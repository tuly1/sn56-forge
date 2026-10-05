"""Pre-deploy defect fixes D2-D7 + window use for release-fullft-default-20260929 (DEFECT-FIXES.md).

Run-level tests reuse the tiny-LFM2 CPU harness of tests/test_clean_small_lfm2_route.py (real sft_v2.run route, recipe,
geometry, LR and planner code; training / timing / dev scoring / export are doubles).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

from forge.tasks import sft_v2
from forge.tuning import selection

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ladder_0929", HERE / "test_fullft_fallback_ladder.py")
lad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lad)
lfm = lad.lfm


def _run(monkeypatch, tmp_path, *, late=None, steps=1, **run_kw):
    """lad._run with ``late(monkeypatch)`` applied after the harness's own patches, right before sft_v2.run starts."""
    if late is not None:
        real = sft_v2.run

        def wrapped(*a, **k):
            late(monkeypatch)
            return real(*a, **k)

        monkeypatch.setattr(sft_v2, "run", wrapped)
    return lad._run(monkeypatch, tmp_path, steps=steps, **run_kw)


# ---- D2: container-aware RAM gate, pool cap, memory.peak --------------------------------------------------------------

def _write(d: Path, **files):
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (d / name.replace("__", ".")).write_text(text)


def test_cgroup_v2_headroom(tmp_path):
    gib = 1024 ** 3
    _write(tmp_path, memory__max=str(110 * gib), memory__current=str(60 * gib),
           memory__stat=f"anon 1\ninactive_file {10 * gib}\n", memory__peak=str(97 * gib))
    assert selection.cgroup_memory_available(str(tmp_path)) == 60 * gib  # 110 - (60 - 10 reclaimable cache)
    assert selection.cgroup_memory_peak(str(tmp_path)) == {"peak_bytes": 97 * gib, "limit_bytes": 110 * gib}


def test_cgroup_v2_unlimited_and_v1(tmp_path):
    _write(tmp_path / "v2", memory__max="max", memory__current="5")
    assert selection.cgroup_memory_available(str(tmp_path / "v2")) is None
    gib = 1024 ** 3
    _write(tmp_path / "v1" / "memory", memory__limit_in_bytes=str(118 * gib), memory__usage_in_bytes=str(20 * gib),
           memory__stat=f"total_inactive_file {4 * gib}\n", memory__max_usage_in_bytes=str(30 * gib))
    assert selection.cgroup_memory_available(str(tmp_path / "v1")) == 102 * gib
    assert selection.cgroup_memory_peak(str(tmp_path / "v1")) == {"peak_bytes": 30 * gib, "limit_bytes": 118 * gib}
    _write(tmp_path / "v1u" / "memory", memory__limit_in_bytes=str(2 ** 63 - 4096))
    assert selection.cgroup_memory_available(str(tmp_path / "v1u")) is None


def test_available_ram_is_the_smaller_of_host_and_container(monkeypatch):
    monkeypatch.setattr(selection, "_host_available_ram", lambda: 1400 * 1024 ** 3)
    monkeypatch.setattr(selection, "cgroup_memory_available", lambda root=None: 50 * 1024 ** 3)
    assert selection.available_ram() == 50 * 1024 ** 3
    monkeypatch.setattr(selection, "cgroup_memory_available", lambda root=None: None)
    assert selection.available_ram() == 1400 * 1024 ** 3


def test_container_limit_now_refuses_snapshots(monkeypatch):
    """4.2 B: 8.4 GB per snapshot. With 118 GB host-free but 20 GB container headroom the gate must refuse."""
    import torch

    class M:
        def parameters(self):
            return [torch.nn.Parameter(torch.empty(0))]

    pool = selection.SnapshotPool(k=selection.snapshot_pool_k(4.2058e9))
    pool._snap_bytes = int(8.4e9)
    monkeypatch.setattr(selection, "_host_available_ram", lambda: 1400 * 10 ** 9)
    monkeypatch.setattr(selection, "cgroup_memory_available", lambda root=None: 20 * 10 ** 9)
    assert pool.k == 3 and pool.can_admit(M()) is False
    monkeypatch.setattr(selection, "cgroup_memory_available", lambda root=None: 60 * 10 ** 9)
    assert pool.can_admit(M()) is True


@pytest.mark.parametrize("params,k", [(4.2058e9, 3), (3.8211e9, 3), (3.0859e9, 3), (2.614e9, 8), (1.17e9, 8)])
def test_pool_k_cap_above_3b(params, k):
    assert selection.snapshot_pool_k(params) == k


def test_adapter_pool_keeps_k8(monkeypatch, tmp_path):
    import inspect

    src = inspect.getsource(sft_v2.run)
    assert 'snapshot_pool_k(float(params_b) * 1e9) if strategy == "full"' in src and "else selection.SNAPSHOT_POOL_K" in src
    assert selection.SNAPSHOT_POOL_K == 8


def test_run_logs_pool_k_and_cgroup(monkeypatch, tmp_path):
    ev, _ = _run(monkeypatch, tmp_path, recipe_key="family")
    pool = lfm._ev(ev, "sft_v2_snapshot_pool")
    assert len(pool) == 1 and pool[0]["k"] == 8 and "peak_bytes" in pool[0]
    assert "cgroup_peak_bytes" in lfm._ev(ev, "sft_v2_final")[0]


# ---- D3: the zero-step raise is in-scope only -------------------------------------------------------------------------

def test_zero_steps_out_of_scope_raises_too(monkeypatch, tmp_path):
    # v9.2 (owner decision 2026-10-05): no floor exists, so zero steps raises on every route (cab930c returned)
    with pytest.raises(sft_v2.ZeroTrainedStepsError):
        _run(monkeypatch, tmp_path, steps=0)  # recipe_key None = not the full-FT default route


def test_zero_steps_in_scope_raises(monkeypatch, tmp_path):
    with pytest.raises(sft_v2.ZeroTrainedStepsError):
        _run(monkeypatch, tmp_path, steps=0, recipe_key="generic")


# ---- D5: a failed final save with no trained artifact raises (in scope) -----------------------------------------------

def _failing_save(m):
    def boom(*a, **k):
        raise OSError("disk full")

    m.setattr(sft_v2, "save_adapter", boom)
    m.setattr(sft_v2, "trained_artifact_present", lambda spec: False)


def test_persist_failure_raises_in_scope(monkeypatch, tmp_path):
    with pytest.raises(sft_v2.PersistFailedError):
        _run(monkeypatch, tmp_path, late=_failing_save, recipe_key="family")


def test_persist_failure_out_of_scope_raises_too(monkeypatch, tmp_path):
    # v9.2: with no floor, a failed final save and no trained artifact raises on every route (cab930c returned)
    with pytest.raises(sft_v2.PersistFailedError):
        _run(monkeypatch, tmp_path, late=_failing_save)


def test_persist_failure_with_partial_artifact_keeps_it(monkeypatch, tmp_path):
    def late(m):
        _failing_save(m)
        m.setattr(sft_v2, "trained_artifact_present", lambda spec: True)

    ev, _ = _run(monkeypatch, tmp_path, late=late, recipe_key="family")
    assert lfm._ev(ev, "sft_v2_persist_failed_no_artifact") == []


def test_ladder_retries_after_persist_failure(monkeypatch):
    calls, events, legacy = lad.ladder(monkeypatch, lad._entry("unsloth/Llama-3.2-1B"),
                                       outcomes=[sft_v2.PersistFailedError("x"), "ok"])
    assert [c["cheap"] for c in calls] == [False, True] and not legacy


# ---- D4: late fixed-LR runs still measure step time ------------------------------------------------------------------

def _no_sweep_floor(m):
    m.setattr(sft_v2, "MIN_TASK_SECONDS_FOR_V2", 10 ** 6)  # the old gate would skip the probe at any remaining time


def test_late_rung_gets_timing_probe(monkeypatch, tmp_path):
    ev, _ = _run(monkeypatch, tmp_path, late=_no_sweep_floor, recipe_key="family", cheap_full=True)
    assert len(lfm._ev(ev, "sft_v2_late_timing_probe")) == 1
    assert lfm._ev(ev, "sft_v2_timing")[0]["t_per_step"] == 1.0
    assert lfm._ev(ev, "sft_v2_plan")[0]["t_per_step"] == 1.0


def test_out_of_scope_probe_gate_unchanged(monkeypatch, tmp_path):
    ev, _ = _run(monkeypatch, tmp_path, late=_no_sweep_floor)
    assert lfm._ev(ev, "sft_v2_timing") == [] and lfm._ev(ev, "sft_v2_late_timing_probe") == []


def test_rung2_entry_floor_unchanged():
    assert sft_v2.CHEAP_FULL_MIN_REMAINING_S == 900.0 and sft_v2.TIMING_PROBE_MIN_REMAINING_S == 600.0


# ---- D6: dev pass optimizer ------------------------------------------------------------------------------------------

def _geo(fp32: bool, optim: str):
    return sft_v2.Geometry(4, 4, 512, True, fp32, optim)


def test_dev_pass_optimizer_choice(monkeypatch):
    default = object()
    assert sft_v2.dev_pass_optimizer_factory(_geo(True, "adamw_torch_fused"), default) is default
    assert sft_v2.dev_pass_optimizer_factory(_geo(False, "adamw_torch"), default, strategy="lora") is default
    assert sft_v2.dev_pass_optimizer_factory(_geo(False, "adamw_torch"), default) is None  # bf16 without 8-bit: skip
    monkeypatch.setattr(sft_v2, "_bnb_available", lambda: True)
    made = {}

    class PagedAdamW8bit:
        def __init__(self, params, **kw):
            made.update(kw, n=len(params))

    monkeypatch.setitem(sys.modules, "bitsandbytes", types.SimpleNamespace(optim=types.SimpleNamespace(PagedAdamW8bit=PagedAdamW8bit)))
    f = sft_v2.dev_pass_optimizer_factory(_geo(False, "paged_adamw_8bit"), default, betas=(0.9, 0.95))
    assert f is not None and f is not default
    assert isinstance(f([1, 2], 1e-5), PagedAdamW8bit)
    assert made == {"lr": 1e-5, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0, "n": 2}


def test_bf16_run_without_8bit_skips_dev_pass(monkeypatch, tmp_path):
    def late(m):
        m.setenv("FORGE_DEV_PASS", "1")
        m.setattr(sft_v2, "stratified_split", lambda r, *a, **k: (r[:-2], r[-2:] * 60))  # >= 100 dev rows

    ev, _ = _run(monkeypatch, tmp_path, late=late, recipe_key="family", cheap_full=True)  # rung 2 = bf16 weights
    assert lfm._ev(ev, "sft_v2_dev_pass_skipped")[0]["reason"] == "bf16_weights_without_8bit_adamw"
    assert lfm._ev(ev, "sft_v2_dev_pass_done") == []


# ---- D7: llama fp32-master switch ------------------------------------------------------------------------------------

def test_llama_switch_off_by_default(monkeypatch):
    monkeypatch.delenv("FORGE_V2_LLAMA_BF16_MIN_B", raising=False)
    r = {"bf16": False, "bf16_min_b": 2.0, "lr_mult": 1.0}
    assert sft_v2.LLAMA_RECIPE_BF16_MIN_B is None
    assert sft_v2.llama_recipe_precision(r, 3.2127) is r


def test_llama_switch_on(monkeypatch):
    monkeypatch.setenv("FORGE_V2_LLAMA_BF16_MIN_B", "3.0")
    r = {"bf16": False, "bf16_min_b": 2.0, "lr_mult": 1.0}
    assert sft_v2.llama_recipe_precision(r, 3.2127)["bf16"] is True
    assert sft_v2.llama_recipe_precision(r, 1.2358) is r


# ---- window use: DEV-gated continuation past the epoch cap -----------------------------------------------------------

NUX = dict(recipe_key="generic", phase=1, extended=False, last_phase_complete=True, stopped_overfit=False,
           best_step=7461, step_offset=7461, epochs_done=3.001, max_epochs=3.0, window_s=2146.0, min_window_s=240.0,
           t_step=0.269, steps_per_epoch=2486)


def test_extension_on_nuextract_shape():
    extra = sft_v2.fullft_epoch_cap_extension(**NUX)
    assert extra == round(min(3.0, 2146.0 * 0.85 / 0.269 / 2486), 3) and extra > 2.5


@pytest.mark.parametrize("change", [
    dict(recipe_key=None), dict(extended=True), dict(phase=2), dict(last_phase_complete=False),
    dict(stopped_overfit=True), dict(best_step=6897), dict(epochs_done=2.2), dict(window_s=200.0), dict(t_step=None),
    dict(window_s=320.0, t_step=30.0),  # fits fewer steps than the planner's minimum continuation phase
])
def test_extension_refused(change):
    assert sft_v2.fullft_epoch_cap_extension(**{**NUX, **change}) == 0.0


def test_extension_env_off(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FULLFT_EXTEND", "0")
    assert sft_v2.fullft_epoch_cap_extension(**NUX) == 0.0


def _capped_improving(m):
    """Every phase completes its plan and each DEV evaluation improves; the cap binds at 1 epoch."""
    import transformers

    m.setenv("FORGE_V2_MAX_EPOCHS", "1")
    m.setenv("FORGE_V2_FIELD_EFF_BATCH", "1")  # enough optimizer steps per epoch for a continuation phase
    losses = iter([1.0 - 0.01 * i for i in range(1000)])
    m.setattr(selection, "per_example_dev_loss", lambda *a, **k: (next(losses), []))
    Base = transformers.Trainer

    class Complete(Base):
        def train(self):
            n = int(self.callbacks[0].total_steps)
            self.state.max_steps = n
            self.state.global_step = n
            self.callbacks[0].on_step_end(self.args, self.state, types.SimpleNamespace(should_training_stop=False),
                                          model=self.model)

    m.setattr(transformers, "Trainer", Complete)


def test_run_extends_capped_improving_full_ft(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path, late=_capped_improving, recipe_key="family")
    ext = lfm._ev(ev, "sft_v2_fullft_epoch_cap_extended")
    plans = lfm._ev(ev, "sft_v2_plan")
    assert len(ext) == 1 and ext[0]["cap"] == 1.0
    assert [p["phase"] for p in plans] == [0, 1]
    assert math.isclose(plans[1]["lr"], plans[0]["lr"] * 0.5)
    assert plans[1]["epochs"] <= ext[0]["extra_epochs"] + 1e-9


def test_run_without_recipe_key_never_extends(monkeypatch, tmp_path):
    ev, _ = _run(monkeypatch, tmp_path, late=_capped_improving)
    assert lfm._ev(ev, "sft_v2_fullft_epoch_cap_extended") == []
    assert [p["phase"] for p in lfm._ev(ev, "sft_v2_plan")] == [0]
