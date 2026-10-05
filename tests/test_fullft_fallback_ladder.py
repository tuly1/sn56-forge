"""Fallback ladder, zero-step guard, recipes and legacy adapter fixes of the full-FT default (release-fullft-default-20260929).

Owner directive 4: full FT -> cheaper full FT (bf16 + paged 8-bit AdamW, smaller micro-batch) -> adapter only as the
crash fallback of last resort; zero trained steps never ship an untrained copy while a rung remains.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from forge.data import packing
from forge.tasks import instruct, sft_v2
from forge.tuning import legacy_adapter_fixes as lfix

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("route_table", HERE / "test_fullft_default_route_table.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


class StopAfterLegacyFixes(BaseException):
    pass


def _entry(model):
    return next(e for e in rt.POOL if e["model"] == model)


def ladder(monkeypatch, entry, *, outcomes, trained_after=None, hours=1.0, legacy_stop="plan", n_gpus=1, rows="short"):
    """Drive instruct.run with a scripted sft_v2.run ("ok" or an exception per call). Returns (v2 calls, events,
    legacy reached)."""
    script = list(outcomes)
    calls: list = []
    monkeypatch.setattr(sft_v2, "trained_artifact_present", lambda spec: False)

    def v2(**kw):
        calls.append({"cheap": kw.get("cheap_full"), "recipe": kw.get("recipe_key"), "field": kw["field_route"]})
        if trained_after is not None and len(calls) >= trained_after:
            monkeypatch.setattr(sft_v2, "trained_artifact_present", lambda spec: True)
        out = script.pop(0)
        if out != "ok":
            raise out

    legacy = {"reached": False}

    def legacy_handler(**kw):
        legacy["reached"] = True
        if legacy_stop == "fixes":
            from forge.tuning.plan import make_sft_plan as real_plan

            def stop(*a, **k):
                raise StopAfterLegacyFixes()

            monkeypatch.setattr(instruct, "conservative_qwen35_plan", stop)
            return real_plan(**kw)
        raise rt.StopAtLegacyPlan()

    events: list = []
    try:
        rt.dispatch(monkeypatch, entry, data="medium", rows=rows, verdict="clean", hours=hours, n_gpus=n_gpus, v2_handler=v2,
                    legacy_handler=legacy_handler, events_out=events)
    except StopAfterLegacyFixes:
        pass
    return calls, events, legacy["reached"]


@pytest.fixture
def llama1b():
    return _entry("unsloth/Llama-3.2-1B")


def test_rung2_cheap_full_after_a_failure(monkeypatch, llama1b):
    calls, events, legacy = ladder(monkeypatch, llama1b, outcomes=[RuntimeError("CUDA out of memory"), "ok"])
    assert [c.get("cheap") for c in calls] == [False, True] and not legacy
    assert all(c["field"] is True and c["recipe"] == "llama" for c in calls)
    assert "sft_v2_fallback_to_cheap_full" in [n for n, _ in events]


def test_zero_trained_steps_move_to_the_next_rung(monkeypatch, llama1b):
    calls, events, legacy = ladder(monkeypatch, llama1b, outcomes=[sft_v2.ZeroTrainedStepsError("x"), "ok"])
    assert [c.get("cheap") for c in calls] == [False, True] and not legacy
    failed = [kw for n, kw in events if n == "sft_v2_failed"]
    assert failed[0]["zero_trained_steps"] is True


def test_both_full_rungs_fail_then_adapter_last(monkeypatch, llama1b):
    calls, events, legacy = ladder(monkeypatch, llama1b,
                                   outcomes=[RuntimeError("boom"), sft_v2.ZeroTrainedStepsError("x")])
    assert [c.get("cheap") for c in calls] == [False, True] and legacy
    assert "sft_v2_fallback_to_lora" in [n for n, _ in events]


def test_adapter_rung_runs_with_the_legacy_fixes(monkeypatch, llama1b):
    calls, events, legacy = ladder(monkeypatch, llama1b, outcomes=[RuntimeError("boom"), RuntimeError("again")],
                                   legacy_stop="fixes")
    names = [n for n, _ in events]
    assert "sft_v2_fallback_to_cheap_full" in names and "sft_v2_fallback_to_lora" in names
    on = [kw for n, kw in events if n == "legacy_fixes_on"]
    assert on and on[0]["lr_scheduler"] == "cosine" and on[0]["fallback"] is True
    assert legacy


def test_a_trained_artifact_is_kept(monkeypatch, llama1b):
    calls, events, legacy = ladder(monkeypatch, llama1b, outcomes=[RuntimeError("late failure")], trained_after=1)
    assert [c.get("cheap") for c in calls] == [False] and not legacy
    assert "sft_v2_fallback_to_lora" not in [n for n, _ in events]


def test_short_window_skips_rung2_but_never_ships_the_floor(monkeypatch, llama1b):
    """< 900 s left: no time for a second full attempt; in scope the adapter rung still runs (cab930c kept the floor)."""
    calls, events, legacy = ladder(monkeypatch, llama1b, outcomes=[RuntimeError("boom")], hours=0.2)
    assert [c.get("cheap") for c in calls] == [False] and legacy
    assert "sft_v2_fallback_to_lora" in [n for n, _ in events]


def test_out_of_scope_keeps_cab930c_failure_handling(monkeypatch):
    big = _entry("Qwen/Qwen3-14B")  # 2 GPUs: cab930c's multi-GPU v2 adapter route
    calls, events, legacy = ladder(monkeypatch, big, outcomes=[RuntimeError("boom")], n_gpus=2, rows="long")
    assert [c.get("cheap") for c in calls] == [False] and legacy  # no cheap rung outside the scope
    # under 900 s there is no second recipe; v9.2 re-raises so the CLI runs the minimal fine-tune (cab930c kept a floor)
    with pytest.raises(RuntimeError, match="boom"):
        ladder(monkeypatch, big, outcomes=[RuntimeError("boom")], hours=0.2, n_gpus=2, rows="long")


def test_quality_exceptions_keep_their_adapter_recipe(monkeypatch):
    assert sft_v2.legacy_adapter_fixes_enabled(fallback=False, crash_exception_route=False,
                                               quality_exception_route=True) is False
    assert sft_v2.legacy_adapter_fixes_enabled(fallback=True, crash_exception_route=False,
                                               quality_exception_route=False) is True
    assert sft_v2.legacy_adapter_fixes_enabled(fallback=False, crash_exception_route=True,
                                               quality_exception_route=False) is True
    monkeypatch.setenv("FORGE_LEGACY_FIXES_ON_EXCEPTIONS", "1")
    assert sft_v2.legacy_adapter_fixes_enabled(fallback=False, crash_exception_route=False,
                                               quality_exception_route=True) is True


# ---- recipes ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mt,params_b,vocab,key,bf16,lr_mult", [
    ("lfm2", 1.1703, 65536, "family", False, 1.0),      # LFM2 small band: fp32 master, LR x1 (v8, phase-bb; 7024722 x2)
    ("lfm2", 0.354, 65536, "family", False, 1.0),
    ("lfm2", 2.697, 128000, "family", True, 1.0),      # LFM2.5-2.6B long rows: bf16 + paged 8-bit, x1 (MathFusion)
    ("qwen2", 0.494, 151936, "family", True, 1.0),     # Qwen FIELD: bf16 from 0.3 B, x1
    ("qwen2", 3.086, 151936, "family", True, 1.0),
    ("qwen3", 4.022, 151936, "family", True, 1.0),
    ("gemma2", 2.614, 256000, "family", True, 1.0),    # + champion port in its band (run-level)
    ("llama", 1.081, 130560, "minicpm5", True, 1.0),   # MiniCPM5 F1
    ("llama", 1.236, 128256, "llama", False, 1.0),     # Llama production fp32 FIELD
    ("llama", 3.213, 128256, "llama", False, 1.0),     # fp32 master up to FP32_MASTER_MAX_B
    ("llama", 1.1, 32000, "llama", False, 1.0),
    ("phi3", 3.821, 32064, "generic", True, 1.0),      # generic: bf16 from 2.0 B, x1
    ("granite", 3.403, 100352, "generic", True, 1.0),
    ("qwen3_5", 4.2058, 248320, "generic", True, 1.0),
    ("qwen3_5", 0.873, 248320, "generic", False, 1.0), # generic below 2.0 B: fp32 master
    ("gpt_neox", 0.405, 50304, "generic", False, 1.0),
    ("phi", 1.418, 51200, "generic", False, 1.0),
])
def test_recipe_per_family(monkeypatch, mt, params_b, vocab, key, bf16, lr_mult):
    for k in list(sys.modules["os"].environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    assert sft_v2.default_recipe_key(mt, params_b, vocab) == key
    rec = sft_v2.run_field_recipe(mt, params_b, field=True, recipe_key=key)  # what sft_v2.run applies
    assert (rec["bf16"], rec["lr_mult"], rec["strategy"]) == (bf16, lr_mult, "full")
    # field_recipe without a recipe key is the family row as the cab930c routes read it (lfm2 small band still x2)
    old = sft_v2.field_recipe(mt, params_b, damaged_base=key in ("family", "minicpm5"), generic=key == "generic")
    small_lfm2 = mt == "lfm2" and params_b <= 1.5
    assert (old["bf16"], old["lr_mult"]) == (bf16, 2.0 if small_lfm2 else lr_mult)


# ---- sft_v2.run on CPU: zero-step guard, cheap rung, recipe keys -----------------------------------------------------

_lfm_spec = importlib.util.spec_from_file_location("clean_lfm", HERE / "test_clean_small_lfm2_route.py")
lfm = importlib.util.module_from_spec(_lfm_spec)
_lfm_spec.loader.exec_module(lfm)


def _run(monkeypatch, tmp_path, *, steps=1, **run_kw):
    """lfm._run (the tiny-LFM2 CPU harness) with extra sft_v2.run keywords and an optional zero-step trainer."""
    monkeypatch.setattr(sft_v2, "run", partial(sft_v2.run, **run_kw))
    if steps == 0:
        import transformers

        real_setattr = monkeypatch.setattr

        def zero_step_trainer(target, name, value, *a, **k):
            if target is transformers and name == "Trainer":
                class ZeroStep(value):
                    def train(self):  # the clock stopped it before the first optimizer step
                        self.state.global_step = 0
                value = ZeroStep
            return real_setattr(target, name, value, *a, **k)

        monkeypatch.setattr = zero_step_trainer  # type: ignore[method-assign]
    return lfm._run(monkeypatch, tmp_path, csl=False)


def test_zero_trained_steps_raise(monkeypatch, tmp_path):
    with pytest.raises(sft_v2.ZeroTrainedStepsError):
        _run(monkeypatch, tmp_path, steps=0, recipe_key="family")


def test_family_recipe_key_is_the_lfm2_small_band(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path, recipe_key="family")
    rec = lfm._ev(ev, "sft_v2_field_recipe")[0]
    assert (rec["recipe_key"], rec["bf16"], rec["lr_mult"], rec["cheap_full"]) == ("family", False, 1.0, False)  # v8
    assert lfm._ev(ev, "sft_v2_cheap_full") == []


def test_cheap_full_rung_is_bf16_8bit_half_micro(monkeypatch, tmp_path):
    base_ev, base_tr = _run(monkeypatch, tmp_path / "a", recipe_key="family")
    monkeypatch.undo()
    ev, tr = _run(monkeypatch, tmp_path / "b", recipe_key="family", cheap_full=True)
    cheap = lfm._ev(ev, "sft_v2_cheap_full")
    assert len(cheap) == 1
    assert cheap[0]["optim"] in ("paged_adamw_8bit", "adamw_torch")  # adamw_torch only without bitsandbytes (CPU)
    t_base, t = base_tr[-1].args, tr[-1].args
    assert t.per_device_train_batch_size <= max(1, t_base.per_device_train_batch_size)
    assert (t.per_device_train_batch_size * t.gradient_accumulation_steps
            >= t_base.per_device_train_batch_size * t_base.gradient_accumulation_steps // 2)


def test_packing_contract_unchanged():
    # the run-level harness above only runs on the image's transformers (lfm._run skips otherwise)
    assert isinstance(packing.QUALIFIED_TRANSFORMERS, str)


# ---- legacy adapter fixes (pure pieces) ------------------------------------------------------------------------------

def test_legacy_fix_epoch_cap_and_schedule():
    assert lfix.cap_epochs(4.0, 14_000, 2.0) == 2.0      # the 803a8f69 stretch 2 -> 4 is capped
    assert lfix.cap_epochs(1.3, 14_000, 2.0) == 1.3
    assert lfix.cap_epochs(None, 14_000, 2.0) is None     # plan default (2) already within the cap
    assert lfix.cap_epochs(4.0, 50_000, 2.0) == 4.0       # large data: the time-aware plan stands
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class P:
        lr_scheduler: str

    assert lfix.apply_plan(P("cosine_with_min_lr")).lr_scheduler == "cosine"


def test_legacy_fix_early_stop_and_snapshots():
    from forge.tuning.selection import SnapshotPool

    pool = SnapshotPool(k=4)
    seen = []
    pool.consider = lambda model, loss, step: seen.append((loss, step)) or True  # type: ignore[method-assign]
    cb = lfix.make_callbacks(pool)[0]
    control = NS(should_training_stop=False)
    for step, loss in enumerate([1.0, 0.9, 0.95, 0.96, 0.97], start=1):
        cb.on_evaluate(None, NS(global_step=step), control, metrics={"eval_loss": loss}, model=object())
    assert control.should_training_stop is True  # 3 evaluations without a new best
    assert [s for _, s in seen] == [1, 2, 3, 4, 5]


def test_legacy_fix_soup_saves_only_an_improving_multi_member_soup(monkeypatch):
    from forge.tuning import selection

    saved = []
    pool = NS(items=[{"loss": 1.0, "step": 1, "state": {}}, {"loss": 1.1, "step": 2, "state": {}}],
              best=lambda: {"loss": 1.0, "step": 1, "state": {}})
    monkeypatch.setattr(selection, "greedy_soup", lambda *a, **k: (0.9, {"w": 1}, 2))
    assert lfix.soup_and_save(object(), pool, [{"input_ids": [1]}], None, save=saved.append, time_budget_s=60) is True
    monkeypatch.setattr(selection, "greedy_soup", lambda *a, **k: (1.0, {"w": 1}, 1))
    assert lfix.soup_and_save(object(), pool, [{"input_ids": [1]}], None, save=saved.append, time_budget_s=60) is False
    assert len(saved) == 1
    assert lfix.soup_and_save(object(), NS(items=[]), [1], None, save=saved.append, time_budget_s=60) is False
