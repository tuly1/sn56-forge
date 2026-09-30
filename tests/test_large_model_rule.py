"""Release candidate v9: the >= 3.0 B large-model batch rule (c1f7501 + c919e4c knobs, phase-ba arm C defaults).

Below 3.0 B the run is 854bf7a's (the rule is never called: poisoned-function tests); at >= 3.0 B the batch grows toward
the champion tokens per step with LR x min(2, (B_new / B_old) ** 0.5). Nothing here is a quality result."""
from __future__ import annotations

import importlib.util
import inspect
import os
from functools import partial
from pathlib import Path

import pytest

from forge.tasks import sft_v2
from forge.tuning import large_model_rule as lm

HERE = Path(__file__).resolve().parent
_lfm_spec = importlib.util.spec_from_file_location("clean_lfm_lm", HERE / "test_clean_small_lfm2_route.py")
lfm = importlib.util.module_from_spec(_lfm_spec)
_lfm_spec.loader.exec_module(lfm)
_rt_spec = importlib.util.spec_from_file_location("route_table_lm", HERE / "test_fullft_default_route_table.py")
rt = importlib.util.module_from_spec(_rt_spec)
_rt_spec.loader.exec_module(rt)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)


# ---- the rule --------------------------------------------------------------------------------------------------------

def test_default_on_and_size_gate(monkeypatch):
    assert lm.enabled() and lm.LARGE_MIN_PARAMS_B == 3.0
    assert lm.applies(3.0) and lm.applies(3.821) and lm.applies(4.2058)
    assert not lm.applies(2.999) and not lm.applies(2.70) and not lm.applies(1.17)
    monkeypatch.setenv(lm.SWITCH_ENV, "0")
    assert not lm.applies(3.821)


def test_no_family_inputs():
    for fn in (lm.plan_growth, lm.applies, lm.token_cap, lm.lr_factor):
        assert not set(inspect.signature(fn).parameters) & {"model_type", "vocab_size", "recipe_key", "family"}


def test_phase_ba_arm_c_nuextract_receipt():
    """880b38e9 phase-ba ba-nx15-0928-c-s7 (alpha 0.5, cap 2 by env on 7024722): 48 -> 96 rows (micro 48), token cap
    33,945.6, LR 6.082e-5 -> 8.601e-5 (x sqrt 2). v9 must reproduce it with NO env."""
    tpu = 13909.6 / 48
    got = lm.plan_growth(train_tokens=tpu * 38_777, tokens_per_unit=tpu, eff_units=48, micro=48, t_step_s=1.371,
                         window_s=4236.0, max_epochs=3.0, champ_len=lm.champion_len(273, 754), fill=0.85, gpu_gb=80.0)
    assert got is not None
    units, info = got
    assert lm.champion_len(273, 754) == 832
    assert units == 96 and info["binding"] == "token_cap" and info["cap_tokens"] == pytest.approx(33945.6, abs=0.1)
    assert info["lr_factor"] == pytest.approx(2 ** 0.5, abs=1e-4) and info["lr_alpha"] == 0.5 and info["lr_cap"] == 2.0
    assert 6.082046138999228e-05 * info["lr_factor"] == pytest.approx(8.601229649772708e-05, rel=1e-4)


def test_phase_ba_arm_c_qwen3b_receipt():
    """522e0294 phase-ba ba-q25-3b-evolcode-c-s7: 11 -> 33 packed blocks (micro 11), cap 85,780.1 (fill 0.8726), LR
    4.463e-5 -> 7.730e-5 (x sqrt 3; c1f7501's alpha 0.75 bounded this at x2)."""
    tpu = 26414.8 / 11
    got = lm.plan_growth(train_tokens=20_500_000, tokens_per_unit=tpu, eff_units=11, micro=11, t_step_s=2.3047,
                         window_s=4876.0, max_epochs=3.0, champ_len=2752, fill=85780.1 / (48 * 2048), gpu_gb=80.0)
    assert got is not None
    units, info = got
    assert units == 33 and info["binding"] == "token_cap"
    assert info["lr_factor"] == pytest.approx(3 ** 0.5, abs=1e-4)
    assert 4.4629058653707016e-05 * info["lr_factor"] == pytest.approx(7.730199249408592e-05, rel=1e-4)


def test_no_growth_without_one_epoch_or_on_small_data():
    kw = dict(tokens_per_unit=290.0, eff_units=48, micro=4, max_epochs=3.0, champ_len=832, fill=0.85, gpu_gb=80.0)
    assert lm.plan_growth(train_tokens=11_437_022, t_step_s=10.0, window_s=4000, **kw) is None  # 0.9 epoch fits
    assert lm.plan_growth(train_tokens=2_000_000, t_step_s=1.0, window_s=4000, **kw) is None  # T/200 = 34 rows < 48
    assert lm.plan_growth(train_tokens=11_437_022, t_step_s=None, window_s=4000, **kw) is None


def test_lr_factor_bounds():
    assert lm.LR_ALPHA == 0.5 and lm.LR_FACTOR_MAX == 2.0  # phase-ba arm C, the code default (no env)
    assert lm.lr_factor(2.46, 1.0) == pytest.approx(2.46 ** 0.5)
    assert lm.lr_factor(4, 1) == 2.0 and lm.lr_factor(10, 1) == 2.0 and lm.lr_factor(1, 2) == 1.0


# ---- sft_v2.run integration (tiny LFM2 CPU harness) ----------------------------------------------------------------------

def _run(monkeypatch, tmp_path, *, params_b=None, **run_kw):
    if params_b is not None:
        monkeypatch.setattr(lfm, "LFM12_B", params_b)
    monkeypatch.setattr(sft_v2, "run", partial(sft_v2.run, **run_kw))
    return lfm._run(monkeypatch, tmp_path, csl=False)


def _sig(events, trainers):
    keep = ("sft_v2_field_recipe", "sft_v2_field_lr", "sft_v2_plan", "sft_v2_train_start")
    # wall-clock fields (elapsed time and the deadline-derived reserve/window) vary with host load; drop them
    clock = ("elapsed_s", "finish_reserve_s", "window_s", "remaining_s")
    ev = [(n, {k: v for k, v in d.items() if k not in clock}) for n, d in events if n in keep]
    args = [(t.args.learning_rate, t.args.per_device_train_batch_size, t.args.gradient_accumulation_steps,
             t.args.num_train_epochs, t.args.optim) for t in trainers[1:]]
    return ev, args


def _poison(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("large_model_rule.plan_growth called below 3.0 B, out of route or with the switch off")
    monkeypatch.setattr(lm, "plan_growth", boom)


@pytest.mark.parametrize("key", ["family", "generic", "llama", "minicpm5"])
def test_below_3b_is_854bf7a(monkeypatch, tmp_path, key):
    monkeypatch.setenv(lm.SWITCH_ENV, "0")
    off = _sig(*_run(monkeypatch, tmp_path / "off", recipe_key=key))
    monkeypatch.undo()
    _poison(monkeypatch)
    on = _sig(*_run(monkeypatch, tmp_path / "on", recipe_key=key))
    assert on == off
    monkeypatch.undo()


def test_at_3b_and_above_the_rule_grows_and_scales_lr(monkeypatch, tmp_path):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return kw["eff_units"] * 2, {"new_units": kw["eff_units"] * 2, "lr_factor": 2 ** 0.5, "binding": "token_cap"}

    monkeypatch.setattr(lm, "plan_growth", fake)
    ev, tr = _run(monkeypatch, tmp_path / "on", params_b=3.5, recipe_key="generic")
    grow = lfm._ev(ev, "sft_v2_large_model_batch")
    assert len(calls) == 1 and len(grow) == 1
    plan = lfm._ev(ev, "sft_v2_plan")[0]
    assert plan["lr"] == pytest.approx(grow[0]["lr_old"] * 2 ** 0.5)
    base_eff = calls[0]["eff_units"]
    a = tr[-1].args
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == pytest.approx(2 * base_eff, abs=a.per_device_train_batch_size)
    monkeypatch.undo()
    _poison(monkeypatch)
    monkeypatch.setenv(lm.SWITCH_ENV, "0")
    ev_off, _ = _run(monkeypatch, tmp_path / "off", params_b=3.5, recipe_key="generic")
    assert not lfm._ev(ev_off, "sft_v2_large_model_batch")
    monkeypatch.undo()


def test_rule_skipped_on_cheap_rung_and_out_of_route(monkeypatch, tmp_path):
    _poison(monkeypatch)
    _run(monkeypatch, tmp_path / "a", params_b=3.5, recipe_key="generic", cheap_full=True)
    monkeypatch.undo()
    _poison(monkeypatch)
    _run(monkeypatch, tmp_path / "b", params_b=3.5, recipe_key=None)
    monkeypatch.undo()


# ---- pool table: which in-scope models the rule can touch ---------------------------------------------------------------

def test_pool_split():
    pool = [m for m in rt.pool() if rt.in_scope_size(m["params_b"])]
    big = sorted({m["model"] for m in pool if lm.applies(m["params_b"])})
    small = [m for m in pool if not lm.applies(m["params_b"])]
    assert all(m["params_b"] < 3.0 for m in small) and small
    assert "numind/NuExtract-v1.5" in big and "unsloth/Qwen2.5-3B" in big
    assert not any("LFM2.5-2.6B" in b or "gemma-2-2b" in b for b in big)
    # routing is untouched by this rule: the route decision is 854bf7a's for every in-scope model
    for m in pool:
        assert sft_v2.fullft_default_route(model_type=m["model_type"], params_b=m["params_b"], vocab_size=m["vocab"]).route == "full"


# ---- research knobs (c919e4c, phase-ba): unset = phase-ba arm C (alpha 0.5, cap 2) ---------------------------------------

def test_research_knobs_default_is_phase_ba_arm_c():
    assert lm.lr_alpha() == 0.5 and lm.lr_cap() == 2.0
    assert lm.lr_factor(2.46, 1.0) == pytest.approx(2.46 ** 0.5) and lm.lr_factor(10, 1) == 2.0


@pytest.mark.parametrize("alpha,cap,ratio,want", [("0.75", None, 2.46, 2.46 ** 0.75), ("0.5", None, 9.0, 2.0),
                                                  ("0", None, 3.0, 1.0), ("0.75", "1.5", 3.0, 1.5),
                                                  ("0.75", "1.5", 1.5, 1.5 ** 0.75)])
def test_research_knobs(monkeypatch, alpha, cap, ratio, want):
    monkeypatch.setenv(lm.ALPHA_ENV, alpha)
    if cap is not None:
        monkeypatch.setenv(lm.CAP_ENV, cap)
    assert lm.lr_factor(ratio, 1.0) == pytest.approx(want)


@pytest.mark.parametrize("bad", ["x", "-0.1", "1.5", ""])
def test_research_knobs_out_of_range_fall_back_to_default(monkeypatch, bad):
    monkeypatch.setenv(lm.ALPHA_ENV, bad)
    monkeypatch.setenv(lm.CAP_ENV, bad if bad != "1.5" else "9")
    assert lm.lr_alpha() == 0.5 and lm.lr_cap() == 2.0


def test_research_knobs_are_logged_in_the_growth_info(monkeypatch):
    monkeypatch.setenv(lm.ALPHA_ENV, "0.75")
    got = lm.plan_growth(train_tokens=20_500_000, tokens_per_unit=2_370, eff_units=11, micro=11, t_step_s=2.34,
                         window_s=4870, max_epochs=3.0, champ_len=lm.champion_len(453, 2482), fill=0.86, gpu_gb=80.0)
    units, info = got
    assert units == 33 and info["lr_alpha"] == 0.75 and info["lr_cap"] == 2.0
    assert info["lr_factor"] == 2.0  # 3 ** 0.75 = 2.28, bounded


# ---- route snapshot declaration (tests/test_route_snapshot_v9.py collects V9_ROUTE_CHANGES) ------------------------------

def _large_model_fullft(key: str, base: dict) -> bool:
    """In-scope full-FT default FIELD cells (recipe_key set, rung 1) with params_b >= 3.0 (production switch)."""
    _model, _data, _rows, _verdict, switch = key.split("|")
    kw = base.get("kw") or {}
    return (switch == "default" and base.get("strategy") == "full" and bool(kw.get("field_route"))
            and bool(kw.get("recipe_key")) and not kw.get("cheap_full") and float(kw.get("params_b") or 0) >= 3.0)


V9_ROUTE_CHANGES = {"large_model_batch_rule": (_large_model_fullft, ("rules", "large_model_batch"), True)}
