"""Release candidate v9: the two phase-ao Qwen3.5-4B planner fixes from c1f7501 (full-FT default route, every size).

1. starved_finish_reserve: drop the dev-pass estimate from the finish reserve when keeping it leaves < max(25 % of the
   remaining time, 3 measured steps) to train.
2. dev_pass_with_oom_retry: on a dev-pass CUDA OOM restore the pre-dev-pass weights; retry once at half the micro-batch
   only if a conservative retry plus the export still fits (RESP-074-1 B1), else keep the restored weights.
phase-aw (c1f7501 on GPU): the long-row reserve drop fired and rung 1 trained 4 steps; on the short item the OOM retry was
SKIPPED for time (a safe degradation) - the half-micro retry and any OOM quality repair are FIX_NOT_PROVEN on GPU."""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

import pytest

from forge.tasks import sft_v2


def test_starved_finish_reserve_phase_ao_numbers():
    # ao-q35-4b-long-thr345: 1,793 s left, reserve 3,695.6 s (dev pass 2,658 s), 211 s / step
    assert sft_v2.starved_finish_reserve(remaining=1793, finish_reserve=3695.6, dev_pass_s=2658, t_step=211) == \
        pytest.approx(1037.6)
    # not starved: unchanged (the 7024722 value)
    assert sft_v2.starved_finish_reserve(remaining=3600, finish_reserve=400, dev_pass_s=100, t_step=2) == 400
    # starved but no dev pass to drop: unchanged
    assert sft_v2.starved_finish_reserve(remaining=500, finish_reserve=600, dev_pass_s=0, t_step=2) == 600


def _retry_env(monkeypatch):
    geos, loads, events = [], [], []

    def fake_dev_pass(model, dev, collator, *, lr, geo, autocast, optimizer_factory):
        geos.append(geo)
        if len(geos) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 8.86 GiB")

    monkeypatch.setattr(sft_v2, "_dev_pass", fake_dev_pass)
    monkeypatch.setattr(sft_v2.selection, "load_state", lambda m, st: loads.append(st))
    monkeypatch.setattr(sft_v2, "_free_cuda", lambda: None)
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **k: events.append(n))
    return geos, loads, events


def _retry(remaining, dev_pass_s=400.0, export=150.0):
    geo = sft_v2.Geometry(24, 1, 2048, True, False, "paged_adamw_8bit")
    model = NS(zero_grad=lambda set_to_none=True: None)
    sft_v2.dev_pass_with_oom_retry(model, [], None, lr=1e-5, geo=geo, autocast=False, optimizer_factory=None,
                                   restore_state={"w": 1}, remaining_s=lambda: remaining, export_reserve_s=export,
                                   dev_pass_s=dev_pass_s)


def test_dev_pass_oom_retry_with_sufficient_time_halves_micro(monkeypatch):
    geos, loads, events = _retry_env(monkeypatch)
    _retry(remaining=150 + 1.5 * 400 + 61)  # export + conservative retry + margin fit
    assert [g.micro_batch for g in geos] == [24, 12] and geos[1].eff_batch >= 24 and loads == [{"w": 1}]
    assert "sft_v2_dev_pass_oom_retry" in events and "sft_v2_dev_pass_retry_skipped" not in events


def test_late_dev_pass_oom_without_time_keeps_restored_weights(monkeypatch):
    geos, loads, events = _retry_env(monkeypatch)
    _retry(remaining=150 + 1.5 * 400 + 59)  # one second short of the conservative budget
    assert [g.micro_batch for g in geos] == [24]  # no second pass
    assert loads == [{"w": 1}] and "sft_v2_dev_pass_retry_skipped" in events


def test_retry_budget_rule():
    assert sft_v2.dev_pass_retry_fits(remaining_s=1000, export_reserve_s=150, dev_pass_s=400) is True
    assert sft_v2.dev_pass_retry_fits(remaining_s=800, export_reserve_s=150, dev_pass_s=400) is False


def test_dev_pass_other_errors_propagate(monkeypatch):
    def bad(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(sft_v2, "_dev_pass", bad)
    geo = sft_v2.Geometry(24, 1, 2048, True, False, "paged_adamw_8bit")
    with pytest.raises(ValueError):
        sft_v2.dev_pass_with_oom_retry(NS(), [], None, lr=1e-5, geo=geo, autocast=False, optimizer_factory=None,
                                       restore_state={"w": 1}, remaining_s=lambda: 1e6, export_reserve_s=150,
                                       dev_pass_s=10)


def test_run_arms_the_fixes_only_on_the_fullft_default_route():
    src = inspect.getsource(sft_v2.run)
    i = src.index("_fr = starved_finish_reserve(")
    assert "if recipe_key:" in src[i - 400:i]
    j = src.index("dev_pass_with_oom_retry(model, dev_ex")
    assert "if recipe_key:" in src[j - 200:j]
    assert "_dev_pass(model, dev_ex, collator, lr=lr * _dp_mult, geo=geo" in src[j:j + 700]  # the 854bf7a call off-route


def test_skipped_retry_does_not_run_a_second_pass_and_restores_before_reading_the_deadline(monkeypatch):
    order = []

    def fake_dev_pass(*a, **k):
        order.append("pass")
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(sft_v2, "_dev_pass", fake_dev_pass)
    monkeypatch.setattr(sft_v2.selection, "load_state", lambda m, st: order.append("restore"))
    monkeypatch.setattr(sft_v2, "_free_cuda", lambda: order.append("free"))
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **k: order.append(n))

    def left():
        order.append("deadline")
        return 10.0

    geo = sft_v2.Geometry(24, 1, 2048, True, False, "paged_adamw_8bit")
    sft_v2.dev_pass_with_oom_retry(NS(zero_grad=lambda set_to_none=True: None), [], None, lr=1e-5, geo=geo,
                                   autocast=False, optimizer_factory=None, restore_state={"w": 1}, remaining_s=left,
                                   export_reserve_s=150, dev_pass_s=400)
    assert order == ["pass", "sft_v2_dev_pass_oom_retry", "restore", "free", "deadline", "sft_v2_dev_pass_retry_skipped"]


def test_oom_without_restorable_state_propagates(monkeypatch):
    monkeypatch.setattr(sft_v2, "_dev_pass", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    geo = sft_v2.Geometry(24, 1, 2048, True, False, "paged_adamw_8bit")
    with pytest.raises(RuntimeError):
        sft_v2.dev_pass_with_oom_retry(NS(), [], None, lr=1e-5, geo=geo, autocast=False, optimizer_factory=None,
                                       restore_state=None, remaining_s=lambda: 1e6, export_reserve_s=150, dev_pass_s=10)


# ---- route snapshot declaration (tests/test_route_snapshot_v9.py collects V9_ROUTE_CHANGES) ------------------------------

def _planner_fixes_fullft(key: str, base: dict) -> bool:
    """Every sft_v2 cell the dispatcher gives a recipe_key (full-FT default route), at every size: the fixes are armed there
    and each acts only when its trigger fires."""
    return bool((base.get("kw") or {}).get("recipe_key"))


V9_ROUTE_CHANGES = {"phase_ao_planner_fixes": (_planner_fixes_fullft, ("rules", "planner_fixes"), True)}
