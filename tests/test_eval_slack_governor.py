"""Window-aware dev-eval governor (FORGE_V2_EVAL_SLACK, research, default off)."""
import importlib

import pytest

from forge.tasks import sft_v2


def test_default_off_keeps_flat_rule(monkeypatch):
    monkeypatch.delenv("FORGE_V2_EVAL_SLACK", raising=False)
    importlib.reload(sft_v2)
    assert sft_v2.EVAL_SLACK_DEFAULT == "0" and not sft_v2._eval_slack()
    # flat rule only when no window is passed (the callback passes None while the switch is off)
    assert sft_v2.governed_eval_every(10, 5.7, 0.73, 264, None) == (131, None)


def test_short_run_keeps_dense_evals():
    # Phase D trdpo FIELD (Qwen2-0.5B): 264 steps at 0.73 s in a 2468 s window, ~5.7 s per eval
    every, info = sft_v2.governed_eval_every(10, 5.7, 0.73, 264, 2468.0)
    assert every == 10 and info["flat_rule"] == 131 and info["slack_s"] > 2000
    assert 264 / every * 5.7 <= 0.5 * (2468.0 - 264 * 0.73)   # spends at most half the slack


def test_window_bound_run_unchanged():
    # Phase D 3B FIELD: 1678 steps at 3.04 s in a 4792 s window -> no slack -> flat rule
    assert sft_v2.governed_eval_every(97, 11.0, 3.035, 1678, 4792.5) == (61, None)  # 61 < 97: callback keeps 97


@pytest.mark.parametrize("slack_frac", [0.05, 0.2, 0.5])
def test_never_sparser_than_flat_never_denser_than_plan(slack_frac):
    total, t_step, per_eval, planned = 500, 1.0, 8.0, 12
    window = total * t_step * (1 + slack_frac)
    every, _ = sft_v2.governed_eval_every(planned, per_eval, t_step, total, window)
    flat, _ = sft_v2.governed_eval_every(planned, per_eval, t_step, total, None)
    assert planned <= every <= max(planned, flat)


def test_switch_reads_env(monkeypatch):
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "1")
    assert sft_v2._eval_slack()
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "0")
    assert not sft_v2._eval_slack()


class _Clock:
    """Deadline stand-in whose remaining time falls as work (the seed-A soup) spends wall clock."""

    def __init__(self, remaining):
        self.remaining = remaining

    def spend(self, s):
        self.remaining -= s

    def remaining_hard(self):
        return self.remaining


def test_seed_b_window_refresh_after_seed_a_soup():
    # Codex review of 6912643 (P2): 1300 s window before the seed-A soup, soup spends 250 s, 800 steps at 1 s, 50 s/eval.
    finish_reserve = 100.0
    clock = _Clock(1300.0 + finish_reserve)
    stale_window = clock.remaining_hard() - finish_reserve
    clock.spend(250.0)                                   # greedy_soup + _reinit_lora
    window, fits = sft_v2.seed_b_window(clock, finish_reserve, min_window=240.0)
    assert window == 1050.0 and fits
    assert sft_v2.governed_eval_every(100, 50.0, 1.0, 800, stale_window)[0] == 160   # stale: too dense
    assert sft_v2.governed_eval_every(100, 50.0, 1.0, 800, window)[0] == 320         # refreshed


def test_seed_b_window_no_fit_after_soup():
    clock = _Clock(500.0)
    clock.spend(300.0)
    window, fits = sft_v2.seed_b_window(clock, finish_reserve=100.0, min_window=240.0)
    assert window == 100.0 and not fits


def test_refresh_is_gated_and_placed_after_seed_a_soup():
    import inspect
    src = inspect.getsource(sft_v2.run)
    flag = src.index("seed_b_no_fit = False")
    soup = src.index("loss_a, state_a, members_a = selection.greedy_soup(")
    reinit = src.index("_reinit_lora(model, _seed_base + 1000)")
    refresh = src.index("window, fits = seed_b_window(deadline, finish_reserve, min_window)")
    handler = src.index('_event_and_print("sft_v2_two_seed_failed", stage="a"')
    restore = src.index("restore_seed_a_or_fail(model, seed_a, pool, state)")
    plan = src.index('_event_and_print("sft_v2_plan"')
    cb = src.index("cb = SelectCallback(eval_every, total_steps, window_s=window)")
    assert flag < soup < reinit < refresh < handler < restore < plan < cb
    gate = src.rindex("if _eval_slack():", 0, refresh)
    assert refresh - gate < 400                           # the refresh only runs with FORGE_V2_EVAL_SLACK=1
    # the restore runs outside the broad stage-A handler (Codex re-review of 01a6b57): nothing in the try body restores
    body = src[soup:handler]
    assert "load_state(model, seed_a" not in body and "restore_seed_a_or_fail" not in body
    tail = src[restore:restore + 300]
    assert "seed_b_active = False" in tail and "two_seed = False" in tail and "break" in tail


class _Pool:
    def __init__(self):
        self.items = []


def _seed_a():
    return {"loss": 1.23, "step": 130, "state": {"w": 1}, "members": 3}


def test_restore_seed_a_success(monkeypatch):
    loaded = []
    monkeypatch.setattr(sft_v2.selection, "load_state", lambda m, st: loaded.append(st))
    pool, state = _Pool(), {"best": float("inf"), "best_step": 0}
    sft_v2.restore_seed_a_or_fail(object(), _seed_a(), pool, state)
    assert loaded == [{"w": 1}] and pool.items == [{"loss": 1.23, "step": 130, "state": {"w": 1}}]
    assert state == {"best": 1.23, "best_step": 130}


def test_restore_seed_a_failure_fails_closed(monkeypatch):
    # Codex repro: 260 s window - 50 s soup = 210 s < 240 s min window, then load_state raises. Before the fix the stage-A
    # handler swallowed it and seed B was planned on the re-initialised adapter (empty pool, best=inf).
    def boom(m, st):
        raise RuntimeError("size mismatch")
    monkeypatch.setattr(sft_v2.selection, "load_state", boom)
    events = []
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda name, **kv: events.append(name))
    clock = _Clock(260.0 + 100.0); clock.spend(50.0)
    window, fits = sft_v2.seed_b_window(clock, 100.0, 240.0)
    assert window == 210.0 and not fits
    pool, state = _Pool(), {"best": float("inf"), "best_step": 0}
    with pytest.raises(sft_v2.SeedARestoreError):
        sft_v2.restore_seed_a_or_fail(object(), _seed_a(), pool, state)
    assert pool.items == [] and state == {"best": float("inf"), "best_step": 0}   # nothing half-applied
    assert events == ["sft_v2_two_seed_restore_failed"]


def test_restore_failure_propagates_to_the_v2_fallback():
    # SeedARestoreError is not caught inside run(): the caller (forge/tasks/instruct.py) keeps the persisted artifact or
    # falls back to the validated LoRA path.
    import inspect
    from forge.tasks import instruct
    assert issubclass(sft_v2.SeedARestoreError, RuntimeError)
    src = inspect.getsource(instruct)
    assert '"sft_v2_failed"' in src and "sft_v2.trained_artifact_present(spec)" in src
