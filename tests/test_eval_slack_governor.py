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
