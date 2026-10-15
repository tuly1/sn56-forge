"""v10e = fa3e6ac + (a) the window-aware eval slack ON BY DEFAULT with the three risks of
research/20261013-eval-slack-default-scope/REPORT.md fixed (measured step time, tail reserve inside the slack, a minimum-epoch
guard on the overfit stop of a slack-densified phase) and (b) the geometry selector's large-model refusal only when the
large-model rule grew run A's batch or run A was window-bound (run S never applies the rule). All CPU; nothing here is GPU
evidence."""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from forge.tasks import geometry_select as gs
from forge.tasks import sft_v2

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v10b_gs_harness", HERE / "test_v10b_geometry_select.py")
v10b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10b)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k)


# ---- (a1) default on ---------------------------------------------------------------------------------------------------

def test_eval_slack_is_on_by_default_on_every_route(monkeypatch):
    assert sft_v2.EVAL_SLACK_DEFAULT == "1"
    assert sft_v2._eval_slack() is True and sft_v2._eval_slack(False) is True and sft_v2._eval_slack(True) is True


@pytest.mark.parametrize("value,on", [("0", False), ("1", True), ("", True), (" 0 ", False)])
def test_eval_slack_switch(monkeypatch, value, on):
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", value)
    assert sft_v2._eval_slack() is on and sft_v2._eval_slack(True) is on


# ---- (a2) slack from the MEASURED step time ----------------------------------------------------------------------------

def test_slack_uses_the_measured_step_time_not_the_probe():
    # 500 steps, probe 1.0 s, 900 s window: by the probe 400 s of slack; measured 1.6 s: 100 s
    planned, per_eval, total, window = 10, 8.0, 500, 900.0
    by_probe, i_probe = sft_v2.governed_eval_every(planned, per_eval, 1.0, total, window)
    by_meas, i_meas = sft_v2.governed_eval_every(planned, per_eval, 1.0, total, window, t_step_measured=1.6)
    assert i_probe["slack_s"] == 400.0 and i_probe["measured"] is False and i_probe["t_step_slack"] == 1.0
    assert i_meas["slack_s"] == 100.0 and i_meas["measured"] is True and i_meas["t_step_slack"] == 1.6
    assert by_meas > by_probe                                    # less real slack: sparser evals
    assert by_meas == max(planned, -(-int(total * per_eval) // int(0.5 * 100.0)))


def test_measured_step_time_that_leaves_no_slack_gives_the_flat_rule():
    flat, _ = sft_v2.governed_eval_every(10, 8.0, 1.0, 500, None)
    assert sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0, t_step_measured=1.8) == (flat, None)   # 900 - 900 = 0


@pytest.mark.parametrize("t_meas", [None, 0.0, -1.0])
def test_without_a_measurement_the_probe_is_used(t_meas):
    assert (sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0, t_step_measured=t_meas)[0]
            == sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0)[0])


def test_the_flat_rule_still_reads_the_probe():
    # the off path and the flat bound are fa3e6ac's: the measured time enters the slack only
    assert sft_v2.governed_eval_every(10, 5.7, 0.73, 264, None, t_step_measured=5.0) == (131, None)


# ---- (a3) the tail reserve inside the slack ----------------------------------------------------------------------------

def test_reserve_is_subtracted_from_the_slack():
    every0, i0 = sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0, t_step_measured=1.0)
    every1, i1 = sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0, t_step_measured=1.0, reserve_s=202.0)
    assert i0["slack_s"] == 400.0 and i1["slack_s"] == 198.0 and i1["reserve_s"] == 202.0 and every1 > every0


def test_a_reserve_that_eats_the_slack_gives_the_flat_rule():
    flat, _ = sft_v2.governed_eval_every(10, 8.0, 1.0, 500, None)
    assert sft_v2.governed_eval_every(10, 8.0, 1.0, 500, 900.0, t_step_measured=1.0, reserve_s=400.0) == (flat, None)


def test_densified_evals_never_spend_more_than_half_the_slack_left_after_the_reserve():
    for t_meas in (1.0, 1.1, 1.3):
        for reserve in (0.0, 96.0, 202.0):
            every, info = sft_v2.governed_eval_every(12, 8.0, 1.0, 500, 1000.0, t_step_measured=t_meas, reserve_s=reserve)
            if info:
                assert (500 / every) * 8.0 <= 0.5 * (1000.0 - 500 * t_meas - reserve) + 1e-9


# ---- (a4) the third risk: no minimum-epoch guard on a final overfit stop -----------------------------------------------

@pytest.mark.parametrize("densified,epochs_done,local_step,spe,wait", [
    (False, 0.0, 10, 100, False),    # flat-rule phase: the fa3e6ac stop
    (True, 0.0, 10, 100, True),      # densified, 0.1 epoch: wait
    (True, 0.0, 99, 100, True),
    (True, 0.0, 100, 100, False),    # one full epoch: the stop may end the phase
    (True, 0.6, 40, 100, False),     # earlier phases count
    (True, 0.0, 5, 0, False),        # degenerate steps per epoch: never blocks forever
])
def test_slack_overfit_guard(densified, epochs_done, local_step, spe, wait):
    assert sft_v2.slack_overfit_guard(densified=densified, epochs_done=epochs_done, local_step=local_step,
                                      steps_per_epoch=spe) is wait
    assert sft_v2.SLACK_OVERFIT_MIN_EPOCHS == 1.0


class _CurveHarness(v10b.Harness):
    """The real sft_v2.run on CPU; the recorder trainer runs `n_steps` steps, calls on_train_begin, sleeps `step_s` per
    step and honours should_training_stop. The DEV curve is `curve(step)`; each DEV eval sleeps `eval_s`."""

    def __init__(self, monkeypatch, tmp_path, *, n_rows, n_steps, curve, step_s=0.0, eval_s=0.0, **kw):
        super().__init__(monkeypatch, tmp_path, **kw)
        import sys

        import torch
        from forge.tuning import selection

        self.source = v10b.rows(n_rows)
        h, self.step_now = self, {"step": 0}

        def dev_loss(model, rs, collator, **k):
            if eval_s:
                time.sleep(eval_s)
            return curve(h.step_now["step"]), []

        monkeypatch.setattr(selection, "per_example_dev_loss", dev_loss)

        class CurveTrainer:
            def __init__(self, model, args, train_dataset, data_collator, callbacks=()):
                self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
                self.callbacks = callbacks
                self.state = NS(global_step=0, max_steps=n_steps)
                self.optimizer, self.lr_scheduler = object(), object()
                h.trainers.append({"micro": args.per_device_train_batch_size, "accum": args.gradient_accumulation_steps,
                                   "lr": args.learning_rate, "trained": False})

            def get_train_dataloader(self):
                return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, collate_fn=self.data_collator)

            def train(self):
                h.trainers[-1]["trained"] = True
                cb = self.callbacks[0]
                control = NS(should_training_stop=False)
                cb.on_train_begin(self.args, self.state, control, model=self.model)
                for i in range(1, n_steps + 1):
                    if step_s:
                        time.sleep(step_s)
                    self.state.global_step = i
                    h.step_now["step"] = i
                    cb.on_step_end(self.args, self.state, control, model=self.model)
                    if control.should_training_stop:
                        break

        monkeypatch.setattr(sys.modules["transformers"], "Trainer", CurveTrainer)


def _v_curve(best_step):
    return lambda step: 1.0 if step <= best_step else 1.3          # a minimum, then every later eval 30 % above it


def test_a_densified_phase_does_not_stop_on_overfit_inside_its_first_epoch(monkeypatch, tmp_path):
    h = _CurveHarness(monkeypatch, tmp_path, n_rows=12300, n_steps=200, curve=_v_curve(10), eval_s=0.02)
    first: dict = {}
    h.run(result=first)
    slack = h.ev("sft_v2_eval_governor_slack")
    assert slack and slack[0]["measured"] is True and slack[0]["reserve_s"] > 0       # densified, measured, reserved
    spe = h.ev("sft_v2_train_start")[0]["steps_per_epoch"]
    assert spe >= 40
    deferred, stop = h.ev("sft_v2_early_stop_deferred"), h.ev("sft_v2_early_stop")
    assert len(deferred) == 1 and deferred[0]["step"] < spe
    assert len(stop) == 1 and spe <= stop[0]["step"] < spe + 2 * h.ev("sft_v2_eval_governor_slack")[0]["eval_every"] + 1
    assert first["stopped_overfit"] is True and first["completed"] and first["saved"]


def test_with_the_slack_off_the_overfit_stop_is_fa3e6acs(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "0")
    h = _CurveHarness(monkeypatch, tmp_path, n_rows=12300, n_steps=200, curve=_v_curve(10), eval_s=0.02)
    first: dict = {}
    h.run(result=first)
    assert h.ev("sft_v2_eval_governor_slack") == [] and h.ev("sft_v2_early_stop_deferred") == []
    # flat rule: evals sparser than planned; three bad evals in a row end the phase wherever they fall
    gov = h.ev("sft_v2_eval_governor")
    assert gov and gov[0]["eval_every"] > 10
    assert len(h.ev("sft_v2_early_stop")) == 1 and first["stopped_overfit"] is True


def test_the_slack_reads_the_phases_measured_step_time(monkeypatch, tmp_path):
    # the probe says 0.01 s per step (harness); the trainer really takes about 0.03 s
    h = _CurveHarness(monkeypatch, tmp_path, n_rows=12300, n_steps=12, curve=lambda s: 1.0, step_s=0.03, eval_s=0.02)
    h.run(result={})
    info = h.ev("sft_v2_eval_governor_slack")[0]
    assert info["measured"] is True and 0.03 <= info["t_step_slack"] < 0.5


def test_the_callback_wiring_in_the_source():
    import inspect

    src = inspect.getsource(sft_v2.run)
    gov = src.index("widened, slack_info = governed_eval_every(self.eval_every, per_eval, t_step, self.total_steps,")
    assert "t_step_measured=self.t_meas, reserve_s=_slack_reserve" in src[gov:gov + 400]
    res = src.index("_slack_reserve = float(tail_reserve_plan(")
    assert res < gov and "pool_n=pool.k" in src[res:res + 300]
    assert src.count("slack_overfit_guard(densified=self.densified") == 1
    assert "self.densified = True" in src[gov:gov + 600]


# ---- (b) geometry-selector eligibility at >= 3.0 B --------------------------------------------------------------------

@pytest.mark.parametrize("over,reason", [
    (dict(large_model_rule=True, large_model_grew=False, window_bound=False), None),              # kept, not bound: runs
    (dict(large_model_rule=True, large_model_grew=False, window_bound=True), "large_model_batch_rule"),   # bound
    (dict(large_model_rule=True, large_model_grew=True, window_bound=False), "large_model_batch_rule"),   # grown
    (dict(large_model_rule=True, window_bound=False), "large_model_batch_rule"),                  # no key: read as grown
    (dict(large_model_rule=False, large_model_grew=False, window_bound=True), "first_run_window_bound"),
    (dict(large_model_rule=False, large_model_grew=False, window_bound=False), None),
])
def test_large_model_refusal_only_when_grown_or_window_bound(over, reason):
    assert gs.first_run_refusal(v10b.first_ok(**over)) == reason


def _big(monkeypatch, tmp_path, *, grow):
    from forge.tuning import large_model_rule as lm

    h = v10b.Harness(monkeypatch, tmp_path, params_b=3.93)
    h.kwargs["recipe_key"] = "llama"
    if not grow:
        monkeypatch.setattr(lm, "plan_growth", lambda **kw: None)
    return h


def test_run_a_at_3_9b_whose_batch_was_kept_goes_to_the_time_gate(monkeypatch, tmp_path):
    h = _big(monkeypatch, tmp_path, grow=False)
    first: dict = {}
    h.run(result=first)
    assert h.ev("sft_v2_large_model_batch_kept") and not h.ev("sft_v2_large_model_batch")
    assert first["large_model_rule"] is True and first["large_model_grew"] is False and not first["window_bound"]
    assert gs.first_run_refusal(first) is None


def test_run_a_at_3_9b_whose_batch_grew_is_refused(monkeypatch, tmp_path):
    from forge.tuning import large_model_rule as lm

    h = _big(monkeypatch, tmp_path, grow=True)
    monkeypatch.setattr(lm, "plan_growth", lambda **kw: (kw["eff_units"] * 2, {"lr_factor": 1.2}))
    first: dict = {}
    h.run(result=first)
    assert h.ev("sft_v2_large_model_batch")
    assert first["large_model_rule"] is True and first["large_model_grew"] is True
    assert gs.first_run_refusal(first) == "large_model_batch_rule"


def test_run_s_at_3_9b_never_applies_the_large_model_rule(monkeypatch, tmp_path):
    from forge.tuning import large_model_rule as lm

    h = _big(monkeypatch, tmp_path, grow=False)
    first = v10b.run_a(h)
    n_events = len(h.events)
    calls = []
    monkeypatch.setattr(lm, "plan_growth", lambda **kw: calls.append(kw) or (kw["eff_units"] * 4, {"lr_factor": 2.0}))
    h.dev_loss["value"] = 0.9
    sel, res = v10b.second(h, first), {}
    h.run(deadline=h.deadline(0.5), geometry_select=sel, result=res)
    later = [n for n, _ in h.events[n_events:]]
    assert calls == [] and "sft_v2_large_model_batch" not in later and "sft_v2_large_model_batch_kept" not in later
    assert res["rows_per_step"] == 32 and sel.record["pick"] == "S"
    a_lr, s_lr = h.ev("sft_v2_field_lr")[0], h.ev("sft_v2_field_lr")[1]
    assert s_lr["prior_lr"] == a_lr["prior_lr"]                     # same fixed FIELD LR as run A (the rule kept it)


def test_large_model_rule_itself_is_unchanged_below_and_in_run_a():
    import inspect

    src = inspect.getsource(sft_v2.run)
    assert "and not field_calibration and geometry_select is None:" in src
    assert sft_v2.large_model_rule_eligible(field=True, recipe_key="llama", cheap_full=False, g2c=False, strategy="full",
                                            params_b=3.93) is True
    assert sft_v2.large_model_rule_eligible(field=True, recipe_key="llama", cheap_full=False, g2c=False, strategy="full",
                                            params_b=2.9) is False


# ---- (c) fake-clock replays of the real run with the v10e DEFAULT (slack on): the finish is kept ----------------------------------

_t94_spec = importlib.util.spec_from_file_location("t94_for_v10e", HERE / "test_v94_tail_reserve.py")
t94 = importlib.util.module_from_spec(_t94_spec)
_t94_spec.loader.exec_module(t94)


@pytest.mark.parametrize("seed", [7, 11])
def test_default_slack_on_the_replay_timelines_keeps_soup_dev_pass_and_the_export_floor(monkeypatch, tmp_path, seed):
    """The v9.4 receipt timelines (window-sized runs): with the v10e default the run still soups over its pool, dev-passes all 400
    rows and ends above the 180 s export reserve (the flat-governor run of the same timeline is the reference only)."""
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "0")
    flat = t94._run(monkeypatch, tmp_path / "flat", t94.Timeline(seed), tail=True)
    monkeypatch.undo()
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "1")
    on = t94._run(monkeypatch, tmp_path / "on", t94.Timeline(seed), tail=True)
    assert t94._ev(on, "soup_result") and t94._ev(on, "sft_v2_soup_skipped") == []
    assert on.dev_pass_rows == [400] and on.saved >= 1
    assert t94._final_remaining(on) > 180.0
    for ev in t94._ev(on, "sft_v2_eval_governor_slack"):          # whenever it densifies, it says it measured and reserved
        assert ev["measured"] is True and ev["reserve_s"] > 0
