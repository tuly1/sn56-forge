"""candidate-eval-cadence-20261007 (behind v9.4 a9de4f9): a step-based dev-eval cadence (FORGE_V2_STEP_CADENCE, default on;
0 = the base commit).

1. pure rules: step_cadence, cadence_cap_interval, overfit_patience_reached;
2. off switch == base commit a9de4f9: the real sft_v2.run on the fake clock (harness of tests/test_v94_tail_reserve.py), eight
   receipt timelines, compared event for event with tests/data/cadence/golden_a9de4f9.json, which was written by running this
   file's --write-golden on a checkout of a9de4f9 (it refuses to run where the cadence code exists);
3. cadence on: the two real seed-7 runs of 01fdf13f (live 9504136: ran to the clock; v9.4 6b1f603: overfit stop at 980) replayed
   on a 1.0 x and a 1.3 x node; the hard cap on eval time.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from forge.tasks import sft_v2

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("t94_for_cadence", HERE / "test_v94_tail_reserve.py")
t94 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t94)
GOLDEN = HERE / "data/cadence/golden_a9de4f9.json"
RECEIPTS = ["bj_nx15_880b38e9_s7", "bv_minicpm5_2b_s7", "bt_q25_05bi_indic_s7", "cb_lfm25_350m_s7", "by_m2bb_01fdf13f_public_s7",
            "bk_lfm12_turkish_s7"]
CAD_EVENTS = ("sft_v2_eval_cadence", "sft_v2_eval_cadence_cap")
t94.RECEIPT["t7"] = json.loads((HERE / "data/cadence/receipt_cf_t_01fdf13f_s7.json").read_text())   # the 6b1f603 run at seed 7
t94.Timeline.PERSIST["t7"] = 3.6


def _timelines():
    """name -> (timeline, run kwargs): seeds 7 and 11 of 01fdf13f (delivered base) and six more receipts."""
    out = {"ce_d0_s7": (t94.Timeline(7), {}), "ce_d0_s11": (t94.Timeline(11), {})}
    for n in RECEIPTS:
        tl = t94.ReceiptTimeline(n)
        out[n] = (tl, dict(rows=tl.rows, n_dev=tl.n_dev, p_b=tl.p_b))
    return out


def _trace(out):
    return {"events": json.loads(json.dumps(out.events)), "args": json.loads(json.dumps(t94._args(out), default=str)),
            "saved": out.saved, "evals": out.evals, "dev_pass_rows": out.dev_pass_rows, "t_export_start": out.t_export_start, "clock": out.clock[0]}


def _eval_steps(out):
    return [e["step"] for e in t94._ev(out, "sft_v2_eval")]


# ---- 1. pure rules ---------------------------------------------------------------------------------------------------

def test_switch(monkeypatch):
    monkeypatch.delenv("FORGE_V2_STEP_CADENCE", raising=False)
    assert sft_v2.step_cadence_enabled() is True
    monkeypatch.setenv("FORGE_V2_STEP_CADENCE", "0")
    assert sft_v2.step_cadence_enabled() is False


def test_interval_is_planned_times_a_power_of_two_and_ignores_probe_noise():
    geo = sft_v2.Geometry(14, 4, 1728, True, True, "adamw_torch_fused")
    seen = set()
    for t_probe in (2.4806, 2.4518, 2.4825, 2.4594, 2.4306, 2.4275):      # six real timing probes of this cell (four runs measured 74, 70, 73, 70)
        est = sft_v2._estimate_eval_seconds(400, t_probe, geo)
        every, info = sft_v2.step_cadence(50, est, t_probe, 1206)
        seen.add(every)
        assert info["planned"] == 50 and every == 50 * 2 ** info["k"] and 69 <= info["target"] <= 76
    assert seen == {100}
    for slow in (1.3, 1.6):                                                # a slower node: same interval here
        est = sft_v2._estimate_eval_seconds(400, 2.48 * slow, geo)
        assert sft_v2.step_cadence(50, est, 2.48 * slow, 1206)[0] == 100
    assert sft_v2.step_cadence(86, 2.6, 1.06, 2079)[0] == 86               # cheap evals: the planned interval (k = 0)
    assert sft_v2.step_cadence(50, 10.0, None, 1206)[0] == 50              # no timing probe: planned
    every, info = sft_v2.step_cadence(10, 190.0, 20.0, 50)                 # very slow evals: far beyond the phase
    assert every == 10 * 2 ** info["k"] and every >= info["target"] >= 159


def test_never_denser_than_planned_never_sparser_than_twice_the_base_target():
    for planned in (10, 31, 50, 86, 200):
        for est in (0.5, 3.0, 11.0, 60.0, 300.0):
            for t in (0.3, 1.0, 2.5, 20.0):
                every, info = sft_v2.step_cadence(planned, est, t, 2000)
                assert every >= planned and (every == planned or every < 2 * info["target"])


def test_cap_only_doubles_and_bounds_the_projection():
    f = sft_v2.cadence_cap_interval
    assert f(100, spent_s=20.0, evals_done=2, steps_left=1000, cap_s=333.0) == 100          # 20 + 10 x 10 = 120 <= 333
    assert f(100, spent_s=200.0, evals_done=2, steps_left=1000, cap_s=333.0) == 800         # 100 s per eval: one more fits
    assert f(100, spent_s=400.0, evals_done=2, steps_left=1000, cap_s=333.0) > 1000         # already over: no further eval in the phase
    assert f(100, spent_s=0.0, evals_done=0, steps_left=1000, cap_s=333.0) == 100
    for spent, n, left, cap in ((50.0, 1, 5000, 300.0), (9.0, 3, 900, 20.0), (120.0, 4, 10_000, 125.0)):
        every = f(64, spent_s=spent, evals_done=n, steps_left=left, cap_s=cap)
        assert every >= 64 and (every // 64) & ((every // 64) - 1) == 0                      # a power-of-two multiple
        assert spent > cap or spent + (left // every) * spent / n <= cap


def test_patience_in_steps_equals_three_evals_at_the_planned_cadence():
    p = sft_v2.overfit_patience_reached
    # planned cadence (interval == planned): bad evals at +I, +2I, +3I -> the base rule exactly, step-wise or not
    for stepwise in (False, True):
        assert [p(c, c * 50, 50, stepwise) for c in (1, 2, 3)] == [False, False, True]
    # doubled cadence: two bad evals already span 4 planned intervals -> stop (base: would wait for a third = 6 intervals)
    assert p(2, 200, 50, True) is True and p(2, 200, 50, False) is False and p(1, 100, 50, True) is False
    assert p(3, 10, 50, False) is True


def test_wiring():
    import inspect

    src = inspect.getsource(sft_v2.run)
    assert src.count("step_cadence_enabled()") == 1 and "_cad_on = step_cadence_enabled()" in src
    assert "self.governed = bool(_cad_on)" in src                                    # off: the base governor block runs untouched
    assert "if not self.governed and t_step:" in src and "governed_eval_every(self.eval_every, per_eval, t_step, self.total_steps," in src
    assert "step_cadence(eval_every, dev_eval_est, t_per_step, total_steps," in src   # the probe and the plan only: not t_step, not dt
    assert "overfit_patience_reached(state[\"overfit\"], step - state.get(\"overfit_from\", step), self.planned_every, _cad_on)" in src
    for fn in (sft_v2.step_cadence, sft_v2.cadence_cap_interval, sft_v2.overfit_patience_reached):
        body = inspect.getsource(fn).lower()
        assert not any(w in body for w in ("model_type", "params_b", "recipe", "qwen", "llama", "lfm", "gemma"))


# ---- 2. off switch == base commit a9de4f9 ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ce_d0_s7", "ce_d0_s11"] + RECEIPTS)
def test_off_switch_is_the_base_commit_event_for_event(monkeypatch, tmp_path, name):
    golden = json.loads(GOLDEN.read_text())
    assert golden["commit"] == "a9de4f9a025e1ad7225063c766e61a2f1678cc03"
    tl, kw = _timelines()[name]
    monkeypatch.setenv("FORGE_V2_STEP_CADENCE", "0")
    out = t94._run(monkeypatch, tmp_path, tl, tail=True, **kw)
    got = _trace(out)
    assert not any(n in CAD_EVENTS for n, _ in out.events)
    want = golden["runs"][name]
    assert len(got["events"]) == len(want["events"])
    for i, (g, w) in enumerate(zip(got["events"], want["events"])):
        assert g == w, (i, g, w)
    assert {k: got[k] for k in got if k != "events"} == {k: want[k] for k in want if k != "events"}


# ---- 3. cadence on ----------------------------------------------------------------------------------------------------

def _pair(monkeypatch, tmp_path, key, slow):
    monkeypatch.setenv("FORGE_V2_STEP_CADENCE", "1")
    out = t94._run(monkeypatch, tmp_path / f"{key}-{slow}", t94.Timeline(key, slow=slow), tail=True)
    monkeypatch.undo()
    return out


def test_the_two_seed7_runs_sample_the_same_steps_on_a_fast_and_a_slow_node(monkeypatch, tmp_path):
    """Base: the live run evaluated every 74 steps and ran to the clock; the 6b1f603 run evaluated every 70 and met the
    overfit stop at 980. Cadence on: both timelines, on a 1.0 x and a 1.3 x node, evaluate at multiples of 100 until they
    stop. (The dev values at those steps are linear interpolations of each receipt's own curve: an approximation.)"""
    assert t94._rev(7, "sft_v2_eval_governor")[0]["eval_every"] == 74 and t94._rev("t7", "sft_v2_eval_governor")[0]["eval_every"] == 70
    runs = {(k, s): _pair(monkeypatch, tmp_path, k, s) for k in (7, "t7") for s in (1.0, 1.3)}
    for (k, s), out in runs.items():
        cad = t94._ev(out, "sft_v2_eval_cadence")
        assert len(cad) == 1 and (cad[0]["planned"], cad[0]["k"], cad[0]["eval_every"]) == (50, 1, 100)
        assert t94._ev(out, "sft_v2_eval_governor") == [] and t94._ev(out, "sft_v2_eval_cadence_cap") == []
        steps = _eval_steps(out)
        assert all(st % 100 == 0 for st in steps[:-1]) and steps[:5] == [100, 200, 300, 400, 500]
        assert t94._ev(out, "sft_v2_final")[0]["saved"] is True and t94._export_start_remaining(out, t94.Timeline(k, slow=s)) >= 180.0 - 1.0
    # same node speed: the two timelines share every eval step up to the earlier stop
    for s in (1.0, 1.3):
        a, b = _eval_steps(runs[(7, s)]), _eval_steps(runs[("t7", s)])
        n = min(len(a), len(b)) - 1
        assert n >= 5 and a[:n] == b[:n]
    # node speed does not move the schedule either
    for k in (7, "t7"):
        a, b = _eval_steps(runs[(k, 1.0)]), _eval_steps(runs[(k, 1.3)])
        n = min(len(a), len(b)) - 1
        assert a[:n] == b[:n]


def test_eval_time_is_bounded_on_a_node_whose_evals_are_far_slower_than_estimated(monkeypatch, tmp_path):
    """Evals 12 x slower than the estimate assumes (98 s each instead of ~11 s) in a 0.5 h task: the cap doubles the
    interval after the first eval; total eval + save time stays within the cap plus one eval in flight; the run exports."""
    monkeypatch.setenv("FORGE_V2_STEP_CADENCE", "1")
    tl = t94.Timeline(7, hard_t=1800.0, eval_scale=12.0)
    out = t94._run(monkeypatch, tmp_path, tl, tail=True)
    cap = t94._ev(out, "sft_v2_eval_cadence")[0]["cap_s"]
    assert cap == pytest.approx(0.10 * t94._ev(out, "sft_v2_plan")[0]["window_s"], abs=0.2)
    hits = t94._ev(out, "sft_v2_eval_cadence_cap")
    assert hits and hits[0]["eval_every"] > hits[0]["was"]
    in_training = [e for e in t94._ev(out, "sft_v2_eval")]
    spent = sum(e["eval_s"] for e in in_training) + tl.persist_s * out.saved
    assert spent <= cap + max(tl.eval_s) + tl.persist_s
    assert t94._ev(out, "sft_v2_final")[0]["saved"] is True and t94._steps(out) > 0


def test_cap_not_reached_on_the_real_timelines(monkeypatch, tmp_path):
    for k in (7, 11):
        out = _pair(monkeypatch, tmp_path, k, 1.3)
        assert t94._ev(out, "sft_v2_eval_cadence_cap") == []


def _write_golden():
    if hasattr(sft_v2, "step_cadence"):
        raise SystemExit("refusing: this checkout has the cadence code; the golden is written on the base commit a9de4f9 only")
    import subprocess
    import tempfile

    from _pytest.monkeypatch import MonkeyPatch

    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    runs = {}
    for name, (tl, kw) in _timelines().items():
        mp = MonkeyPatch()
        with tempfile.TemporaryDirectory() as td:
            runs[name] = _trace(t94._run(mp, Path(td), tl, tail=True, **kw))
        mp.undo()
    Path(sys.argv[2]).write_text(json.dumps({"commit": commit, "note": "fake-clock traces of the real sft_v2.run on the base commit (tail reserve on)", "runs": runs}) + "\n")
    print("golden written", commit, {k: len(v["events"]) for k, v in runs.items()})


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--write-golden":
    _write_golden()
