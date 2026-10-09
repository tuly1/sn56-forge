"""release-candidate-v10c-merged-20261008: live v10b 2d6eaec + the tail reserve (a9de4f9) + the step cadence (d09fdc1).

Three switches, one per part:
  FORGE_V2_GEOMETRY_SELECT  default 1 (as live 2d6eaec)
  FORGE_V2_TAIL_RESERVE     default 1 (a safety mechanism: an earlier training stop so the finish stages fit)
  FORGE_V2_STEP_CADENCE     default 0 (no GPU result; a two-factor bundle per Codex RESP-148-1; never run with the selector)

1. Off switches: with the tail reserve and the cadence off, the real sft_v2.run on the fake clock is 2d6eaec event for event,
   on nine real receipt timelines. tests/data/v10c/golden_2d6eaec.json was written by this file's --write-golden on an
   extracted archive of 2d6eaec (it refuses to run where the tail-reserve code exists). This is the FIRST run of a task
   (geometry_select=None); the selector's own orchestration is covered by tests/test_v10b_geometry_select.py, unchanged.
2. Tail reserve on, cadence off: the a9de4f9 golden (tests/test_cadence_step_eval.py) still holds, with one declared
   difference - none on these runs, since they all have the tail reserve on. (release-candidate-v10d-20261009: no longer
   byte for byte. The reserve is smaller by the dev-pass estimate; tests/test_cadence_step_eval.py section 2 states the
   two declared differences. Item 1, switches off == 2d6eaec, is unchanged.)
3. The export-start line is written only with the tail reserve on (RESP-147-1 B2).
4. Run S (the selector's second run) starts its save measure from run A's slowest save.
5. Route snapshot: every cell is 2d6eaec's under every switch combination.

These are fake-clock replays with interpolated dev curves on a tiny model. They are not GPU evidence.
"""
from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import sys
from pathlib import Path

import pytest

from forge.tasks import sft_v2

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("t94_for_v10c", HERE / "test_v94_tail_reserve.py")
t94 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t94)
GOLDEN = HERE / "data/v10c/golden_2d6eaec.json"
BASE_COMMIT = "2d6eaecbb7d644a222fc1c928b1afef0de2ffed7"
ROUTE_SHA_2D6EAEC = "28da141c3a731d13410bc876457e0b7fdc4d656368e77a58c2d5d372c947f9f0"   # this harness on an extracted archive of 2d6eaec (4,068 cells, 1,540 armed); not the served route digest
RECEIPTS = ["bj_nx15_880b38e9_s7", "bv_minicpm5_2b_s7", "bt_q25_05bi_indic_s7", "cb_lfm25_350m_s7", "by_m2bb_01fdf13f_public_s7",
            "bk_lfm12_turkish_s7"]
t94.RECEIPT["t7"] = json.loads((HERE / "data/cadence/receipt_cf_t_01fdf13f_s7.json").read_text())
t94.Timeline.PERSIST["t7"] = 3.6
NAMES = ["ce_d0_s7", "ce_d0_s11", "cf_t_s7"] + RECEIPTS
NEW_EVENTS = ("sft_v2_tail_reserve", "sft_v2_tail_finish", "sft_v2_export_start", "sft_v2_tail_soup_budget", "sft_v2_tail_second_run_seed",
              "sft_v2_eval_cadence", "sft_v2_eval_cadence_cap")


def _timelines():
    out = {"ce_d0_s7": (t94.Timeline(7), {}), "ce_d0_s11": (t94.Timeline(11), {}), "cf_t_s7": (t94.Timeline("t7"), {})}
    for n in RECEIPTS:
        tl = t94.ReceiptTimeline(n)
        out[n] = (tl, dict(rows=tl.rows, n_dev=tl.n_dev, p_b=tl.p_b))
    return out


def _trace(out):
    return {"events": json.loads(json.dumps(out.events)), "args": json.loads(json.dumps(t94._args(out), default=str)),
            "saved": out.saved, "evals": out.evals, "dev_pass_rows": out.dev_pass_rows, "t_export_start": out.t_export_start, "clock": out.clock[0]}


# ---- defaults ---------------------------------------------------------------------------------------------------------

def test_default_states(monkeypatch):
    from forge.tasks import geometry_select as gs

    for k in ("FORGE_V2_GEOMETRY_SELECT", "FORGE_V2_TAIL_RESERVE", "FORGE_V2_STEP_CADENCE"):
        monkeypatch.delenv(k, raising=False)
    assert gs.enabled() is True and sft_v2.tail_reserve_enabled() is True and sft_v2.step_cadence_enabled() is False
    monkeypatch.setenv("FORGE_V2_STEP_CADENCE", "1")
    assert sft_v2.step_cadence_enabled() is True


# ---- 1. off switches == live 2d6eaec, event for event -----------------------------------------------------------------

@pytest.mark.parametrize("cadence_env", ["0", None])
@pytest.mark.parametrize("name", NAMES)
def test_switches_off_is_2d6eaec_event_for_event(monkeypatch, tmp_path, name, cadence_env):
    golden = json.loads(GOLDEN.read_text())
    assert golden["commit"] == BASE_COMMIT
    tl, kw = _timelines()[name]
    if cadence_env is None:      # the cadence default is off: unset must behave as 0. _run sets 0 when the name is unset,
        monkeypatch.setattr(sft_v2, "STEP_CADENCE_DEFAULT", sft_v2.STEP_CADENCE_DEFAULT)  # so pin the default instead
        assert sft_v2.STEP_CADENCE_DEFAULT == "0"
    else:
        monkeypatch.setenv("FORGE_V2_STEP_CADENCE", cadence_env)
    out = t94._run(monkeypatch, tmp_path, tl, tail=False, **kw)
    assert not any(n in NEW_EVENTS for n, _ in out.events)
    got, want = _trace(out), golden["runs"][name]
    assert len(got["events"]) == len(want["events"])
    for i, (g, w) in enumerate(zip(got["events"], want["events"])):
        assert g == w, (i, g, w)
    assert {k: got[k] for k in got if k != "events"} == {k: want[k] for k in want if k != "events"}


# ---- 3. the export-start line follows the tail switch ------------------------------------------------------------------

def test_export_start_line_only_with_the_tail_reserve_on(monkeypatch, tmp_path):
    off = t94._run(monkeypatch, tmp_path / "off", t94.Timeline(11), tail=False)
    monkeypatch.undo()
    on = t94._run(monkeypatch, tmp_path / "on", t94.Timeline(11), tail=True)
    assert t94._ev(off, "sft_v2_export_start") == [] and len(t94._ev(on, "sft_v2_export_start")) == 1
    assert off.t_export_start == on.t_export_start          # seed 11: the guard never fires, the save happens at the same time


# ---- 4. run S inherits run A's slowest save ------------------------------------------------------------------------------

@pytest.mark.parametrize("first,want", [(None, 0.0), ({}, 0.0), ({"persist_max_s": None}, 0.0), ({"persist_max_s": "x"}, 0.0),
                                        ({"persist_max_s": -3}, 0.0), ({"persist_max_s": float("nan")}, 0.0),
                                        ({"persist_max_s": float("inf")}, 0.0), ({"persist_max_s": 41.5}, 41.5), ("junk", 0.0)])
def test_second_run_persist_seed_is_finite_and_never_negative(first, want):
    assert sft_v2.second_run_persist_seed(first) == want


def test_a_slow_first_run_save_raises_run_s_export_floor_and_its_stop():
    seed = sft_v2.second_run_persist_seed({"persist_max_s": 120.0})
    assert sft_v2.tail_export_floor(150.0, 0.0) == 150.0                       # blind run S (2d6eaec + a9de4f9 as merged text)
    assert sft_v2.tail_export_floor(150.0, seed) > 150.0                       # seeded: the floor is raised before any save
    kw = dict(export_reserve_s=150.0, finish_reserve_s=400.0, eval_s=20.0, pool_n=3, trainable_params=2.5e9, dev_pass_s=60.0,
              t_step=2.0, window_s=4000.0)
    blind, seeded = sft_v2.tail_reserve_plan(persist_max_s=0.0, **kw), sft_v2.tail_reserve_plan(persist_max_s=seed, **kw)
    assert seeded["extra_s"] >= blind["extra_s"] and seeded["extra_s"] <= 0.15 * 4000.0 + 1e-6
    assert sft_v2.tail_reserve_plan(persist_max_s=sft_v2.second_run_persist_seed({"persist_max_s": 6.6}), **kw) == \
        sft_v2.tail_reserve_plan(persist_max_s=6.6, **kw)


def _g():
    sys.path.insert(0, str(HERE))
    import test_v10b_geometry_select as g

    return g


@pytest.mark.parametrize("tail", ["1", "0"])
def test_real_run_s_starts_from_run_a_slowest_save(monkeypatch, tmp_path, tail):
    """The selector tests' own harness (real sft_v2.run, recorder trainer): run A, then run S with the selector's record."""
    g = _g()
    monkeypatch.setenv("FORGE_V2_TAIL_RESERVE", tail)
    h = g.Harness(monkeypatch, tmp_path)
    first = g.run_a(h)
    if tail == "0":
        assert "persist_max_s" not in first and h.ev("sft_v2_export_start") == []      # 2d6eaec record, 2d6eaec events
    else:
        assert first["persist_max_s"] >= 0.0 and len(h.ev("sft_v2_export_start")) == 1
        first["persist_max_s"] = 150.0                                                  # a slow store: run A's save took 150 s
    h.dev_loss["value"] = 0.9
    sel, res = g.second(h, first), {}
    h.run(deadline=h.deadline(0.5), geometry_select=sel, result=res)
    assert sel.record["pick"] == "S" and res["saved"] and "persist_max_s" not in res   # run S hands nothing on
    seeds = h.ev("sft_v2_tail_second_run_seed")
    if tail == "0":
        assert seeds == [] and h.ev("sft_v2_export_start") == []
    else:
        assert seeds == [{"persist_max_s": 150.0}]
        start = h.ev("sft_v2_export_start")[-1]
        assert start["persist_max_s"] >= 150.0 and start["export_floor_s"] == sft_v2.tail_export_floor(180.0, 150.0) > 180.0


def test_wiring_of_the_handover_and_of_the_switches():
    src = inspect.getsource(sft_v2.run)
    assert src.count("tail_reserve_enabled()") == 1 and src.count("step_cadence_enabled()") == 1
    # run S: seeded before any training, only with the tail reserve on
    i = src.index("if geometry_select is not None and _tail_on:")
    assert 'state["persist_max"] = second_run_persist_seed(getattr(geometry_select, "first", None))' in src[i:i + 500]
    assert i < src.index("stop_at = _StopAt(") or i < src.index("def persist(")
    # run A: handed over after its final save, only with the tail reserve on, never by run S
    j = src.index("if _tail_on and geometry_select is None:")
    assert 'result["persist_max_s"]' in src[j:j + 500] and src.index('ok = persist(model, final_state') < j
    # run S still saves nothing before it has won, and its one save still carries the generation marker
    assert "if geometry_select is not None and geometry_select.defer_persist:\n            return False" in src
    assert '"sft_v2_final" if geometry_select is None else geometry_select.truth_reason' in src
    # the selector module does not read any of the new switches
    from forge.tasks import geometry_select as gs
    from forge.tasks import instruct
    for mod in (gs, instruct):
        body = inspect.getsource(mod)
        assert "TAIL_RESERVE" not in body and "STEP_CADENCE" not in body and "persist_max" not in body


def test_a_run_the_tail_reserve_ends_is_window_bound_for_the_selector():
    """A first run stopped by the tail reserve neither completed its plan nor stopped on DEV: the selector reads it as
    window-bound and refuses the second run (geometry_select.first_run_refusal), like a clock-ended run on 2d6eaec."""
    from forge.tasks import geometry_select as gs

    src = inspect.getsource(sft_v2.run)
    assert 'window_bound=bool(not state["last_phase_complete"] and not state["stopped_overfit"])' in src
    assert "window_bound" in inspect.getsource(gs.first_run_refusal)


# ---- 5. route snapshot ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tail,cad", list(itertools.product([None, "0", "1"], [None, "1"])))
def test_route_snapshot_is_2d6eaecs_under_every_switch(monkeypatch, tail, cad):
    sys.path.insert(0, str(HERE))
    import test_route_snapshot_v10b as rs

    for k, v in (("FORGE_V2_TAIL_RESERVE", tail), ("FORGE_V2_STEP_CADENCE", cad)):
        monkeypatch.delenv(k, raising=False) if v is None else monkeypatch.setenv(k, v)
    snap = rs.build_snapshot()
    assert len(snap) == 4068 and rs.route_sha(snap) == ROUTE_SHA_2D6EAEC


def _write_golden():
    if hasattr(sft_v2, "tail_reserve_enabled") or hasattr(sft_v2, "step_cadence"):
        raise SystemExit("refusing: this checkout has the tail-reserve or cadence code; the golden is written on 2d6eaec only")
    import tempfile

    from _pytest.monkeypatch import MonkeyPatch

    runs = {}
    for name, (tl, kw) in _timelines().items():
        mp = MonkeyPatch()
        with tempfile.TemporaryDirectory() as td:
            runs[name] = _trace(t94._run(mp, Path(td), tl, tail=False, **kw))
        mp.undo()
    Path(sys.argv[2]).write_text(json.dumps({"commit": sys.argv[3], "note": "fake-clock traces of the real sft_v2.run on an extracted archive of the live release (no tail reserve, no cadence code)", "runs": runs}) + "\n")
    print("golden written", sys.argv[3], {k: len(v["events"]) for k, v in runs.items()})


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--write-golden":
    _write_golden()
