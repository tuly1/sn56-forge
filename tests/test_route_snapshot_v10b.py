"""Route snapshot vs 9504136 (release-candidate-v10b; live v9.2).

Same harness as tests/test_route_snapshot_v9.py (the REAL dispatcher for every pool model x small / medium / large data x
short / long rows x clean / damaged / unknown verdict, production switches and FORGE_V2_FULLFT_DEFAULT=0, with the v9 rules
block), plus one rule for the v10b change. The selector acts AFTER the first sft_v2.run of a task has exported its artifact,
so no dispatcher output moves: handler, strategy, kwargs and recipe of every cell are 9504136's.

  geometry_select   forge.tasks.geometry_select.armed for the first call instruct.run makes: the full-weight FIELD route,
                    rung "full", switch on, no explicit batch env. True = the run is ARMED: after it has exported, a second
                    training at about 32 rows per step MAY follow, if the first-run checks and both time gates pass at run
                    time (first_run_refusal, time_gate_stage1, time_gate_stage2). False when the module is absent (9504136).

Armed is static; whether a second run happens is decided at run time from measured seconds. The cells that can never get one
are stated here as well (`never_runs`): the >= 3.0 B large-model batch rule and the damaged Gemma2 champion geometry.

Regenerate the baseline (only with forge = 9504136):
    cd <9504136 checkout> && PYTHONPATH=$PWD <venv python> <this file> --write-baseline
Write the table of armed cells:
    <venv python> tests/test_route_snapshot_v10b.py --delta OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_route_snapshot_v9 as v9  # noqa: E402

try:
    from forge.tasks import geometry_select as gs  # noqa: E402
except ImportError:  # forge = 9504136 (baseline generation)
    gs = None

BASELINE = Path(__file__).resolve().parent / "data" / "route_snapshot_9504136_v10b.json"
ROUTE_SHA_BASELINE = "6abe499456d8021b4793439b4dd53d2cbf6613bff670ed7716f438a6f1c7b599"  # canonical sha256 of the baseline (9504136 cells + geometry_select False)
RULE = "geometry_select"
canonical, route_sha = v9.canonical, v9.route_sha


def rule_for(cell: dict) -> bool:
    kw = cell.get("kw")
    if kw is None or gs is None:
        return False
    return bool(gs.armed(field_route=bool(kw.get("field_route")), strategy=cell["strategy"],
                         rung="cheap_full" if kw.get("cheap_full") else "full"))


def build_snapshot() -> dict[str, dict]:
    snap = v9.build_snapshot()
    for cell in snap.values():
        cell["rules"][RULE] = rule_for(cell)
    return snap


def load_baseline() -> dict[str, dict]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def predicted(base: dict) -> bool:
    """Where v10b arms the selector, stated on the 9504136 cell alone: sft_v2, full weights, the FIELD route."""
    kw = base.get("kw")
    return bool(kw is not None and base["strategy"] == "full" and kw.get("field_route") and not kw.get("cheap_full"))


def never_runs(base: dict) -> str | None:
    """Armed cells where the first-run check refuses every time (static part of geometry_select.first_run_refusal)."""
    if base["rules"].get("large_model_batch"):
        return "large_model_batch_rule"
    return None


def expected_cell(base: dict) -> dict:
    exp = json.loads(json.dumps(base))
    exp["rules"][RULE] = predicted(base)
    return exp


@pytest.fixture(scope="module")
def snapshots():
    return load_baseline(), build_snapshot()


def test_baseline_is_the_recorded_9504136_snapshot_with_the_rule_off_everywhere():
    base = load_baseline()
    assert route_sha(base) == ROUTE_SHA_BASELINE and len(base) == 4068
    assert all(c["rules"][RULE] is False for c in base.values())


def test_dispatch_kwargs_recipes_and_v9_rules_are_untouched(snapshots):
    """Stripping the new key gives 9504136's cells: the selector moves no dispatcher output."""
    base, cur = snapshots
    assert set(cur) == set(base)
    for k in base:
        a = json.loads(json.dumps(base[k])); b = json.loads(json.dumps(cur[k]))
        a["rules"].pop(RULE); b["rules"].pop(RULE)
        assert canonical(a) == canonical(b), k


def test_every_cell_equals_9504136_plus_the_declared_rule(snapshots):
    base, cur = snapshots
    diffs = [k for k in sorted(base) if canonical(cur[k]) != canonical(expected_cell(base[k]))]
    assert not diffs, f"{len(diffs)} cells differ from 9504136 + the declared rule; first: {diffs[:3]}"


def test_armed_cells_are_exactly_the_full_weight_field_cells(snapshots):
    base, cur = snapshots
    armed = {k for k in cur if cur[k]["rules"][RULE]}
    assert armed == {k for k in base if predicted(base[k])} and armed
    assert all(cur[k]["strategy"] == "full" and cur[k]["kw"]["field_route"] for k in armed)
    assert not any(cur[k]["rules"][RULE] for k in cur if cur[k].get("kw") is None)           # no adapter / legacy handler


def test_cells_that_are_not_full_weight_field_do_not_change(snapshots):
    base, cur = snapshots
    other = [k for k in base if not predicted(base[k])]
    assert other and route_sha({k: cur[k] for k in other}) == route_sha({k: base[k] for k in other})


def test_scope_is_not_a_size_or_family_threshold(snapshots):
    """Armed cells span every size band; the time gate, not a params threshold, limits where a second run happens."""
    base, cur = snapshots
    armed = [k for k in cur if cur[k]["rules"][RULE]]
    sizes = [cur[k]["kw"]["params_b"] for k in armed]
    assert min(sizes) < 0.2 and max(sizes) >= 3.0
    assert any(s < 1.0 for s in sizes) and any(1.0 <= s < 3.0 for s in sizes)
    families = {v9.ENTRY[k.split("|")[0]]["model_type"] for k in armed}
    assert {"llama", "qwen2", "lfm2"} <= families and len(families) >= 6
    # stated limit: the >= 3.0 B large-model batch rule cells are armed but refused by the first-run check every time
    big = [k for k in armed if never_runs(base[k])]
    assert big and all(cur[k]["kw"]["params_b"] >= 3.0 for k in big)


def test_switch_off_restores_9504136_in_every_cell(monkeypatch, snapshots):
    base, _ = snapshots
    monkeypatch.setenv("FORGE_V2_GEOMETRY_SELECT", "0")
    snap = v9.build_snapshot()                       # build_snapshot clears FORGE_ env per cell: evaluate the gate here
    for cell in snap.values():
        cell["rules"][RULE] = rule_for(cell)
    assert route_sha(snap) == route_sha(base)


def test_a_research_batch_arm_is_never_armed(monkeypatch, snapshots):
    monkeypatch.setenv("FORGE_V2_FIELD_EFF_BATCH", "32")
    base, _ = snapshots
    assert not any(rule_for(c) for c in base.values())


def historical(cur: dict) -> list[str]:
    hit = []
    for tid, _date, model, p50 in v9.rt.HISTORICAL_R1:
        lengths = ["long" if p50 >= 256 else "short"] if p50 is not None else ["short", "long"]
        got = {cur[f"{model}|medium|{rl}|unknown|default"]["rules"][RULE] for rl in lengths}
        assert len(got) == 1, (tid, model)
        if got.pop():
            hit.append(tid)
    return hit


def test_historical_r1_coverage_of_the_armed_route(snapshots):
    """Release disclosure 'Historical coverage': which of the 34 recorded R1 draws (07-13 .. 09-21) are ARMED. Armed is an
    upper bound on second runs: the time gates decide per task and are not replayable from the draw list."""
    _, cur = snapshots
    hit = historical(cur)
    assert len(v9.rt.HISTORICAL_R1) == 34
    full_field = [t for t, _d, m, p50 in v9.rt.HISTORICAL_R1
                  if all(cur[f"{m}|medium|{rl}|unknown|default"]["strategy"] == "full"
                         and (cur[f"{m}|medium|{rl}|unknown|default"].get("kw") or {}).get("field_route")
                         for rl in (["long" if p50 >= 256 else "short"] if p50 is not None else ["short", "long"]))]
    assert hit == full_field and len(hit) == HISTORICAL_ARMED


HISTORICAL_ARMED = 33  # of 34; the one unarmed draw is an adapter exception


def summary(base: dict, cur: dict) -> dict:
    armed = [k for k in sorted(cur) if cur[k]["rules"][RULE]]
    models: dict[str, dict] = {}
    for k in armed:
        e = v9.ENTRY[k.split("|")[0]]
        row = models.setdefault(e["model"], {"model_type": e["model_type"], "params_b": e["params_b"], "armed_cells": 0,
                                             "never_runs_cells": 0})
        row["armed_cells"] += 1
        row["never_runs_cells"] += 1 if never_runs(base[k]) else 0
    return {"baseline": "9504136", "cells_total": len(base), "cells_armed": len(armed),
            "cells_armed_never_runs": sum(m["never_runs_cells"] for m in models.values()),
            "dispatcher_cells_changed": sum(1 for k in base if canonical({**cur[k], "rules": {}}) != canonical({**base[k], "rules": {}})),
            "historical_r1_armed": historical(cur), "models": models}


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        assert gs is None, "the baseline is generated with forge = 9504136"
        snap = build_snapshot()
        BASELINE.write_text("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v, sort_keys=True, separators=(',', ':'))}"
                                                 for k, v in sorted(snap.items())) + "\n}\n", encoding="utf-8")
        print(len(snap), route_sha(snap))
    elif "--delta" in sys.argv:
        base, cur = load_baseline(), build_snapshot()
        out = summary(base, cur)
        Path(sys.argv[sys.argv.index("--delta") + 1]).write_text(json.dumps(out, indent=1) + "\n")
        print(out["cells_armed"], "of", out["cells_total"], "cells armed;", out["cells_armed_never_runs"], "never run;",
              len(out["models"]), "models; historical", len(out["historical_r1_armed"]))
