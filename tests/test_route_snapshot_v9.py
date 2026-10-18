"""Route snapshot vs 854bf7a (release-candidate-v9-20260930; live v8).

Same harness as tests/test_route_snapshot_v8.py (the REAL dispatcher, forge.tasks.instruct.run via the
tests/test_fullft_default_route_table.py doubles, for every pool model x small / medium / large data x short / long rows x
clean / damaged / unknown detector verdict, production switches and FORGE_V2_FULLFT_DEFAULT=0), plus a "rules" block per
cell for the v9 changes that act at run time or load time rather than in the dispatcher:

  large_model_batch   sft_v2.large_model_rule_eligible for the kwargs sft_v2.run receives (the static gate of the >= 3.0 B
                      batch/LR rule; at run time plan_growth may still keep the batch). False when the function is absent.
  planner_fixes       the phase-ao Qwen3.5-4B planner fixes are armed (full-FT default route: a recipe_key, i.e. every
                      in-scope sft_v2 cell at every size). False when sft_v2.starved_finish_reserve is absent.
  hub_kernels_guard   forge.model.load_base runs disable_hub_kernels_offline before construction (process-level, every
                      cell that loads a base; behaviour differs only where Transformers reaches lazy_load_kernel).

The committed baseline tests/data/route_snapshot_854bf7a.json (plain JSON, one cell per line) was generated with this file on 854bf7a (every rule
False). Each v9 commit declares the cells it changes in CHANGES; any other difference FAILS.

Regenerate the baseline (only on 854bf7a, i.e. with CHANGES empty):
    <venv python> tests/test_route_snapshot_v9.py --write-baseline
Write the delta table:
    <venv python> tests/test_route_snapshot_v9.py --delta OUT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fullft_default_route_table as rt  # noqa: E402
import test_route_snapshot_v8 as v8  # noqa: E402

from forge import model as forge_model  # noqa: E402
from forge.tasks import sft_v2  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "data" / "route_snapshot_854bf7a.json"
ROUTE_SHA_854BF7A = "060730bddcb759db55b01d020499bb8bf1c1901abe71e8faaa74fa0aaa1a8395"  # canonical sha256 of the baseline; v10f 2026-10-18: the earlier pin minus the 7 removed past-winner models' cells (research/20261018-f2f3-decision)
RULE_KEYS = ("large_model_batch", "planner_fixes", "hub_kernels_guard")


def rules_for(entry: dict, cell: dict) -> dict:
    kw = cell.get("kw")
    if kw is None:  # not sft_v2 (legacy plan / adapter handlers): only the load-time guard can apply
        return {"large_model_batch": False, "planner_fixes": False,
                "hub_kernels_guard": hasattr(forge_model, "disable_hub_kernels_offline")}
    strategy = cell["strategy"]
    field = bool(kw.get("field_route")) and strategy == "full"
    recipe_key = kw.get("recipe_key") if field else None
    fam = bool(field and recipe_key in ("family", "minicpm5"))
    g2c = bool(field and not kw.get("cheap_full")
               and sft_v2.gemma2_champion_admits(entry["model_type"], kw["params_b"], entry["vocab"],
                                                 damaged_base_route=bool(kw.get("damaged_base_route")) or fam))
    lm = False
    if hasattr(sft_v2, "large_model_rule_eligible"):
        lm = sft_v2.large_model_rule_eligible(field=field, recipe_key=recipe_key, cheap_full=bool(kw.get("cheap_full")),
                                              g2c=g2c, strategy=strategy, params_b=kw["params_b"])
    return {"large_model_batch": bool(lm),
            "planner_fixes": bool(hasattr(sft_v2, "starved_finish_reserve") and kw.get("recipe_key")),
            "hub_kernels_guard": hasattr(forge_model, "disable_hub_kernels_offline")}


def build_snapshot() -> dict[str, dict]:
    mp = MonkeyPatch()
    snap: dict[str, dict] = {}
    try:
        for e in rt.POOL:
            for data in rt.DATA:
                for rows in rt.ROWS:
                    for v in v8.VERDICTS:
                        for sw in v8.SWITCHES:
                            cell = v8.snapshot_cell(mp, e, data, rows, v, sw)
                            mp.undo()
                            cell["rules"] = rules_for(e, cell)
                            snap[v8.cell_key(e, data, rows, v, sw)] = cell
    finally:
        mp.undo()
    return snap


canonical = v8.canonical
route_sha = v8.route_sha
ENTRY = v8.ENTRY


def load_baseline() -> dict[str, dict]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


# ---- declared changes (one entry per release-candidate-v9 commit) -----------------------------------------------------
# name -> (predicate(cell_key, baseline_cell) -> bool, field path, expected new value)
# Each v9 commit declares its cells in ITS OWN test module as a module-level dict V9_ROUTE_CHANGES (same shape), so that
# reverting that commit removes the declaration with it and `git revert` of any single commit stays conflict-free.
CHANGES: dict[str, tuple] = {}


def _collect_declarations() -> None:
    import importlib

    here = Path(__file__).resolve().parent
    for p in sorted(here.glob("test_*.py")):
        if p.name == Path(__file__).name or "V9_ROUTE_CHANGES = " not in p.read_text():
            continue
        mod = importlib.import_module(p.stem)
        for name, decl in mod.V9_ROUTE_CHANGES.items():
            assert name not in CHANGES, f"duplicate v9 route change {name}"
            CHANGES[name] = decl


_collect_declarations()


def expected_cell(key: str, base: dict) -> tuple[dict, list[str]]:
    exp = json.loads(json.dumps(base))
    hits = []
    for name, (pred, path, value) in CHANGES.items():
        if pred(key, base):
            node = exp
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] = value
            hits.append(name)
    return exp, hits


@pytest.fixture(scope="module")
def snapshots():
    return load_baseline(), build_snapshot()


def test_baseline_is_the_recorded_854bf7a_route_hash():
    assert route_sha(load_baseline()) == ROUTE_SHA_854BF7A


def test_baseline_dispatch_equals_the_v8_snapshot_harness():
    """Without the rules block the 854bf7a baseline is v8's harness output: 7024722 + v8's declared changes."""
    base = load_baseline()
    v8_base = v8.load_baseline()
    assert set(base) == set(v8_base)
    for k in base:
        cell = {kk: vv for kk, vv in base[k].items() if kk != "rules"}
        assert canonical(cell) == canonical(v8.expected_cell(k, v8_base[k])[0]), k
        assert base[k]["rules"] == {r: False for r in RULE_KEYS}, k


def test_every_undeclared_cell_is_byte_identical_to_854bf7a(snapshots):
    base, cur = snapshots
    assert set(cur) == set(base)
    diffs = []
    for key in sorted(base):
        exp, _ = expected_cell(key, base[key])
        if canonical(cur[key]) != canonical(exp):
            diffs.append((key, base[key], cur[key]))
    assert not diffs, f"{len(diffs)} cells differ from 854bf7a + declared changes; first: {diffs[:3]}"


def test_unchanged_subset_route_hash(snapshots):
    base, cur = snapshots
    keep = [k for k in base if not expected_cell(k, base[k])[1]]
    assert route_sha({k: cur[k] for k in keep}) == route_sha({k: base[k] for k in keep})


def test_each_declared_change_hits_cells(snapshots):
    base, _ = snapshots
    for name, (pred, _path, _value) in CHANGES.items():
        assert any(pred(k, base[k]) for k in base), name


def delta(base: dict, cur: dict) -> dict:
    changed = {}
    for k in sorted(base):
        if canonical(base[k]) == canonical(cur[k]):
            continue
        rb, rc = base[k].get("rules", {}), cur[k].get("rules", {})
        changed[k] = {"change": expected_cell(k, base[k])[1],
                      "rules_854bf7a": rb, "rules_v9": rc,
                      "dispatch_identical": canonical({x: y for x, y in base[k].items() if x != "rules"})
                      == canonical({x: y for x, y in cur[k].items() if x != "rules"}),
                      "recipe_key": (cur[k].get("kw") or {}).get("recipe_key"),
                      "params_b": (cur[k].get("kw") or {}).get("params_b")}
    return changed


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        assert not CHANGES, "the baseline is 854bf7a: write it with no declared changes"
        snap = build_snapshot()
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_bytes(v8.readable(snap))
        print(BASELINE, len(snap), route_sha(snap))
    elif "--delta" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--delta") + 1])
        base, cur = load_baseline(), build_snapshot()
        ch = delta(base, cur)
        out.write_text(json.dumps({"baseline_route_sha_854bf7a": route_sha(base), "head_route_sha": route_sha(cur),
                                   "n_cells": len(cur), "n_changed": len(ch), "changed": ch}, indent=1, sort_keys=True))
        print(out, len(cur), len(ch), route_sha(cur))
    else:
        snap = build_snapshot()
        print(len(snap), route_sha(snap))
