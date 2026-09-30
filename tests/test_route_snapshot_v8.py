"""Route snapshot vs 7024722 (release-candidate-v8-20260930).

For every pool model x small / medium / large data x short / long rows x clean / damaged / unknown detector verdict, with
the production switches (no env) and with FORGE_V2_FULLFT_DEFAULT=0 (the cab930c research switch), the REAL dispatcher
(forge.tasks.instruct.run, via the tests/test_fullft_default_route_table.py harness) runs to the handler call. Each cell
records the handler, the routing kwargs sft_v2.run receives and the recipe run() would apply
(sft_v2.run_field_recipe: precision, LR multiplier, bucket). The committed baseline
tests/data/route_snapshot_7024722.json.gz was generated from this harness on 7024722's routing (commit "v8 prep", which
only extracts run_field_recipe). Its canonical sha256 is the route hash ROUTE_SHA_7024722.

Each release-candidate-v8 change declares the cells it changes (CHANGES below, one entry per commit). The test FAILS if
any other cell differs from 7024722 in any byte, or if a declared cell changes in anything but the declared field.

Regenerate the baseline (only on 7024722 routing, i.e. with CHANGES empty):
    <venv python> tests/test_route_snapshot_v8.py --write-baseline
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fullft_default_route_table as rt  # noqa: E402  (the release route-table harness)

from forge.tasks import sft_v2  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "data" / "route_snapshot_7024722.json.gz"
# canonical sha256 of the uncompressed baseline (json.dumps(sort_keys=True, separators=(",", ":")))
ROUTE_SHA_7024722 = "ebdc33e82e7d6e2d6e510a8b625fa1186fa63a49037bbf6f01fb2b4bf3f6a4b6"
VERDICTS = ("clean", "damaged", "unknown")
SWITCHES = (None, "0")  # production default; FORGE_V2_FULLFT_DEFAULT=0 (cab930c research switch)
KW_KEYS = ("params_b", "per_gpu_gb", "strategy_override", "route_lr", "field_route", "damaged_base_route",
           "clean_small_qwen_route", "clean_small_lfm2_route", "recipe_key", "cheap_full")


def cell_key(entry: dict, data: str, rows: str, verdict: str, switch: str | None) -> str:
    return "|".join((entry["model"], data, rows, verdict, "default" if switch is None else f"switch={switch}"))


def _recipe_for(kw: dict) -> dict | None:
    if not kw.get("field_route"):
        return None  # run() reads the recipe only on the field route
    if not hasattr(sft_v2, "run_field_recipe"):
        return _recipe_7024722_inline(kw)  # cross-check mode on a pristine 7024722 checkout
    params = inspect.signature(sft_v2.run_field_recipe).parameters
    extra = {k: kw.get(k) for k in params if k not in ("model_type", "params_b", "field") and k in kw}
    return sft_v2.run_field_recipe(_MT[0], kw["params_b"], field=bool(kw["field_route"]), **extra)


_MT = [None]


def _recipe_7024722_inline(kw: dict) -> dict:
    """7024722 sft_v2.run lines 1566-1620, verbatim logic (used only when run_field_recipe does not exist)."""
    field, recipe_key = bool(kw["field_route"]), kw.get("recipe_key")
    fam = bool(field and recipe_key in ("family", "minicpm5"))
    gen = bool(field and recipe_key == "generic")
    dmg = bool(field and (kw.get("damaged_base_route") or kw.get("clean_small_qwen_route") or fam))
    csl = bool(field and kw.get("clean_small_lfm2_route") and not dmg)
    r = sft_v2.field_recipe(_MT[0], kw["params_b"], damaged_base=dmg, clean_small_lfm2=csl, generic=gen)
    if field and recipe_key == "llama":
        r = sft_v2.llama_recipe_precision(r, kw["params_b"])
    return r


def snapshot_cell(mp: MonkeyPatch, entry: dict, data: str, rows: str, verdict: str, switch: str | None) -> dict:
    seen: list[dict] = []
    out = rt.dispatch(mp, entry, data=data, rows=rows, verdict=verdict, switch=switch,
                      v2_handler=lambda **kw: seen.append(kw))
    routes = out["routes"]
    assert len(routes) == 1, (entry["model"], routes)
    r = routes[0]
    if r["handler"] != "sft_v2":
        return {"handler": r["handler"], "strategy": r["strategy"], "kw": None, "recipe": None}
    kw = seen[0]
    _MT[0] = entry["model_type"]
    return {"handler": "sft_v2", "strategy": r["strategy"], "kw": {k: kw.get(k) for k in KW_KEYS},
            "recipe": _recipe_for(kw)}


def build_snapshot() -> dict[str, dict]:
    mp = MonkeyPatch()
    snap: dict[str, dict] = {}
    try:
        for e in rt.POOL:
            for data in rt.DATA:
                for rows in rt.ROWS:
                    for v in VERDICTS:
                        for sw in SWITCHES:
                            snap[cell_key(e, data, rows, v, sw)] = snapshot_cell(mp, e, data, rows, v, sw)
                            mp.undo()
    finally:
        mp.undo()
    return snap


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def route_sha(snap: dict) -> str:
    return hashlib.sha256(canonical(snap)).hexdigest()


def load_baseline() -> dict[str, dict]:
    return json.loads(gzip.decompress(BASELINE.read_bytes()))


ENTRY = {e["model"]: e for e in rt.POOL}

# ---- declared changes (one per release-candidate-v8 commit) -----------------------------------------------------------
# name -> (predicate(cell_key, baseline_cell) -> bool, field path, expected new value)
CHANGES: dict[str, tuple] = {}


def _lfm2_small_fullft(key: str, base: dict) -> bool:
    model, _data, _rows, _verdict, switch = key.split("|")
    kw = base.get("kw") or {}
    return (switch == "default" and ENTRY[model]["model_type"] == "lfm2" and kw.get("recipe_key") == "family"
            and 0 < kw["params_b"] <= 1.5)


# commit "lfm2 small band": full-FT default "family" recipe, LR x2 -> x1 (phase-bb)
CHANGES["lfm2_small_lr_x1"] = (_lfm2_small_fullft, ("recipe", "lr_mult"), 1.0)


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


def test_baseline_is_the_recorded_7024722_route_hash():
    assert route_sha(load_baseline()) == ROUTE_SHA_7024722


def test_every_undeclared_cell_is_byte_identical_to_7024722(snapshots):
    base, cur = snapshots
    assert set(cur) == set(base)
    diffs = []
    for key in sorted(base):
        exp, _ = expected_cell(key, base[key])
        if canonical(cur[key]) != canonical(exp):
            diffs.append((key, base[key], cur[key]))
    assert not diffs, f"{len(diffs)} cells differ from 7024722 + declared changes; first: {diffs[:3]}"


def test_unchanged_subset_route_hash(snapshots):
    base, cur = snapshots
    keep = [k for k in base if not expected_cell(k, base[k])[1]]
    assert route_sha({k: cur[k] for k in keep}) == route_sha({k: base[k] for k in keep})


def test_each_declared_change_hits_cells(snapshots):
    base, _ = snapshots
    for name, (pred, _path, _value) in CHANGES.items():
        assert any(pred(k, base[k]) for k in base), name


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        assert not CHANGES, "the baseline is 7024722 routing: write it with no declared changes"
        snap = build_snapshot()
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_bytes(gzip.compress(canonical(snap), mtime=0))
        print(BASELINE, len(snap), route_sha(snap))
    else:
        snap = build_snapshot()
        print(len(snap), route_sha(snap))
