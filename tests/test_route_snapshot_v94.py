"""Route snapshot of release-candidate-v9.4-20261007 vs live 9504136.

v9.4 changes no dispatch decision: the tail reserve acts inside sft_v2.run, after the route, the recipe, the geometry, the
learning rate and the plan are fixed. The snapshot (the REAL dispatcher over every pool model x data size x row length x
detector verdict, tests/test_route_snapshot_v9.py) must therefore equal 9504136's, which equals d44a582's, in all 4,068
cells, with the tail reserve on (default) and off. Finite, mocked dispatch coverage; not run-time equivalence (the run-time
behaviour is tested in tests/test_v94_tail_reserve.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_route_snapshot_v9 as rs  # noqa: E402

from forge.tasks import sft_v2  # noqa: E402

ROUTE_SHA_9504136 = "d7d71afd23fee7fa8c40b2f71dc1d001f3ad7d024c773051f01aaf455f406c4e"  # = d44a582 (tests/test_v91_dev_pass_soup_restore.py)


@pytest.mark.parametrize("switch", [None, "1", "0"])
def test_route_snapshot_is_9504136s(monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv("FORGE_V2_TAIL_RESERVE", raising=False)
    else:
        monkeypatch.setenv("FORGE_V2_TAIL_RESERVE", switch)
    snap = rs.build_snapshot()
    assert len(snap) == 3816 and rs.route_sha(snap) == ROUTE_SHA_9504136
    assert sft_v2.tail_reserve_enabled() is (switch != "0")


def test_the_rule_is_route_independent():
    """One rule for every sft_v2 run: the switch is read once per run and nothing in the dispatcher or in the kwargs
    sft_v2.run receives selects it."""
    import inspect

    from forge.tasks import instruct

    assert "tail_reserve" not in inspect.getsource(instruct) and "TAIL_RESERVE" not in inspect.getsource(instruct)
    src = inspect.getsource(sft_v2.run)
    assert src.count("tail_reserve_enabled()") == 1
    line = next(l for l in src.splitlines() if "tail_reserve_enabled()" in l)
    assert line.strip() == "_tail_on = tail_reserve_enabled()"
