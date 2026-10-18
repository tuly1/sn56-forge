"""release-candidate-v9.1-20261002: a failed dev pass restores the weights chosen before it (the soup), not pool.best.

d44a582 sft_v2.py:2793-2796 reloaded pool.best() on any dev-pass exception, discarding an accepted greedy soup. v9.1 loads
the pre-dev-pass state (_dp_pre = the in-memory chosen state; pool.best only when nothing was chosen).

Run-level tests reuse the tiny-LFM2 CPU harness (real sft_v2.run route, recipe, geometry, LR, planner, soup bookkeeping and
dev-pass control flow; training / timing / dev scoring / export are doubles). The soup is a real CPU state of the model with
every trainable tensor set to SOUP_V; the failing dev pass writes POISON_V into the weights first, then raises.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from forge.tasks import sft_v2
from forge.tuning import selection

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("defects_0929_v91", HERE / "test_release_defects_0929.py")
d29 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d29)
lfm = d29.lfm

SOUP_V, POISON_V = 0.125, 9.0  # exact in bf16 (snapshots are bf16)


def _harness(monkeypatch, tmp_path, *, dev_pass, soup=True, recipe_key="family"):
    """Run sft_v2.run with an accepted 2-member soup (or no soup) and ``dev_pass`` as the dev-pass body.
    Returns (events, model, saved_state_dicts, soup_state, pool_best_state)."""
    import torch

    seen: dict = {}

    def fake_soup(model, pool, eval_dev, *, time_budget_s, log):
        seen["pool_best"] = pool.best()["state"] if pool.best() is not None else None
        if not soup:
            return float("inf"), None, 0
        st = {n: torch.full_like(t, SOUP_V) for n, t in selection.snapshot_trainable(model).items()}
        selection.load_state(model, st)  # the real greedy_soup leaves the winning weights in the model
        seen["soup"] = st
        return 0.01, st, 2

    def poison_then(model, *a, **k):
        seen["model"] = model
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.fill_(POISON_V)
        dev_pass(model)

    saved: list = []

    def late(m):
        m.setenv("FORGE_DEV_PASS", "1")
        m.setattr(sft_v2, "stratified_split", lambda r, *a, **k: (r[:-2], r[-2:] * 60))  # >= 100 dev rows
        m.setattr(selection, "greedy_soup", fake_soup)
        m.setattr(sft_v2, "_dev_pass", poison_then)
        m.setattr(sft_v2, "save_adapter", lambda *a, state_dict=None, **k: saved.append(state_dict))

    ev, _ = d29._run(monkeypatch, tmp_path, late=late, recipe_key=recipe_key)
    return ev, seen.get("model"), saved, seen.get("soup"), seen.get("pool_best")


def _all_equal(model, value):
    return all(bool((p.detach() == value).all()) for p in model.parameters() if p.requires_grad)


def _raise(exc):
    def f(model):
        raise exc
    return f


def test_non_oom_dev_pass_failure_keeps_the_soup(monkeypatch, tmp_path):
    ev, model, saved, soup, _ = _harness(monkeypatch, tmp_path, dev_pass=_raise(ValueError("boom: not an OOM")))
    fail = lfm._ev(ev, "sft_v2_dev_pass_failed")
    assert len(fail) == 1 and fail[0]["restored"] == "chosen" and "ValueError" in fail[0]["error"]
    assert lfm._ev(ev, "sft_v2_dev_pass_done") == []
    assert _all_equal(model, SOUP_V)          # the poisoned dev-pass weights were replaced by the soup
    assert saved[-1] is soup                  # the final export is the soup state
    final = lfm._ev(ev, "sft_v2_final")[-1]
    assert final["saved"] is True and final["chosen_loss"] == 0.01


def test_oom_escaping_the_retry_also_keeps_the_soup(monkeypatch, tmp_path):
    """recipe_key route: dev_pass_with_oom_retry restores _dp_pre and retries once; if the retry fails too, the handler
    restores _dp_pre again (d44a582: pool.best)."""
    ev, model, saved, soup, _ = _harness(monkeypatch, tmp_path, dev_pass=_raise(RuntimeError("CUDA out of memory")))
    assert lfm._ev(ev, "sft_v2_dev_pass_failed")[0]["restored"] == "chosen"
    assert _all_equal(model, SOUP_V) and saved[-1] is soup


def test_without_a_soup_the_fallback_is_pool_best_as_in_d44a582(monkeypatch, tmp_path):
    ev, model, saved, soup, pool_best = _harness(monkeypatch, tmp_path, soup=False,
                                                 dev_pass=_raise(ValueError("boom")))
    assert soup is None and pool_best is not None
    # chosen = pool.best (no soup); .get: the same assertions pass on d44a582, whose event has no "restored" field
    assert lfm._ev(ev, "sft_v2_dev_pass_failed")[0].get("restored", "chosen") == "chosen"
    assert saved[-1] is pool_best
    assert not _all_equal(model, POISON_V)


def test_successful_dev_pass_is_unchanged(monkeypatch, tmp_path):
    ev, model, saved, soup, _ = _harness(monkeypatch, tmp_path, dev_pass=lambda m: None)
    assert lfm._ev(ev, "sft_v2_dev_pass_failed") == []
    assert len(lfm._ev(ev, "sft_v2_dev_pass_done")) == 1
    assert _all_equal(model, POISON_V)  # the dev-pass weights are the final weights (snapshot of the model)
    assert saved[-1] is not soup


def test_handler_source_restores_dp_pre_not_pool_best():
    src = Path(sft_v2.__file__).read_text()
    i = src.index('_event_and_print("sft_v2_dev_pass_failed"')
    block = src[i:i + 700]
    assert "chosen_state = _dp_pre" in block and "selection.load_state(model, chosen_state)" in block
    assert 'pool.best()["state"]' not in block


# ---- route snapshot: v9.1 changes no route (failure path only) -------------------------------------------------------
ROUTE_SHA_D44A582 = "d7d71afd23fee7fa8c40b2f71dc1d001f3ad7d024c773051f01aaf455f406c4e"  # ROUTE-SNAPSHOT-DELTA head_route_sha; v10f 2026-10-18: the earlier pin minus the 7 removed past-winner models' cells (research/20261018-f2f3-decision)


def test_route_snapshot_is_d44a582s():
    import sys

    sys.path.insert(0, str(HERE))
    import test_route_snapshot_v9 as rs

    snap = rs.build_snapshot()
    assert len(snap) == 3816 and rs.route_sha(snap) == ROUTE_SHA_D44A582
