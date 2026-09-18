"""The production adapter path drops NEFTune on short-row corpora (2026-09-18: tinyllama/qa-news 3.77165 -> 1.64690)."""
import pytest

from forge.tasks import instruct


def _rows(n_tokens: int, n: int = 11) -> list:
    return [{"input_ids": list(range(n_tokens)), "labels": list(range(n_tokens))} for _ in range(n)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FORGE_LEGACY_NEFTUNE_ALPHA", raising=False)


def test_short_rows_disable_neftune():
    assert instruct.legacy_neftune_alpha(_rows(79), is_kl=False) is None
    assert instruct.legacy_neftune_alpha(_rows(255), is_kl=False) is None


def test_long_rows_keep_the_live_winning_setting():
    assert instruct.legacy_neftune_alpha(_rows(256), is_kl=False) == 5.0
    assert instruct.legacy_neftune_alpha(_rows(1200), is_kl=False) == 5.0


def test_gate_percentile_decides(monkeypatch):
    rows = _rows(100, n=17) + _rows(4000, n=3)  # short median, long tail (indic-like)
    monkeypatch.setattr(instruct, "LEGACY_NEFTUNE_GATE_PERCENTILE", 0.5)
    assert instruct.legacy_neftune_alpha(rows, is_kl=False) is None
    monkeypatch.setattr(instruct, "LEGACY_NEFTUNE_GATE_PERCENTILE", 0.95)
    assert instruct.legacy_neftune_alpha(rows, is_kl=False) == 5.0  # only uniformly short corpora lose NEFTune
    assert instruct.legacy_neftune_alpha(_rows(100, n=20), is_kl=False) is None


def test_kl_tasks_never_use_neftune():
    assert instruct.legacy_neftune_alpha(_rows(1200), is_kl=True) is None


def test_no_rows_keeps_previous_behaviour():
    assert instruct.legacy_neftune_alpha([], is_kl=False) == 5.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("FORGE_LEGACY_NEFTUNE_ALPHA", "0")
    assert instruct.legacy_neftune_alpha(_rows(1200), is_kl=False) is None
    monkeypatch.setenv("FORGE_LEGACY_NEFTUNE_ALPHA", "3")
    assert instruct.legacy_neftune_alpha(_rows(79), is_kl=False) == 3.0
