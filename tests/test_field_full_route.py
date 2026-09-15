"""Field route gates (2026-09-15): full weights at the champion's geometry for
single-GPU non-KL instruct tasks up to 3.5B, only when FORGE_V2_FIELD=1."""
from __future__ import annotations

import pytest

from forge.tasks import sft_v2


def test_eff_batch_buckets(monkeypatch):
    monkeypatch.delenv("FORGE_V2_FIELD_EFF_BATCH", raising=False)
    assert sft_v2.field_eff_batch(0.49) == 140
    assert sft_v2.field_eff_batch(1.24) == 100
    assert sft_v2.field_eff_batch(2.7) == 48
    monkeypatch.setenv("FORGE_V2_FIELD_EFF_BATCH", "64")
    assert sft_v2.field_eff_batch(1.24) == 64


def test_route_off_by_default(monkeypatch):
    monkeypatch.delenv("FORGE_V2_FIELD", raising=False)
    assert not sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False)


@pytest.mark.parametrize("params_b,n_gpus,is_kl,expected", [
    (0.49, 1, False, True), (1.24, 1, False, True), (2.7, 1, False, True), (3.5, 1, False, True),
    (3.6, 1, False, False), (1.24, 2, False, False), (1.24, 1, True, False), (0.0, 1, False, False),
])
def test_route_gates(monkeypatch, params_b, n_gpus, is_kl, expected):
    monkeypatch.setenv("FORGE_V2_FIELD", "1")
    monkeypatch.delenv("FORGE_V2_FIELD_MAX_PARAMS_B", raising=False)
    assert sft_v2.field_full_route(params_b=params_b, n_gpus=n_gpus, is_kl=is_kl) is expected


def test_lfm_pin_env_override(monkeypatch):
    monkeypatch.delenv("FORGE_V2_TAKES_LFM25", raising=False)
    assert sft_v2.v2_takes_lfm25() is sft_v2.V2_TAKES_LFM25
    monkeypatch.setenv("FORGE_V2_TAKES_LFM25", "1")
    assert sft_v2.v2_takes_lfm25() is True
    monkeypatch.setenv("FORGE_V2_TAKES_LFM25", "0")
    assert sft_v2.v2_takes_lfm25() is False


def test_min_lr_rate_default(monkeypatch):
    monkeypatch.delenv("FORGE_V2_MIN_LR_RATE", raising=False)
    assert sft_v2._min_lr_rate() == 0.1
    assert sft_v2._min_lr_rate(0.25) == 0.25
    monkeypatch.setenv("FORGE_V2_MIN_LR_RATE", "0.3")
    assert sft_v2._min_lr_rate(0.25) == 0.3
