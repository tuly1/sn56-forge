"""release-candidate-v8 commit 1: lfm2 small band trains the full-FT default "family" recipe at LR x1 (phase-bb).

Cell-level byte identity against 7024722 is tests/test_route_snapshot_v8.py (declared change "lfm2_small_lr_x1")."""
from __future__ import annotations

import pytest

from forge.tasks import sft_v2

LFM12 = 1.1703  # LFM2.5-1.2B-Instruct / -Base / LFM2-1.2B (loaded params, fingerprint table)
LFM350 = 0.3545
LFM700 = 0.7425
LFM26 = 2.6972


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)


def fullft(params_b, **kw):
    return sft_v2.run_field_recipe("lfm2", params_b, field=True, recipe_key="family", **kw)


def test_constant_and_band():
    assert sft_v2.FULLFT_LFM2_SMALL_LR_MULT == 1.0
    fam = sft_v2.DAMAGED_BASE_FAMILIES["lfm2"]
    assert (fam["small_max_b"], fam["small_lr_mult"]) == (1.5, 2.0)  # the table itself is unchanged (cab930c routes)


@pytest.mark.parametrize("params_b", [LFM350, LFM700, LFM12, 1.5])
@pytest.mark.parametrize("damaged_base_route", [False, True])
def test_small_band_full_ft_is_lr_x1_fp32_master_bucket_unchanged(params_b, damaged_base_route):
    r = fullft(params_b, damaged_base_route=damaged_base_route)
    assert r["lr_mult"] == 1.0
    assert r["bf16"] is False  # fp32 master + fused AdamW, as in 7024722
    assert r["eff_batch"] == sft_v2.field_eff_batch(params_b)
    assert r["strategy"] == "full"


def test_equals_phase_bb_arm_b_on_7024722(monkeypatch):
    """Arm B = 7024722 + FORGE_V2_FIELD_LR_MULT=1. With the constant off (7024722) and that env, the recipe equals the
    new default without env."""
    new = fullft(LFM12)
    monkeypatch.setattr(sft_v2, "FULLFT_LFM2_SMALL_LR_MULT", None)
    assert fullft(LFM12)["lr_mult"] == 2.0  # 7024722 arm A
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "1")
    assert fullft(LFM12) == new


def test_revert_switch_is_7024722(monkeypatch):
    monkeypatch.setattr(sft_v2, "FULLFT_LFM2_SMALL_LR_MULT", None)
    for pb in (LFM350, LFM700, LFM12):
        assert fullft(pb)["lr_mult"] == 2.0


def test_env_override_keeps_precedence(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "2")
    assert fullft(LFM12)["lr_mult"] == 2.0
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "0.5")
    assert fullft(LFM12)["lr_mult"] == 0.5


def test_above_band_unchanged():
    r = fullft(LFM26)
    assert (r["lr_mult"], r["bf16"]) == (1.0, True)
    assert fullft(1.5001)["lr_mult"] == 1.0  # family lr_mult outside the band (was already 1.0)


def test_cab930c_research_routes_keep_x2():
    # FORGE_V2_FULLFT_DEFAULT=0 routes: no recipe_key
    assert sft_v2.run_field_recipe("lfm2", LFM12, field=True, damaged_base_route=True)["lr_mult"] == 2.0
    assert sft_v2.run_field_recipe("lfm2", LFM12, field=True, clean_small_lfm2_route=True)["lr_mult"] == 2.0
    assert sft_v2.CLEAN_SMALL_LFM2_LR_MULT == 2.0


def test_other_families_and_recipes_untouched():
    for mt, pb in (("qwen2", 0.494), ("qwen3", 1.72), ("gemma2", 2.61)):
        a = sft_v2.run_field_recipe(mt, pb, field=True, recipe_key="family")
        b = sft_v2.field_recipe(mt, pb, damaged_base=True)
        assert a == b
    # a non-family recipe key on an lfm2 shape never reaches the band override
    assert sft_v2._field_lr_mult("lfm2", damaged_base=True, params_b=LFM12, recipe_key="generic") == 2.0
    assert sft_v2._field_lr_mult("lfm2", damaged_base=True, params_b=LFM12, recipe_key=None) == 2.0
    # without damaged_base (not the family row) nothing changes
    assert sft_v2._field_lr_mult("lfm2", params_b=LFM12, recipe_key="family") == 1.0
