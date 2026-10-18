"""release-candidate-v8 commit 2: MiniCPM5-1B trains at LR x0.5 only when the damage detectors read it CLEAN and name
openbmb/MiniCPM5-1B (phase-bc CONDITIONAL_CLEAN_ONLY). Every other verdict keeps the release LR x1.0.

Cell-level byte identity against 7024722 is tests/test_route_snapshot_v8.py (declared change "minicpm5_clean_lr_x05");
the real-detector checks on MiniCPM5-shaped safetensors are in the last tests of this file."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from forge.tasks import sft_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fullft_default_route_table as rt  # noqa: E402

MINICPM5 = 1.0806
REF = "openbmb/MiniCPM5-1B"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)


def recipe(base_damage, base_clean_ref, *, recipe_key="minicpm5", **kw):
    return sft_v2.run_field_recipe("llama", MINICPM5, field=True, recipe_key=recipe_key, base_damage=base_damage,
                                   base_clean_ref=base_clean_ref, **kw)


def test_constants():
    assert sft_v2.CLEAN_MINICPM5_LR_MULT == 0.5
    assert sft_v2.CLEAN_MINICPM5_REFERENCES == frozenset({"openbmb/minicpm5-1b"})
    assert sft_v2.CLEAN_MINICPM5_LR_DEFAULT == "1"


def test_clean_verdict_naming_minicpm5_halves_the_lr_and_nothing_else():
    new, old = recipe("clean", REF), recipe("damaged", REF)
    assert new["lr_mult"] == 0.5 and old["lr_mult"] == 1.0
    assert {k: v for k, v in new.items() if k != "lr_mult"} == {k: v for k, v in old.items() if k != "lr_mult"}
    assert new["bf16"] is True  # the F1 recipe (bf16 + paged 8-bit AdamW) is unchanged
    # reference_id forms accepted: table key, fingerprint source, lower case
    for ref in ("openbmb__MiniCPM5-1B", "openbmb/minicpm5-1b", " openbmb/MiniCPM5-1B "):
        assert recipe("clean", ref)["lr_mult"] == 0.5


@pytest.mark.parametrize("verdict,ref,reason", [
    ("damaged", REF, "verdict_damaged"),
    ("unknown", None, "verdict_unknown"),
    (None, None, "verdict_not_assessed"),
    ("clean", None, "clean_reference_not_minicpm5"),
    ("clean", "meta-llama/Llama-3.2-1B-Instruct", "clean_reference_not_minicpm5"),
    ("clean", "openbmb/MiniCPM5-1B-Base", "clean_reference_not_minicpm5"),
])
def test_any_doubt_keeps_the_release_lr(verdict, ref, reason):
    assert recipe(verdict, ref)["lr_mult"] == 1.0
    assert sft_v2.clean_minicpm5_lr_rule("minicpm5", verdict, ref) == (None, reason)


def test_rule_is_scoped_to_the_minicpm5_recipe():
    assert sft_v2.clean_minicpm5_lr_rule("llama", "clean", REF) == (None, "not_minicpm5_recipe")
    assert sft_v2.clean_minicpm5_lr_rule(None, "clean", REF) == (None, "not_minicpm5_recipe")
    # the cab930c damaged MiniCPM5 route (no recipe_key) and a non-field run never see the rule
    assert sft_v2.run_field_recipe("llama", MINICPM5, field=True, damaged_base_route=True, base_damage="clean",
                                   base_clean_ref=REF)["lr_mult"] == 1.0
    assert sft_v2.run_field_recipe("llama", MINICPM5, field=False, recipe_key="minicpm5", base_damage="clean",
                                   base_clean_ref=REF)["lr_mult"] == 1.0
    for key, mt, pb in (("family", "qwen2", 0.494), ("family", "lfm2", 1.1703), ("llama", "llama", 1.236),
                        ("generic", "phi3", 3.821)):
        a = sft_v2.run_field_recipe(mt, pb, field=True, recipe_key=key, base_damage="clean", base_clean_ref=REF)
        b = sft_v2.run_field_recipe(mt, pb, field=True, recipe_key=key)
        assert a == b, (key, mt)


def test_switch_off_is_7024722(monkeypatch):
    monkeypatch.setenv("FORGE_V2_CLEAN_MINICPM5_LR", "0")
    assert recipe("clean", REF)["lr_mult"] == 1.0
    assert sft_v2.clean_minicpm5_lr_rule("minicpm5", "clean", REF) == (None, "switch_off")


def test_env_lr_mult_keeps_precedence(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "1")
    assert recipe("clean", REF)["lr_mult"] == 1.0
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "0.5")
    assert recipe("damaged", REF)["lr_mult"] == 0.5  # = phase-bc arm lrh on 7024722


def test_equals_phase_bc_arm_lrh_on_7024722(monkeypatch):
    new = recipe("clean", REF)
    monkeypatch.setenv("FORGE_V2_CLEAN_MINICPM5_LR", "0")
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "0.5")
    assert recipe("clean", REF) == new


# ---- dispatcher: the verdict reaches sft_v2.run ------------------------------------------------------------------------
ENTRY = next(e for e in rt.POOL if e["model"] == REF)


@pytest.mark.parametrize("verdict", ["clean", "damaged", "unknown"])
def test_dispatcher_passes_the_verdict_and_reference(monkeypatch, verdict):
    seen = []
    rt.dispatch(monkeypatch, ENTRY, data="medium", rows="short", verdict=verdict, v2_handler=lambda **kw: seen.append(kw))
    kw = seen[0]
    assert kw["recipe_key"] == "minicpm5" and kw["field_route"] is True
    assert kw["base_damage"] == verdict
    assert kw["base_clean_ref"] == ("openbmb/minicpm5-1b" if verdict == "clean" else None)
    r = sft_v2.run_field_recipe("llama", kw["params_b"], field=True, recipe_key=kw["recipe_key"],
                                base_damage=kw["base_damage"], base_clean_ref=kw["base_clean_ref"])
    assert r["lr_mult"] == (0.5 if verdict == "clean" else 1.0)


def test_detection_not_run_keeps_release_lr(monkeypatch):
    """Damage detection switched off (FORGE_V2_DAMAGED_MINICPM5=0 -> no detection pass for llama): verdict None."""
    seen = []
    rt.dispatch(monkeypatch, ENTRY, data="medium", rows="short", verdict="clean",
                env={"FORGE_V2_DAMAGED_MINICPM5": "0"}, v2_handler=lambda **kw: seen.append(kw))
    kw = seen[0]
    assert kw["base_damage"] is None and kw["recipe_key"] == "minicpm5"
    assert sft_v2.run_field_recipe("llama", kw["params_b"], field=True, recipe_key="minicpm5",
                                   base_damage=None, base_clean_ref=None)["lr_mult"] == 1.0


def test_other_llama_shapes_never_get_a_verdict(monkeypatch):
    for model in ("meta-llama/Llama-3.2-1B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        e = next((x for x in rt.POOL if x["model"] == model), None)
        if e is None:
            continue
        seen = []
        rt.dispatch(monkeypatch, e, data="medium", rows="short", verdict="clean", v2_handler=lambda **kw: seen.append(kw))
        monkeypatch.undo()
        assert seen[0]["base_damage"] is None and seen[0]["recipe_key"] == "llama"


# ---- the real detector on MiniCPM5-shaped bytes -----------------------------------------------------------------------
def _write_st(path: Path, tensors: dict):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def test_real_detector_prefix_rule_on_a_minicpm5_table_entry(tmp_path):
    """The shipped rule (damage.fingerprint_assess) against a table entry built from the same bytes: identical -> clean
    naming the entry; any non-protected prefix change (scaling / pruning / noise / re-init all change it) -> damaged;
    a norm change -> unknown. Only "clean" + the MiniCPM5 reference unlocks x0.5."""
    import torch

    from forge.routing import damage

    g = torch.Generator().manual_seed(7)
    base = {"model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, generator=g).to(torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.randn(64, 64, generator=g).to(torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.ones(64, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(64, dtype=torch.bfloat16) * 1.5,
            "model.embed_tokens.weight": torch.randn(32, 64, generator=g).to(torch.bfloat16)}
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _write_st(ref_dir / "model.safetensors", base)
    fp = damage.local_fingerprint(str(ref_dir))
    table = {"models": {REF: {"tensors": fp["tensors"], "one_d": fp["one_d"]}}}

    def assess(tensors):
        d = tmp_path / f"t{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        _write_st(d / "model.safetensors", tensors)
        files = damage._local_files(str(d))
        rep = damage.fingerprint_assess(files, damage._read_local, table=table, fallback=False)
        verdict, _ = damage.combine({"fingerprint": rep})
        full = {"verdict": verdict, "detectors": {"fingerprint": rep}}
        return verdict, sft_v2.clean_base_reference(full)

    v, ref = assess(dict(base))
    assert (v, ref) == ("clean", "openbmb/minicpm5-1b") and recipe(v, ref)["lr_mult"] == 0.5
    scaled = dict(base, **{"model.layers.0.mlp.down_proj.weight": base["model.layers.0.mlp.down_proj.weight"] * 0.7})
    v, ref = assess(scaled)
    assert v == "damaged" and recipe(v, ref)["lr_mult"] == 1.0
    norm = dict(base, **{"model.norm.weight": base["model.norm.weight"] * 2})
    v, ref = assess(norm)
    assert v == "unknown" and recipe(v, ref)["lr_mult"] == 1.0


# ---- sft_v2.run on CPU: the event records the verdict and the applied LR -----------------------------------------------
def _cpu_run(monkeypatch, tmp_path, **run_kw):
    """The tiny-model CPU harness of tests/test_fullft_fallback_ladder.py (an LFM2-architecture double; the recipe key
    is forced to "minicpm5", so the family row it reads is lfm2's, but the rule and the event are exercised in run())."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ladder", Path(__file__).resolve().parent / "test_fullft_fallback_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    ev, _tr = ladder._run(monkeypatch, tmp_path, recipe_key="minicpm5", **run_kw)
    return ladder.lfm._ev(ev, "sft_v2_minicpm5_lr_rule"), ladder.lfm._ev(ev, "sft_v2_field_lr")


def test_run_emits_verdict_and_applied_lr(monkeypatch, tmp_path):
    rule_c, lr_c = _cpu_run(monkeypatch, tmp_path / "c", base_damage="clean", base_clean_ref="openbmb/minicpm5-1b")
    monkeypatch.undo()
    rule_d, lr_d = _cpu_run(monkeypatch, tmp_path / "d", base_damage="damaged", base_clean_ref=None)
    assert len(rule_c) == 1 and len(rule_d) == 1
    c, d = rule_c[0], rule_d[0]
    assert (c["verdict"], c["clean_ref"], c["reason"], c["rule_mult"], c["lr_mult"]) == (
        "clean", "openbmb/minicpm5-1b", "clean_minicpm5", 0.5, 0.5)
    assert (d["verdict"], d["reason"], d["rule_mult"]) == ("damaged", "verdict_damaged", None)
    assert c["lr"] == lr_c[0]["prior_lr"] and d["lr"] == lr_d[0]["prior_lr"]
    assert lr_c[0]["raw"] == pytest.approx(lr_d[0]["raw"])
    assert c["lr"] == pytest.approx(d["lr"] * 0.5 / d["lr_mult"])  # only the multiplier differs


def test_run_without_the_minicpm5_recipe_emits_nothing(monkeypatch, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ladder", Path(__file__).resolve().parent / "test_fullft_fallback_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    ev, _ = ladder._run(monkeypatch, tmp_path, recipe_key="family", base_damage="clean", base_clean_ref=REF)
    assert ladder.lfm._ev(ev, "sft_v2_minicpm5_lr_rule") == []
    assert ladder.lfm._ev(ev, "sft_v2_field_recipe")[0]["lr_mult"] != 0.5  # the family recipe's own multiplier
