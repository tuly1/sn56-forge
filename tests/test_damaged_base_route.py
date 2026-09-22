"""Damaged-base full route (2026-09-22): on a validator-damaged base an enabled family takes the FIELD full route with
the recipe of the 2026-09-22 pairs (bf16 + paged 8-bit AdamW from 0.3 B, LR x1.0, no family vocabulary / minimum-size
gate); the row / token floors still apply. Clean and unknown bases keep today's routes. Families are data-driven
(sft_v2.DAMAGED_BASE_FAMILIES): qwen2 and lfm2 on, llama / gemma2 / qwen3 off."""
import importlib.util
import json
import struct
from pathlib import Path

import pytest

from forge.routing import base_identity as bid
from forge.routing import damage
from forge.tasks import sft_v2

ENV = ("FORGE_V2_FIELD", "FORGE_V2_FIELD_TYPES", "FORGE_V2_FIELD_MIN_ROWS", "FORGE_V2_FIELD_MIN_TOKENS",
       "FORGE_V2_FIELD_MAX_PARAMS_B", "FORGE_V2_FIELD_IDENTICAL_BASE", "FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B",
       "FORGE_V2_FIELD_BF16", "FORGE_V2_FIELD_BF16_MIN_B", "FORGE_V2_FIELD_LR_MULT", "FORGE_V2_DAMAGED_BASE_ROUTE",
       "FORGE_V2_DAMAGED_BASE_TYPES", "FORGE_V2_DAMAGE_DETECTORS")

# (model_type, params_b, vocab) of the checkpoints behind each family's evidence or planned confirmation
FAMILY = {
    "qwen2": (3.09, 151936),   # Qwen2.5-3B(-Instruct) / Qwen2-0.5B / Qwen2.5-1.5B
    "lfm2": (1.17, 65536),     # LFM2.5-1.2B-Instruct (pruned tournament base): outside the lfm2 vocab + 2.0 B gates
    "llama": (1.24, 128256),   # Llama-3.2-1B-Instruct
    "gemma2": (2.61, 256000),  # gemma-2-2b
    "qwen3": (1.72, 151936),   # Qwen3-1.7B-Base
}
ENABLED = {"qwen2", "lfm2"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def _gate(mt, **kw):
    params_b, vocab = FAMILY[mt]
    base = dict(params_b=params_b, n_gpus=1, is_kl=False, model_type=mt, n_rows=70052, total_tokens=21_339_784,
                vocab_size=vocab)
    base.update(kw)
    return sft_v2.field_full_route(**base)


def test_family_table_defaults():
    assert set(sft_v2.DAMAGED_BASE_FAMILIES) == {"qwen2", "lfm2", "llama", "gemma2", "qwen3"}
    assert sft_v2.damaged_base_types() == frozenset(ENABLED)
    assert sft_v2.DAMAGED_BASE_ROUTE_DEFAULT == "1" and sft_v2._damaged_base_route() is True
    assert sft_v2.damage_detectors() == ("identity",)  # zero_fraction is opt-in
    assert all(v["max_params_b"] <= sft_v2.FIELD_FULL_MAX_PARAMS_B for v in sft_v2.DAMAGED_BASE_FAMILIES.values())


@pytest.mark.parametrize("mt", sorted(FAMILY))
def test_route_per_family_and_verdict(mt):
    today = _gate(mt)  # the deployed decision without any damage verdict
    for verdict in (None, "clean", "unknown"):
        assert _gate(mt, base_damage=verdict) is today  # clean / unknown bases keep today's route
    if mt in ENABLED:
        assert _gate(mt, base_damage="damaged") is True
    else:
        assert _gate(mt, base_damage="damaged") is today


def test_todays_routes_on_these_shapes_are_unchanged():
    # pinned for reference: qwen2 / gemma2 / qwen3 are adapter families; the 1.2B lfm2 fails the vocab + size gates;
    # Llama-3.2-1B is already on the field route (deployed default since 2026-09-15), damaged or not
    assert {mt: _gate(mt) for mt in FAMILY} == {"qwen2": False, "lfm2": False, "llama": True, "gemma2": False,
                                                "qwen3": False}
    # a damaged llama outside its validated vocabulary stays off while llama is disabled
    assert _gate("llama", vocab_size=32000, base_damage="damaged") is False


@pytest.mark.parametrize("mt", sorted(ENABLED))
def test_floors_and_hard_gates_still_apply_on_a_damaged_base(mt, monkeypatch):
    d = dict(base_damage="damaged")
    assert _gate(mt, **d, n_rows=14999) is False                    # FIELD_FULL_MIN_ROWS 15000
    assert _gate(mt, **d, total_tokens=7_999_999) is False          # FIELD_FULL_MIN_TOKENS 8M
    assert _gate(mt, **d, total_tokens=None, n_rows=None) is True   # unknown volume: same as the field route today
    assert _gate(mt, **d, params_b=3.51) is False                   # FIELD size cap
    assert _gate(mt, **d, params_b=7.6) is False
    assert _gate(mt, **d, n_gpus=2) is False
    assert _gate(mt, **d, is_kl=True) is False
    monkeypatch.setenv("FORGE_V2_FIELD", "0")                       # the field master switch covers this route too
    assert _gate(mt, **d) is False


def test_master_switch_and_family_override(monkeypatch):
    monkeypatch.setenv("FORGE_V2_DAMAGED_BASE_ROUTE", "0")
    assert not any(_gate(mt, base_damage="damaged") and not _gate(mt) for mt in FAMILY)
    assert sft_v2.damaged_base_family_enabled("qwen2") is False
    monkeypatch.setenv("FORGE_V2_DAMAGED_BASE_ROUTE", "1")
    monkeypatch.setenv("FORGE_V2_DAMAGED_BASE_TYPES", "gemma2, qwen3,mystery")
    assert sft_v2.damaged_base_types() == frozenset({"gemma2", "qwen3"})  # a family missing from the table is ignored
    assert _gate("gemma2", base_damage="damaged") is True and _gate("qwen3", base_damage="damaged") is True
    assert _gate("qwen2", base_damage="damaged") is False and _gate("lfm2", base_damage="damaged") is False
    monkeypatch.setenv("FORGE_V2_DAMAGED_BASE_TYPES", "none")
    assert sft_v2.damaged_base_types() == frozenset()
    assert _gate("qwen2", base_damage="damaged") is False


def test_family_switch_is_data(monkeypatch):
    monkeypatch.setitem(sft_v2.DAMAGED_BASE_FAMILIES, "llama", {"enabled": True, "max_params_b": 3.5})
    assert _gate("llama", vocab_size=32000, base_damage="damaged") is True  # TinyLlama-shaped vocab now admitted
    monkeypatch.setitem(sft_v2.DAMAGED_BASE_FAMILIES, "qwen2", {"enabled": False, "max_params_b": 3.5})
    assert _gate("qwen2", base_damage="damaged") is False
    monkeypatch.setitem(sft_v2.DAMAGED_BASE_FAMILIES, "lfm2", {"enabled": True, "max_params_b": 1.0})
    assert _gate("lfm2", base_damage="damaged") is False  # per-family size cap below the 1.17 B model


def test_identical_base_admission_is_untouched():
    # the 2026-09-17..19 identical-base admissions do not depend on the damage verdict
    q = dict(params_b=0.494, base_identity="identical", base_candidate="Qwen__Qwen2.5-0.5B-Instruct")
    assert _gate("qwen2", **q) is True and _gate("qwen2", **q, base_damage="clean") is True
    assert _gate("qwen2", base_identity="altered") is False
    assert _gate("qwen2", base_identity="altered", base_damage="damaged") is True


def test_damaged_recipe_defaults(monkeypatch):
    for mt in FAMILY:
        assert sft_v2._field_bf16_enabled(mt, damaged_base=True) is True
        assert sft_v2._field_bf16_min_b(mt, damaged_base=True) == 0.3
        assert sft_v2._field_lr_mult(mt, damaged_base=True) == 1.0
    # without the damaged flag the deployed defaults stand
    assert sft_v2._field_bf16_enabled("llama") is False and sft_v2._field_bf16_min_b("llama") == 2.0
    assert sft_v2._field_bf16_enabled("lfm2") is True and sft_v2._field_bf16_min_b("lfm2") == 2.0
    assert sft_v2._field_bf16_enabled("gemma2") is False
    # research env still wins
    monkeypatch.setenv("FORGE_V2_FIELD_BF16", "0")
    monkeypatch.setenv("FORGE_V2_FIELD_BF16_MIN_B", "1.5")
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "0.5")
    assert sft_v2._field_bf16_enabled("qwen2", damaged_base=True) is False
    assert sft_v2._field_bf16_min_b("qwen2", damaged_base=True) == 1.5
    assert sft_v2._field_lr_mult("qwen2", damaged_base=True) == 0.5


# ---- detectors ------------------------------------------------------------------------------------------------

def _report(**kw):
    rep = dict(status="altered", candidate="Qwen__Qwen2.5-3B-Instruct", n_tensors=434, altered=30, missing=0, extra=0,
               protected=74, protected_altered=0)
    rep.update(kw)
    return rep


def test_identity_signature():
    assert damage.identity_signature("", _report())["verdict"] == "damaged"
    assert damage.identity_signature("", _report(status="identical", altered=0))["verdict"] == "clean"
    for kw in (dict(protected_altered=3),                       # a different fine-tune of the same shape
               dict(missing=1), dict(extra=2), dict(altered=0),
               dict(protected=0), dict(protected=None),          # a report from before this patch
               dict(status="unknown"), dict(status="skipped"), dict(status="error")):
        assert damage.identity_signature("", _report(**kw))["verdict"] == "unknown", kw
    assert damage.identity_signature("", None)["verdict"] == "unknown"


def test_shipped_same_shape_checkpoints_are_not_mistaken_for_damage():
    # Qwen2-1.5B-Instruct and Qwen2.5-1.5B-Instruct share every shape; their norms differ, so one read against the
    # other's table is "altered" but not "damaged"
    a = json.loads((bid.TABLES_DIR / "Qwen__Qwen2-1.5B-Instruct.json").read_text())["tensors"]
    b = json.loads((bid.TABLES_DIR / "Qwen__Qwen2.5-1.5B-Instruct.json").read_text())["tensors"]
    rep = bid.compare(a, b)
    assert rep["status"] == "altered" and rep["protected"] > 0 and rep["protected_altered"] > 0
    assert damage.identity_signature("", rep)["verdict"] == "unknown"
    # a validator-style augmentation of b (weights changed, norms / embeddings untouched) is "damaged"
    damaged = {k: (dict(v, sha256="0" * 64) if not bid.is_protected(k) and k.endswith("proj.weight") else v)
               for k, v in b.items()}
    rep = bid.compare(damaged, b)
    assert rep["protected_altered"] == 0 and rep["altered"] > 0
    assert damage.identity_signature("", rep)["verdict"] == "damaged"


def _bf16_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, int], float]]) -> None:
    """tensors: name -> (shape, fraction of exact zeros); non-zero entries are bf16 1.0 (0x3f80)."""
    header, blob, off = {}, b"", 0
    for name, (shape, zf) in tensors.items():
        n = shape[0] * shape[1]
        nz = int(n * zf)
        data = b"\x00\x00" * nz + struct.pack("<H", 0x3F80) * (n - nz)
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + blob)


def test_zero_fraction_detector(tmp_path):
    def model(name, zf_per_layer):
        d = tmp_path / name
        d.mkdir()
        t = {"model.embed_tokens.weight": ((64, 32), 0.9)}  # embeddings / norms never count
        for i, zf in enumerate(zf_per_layer):
            t[f"model.layers.{i}.mlp.up_proj.weight"] = ((32, 32), zf)
            t[f"model.layers.{i}.input_layernorm.weight"] = ((32, 32), 0.9)
        _bf16_safetensors(d / "model.safetensors", t)
        return d
    assert damage.zero_fraction(str(model("clean", [0.0] * 8)))["verdict"] == "unknown"
    assert damage.zero_fraction(str(model("pruned", [0.4] * 8)))["verdict"] == "damaged"
    assert damage.zero_fraction(str(model("multi", [0.0] * 5 + [0.3] * 3)))["verdict"] == "damaged"  # 3 layers hit
    assert damage.zero_fraction(str(model("two", [0.0] * 6 + [0.45] * 2)))["verdict"] == "unknown"
    assert damage.zero_fraction(str(model("sparseish", [0.1] * 8)))["verdict"] == "unknown"
    assert damage.zero_fraction(str(tmp_path / "missing"))["verdict"] == "unknown"


def test_assess_combines_detectors_and_never_raises(tmp_path, monkeypatch):
    assert damage.assess("x", identity=_report(), detectors=("identity",))["verdict"] == "damaged"
    assert damage.assess("x", identity=_report(protected_altered=1), detectors=("identity",))["verdict"] == "unknown"
    assert damage.assess("x", identity=None, detectors=())["verdict"] == "unknown"
    assert damage.assess("x", identity=None, detectors=("nope",))["verdict"] == "unknown"
    monkeypatch.setitem(damage.DETECTORS, "boom", lambda d, i: 1 / 0)
    assert damage.assess("x", identity=None, detectors=("boom",))["verdict"] == "unknown"
    damage.register("always", lambda d, i: {"verdict": "damaged"})
    try:
        rep = damage.assess("x", identity=None, detectors=("identity", "always"))
        assert rep["verdict"] == "damaged" and rep["by"] == ["always"]
        # a byte-identical public checkpoint overrides every statistic
        rep = damage.assess("x", identity=_report(status="identical", altered=0), detectors=("identity", "always"))
        assert rep["verdict"] == "clean"
    finally:
        damage.DETECTORS.pop("always", None)


# ---- dispatcher (deployed CLI / parser / dispatcher; model, identity and training are doubles) ------------------

_contract_path = Path(__file__).resolve().parent / "test_anonymous_field_dispatch_contract.py"
_spec = importlib.util.spec_from_file_location("_anon_contract", _contract_path)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def _dispatch(monkeypatch, tmp_path, model_type, *, identity=None, params_b=None, long_rows=True, env=None,
              zero=None):
    from forge import cli
    from forge.tasks import instruct

    tmp_path.mkdir(parents=True, exist_ok=True)
    args, seen, _ = contract.prepare(monkeypatch, tmp_path, model_type, n_rows=20000, long_rows=long_rows)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if params_b is not None:
        monkeypatch.setattr(instruct, "model_param_billions", lambda m: params_b)
    calls = {"identify": 0}

    def identify(model_dir, **kw):
        calls["identify"] += 1
        return dict(identity or {"status": "unknown", "reason": "no_matching_table"})

    monkeypatch.setattr(bid, "identify", identify)
    if zero is not None:
        monkeypatch.setitem(damage.DETECTORS, "zero_fraction", lambda d, i: {"verdict": zero})

    def stop_v2(spec, deadline, rows, loaded_arg, tokenizer, **kw):
        seen["routes"].append({"handler": "sft_v2", "field": kw["field_route"], "damaged": kw["damaged_base_route"],
                               "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], model_type)})

    monkeypatch.setattr(sft_v2, "run", stop_v2)
    try:
        cli.main(args)
    except contract.StopAtLegacyPlan:
        pass
    events = dict(seen["events"])
    return seen["routes"], events, calls


def test_dispatch_qwen2_damaged_identical_unknown(monkeypatch, tmp_path):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, "qwen2", identity=_report())
    assert routes == [{"handler": "sft_v2", "field": True, "damaged": True, "strategy": "full"}]
    assert ev["base_damage"]["verdict"] == "damaged" and ev["sft_v2_field_gate"]["damaged_base_route"] is True
    assert calls["identify"] == 1  # one hash pass serves both the identity admission and the damage check
    # identical (untested 3B checkpoint) and unknown bases keep today's route: v2 LoRA
    for rep in (_report(status="identical", altered=0), None, _report(protected_altered=2)):
        routes, ev, _ = _dispatch(monkeypatch, tmp_path / str(id(rep)), "qwen2", identity=rep)
        assert routes == [{"handler": "sft_v2", "field": False, "damaged": False, "strategy": "lora"}]


def test_dispatch_qwen2_damaged_short_rows_takes_the_field_route(monkeypatch, tmp_path):
    # 20k short rows = 160k tokens: below the 8M floor the route stays legacy even on a damaged base ...
    routes, ev, _ = _dispatch(monkeypatch, tmp_path / "a", "qwen2", identity=_report(), long_rows=False)
    assert routes == [{"handler": "legacy", "field": False, "strategy": "lora"}]
    # ... and above the floor (research override) short rows take full weights
    routes, ev, _ = _dispatch(monkeypatch, tmp_path / "b", "qwen2", identity=_report(), long_rows=False,
                              env={"FORGE_V2_FIELD_MIN_TOKENS": "0"})
    assert routes == [{"handler": "sft_v2", "field": True, "damaged": True, "strategy": "full"}]


def test_dispatch_switches_off(monkeypatch, tmp_path):
    for i, env in enumerate(({"FORGE_V2_DAMAGED_BASE_ROUTE": "0"}, {"FORGE_V2_DAMAGED_BASE_TYPES": "lfm2"})):
        routes, ev, _ = _dispatch(monkeypatch, tmp_path / str(i), "qwen2", identity=_report(), env=env)
        assert routes == [{"handler": "sft_v2", "field": False, "damaged": False, "strategy": "lora"}]
        assert "base_damage" not in ev


def test_dispatch_lfm2_pruned_needs_a_detector(monkeypatch, tmp_path):
    # LFM2.5-1.2B: no shipped table, so the identity detector alone reads "unknown" and today's route stands
    routes, ev, _ = _dispatch(monkeypatch, tmp_path / "a", "lfm2", params_b=1.17)
    assert routes and routes[0]["field"] is False
    assert ev["base_damage"]["verdict"] == "unknown"
    # with the opt-in pruning detector firing, the 1.2 B model takes the field route (vocab / 2.0 B gates bypassed)
    routes, ev, _ = _dispatch(monkeypatch, tmp_path / "b", "lfm2", params_b=1.17, zero="damaged",
                              env={"FORGE_V2_DAMAGE_DETECTORS": "identity,zero_fraction"})
    assert routes == [{"handler": "sft_v2", "field": True, "damaged": True, "strategy": "full"}]
    assert ev["base_damage"]["by"] == ["zero_fraction"]


@pytest.mark.parametrize("mt", ["gemma2", "qwen3", "llama"])
def test_dispatch_disabled_families_do_not_run_detection(monkeypatch, tmp_path, mt):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, mt, identity=_report())
    assert "base_damage" not in ev and calls["identify"] == 0
    assert routes and routes[0].get("damaged", False) is False
