"""Damaged-base route reproduces the executed 2026-09-22/23 candidate recipes exactly (CPU contract, no training).

For every scored damaged-base candidate arm the source was c0dc300 plus the plan's ``route_env``
(research/20260922-dmg-*-exec/plans/*candidate*.json) and the arm's train.log recorded the resulting route / geometry
(``sft_v2_field_route``, ``sft_v2_field_bf16``, ``sft_v2_field_lr``, ``sft_v2_packing[_skipped]``, ``sft_v2_data``).
These tests assert that, with NO environment, a base read "damaged" gets the same gate decision and the same knobs as
the executed env produced, and that clean / unknown bases keep today's decision per family.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from forge.data import packing
from forge.routing import damage
from forge.tasks import sft_v2

# name -> (model_type, params_b, vocab, route_env of the candidate plan, executed knobs from its train.log)
#   eff: sft_v2_field_route eff_batch; unit: max_len (== pack_len when packed); packed: sft_v2_packing vs _skipped;
#   rows_per_block: packing stats rows / blocks; micro / accum: sft_v2_field_bf16; lr_mult: sft_v2_field_lr mult
_BASE_ENV = {"FORGE_V2_FIELD": "1", "FORGE_V2_FIELD_BF16": "1", "FORGE_V2_FIELD_LR_MULT": "1"}
EXECUTED = {
    # 20260922-dmg-qwen25-15b-exec (Qwen2.5-1.5B pruned, augmented-1f003e4fb12a653d), s17 -1.895%
    "qwen2-1.5b-pruned": ("qwen2", 1.544, 151936, dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="qwen2"),
                          dict(eff=100, unit=1472, packed=True, rows_per_block=69610 / 16468, micro=24, accum=1)),
    # 20260922-dmg-qwen25-3bi{,-s7}-exec (Qwen2.5-3B-Instruct scaled, augmented-f96977dba6704cd1), -1.247% / -1.466%
    "qwen2-3b-scaled": ("qwen2", 3.086, 151936, dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="qwen2"),
                        dict(eff=48, unit=1472, packed=True, rows_per_block=69410 / 16423, micro=12, accum=1)),
    # 20260922-dmg-qwen3-17b-exec (Qwen3-1.7B-Base re-init, augmented-ea551e5599508c26), -2.696%
    "qwen3-1.7b-reinit": ("qwen3", 1.721, 151936,
                          dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="qwen3", FORGE_V2_FIELD_BF16_MIN_B="0.3"),
                          dict(eff=100, unit=1472, packed=False, micro=33, accum=4)),
    # 20260922-dmg-llama32-1bi-exec (Llama-3.2-1B-Instruct re-init, augmented-8a3f84c4cc4a7528), -1.537%
    "llama-1b-reinit": ("llama", 1.236, 128256,
                        dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="llama", FORGE_V2_FIELD_BF16_MIN_B="0.3"),
                        dict(eff=100, unit=1472, packed=True, rows_per_block=69615 / 16463, micro=24, accum=1)),
    # 20260922-dmg-gemma2-2b-exec (gemma-2-2b re-init, augmented-029b5c5241c15cdc), -3.266%
    "gemma2-2b-reinit": ("gemma2", 2.614, 256000, dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="gemma2"),
                         dict(eff=48, unit=1472, packed=False, micro=25, accum=2)),
    # 20260922-dmg-lfm12-s17-exec / 20260922-lfm12-exec (LFM2.5-1.2B-Instruct pruned, augmented-a22a7e3362beba00)
    # 2026-09-24: the lfm2 <= 1.5 B recipe is now the replay-bench Phase H arm (FORGE_V2_FP32_MASTER=1 + FIELD_LR_MULT=2 on
    # the damaged route). In recipe terms that is bf16 off (min 1.5 B) + LR x2: _recipe["bf16"] is read only by the bf16
    # switch (sft_v2.py field bf16 block), so both leave the geometry's fp32 master + fused AdamW in place.
    "lfm2-1.2b-pruned": ("lfm2", 1.17, 65536,
                         dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="lfm2", FORGE_V2_FIELD_BF16_MIN_B="1.5",
                              FORGE_V2_FIELD_LR_MULT="2", FORGE_V2_STRATEGY="full"),
                         dict(eff=100, unit=1536, packed=False, micro=32, accum=4)),
    # 20260922-dmg-lfm26-exec (LFM2.5-2.6B scaled, augmented-973212c10d42d89f), -1.311%
    "lfm2-2.6b-scaled": ("lfm2", 2.697, 128000,
                         dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="lfm2", FORGE_V2_STRATEGY="full"),
                         dict(eff=48, unit=1472, packed=False, micro=33, accum=2)),
}
GATE_DATA = dict(n_gpus=1, is_kl=False, n_rows=70052, total_tokens=21_339_784)  # closed q2-grammar fixture
# llama is OFF for this release (executor decision 2026-09-23): production already runs Llama-3.2-1B on FIELD fp32 +
# AdamW and the llama arm's control was the old adapter, so bf16 vs production fp32 FIELD is untested
ENABLED_ARMS = sorted(n for n, v in EXECUTED.items() if v[0] != "llama")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    # cab930c-equivalence suite: these tests pin the pre-full-FT-default admission, reachable only with the research
    # switch FORGE_V2_FULLFT_DEFAULT=0; the release behaviour is pinned by tests/test_fullft_default_route_table.py
    monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")


def _knobs(mt, params_b, *, damaged):
    r = sft_v2.field_recipe(mt, params_b, damaged_base=damaged)
    return {k: r[k] for k in ("bf16", "lr_mult", "eff_batch", "strategy")}


@pytest.mark.parametrize("name", ENABLED_ARMS)
def test_damaged_route_equals_the_executed_env(monkeypatch, name):
    mt, params_b, vocab, env, ex = EXECUTED[name]
    gate = dict(GATE_DATA, params_b=params_b, model_type=mt, vocab_size=vocab)
    # 1. what the executed arm got: c0dc300 routing + its route_env, no damage verdict
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    executed_gate = sft_v2.field_full_route(**gate)
    executed_knobs = _knobs(mt, params_b, damaged=False)
    executed_strategy = sft_v2.strategy_for(params_b, mt)
    # 2. production defaults (no env) on a base the detectors read "damaged"
    for k in env:
        monkeypatch.delenv(k)
    assert sft_v2.field_full_route(**gate, base_damage="damaged") is executed_gate is True
    assert sft_v2.damaged_base_admits(mt, "damaged", params_b, vocab) is True
    assert _knobs(mt, params_b, damaged=True) == executed_knobs
    # 3. the values the arm's train.log recorded
    if mt == "lfm2" and params_b <= 1.5:
        # 2026-09-24: the lfm2 <= 1.5 B recipe is the Phase H fp32-master x2 arm (geometry covered by
        # test_lfm2_small_recipe_matches_the_phase_h_fp32_arm); the Sept 22 bf16 micro/accum pin below no longer applies
        assert executed_knobs == {"bf16": False, "lr_mult": 2.0, "eff_batch": ex["eff"], "strategy": "full"}
        return
    assert executed_knobs == {"bf16": True, "lr_mult": 1.0, "eff_batch": ex["eff"], "strategy": "full"}
    assert executed_strategy in ("full", "lora")  # both enter the v2 block; the field route forces "full"
    # 4. geometry: bf16 micro batch / accumulation at the executed unit length (packed runs: blocks per step)
    eff_units = ex["eff"]
    if ex["packed"]:
        plan = packing.PackingPlan("sdpa", "sdpa", ex["unit"], [], None, {"rows": ex["rows_per_block"] * 1000,
                                                                            "blocks": 1000})
        eff_units = plan.eff_blocks(ex["eff"])
    assert sft_v2.field_bf16_micro_accum(unit_len=ex["unit"], vocab=vocab, eff_default=eff_units) == (
        ex["micro"], ex["accum"])


def test_family_table_matches_the_executed_thresholds():
    fam = sft_v2.DAMAGED_BASE_FAMILIES
    assert {k for k, v in fam.items() if v["enabled"]} == {"qwen2", "qwen3", "gemma2", "lfm2"}
    assert fam["llama"]["enabled"] is False
    assert {k: v["bf16_min_b"] for k, v in fam.items()} == {"qwen2": 0.3, "qwen3": 0.3, "llama": 0.3, "gemma2": 2.0,
                                                            "lfm2": 1.5}
    assert all(v["lr_mult"] == 1.0 for v in fam.values())
    assert {k: (v.get("small_max_b"), v.get("small_lr_mult")) for k, v in fam.items() if v.get("small_max_b")} == {"lfm2": (1.5, 2.0)}
    assert all(v["max_params_b"] == sft_v2.FIELD_FULL_MAX_PARAMS_B == 3.5 for v in fam.values())
    assert fam["lfm2"]["vocabs"] is None  # executed with the vocabulary gate bypassed (FIELD_TYPES=lfm2)
    for name, (mt, params_b, vocab, _, _) in EXECUTED.items():
        lo, hi = fam[mt]["tested_params_b"]
        assert lo <= round(params_b, 2) <= hi, name
        assert fam[mt]["vocabs"] is None or vocab in fam[mt]["vocabs"], name


@pytest.mark.parametrize("mt", ["qwen2", "qwen3", "gemma2", "lfm2"])
def test_size_cap_is_the_field_cap(mt):
    vocab = {"qwen2": 151936, "qwen3": 151936, "llama": 128256, "gemma2": 256000, "lfm2": 128000}[mt]
    gate = dict(GATE_DATA, model_type=mt, vocab_size=vocab, base_damage="damaged")
    assert sft_v2.field_full_route(**gate, params_b=3.5) is True
    assert sft_v2.field_full_route(**gate, params_b=3.51) is False
    assert sft_v2.field_full_route(**gate, params_b=9.24) is False  # gemma-2-9b, Qwen2.5-7B, Llama-3.1-8B ...
    assert sft_v2.damaged_base_admits(mt, "damaged", 3.51, vocab) is False


# ---- packing eligibility is unchanged by the route and matches the executed arms --------------------------------

def _tiny(mt, vocab):
    import transformers as tf

    common = dict(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8)
    cfg = {"qwen2": lambda: tf.Qwen2Config(**common), "llama": lambda: tf.LlamaConfig(**common),
           "qwen3": lambda: tf.Qwen3Config(**common), "gemma2": lambda: tf.Gemma2Config(**common)}[mt]()
    cls = {"qwen2": tf.Qwen2ForCausalLM, "llama": tf.LlamaForCausalLM, "qwen3": tf.Qwen3ForCausalLM,
           "gemma2": tf.Gemma2ForCausalLM}[mt]
    cfg._attn_implementation = "sdpa"
    model = cls(cfg)
    model.config.use_cache = False
    return model


@pytest.mark.parametrize("mt,vocab,packed", [("qwen2", 151936, True), ("llama", 128256, True),
                                             ("qwen3", 151936, False), ("gemma2", 256000, False)])
def test_packing_eligibility_matches_the_executed_arms(mt, vocab, packed):
    pytest.importorskip("transformers")
    import transformers

    if transformers.__version__ != packing.QUALIFIED_TRANSFORMERS:
        pytest.skip("packing is qualified on the image's transformers only")
    reason = packing._eligibility(_tiny(mt, vocab), field_route=True, field_hygiene=False)
    assert (reason is None) is packed, reason
    if not packed:
        assert reason == "not_native_llama"  # the reason the gemma2 / qwen3 / lfm2 arms logged


# ---- dispatcher: damaged / clean / unknown per family (deployed CLI; model, detector and training are doubles) -----

_contract_path = Path(__file__).resolve().parent / "test_anonymous_field_dispatch_contract.py"
_spec = importlib.util.spec_from_file_location("_anon_contract_recipe", _contract_path)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def _dispatch(monkeypatch, tmp_path, mt, params_b, vocab, verdict, env=None):
    from forge import cli
    from forge.routing import identity_report as idr
    from forge.tasks import instruct

    tmp_path.mkdir(parents=True, exist_ok=True)
    args, seen, _ = contract.prepare(monkeypatch, tmp_path, mt, n_rows=20000, long_rows=True)  # clears FORGE_* env
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    loaded = instruct.load_base("test-only")
    seen["model_paths"].clear()
    loaded.model.config.vocab_size = vocab
    monkeypatch.setattr(instruct, "model_param_billions", lambda m: params_b)
    monkeypatch.setattr(idr, "identify", lambda d, **kw: {"status": "unknown", "reason": "no_matching_table"})
    calls = []

    def assess(model_dir, *, identity, detectors, model_name=None):
        calls.append(tuple(detectors))
        return {"verdict": verdict, "by": ["fingerprint"] if verdict != "unknown" else [], "detectors": {}, "seconds": 0}

    monkeypatch.setattr(damage, "assess", assess)

    def stop_v2(spec, deadline, rows, loaded_arg, tokenizer, **kw):
        d = bool(kw["damaged_base_route"] or kw["clean_small_qwen_route"])
        seen["routes"].append({"handler": "sft_v2", "field": kw["field_route"], "damaged": kw["damaged_base_route"],
                               "clean_small_qwen": kw["clean_small_qwen_route"],
                               "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], mt),
                               "recipe": _knobs(mt, kw["params_b"], damaged=d) if kw["field_route"] else None})

    monkeypatch.setattr(sft_v2, "run", stop_v2)
    try:
        cli.main(args)
    except contract.StopAtLegacyPlan:
        pass
    return seen["routes"], dict(seen["events"]), calls


# today's (c0dc300) decision on a clean / unknown base of each executed shape with 20k long rows (12M tokens):
# llama-3.2-1B and LFM2.5-2.6B already take the field route; the rest train the v2 LoRA
TODAY = {"qwen2-1.5b-pruned": (False, "lora"), "qwen2-3b-scaled": (False, "lora"), "qwen3-1.7b-reinit": (False, "lora"),
         "llama-1b-reinit": (True, "full"), "gemma2-2b-reinit": (False, "lora"), "lfm2-1.2b-pruned": (False, "lora"),
         "lfm2-2.6b-scaled": (True, "full")}


@pytest.mark.parametrize("name", ENABLED_ARMS)
def test_dispatch_damaged_clean_unknown(monkeypatch, tmp_path, name):
    mt, params_b, vocab, _, ex = EXECUTED[name]
    routes, ev, calls = _dispatch(monkeypatch, tmp_path / "damaged", mt, params_b, vocab, "damaged")
    assert calls == [("identity", "fingerprint")]
    assert routes == [{"handler": "sft_v2", "field": True, "damaged": True, "clean_small_qwen": False,
                       "strategy": "full",
                       "recipe": _knobs(mt, params_b, damaged=True)}]
    small_lfm = mt == "lfm2" and params_b <= 1.5
    assert routes[0]["recipe"] == {"bf16": not small_lfm, "lr_mult": 2.0 if small_lfm else 1.0, "eff_batch": ex["eff"],
                                   "strategy": "full"}
    assert ev["sft_v2_field_gate"]["damaged_base_route"] is True and ev["base_damage"]["verdict"] == "damaged"
    for verdict in ("clean", "unknown"):
        routes, ev, _ = _dispatch(monkeypatch, tmp_path / verdict, mt, params_b, vocab, verdict)
        field, strategy = TODAY[name]
        assert len(routes) == 1 and routes[0]["handler"] == "sft_v2"
        assert (routes[0]["field"], routes[0]["strategy"], routes[0]["damaged"]) == (field, strategy, False)
        if field:  # today's field recipe on a clean base (llama-1B stays fp32 master; LFM-2.6B bf16)
            assert routes[0]["recipe"] == _knobs(mt, params_b, damaged=False)


@pytest.mark.parametrize("vocab", [128256, 32000])
def test_damaged_llama_keeps_the_exact_production_route(monkeypatch, tmp_path, vocab):
    """Family OFF: a damaged Llama-3.2-1B(-shaped) base takes exactly the c0dc300 route - FIELD full, fp32 master +
    AdamW (no bf16 / paged 8-bit), LR x1.0, champion bucket 100 - and damage detection does not even run."""
    mt, params_b, _, _, ex = EXECUTED["llama-1b-reinit"]
    assert sft_v2.damaged_base_admits(mt, "damaged", params_b, vocab) is False
    gate = dict(GATE_DATA, params_b=params_b, model_type=mt, vocab_size=vocab)
    assert sft_v2.field_full_route(**gate, base_damage="damaged") is sft_v2.field_full_route(**gate) is (vocab == 128256)
    # c0dc300's recipe helpers, called exactly as c0dc300 called them (no damaged flag)
    c0dc300 = {"bf16": sft_v2._field_bf16_enabled(mt) and params_b >= sft_v2._field_bf16_min_b(mt),
               "lr_mult": sft_v2._field_lr_mult(mt), "eff_batch": sft_v2.field_eff_batch(params_b), "strategy": "full"}
    assert c0dc300 == {"bf16": False, "lr_mult": 1.0, "eff_batch": 100, "strategy": "full"}
    for verdict in ("damaged", "clean", "unknown"):
        routes, ev, calls = _dispatch(monkeypatch, tmp_path / f"{vocab}-{verdict}", mt, params_b, vocab, verdict)
        assert calls == [] and "base_damage" not in ev
        if vocab == 128256:
            assert routes == [{"handler": "sft_v2", "field": True, "damaged": False, "clean_small_qwen": False,
                               "strategy": "full", "recipe": c0dc300}]
        else:  # TinyLlama-shaped vocabulary: not the validated Llama-3.2, today's adapter route
            assert (routes[0]["field"], routes[0]["damaged"]) == (False, False)


def test_dispatch_family_switch_off_restores_today(monkeypatch, tmp_path):
    mt, params_b, vocab, _, _ = EXECUTED["gemma2-2b-reinit"]
    monkeypatch.setitem(sft_v2.DAMAGED_BASE_FAMILIES, "gemma2",
                        dict(sft_v2.DAMAGED_BASE_FAMILIES["gemma2"], enabled=False))
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, mt, params_b, vocab, "damaged")
    assert calls == [] and "base_damage" not in ev
    assert (routes[0]["field"], routes[0]["strategy"]) == (False, "lora")


# ---- clean small-Qwen: default ON (2026-09-24), with its own data floors ---------------------------------------------

TRDPO = dict(n_gpus=1, is_kl=False, n_rows=9282, total_tokens=2_812_952)      # replay r-q2-05b-trdpo (below generic floors)
DOLLY_TR = dict(n_gpus=1, is_kl=False, n_rows=13976, total_tokens=3_682_451)  # Phase G analogue a-q2-05b-dollytr


def test_clean_small_qwen_is_on_by_default(monkeypatch, tmp_path):
    assert sft_v2.CLEAN_SMALL_QWEN_FIELD_ROUTE_DEFAULT == "1"
    assert (sft_v2.CLEAN_SMALL_QWEN_MIN_ROWS, sft_v2.CLEAN_SMALL_QWEN_MIN_TOKENS) == (8000, 2_500_000)
    q = dict(params_b=0.494, model_type="qwen2", vocab_size=151936)
    assert sft_v2.clean_small_qwen_admits("qwen2", "clean", 0.494, 151936) is True
    for data in (GATE_DATA, TRDPO, DOLLY_TR):
        assert sft_v2.field_full_route(**data, **q, base_damage="clean") is True
    # its own floors, not the generic 15k rows / 8M tokens
    assert sft_v2.field_full_route(**dict(TRDPO, n_rows=7999), **q, base_damage="clean") is False
    assert sft_v2.field_full_route(**dict(TRDPO, total_tokens=2_499_999), **q, base_damage="clean") is False
    # scope: clean verdict only, qwen2 only, <= 1.0 B only
    for verdict in ("unknown", None):
        assert sft_v2.field_full_route(**TRDPO, **q, base_damage=verdict) is False
    assert sft_v2.field_full_route(**TRDPO, **dict(q, params_b=1.5), base_damage="clean") is False
    assert sft_v2.field_full_route(**TRDPO, **dict(q, model_type="qwen3"), base_damage="clean") is False
    # the damaged-base qwen2 route keeps the generic floors (untested below them)
    assert sft_v2.damaged_base_admits("qwen2", "damaged", 0.494, 151936) is True
    assert sft_v2.field_full_route(**TRDPO, **q, base_damage="damaged") is False
    assert sft_v2.field_full_route(**GATE_DATA, **q, base_damage="damaged") is True
    # research overrides still work
    monkeypatch.setenv("FORGE_V2_CLEAN_SMALL_QWEN", "0")
    assert sft_v2.field_full_route(**TRDPO, **q, base_damage="clean") is False
    monkeypatch.delenv("FORGE_V2_CLEAN_SMALL_QWEN")
    monkeypatch.setenv("FORGE_V2_FIELD_MIN_ROWS", "10000")
    assert sft_v2.field_full_route(**TRDPO, **q, base_damage="clean") is False
    monkeypatch.delenv("FORGE_V2_FIELD_MIN_ROWS")
    routes, _, _ = _dispatch(monkeypatch, tmp_path, "qwen2", 0.494, 151936, "clean")
    assert (routes[0]["field"], routes[0]["clean_small_qwen"], routes[0]["damaged"]) == (True, True, False)
    assert routes[0]["recipe"] == {"bf16": True, "lr_mult": 1.0, "eff_batch": 140, "strategy": "full"}
    routes, _, _ = _dispatch(monkeypatch, tmp_path / "u", "qwen2", 0.494, 151936, "unknown")
    assert (routes[0]["field"], routes[0]["clean_small_qwen"]) == (False, False)


def test_eval_slack_on_only_for_the_clean_small_qwen_route(monkeypatch):
    # v10e: the slack is on by default on every route; the clean small-Qwen route default stays on
    monkeypatch.delenv("FORGE_V2_EVAL_SLACK", raising=False)
    assert sft_v2.EVAL_SLACK_DEFAULT == "1"
    assert sft_v2._eval_slack() is True and sft_v2._eval_slack(True) is True
    monkeypatch.setattr(sft_v2, "EVAL_SLACK_DEFAULT", "0")      # the fa3e6ac default: route-only
    assert sft_v2._eval_slack() is False and sft_v2._eval_slack(True) is True
    monkeypatch.setattr(sft_v2, "EVAL_SLACK_DEFAULT", "1")
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "0")
    assert sft_v2._eval_slack(True) is False
    monkeypatch.setenv("FORGE_V2_EVAL_SLACK", "1")
    assert sft_v2._eval_slack(False) is True
    import inspect
    src = inspect.getsource(sft_v2.run)
    assert "_csq_field = bool(field and clean_small_qwen_route)" in src
    assert "_slack_on = _eval_slack(_csq_field)" in src and "self.window_s if _slack_on else None" in src
    # the two-seed (LoRA-only) refresh keeps the switch-only gate
    assert src.count("if _eval_slack():") == 1


def test_clean_small_qwen_switch_reproduces_the_clean_05b_arm(monkeypatch, tmp_path):
    # research/20260922-clean-qwen2-05b-s17-exec candidate: FIELD=1, FIELD_BF16=1, FIELD_BF16_MIN_B=0.3,
    # FIELD_LR_MULT=1, FIELD_TYPES=qwen2 on clean unsloth/Qwen2-0.5B (0.494 B)
    env = dict(_BASE_ENV, FORGE_V2_FIELD_TYPES="qwen2", FORGE_V2_FIELD_BF16_MIN_B="0.3")
    gate = dict(GATE_DATA, params_b=0.494, model_type="qwen2", vocab_size=151936)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    executed = (sft_v2.field_full_route(**gate), _knobs("qwen2", 0.494, damaged=False))
    for k in env:
        monkeypatch.delenv(k)
    monkeypatch.setenv("FORGE_V2_CLEAN_SMALL_QWEN", "1")
    assert (sft_v2.field_full_route(**gate, base_damage="clean"), _knobs("qwen2", 0.494, damaged=True)) == executed
    assert executed == (True, {"bf16": True, "lr_mult": 1.0, "eff_batch": 140, "strategy": "full"})
    assert sft_v2.field_full_route(**dict(gate, params_b=1.5), base_damage="clean") is False  # above its 1.0 B cap
    assert sft_v2.field_full_route(**dict(gate, model_type="qwen3"), base_damage="clean") is False
    routes, ev, _ = _dispatch(monkeypatch, tmp_path, "qwen2", 0.494, 151936, "clean",
                              env={"FORGE_V2_CLEAN_SMALL_QWEN": "1"})
    assert routes[0]["field"] is True and routes[0]["clean_small_qwen"] is True and routes[0]["damaged"] is False
    assert routes[0]["recipe"] == {"bf16": True, "lr_mult": 1.0, "eff_batch": 140, "strategy": "full"}


# ---- shipped table: plain readable JSON, no compressed blob in the submitted tree ---------------------------------

def test_shipped_table_is_plain_json():
    root = Path(damage.__file__).resolve().parents[1]
    assert damage.FINGERPRINT_TABLE.suffix == ".json" and damage.FINGERPRINT_TABLE.exists()
    assert not list(root.rglob("*.gz"))
    raw = damage.FINGERPRINT_TABLE.read_bytes()
    assert max(b for b in raw) < 128 and max(len(line) for line in raw.split(b"\n")) < 4096
    damage._TABLE_CACHE.clear()
    t = damage.load_table()
    assert t is not None and len(t["models"]) == 109  # 73 (2026-09-23) + 36 small public bases (2026-09-27 addendum)


def test_lfm2_small_recipe_matches_the_phase_h_fp32_arm(monkeypatch):
    """Damaged LFM2 <= 1.5 B: bf16 off + LR x2 by default; 2.6 B unchanged (bf16, x1). The Phase H arm ran the old recipe
    with FORGE_V2_FP32_MASTER=1 + FIELD_LR_MULT=2: same geometry because only the bf16 switch reads _recipe["bf16"]."""
    import inspect
    small = sft_v2.field_recipe("lfm2", 1.17, damaged_base=True)
    big = sft_v2.field_recipe("lfm2", 2.70, damaged_base=True)
    assert (small["bf16"], small["lr_mult"]) == (False, 2.0) and (big["bf16"], big["lr_mult"]) == (True, 1.0)
    # precision AND LR agree below / at / above the inclusive small_max_b boundary (Codex review of cae3b6c: exactly
    # 1.5 B used to get bf16 + LR x2, because bf16 used >= bf16_min_b while the LR band used <= small_max_b)
    for params_b, want in ((1.17, (False, 2.0)), (1.4999, (False, 2.0)), (1.5, (False, 2.0)),
                           (1.5001, (True, 1.0)), (1.51, (True, 1.0)), (2.70, (True, 1.0))):
        r = sft_v2.field_recipe("lfm2", params_b, damaged_base=True)
        assert (r["bf16"], r["lr_mult"]) == want, (params_b, r)
    assert sft_v2.field_recipe("lfm2", 1.5, damaged_base=False)["lr_mult"] == 1.0          # small band is damaged-only
    # explicit research controls keep precedence over the small-band precision default (Codex REVIEW-1041d12)
    for env, want_bf16 in (({"FORGE_V2_FIELD_BF16": "1", "FORGE_V2_FIELD_BF16_MIN_B": "0.3"}, True),
                           ({"FORGE_V2_FIELD_BF16_MIN_B": "0.3"}, True),
                           ({"FORGE_V2_FIELD_BF16": "0"}, False),
                           ({"FORGE_V2_FIELD_BF16": "1"}, False),           # 1.17 < default min 1.5: parent behaviour
                           ({}, False)):
        with monkeypatch.context() as m:
            for k in ("FORGE_V2_FIELD_BF16", "FORGE_V2_FIELD_BF16_MIN_B"):
                m.delenv(k, raising=False)
            for k, v in env.items():
                m.setenv(k, v)
            r = sft_v2.field_recipe("lfm2", 1.17, damaged_base=True)
            assert (r["bf16"], r["lr_mult"]) == (want_bf16, 2.0), (env, r)
    with monkeypatch.context() as m:
        m.setenv("FORGE_V2_FIELD_LR_MULT", "3")
        assert sft_v2.field_recipe("lfm2", 1.5, damaged_base=True)["lr_mult"] == 3.0       # LR override still wins
    assert sft_v2.field_recipe("qwen2", 1.17, damaged_base=True)["lr_mult"] == 1.0        # other families unchanged
    assert sft_v2.choose_geometry(params_b=1.17, max_len=1536, vocab=65536, per_gpu_gb=80, bnb_ok=True).fp32_master is True
    src = inspect.getsource(sft_v2.run)
    assert src.count('_recipe["bf16"]') == 1 and 'if field and geo.fp32_master and _recipe["bf16"]' in src
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "1")                                      # research override still wins
    assert sft_v2.field_recipe("lfm2", 1.17, damaged_base=True)["lr_mult"] == 1.0

