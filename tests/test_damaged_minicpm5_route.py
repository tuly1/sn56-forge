"""Damaged MiniCPM5-1B FIELD route (release candidate 2026-09-27, branches release-r1-candidate-20260927
and release-r1-noqwen-20260927).

A llama base with vocabulary 130560 (openbmb/MiniCPM5-1B) in the (0.9, 1.3] B band that the damage detectors read
"damaged" takes the damaged-base FIELD full BF16 route at the formula LR (clamped to 1e-4): the recipe replay-bench
Phase Y arm F1 executed as 52d319d + FORGE_V2_FIELD_TYPES=llama + FORGE_V2_FIELD_BF16=1 + FORGE_V2_FIELD_BF16_MIN_B=0.3
(research/20260923-replay-bench/phase-y/CLOSEOUT-20260927.md). Clean / unknown MiniCPM5 and every other llama base keep
their 52d319d routes, and detection runs for the MiniCPM5 shape only. FORGE_V2_DAMAGED_MINICPM5=0 restores 52d319d."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from forge.data import packing
from forge.routing import damage
from forge.routing import identity_report as idr
from forge.tasks import sft_v2
from forge.tuning import lr_probe

HERE = Path(__file__).resolve().parent
SWITCH = "FORGE_V2_DAMAGED_MINICPM5"
F1_ENV = {"FORGE_V2_FIELD_TYPES": "llama", "FORGE_V2_FIELD_BF16": "1", "FORGE_V2_FIELD_BF16_MIN_B": "0.3"}
MINICPM_VOCAB = 130560
MINICPM_B = 1.080633  # openbmb/MiniCPM5-1B, 1,080,632,832 parameters
OPENORCA = dict(n_rows=26000, total_tokens=13_000_000)  # Phase Y fixture scale (above the llama floors 10k / 5M)
HW = dict(n_gpus=1, is_kl=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    # cab930c-equivalence suite: these tests pin the pre-full-FT-default admission, reachable only with the research
    # switch FORGE_V2_FULLFT_DEFAULT=0; the release behaviour is pinned by tests/test_fullft_default_route_table.py
    monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")


def _route(params_b=MINICPM_B, *, mt="llama", verdict="damaged", vocab=MINICPM_VOCAB, data=OPENORCA):
    return sft_v2.field_full_route(params_b=params_b, model_type=mt, vocab_size=vocab, base_damage=verdict, **HW, **data)


def _decisions(params_b, mt, vocab, verdict, data=OPENORCA):
    return {"detect": sft_v2.damage_detection_enabled(mt, vocab_size=vocab, params_b=params_b),
            "field": _route(params_b, mt=mt, verdict=verdict, vocab=vocab, data=data),
            "damaged": sft_v2.damaged_base_admits(mt, verdict, params_b, vocab)}


# ---- constants and admission -------------------------------------------------------------------------------------

def test_constants_and_defaults():
    assert sft_v2.DAMAGED_MINICPM5_ROUTE_DEFAULT == "1"
    assert (sft_v2.DAMAGED_MINICPM5_MODEL_TYPE, sft_v2.DAMAGED_MINICPM5_VOCAB) == ("llama", 130560)
    assert sft_v2.DAMAGED_MINICPM5_MIN_PARAMS_B < MINICPM_B <= sft_v2.DAMAGED_MINICPM5_MAX_PARAMS_B
    # the llama family row stays disabled and its tested vocabulary is Llama-3.x only
    llama = sft_v2.DAMAGED_BASE_FAMILIES["llama"]
    assert llama["enabled"] is False and llama["vocabs"] == frozenset({128256})
    assert "llama" not in sft_v2.damaged_base_types()
    assert sft_v2.FIELD_FULL_VOCABS["llama"] == frozenset({128256})  # clean MiniCPM5 still refused by the vocab gate


def test_damaged_minicpm5_is_admitted():
    assert _decisions(MINICPM_B, "llama", MINICPM_VOCAB, "damaged") == {"detect": True, "field": True, "damaged": True}
    assert sft_v2.damaged_minicpm5_admits("llama", "damaged", MINICPM_B, MINICPM_VOCAB) is True
    assert sft_v2.damaged_minicpm5_admits("LLAMA", "damaged", 1.3, MINICPM_VOCAB) is True  # inclusive upper bound


@pytest.mark.parametrize("verdict", [None, "clean", "unknown"])
def test_clean_or_unknown_minicpm5_keeps_the_adapter_route(verdict):
    # detection runs for the shape, but only a "damaged" verdict admits; the vocabulary gate refuses 130560 as before
    assert _decisions(MINICPM_B, "llama", MINICPM_VOCAB, verdict) == {"detect": True, "field": False, "damaged": False}


@pytest.mark.parametrize("params_b", [0.5, 0.9, 1.31, 2.0, 3.2])
def test_outside_the_size_band_nothing_changes(params_b):
    assert _decisions(params_b, "llama", MINICPM_VOCAB, "damaged") == {"detect": False, "field": False, "damaged": False}


@pytest.mark.parametrize("vocab,params_b,today", [
    (128256, 1.2358, True),    # Llama-3.2-1B: already FIELD fp32 in 52d319d, damaged or not; no detection
    (128256, 3.2127, True),    # Llama-3.2-3B
    (32000, 1.1001, False),    # TinyLlama
    (49152, 1.7115, False),    # SmolLM2-1.7B
    (None, MINICPM_B, True),   # unknown vocabulary never matches the shape (52d319d skips the vocab gate)
])
@pytest.mark.parametrize("verdict", [None, "clean", "unknown", "damaged"])
def test_other_llama_bases_are_unchanged(vocab, params_b, today, verdict):
    assert _decisions(params_b, "llama", vocab, verdict) == {"detect": False, "field": today, "damaged": False}


@pytest.mark.parametrize("mt", ["qwen2", "qwen3", "gemma2", "lfm2", "mistral", "minicpm"])
def test_other_model_types_with_this_vocabulary_are_not_admitted(mt):
    assert sft_v2.damaged_minicpm5_shape(mt, MINICPM_B, MINICPM_VOCAB) is False
    assert sft_v2.damaged_minicpm5_admits(mt, "damaged", MINICPM_B, MINICPM_VOCAB) is False


def test_floors_and_hard_gates_still_apply():
    assert _route(data=dict(n_rows=9999, total_tokens=13_000_000)) is False       # llama row floor 10k
    assert _route(data=dict(n_rows=26000, total_tokens=4_999_999)) is False       # llama token floor 5M
    assert _route(data=dict(n_rows=10000, total_tokens=5_000_000)) is True
    kw = dict(params_b=MINICPM_B, model_type="llama", vocab_size=MINICPM_VOCAB, base_damage="damaged", **OPENORCA)
    assert sft_v2.field_full_route(n_gpus=2, is_kl=False, **kw) is False
    assert sft_v2.field_full_route(n_gpus=1, is_kl=True, **kw) is False


@pytest.mark.parametrize("env", [{SWITCH: "0"}, {"FORGE_V2_DAMAGED_BASE_ROUTE": "0"}, {"FORGE_V2_FIELD": "0"}])
def test_off_switches_restore_52d319d(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # 52d319d: no llama detection, no damaged admission, vocabulary 130560 refused by the FIELD gate
    for verdict in (None, "clean", "unknown", "damaged"):
        assert _decisions(MINICPM_B, "llama", MINICPM_VOCAB, verdict) == {"detect": False, "field": False,
                                                                           "damaged": False}
    assert sft_v2.damage_detection_enabled("llama") is False


def test_family_override_env_does_not_reach_minicpm5(monkeypatch):
    # FORGE_V2_DAMAGED_BASE_TYPES restricts the family table only; the MiniCPM5 admission has its own switch
    monkeypatch.setenv("FORGE_V2_DAMAGED_BASE_TYPES", "none")
    assert _route() is True
    monkeypatch.setenv(SWITCH, "0")
    assert _route() is False


def test_detection_signature_is_backward_compatible():
    # callers that pass only the model type (older harnesses) see the 52d319d answer for llama
    assert sft_v2.damage_detection_enabled("llama") is False
    for mt in ("qwen2", "qwen3", "gemma2", "lfm2"):
        assert sft_v2.damage_detection_enabled(mt) is True


# ---- recipe = Phase Y F1 -----------------------------------------------------------------------------------------

def test_recipe_equals_the_f1_env(monkeypatch):
    route = sft_v2.field_recipe("llama", MINICPM_B, damaged_base=True)
    assert route == {"bf16": True, "bf16_min_b": 0.3, "lr_mult": 1.0, "eff_batch": 100, "strategy": "full"}
    for k, v in F1_ENV.items():
        monkeypatch.setenv(k, v)
    assert sft_v2.field_recipe("llama", MINICPM_B, damaged_base=False) == route


# ---- dispatcher (deployed CLI / parser / dispatcher; model, identity, detectors and training are doubles) ----------

_contract_path = HERE / "test_anonymous_field_dispatch_contract.py"
_spec = importlib.util.spec_from_file_location("_anon_contract_minicpm5", _contract_path)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def _dispatch(monkeypatch, tmp_path, *, vocab, params_b, verdict, env=None):
    from forge import cli
    from forge.tasks import instruct

    tmp_path.mkdir(parents=True, exist_ok=True)
    args, seen, _ = contract.prepare(monkeypatch, tmp_path, "llama", n_rows=20000, long_rows=True)
    loaded = instruct.load_base("probe")
    seen["model_paths"].clear()
    loaded.model.config.vocab_size = vocab
    monkeypatch.setattr(instruct, "model_param_billions", lambda m: params_b)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    calls = {"identify": 0}

    def identify(model_dir, **kw):
        calls["identify"] += 1
        return {"status": "unknown", "reason": "no_matching_table"}

    monkeypatch.setattr(idr, "identify", identify)
    monkeypatch.setitem(damage.DETECTORS, "fingerprint", lambda d, i, **kw: {"verdict": verdict, "reason": "double", "source": "openbmb/MiniCPM5-1B"})

    def stop_v2(spec, deadline, rows, loaded_arg, tokenizer, **kw):
        seen["routes"].append({"handler": "sft_v2", "field": kw["field_route"], "damaged": kw["damaged_base_route"],
                               "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], "llama")})

    monkeypatch.setattr(sft_v2, "run", stop_v2)
    try:
        cli.main(args)
    except contract.StopAtLegacyPlan:
        pass
    return seen["routes"], dict(seen["events"]), calls


def test_dispatch_damaged_minicpm5(monkeypatch, tmp_path):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, vocab=MINICPM_VOCAB, params_b=MINICPM_B, verdict="damaged")
    assert routes == [{"handler": "sft_v2", "field": True, "damaged": True, "strategy": "full"}]
    assert calls["identify"] == 1 and ev["base_damage"]["verdict"] == "damaged"
    assert ev["sft_v2_field_gate"]["damaged_base_route"] is True


def test_dispatch_clean_minicpm5_keeps_v2_lora(monkeypatch, tmp_path):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, vocab=MINICPM_VOCAB, params_b=MINICPM_B, verdict="clean")
    assert routes == [{"handler": "sft_v2", "field": False, "damaged": False, "strategy": "lora"}]
    assert calls["identify"] == 1 and ev["base_damage"]["verdict"] == "clean"


def test_dispatch_off_switch_is_52d319d(monkeypatch, tmp_path):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, vocab=MINICPM_VOCAB, params_b=MINICPM_B, verdict="damaged",
                                  env={SWITCH: "0"})
    assert routes == [{"handler": "sft_v2", "field": False, "damaged": False, "strategy": "lora"}]
    assert calls["identify"] == 0 and "base_damage" not in ev


@pytest.mark.parametrize("vocab,params_b", [(128256, 1.2358), (32000, 1.1001), (MINICPM_VOCAB, 2.0)])
def test_dispatch_other_llama_bases_run_no_detection(monkeypatch, tmp_path, vocab, params_b):
    routes, ev, calls = _dispatch(monkeypatch, tmp_path, vocab=vocab, params_b=params_b, verdict="damaged")
    assert calls["identify"] == 0 and "base_damage" not in ev
    assert routes and routes[0].get("damaged", False) is False
    today = vocab == 128256  # Llama-3.x keeps its deployed FIELD fp32 route
    assert routes[0]["field"] is today


# ---- sft_v2.run on CPU: candidate route == F1 env on the plain FIELD route -----------------------------------------

def _tiny_minicpm5_like():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=MINICPM_VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, pad_token_id=1,
        tie_word_embeddings=False, use_cache=False, attn_implementation="sdpa")).eval()


def _rows(n=34):
    out = []
    for i in range(n):
        ids = [0] + [5 + i] * (3 + i % 5) + [17, 1]
        out.append(dict(input_ids=ids, labels=([-100, -100] + ids[2:]) if i % 2 else ids.copy()))
    return out


def _run(monkeypatch, tmp_path, *, damaged: bool, env=None, rms=1.0):
    """sft_v2.run on CPU with a tiny llama (vocab 130560) at params_b 1.0806: the real route, packing, recipe, geometry
    and LR code; training, timing, dev scoring and export are doubles. Returns (events, trainers)."""
    import torch
    from forge.data.schema import InstructColumns
    from forge.tasks import fallback
    from forge.tuning import selection

    import transformers
    if transformers.__version__ != packing.QUALIFIED_TRANSFORMERS:
        pytest.skip("route geometry is qualified on the image's transformers only")
    torch.set_num_threads(2)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("FORGE_DEV_PASS", "0")
    events, trainers = [], []
    source = _rows()
    monkeypatch.setattr(sft_v2, "_liger_ok", lambda m: False)
    monkeypatch.setattr(sft_v2, "_bnb_available", lambda: False)
    monkeypatch.setattr(sft_v2, "workdir", lambda spec: str(tmp_path / "work"))
    monkeypatch.setattr(sft_v2, "_layer_weight_rms", lambda m: rms)
    monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda *a: [])
    monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct", lambda *a: copy.deepcopy(source))
    monkeypatch.setattr(sft_v2, "adaptive_max_len", lambda *a, **k: 32)
    monkeypatch.setattr(sft_v2, "dev_size_for", lambda *a, **k: 2)
    monkeypatch.setattr(sft_v2, "stratified_split", lambda r, *a, **k: (r[:-2], r[-2:]))
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **d: events.append((n, d)))
    for name in ("set_meta", "eval_point", "write_into"):
        monkeypatch.setattr(sft_v2.telemetry, name, lambda *a, **k: None)
    monkeypatch.setattr(sft_v2.telemetry, "make_trainer_callback", lambda *a: object())
    monkeypatch.setattr(sft_v2, "write_artifact_truth", lambda *a, **k: None)
    monkeypatch.setattr(sft_v2, "save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(selection, "per_example_dev_loss", lambda model, rs, collator, **kw: (0.5, []))
    monkeypatch.setattr(selection, "available_ram", lambda: 10 ** 12)
    monkeypatch.setattr(sft_v2.lr_probe, "measure_step_time", lambda model, batches, **kw: (1.0, [1.0]))

    class Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setattr(sft_v2, "compatible_dataclass_kwargs", lambda cls, kw, **_: kw)
    monkeypatch.setattr(sys.modules["transformers"], "TrainingArguments", Args)

    class RecordingTrainer:
        def __init__(self, model, args, train_dataset, data_collator, callbacks=()):
            self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
            self.callbacks = callbacks
            self.state = NS(global_step=0, max_steps=100)
            self.optimizer = object()
            self.lr_scheduler = object()
            trainers.append(self)

        def get_train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, collate_fn=self.data_collator)

        def train(self):
            self.state.global_step = 1
            self.callbacks[0].on_step_end(self.args, self.state, NS(should_training_stop=False), model=self.model)

    monkeypatch.setattr(sys.modules["transformers"], "Trainer", RecordingTrainer)
    spec = NS(output_dir=str(tmp_path / "out"), task_id="synthetic", instruct=InstructColumns("instruction", "output"))
    deadline = NS(remaining_hard=lambda: 3600.0, export_reserve_s=180.0)
    sft_v2.run(spec, deadline, source, NS(model=_tiny_minicpm5_like()), NS(pad_token_id=1), baseline_summary=None,
               params_b=MINICPM_B, per_gpu_gb=80.0, strategy_override="full", field_route=True,
               damaged_base_route=damaged)
    return events, trainers


def _ev(events, name):
    return [d for n, d in events if n == name]


def _trace(events, trainers):
    plan0 = _ev(events, "sft_v2_plan")[0]
    return {"field_lr": _ev(events, "sft_v2_field_lr"), "override": _ev(events, "sft_v2_lr_override"),
            "sweep_gate_lr": _ev(events, "sft_v2_sweep_gate")[0]["lr_override"],
            "train_start_lr": _ev(events, "sft_v2_train_start")[0]["lr"], "plan_lr": plan0["lr"],
            "plan_warmup": plan0["warmup"], "plan_steps": plan0["total_steps"],
            "trainer_lr": trainers[-1].args.learning_rate, "trainer_warmup": trainers[-1].args.warmup_steps,
            "packing": _ev(events, "sft_v2_packing"), "bf16": _ev(events, "sft_v2_field_bf16"),
            "route": _ev(events, "sft_v2_field_route"),
            "recipe": [{k: v for k, v in d.items() if k != "damaged_base"} for d in _ev(events, "sft_v2_field_recipe")]}


@pytest.mark.parametrize("rms", [1.0, 0.025])  # clamped at 1e-4 (as Phase Y's raw 1.03e-4); inside the clamp
def test_run_candidate_equals_phase_y_f1(monkeypatch, tmp_path, rms):
    ev, tr = _run(monkeypatch, tmp_path / "route", damaged=True, rms=rms)
    t = _trace(ev, tr)
    raw = lr_probe.analytic_lr(weight_rms=rms, params_b=MINICPM_B, eff_batch=1, gradient_noise_scale=None)
    want = max(2.0e-5, min(1.0e-4, raw)) * 1.0
    assert [d["prior_lr"] for d in t["field_lr"]] == [want] and t["override"] == []
    assert t["sweep_gate_lr"] == t["train_start_lr"] == t["plan_lr"] == t["trainer_lr"] == want
    if rms == 1.0:
        assert want == 1.0e-4
    assert _ev(ev, "sft_v2_field_recipe")[0]["damaged_base"] is True
    assert t["recipe"] == [{"clean_small_qwen": False, "recipe_key": None, "cheap_full": False, "bf16": True, "bf16_min_b": 0.3, "lr_mult": 1.0,
                            "eff_batch": 100, "strategy": "full"}]
    assert len(t["bf16"]) == 1  # bf16 weights (+ paged 8-bit AdamW where bitsandbytes is present)
    assert _ev(ev, "sft_v2_damaged_minicpm5_route") == [{"params_b": round(MINICPM_B, 4), "vocab": MINICPM_VOCAB}]
    # F1 = 52d319d + the research env on the plain FIELD route: same recipe, geometry, packing and LR at every step
    ev_f1, tr_f1 = _run(monkeypatch, tmp_path / "f1", damaged=False, env=F1_ENV, rms=rms)
    assert _trace(ev_f1, tr_f1) == t
    assert _ev(ev_f1, "sft_v2_damaged_minicpm5_route") == []


# ---- export: plain LlamaForCausalLM full weights, no remote code ---------------------------------------------------

def test_full_weight_export_has_no_remote_code(monkeypatch, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    from forge import telemetry
    from forge.tasks import common

    monkeypatch.setattr(telemetry, "write_into", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "event", lambda *a, **k: None)
    base = tmp_path / "base"
    _tiny_minicpm5_like().save_pretrained(base, safe_serialization=True)
    model = LlamaForCausalLM.from_pretrained(base, dtype=torch.bfloat16)

    class Tok:
        def save_pretrained(self, d):
            Path(d, "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))

    out = tmp_path / "out"
    state = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    common.save_adapter(model, Tok(), str(out), artifact_truth=sft_v2.ARTIFACT_COMPLETE_BEST, optimizer_step=1,
                        truth_reason="test", state_dict=state)
    files = sorted(os.listdir(out))
    assert not [f for f in files if f.endswith(".py")]
    assert "model.safetensors" in files and not [f for f in files if f.startswith("adapter_")]
    cfg = json.loads((out / "config.json").read_text())
    assert "auto_map" not in cfg and cfg["model_type"] == "llama" and cfg["vocab_size"] == MINICPM_VOCAB
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    re = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, trust_remote_code=False, local_files_only=True)
    assert type(re).__name__ == "LlamaForCausalLM"
