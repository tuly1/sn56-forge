"""Clean LFM2.5-1.2B-Instruct FIELD route (CONTINGENT release candidate 2026-09-28, branch
release-clean-lfm12-field-20260928).

An lfm2 base that the damage detectors read "clean" against exactly LiquidAI/LFM2.5-1.2B-Instruct, 0 < params_b <= 1.5,
takes the FIELD full route with the recipe replay-bench Phase AJ arm F executed as 11c41a7 + FORGE_V2_FIELD_TYPES=lfm2 +
FORGE_V2_FIELD_LR_MULT=2 (research/20260923-replay-bench/phase-aj/CARD.md): fp32 master + fused AdamW, LR clamp(analytic)
x 2, eff batch 100, generic floors 15k rows / 8M tokens, on SHORT-ROW tasks only (long_row_task False: sampled median <
256 tokens, where 11c41a7 runs the legacy LoRA; RESP-065). Long-row tasks keep the v2 LoRA. Every other base keeps its
11c41a7 route; FORGE_V2_CLEAN_SMALL_LFM2=0 restores 11c41a7 exactly."""
from __future__ import annotations

import copy
import importlib.util
import itertools
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
SWITCH = "FORGE_V2_CLEAN_SMALL_LFM2"
F_ENV = {"FORGE_V2_FIELD_TYPES": "lfm2", "FORGE_V2_FIELD_LR_MULT": "2"}
OFF = {SWITCH: "0"}  # == 11c41a7 routing
LFM12_B = 1.170340608  # LiquidAI/LFM2.5-1.2B-Instruct @0f604ada, 1,170,340,608 parameters
LFM_VOCAB = 65536
REF = "LiquidAI/LFM2.5-1.2B-Instruct"
AJ = dict(n_rows=80704, total_tokens=16_790_709)  # Phase AJ TRAIN (reduced cc4550ab)
HW = dict(n_gpus=1, is_kl=False)
F_RECIPE = {"bf16": False, "bf16_min_b": 2.0, "lr_mult": 2.0, "eff_batch": 100, "strategy": "full"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    # cab930c-equivalence suite: these tests pin the pre-full-FT-default admission, reachable only with the research
    # switch FORGE_V2_FULLFT_DEFAULT=0; the release behaviour is pinned by tests/test_fullft_default_route_table.py
    monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")


def _gate(params_b=LFM12_B, *, mt="lfm2", vocab=LFM_VOCAB, verdict="clean", ref=REF, data=AJ, long_rows=False, **kw):
    """The instruct.py gate block (field_full_route -> damaged / clean-small-Qwen / clean-small-LFM2 -> field_recipe)."""
    field = bool(sft_v2.field_full_route(params_b=params_b, model_type=mt, vocab_size=vocab, base_damage=verdict,
                                         base_clean_ref=ref, long_rows=long_rows, **HW, **data, **kw))
    dmg = bool(field and sft_v2.damaged_base_admits(mt, verdict, params_b, vocab))
    csq = bool(field and not dmg and sft_v2.clean_small_qwen_admits(mt, verdict, params_b, vocab))
    csl = bool(field and not dmg and not csq and sft_v2.clean_small_lfm2_admits(mt, verdict, params_b, ref,
                                                                                long_rows=long_rows))
    rec = sft_v2.field_recipe(mt, params_b, damaged_base=dmg or csq, clean_small_lfm2=csl) if field else None
    return {"field": field, "damaged": dmg, "csq": csq, "csl": csl, "recipe": rec}


def _with_env(monkeypatch, env, fn):
    with monkeypatch.context() as m:
        for k, v in env.items():
            m.setenv(k, v)
        return fn()


# ---- constants, reference, admission --------------------------------------------------------------------------------

def test_constants_and_defaults():
    assert sft_v2.CLEAN_SMALL_LFM2_ROUTE_DEFAULT == "1"
    assert sft_v2.CLEAN_SMALL_LFM2_REFERENCES == frozenset({damage.reference_id(REF)})
    assert 0 < LFM12_B <= sft_v2.CLEAN_SMALL_LFM2_MAX_PARAMS_B == 1.5
    assert sft_v2.CLEAN_SMALL_LFM2_LR_MULT == 2.0
    # the damaged small lfm2 band (fp32 master, LR x2) is untouched
    fam = sft_v2.DAMAGED_BASE_FAMILIES["lfm2"]
    assert (fam["small_max_b"], fam["small_lr_mult"], fam["bf16_min_b"]) == (1.5, 2.0, 1.5)


def test_clean_base_reference():
    fp = lambda v, src: {"verdict": v, "reason": "x", "source": src}  # noqa: E731
    rep = {"verdict": "clean", "detectors": {"identity": {"verdict": "unknown", "reason": "identity_unknown"},
                                              "fingerprint": fp("clean", REF)}}
    assert sft_v2.clean_base_reference(rep) == "liquidai/lfm2.5-1.2b-instruct"
    assert sft_v2.clean_base_reference({**rep, "verdict": "damaged"}) is None
    assert sft_v2.clean_base_reference({**rep, "verdict": "unknown"}) is None
    assert sft_v2.clean_base_reference(None) is None
    two = {"verdict": "clean", "detectors": {"a": fp("clean", REF), "b": fp("clean", "LiquidAI/LFM2-1.2B")}}
    assert sft_v2.clean_base_reference(two) is None
    assert sft_v2.clean_base_reference({"verdict": "clean", "detectors": {"a": fp("clean", None)}}) is None
    # an exact identity table key form maps to the same id
    assert sft_v2.clean_base_reference({"verdict": "clean", "detectors": {
        "identity": {"verdict": "clean", "reference": "LiquidAI__LFM2.5-1.2B-Instruct"}}}) == "liquidai/lfm2.5-1.2b-instruct"


def test_phase_aj_task_takes_exactly_arm_f(monkeypatch):
    new = _gate()
    assert new == {"field": True, "damaged": False, "csq": False, "csl": True, "recipe": F_RECIPE}
    f = _with_env(monkeypatch, {**OFF, **F_ENV}, _gate)
    assert {**f, "csl": True} == new and f["csl"] is False  # 11c41a7 + F env: plain FIELD route, same recipe
    c = _with_env(monkeypatch, OFF, _gate)
    assert c == {"field": False, "damaged": False, "csq": False, "csl": False, "recipe": None}  # 11c41a7 unset (arm C)


@pytest.mark.parametrize("long_rows", [True, None])
def test_long_or_unknown_row_length_keeps_11c41a7(monkeypatch, long_rows):
    got = _gate(long_rows=long_rows)
    assert got == _with_env(monkeypatch, OFF, lambda: _gate(long_rows=long_rows))
    assert got["field"] is False and got["csl"] is False
    assert sft_v2.clean_small_lfm2_admits("lfm2", "clean", LFM12_B, REF, long_rows=long_rows) is False
    assert sft_v2.clean_small_lfm2_admits("lfm2", "clean", LFM12_B, REF) is False  # default None fails closed


def test_env_f_on_the_new_commit_is_unchanged(monkeypatch):
    assert _with_env(monkeypatch, F_ENV, _gate)["recipe"] == F_RECIPE


@pytest.mark.parametrize("ref", [None, "LiquidAI/LFM2-1.2B", "LiquidAI/LFM2.5-1.2B-Base", "LiquidAI/LFM2.5-350M",
                                 "LiquidAI/LFM2-700M", "LiquidAI/LFM2.5-2.6B", "Qwen/Qwen2.5-0.5B-Instruct"])
@pytest.mark.parametrize("params_b", [0.36, 0.74, LFM12_B, 1.5])
def test_other_clean_lfm2_references_are_unchanged(monkeypatch, ref, params_b):
    got = _gate(params_b, ref=ref)
    assert got == _with_env(monkeypatch, OFF, lambda: _gate(params_b, ref=ref))
    assert got["field"] is False


@pytest.mark.parametrize("verdict", [None, "unknown", "damaged"])
@pytest.mark.parametrize("params_b", [0.36, LFM12_B, 1.5, 2.7])
@pytest.mark.parametrize("vocab", [65536, 128000])
def test_non_clean_verdicts_are_unchanged(monkeypatch, verdict, params_b, vocab):
    got = _gate(params_b, verdict=verdict, vocab=vocab)
    assert got == _with_env(monkeypatch, OFF, lambda: _gate(params_b, verdict=verdict, vocab=vocab))
    assert got["csl"] is False


@pytest.mark.parametrize("params_b", [1.5001, 2.0, 2.7, 3.5])
@pytest.mark.parametrize("vocab", [65536, 128000])
def test_above_the_size_band_nothing_changes(monkeypatch, params_b, vocab):
    got = _gate(params_b, vocab=vocab)
    assert got == _with_env(monkeypatch, OFF, lambda: _gate(params_b, vocab=vocab)) and got["csl"] is False


@pytest.mark.parametrize("mt", ["llama", "qwen2", "qwen3", "gemma2", "lfm2_moe", None])
def test_other_model_types_are_not_admitted(mt):
    assert sft_v2.clean_small_lfm2_admits(mt, "clean", LFM12_B, REF, long_rows=False) is False


def test_floors_equal_arm_f(monkeypatch):
    """The admission's row / token floors are F's (generic FIELD 15k / 8M) at every data regime, unknown tokens included."""
    rows = [None, 8000, 14999, 15000, 80704]
    toks = [None, 2_500_000, 7_999_999, 8_000_000, 16_790_709]
    for r, t in itertools.product(rows, toks):
        data = dict(n_rows=r, total_tokens=t)
        new = _gate(data=data)
        f = _with_env(monkeypatch, {**OFF, **F_ENV}, lambda: _gate(data=data))
        assert new["field"] == f["field"] and new["recipe"] == f["recipe"], (r, t)
        assert new["field"] == ((r is None or r >= 15000) and (t is None or t >= 8_000_000))


def test_hard_gates_still_apply():
    base = dict(params_b=LFM12_B, model_type="lfm2", vocab_size=LFM_VOCAB, base_damage="clean", base_clean_ref=REF, long_rows=False, **AJ)
    assert sft_v2.field_full_route(n_gpus=1, is_kl=False, **base) is True
    assert sft_v2.field_full_route(n_gpus=2, is_kl=False, **base) is False
    assert sft_v2.field_full_route(n_gpus=1, is_kl=True, **base) is False


@pytest.mark.parametrize("env", [OFF, {"FORGE_V2_FIELD": "0"}])
def test_off_switches_restore_11c41a7(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _gate()["field"] is False
    assert sft_v2.clean_small_lfm2_admits("lfm2", "clean", LFM12_B, REF, long_rows=False) is False


def test_lr_mult_env_keeps_precedence(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "1")
    assert _gate()["recipe"]["lr_mult"] == 1.0


def test_default_signatures_are_backward_compatible():
    # callers that do not pass the new keywords see 11c41a7 answers
    assert sft_v2.field_recipe("lfm2", LFM12_B) == {**F_RECIPE, "lr_mult": 1.0}
    assert sft_v2.field_full_route(params_b=LFM12_B, model_type="lfm2", vocab_size=LFM_VOCAB, base_damage="clean",
                                   **HW, **AJ) is False
    assert sft_v2.field_recipe("lfm2", LFM12_B, damaged_base=True, clean_small_lfm2=True) == \
        sft_v2.field_recipe("lfm2", LFM12_B, damaged_base=True)


# ---- dispatcher (deployed CLI / parser / dispatcher; model, identity, detectors and training are doubles) ----------

_contract_path = HERE / "test_anonymous_field_dispatch_contract.py"
_spec = importlib.util.spec_from_file_location("_anon_contract_clean_lfm2", _contract_path)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def _dispatch(monkeypatch, tmp_path, *, long_rows, ref=REF, verdict="clean", params_b=LFM12_B, env=None, median=None):
    from forge import cli
    from forge.tasks import instruct

    tmp_path.mkdir(parents=True, exist_ok=True)
    args, seen, stats = contract.prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=long_rows)
    st = json.loads(stats.read_text())
    st["dataset"]["total_tokens"] = AJ["total_tokens"]  # Phase AJ volume: short rows, but above the 8M token floor
    stats.write_text(json.dumps(st))
    loaded = instruct.load_base("probe")
    seen["model_paths"].clear()
    loaded.model.config.vocab_size = LFM_VOCAB
    monkeypatch.setattr(instruct, "model_param_billions", lambda m: params_b)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if median is not None:
        # the REAL long_row_task decides; only its tokenization is a double (every sampled row has `median` tokens)
        monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda picked, cols: list(picked))
        monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct",
                            lambda ex, tok, cap: [{"input_ids": [5] * median} for _ in ex])
    monkeypatch.setattr(idr, "identify", lambda model_dir, **kw: {"status": "unknown", "reason": "no_matching_table"})
    monkeypatch.setitem(damage.DETECTORS, "fingerprint",
                        lambda d, i, **kw: {"verdict": verdict, "reason": "double", "source": ref})

    def stop_v2(spec, deadline, rows, loaded_arg, tokenizer, **kw):
        seen["routes"].append({"handler": "sft_v2", "field": kw["field_route"], "damaged": kw["damaged_base_route"],
                               "csl": kw.get("clean_small_lfm2_route", False),
                               "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], "lfm2")})

    monkeypatch.setattr(sft_v2, "run", stop_v2)
    try:
        cli.main(args)
    except contract.StopAtLegacyPlan:
        pass
    return [{k: r[k] for k in ("handler", "field", "strategy") if k in r} | ({"csl": r["csl"]} if "csl" in r else {})
            for r in seen["routes"]], dict(seen["events"])


def test_dispatch_clean_lfm12_instruct_short_rows_take_field_full(monkeypatch, tmp_path):
    routes, ev = _dispatch(monkeypatch, tmp_path, long_rows=False)
    assert routes == [{"handler": "sft_v2", "field": True, "strategy": "full", "csl": True}]
    assert ev["sft_v2_field_gate"]["clean_small_lfm2_route"] is True
    assert ev["sft_v2_field_full_route"]["clean_small_lfm2"] is True


def test_dispatch_clean_lfm12_instruct_long_rows_keep_v2_lora(monkeypatch, tmp_path):
    routes, ev = _dispatch(monkeypatch, tmp_path, long_rows=True)
    assert routes == [{"handler": "sft_v2", "field": False, "strategy": "lora", "csl": False}]
    assert ev["sft_v2_field_gate"]["clean_small_lfm2_route"] is False
    off, _ = _dispatch(monkeypatch, tmp_path / "off", long_rows=True, env=OFF)
    assert off == routes  # == 11c41a7


@pytest.mark.parametrize("median,want_field", [(1, True), (254, True), (255, True), (256, False), (257, False), (403, False)])
def test_dispatch_row_median_boundary(monkeypatch, tmp_path, median, want_field):
    """Real long_row_task at the 256-token cut: median 255 -> short branch -> new FIELD route; 256 / 257 -> long branch
    -> the v2 LoRA exactly as 11c41a7; switch off -> 11c41a7 on both sides (legacy LoRA below, v2 LoRA at/above)."""
    assert sft_v2.LORA_MIN_MEDIAN_TOKENS == 256
    routes, ev = _dispatch(monkeypatch, tmp_path / "new", long_rows=True, median=median)
    assert ev["sft_v2_row_length_gate"]["median_tokens"] == median
    assert ev["sft_v2_row_length_gate"]["long_rows"] is (median >= 256)
    if want_field:
        assert routes == [{"handler": "sft_v2", "field": True, "strategy": "full", "csl": True}]
    else:
        assert routes == [{"handler": "sft_v2", "field": False, "strategy": "lora", "csl": False}]
    off, _ = _dispatch(monkeypatch, tmp_path / "off", long_rows=True, median=median, env=OFF)
    assert off == ([{"handler": "legacy", "field": False, "strategy": "lora"}] if want_field else routes)
    f, _ = _dispatch(monkeypatch, tmp_path / "f", long_rows=True, median=median, env={**OFF, **F_ENV})
    assert f == [{"handler": "sft_v2", "field": True, "strategy": "full", "csl": False}]  # 11c41a7 + F env, both sides


@pytest.mark.parametrize("long_rows", [False, True])
def test_dispatch_arm_f_env_on_11c41a7_semantics_is_the_same_route(monkeypatch, tmp_path, long_rows):
    routes, _ = _dispatch(monkeypatch, tmp_path, long_rows=long_rows, env={**OFF, **F_ENV})
    assert routes == [{"handler": "sft_v2", "field": True, "strategy": "full", "csl": False}]


@pytest.mark.parametrize("long_rows,want", [(False, {"handler": "legacy", "field": False, "strategy": "lora"}),
                                            (True, {"handler": "sft_v2", "field": False, "strategy": "lora", "csl": False})])
def test_dispatch_off_switch_is_11c41a7(monkeypatch, tmp_path, long_rows, want):
    routes, _ = _dispatch(monkeypatch, tmp_path, long_rows=long_rows, env=OFF)
    assert routes == [want]


@pytest.mark.parametrize("ref", ["LiquidAI/LFM2-1.2B", "LiquidAI/LFM2.5-1.2B-Base"])
def test_dispatch_other_clean_lfm12_bases_keep_11c41a7(monkeypatch, tmp_path, ref):
    routes, _ = _dispatch(monkeypatch, tmp_path, long_rows=False, ref=ref)
    assert routes == [{"handler": "legacy", "field": False, "strategy": "lora"}]  # short rows: 11c41a7 legacy LoRA


# ---- sft_v2.run on CPU: candidate route == arm F env on the plain FIELD route ---------------------------------------

def _tiny_lfm2():
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM

    torch.manual_seed(17)
    return Lfm2ForCausalLM(Lfm2Config(
        vocab_size=LFM_VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, layer_types=["conv", "full_attention"], max_position_embeddings=4096, bos_token_id=0,
        eos_token_id=1, pad_token_id=1, tie_word_embeddings=True, use_cache=False)).eval()


def _rows(n=34):
    out = []
    for i in range(n):
        ids = [0] + [5 + i] * (3 + i % 5) + [17, 1]
        out.append(dict(input_ids=ids, labels=([-100, -100] + ids[2:]) if i % 2 else ids.copy()))
    return out


def _run(monkeypatch, tmp_path, *, csl: bool, env=None, rms=1.0):
    """sft_v2.run on CPU with a tiny lfm2 (vocab 65536) at params_b 1.1703: the real route, recipe, geometry and LR code;
    training, timing, dev scoring and export are doubles. Returns (events, trainers)."""
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
    sft_v2.run(spec, deadline, source, NS(model=_tiny_lfm2()), NS(pad_token_id=1), baseline_summary=None,
               params_b=LFM12_B, per_gpu_gb=80.0, strategy_override="full", field_route=True,
               damaged_base_route=False, clean_small_lfm2_route=csl)
    return events, trainers


def _ev(events, name):
    return [d for n, d in events if n == name]


def _trace(events, trainers):
    plan0 = _ev(events, "sft_v2_plan")[0]
    t = trainers[-1].args
    return {"field_lr": _ev(events, "sft_v2_field_lr"), "override": _ev(events, "sft_v2_lr_override"),
            "sweep_gate_lr": _ev(events, "sft_v2_sweep_gate")[0]["lr_override"],
            "train_start_lr": _ev(events, "sft_v2_train_start")[0]["lr"], "plan": plan0,
            "trainer_lr": t.learning_rate, "trainer_warmup": t.warmup_steps,
            "trainer_batch": (t.per_device_train_batch_size, t.gradient_accumulation_steps),
            "trainer_optim": str(t.optim), "trainer_precision": (getattr(t, "bf16", None), getattr(t, "fp16", None)),
            "packing": _ev(events, "sft_v2_packing"), "bf16": _ev(events, "sft_v2_field_bf16"),
            "route": _ev(events, "sft_v2_field_route"), "recipe": _ev(events, "sft_v2_field_recipe")}


@pytest.mark.parametrize("rms", [1.0, 0.011690166778862476])  # clamped at 1e-4; the real base's layer RMS
def test_run_candidate_equals_phase_aj_arm_f(monkeypatch, tmp_path, rms):
    ev, tr = _run(monkeypatch, tmp_path / "route", csl=True, rms=rms)
    t = _trace(ev, tr)
    assert _ev(ev, "sft_v2_clean_small_lfm2_route") == [{"params_b": round(LFM12_B, 4), "vocab": LFM_VOCAB, "lr_mult": 2.0}]
    assert t["recipe"] == [{"damaged_base": False, "clean_small_qwen": False, "recipe_key": None, "cheap_full": False,
                           **F_RECIPE}]
    assert t["bf16"] == []  # fp32 master, no bf16 switch
    fl = t["field_lr"]
    assert len(fl) == 1 and fl[0]["mult"] == 2.0 and t["override"] == []
    assert fl[0]["prior_lr"] == max(2.0e-5, min(1.0e-4, fl[0]["raw"])) * 2.0
    assert t["sweep_gate_lr"] == t["train_start_lr"] == t["trainer_lr"] == fl[0]["prior_lr"]
    # arm F = 11c41a7 semantics (switch off) + the research env on the plain FIELD route: identical at every step
    with monkeypatch.context() as m:
        ev_f, tr_f = _run(m, tmp_path / "f", csl=False, env={**OFF, **F_ENV}, rms=rms)
    assert _trace(ev_f, tr_f) == t
    assert _ev(ev_f, "sft_v2_clean_small_lfm2_route") == []
    # without the flag and without env, 11c41a7's plain FIELD route would run LR x1: the flag is what carries x2
    with monkeypatch.context() as m:
        ev_1, tr_1 = _run(m, tmp_path / "x1", csl=False, rms=rms)
    assert _ev(ev_1, "sft_v2_field_lr")[0]["mult"] == 1.0
