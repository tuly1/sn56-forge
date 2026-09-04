"""CPU contracts for the study-only recipe override hook.

Byte-identity when unset is the load-bearing property: every helper must hand
back its input object untouched, the SFT handler must read the environment
exactly once before the adapter is attached, and the survival machinery
(geometry, checkpointing, admission, wall planner, artifact truth) must remain
unreachable from the payload.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from forge.data import tokenize
from forge.tasks import instruct
from forge.tuning import overrides
from forge.tuning.overrides import (
    ENV_NAME,
    RecipeOverrides,
    apply_plan_overrides,
    apply_training_kwargs_overrides,
    cap_epochs,
    load_recipe_overrides,
    long_rows_policy,
    neftune_alpha,
    parse_recipe_overrides,
    plan_override_diff,
)
from forge.tuning.plan import TrainPlan


def _plan(**changes) -> TrainPlan:
    base = dict(
        lora_r=32, lora_alpha=64, lora_dropout=0.05, learning_rate=1.5e-4,
        per_device_batch_size=4, grad_accum_steps=4, max_seq_len=4096,
        num_epochs=2, warmup_ratio=0.03, weight_decay=0.0,
        optimizer="adamw_torch_fused", lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=False, bf16=True, fp16=False, strategy="lora",
    )
    base.update(changes)
    return TrainPlan(**base)


def _kwargs(scheduler: str = "cosine_with_min_lr") -> dict:
    return {
        "learning_rate": 1.5e-4,
        "lr_scheduler_type": scheduler,
        "lr_scheduler_kwargs": {"min_lr_rate": 0.25} if scheduler == "cosine_with_min_lr" else {},
        "neftune_noise_alpha": 5.0,
        "num_train_epochs": 2,
    }


class _CharTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, **_kw):
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


# --- unset => inert ----------------------------------------------------------

def test_unset_environment_is_inert_everywhere():
    recipe = load_recipe_overrides({})
    assert recipe == RecipeOverrides()
    assert recipe.present is False and recipe.active is False
    plan = _plan()
    assert apply_plan_overrides(plan, recipe) is plan
    kwargs = _kwargs()
    assert apply_training_kwargs_overrides(kwargs, recipe) is kwargs
    assert apply_training_kwargs_overrides(kwargs, None) is kwargs
    assert neftune_alpha(False, recipe) == 5.0
    assert neftune_alpha(False, None) == 5.0
    assert neftune_alpha(True, recipe) is None
    assert cap_epochs(3.37, recipe) == 3.37
    assert cap_epochs(None, recipe) is None
    assert long_rows_policy(recipe) == "drop"
    assert long_rows_policy(None) == "drop"
    assert plan_override_diff(plan, plan) == {}


def test_empty_object_is_present_but_inert():
    recipe = load_recipe_overrides({ENV_NAME: "{}"})
    assert recipe.present is True and recipe.active is False and recipe.error is None
    plan = _plan()
    assert apply_plan_overrides(plan, recipe) is plan
    assert recipe.record() == {"source": ENV_NAME, "accepted": True, "error": None, "values": {}}


# --- parsing -----------------------------------------------------------------

def test_valid_payload_parses_every_whitelisted_key():
    payload = {
        "learning_rate": 7.5e-5, "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1,
        "num_epochs": 1.2, "epochs_cap": 1.2, "warmup_ratio": 0.05, "weight_decay": 0.01,
        "lr_scheduler": "cosine_with_min_lr", "min_lr_rate": 0.0, "neftune_alpha": 10,
        "max_seq_len": 2048, "long_rows": "truncate",
    }
    values = parse_recipe_overrides(json.dumps(payload))
    assert set(values) == set(payload) == set(overrides.ALLOWED_KEYS)
    assert values["neftune_alpha"] == 10.0 and values["min_lr_rate"] == 0.0
    assert values["lora_r"] == 64 and values["num_epochs"] == 1.2
    recipe = load_recipe_overrides({ENV_NAME: json.dumps(payload)})
    assert recipe.active and recipe.error is None and recipe.record()["accepted"] is True


@pytest.mark.parametrize("raw", [
    "not json",
    "[1, 2]",
    '{"learning_rate": "fast"}',
    '{"learning_rate": 0.5}',
    '{"lora_r": 2.5}',
    '{"lora_r": true}',
    '{"lr_scheduler": "polynomial"}',
    '{"long_rows": "pad"}',
    '{"neftune_alpha": -1}',
    '{"max_seq_len": 16}',
    '{"epochs_cap": 0}',
    '{"learning_rate": 1e-4, "unknown_knob": 1}',
])
def test_bad_payloads_disable_the_whole_payload(raw):
    recipe = load_recipe_overrides({ENV_NAME: raw})
    assert recipe.present is True
    assert recipe.active is False
    assert recipe.values == {}
    assert recipe.error and recipe.error.startswith("ValueError:")
    assert recipe.record()["accepted"] is False
    plan = _plan()
    assert apply_plan_overrides(plan, recipe) is plan


@pytest.mark.parametrize("key", [
    "per_device_batch_size", "grad_accum_steps", "gradient_checkpointing",
    "strategy", "optimizer", "bf16", "hours_to_complete", "max_steps",
    "admission_ceiling", "eval_steps",
])
def test_survival_and_geometry_fields_are_not_overridable(key):
    recipe = load_recipe_overrides({ENV_NAME: json.dumps({"learning_rate": 1e-4, key: 1})})
    assert recipe.active is False and "unknown override keys" in recipe.error


def test_oversized_payload_is_rejected():
    raw = json.dumps({"learning_rate": 1e-4, "lr_scheduler": "cosine" + " " * 9000})
    recipe = load_recipe_overrides({ENV_NAME: raw})
    assert recipe.active is False and "exceeds" in recipe.error


# --- plan application ----------------------------------------------------------

def test_plan_fields_replace_and_geometry_is_untouched():
    plan = _plan()
    recipe = load_recipe_overrides({ENV_NAME: json.dumps({
        "learning_rate": 7.5e-5, "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1,
        "num_epochs": 1.2, "warmup_ratio": 0.05, "weight_decay": 0.01, "lr_scheduler": "cosine",
    })})
    applied = apply_plan_overrides(plan, recipe)
    assert applied is not plan
    assert (applied.learning_rate, applied.lora_r, applied.lora_alpha, applied.lora_dropout) == (7.5e-5, 64, 128, 0.1)
    assert (applied.num_epochs, applied.warmup_ratio, applied.weight_decay, applied.lr_scheduler) == (1.2, 0.05, 0.01, "cosine")
    for name in ("per_device_batch_size", "grad_accum_steps", "max_seq_len", "gradient_checkpointing",
                 "optimizer", "bf16", "fp16", "strategy"):
        assert getattr(applied, name) == getattr(plan, name)
    assert set(plan_override_diff(plan, applied)) == {
        "learning_rate", "lora_r", "lora_alpha", "lora_dropout", "num_epochs",
        "warmup_ratio", "weight_decay", "lr_scheduler",
    }
    # the survival rung ladder keeps its geometry; the recipe rides along on every rung
    assert [(p.per_device_batch_size, p.grad_accum_steps, p.gradient_checkpointing, p.max_seq_len)
            for p in instruct._plans(applied, (1280,))] == \
           [(p.per_device_batch_size, p.grad_accum_steps, p.gradient_checkpointing, p.max_seq_len)
            for p in instruct._plans(plan, (1280,))]
    assert all(p.lora_r == 64 and p.learning_rate == 7.5e-5 for p in instruct._plans(applied, (1280,)))


def test_sequence_cap_override_only_lowers_the_evaluator_cap():
    plan = _plan()
    lower = apply_plan_overrides(plan, load_recipe_overrides({ENV_NAME: '{"max_seq_len": 2048}'}))
    assert lower.max_seq_len == 2048
    higher = apply_plan_overrides(plan, load_recipe_overrides({ENV_NAME: '{"max_seq_len": 8192}'}))
    assert higher is plan  # no change => same object


def test_epochs_cap_lowers_plan_epochs_and_time_aware_result():
    plan = _plan()
    recipe = load_recipe_overrides({ENV_NAME: '{"epochs_cap": 1.25}'})
    assert apply_plan_overrides(plan, recipe).num_epochs == 1.25
    assert cap_epochs(3.62, recipe) == 1.25
    assert cap_epochs(1.0, recipe) == 1.0
    assert cap_epochs(None, recipe) is None
    both = load_recipe_overrides({ENV_NAME: '{"num_epochs": 3.0, "epochs_cap": 1.5}'})
    assert apply_plan_overrides(plan, both).num_epochs == 1.5
    only_epochs = load_recipe_overrides({ENV_NAME: '{"num_epochs": 1.2}'})
    assert apply_plan_overrides(plan, only_epochs).num_epochs == 1.2
    assert cap_epochs(4.0, only_epochs) == 4.0  # no cap => time-aware result untouched


def test_identical_values_do_not_create_a_new_plan():
    plan = _plan()
    recipe = load_recipe_overrides({ENV_NAME: '{"learning_rate": 0.00015, "lora_r": 32}'})
    assert recipe.active
    assert apply_plan_overrides(plan, recipe) is plan


# --- training kwargs -------------------------------------------------------------

def test_min_lr_rate_reaches_the_floored_cosine_only():
    recipe = load_recipe_overrides({ENV_NAME: '{"min_lr_rate": 0.0}'})
    kwargs = _kwargs()
    out = apply_training_kwargs_overrides(kwargs, recipe)
    assert out is not kwargs and out["lr_scheduler_kwargs"] == {"min_lr_rate": 0.0}
    assert kwargs["lr_scheduler_kwargs"] == {"min_lr_rate": 0.25}  # input untouched
    linear = _kwargs("linear")
    assert apply_training_kwargs_overrides(linear, recipe) is linear


def test_neftune_override_semantics():
    off_zero = load_recipe_overrides({ENV_NAME: '{"neftune_alpha": 0}'})
    off_null = load_recipe_overrides({ENV_NAME: '{"neftune_alpha": null}'})
    ten = load_recipe_overrides({ENV_NAME: '{"neftune_alpha": 10}'})
    assert neftune_alpha(False, off_zero) is None
    assert neftune_alpha(False, off_null) is None
    assert neftune_alpha(False, ten) == 10.0
    # KL tasks never get NEFTune, override or not (base reference must stay clean)
    assert neftune_alpha(True, ten) is None
    unrelated = load_recipe_overrides({ENV_NAME: '{"learning_rate": 1e-4}'})
    assert neftune_alpha(False, unrelated) == 5.0


# --- tokenization: long-row policy -------------------------------------------------

def test_long_row_policy_truncate_keeps_prompt_and_cuts_completion():
    tok = _CharTokenizer()
    rows = [{"prompt_text": "abcd", "completion_text": "0123456789"}]  # 5 prompt ids + 10 + EOS = 16
    assert tokenize.tokenize_instruct(rows, tok, 16) == tokenize.tokenize_instruct(rows, tok, 16, on_overflow="drop")
    assert tokenize.tokenize_instruct(rows, tok, 8) == []
    kept = tokenize.tokenize_instruct(rows, tok, 8, on_overflow="truncate")
    assert len(kept) == 1
    assert kept[0]["input_ids"] == [1, 97, 98, 99, 100, 48, 49, 50]
    assert kept[0]["labels"] == [-100] * 5 + [48, 49, 50]
    # prompt alone fills the cap => still dropped (nothing supervised would remain)
    assert tokenize.tokenize_instruct(rows, tok, 5, on_overflow="truncate") == []
    # a row that fits is byte-identical under both policies
    short = tokenize.tokenize_instruct(rows, tok, 64, on_overflow="truncate")
    assert short == tokenize.tokenize_instruct(rows, tok, 64)
    assert short[0]["input_ids"][-1] == tok.eos_token_id
    with pytest.raises(ValueError):
        tokenize.tokenize_instruct(rows, tok, 8, on_overflow="pad")


def test_long_rows_policy_helper():
    assert long_rows_policy(load_recipe_overrides({ENV_NAME: '{"long_rows": "truncate"}'})) == "truncate"
    assert long_rows_policy(load_recipe_overrides({ENV_NAME: '{"long_rows": "drop"}'})) == "drop"
    assert long_rows_policy(load_recipe_overrides({ENV_NAME: '{"learning_rate": 1e-4}'})) == "drop"


# --- wiring into the SFT handler (source shape) ---------------------------------------

def test_handler_reads_overrides_once_before_the_adapter_and_after_the_routes():
    run = inspect.getsource(instruct.run)
    assert run.count("load_recipe_overrides()") == 1
    read = run.index("recipe = load_recipe_overrides()")
    route = run.index("conservative_qwen35_plan(loaded.model, plan)")
    quasar = run.index("conservative_quasar_plan(loaded.model, plan)")
    applied = run.index("plan = apply_plan_overrides(plan, recipe)")
    strategy = run.index('"strategy_chosen"')
    lora = run.index("model = attach_lora(")
    seq = run.index("initial_seq_len = effective_sft_seq_len(model, plan.max_seq_len)")
    assert read < route < quasar < applied < strategy < lora < seq
    assert "on_overflow=long_rows_policy(recipe)" in run
    assert "candidate_epochs = cap_epochs(candidate_epochs, recipe)" in run
    assert run.index("candidate_epochs = cap_epochs(candidate_epochs, recipe)") < run.index(
        'timing["probe_per_step"] = probe_per_step')
    assert "recipe=recipe)" in run  # make() threads the recipe into the exact production trainer factory
    assert "neftune=recipe_neftune_alpha(is_kl, recipe) is not None" in run
    factory = inspect.getsource(instruct._make_trainer)
    assert "apply_training_kwargs_overrides(" in factory
    assert factory.index("build_training_kwargs(") < factory.index("TrainingArguments(")
    assert "neftune_alpha=recipe_neftune_alpha(is_kl, recipe)" in factory


def test_survival_modules_stay_hash_pinned_and_untouched():
    # The hook lives in overrides.py/instruct.py/tokenize.py only; the hash-pinned
    # survival modules are asserted unchanged by tests/test_2026_pool_robustness.py.
    source = inspect.getsource(overrides)
    for forbidden in ("per_device_batch_size", "grad_accum_steps", "gradient_checkpointing",
                      "_ADMISSION_CEILING", "_STEADY_STATE_HEADROOM", "max_steps", "artifact_truth"):
        assert forbidden not in source


def test_environment_roundtrip(monkeypatch):
    monkeypatch.setenv(ENV_NAME, json.dumps({"learning_rate": 1e-4, "neftune_alpha": None}))
    recipe = load_recipe_overrides()
    assert recipe.values == {"learning_rate": 1e-4, "neftune_alpha": None}
    monkeypatch.delenv(ENV_NAME)
    assert load_recipe_overrides() == RecipeOverrides()


def test_telemetry_records_only_when_present(monkeypatch):
    events: list[tuple[str, dict]] = []
    metas: list[dict] = []
    monkeypatch.setattr(instruct.telemetry, "event", lambda name, **kv: events.append((name, kv)))
    monkeypatch.setattr(instruct.telemetry, "set_meta", lambda **kv: metas.append(kv))
    # Mirror the handler's read-once block exactly (the handler itself needs a model to run).
    block = inspect.getsource(instruct.run)
    start = block.rindex("\n", 0, block.index("recipe = load_recipe_overrides()")) + 1
    end = block.index("loaded = load_base(")
    snippet = "\n".join(line[4:] for line in block[start:end].splitlines())
    monkeypatch.delenv(ENV_NAME, raising=False)
    namespace = {"load_recipe_overrides": load_recipe_overrides, "telemetry": instruct.telemetry}
    exec(snippet, namespace)
    assert events == [] and metas == []
    monkeypatch.setenv(ENV_NAME, '{"learning_rate": 1e-4}')
    exec(snippet, namespace)
    assert events[0][0] == "recipe_overrides" and events[0][1]["accepted"] is True
    assert metas[0]["recipe_overrides"]["values"] == {"learning_rate": 1e-4}
    monkeypatch.setenv(ENV_NAME, "{bad")
    exec(snippet, namespace)
    assert events[-1][1]["accepted"] is False and events[-1][1]["values"] == {}
