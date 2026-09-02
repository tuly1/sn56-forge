"""Focused CPU contract for the twelve-model 2026 pool robustness route."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from forge import model
from forge.tasks import instruct


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "ops/2026-pool-route-matrix.json").read_text())
EXPECTED_ROWS = [
    (
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "qwen3_5",
        ("Qwen3_5ForConditionalGeneration",),
        "QWEN35_BATCH1_ACC16",
    ),
    (
        "Qwen/Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        "qwen3_5",
        ("Qwen3_5ForConditionalGeneration",),
        "QWEN35_BATCH1_ACC16",
    ),
    (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
        "qwen3_5",
        ("Qwen3_5ForConditionalGeneration",),
        "QWEN35_BATCH1_ACC16",
    ),
    (
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
        "qwen3_5",
        ("Qwen3_5ForConditionalGeneration",),
        "QWEN35_BATCH1_ACC16",
    ),
    (
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
        "gemma4",
        ("Gemma4ForConditionalGeneration",),
        "PRODUCTION_LORA",
    ),
    (
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
        "gemma4",
        ("Gemma4ForConditionalGeneration",),
        "PRODUCTION_LORA",
    ),
    (
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        "15d6157fb6df3f8272e2fe90e18f57727ccf02a125c94469198b0f3281510185",
        "lfm2",
        ("Lfm2ForCausalLM",),
        "PRODUCTION_LORA",
    ),
    (
        "LiquidAI/LFM2.5-2.6B",
        "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
        "480f63fa8e1efa534ae8b92774b3b53b8d6812d62a726e9ecfc866933662f273",
        "lfm2",
        ("Lfm2ForCausalLM",),
        "PRODUCTION_LORA",
    ),
    (
        "LiquidAI/LFM2.5-8B-A1B",
        "5dd22602c2e9f6a097b1de4c4efe0658b605015c",
        "9c0255c2d5c744c99b760a12edca2572935348dae340e79e2e6625af975d2d68",
        "lfm2_moe",
        ("Lfm2MoeForCausalLM",),
        "PRODUCTION_LORA",
    ),
    (
        "ibm-granite/granite-4.1-3b",
        "c0650403e44e78ec0262dab1c90914c65b196c4e",
        "9a0e589b69e7d3ad9fb9fb2c844aa7d7156e052cb7ea4211de7de48ab7c8525c",
        "granite",
        ("GraniteForCausalLM",),
        "PRODUCTION_LORA",
    ),
    (
        "ibm-granite/granite-4.1-8b",
        "1504002f650e656a0a3789d99574df12e3e94ed0",
        "dea9d856cb57018117fe2fe3366f37cb4aa39424890061db2c0045a6a4efbda0",
        "granite",
        ("GraniteForCausalLM",),
        "PRODUCTION_LORA",
    ),
    (
        "openbmb/MiniCPM5-1B",
        "87179e5c1f455ef22e6223592d2d61351b525bfc",
        "6a6509b646cb3169616c5ffc3196e7ccaf9d4d6bc17b266581d241a31c217714",
        "llama",
        ("LlamaForCausalLM",),
        "PRODUCTION_LORA",
    ),
]


@dataclass(frozen=True)
class _Plan:
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    gradient_checkpointing: bool = True


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_exact_twelve_config_model_routes_are_complete_and_deterministic():
    rows = MATRIX["models"]
    assert MATRIX["authority"] == {
        "public_god_commit": "f11b6fca4244a9227ed1531f5fc9b189972bf9ac",
        "public_god_tree": "d7b42dc1a941c28e613792ce60eef7c62acd295c",
        "pool_path": "core/oversampled_later_models.json",
        "pool_sha256": "734cfb9fac917e5e4f0b2ada383bc8f125f0929cec5462ac0afe29bf3ba5ba2e",
        "row_count": 12,
        "config_source": (
            "public Hugging Face model repository at the recorded immutable revision"
        ),
    }
    assert len(rows) == 12
    assert len({row["model_id"] for row in rows}) == 12
    assert all(len(row["revision"]) == 40 for row in rows)
    assert all(len(row["config_sha256"]) == 64 for row in rows)
    assert [
        (
            row["model_id"],
            row["revision"],
            row["config_sha256"],
            row["model_type"],
            tuple(row["architectures"]),
            row["route"],
        )
        for row in rows
    ] == EXPECTED_ROWS

    for row in rows:
        base = _Plan()
        routed, changed = model.conservative_qwen35_plan(
            SimpleNamespace(config=SimpleNamespace(model_type=row["model_type"])),
            base,
        )
        if row["route"] == "QWEN35_BATCH1_ACC16":
            assert row["model_type"] == "qwen3_5"
            assert changed is True
            assert routed.per_device_batch_size == 1
            assert routed.grad_accum_steps == 16
            assert routed.per_device_batch_size * routed.grad_accum_steps == 16
            assert routed.gradient_checkpointing is base.gradient_checkpointing
        else:
            assert row["route"] == "PRODUCTION_LORA"
            assert changed is False
            assert routed is base

    nested = SimpleNamespace(
        config=SimpleNamespace(
            model_type="conditional_wrapper",
            text_config=SimpleNamespace(model_type="qwen3_5_text"),
        )
    )
    nested_route, nested_changed = model.conservative_qwen35_plan(nested, _Plan())
    assert nested_changed is True
    assert (nested_route.per_device_batch_size, nested_route.grad_accum_steps) == (1, 16)

    already_safe = _Plan(per_device_batch_size=1, grad_accum_steps=16)
    safe_route, safe_changed = model.conservative_qwen35_plan(nested, already_safe)
    assert safe_changed is False
    assert safe_route is already_safe

    drifted = _Plan(per_device_batch_size=2, grad_accum_steps=4)
    corrected, corrected_changed = model.conservative_qwen35_plan(nested, drifted)
    assert corrected_changed is True
    assert (corrected.per_device_batch_size, corrected.grad_accum_steps) == (1, 16)


def test_floor_best_export_and_non_sft_handlers_remain_unchanged():
    source = inspect.getsource(instruct.run)
    route = source.index("conservative_qwen35_plan(loaded.model, plan)")
    lora = source.index("model = attach_lora(")
    floor = source.index("truth_reason=\"pretraining_floor\"")
    tokenize_start = source.index("initial_seq_len = effective_sft_seq_len(")
    trainer_policy = source.index("return _make_trainer(")
    train = source.index("_train_ladder(")
    final_guard = source.index("if should_final_save(tracker, final_step=final_step):")
    assert route < lora < floor < tokenize_start
    assert trainer_policy < train < final_guard

    expected = {
        "forge/tasks/common.py": (
            "9a7c9d546e0c9e4c8a344e767480d252b6aebb7c72f79acca62ef118111f93a0"
        ),
        "forge/tasks/fallback.py": (
            "16a556b7c5104f45584b0bdc24bb2f050835eb2945be496a9e659cef6e9ceed0"
        ),
        "forge/tasks/dpo.py": (
            "25bb3d48c2ff1a568d434d3983d4384af1e060967072429cbab68a0b74e1ccd5"
        ),
        "forge/tasks/grpo.py": (
            "16ecd365a984b807b7849eb86337092b789ea5bea1e2585768fd5f7cdc92897d"
        ),
        "forge/tuning/plan.py": (
            "302b0c1da3de865ed08af186cf0b2fc8452173dc9a04ab88eee41a1de6578787"
        ),
    }
    assert {path: _sha256(path) for path in expected} == expected
