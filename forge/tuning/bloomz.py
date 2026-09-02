"""Fail-closed BloomZ-560M public-fixture experiment controls.

This route is intentionally unreachable during ordinary tournament handling.
It exists only when an operator supplies an explicit experiment arm plus the
frozen fixture manifest. Both arms use identical unpacked sequence geometry and
retain exactly four externally scored decision states.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import stat
import time
from typing import Any, Iterable, Mapping

from forge.tuning.plan import TrainPlan


MODEL_REPO = "bigscience/bloomz-560m"
MODEL_REVISION = "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05"
MODEL_CONFIG_SHA256 = "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb"
MODEL_WEIGHTS_SHA256 = "365b2c5e9bd1057eb1e3f1a4fc3f89ae6584d20f24b682d2406bc7e90178ec13"
MODEL_TOKENIZER_SHA256 = "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3"
MODEL_TOKENIZER_CONFIG_BYTES = 222
MODEL_TOKENIZER_CONFIG_SHA256 = (
    "ae85f7ec32efe4ba09f3914743b0187528eab0322fe90c4e077a9229d1de64a9"
)
MODEL_SPECIAL_TOKENS_MAP_BYTES = 85
MODEL_SPECIAL_TOKENS_MAP_SHA256 = (
    "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"
)
MODEL_FORBIDDEN_TOKENIZER_FILES = frozenset(
    {"added_tokens.json", "chat_template.jinja"}
)
DATASET_REPO = "AlekseyKorshuk/evol-codealpaca-v1-dpo"
DATASET_REVISION = "31c087a1492db443a3ace4247ef1880678b27aa4"
DATASET_PARQUET_SHA256 = "b7d98f92731ad075bd01c1088c59816f05cf0f49605856b7e3007482a419535a"
FIXTURE_MANIFEST_SHA256 = "f5e7ffb590a05ba3bf4ab925b442be6bcd4f743d8ddc9603f8e3fa6c93c327c7"
TRAINING_MANIFEST_SHA256 = "3efcd00e9cd8d70c15bb324c264723ea292b5ed8c64bcdbf669f4a034372c336"
TRAINING_IMAGE = (
    "axolotlai/axolotl@"
    "sha256:97fba6ae924a55059bf48c5996014f0675d569df1b9c96e0cb0a0f922f355883"
)
EVALUATOR_IMAGE = (
    "gradientsio/text-evaluator:basilica@"
    "sha256:860d49c7317a82b68d93b7e0e257091d810fdea12eee3013f373903092d279d0"
)
SCORE_DRIVER_SHA256 = "6952bf4a9b365fa00387b87dd813eaf69d1ad8d0a555a668751990a673a1b0a3"

FULL_LR = 1.0e-4
ALLOWED_FULL_LRS = frozenset({FULL_LR})
CONTROL_LR = 1.5e-4
SEQUENCE_LENGTH = 2048
SELECTION_SEQUENCE_LENGTH = 4096
MICROBATCH = 1
GRAD_ACCUM = 16
MAX_RETAINED_CHECKPOINTS = 4
MIN_COMPLETED_EVALS = 4
MATCHED_DECISION_STEPS = 256
MODEL_PARAMS_B = 0.559214592
EXPECTED_TRAIN_ROWS = 38_346
EXPECTED_TRAIN_TOKENIZED_ROWS = 38_099
EXPECTED_TRAIN_DROPPED_ROWS = 247
EXPECTED_DEV_ROWS = 1_024
EXPECTED_DEV_TOKENIZED_ROWS = 1_021

LEASE_TOTAL_SECONDS = 30_600
LEASE_BOOTSTRAP_ALLOWANCE_SECONDS = 1_200
LEASE_SCIENCE_WINDOW_SECONDS = 24_000
LEASE_HOURLY_RATE = "2.006720430"
LEASE_MAX_COST = "17.057123655"
LEASE_CLOSURE_RESERVE_SECONDS = 5_400
LEASE_STAGE_MAXIMA = {
    "admission": {"count": 2, "max_each_seconds": 600},
    "training": {"count": 2, "max_each_seconds": 7_200},
    "dev_score": {"count": 8, "max_each_seconds": 570},
    "validation": {"count": 8, "max_each_seconds": 270},
    "confirmation_score": {"count": 2, "max_each_seconds": 570},
}

NETWORK_DEFAULT_DENY_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.EACCES,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
        errno.ETIMEDOUT,
    }
)
IPV4_TIMEOUT_ERRNOS = frozenset({11, errno.EAGAIN, errno.EWOULDBLOCK})

GPU_ADMISSION_PROOF = (
    "actual_b1_s2048_forward_backward_fused_adamw_step_plus_"
    "optimizer_resident_b1_s4096_labeled_eval"
)
GPU_ADMISSION_GEOMETRY = {
    "microbatch": MICROBATCH,
    "gradient_accumulation_configured": GRAD_ACCUM,
    "optimizer_probe_microsteps_executed": 1,
    "configured_effective_batch": MICROBATCH * GRAD_ACCUM,
    "sequence_length": SEQUENCE_LENGTH,
    "packing": False,
    "gradient_checkpointing": True,
}
EXPECTED_LORA_TARGETS = frozenset(
    name
    for layer in range(24)
    for name in (
        f"transformer.h.{layer}.self_attention.query_key_value",
        f"transformer.h.{layer}.self_attention.dense",
        f"transformer.h.{layer}.mlp.dense_h_to_4h",
        f"transformer.h.{layer}.mlp.dense_4h_to_h",
    )
)
EXPERIMENT_PATH = "experiments/20260831-bloomz-memory-fullft-v1"

RUNTIME_SOURCE_FILES = (
    "forge/__init__.py",
    "forge/model.py",
    "forge/baseline.py",
    "forge/clock.py",
    "forge/telemetry.py",
    "forge/data/__init__.py",
    "forge/data/loader.py",
    "forge/data/prompts.py",
    "forge/data/schema.py",
    "forge/data/tokenize.py",
    "forge/tasks/__init__.py",
    "forge/tasks/common.py",
    "forge/tuning/__init__.py",
    "forge/tuning/bloomz.py",
    "forge/tuning/callbacks.py",
    "forge/tuning/memory.py",
    "forge/tuning/plan.py",
    "pyproject.toml",
    "ops/docker/standalone-text-trainer.dockerfile",
    f"{EXPERIMENT_PATH}/build_fixture.py",
    f"{EXPERIMENT_PATH}/gpu_memory_probe.py",
    f"{EXPERIMENT_PATH}/inspect_runtime.py",
    f"{EXPERIMENT_PATH}/run_training.py",
    f"{EXPERIMENT_PATH}/score_external.py",
    f"{EXPERIMENT_PATH}/decide_external.py",
    f"{EXPERIMENT_PATH}/validate_artifact.py",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class BloomzExperimentError(RuntimeError):
    """The explicit experiment no longer matches its frozen authority."""


@dataclass(frozen=True)
class BloomzRequest:
    arm: str
    phase: str
    learning_rate: float
    max_steps: int
    manifest_path: Path
    train_path: Path
    dev_path: Path
    runtime_authority_path: Path
    runtime_authority_sha256: str
    runtime_authority: Mapping[str, Any]
    gpu_admission_path: Path
    gpu_admission_sha256: str
    gpu_admission: Mapping[str, Any]


@dataclass(frozen=True)
class BloomCheckpoint:
    path: Path
    eval_loss: float
    step: int
    ordinal: int


@dataclass
class BloomCheckpointPool:
    root: Path
    limit: int = MAX_RETAINED_CHECKPOINTS
    eval_count: int = 0
    enabled: bool = True
    entries: list[BloomCheckpoint] = field(default_factory=list)
    capture_errors: list[str] = field(default_factory=list)
    visible_export_errors: list[str] = field(default_factory=list)
    seen_eval_steps: set[int] = field(default_factory=set)
    best_loss: float | None = None
    best_step: int | None = None

    def sorted_entries(self) -> list[BloomCheckpoint]:
        return sorted(self.entries, key=lambda item: (item.eval_loss, item.step, item.ordinal))


@dataclass
class BloomEvalCounter:
    completed: int = 0


def request_from_environment() -> BloomzRequest | None:
    """Parse the route from a mount that physically contains only train/dev."""
    arm = os.environ.get("FORGE_BLOOMZ_EXPERIMENT_ARM")
    if arm is None:
        return None
    if arm not in {"control", "full"}:
        raise BloomzExperimentError("FORGE_BLOOMZ_EXPERIMENT_ARM must be control or full")
    manifest_raw = os.environ.get("FORGE_BLOOMZ_TRAINING_MANIFEST")
    if not manifest_raw:
        raise BloomzExperimentError("FORGE_BLOOMZ_TRAINING_MANIFEST is required")
    manifest_path = Path(manifest_raw).expanduser().resolve()
    manifest = _read_training_manifest(manifest_path)
    _verify_manifest_identities(manifest)
    _verify_training_fixture_directory(manifest_path, manifest)
    train_path = _verified_split_path(manifest_path, manifest, "train")
    dev_path = _verified_split_path(manifest_path, manifest, "dev")
    if train_path == dev_path:
        raise BloomzExperimentError("train and dev fixture paths must differ")

    if arm == "full":
        if os.environ.get("FORGE_ENABLE_EXPERIMENTAL_FULL_FT") != "1":
            raise BloomzExperimentError(
                "full arm also requires FORGE_ENABLE_EXPERIMENTAL_FULL_FT=1"
            )
        lr = _parse_learning_rate(os.environ.get("FORGE_BLOOMZ_LR"))
        if lr not in ALLOWED_FULL_LRS:
            raise BloomzExperimentError("full-arm LR is frozen at 1e-4")
        phase = os.environ.get("FORGE_BLOOMZ_PHASE")
        if phase != "candidate":
            raise BloomzExperimentError(
                "full arm requires FORGE_BLOOMZ_PHASE=candidate"
            )
    else:
        supplied = os.environ.get("FORGE_BLOOMZ_LR")
        lr = CONTROL_LR if supplied in (None, "") else _parse_learning_rate(supplied)
        if lr != CONTROL_LR:
            raise BloomzExperimentError("control LR is frozen at 1.5e-4")
        phase = os.environ.get("FORGE_BLOOMZ_PHASE", "control")
        if phase != "control":
            raise BloomzExperimentError("control arm phase must be control")

    max_steps = _optional_positive_int(os.environ.get("FORGE_BLOOMZ_MAX_STEPS"))
    if max_steps != MATCHED_DECISION_STEPS:
        raise BloomzExperimentError(
            f"{phase} requires FORGE_BLOOMZ_MAX_STEPS={MATCHED_DECISION_STEPS}"
        )
    runtime_raw = os.environ.get("FORGE_BLOOMZ_RUNTIME_AUTHORITY")
    if not runtime_raw:
        raise BloomzExperimentError("FORGE_BLOOMZ_RUNTIME_AUTHORITY is required")
    runtime_path, runtime_authority, runtime_sha = load_runtime_authority(runtime_raw)
    admission_raw = os.environ.get("FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT")
    if not admission_raw:
        raise BloomzExperimentError(
            "FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT is required"
        )
    admission_path, admission, admission_sha = load_gpu_admission_receipt(
        admission_raw,
        arm=arm,
        learning_rate=lr,
        runtime_authority_path=runtime_path,
        runtime_authority=runtime_authority,
        runtime_authority_sha256=runtime_sha,
    )
    return BloomzRequest(
        arm=arm,
        phase=phase,
        learning_rate=lr,
        max_steps=max_steps,
        manifest_path=manifest_path,
        train_path=train_path,
        dev_path=dev_path,
        runtime_authority_path=runtime_path,
        runtime_authority_sha256=runtime_sha,
        runtime_authority=runtime_authority,
        gpu_admission_path=admission_path,
        gpu_admission_sha256=admission_sha,
        gpu_admission=admission,
    )


def validate_model_identity(model: Any, model_dir: str | os.PathLike[str]) -> None:
    """Bind the route to the immutable public BloomZ files and raw config."""
    root = Path(model_dir).resolve()
    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    special_tokens_path = root / "special_tokens_map.json"
    weights_path = root / "model.safetensors"
    for path, expected, expected_bytes in (
        (config_path, MODEL_CONFIG_SHA256, None),
        (tokenizer_path, MODEL_TOKENIZER_SHA256, None),
        (
            tokenizer_config_path,
            MODEL_TOKENIZER_CONFIG_SHA256,
            MODEL_TOKENIZER_CONFIG_BYTES,
        ),
        (
            special_tokens_path,
            MODEL_SPECIAL_TOKENS_MAP_SHA256,
            MODEL_SPECIAL_TOKENS_MAP_BYTES,
        ),
        (weights_path, MODEL_WEIGHTS_SHA256, None),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or (expected_bytes is not None and path.stat().st_size != expected_bytes)
            or _sha256(path) != expected
        ):
            raise BloomzExperimentError(f"immutable model file mismatch: {path.name}")
    for name in MODEL_FORBIDDEN_TOKENIZER_FILES:
        path = root / name
        if path.exists() or path.is_symlink():
            raise BloomzExperimentError(
                f"immutable tokenizer unexpectedly contains {name}"
            )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BloomzExperimentError(f"invalid pinned config: {exc}") from exc
    expected = {
        "architectures": ["BloomForCausalLM"],
        "model_type": "bloom",
        "n_embed": 1024,
        "n_layer": 24,
        "num_attention_heads": 16,
        "seq_length": 2048,
        "vocab_size": 250880,
        "unk_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "use_cache": True,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise BloomzExperimentError("BloomZ architecture/config identity drift")
    if "max_position_embeddings" in raw:
        raise BloomzExperimentError("pinned Bloom config must not gain max_position_embeddings")
    config = getattr(model, "config", None)
    if (
        getattr(config, "model_type", None) != "bloom"
        or tuple(getattr(config, "architectures", ()) or ()) != ("BloomForCausalLM",)
        or getattr(config, "vocab_size", None) != 250880
    ):
        raise BloomzExperimentError("loaded model is not exact BloomForCausalLM")


def validate_task_contract(spec: Any) -> None:
    """Require the exact standardized public-fixture instruct schema."""
    cols = getattr(spec, "instruct", None)
    if (
        getattr(spec, "task_type", None) != "InstructTextTask"
        or bool(getattr(spec, "use_kl", False))
        or getattr(spec, "model", None) != MODEL_REPO
        or cols is None
        or cols.instruction != "instruct"
        or cols.output != "output"
        or cols.system != "system"
        or cols.input is not None
        or cols.system_prompt != ""
        or cols.fmt is not None
        or cols.no_input_fmt not in (None, "{instruction}")
        or cols.system_format != "{system}"
    ):
        raise BloomzExperimentError("task does not match frozen system/instruct/output contract")


def require_single_h100(
    *,
    n_gpus: int,
    per_gpu_gb: float,
    gpu_admission: Mapping[str, Any] | None = None,
) -> None:
    """Fail unless training is on the exact H100 described by its proof."""
    card_bytes = 0
    try:
        import torch

        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0))
            card_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        else:
            name = ""
    except Exception:
        name = ""
    if n_gpus != 1 or per_gpu_gb < 70.0 or "H100" not in name.upper():
        raise BloomzExperimentError(
            f"BloomZ experiment requires one >=70 GB H100; got {n_gpus} GPU, "
            f"{per_gpu_gb:.1f} GB, device={name!r}"
        )
    if gpu_admission is not None:
        admitted_gpu = gpu_admission.get("gpu")
        if (
            not isinstance(admitted_gpu, Mapping)
            or admitted_gpu.get("name") != name
            or admitted_gpu.get("card_bytes") != card_bytes
        ):
            raise BloomzExperimentError(
                "live training GPU differs from measured H100 admission receipt"
            )


def apply_plan(plan: TrainPlan, request: BloomzRequest) -> TrainPlan:
    """Freeze identical, unpacked memory geometry for both experiment arms."""
    expected_strategy = "full" if request.arm == "full" else "lora"
    if plan.strategy != expected_strategy:
        raise BloomzExperimentError(
            f"plan strategy {plan.strategy!r} does not match arm {request.arm!r}"
        )
    return replace(
        plan,
        learning_rate=request.learning_rate,
        per_device_batch_size=MICROBATCH,
        grad_accum_steps=GRAD_ACCUM,
        max_seq_len=SEQUENCE_LENGTH,
        gradient_checkpointing=True,
    )


def validate_tokenization_counts(*, train_count: int, dev_count: int) -> None:
    """Reopen the frozen post-tokenization row-count attestation."""
    if (
        train_count != EXPECTED_TRAIN_TOKENIZED_ROWS
        or EXPECTED_TRAIN_ROWS - train_count != EXPECTED_TRAIN_DROPPED_ROWS
    ):
        raise BloomzExperimentError("frozen train tokenization count drift")
    if dev_count != EXPECTED_DEV_TOKENIZED_ROWS:
        raise BloomzExperimentError("frozen dev tokenization count drift")


def run_matched_training(spec: Any, deadline: Any) -> Path:
    """Run the experiment-local matched LoRA/full training protocol.

    This deliberately does not route through the production task handler. Both
    arms consume the same frozen row order and tokenizer path, use the same
    geometry and schedule, and publish only four bounded nomination artifacts.
    """
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    from forge import telemetry
    from forge.data import loader, prompts, tokenize
    from forge.model import (
        attach_lora,
        gpu_topology,
        load_base,
        median_weight_rms,
        model_param_billions,
        prepare_full_finetune,
    )
    from forge.tasks.common import (
        build_training_kwargs,
        compatible_dataclass_kwargs,
        safe_train,
        workdir,
    )
    from forge.tuning.callbacks import DeadlineCallback
    from forge.tuning.memory import infer_output_width, require_sft_admission
    from forge.tuning.plan import make_sft_plan

    request = request_from_environment()
    if request is None:
        raise BloomzExperimentError("explicit BloomZ arm environment is absent")
    validate_task_contract(spec)
    train_rows = loader.load_rows(
        str(request.train_path),
        dataset_arg=str(request.train_path),
        file_format="json",
    )
    dev_rows = loader.load_rows(
        str(request.dev_path),
        dataset_arg=str(request.dev_path),
        file_format="json",
    )
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(dev_rows) != EXPECTED_DEV_ROWS:
        raise BloomzExperimentError("frozen fixture row count drift")

    loaded = load_base(spec.cached_model_dir, for_generation=False)
    tokenizer = loaded.tokenizer
    validate_model_identity(loaded.model, loaded.model_dir)
    params_b = model_param_billions(loaded.model)
    if not math.isclose(params_b, MODEL_PARAMS_B, rel_tol=0.0, abs_tol=1e-9):
        raise BloomzExperimentError("loaded BloomZ parameter count drift")
    n_gpus, per_gpu_gb = gpu_topology()
    require_single_h100(
        n_gpus=n_gpus,
        per_gpu_gb=per_gpu_gb,
        gpu_admission=request.gpu_admission,
    )
    strategy = "full" if request.arm == "full" else "lora"
    plan = make_sft_plan(
        use_kl=False,
        strategy=strategy,
        params_b=params_b,
        weight_rms=(median_weight_rms(loaded.model) if strategy == "full" else None),
        n_gpus=n_gpus,
        per_gpu_gb=per_gpu_gb,
    )
    plan = apply_plan(plan, request)
    output_width = infer_output_width(loaded.model, tokenizer)
    if output_width != 250_880:
        raise BloomzExperimentError("BloomZ output width drift")
    for purpose, sequence_length in (
        ("train", SEQUENCE_LENGTH),
        ("selection", SELECTION_SEQUENCE_LENGTH),
    ):
        receipt = require_sft_admission(
            params_b=params_b,
            vocab_size=output_width,
            sequence_length=sequence_length,
            microbatch=MICROBATCH,
            strategy=strategy,
            gradient_checkpointing=True,
            card_gb=per_gpu_gb,
        )
        telemetry.event(
            "bloomz_memory_admission", purpose=purpose, **receipt.telemetry_fields()
        )

    model = (
        prepare_full_finetune(loaded.model, gradient_checkpointing=True)
        if strategy == "full"
        else attach_lora(
            loaded.model,
            r=plan.lora_r,
            alpha=plan.lora_alpha,
            dropout=plan.lora_dropout,
        )
    )
    columns = spec.instruct
    if columns is None:
        raise BloomzExperimentError("frozen instruct columns are absent")
    train_examples = prompts.build_instruct_examples(train_rows, columns)
    dev_examples = prompts.build_instruct_examples(dev_rows, columns)
    train_tokens = tokenize.tokenize_instruct(
        train_examples, tokenizer, SEQUENCE_LENGTH
    )
    dev_tokens = tokenize.tokenize_instruct(
        dev_examples, tokenizer, SELECTION_SEQUENCE_LENGTH
    )
    validate_tokenization_counts(
        train_count=len(train_tokens),
        dev_count=len(dev_tokens),
    )

    kwargs = build_training_kwargs(spec, plan, neftune_alpha=5.0)
    kwargs.pop("num_train_epochs", None)
    kwargs.update(
        max_steps=MATCHED_DECISION_STEPS,
        eval_strategy="steps",
        eval_steps=MATCHED_DECISION_STEPS // MIN_COMPLETED_EVALS,
        per_device_eval_batch_size=MICROBATCH,
        save_strategy="no",
        report_to=[],
    )
    training_args = TrainingArguments(
        **compatible_dataclass_kwargs(
            TrainingArguments,
            kwargs,
            allow_removed={"overwrite_output_dir"},
        )
    )
    checkpoint_root = Path(workdir(spec)) / "bloomz-decision-checkpoints"
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise BloomzExperimentError("decision-checkpoint directory is not empty")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pool = BloomCheckpointPool(root=checkpoint_root)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_tokens),
        eval_dataset=Dataset.from_list(dev_tokens),
        data_collator=tokenize.PadCollator(tokenizer.pad_token_id),
        callbacks=[
            DeadlineCallback(deadline),
            make_checkpoint_callback(
                pool,
                tokenizer=tokenizer,
                strategy=strategy,
                metadata_source_dir=loaded.model_dir,
                visible_output_dir=None,
            ),
        ],
    )
    safe_train(trainer)
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    planned_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    inventory_path = Path(workdir(spec)) / "bloomz-checkpoint-inventory.json"
    eligible = write_checkpoint_inventory(
        pool,
        inventory_path,
        strategy=strategy,
        phase=request.phase,
        schedule_completed=(
            final_step == planned_steps == MATCHED_DECISION_STEPS
        ),
        planned_steps=planned_steps,
        final_step=final_step,
        runtime_authority_path=request.runtime_authority_path,
        runtime_authority_sha256=request.runtime_authority_sha256,
        runtime_authority=request.runtime_authority,
        gpu_admission_path=request.gpu_admission_path,
        gpu_admission_sha256=request.gpu_admission_sha256,
        gpu_admission=request.gpu_admission,
    )
    if not eligible:
        raise BloomzExperimentError(
            f"STOP_NO_SCIENCE: incomplete matched run; receipt={inventory_path}"
        )
    return inventory_path


def experiment_config() -> dict[str, Any]:
    """Return the immutable scientific/runtime contract bound into every receipt."""
    return {
        "schema_version": "sn56.bloomz-experiment-config.v1",
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "tokenizer_sha256": MODEL_TOKENIZER_SHA256,
            "tokenizer_config": {
                "bytes": MODEL_TOKENIZER_CONFIG_BYTES,
                "sha256": MODEL_TOKENIZER_CONFIG_SHA256,
            },
            "special_tokens_map": {
                "bytes": MODEL_SPECIAL_TOKENS_MAP_BYTES,
                "sha256": MODEL_SPECIAL_TOKENS_MAP_SHA256,
            },
            "forbidden_tokenizer_files": sorted(
                MODEL_FORBIDDEN_TOKENIZER_FILES
            ),
        },
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "parquet_sha256": DATASET_PARQUET_SHA256,
            "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        },
        "geometry": {
            "microbatch": MICROBATCH,
            "gradient_accumulation": GRAD_ACCUM,
            "effective_batch": MICROBATCH * GRAD_ACCUM,
            "train_sequence_length": SEQUENCE_LENGTH,
            "selection_sequence_length": SELECTION_SEQUENCE_LENGTH,
            "packing": False,
            "gradient_checkpointing": True,
        },
        "arms": {
            "control": {"strategy": "lora", "learning_rate": CONTROL_LR},
            "full": {
                "strategy": "full",
                "learning_rate": FULL_LR,
            },
        },
        "decision": {
            "trainer_loss_role": "nomination_only",
            "external_scores_per_arm": MIN_COMPLETED_EVALS,
            "max_retained_checkpoints": MAX_RETAINED_CHECKPOINTS,
            "matched_control_candidate_steps": MATCHED_DECISION_STEPS,
            "owner_paired_gate": {
                "seed": 20_260_808,
                "bootstrap_resamples": 10_000,
                "confidence": 0.99,
                "one_sided_tail": 0.01,
                "candidate_win_rate_lower_bound": 0.55,
                "mean_gap_lower_bound_nat_floor": 0.01,
                "mean_gap_lower_bound_control_fraction": 0.01,
                "mean_gap_direction": "control_minus_candidate",
            },
            "evidence_label": (
                "composite_train_s2048_external_selection_s4096_"
                "trainer_loss_nomination_only"
            ),
        },
        "runtime": {
            "training_image": TRAINING_IMAGE,
            "evaluator_image": EVALUATOR_IMAGE,
            "score_driver_sha256": SCORE_DRIVER_SHA256,
            "lease_budget": lease_budget(),
        },
    }


def experiment_config_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(experiment_config())).hexdigest()


def lease_budget() -> dict[str, Any]:
    """Return the exact single-lease envelope shared by every GPU stage."""
    stages = {
        name: {
            **facts,
            "total_seconds": facts["count"] * facts["max_each_seconds"],
        }
        for name, facts in LEASE_STAGE_MAXIMA.items()
    }
    stage_seconds = sum(item["total_seconds"] for item in stages.values())
    science_reserve = LEASE_SCIENCE_WINDOW_SECONDS - stage_seconds
    if (
        science_reserve < 0
        or LEASE_BOOTSTRAP_ALLOWANCE_SECONDS
        + LEASE_SCIENCE_WINDOW_SECONDS
        + LEASE_CLOSURE_RESERVE_SECONDS
        != LEASE_TOTAL_SECONDS
    ):
        raise BloomzExperimentError("lease-stage arithmetic exceeds provider cap")
    return {
        "schema_version": "sn56.bloomz-lease-budget.v2",
        "total_seconds": LEASE_TOTAL_SECONDS,
        "hourly_rate_usd": LEASE_HOURLY_RATE,
        "maximum_cost_usd": LEASE_MAX_COST,
        "stages": stages,
        "stage_seconds": stage_seconds,
        "bootstrap_start_allowance_seconds": LEASE_BOOTSTRAP_ALLOWANCE_SECONDS,
        "science_window_seconds": LEASE_SCIENCE_WINDOW_SECONDS,
        "decision_reserve_seconds": science_reserve,
        "ceo_custody_close_reserve_seconds": LEASE_CLOSURE_RESERVE_SECONDS,
    }


def classify_outbound_connect_result(
    *,
    family: str,
    connect_ex: int,
    connected: bool,
    elapsed_seconds: float,
    timeout_seconds: float,
) -> bool:
    """Classify one exact TCP probe without treating a connection as denial."""
    if family not in {"IPv4", "IPv6"}:
        raise BloomzExperimentError("network probe family is invalid")
    if (
        not isinstance(connect_ex, int)
        or isinstance(connect_ex, bool)
        or connect_ex < 0
    ):
        raise BloomzExperimentError("network probe connect_ex is invalid")
    if not isinstance(connected, bool) or connected != (connect_ex == 0):
        raise BloomzExperimentError("network probe connected state is inconsistent")
    for value, label in (
        (elapsed_seconds, "elapsed seconds"),
        (timeout_seconds, "timeout seconds"),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise BloomzExperimentError(f"network probe {label} is invalid")
    if timeout_seconds <= 0:
        raise BloomzExperimentError("network probe timeout seconds must be positive")
    if connected:
        return False
    if connect_ex in NETWORK_DEFAULT_DENY_ERRNOS:
        return True
    return (
        family == "IPv4"
        and connect_ex in IPV4_TIMEOUT_ERRNOS
        and elapsed_seconds >= timeout_seconds
    )


def lease_authority(
    provider_start_epoch: Any,
    science_started_epoch: Any,
) -> dict[str, Any]:
    """Bind actual science start and immutable decision/provider deadlines."""
    if not isinstance(provider_start_epoch, int) or isinstance(provider_start_epoch, bool):
        raise BloomzExperimentError("provider start must be an integer epoch")
    if provider_start_epoch <= 0:
        raise BloomzExperimentError("provider start must be positive")
    if not isinstance(science_started_epoch, int) or isinstance(
        science_started_epoch, bool
    ):
        raise BloomzExperimentError("science start must be an integer epoch")
    science_start_deadline = provider_start_epoch + LEASE_BOOTSTRAP_ALLOWANCE_SECONDS
    if not provider_start_epoch <= science_started_epoch <= science_start_deadline:
        raise BloomzExperimentError("science start exceeds bootstrap allowance")
    budget = lease_budget()
    decision_deadline = science_started_epoch + LEASE_SCIENCE_WINDOW_SECONDS
    provider_deadline = provider_start_epoch + LEASE_TOTAL_SECONDS
    if decision_deadline + LEASE_CLOSURE_RESERVE_SECONDS > provider_deadline:
        raise BloomzExperimentError("lease arithmetic leaves insufficient CEO reserve")
    return {
        "schema_version": "sn56.bloomz-lease-authority.v2",
        "provider_start_epoch": provider_start_epoch,
        "science_start_deadline_epoch": science_start_deadline,
        "science_started_epoch": science_started_epoch,
        "decision_deadline_epoch": decision_deadline,
        "provider_deadline_epoch": provider_deadline,
        "budget": budget,
        "budget_sha256": hashlib.sha256(_canonical_bytes(budget)).hexdigest(),
    }


def require_science_stage(
    lease: Mapping[str, Any],
    *,
    stage_max_seconds: int,
    remaining_planned_seconds: int,
    now_epoch: float | None = None,
    claimed_decision_deadline_epoch: int | None = None,
) -> dict[str, Any]:
    """Fail before a stage if authority or remaining science time has drifted."""
    expected = lease_authority(
        lease.get("provider_start_epoch"), lease.get("science_started_epoch")
    )
    if dict(lease) != expected:
        raise BloomzExperimentError("runtime lease authority drift")
    cutoff = expected["decision_deadline_epoch"]
    if (
        claimed_decision_deadline_epoch is not None
        and claimed_decision_deadline_epoch != cutoff
    ):
        raise BloomzExperimentError("shell decision deadline differs from runtime authority")
    if (
        not isinstance(stage_max_seconds, int)
        or isinstance(stage_max_seconds, bool)
        or stage_max_seconds <= 0
        or not isinstance(remaining_planned_seconds, int)
        or isinstance(remaining_planned_seconds, bool)
        or remaining_planned_seconds < stage_max_seconds
    ):
        raise BloomzExperimentError("invalid science-stage budget")
    now = time.time() if now_epoch is None else float(now_epoch)
    if (
        not math.isfinite(now)
        or now < expected["science_started_epoch"]
        or math.ceil(now) + remaining_planned_seconds > cutoff
    ):
        raise BloomzExperimentError("science stage is outside the authority cutoff")
    return expected


def runtime_source_inventory(
    root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Hash the runtime-critical worktree files mounted into the trainer."""
    repository = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    inventory: list[dict[str, Any]] = []
    for relative in RUNTIME_SOURCE_FILES:
        path = repository / relative
        if not path.is_file() or path.is_symlink():
            raise BloomzExperimentError(f"runtime source file is absent or unsafe: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def runtime_source_inventory_sha256(
    root: str | os.PathLike[str] | None = None,
) -> str:
    return hashlib.sha256(_canonical_bytes(runtime_source_inventory(root))).hexdigest()


def source_child_inventory(
    root: str | os.PathLike[str],
    prefix: str,
    tracked: Iterable[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Bind exact source bytes while allowing ignored interpreter caches.

    Production callers pass the complete ``git ls-files -s`` path/mode set from
    a clean checkout.  The optional discovery form exists for isolated tests;
    it deliberately ignores only Python bytecode/cache directories.
    """
    repository = Path(root).expanduser().resolve(strict=True)
    child = repository / prefix
    if child.is_symlink() or not child.is_dir():
        raise BloomzExperimentError(f"source child is absent or unsafe: {prefix}")
    if tracked is None:
        discovered: list[tuple[str, str]] = []
        for path in child.rglob("*"):
            relative = path.relative_to(repository).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_file() and not path.is_symlink():
                mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
                discovered.append((relative, mode))
        tracked = discovered
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    normalized_prefix = prefix.rstrip("/") + "/"
    for relative, mode in sorted(tracked):
        if (
            relative in seen
            or not relative.startswith(normalized_prefix)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or mode not in {"100644", "100755"}
        ):
            raise BloomzExperimentError("invalid tracked source inventory entry")
        seen.add(relative)
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise BloomzExperimentError(f"tracked source file is absent or unsafe: {relative}")
        actual_mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        if actual_mode != mode:
            raise BloomzExperimentError(f"tracked source mode drift: {relative}")
        content = path.read_bytes()
        blob_header = b"blob " + str(len(content)).encode("ascii") + b"\0"
        inventory.append(
            {
                "path": relative,
                "mode": mode,
                "git_blob_sha1": hashlib.sha1(blob_header + content).hexdigest(),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    if not inventory:
        raise BloomzExperimentError(f"tracked source inventory is empty: {prefix}")
    return inventory


def verify_source_child_inventory(
    root: str | os.PathLike[str], prefix: str, recorded: Any
) -> None:
    """Verify every recorded tracked file, without treating ignored pyc as source."""
    if not isinstance(recorded, list) or not recorded:
        raise BloomzExperimentError(f"missing tracked source inventory: {prefix}")
    pairs: list[tuple[str, str]] = []
    for item in recorded:
        if not isinstance(item, Mapping) or set(item) != {
            "path", "mode", "git_blob_sha1", "bytes", "sha256"
        }:
            raise BloomzExperimentError(f"tracked source inventory shape drift: {prefix}")
        pairs.append((str(item["path"]), str(item["mode"])))
    live = source_child_inventory(root, prefix, pairs)
    if live != recorded:
        raise BloomzExperimentError(f"tracked source bytes drift: {prefix}")


def load_runtime_authority(
    raw_path: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Verify the CPU-generated git/image authority against live source bytes."""
    path = Path(raw_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
        raise BloomzExperimentError("runtime authority is absent, unsafe, or too large")
    try:
        encoded = path.read_bytes()
        if not 0 < len(encoded) <= 2_000_000:
            raise BloomzExperimentError("GPU admission receipt is absent or unsafe")
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BloomzExperimentError(f"invalid runtime authority: {exc}") from exc
    if not isinstance(payload, dict):
        raise BloomzExperimentError("runtime authority is not an object")
    source = payload.get("source")
    image = payload.get("training_image")
    lease = payload.get("lease")
    if (
        payload.get("schema_version") != "sn56.bloomz-runtime-authority.v2"
        or payload.get("status") != "PASS"
        or payload.get("experiment_config_sha256") != experiment_config_sha256()
        or not isinstance(source, Mapping)
        or not isinstance(image, Mapping)
        or not isinstance(lease, Mapping)
    ):
        raise BloomzExperimentError("runtime authority contract drift")
    expected_lease = lease_authority(
        lease.get("provider_start_epoch"), lease.get("science_started_epoch")
    )
    if dict(lease) != expected_lease:
        raise BloomzExperimentError("runtime lease authority drift")
    repository = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if (
        source.get("clean") is not True
        or not isinstance(source.get("parent"), str)
        or _HEX40.fullmatch(str(source.get("parent"))) is None
        or not isinstance(source.get("commit"), str)
        or _HEX40.fullmatch(str(source.get("commit"))) is None
        or not isinstance(source.get("tree"), str)
        or _HEX40.fullmatch(str(source.get("tree"))) is None
        or not isinstance(source.get("forge_child_tree"), str)
        or _HEX40.fullmatch(str(source.get("forge_child_tree"))) is None
        or not isinstance(source.get("experiment_child_tree"), str)
        or _HEX40.fullmatch(str(source.get("experiment_child_tree"))) is None
        or source.get("runtime_source_inventory_sha256")
        != runtime_source_inventory_sha256(repository)
    ):
        raise BloomzExperimentError("runtime source authority drift")
    verify_source_child_inventory(repository, "forge", source.get("forge_inventory"))
    verify_source_child_inventory(
        repository, EXPERIMENT_PATH, source.get("experiment_inventory")
    )
    if (
        image.get("reference") != TRAINING_IMAGE
        or not isinstance(image.get("image_id"), str)
        or _SHA256_ID.fullmatch(str(image.get("image_id"))) is None
        or image.get("os") != "linux"
        or image.get("architecture") != "amd64"
        or not isinstance(image.get("repo_digests"), list)
        or TRAINING_IMAGE not in image.get("repo_digests", [])
    ):
        raise BloomzExperimentError("inspected training-image authority drift")
    return path, payload, hashlib.sha256(encoded).hexdigest()


def authority_fields(payload: Mapping[str, Any], receipt_sha256: str) -> dict[str, Any]:
    """Compact receipt-chain identity after :func:`load_runtime_authority`."""
    source = payload["source"]
    image = payload["training_image"]
    return {
        "runtime_authority_sha256": receipt_sha256,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "forge_child_tree": source["forge_child_tree"],
        "experiment_child_tree": source["experiment_child_tree"],
        "runtime_source_inventory_sha256": source[
            "runtime_source_inventory_sha256"
        ],
        "training_image_reference": image["reference"],
        "training_image_id": image["image_id"],
        "experiment_config_sha256": experiment_config_sha256(),
        "provider_start_epoch": payload["lease"]["provider_start_epoch"],
        "science_start_deadline_epoch": payload["lease"][
            "science_start_deadline_epoch"
        ],
        "science_started_epoch": payload["lease"]["science_started_epoch"],
        "decision_deadline_epoch": payload["lease"]["decision_deadline_epoch"],
        "provider_deadline_epoch": payload["lease"]["provider_deadline_epoch"],
        "lease_budget_sha256": payload["lease"]["budget_sha256"],
    }


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise BloomzExperimentError(f"GPU admission has invalid {label}")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BloomzExperimentError(f"GPU admission has invalid {label}")
    return value


def load_gpu_admission_receipt(
    raw_path: str | os.PathLike[str],
    *,
    arm: str,
    learning_rate: float,
    runtime_authority_path: str | os.PathLike[str],
    runtime_authority: Mapping[str, Any],
    runtime_authority_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    """Verify one measured H100 proof for the exact requested training arm."""
    expanded = Path(raw_path).expanduser()
    if expanded.is_symlink():
        raise BloomzExperimentError("GPU admission receipt must not be a symlink")
    try:
        path = expanded.resolve(strict=True)
    except OSError as exc:
        raise BloomzExperimentError("GPU admission receipt is absent or unsafe") from exc
    if not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
        raise BloomzExperimentError("GPU admission receipt is absent or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BloomzExperimentError(f"invalid GPU admission receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise BloomzExperimentError("GPU admission receipt is not an object")

    expected_keys = {
        "schema_version",
        "status",
        "proof",
        "analytic_estimate_role",
        "arm",
        "strategy",
        "learning_rate",
        "geometry",
        "model",
        "gpu",
        "loss",
        "selection_loss",
        "optimizer_state_tensor_count",
        "optimizer_state_bytes",
        "analytic_estimate",
        "runtime_authority_path",
        "authority",
        "runtime",
    }
    expected_learning_rate = FULL_LR if arm == "full" else CONTROL_LR
    if arm not in {"control", "full"} or learning_rate != expected_learning_rate:
        raise BloomzExperimentError("GPU admission requested learning rate drift")
    strategy = "full" if arm == "full" else "lora"
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("proof") != GPU_ADMISSION_PROOF
        or payload.get("analytic_estimate_role")
        != "admission_support_only_not_hardware_proof"
        or payload.get("arm") != arm
        or payload.get("strategy") != strategy
        or _finite_number(payload.get("learning_rate"), "learning rate")
        != learning_rate
        or payload.get("geometry") != GPU_ADMISSION_GEOMETRY
    ):
        raise BloomzExperimentError("GPU admission experiment contract drift")

    model = payload.get("model")
    if not isinstance(model, Mapping) or set(model) != {
        "repo",
        "revision",
        "config_sha256",
        "weights_sha256",
        "output_width",
        "params_b",
    }:
        raise BloomzExperimentError("GPU admission model identity is absent")
    if (
        model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or model.get("config_sha256") != MODEL_CONFIG_SHA256
        or model.get("weights_sha256") != MODEL_WEIGHTS_SHA256
        or model.get("output_width") != 250_880
        or not math.isclose(
            _finite_number(model.get("params_b"), "model parameter count"),
            MODEL_PARAMS_B,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise BloomzExperimentError("GPU admission model identity drift")

    gpu = payload.get("gpu")
    expected_gpu_keys = {
        "name",
        "card_bytes",
        "train_peak_allocated_bytes",
        "train_peak_reserved_bytes",
        "train_peak_reserved_ratio",
        "selection_peak_allocated_bytes",
        "selection_peak_reserved_bytes",
        "selection_peak_reserved_ratio",
        "free_before_bytes",
        "train_free_after_bytes",
        "free_after_bytes",
        "max_reserved_ratio",
    }
    if not isinstance(gpu, Mapping) or set(gpu) != expected_gpu_keys:
        raise BloomzExperimentError("GPU admission hardware identity is absent")
    name = gpu.get("name")
    if not isinstance(name, str) or "H100" not in name.upper():
        raise BloomzExperimentError("GPU admission was not measured on an H100")
    card_bytes = _positive_int(gpu.get("card_bytes"), "card bytes")
    if card_bytes < 70_000_000_000:
        raise BloomzExperimentError("GPU admission H100 has less than 70 GB")
    train_allocated = _positive_int(
        gpu.get("train_peak_allocated_bytes"), "train peak allocated bytes"
    )
    train_reserved = _positive_int(
        gpu.get("train_peak_reserved_bytes"), "train peak reserved bytes"
    )
    selection_allocated = _positive_int(
        gpu.get("selection_peak_allocated_bytes"),
        "selection peak allocated bytes",
    )
    selection_reserved = _positive_int(
        gpu.get("selection_peak_reserved_bytes"),
        "selection peak reserved bytes",
    )
    if not (
        train_allocated <= train_reserved <= card_bytes
        and selection_allocated <= selection_reserved <= card_bytes
    ):
        raise BloomzExperimentError("GPU admission peak byte accounting drift")
    for key in ("free_before_bytes", "train_free_after_bytes", "free_after_bytes"):
        value = _positive_int(gpu.get(key), key.replace("_", " "))
        if value > card_bytes:
            raise BloomzExperimentError("GPU admission free-memory accounting drift")
    train_ratio = _finite_number(
        gpu.get("train_peak_reserved_ratio"), "train peak reserved ratio"
    )
    selection_ratio = _finite_number(
        gpu.get("selection_peak_reserved_ratio"),
        "selection peak reserved ratio",
    )
    maximum_ratio = _finite_number(
        gpu.get("max_reserved_ratio"), "maximum reserved ratio"
    )
    if (
        not 0.50 <= maximum_ratio <= 0.80
        or max(train_ratio, selection_ratio) > maximum_ratio
        or not math.isclose(
            train_ratio,
            train_reserved / card_bytes,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            selection_ratio,
            selection_reserved / card_bytes,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise BloomzExperimentError("GPU admission reserved-ratio proof drift")

    for key in ("loss", "selection_loss"):
        if _finite_number(payload.get(key), key.replace("_", " ")) < 0:
            raise BloomzExperimentError(f"GPU admission has invalid {key}")
    _positive_int(
        payload.get("optimizer_state_tensor_count"),
        "optimizer state tensor count",
    )
    _positive_int(payload.get("optimizer_state_bytes"), "optimizer state bytes")

    analytic = payload.get("analytic_estimate")
    if not isinstance(analytic, Mapping) or set(analytic) != {"train", "selection"}:
        raise BloomzExperimentError("GPU admission analytic support is absent")
    for purpose, expected_sequence_length in (
        ("train", SEQUENCE_LENGTH),
        ("selection", SELECTION_SEQUENCE_LENGTH),
    ):
        estimate = analytic[purpose]
        if (
            not isinstance(estimate, Mapping)
            or estimate.get("memory_admitted") is not True
            or estimate.get("memory_strategy") != strategy
            or estimate.get("memory_vocab_size") != 250_880
            or estimate.get("memory_sequence_length") != expected_sequence_length
            or estimate.get("memory_microbatch") != MICROBATCH
            or estimate.get("memory_gradient_checkpointing") is not True
            or _finite_number(
                estimate.get("memory_params_b"),
                f"{purpose} analytic parameter count",
            )
            != round(MODEL_PARAMS_B, 4)
            or _finite_number(
                estimate.get("memory_card_gb"),
                f"{purpose} analytic card GB",
            )
            != round(card_bytes / 1_000_000_000.0, 4)
            or _finite_number(
                estimate.get("memory_budget_ratio"),
                f"{purpose} analytic budget ratio",
            )
            != 0.7
        ):
            raise BloomzExperimentError("GPU admission analytic support drift")

    expected_authority = authority_fields(
        runtime_authority, runtime_authority_sha256
    )
    if (
        payload.get("runtime_authority_path")
        != str(Path(runtime_authority_path).expanduser().resolve())
        or payload.get("authority") != expected_authority
    ):
        raise BloomzExperimentError("GPU admission runtime authority drift")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"python", "platform", "torch", "transformers", "peft"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise BloomzExperimentError("GPU admission runtime identity is absent")
    return path, payload, _sha256(path)


def gpu_admission_fields(
    payload: Mapping[str, Any],
    receipt_path: str | os.PathLike[str],
    receipt_sha256: str,
) -> dict[str, Any]:
    """Project verified measured-admission facts into training provenance."""
    gpu = payload["gpu"]
    return {
        "receipt_path": str(Path(receipt_path).expanduser().resolve()),
        "receipt_sha256": receipt_sha256,
        "status": payload["status"],
        "proof": payload["proof"],
        "arm": payload["arm"],
        "strategy": payload["strategy"],
        "learning_rate": payload["learning_rate"],
        "geometry": dict(payload["geometry"]),
        "model": dict(payload["model"]),
        "gpu": {
            "name": gpu["name"],
            "card_bytes": gpu["card_bytes"],
            "train_peak_reserved_bytes": gpu["train_peak_reserved_bytes"],
            "train_peak_reserved_ratio": gpu["train_peak_reserved_ratio"],
            "selection_peak_reserved_bytes": gpu[
                "selection_peak_reserved_bytes"
            ],
            "selection_peak_reserved_ratio": gpu[
                "selection_peak_reserved_ratio"
            ],
            "max_reserved_ratio": gpu["max_reserved_ratio"],
        },
        "authority": dict(payload["authority"]),
    }


def make_checkpoint_callback(
    pool: BloomCheckpointPool,
    *,
    tokenizer: Any,
    strategy: str,
    metadata_source_dir: str,
    visible_output_dir: str | None = None,
) -> Any:
    """Nominate at most four externally loadable decision artifacts.

    Trainer loss is used only to bound the candidate set.  It is not accepted as
    checkpoint evidence: the experiment package separately runs every retained
    artifact through the digest-pinned external scorer with raw vectors.
    """
    from transformers import TrainerCallback

    class BloomCheckpointCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            if not pool.enabled:
                return control
            loss = _finite_loss((metrics or {}).get("eval_loss"))
            if loss is None:
                return control
            step = int(state.global_step)
            if step in pool.seen_eval_steps:
                return control
            pool.seen_eval_steps.add(step)
            pool.eval_count += 1
            ordinal = pool.eval_count
            candidate = BloomCheckpoint(
                path=pool.root / f"step-{step:08d}-eval-{ordinal:03d}",
                eval_loss=loss,
                step=step,
                ordinal=ordinal,
            )
            if len(pool.entries) >= pool.limit:
                worst = max(
                    pool.entries,
                    key=lambda item: (item.eval_loss, item.step, item.ordinal),
                )
                if (loss, candidate.step, candidate.ordinal) >= (
                    worst.eval_loss,
                    worst.step,
                    worst.ordinal,
                ):
                    return control
            model = kwargs.get("model")
            if model is None:
                pool.capture_errors.append(f"eval {ordinal}: model missing")
                return control
            try:
                if strategy == "full":
                    save_full_export(
                        model,
                        tokenizer,
                        str(candidate.path),
                        metadata_source_dir,
                    )
                elif strategy == "lora":
                    from forge.tasks.common import save_adapter

                    save_adapter(model, tokenizer, str(candidate.path))
                else:
                    raise BloomzExperimentError(f"unsupported checkpoint strategy: {strategy}")
                pool.entries.append(candidate)
                if len(pool.entries) > pool.limit:
                    worst = max(
                        pool.entries,
                        key=lambda item: (item.eval_loss, item.step, item.ordinal),
                    )
                    pool.entries.remove(worst)
                    if worst.path != candidate.path:
                        _remove_checkpoint_dir(worst.path)
                if pool.best_loss is None or loss < pool.best_loss:
                    pool.best_loss = loss
                    pool.best_step = step
                    if visible_output_dir is not None:
                        try:
                            if strategy == "full":
                                save_full_export(
                                    model,
                                    tokenizer,
                                    visible_output_dir,
                                    metadata_source_dir,
                                )
                            else:
                                from forge.tasks.common import save_adapter

                                save_adapter(model, tokenizer, visible_output_dir)
                        except Exception as exc:
                            message = (
                                f"eval {ordinal} visible export: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            pool.visible_export_errors.append(message)
                            _emit("bloomz_visible_export_failed", error=message)
                _emit(
                    "bloomz_checkpoint_retained",
                    step=candidate.step,
                    eval_ordinal=ordinal,
                    eval_loss=round(loss, 6),
                    retained=len(pool.entries),
                    cap=pool.limit,
                )
            except Exception as exc:
                _remove_checkpoint_dir(candidate.path)
                message = f"eval {ordinal}: {type(exc).__name__}: {exc}"
                pool.capture_errors.append(message)
                _emit("bloomz_checkpoint_capture_failed", error=message)
            return control

    return BloomCheckpointCallback()


def make_eval_counter_callback(counter: BloomEvalCounter) -> Any:
    """Count finite scheduled dev evaluations without retaining weights."""
    from transformers import TrainerCallback

    class BloomEvalCounterCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            if _finite_loss((metrics or {}).get("eval_loss")) is not None:
                counter.completed += 1
            return control

    return BloomEvalCounterCallback()


def write_checkpoint_inventory(
    pool: BloomCheckpointPool,
    path: str | os.PathLike[str],
    *,
    strategy: str,
    phase: str,
    schedule_completed: bool,
    planned_steps: int,
    final_step: int,
    runtime_authority_path: str | os.PathLike[str],
    runtime_authority_sha256: str,
    runtime_authority: Mapping[str, Any],
    gpu_admission_path: str | os.PathLike[str],
    gpu_admission_sha256: str,
    gpu_admission: Mapping[str, Any],
) -> bool:
    """Seal the bounded candidate inventory for the external scoring phase."""
    pool.enabled = False
    entries = pool.sorted_entries()
    eligible = (
        schedule_completed
        and pool.eval_count >= MIN_COMPLETED_EVALS
        and len(entries) >= 4
        and not pool.capture_errors
    )
    payload = {
        "schema_version": 1,
        "status": "EXTERNAL_SCORE_READY" if eligible else "STOP_NO_SCIENCE",
        "strategy": strategy,
        "phase": phase,
        "trainer_loss_role": "nomination_only_not_decision_evidence",
        "evidence_label": (
            "composite_train_s2048_external_selection_s4096_"
            "trainer_loss_nomination_only"
        ),
        "required_external_scores": 4,
        "schedule_completed": schedule_completed,
        "planned_steps": planned_steps,
        "final_step": final_step,
        "completed_scheduled_evals": pool.eval_count,
        "max_retained_checkpoints": pool.limit,
        "capture_errors": list(pool.capture_errors),
        "visible_export_errors": list(pool.visible_export_errors),
        "runtime_authority_path": str(Path(runtime_authority_path).resolve()),
        "authority": authority_fields(runtime_authority, runtime_authority_sha256),
        "gpu_admission": gpu_admission_fields(
            gpu_admission,
            gpu_admission_path,
            gpu_admission_sha256,
        ),
        "checkpoints": [
            {
                "path": str(entry.path.resolve()),
                "tree_sha256": _tree_sha256(entry.path),
                "nomination_eval_loss": entry.eval_loss,
                "step": entry.step,
                "ordinal": entry.ordinal,
            }
            for entry in entries
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, destination)
    _emit(
        "bloomz_checkpoint_inventory",
        status=payload["status"],
        scheduled_evals=pool.eval_count,
        retained=len(entries),
        strategy=strategy,
        phase=phase,
    )
    return eligible


_EXPORT_METADATA_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "merges.txt",
    }
)


def save_full_export(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    metadata_source_dir: str,
) -> None:
    """Atomically export full weights with byte-exact base metadata.

    ``load_base`` changes ``use_cache`` and tokenizer padding side for training.
    These proxies restore the immutable config/tokenizer files *inside* the
    existing atomic staging transaction, before structural validation and the
    readiness marker.  The shared production saver remains byte-for-byte
    unchanged.
    """
    from forge.tasks.common import save_adapter

    if getattr(model, "peft_config", None):
        raise BloomzExperimentError("full export unexpectedly contains a PEFT adapter")
    source = Path(metadata_source_dir).resolve()
    if not (source / "config.json").is_file():
        raise BloomzExperimentError("full export metadata source lacks config.json")

    class _ModelProxy:
        def save_pretrained(self, path: str, **kwargs: Any) -> Any:
            return model.save_pretrained(path, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

    class _TokenizerProxy:
        def save_pretrained(self, path: str, **kwargs: Any) -> Any:
            result = tokenizer.save_pretrained(path, **kwargs)
            _restore_metadata(Path(path), source)
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(tokenizer, name)

    save_adapter(_ModelProxy(), _TokenizerProxy(), output_dir)
    exported = Path(output_dir)
    if (exported / "adapter_config.json").exists():
        raise BloomzExperimentError("full export contains adapter_config.json")
    if _sha256(exported / "config.json") != _sha256(source / "config.json"):
        raise BloomzExperimentError("exported config is not byte-identical to base")


def _restore_metadata(staged: Path, source: Path) -> None:
    import shutil

    for name in _EXPORT_METADATA_FILES:
        source_path = source / name
        staged_path = staged / name
        if source_path.is_file():
            shutil.copyfile(source_path, staged_path)
        elif staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()


def training_manifest_contract() -> dict[str, Any]:
    """Return the confirmation-blind manifest allowed in the training mount."""
    return {
        "schema_version": "sn56.bloomz-training-fixture.v1",
        "identities": {
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "parquet_sha256": DATASET_PARQUET_SHA256,
            },
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "config_sha256": MODEL_CONFIG_SHA256,
                "tokenizer_json_sha256": MODEL_TOKENIZER_SHA256,
            },
        },
        "splits": {
            "train": {
                "filename": "train.jsonl",
                "row_count": EXPECTED_TRAIN_ROWS,
                "sha256": "e4a0e2d83c9b39d0388931e868e04fdc0cc45288a29c18228ad13555ee1c52c0",
            },
            "dev": {
                "filename": "dev.jsonl",
                "row_count": EXPECTED_DEV_ROWS,
                "sha256": "f5548b1864a55c208f9f8061cb0e1d2471a6e58b976bb532ffdbb7a584bbfad6",
            },
        },
        "schema": {
            "fields": ["system", "instruct", "output"],
            "dataset_type": {
                "filename": "dataset-type.json",
                "sha256": "6a43eb4f03c0979e910e1a0f13d3510b9173d92063c091ece1d707769bf5d012",
            },
        },
        "artifacts": {
            "baseline_stats": {
                "filename": "baseline-stats.json",
                "sha256": "31c9c00c29fdd147a221c5c934170c4c422dba703350f3fee8952fc8de095b6f",
            }
        },
    }


def _read_training_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise BloomzExperimentError("training manifest missing or too large")
    if path.name != "training-manifest.json" or _sha256(path) != TRAINING_MANIFEST_SHA256:
        raise BloomzExperimentError("training manifest SHA-256 drift")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BloomzExperimentError(f"invalid fixture manifest: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value != training_manifest_contract()
    ):
        raise BloomzExperimentError("training manifest contract drift")
    return value


def _verify_training_fixture_directory(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> None:
    """Require a flat mount that physically contains no confirmation material."""
    root = manifest_path.parent.resolve()
    allowed = {
        "training-manifest.json",
        "train.jsonl",
        "dev.jsonl",
        "dataset-type.json",
        "baseline-stats.json",
    }
    children = list(root.iterdir())
    if (
        {child.name for child in children} != allowed
        or any(not child.is_file() or child.is_symlink() for child in children)
    ):
        raise BloomzExperimentError(
            "training fixture must physically contain only train/dev authority files"
        )
    schema = manifest["schema"]["dataset_type"]
    baseline = manifest["artifacts"]["baseline_stats"]
    for facts in (schema, baseline):
        path = root / facts["filename"]
        if _sha256(path) != facts["sha256"]:
            raise BloomzExperimentError("training fixture support-file bytes drift")


def _verify_manifest_identities(manifest: Mapping[str, Any]) -> None:
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping):
        raise BloomzExperimentError("fixture identities missing")
    dataset = identities.get("dataset")
    model = identities.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise BloomzExperimentError("fixture model/dataset identities missing")
    if (
        dataset.get("repo") != DATASET_REPO
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("parquet_sha256") != DATASET_PARQUET_SHA256
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or model.get("config_sha256") != MODEL_CONFIG_SHA256
        or model.get("tokenizer_json_sha256") != MODEL_TOKENIZER_SHA256
    ):
        raise BloomzExperimentError("fixture public authority drift")


def _verified_split_path(
    manifest_path: Path, manifest: Mapping[str, Any], split_name: str
) -> Path:
    splits = manifest.get("splits")
    split = splits.get(split_name) if isinstance(splits, Mapping) else None
    if not isinstance(split, Mapping):
        raise BloomzExperimentError(f"fixture split missing: {split_name}")
    filename = split.get("filename")
    expected = split.get("sha256")
    row_count = split.get("row_count")
    if (
        not isinstance(filename, str)
        or not filename
        or not isinstance(expected, str)
        or len(expected) != 64
        or not isinstance(row_count, int)
        or row_count <= 0
    ):
        raise BloomzExperimentError(f"invalid fixture split metadata: {split_name}")
    root = manifest_path.parent.resolve()
    path = (root / filename).resolve()
    try:
        inside = os.path.commonpath((str(root), str(path))) == str(root)
    except ValueError:
        inside = False
    if not inside or not path.is_file() or _sha256(path) != expected:
        raise BloomzExperimentError(f"fixture split bytes mismatch: {split_name}")
    return path


def _parse_learning_rate(raw: str | None) -> float:
    try:
        value = float(raw) if raw is not None else math.nan
    except (TypeError, ValueError, OverflowError) as exc:
        raise BloomzExperimentError("invalid FORGE_BLOOMZ_LR") from exc
    if not math.isfinite(value) or value <= 0:
        raise BloomzExperimentError("invalid FORGE_BLOOMZ_LR")
    return value


def _optional_positive_int(raw: str | None) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BloomzExperimentError("invalid FORGE_BLOOMZ_MAX_STEPS") from exc
    if value <= 0:
        raise BloomzExperimentError("FORGE_BLOOMZ_MAX_STEPS must be positive")
    return value


def _finite_loss(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise BloomzExperimentError(f"checkpoint directory missing: {path}") from exc
    if not root.is_dir():
        raise BloomzExperimentError(f"checkpoint directory missing: {path}")
    files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise BloomzExperimentError(f"checkpoint contains symlink: {item}")
        if not item.is_dir() and not item.is_file():
            raise BloomzExperimentError(f"checkpoint contains special file: {item}")
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
            )
    return hashlib.sha256(_canonical_bytes(files)).hexdigest()


def _remove_checkpoint_dir(path: Path) -> None:
    import shutil

    if path.is_dir() and path.parent.name == "bloomz-decision-checkpoints":
        shutil.rmtree(path)


def _emit(name: str, **fields: Any) -> None:
    try:
        from forge import telemetry

        telemetry.event(name, **fields)
    except Exception:
        pass
