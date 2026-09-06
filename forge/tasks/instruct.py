"""Supervised fine-tuning for InstructTextTask and ChatTask.

Strategy defaults to the validated LoRA path.  Experimental full fine-tuning is
fail-closed behind an explicit environment opt-in and remains subject to KL,
topology, and memory-fit gates; it must be GPU-certified before deployment.

Loss is completion-only, matching the evaluator. We also hold out a small slice
and durably export the best measured checkpoint; later unevaluated weights never
replace that measured minimum.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from forge import telemetry
from forge.baseline import (
    load_baseline_summary,
    telemetry_fields,
)
from forge.clock import Deadline
from forge.data import loader, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.model import (
    attach_lora,
    conservative_qwen35_plan,
    conservative_quasar_plan,
    decide_full_finetune,
    effective_sft_seq_len,
    gpu_topology,
    is_quasar_model,
    load_base,
    model_param_billions,
    prepare_full_finetune,
)
from forge.tasks.common import (
    ARTIFACT_COMPLETE_BEST,
    ARTIFACT_FLOOR,
    ARTIFACT_PARTIAL_TRAINED_BEST,
    BestTracker,
    _free_cuda,
    _is_oom,
    _make_best_checkpoint_callback,
    _make_periodic_save_callback,
    build_training_kwargs,
    compatible_dataclass_kwargs,
    safe_train,
    save_adapter,
    should_final_save,
    time_aware_epochs,
    write_artifact_truth,
    workdir,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.lfm25_epoch_cap import cap_lfm25_production_epochs
from forge.tuning.granite41_epoch_cap import cap_granite41_production_epochs
from forge.tuning.plan import TrainPlan, make_sft_plan
from forge.tuning.qwen35_soup import (
    apply_qwen35_soup_override,
    eligible_qwen35_soup_route,
    make_qwen35_soup_capture_callback,
)

# Hold out a small fixed validation slice for measured checkpoint selection.
# Skip it on datasets too small to spare rows.
_EVAL_VAL_ROWS = 256
_EVAL_MIN_DATASET = 1000
_BATCHES = (4, 2, 1)
_CAP_QUANTUM = 256

# Promotion of the externally scored BloomZ candidate is deliberately bound to
# the immutable base files, not just the caller-controlled model name. All
# other model revisions continue through the survival path below unchanged.
_BLOOMZ_REPO = "bigscience/bloomz-560m"
_BLOOMZ_REVISION = "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05"
_BLOOMZ_CONFIG_SHA256 = (
    "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb"
)
_BLOOMZ_WEIGHTS_SHA256 = (
    "365b2c5e9bd1057eb1e3f1a4fc3f89ae6584d20f24b682d2406bc7e90178ec13"
)
_BLOOMZ_TOKENIZER_SHA256 = (
    "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3"
)
_BLOOMZ_TOKENIZER_CONFIG_SHA256 = (
    "ae85f7ec32efe4ba09f3914743b0187528eab0322fe90c4e077a9229d1de64a9"
)
_BLOOMZ_SPECIAL_TOKENS_SHA256 = (
    "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"
)
_BLOOMZ_PARAMS_B = 0.559214592
_BLOOMZ_MAX_STEPS = 256
_BLOOMZ_SEQ_LEN = 2048
_BLOOMZ_SAVE_STEPS = frozenset({64, 128, 192, 256})
_BLOOMZ_REQUIRED_METADATA = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
)
_BLOOMZ_METADATA_FILES = frozenset(
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


class _BloomzPromotionError(RuntimeError):
    """The pinned production candidate cannot be exported safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_exact_bloomz_revision(model: Any, model_dir: str) -> bool:
    """Verify the immutable BloomZ revision from local bytes, without network."""
    root = Path(model_dir).resolve()
    required = (
        ("config.json", _BLOOMZ_CONFIG_SHA256, None),
        ("tokenizer.json", _BLOOMZ_TOKENIZER_SHA256, None),
        ("tokenizer_config.json", _BLOOMZ_TOKENIZER_CONFIG_SHA256, 222),
        ("special_tokens_map.json", _BLOOMZ_SPECIAL_TOKENS_SHA256, 85),
        ("model.safetensors", _BLOOMZ_WEIGHTS_SHA256, None),
    )
    try:
        for name, expected_sha, expected_size in required:
            path = root / name
            if not path.is_file():
                return False
            if expected_size is not None and path.stat().st_size != expected_size:
                return False
            if _sha256(path) != expected_sha:
                return False
        if any(
            (root / name).exists()
            for name in ("added_tokens.json", "chat_template.jinja")
        ):
            return False
        raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    expected_config = {
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
    config = getattr(model, "config", None)
    return (
        isinstance(raw, dict)
        and all(raw.get(key) == value for key, value in expected_config.items())
        and "max_position_embeddings" not in raw
        and getattr(config, "model_type", None) == "bloom"
        and tuple(getattr(config, "architectures", ()) or ())
        == ("BloomForCausalLM",)
        and getattr(config, "vocab_size", None) == 250880
    )


def _is_bloomz_promotion(spec: TaskSpec, loaded: Any, params_b: float) -> bool:
    """Fail closed unless the request and loaded base match the winning route."""
    return (
        spec.task_type == "InstructTextTask"
        and spec.instruct is not None
        and spec.instruct.output is not None
        and not spec.use_kl
        and _supported_bloomz_request(spec)
        and abs(params_b - _BLOOMZ_PARAMS_B) <= 1e-9
        and _has_exact_bloomz_revision(loaded.model, loaded.model_dir)
    )


def _supported_bloomz_request(spec: TaskSpec) -> bool:
    """Recognize named BloomZ and the validator's anonymous cache contract."""
    cached_model_dir = str(spec.cached_model_dir)
    if spec.model == _BLOOMZ_REPO:
        return cached_model_dir == f"/cache/models/{_BLOOMZ_REPO.replace('/', '--')}"
    return bool(
        isinstance(spec.model, str)
        and re.fullmatch(r"[0-9a-f]{16}", spec.model) is not None
        and isinstance(spec.baseline_stats_path, str)
        and spec.baseline_stats_path
        and cached_model_dir == f"/cache/models/{spec.model}"
    )


def _bloomz_plan(plan: TrainPlan) -> TrainPlan:
    """Apply the exact promotion-grade full-FT geometry."""
    return replace(
        plan,
        strategy="full",
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        learning_rate=1.0e-4,
        per_device_batch_size=1,
        grad_accum_steps=16,
        max_seq_len=_BLOOMZ_SEQ_LEN,
        warmup_ratio=0.03,
        weight_decay=0.0,
        optimizer="adamw_torch_fused",
        lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
    )


def _prepare_bloomz_full_finetune(model: Any) -> Any:
    """Require the full-FT master parameters proven by the winning run."""
    import torch

    model = prepare_full_finetune(model, gradient_checkpointing=True)
    parameters = list(model.parameters())
    if not parameters or any(
        not parameter.requires_grad or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise _BloomzPromotionError(
            "BloomZ full fine-tune requires every parameter trainable in fp32"
        )
    return model


def _validate_bloomz_staging(staged: Path, source: Path) -> None:
    """Validate metadata, serialization kind, and native offline loading."""

    def validate_tree() -> None:
        pending = [(staged, ())]
        while pending:
            directory, parent_parts = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    parts = (*parent_parts, entry.name)
                    if any("adapter" in part.casefold() for part in parts):
                        raise _BloomzPromotionError(
                            "full export contains an adapter artifact"
                        )
                    if entry.is_symlink():
                        raise _BloomzPromotionError("full export contains a symlink")
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        pending.append((Path(entry.path), parts))
                    elif not stat.S_ISREG(mode):
                        raise _BloomzPromotionError(
                            "full export contains a non-regular node"
                        )

    def unlink_metadata_destination(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise _BloomzPromotionError(
                f"unsafe staged metadata destination: {path.name}"
            )
        path.unlink()

    validate_tree()
    for name in _BLOOMZ_REQUIRED_METADATA:
        if not (source / name).is_file():
            raise _BloomzPromotionError(f"pinned base lacks required {name}")
    for name in _BLOOMZ_METADATA_FILES:
        source_path = source / name
        staged_path = staged / name
        if source_path.is_file():
            unlink_metadata_destination(staged_path)
            shutil.copyfile(source_path, staged_path)
            if staged_path.read_bytes() != source_path.read_bytes():
                raise _BloomzPromotionError(f"staged {name} differs from pinned base")
        elif staged_path.exists() or staged_path.is_symlink():
            unlink_metadata_destination(staged_path)
    validate_tree()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        staged, local_files_only=True
    )
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        staged, local_files_only=True
    )
    del reloaded_tokenizer, reloaded_model


def _save_bloomz_full_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    metadata_source_dir: str,
    *,
    artifact_truth: str | None = None,
    optimizer_step: int = 0,
    truth_reason: str = "unspecified",
) -> None:
    """Atomically save full weights while restoring native base metadata."""
    if getattr(model, "peft_config", None):
        raise _BloomzPromotionError("full export unexpectedly contains an adapter")
    source = Path(metadata_source_dir).resolve()
    if not (source / "config.json").is_file():
        raise _BloomzPromotionError("full export metadata source lacks config.json")

    class _ModelProxy:
        def save_pretrained(self, path: str, **kwargs: Any) -> Any:
            return model.save_pretrained(path, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

    class _TokenizerProxy:
        def save_pretrained(self, path: str, **kwargs: Any) -> Any:
            result = tokenizer.save_pretrained(path, **kwargs)
            _validate_bloomz_staging(Path(path), source)
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(tokenizer, name)

    save_adapter(
        _ModelProxy(),
        _TokenizerProxy(),
        output_dir,
        artifact_truth=artifact_truth,
        optimizer_step=optimizer_step,
        truth_reason=truth_reason,
    )


def _make_bloomz_save_callback(
    spec: TaskSpec, tokenizer: Any, metadata_source_dir: str
) -> Any:
    """Publish only native-validated full models at the science cadence."""
    from transformers import TrainerCallback

    class BloomzSaveCallback(TrainerCallback):
        last_saved_step = 0

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            step = int(getattr(state, "global_step", 0) or 0)
            model = kwargs.get("model")
            if (
                step not in _BLOOMZ_SAVE_STEPS
                or step == self.last_saved_step
                or model is None
            ):
                return control
            try:
                truth = (
                    ARTIFACT_COMPLETE_BEST
                    if step == _BLOOMZ_MAX_STEPS
                    else ARTIFACT_PARTIAL_TRAINED_BEST
                )
                _save_bloomz_full_model(
                    model,
                    tokenizer,
                    spec.output_dir,
                    metadata_source_dir,
                    artifact_truth=truth,
                    optimizer_step=step,
                    truth_reason="bloomz_science_cadence",
                )
                self.last_saved_step = step
            except Exception as exc:
                telemetry.event(
                    "bloomz_checkpoint_export_failed",
                    step=step,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return control

    return BloomzSaveCallback()


@dataclass(frozen=True)
class _DataView:
    """One sequence-cap-specific dataset, split, and ordering authority."""

    cap: int
    cap_basis: str
    source_rows: int
    retained_source_rows: int
    retained_rows: int
    retained_fraction: float
    ordering_sha256: str
    train: list
    validation: list
    authorized: bool = True
    required_retained_fraction: float = 0.0

    def identity(self) -> dict[str, Any]:
        return {
            "sequence_cap": self.cap,
            "sequence_cap_basis": self.cap_basis,
            "source_rows": self.source_rows,
            "retained_source_rows": self.retained_source_rows,
            "rows_retained": self.retained_source_rows,
            "retained_rows": self.retained_rows,
            "tokenized_examples": self.retained_rows,
            "retained_fraction": self.retained_fraction,
            "train_rows": len(self.train),
            "train_n": len(self.train),
            "validation_rows": len(self.validation),
            "val_n": len(self.validation),
            "ordering_sha256": self.ordering_sha256,
            "cap_authorized": self.authorized,
            "required_retained_fraction": self.required_retained_fraction,
        }


def _run_bloomz_promotion(
    spec: TaskSpec,
    deadline: Deadline,
    rows: list,
    loaded: Any,
    tokenizer: Any,
    *,
    params_b: float,
    n_gpus: int,
    per_gpu_gb: float,
) -> None:
    """Run the exact externally promoted BloomZ recipe and export cadence."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    plan = _bloomz_plan(
        make_sft_plan(
            use_kl=False,
            strategy="full",
            params_b=params_b,
            weight_rms=None,
            n_gpus=n_gpus,
            per_gpu_gb=per_gpu_gb,
        )
    )
    telemetry.event(
        "bloomz_full_ft_promoted",
        revision=_BLOOMZ_REVISION,
        max_steps=_BLOOMZ_MAX_STEPS,
    )
    telemetry.event(
        "strategy_chosen",
        strategy="full",
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        per_gpu_gb=per_gpu_gb,
    )

    model = _prepare_bloomz_full_finetune(loaded.model)
    loaded.model = None
    telemetry.event("model_loaded", rows=len(rows))

    from forge.tasks.fallback import emit_untrained_copy

    telemetry.event("full_ft_floor")
    emit_untrained_copy(spec)
    write_artifact_truth(
        spec.output_dir,
        ARTIFACT_FLOOR,
        optimizer_step=0,
        reason="pretraining_floor",
    )

    assert spec.instruct is not None
    examples = prompts.build_instruct_examples(rows, spec.instruct)
    tokenized = tokenize.tokenize_instruct(
        examples, tokenizer, _BLOOMZ_SEQ_LEN
    )
    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")

    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        handler="instruct",
        strategy="full",
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        gpu_gb=per_gpu_gb,
        seq_len=_BLOOMZ_SEQ_LEN,
        seq_len_candidates=[_BLOOMZ_SEQ_LEN],
        baseline_seq_policy="pinned_bloomz_promotion",
        tokenized=len(tokenized),
        train_n=len(tokenized),
        val_n=0,
        lora_r=plan.lora_r,
        lr=plan.learning_rate,
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=eff_batch,
        gradient_checkpointing=plan.gradient_checkpointing,
        epochs=plan.num_epochs,
        neftune=True,
        tokens_per_step_cap=eff_batch * _BLOOMZ_SEQ_LEN,
    )

    kwargs = build_training_kwargs(spec, plan, neftune_alpha=5.0)
    kwargs.pop("num_train_epochs", None)
    kwargs["max_steps"] = _BLOOMZ_MAX_STEPS
    args = TrainingArguments(
        **compatible_dataclass_kwargs(
            TrainingArguments,
            kwargs,
            allow_removed={"overwrite_output_dir"},
        )
    )
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    saver = _make_bloomz_save_callback(spec, tokenizer, loaded.model_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(tokenized),
        eval_dataset=None,
        data_collator=collator,
        callbacks=[
            DeadlineCallback(deadline),
            saver,
            telemetry.make_trainer_callback(spec.output_dir),
        ],
    )

    safe_train(trainer)
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if final_step == _BLOOMZ_MAX_STEPS:
        if saver.last_saved_step != final_step:
            _save_bloomz_full_model(
                model,
                tokenizer,
                spec.output_dir,
                loaded.model_dir,
                artifact_truth=ARTIFACT_COMPLETE_BEST,
                optimizer_step=final_step,
                truth_reason="bloomz_fixed_schedule_complete",
            )
    else:
        # Keep the latest validated cadence generation (or the eager LoRA floor
        # before step 64); never replace either with an arbitrary partial step.
        telemetry.event(
            "bloomz_schedule_incomplete",
            final_step=final_step,
            required_step=_BLOOMZ_MAX_STEPS,
        )


def run(spec: TaskSpec, deadline: Deadline) -> None:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    rows = loader.load_rows(
        spec.cached_dataset_path, dataset_arg=spec.dataset, file_format=spec.file_format
    )
    if not rows:
        raise RuntimeError("empty dataset")

    loaded = load_base(spec.cached_model_dir, for_generation=False)
    tokenizer = loaded.tokenizer
    telemetry.collect_env()

    baseline_summary = None
    try:
        baseline_summary = load_baseline_summary(
            spec.baseline_stats_path, expected_task_type=spec.task_type
        )
    except Exception as exc:
        # Stats are untrusted validator input.  A rejected payload is diagnostic
        # only; preserve the static plan and never publish the raw contents.
        telemetry.event(
            "baseline_stats_invalid", error=f"{type(exc).__name__}: {exc}"
        )
    if baseline_summary is not None:
        telemetry.set_meta(**telemetry_fields(baseline_summary))

    is_kl = spec.use_kl and spec.kl_coef > 0
    params_b = model_param_billions(loaded.model)
    n_gpus, per_gpu_gb = gpu_topology()
    if _is_bloomz_promotion(spec, loaded, params_b):
        _run_bloomz_promotion(
            spec,
            deadline,
            rows,
            loaded,
            tokenizer,
            params_b=params_b,
            n_gpus=n_gpus,
            per_gpu_gb=per_gpu_gb,
        )
        return
    use_full = decide_full_finetune(
        use_kl=is_kl, params_b=params_b, n_gpus=n_gpus, per_gpu_gb=per_gpu_gb
    )
    strategy = "full" if use_full else "lora"
    from forge.model import median_weight_rms

    plan = make_sft_plan(
        use_kl=is_kl,
        strategy=strategy,
        params_b=params_b,
        weight_rms=median_weight_rms(loaded.model) if use_full else None,
        n_gpus=n_gpus,
        per_gpu_gb=per_gpu_gb,
    )
    original_batch = plan.per_device_batch_size
    original_grad_accum = plan.grad_accum_steps
    plan, qwen35_geometry_changed = conservative_qwen35_plan(loaded.model, plan)
    if qwen35_geometry_changed:
        telemetry.event(
            "qwen35_conservative_geometry",
            original_batch=original_batch,
            original_grad_accum=original_grad_accum,
            batch=plan.per_device_batch_size,
            grad_accum=plan.grad_accum_steps,
            effective_batch=plan.per_device_batch_size * plan.grad_accum_steps,
            reason="h100_proven_pretrainer_geometry",
        )
    plan, quasar_geometry_changed = conservative_quasar_plan(loaded.model, plan)
    checkpointing_supported = not is_quasar_model(loaded.model)
    if quasar_geometry_changed:
        # The mandatory Quasar remote code advertises gradient checkpointing,
        # but its decoder never invokes Transformers' checkpoint function. Start
        # at microbatch 1 so the 10B forced rounds do not depend on a fictitious
        # memory saving; preserve the original effective batch via accumulation.
        telemetry.event(
            "quasar_conservative_geometry",
            original_batch=original_batch,
            batch=plan.per_device_batch_size,
            grad_accum=plan.grad_accum_steps,
            reason="remote_gradient_checkpointing_is_noop",
        )
    telemetry.event(
        "strategy_chosen",
        strategy=strategy,
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        per_gpu_gb=per_gpu_gb,
    )

    init_rng = _torch_rng()
    if use_full:
        model = prepare_full_finetune(
            loaded.model, gradient_checkpointing=plan.gradient_checkpointing
        )
    else:
        model = attach_lora(
            loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
        )
    loaded.model = None

    telemetry.event("model_loaded", rows=len(rows))

    # Floor first, before the minutes-long tokenization of a large dataset: write
    # a valid (untrained) artifact so a kill anywhere in setup still leaves a
    # scoreable model at the output path. Training overwrites it (the atomic swap
    # replaces the whole dir).
    #
    # For LoRA the untrained adapter is a valid finetune (adapter_config.json is
    # always detected as a finetune). For FULL-FT, saving the untrained full model
    # would be byte-identical to the base — which the evaluator scores as
    # non-finetuned AND which would trap the fallback (its _has_weights guard
    # keeps it). So use the LoRA-adapter floor (loaded from the cached base on CPU,
    # zero GPU cost): a valid, non-identical finetune until real training lands.
    if strategy == "full":
        from forge.tasks.fallback import emit_untrained_copy

        # A `fallback_emitted` event here is the intentional floor, not a failure —
        # a real fallback would have no later `train_end`.
        telemetry.event("full_ft_floor")
        emit_untrained_copy(spec)
        write_artifact_truth(
            spec.output_dir, ARTIFACT_FLOOR, optimizer_step=0, reason="pretraining_floor"
        )
    else:
        save_adapter(model, tokenizer, spec.output_dir, artifact_truth=ARTIFACT_FLOOR,
                     optimizer_step=0, truth_reason="pretraining_floor")

    # Start at G.O.D's evaluator-equivalent cap. If measured memory rejects every
    # full-cap geometry, later rungs may use smaller caps, but only when those
    # caps are derived from a validated task distribution and the exact rows are
    # retokenized, fingerprinted, and threaded through every downstream phase.
    initial_seq_len = effective_sft_seq_len(model, plan.max_seq_len)
    seq_candidates = tokenize.sft_sequence_len_candidates(
        model, tokenizer, initial_seq_len
    )
    if spec.chat is not None:
        source_items = prompts.build_chat_conversations(rows, spec.chat)
        template_resolution = tokenize.resolve_chat_template(
            spec.chat.chat_template, tokenizer
        )
        if template_resolution.degraded:
            # The valid floor already exists. Persist the compatibility choice
            # before processing rows so even a kill during tokenization explains
            # why native/ChatML semantics replaced the requested future name.
            telemetry.write_into(spec.output_dir)
        def tokenize_one(source_index: int, candidate: int) -> list:
            return tokenize.tokenize_chat_resolved(
                [source_items[source_index]], tokenizer, candidate,
                resolution=template_resolution,
            )
    else:
        assert spec.instruct is not None, "instruct task missing instruct columns"
        if spec.instruct.output is None:
            source_items = prompts.build_completion_documents(rows, spec.instruct)

            def tokenize_one(source_index: int, candidate: int) -> list:
                return tokenize.tokenize_completion(
                    [source_items[source_index]], tokenizer, candidate
                )
        else:
            source_items = prompts.build_instruct_examples(rows, spec.instruct)

            def tokenize_one(source_index: int, candidate: int) -> list:
                return tokenize.tokenize_instruct(
                    [source_items[source_index]], tokenizer, candidate
                )

    initial_view = None
    for candidate in seq_candidates:
        basis = ("evaluator_initial" if candidate == seq_candidates[0]
                 else "evaluator_nonempty_retry")
        view = _make_data_view(
            candidate, basis, len(source_items), tokenize_one, is_kl
        )
        if view.retained_rows:
            initial_view = view
            break
    if initial_view is None:
        raise RuntimeError("no trainable examples after tokenization")
    seq_len = initial_view.cap
    plan = replace(plan, max_seq_len=seq_len)
    reduced_cap_rows, cap_authority = _reduced_sequence_caps(
        baseline_summary, initial_view, expected_records=len(rows)
    )
    plans = _plans(
        plan,
        tuple(cap for cap, _basis, _minimum in reduced_cap_rows),
        checkpointing_supported=checkpointing_supported,
    )
    cap_basis = {
        seq_len: initial_view.cap_basis,
        **{cap: basis for cap, basis, _minimum in reduced_cap_rows},
    }
    cap_minimum = {
        cap: minimum for cap, _basis, minimum in reduced_cap_rows
    }
    views = {seq_len: initial_view}

    telemetry.event("sequence_cap_view", **initial_view.identity())
    telemetry.event("sequence_cap_authority", **cap_authority)
    initial_view = None
    view = None

    def data_for(candidate: TrainPlan) -> _DataView:
        cap = int(candidate.max_seq_len)
        if cap not in views:
            views.clear()
            gc.collect()
            views[cap] = _make_data_view(
                cap,
                cap_basis[cap],
                len(source_items),
                tokenize_one,
                is_kl,
                required_retained_fraction=cap_minimum.get(cap, 0.0),
            )
            telemetry.event("sequence_cap_view", **views[cap].identity())
        return views[cap]

    collator = tokenize.PadCollator(tokenizer.pad_token_id)

    def rebuild(candidate: TrainPlan) -> Any:
        reopened = load_base(spec.cached_model_dir, for_generation=False)
        base = reopened.model
        reopened.model = None
        _restore_torch_rng(init_rng)
        if use_full:
            return prepare_full_finetune(base, gradient_checkpointing=candidate.gradient_checkpointing)
        return attach_lora(base, r=candidate.lora_r, alpha=candidate.lora_alpha,
                           dropout=candidate.lora_dropout)

    def make(candidate: TrainPlan, candidate_model: Any, view: _DataView,
             probe_rows: list | None = None, epochs: float | None = None):
        return _make_trainer(
            spec, deadline, candidate, candidate_model, tokenizer,
            probe_rows if probe_rows is not None else view.train,
            [] if probe_rows is not None else view.validation,
            collator, is_kl, strategy, n_gpus,
            probe=probe_rows is not None, epochs=epochs)

    holder = [model]
    del model
    plan, admitted_view, admitted_index, admission, admitted = _admit(
        plans, holder.pop(), data_for, rebuild, make
    )
    views.clear()
    views[admitted_view.cap] = admitted_view
    gc.collect()
    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        handler="chat" if spec.chat is not None else "instruct",
        strategy=strategy,
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        gpu_gb=per_gpu_gb,
        seq_len=plan.max_seq_len,
        sequence_cap_basis=admitted_view.cap_basis,
        seq_len_candidates=seq_candidates,
        survival_seq_caps=list(dict.fromkeys(
            candidate.max_seq_len for candidate in plans
        )),
        baseline_seq_policy="validated_distribution_survival_fallback_only",
        tokenized=admitted_view.retained_rows,
        source_rows=admitted_view.source_rows,
        rows_retained=admitted_view.retained_source_rows,
        retained_source_rows=admitted_view.retained_source_rows,
        retained_fraction=admitted_view.retained_fraction,
        dataset_ordering_sha256=admitted_view.ordering_sha256,
        train_n=len(admitted_view.train),
        val_n=len(admitted_view.validation),
        lora_r=plan.lora_r,
        lr=plan.learning_rate,
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=eff_batch,
        gradient_checkpointing=plan.gradient_checkpointing,
        epochs=plan.num_epochs,
        neftune=not is_kl,
        tokens_per_step_cap=eff_batch * plan.max_seq_len,
        geometry_admission=admission,
        geometry_admitted=admitted,
    )
    admitted_view = None

    if is_kl:
        from forge.tuning.kl import KLSFTTrainer as probe_cls

        probe_extra = {"kl_coef": spec.kl_coef}
    else:
        probe_cls = Trainer
        probe_extra = None

    def measure_epochs(candidate: TrainPlan, view: _DataView) -> float | None:
        kwargs = build_training_kwargs(
            spec, candidate, neftune_alpha=None if is_kl else 5.0
        )
        timing_model = rebuild(candidate)
        try:
            candidate_epochs, probe_per_step = time_aware_epochs(
                trainer_cls=probe_cls,
                model=timing_model,
                kwargs=kwargs,
                train_ex=view.train,
                collator=collator,
                deadline=deadline,
                eff_batch=(candidate.per_device_batch_size
                           * candidate.grad_accum_steps),
                strategy=strategy,
                trainer_extra=probe_extra,
            )
        finally:
            holder = [timing_model]
            timing_model = None
            _discard(holder.pop())
        if candidate_epochs is not None:
            telemetry.event(
                "time_aware_epochs",
                epochs=candidate_epochs,
                planned=candidate.num_epochs,
                probe_per_step_s=round(probe_per_step, 4),
                **_plan_identity(candidate),
                **view.identity(),
            )
        return candidate_epochs

    trainer, tracker, soup_route, plan, training_view, training_outcome = _train_ladder(
        plans[admitted_index:], data_for, rebuild, make, measure_epochs,
        spec, tokenizer,
    )
    _record_training_selection(plan, training_view, training_outcome)
    if training_outcome == "progressed_oom_preserved":
        telemetry.event(
            "training_progressed_oom_preserved",
            **_plan_identity(plan),
            **training_view.identity(),
        )
        telemetry.write_into(spec.output_dir)
        return
    if training_outcome == "progressed_oom_export_failed":
        telemetry.event(
            "training_progressed_oom_export_failed",
            **_plan_identity(plan),
            **training_view.identity(),
        )
        telemetry.write_into(spec.output_dir)
        return
    if trainer is None or tracker is None or training_view is None:
        telemetry.event(
            "training_geometry_exhausted",
            terminal_artifact_truth=ARTIFACT_FLOOR,
            training_outcome=training_outcome,
            **_plan_identity(plan),
            **training_view.identity(),
        )
        write_artifact_truth(
            spec.output_dir,
            ARTIFACT_FLOOR,
            optimizer_step=0,
            reason="zero_progress_oom_ladder_exhausted",
        )
        telemetry.write_into(spec.output_dir)
        return
    model = trainer.model
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    truth = _truth(final_step, int(getattr(trainer.state, "max_steps", 0) or 0))
    if truth == ARTIFACT_FLOOR:
        write_artifact_truth(spec.output_dir, truth, optimizer_step=0, reason="no_progress")
        return
    selected_step = final_step
    if should_final_save(tracker, final_step=final_step):
        save_adapter(model, tokenizer, spec.output_dir, artifact_truth=truth,
                     optimizer_step=final_step, truth_reason="schedule_return")
    else:
        selected_step = int(tracker.best_step or 0)
        write_artifact_truth(spec.output_dir, truth, optimizer_step=selected_step,
                             reason="retained_best")
    apply_qwen35_soup_override(soup_route, tracker=tracker, model=model,
                               tokenizer=tokenizer)
    write_artifact_truth(spec.output_dir, truth, optimizer_step=selected_step,
                         reason="selected_export")


def _torch_rng() -> tuple[Any, Any]:
    import torch

    cpu = torch.random.get_rng_state().clone()
    cuda = [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    return cpu, cuda


def _restore_torch_rng(state: tuple[Any, Any]) -> None:
    import torch

    torch.random.set_rng_state(state[0])
    if state[1] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[1])


def _make_data_view(
    cap: int,
    cap_basis: str,
    source_rows: int,
    tokenize_one: Callable[[int, int], list],
    is_kl: bool,
    *,
    required_retained_fraction: float = 0.0,
) -> _DataView:
    """Retokenize exact source rows at ``cap`` and fingerprint their ordering."""
    tokenized = []
    retained_sources: set[int] = set()
    ordering = hashlib.sha256()
    for source_index in range(source_rows):
        chunks = tokenize_one(source_index, cap)
        if chunks:
            retained_sources.add(source_index)
        for chunk_index, row in enumerate(chunks):
            payload = json.dumps(
                {
                    "source_index": source_index,
                    "chunk_index": chunk_index,
                    "row": row,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            ordering.update(len(payload).to_bytes(8, "big"))
            ordering.update(payload)
            tokenized.append(row)
    train, validation = (tokenized, []) if is_kl else _split_for_eval(tokenized)
    retained = len(retained_sources)
    retained_fraction = round(retained / source_rows, 12) if source_rows else 0.0
    return _DataView(
        cap=int(cap),
        cap_basis=cap_basis,
        source_rows=source_rows,
        retained_source_rows=retained,
        retained_rows=len(tokenized),
        retained_fraction=retained_fraction,
        ordering_sha256=ordering.hexdigest(),
        train=train,
        validation=validation,
        authorized=(
            bool(train) and retained_fraction >= required_retained_fraction
        ),
        required_retained_fraction=required_retained_fraction,
    )


def _reduced_sequence_caps(
    summary: Any,
    initial: _DataView,
    *,
    expected_records: int,
):
    """Return only distribution-authorized caps below the evaluator cap.

    The validator summary is schema-checked before reaching this function.  p99
    is the first quality-preserving rescue and p95 is the final survival rung;
    both round upward so the reported quantile remains inside the cap.
    """
    authority = {
        "policy": "validated_p99_then_p95_round_up_256",
        "quantum": _CAP_QUANTUM,
        "initial_cap": initial.cap,
        "source": "unavailable",
        "current_task_records": int(expected_records),
        "caps": [],
    }
    if summary is None:
        return (), authority
    authority.update(
        source="validated_baseline_summary",
        baseline_stats_sha256=summary.sha256,
        baseline_num_records=summary.num_records,
        sequence_p99=summary.sequence_p99,
        sequence_p95=summary.sequence_p95,
        record_count_matches=(summary.num_records == int(expected_records)),
    )
    if summary.num_records != int(expected_records):
        authority["source"] = "rejected_record_count_mismatch"
        return (), authority
    rows: list[tuple[int, str, float]] = []
    for quantile, value, minimum in (
        ("p99", summary.sequence_p99, 0.99),
        ("p95", summary.sequence_p95, 0.95),
    ):
        if value <= 0:
            continue
        cap = min(
            initial.cap,
            math.ceil(int(value) / _CAP_QUANTUM) * _CAP_QUANTUM,
        )
        if cap >= initial.cap:
            continue
        duplicate = next(
            (index for index, (existing, _, _) in enumerate(rows)
             if existing == cap),
            None,
        )
        if duplicate is not None:
            _cap, old_basis, old_minimum = rows[duplicate]
            rows[duplicate] = (
                cap,
                old_basis.replace("_round_up_", f"_{quantile}_round_up_"),
                min(old_minimum, minimum),
            )
            continue
        rows.append(
            (
                cap,
                f"validated_baseline_{quantile}_round_up_{_CAP_QUANTUM}",
                minimum,
            )
        )
    authority["caps"] = [cap for cap, _, _ in rows]
    return tuple(rows), authority


def _plan_identity(plan: TrainPlan) -> dict[str, Any]:
    batch = int(plan.per_device_batch_size)
    accum = int(plan.grad_accum_steps)
    return {
        "batch": batch,
        "grad_accum": accum,
        "effective_batch": batch * accum,
        "gradient_checkpointing": bool(plan.gradient_checkpointing),
        "seq_cap": int(plan.max_seq_len),
    }


def _record_training_selection(
    plan: TrainPlan, view: _DataView, outcome: str
) -> None:
    """Refresh top-level telemetry to the rung that actually ended training."""
    telemetry.set_meta(
        seq_len=plan.max_seq_len,
        sequence_cap_basis=view.cap_basis,
        tokenized=view.retained_rows,
        rows_retained=view.retained_source_rows,
        retained_source_rows=view.retained_source_rows,
        retained_fraction=view.retained_fraction,
        dataset_ordering_sha256=view.ordering_sha256,
        train_n=len(view.train),
        val_n=len(view.validation),
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=plan.per_device_batch_size * plan.grad_accum_steps,
        gradient_checkpointing=plan.gradient_checkpointing,
        tokens_per_step_cap=(plan.per_device_batch_size
                             * plan.grad_accum_steps * plan.max_seq_len),
        training_outcome=outcome,
    )


def _plans(
    plan: TrainPlan,
    reduced_caps: tuple[int, ...] = (),
    *,
    checkpointing_supported: bool = True,
) -> tuple[TrainPlan, ...]:
    """Ordered survival rungs with one fixed effective batch of 16."""
    requested = min(4, max(1, int(plan.per_device_batch_size)))
    start = next(batch for batch in _BATCHES if batch <= requested)
    first = replace(
        plan,
        per_device_batch_size=start,
        grad_accum_steps=16 // start,
        gradient_checkpointing=(
            plan.gradient_checkpointing if checkpointing_supported else False
        ),
    )
    candidates = [first]
    rescue_gc = checkpointing_supported
    if checkpointing_supported and not first.gradient_checkpointing:
        candidates.append(replace(first, gradient_checkpointing=True))
    for batch in _BATCHES:
        if batch >= start:
            continue
        candidates.append(
            replace(
                first,
                per_device_batch_size=batch,
                grad_accum_steps=16 // batch,
                gradient_checkpointing=rescue_gc,
            )
        )
    for cap in reduced_caps:
        cap = int(cap)
        if cap <= 0 or cap >= first.max_seq_len:
            continue
        candidates.append(
            replace(
                first,
                per_device_batch_size=1,
                grad_accum_steps=16,
                gradient_checkpointing=rescue_gc,
                max_seq_len=cap,
            )
        )
    unique: list[TrainPlan] = []
    identities = set()
    for candidate in candidates:
        identity = (
            candidate.per_device_batch_size,
            candidate.grad_accum_steps,
            candidate.gradient_checkpointing,
            candidate.max_seq_len,
        )
        if identity not in identities:
            identities.add(identity)
            unique.append(candidate)
    return tuple(unique)


def _probe_rows(rows: list, batch: int) -> list[tuple[str, list, dict[str, Any]]]:
    size = min(batch, len(rows))
    lengths = [len(row.get("input_ids", [])) for row in rows]
    ranked = sorted(range(len(rows)), key=lambda i: (lengths[i], i))
    p99 = lengths[ranked[min(len(rows) - 1, math.ceil(0.99 * len(rows)) - 1)]]
    groups = (("p99", sorted(range(len(rows)), key=lambda i: (abs(lengths[i] - p99), -lengths[i], i))[:size]),
              ("worst", sorted(range(len(rows)), key=lambda i: (-lengths[i], i))[:size]))
    return [
        (label, [rows[i] for i in indices],
         {"row_indices": indices, "row_lengths": [lengths[i] for i in indices]})
        for label, indices in groups
    ]


def _discard(value: Any, *, trainer: bool = False) -> None:
    if trainer:
        try:
            value.accelerator.free_memory()
        except Exception:
            pass
        for name in ("optimizer", "lr_scheduler", "model_wrapped", "model"):
            try:
                setattr(value, name, None)
            except Exception:
                pass
    else:
        try:
            value.zero_grad(set_to_none=True)
        except Exception:
            pass
    del value
    gc.collect()
    _free_cuda()


def _make_trainer(
    spec: TaskSpec, deadline: Deadline, plan: TrainPlan, model: Any,
    tokenizer: Any, train_ex: list, val_ex: list, collator: Any,
    is_kl: bool, strategy: str, n_gpus: int,
    *,
    probe: bool = False,
    epochs: float | None = None,
):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    if probe:
        kwargs["max_steps"] = 1
    elif epochs is not None:
        kwargs["num_train_epochs"] = epochs
        capped_epochs = cap_lfm25_production_epochs(
            spec,
            model,
            strategy=strategy,
            n_gpus=n_gpus,
            native_epochs=epochs,
        )
        if capped_epochs is not None:
            kwargs["num_train_epochs"] = capped_epochs
            telemetry.event(
                "lfm25_production_epoch_cap",
                observed_time_aware_epochs=float(epochs),
                applied_epochs=capped_epochs,
                override_applied=capped_epochs < float(epochs),
            )
        capped_epochs = cap_granite41_production_epochs(
            spec,
            model,
            strategy=strategy,
            n_gpus=n_gpus,
            native_epochs=epochs,
        )
        if capped_epochs is not None:
            kwargs["num_train_epochs"] = capped_epochs
            telemetry.event(
                "granite41_production_epoch_cap",
                observed_time_aware_epochs=float(epochs),
                applied_epochs=capped_epochs,
                override_applied=capped_epochs < float(epochs),
            )
    tracker = BestTracker()
    route = None if probe else eligible_qwen35_soup_route(
        spec, model, strategy=strategy, n_gpus=n_gpus,
        capture_root=os.path.join(workdir(spec), "qwen35-r4-r2-captures"))
    if val_ex:
        eff = plan.per_device_batch_size * plan.grad_accum_steps
        kwargs.update(eval_strategy="steps", eval_steps=max(1, (len(train_ex) // eff) // 4),
                      per_device_eval_batch_size=max(1, plan.per_device_batch_size))
    args = TrainingArguments(
        **compatible_dataclass_kwargs(
            TrainingArguments, kwargs, allow_removed={"overwrite_output_dir"}
        )
    )
    callbacks = []
    if not probe:
        callbacks = [
            DeadlineCallback(deadline),
            _make_periodic_save_callback(
                spec, tokenizer, every=100 if strategy == "full" else 25,
                tracker=tracker, label_artifact_truth=True,
            ),
            telemetry.make_trainer_callback(spec.output_dir),
        ]
        if val_ex:
            callbacks.append(
                _make_best_checkpoint_callback(
                    spec, tokenizer, tracker, label_artifact_truth=True
                ))
            if route is not None:
                callbacks.append(make_qwen35_soup_capture_callback(route, tokenizer))
    fields = dict(model=model, args=args, train_dataset=Dataset.from_list(train_ex),
                  eval_dataset=Dataset.from_list(val_ex) if val_ex else None,
                  data_collator=collator, callbacks=callbacks)
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer

        trainer = KLSFTTrainer(kl_coef=spec.kl_coef, **fields)
    else:
        trainer = Trainer(**fields)
    return trainer, tracker, route


def _probe_once(
    plan: TrainPlan,
    model: Any,
    rows: list,
    identity: dict[str, Any],
    view: _DataView,
    make: Callable,
) -> dict[str, Any]:
    import torch

    devices = tuple(range(torch.cuda.device_count()))
    trainer = None
    try:
        torch.cuda.empty_cache()
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        trainer = make(plan, model, view, rows)[0]
        model = None
        trainer.train()
        if int(getattr(trainer.state, "global_step", 0) or 0) != 1:
            raise RuntimeError("exact Trainer probe did not complete one optimizer step")
        for device in devices:
            torch.cuda.synchronize(device)
        peak = max(
            torch.cuda.max_memory_reserved(device)
            / torch.cuda.get_device_properties(device).total_memory
            for device in devices)
        return {**identity, "status": "PASS" if peak <= 0.90 else "HEADROOM_EXCEEDED",
                "peak_reserved_fraction": round(peak, 6)}
    except Exception as exc:
        if not _is_oom(exc):
            raise RuntimeError("exact production geometry measurement failed") from exc
        return {**identity, "status": "OOM", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if trainer is not None:
            holder = [trainer]
            trainer = None
            _discard(holder.pop(), trainer=True)
        elif model is not None:
            holder = [model]
            model = None
            _discard(holder.pop())


def _admit(
    plans: tuple[TrainPlan, ...],
    model: Any,
    data_for: Callable[[TrainPlan], _DataView],
    rebuild: Callable,
    make: Callable,
):
    import torch

    if not torch.cuda.is_available():
        _discard(model)
        view = data_for(plans[0])
        attempt = {
            **_plan_identity(plans[0]),
            **view.identity(),
            "status": "SKIP_NO_CUDA",
        }
        return plans[0], view, 0, [attempt], False
    attempts = []
    current = model
    model = None
    fallback = None
    view = None
    current_cap = None
    for plan_index, candidate in enumerate(plans):
        if current_cap is not None and candidate.max_seq_len != current_cap:
            view = None
            gc.collect()
        view = data_for(candidate)
        current_cap = candidate.max_seq_len
        if not view.authorized:
            attempt = {
                **_plan_identity(candidate),
                **view.identity(),
                "status": "CAP_AUTHORITY_REJECTED",
                "batches": [],
            }
            attempts.append(attempt)
            telemetry.event("geometry_admission_attempt", **attempt)
            continue
        if not view.train:
            attempt = {
                **_plan_identity(candidate),
                **view.identity(),
                "status": "NO_RETAINED_ROWS",
                "batches": [],
            }
            attempts.append(attempt)
            telemetry.event("geometry_admission_attempt", **attempt)
            continue
        fallback = (candidate, plan_index)
        observations = []
        for batch_index, (label, selected, identity) in enumerate(
                _probe_rows(view.train, candidate.per_device_batch_size)):
            if plan_index or batch_index:
                current = rebuild(candidate)
            holder = [current]
            current = None
            observation = _probe_once(
                candidate,
                holder.pop(),
                selected,
                {
                    "label": label,
                    **_plan_identity(candidate),
                    **view.identity(),
                    **identity,
                },
                view,
                make,
            )
            observations.append(observation)
            if observation["status"] != "PASS":
                break
        status = ("PASS" if len(observations) == 2
                  and all(o["status"] == "PASS" for o in observations)
                  else observations[-1]["status"])
        attempt = {
            **_plan_identity(candidate),
            **view.identity(),
            "status": status,
            "batches": observations,
        }
        attempts.append(attempt)
        telemetry.event("geometry_admission_attempt", **attempt)
        if status == "PASS":
            return candidate, view, plan_index, attempts, True
    if fallback is None:
        raise RuntimeError("all distribution-authorized SFT geometries were empty")
    candidate, plan_index = fallback
    view = None
    gc.collect()
    view = data_for(candidate)
    telemetry.event(
        "geometry_admission_exhausted_fallback",
        attempts=len(attempts),
        admitted=False,
        **_plan_identity(candidate),
        **view.identity(),
    )
    return candidate, view, plan_index, attempts, False


def _truth_matches(output_dir: str, truth: str, step: int) -> bool:
    try:
        with open(
            os.path.join(output_dir, "forge_artifact_truth.json"),
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("truth") == truth
            and int(payload.get("optimizer_step", -1)) == int(step)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _preserve_progress(trainer: Any, tracker: BestTracker, spec: TaskSpec,
                       tokenizer: Any, step: int) -> bool:
    if tracker.persisted_best is not None:
        selected_step = int(tracker.persisted_best_step or 0)
        written = write_artifact_truth(
            spec.output_dir,
            ARTIFACT_PARTIAL_TRAINED_BEST,
            optimizer_step=selected_step,
            reason="progressed_oom_retained_best",
        )
        return written and _truth_matches(
            spec.output_dir, ARTIFACT_PARTIAL_TRAINED_BEST, selected_step
        )
    try:
        save_adapter(
            trainer.model, tokenizer, spec.output_dir,
            artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST,
            optimizer_step=step, truth_reason="progressed_oom_latest")
        return _truth_matches(
            spec.output_dir, ARTIFACT_PARTIAL_TRAINED_BEST, step
        )
    except Exception as exc:
        telemetry.event("progressed_oom_export_failed", error=repr(exc), step=step)
        return False


def _train_ladder(
    plans: tuple[TrainPlan, ...],
    data_for: Callable[[TrainPlan], _DataView],
    rebuild: Callable,
    make: Callable,
    measure_epochs: Callable[[TrainPlan, _DataView], float | None],
    spec: TaskSpec,
    tokenizer: Any,
):
    last_plan = None
    view = None
    current_cap = None
    for index, candidate in enumerate(plans):
        if current_cap is not None and candidate.max_seq_len != current_cap:
            view = None
            gc.collect()
        view = data_for(candidate)
        current_cap = candidate.max_seq_len
        if not view.authorized:
            telemetry.event(
                "training_geometry_attempt",
                status="CAP_AUTHORITY_REJECTED",
                **_plan_identity(candidate),
                **view.identity(),
            )
            continue
        if not view.train:
            telemetry.event(
                "training_geometry_attempt",
                status="NO_RETAINED_ROWS",
                **_plan_identity(candidate),
                **view.identity(),
            )
            continue
        last_plan = candidate
        trainer = None
        tracker = None
        route = None
        current = None
        try:
            epochs = measure_epochs(candidate, view)
            current = rebuild(candidate)
            trainer, tracker, route = make(candidate, current, view, None, epochs)
            current = None
            trainer.train()
            telemetry.event(
                "training_geometry_attempt",
                status="PASS",
                **_plan_identity(candidate),
                **view.identity(),
            )
            return trainer, tracker, route, candidate, view, "trained"
        except Exception as exc:
            if not _is_oom(exc):
                raise
            step = int(getattr(getattr(trainer, "state", None),
                               "global_step", 0) or 0)
            if step:
                preserved = _preserve_progress(
                    trainer, tracker, spec, tokenizer, step
                )
                status = (
                    "PROGRESSED_OOM_PRESERVED"
                    if preserved else "PROGRESSED_OOM_EXPORT_FAILED"
                )
                telemetry.event(
                    "training_geometry_attempt",
                    status=status,
                    optimizer_step=step,
                    **_plan_identity(candidate),
                    **view.identity(),
                )
                return (
                    None,
                    tracker,
                    route,
                    candidate,
                    view,
                    (
                        "progressed_oom_preserved"
                        if preserved else "progressed_oom_export_failed"
                    ),
                )
            telemetry.event(
                "training_geometry_attempt",
                status="ZERO_PROGRESS_OOM",
                **_plan_identity(candidate),
                **view.identity(),
            )
            if trainer is not None:
                holder = [trainer]
                trainer = None
                _discard(holder.pop(), trainer=True)
            else:
                holder = [current]
                current = None
                _discard(holder.pop())
    if last_plan is None:
        raise RuntimeError("training ladder contained no nonempty dataset view")
    view = None
    gc.collect()
    terminal_view = data_for(last_plan)
    return (
        None,
        None,
        None,
        last_plan,
        terminal_view,
        "zero_progress_exhausted",
    )


def _truth(step: int, max_steps: int) -> str:
    if step <= 0:
        return ARTIFACT_FLOOR
    return (ARTIFACT_COMPLETE_BEST if max_steps > 0 and step >= max_steps
            else ARTIFACT_PARTIAL_TRAINED_BEST)


def _split_for_eval(tokenized: list) -> tuple[list, list]:
    """Small held-out slice for eval-loss logging; empty on tiny datasets."""
    n = len(tokenized)
    if n < _EVAL_MIN_DATASET:
        return tokenized, []
    idx = list(range(n))
    random.Random(7).shuffle(idx)
    val_idx = set(idx[:_EVAL_VAL_ROWS])
    train = [ex for i, ex in enumerate(tokenized) if i not in val_idx]
    val = [tokenized[i] for i in sorted(val_idx)]
    return train, val
