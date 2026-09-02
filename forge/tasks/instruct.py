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
import math
import os
import random
from dataclasses import replace
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
    read_artifact_truth,
    save_adapter,
    should_final_save,
    time_aware_epochs,
    write_artifact_truth,
    workdir,
)
from forge.tuning.callbacks import DeadlineCallback
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
_GEOMETRY_BATCH_LADDER = (4, 2, 1)
_GEOMETRY_TARGET_EFFECTIVE_BATCH = 16
_GEOMETRY_MAX_RESERVED_FRACTION = 0.90


def run(spec: TaskSpec, deadline: Deadline) -> None:
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
    plan, qwen35_geometry_changed = (
        conservative_qwen35_plan(loaded.model, plan)
        if n_gpus == 0
        else (plan, False)
    )
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

    model_init_rng = _capture_random_state()
    if use_full:
        model = prepare_full_finetune(
            loaded.model, gradient_checkpointing=plan.gradient_checkpointing
        )
    else:
        model = attach_lora(
            loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
        )
    loaded.model = None
    del loaded

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
    else:
        save_adapter(
            model,
            tokenizer,
            spec.output_dir,
            artifact_truth=ARTIFACT_FLOOR,
            optimizer_step=0,
            truth_reason="pretraining_floor",
        )

    # Baseline stats are provenance-only for SFT length. Shrinking below G.O.D's
    # evaluator ladder changes which rows retain supervised completion tokens.
    initial_seq_len = effective_sft_seq_len(model, plan.max_seq_len)
    seq_candidates = tokenize.sft_sequence_len_candidates(
        model, tokenizer, initial_seq_len
    )
    if spec.chat is not None:
        conversations = prompts.build_chat_conversations(rows, spec.chat)
        template_resolution = tokenize.resolve_chat_template(
            spec.chat.chat_template, tokenizer
        )
        if template_resolution.degraded:
            # The valid floor already exists. Persist the compatibility choice
            # before processing rows so even a kill during tokenization explains
            # why native/ChatML semantics replaced the requested future name.
            telemetry.write_into(spec.output_dir)
        tokenized, seq_len = tokenize.first_nonempty_tokenization(
            seq_candidates,
            lambda candidate: tokenize.tokenize_chat_resolved(
                conversations,
                tokenizer,
                candidate,
                resolution=template_resolution,
            ),
        )
    else:
        assert spec.instruct is not None, "instruct task missing instruct columns"
        if spec.instruct.output is None:
            documents = prompts.build_completion_documents(rows, spec.instruct)
            seq_len = seq_candidates[0]
            tokenized = tokenize.tokenize_completion(documents, tokenizer, seq_len)
        else:
            examples = prompts.build_instruct_examples(rows, spec.instruct)
            tokenized, seq_len = tokenize.first_nonempty_tokenization(
                seq_candidates,
                lambda candidate: tokenize.tokenize_instruct(
                    examples, tokenizer, candidate
                ),
            )

    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")

    # Eval-loss selection holds out a small slice — but NOT on KL tasks, where
    # each eval reruns the full KL double-forward and would eat the time budget.
    train_ex, val_ex = (tokenized, []) if is_kl else _split_for_eval(tokenized)
    collator = tokenize.PadCollator(tokenizer.pad_token_id)

    def rebuild_trainable(candidate_plan: TrainPlan) -> Any:
        """Reload exact base bytes and reproduce the original trainable init."""
        rebuilt = load_base(spec.cached_model_dir, for_generation=False)
        base_model = rebuilt.model
        rebuilt.model = None
        del rebuilt
        _restore_random_state(model_init_rng)
        if use_full:
            return prepare_full_finetune(
                base_model,
                gradient_checkpointing=candidate_plan.gradient_checkpointing,
            )
        return attach_lora(
            base_model,
            r=candidate_plan.lora_r,
            alpha=candidate_plan.lora_alpha,
            dropout=candidate_plan.lora_dropout,
        )

    def build_probe_trainer(
        candidate_plan: TrainPlan, candidate_model: Any, examples: list
    ):
        return _make_sft_probe_trainer(
            spec=spec,
            plan=candidate_plan,
            model=candidate_model,
            train_ex=examples,
            collator=collator,
            is_kl=is_kl,
        )

    model_holder = [model]
    del model
    plan, geometry_attempts = _admit_sft_geometry(
        plan=plan,
        model=model_holder.pop(),
        train_ex=train_ex,
        rebuild_model=rebuild_trainable,
        build_probe_trainer=build_probe_trainer,
    )
    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        handler="chat" if spec.chat is not None else "instruct",
        strategy=strategy,
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        gpu_gb=per_gpu_gb,
        seq_len=seq_len,
        seq_len_candidates=seq_candidates,
        baseline_seq_policy="provenance_only",
        tokenized=len(tokenized),
        train_n=len(train_ex),
        val_n=len(val_ex),
        lora_r=plan.lora_r,
        lr=plan.learning_rate,
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=eff_batch,
        epochs=plan.num_epochs,
        neftune=not is_kl,
        tokens_per_step_cap=eff_batch * seq_len,
        geometry_admission=geometry_attempts,
    )

    timing_model = rebuild_trainable(plan)
    scheduled_epochs, probe_per_step = _time_sft_schedule(
        plan=plan,
        model=timing_model,
        train_ex=train_ex,
        collator=collator,
        deadline=deadline,
        strategy=strategy,
        is_kl=is_kl,
        kl_coef=spec.kl_coef,
        spec=spec,
    )
    if scheduled_epochs is not None:
        telemetry.event(
            "time_aware_epochs",
            epochs=scheduled_epochs,
            planned=plan.num_epochs,
            probe_per_step_s=round(probe_per_step, 4),
            batch=int(plan.per_device_batch_size),
            grad_accum=int(plan.grad_accum_steps),
        )

    def build_trainer(candidate_plan: TrainPlan, candidate_model: Any):
        return _make_sft_trainer(
            spec=spec,
            deadline=deadline,
            plan=candidate_plan,
            model=candidate_model,
            tokenizer=tokenizer,
            train_ex=train_ex,
            val_ex=val_ex,
            collator=collator,
            is_kl=is_kl,
            strategy=strategy,
            n_gpus=n_gpus,
            scheduled_epochs=scheduled_epochs,
        )

    model = rebuild_trainable(plan)
    model_holder = [model]
    del model
    trainer, tracker, soup_route, plan = _train_sft_geometry_ladder(
        initial_plan=plan,
        initial_model=model_holder.pop(),
        build_trainer=build_trainer,
        rebuild_model=rebuild_trainable,
        spec=spec,
        tokenizer=tokenizer,
    )
    model = trainer.model
    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=eff_batch,
    )
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    terminal_truth = _completed_artifact_truth(final_step, max_steps)
    if terminal_truth == ARTIFACT_FLOOR:
        write_artifact_truth(
            spec.output_dir,
            ARTIFACT_FLOOR,
            optimizer_step=0,
            reason="normal_return_without_optimizer_progress",
        )
        telemetry.event("training_returned_without_optimizer_progress")
        return
    selected_artifact_step = final_step
    completed = terminal_truth == ARTIFACT_COMPLETE_BEST
    phase = "completed_schedule" if completed else "deadline_cut"
    if should_final_save(tracker, final_step=final_step):
        save_adapter(
            model,
            tokenizer,
            spec.output_dir,
            artifact_truth=terminal_truth,
            optimizer_step=final_step,
            truth_reason=f"{phase}_final_is_best",
        )
    else:
        telemetry.event(
            "kept_best_checkpoint",
            best=round(tracker.best, 5),
            best_step=tracker.best_step,
            last_eval=round(tracker.last, 5),
            last_eval_step=tracker.last_step,
            final_step=final_step,
        )
        write_artifact_truth(
            spec.output_dir,
            terminal_truth,
            optimizer_step=int(tracker.best_step or 0),
            reason=f"{phase}_retained_eval_minimum",
        )
        selected_artifact_step = int(tracker.best_step or 0)
    apply_qwen35_soup_override(
        soup_route,
        tracker=tracker,
        model=model,
        tokenizer=tokenizer,
    )
    write_artifact_truth(
        spec.output_dir,
        terminal_truth,
        optimizer_step=selected_artifact_step,
        reason=f"{phase}_selected_export",
    )


def _completed_artifact_truth(final_step: int, max_steps: int) -> str:
    """Label only a genuinely exhausted Trainer schedule as complete."""
    step = int(final_step)
    if step <= 0:
        return ARTIFACT_FLOOR
    if int(max_steps) > 0 and step >= int(max_steps):
        return ARTIFACT_COMPLETE_BEST
    return ARTIFACT_PARTIAL_TRAINED_BEST


def _capture_random_state() -> dict[str, Any]:
    """Snapshot adapter-initialization RNG without imposing a new seed."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch

        state["torch_cpu"] = torch.random.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    except Exception:
        pass
    return state


def _restore_random_state(state: dict[str, Any]) -> None:
    """Reproduce the first trainable model rather than reseeding the recipe."""
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch

        torch.random.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _geometry_plans(plan: TrainPlan) -> tuple[TrainPlan, ...]:
    """Return only the 4 -> 2 -> 1 ladder at effective batch 16."""
    start = min(4, max(1, int(plan.per_device_batch_size)))
    batches = [batch for batch in _GEOMETRY_BATCH_LADDER if batch <= start]
    return tuple(
        replace(
            plan,
            per_device_batch_size=batch,
            grad_accum_steps=math.ceil(_GEOMETRY_TARGET_EFFECTIVE_BATCH / batch),
        )
        for batch in batches
    )


def _memory_probe_rows(
    train_ex: list, *, batch_size: int
) -> list[tuple[str, list, dict[str, Any]]]:
    """Choose actual p99 and worst padded batches, deterministically."""
    if not train_ex:
        raise ValueError("memory admission requires non-empty training examples")
    size = max(1, min(int(batch_size), len(train_ex)))
    lengths = [len(row.get("input_ids", [])) for row in train_ex]
    ranked = sorted(range(len(train_ex)), key=lambda i: (lengths[i], i))
    p99_rank = min(len(ranked) - 1, max(0, math.ceil(0.99 * len(ranked)) - 1))
    p99_length = lengths[ranked[p99_rank]]
    p99_indices = sorted(
        range(len(train_ex)),
        key=lambda i: (abs(lengths[i] - p99_length), -lengths[i], i),
    )[:size]
    worst_indices = sorted(
        range(len(train_ex)), key=lambda i: (-lengths[i], i)
    )[:size]

    def selected(label: str, indices: list[int]) -> tuple[str, list, dict[str, Any]]:
        return (
            label,
            [train_ex[index] for index in indices],
            {
                "row_indices": indices,
                "row_lengths": [lengths[index] for index in indices],
            },
        )

    return [selected("p99", p99_indices), selected("worst", worst_indices)]


def _model_cuda_devices(model: Any) -> tuple[Any, ...]:
    """All CUDA devices carrying parameters, in stable index order."""
    import torch

    indices = set(range(int(torch.cuda.device_count())))
    indices.update(
        {
            int(parameter.device.index)
            if parameter.device.index is not None
            else int(torch.cuda.current_device())
            for parameter in model.parameters()
            if getattr(parameter, "device", None) is not None
            and parameter.device.type == "cuda"
        }
    )
    if not indices:
        indices.add(int(torch.cuda.current_device()))
    return tuple(torch.device("cuda", index) for index in sorted(indices))


def _measure_sft_geometry(
    *,
    model: Any,
    examples: list,
    identity: dict[str, Any],
    plan: TrainPlan,
    build_probe_trainer: Callable[[TrainPlan, Any, list], Any],
) -> dict[str, Any]:
    """Measure one exact Trainer step and discard its complete generation."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("exact CUDA geometry measurement is unavailable")
    devices = _model_cuda_devices(model)
    trainer = None
    try:
        torch.cuda.empty_cache()
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        trainer = build_probe_trainer(plan, model, examples)
        model = None
        trainer.train()
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        if step != 1:
            raise RuntimeError(f"exact geometry probe completed {step} optimizer steps")
        for device in devices:
            torch.cuda.synchronize(device)
        device_peaks = []
        for device in devices:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            reserved = int(torch.cuda.max_memory_reserved(device))
            allocated = int(torch.cuda.max_memory_allocated(device))
            device_peaks.append(
                {
                    "device": str(device),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "total_memory_bytes": total,
                    "peak_reserved_fraction": round(
                        (reserved / total) if total else 1.0, 6
                    ),
                }
            )
        peak = max(device_peaks, key=lambda item: item["peak_reserved_fraction"])
        fraction = peak["peak_reserved_fraction"]
        return {
            **identity,
            "status": (
                "PASS"
                if fraction <= _GEOMETRY_MAX_RESERVED_FRACTION
                else "HEADROOM_EXCEEDED"
            ),
            "optimizer": str(plan.optimizer),
            "peak_allocated_bytes": peak["peak_allocated_bytes"],
            "peak_reserved_bytes": peak["peak_reserved_bytes"],
            "total_memory_bytes": peak["total_memory_bytes"],
            "peak_reserved_fraction": fraction,
            "device_peaks": device_peaks,
        }
    except Exception as exc:
        if not _is_oom(exc):
            raise RuntimeError(
                "exact production geometry measurement could not be established"
            ) from exc
        return {
            **identity,
            "status": "OOM",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if trainer is not None:
            holder = [trainer]
            trainer = None
            _discard_trainer(holder.pop())
        elif model is not None:
            holder = [model]
            model = None
            _discard_model(holder.pop())


def _discard_model(model: Any) -> None:
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    del model
    gc.collect()
    _free_cuda()


def _admit_sft_geometry(
    *,
    plan: TrainPlan,
    model: Any,
    train_ex: list,
    rebuild_model: Callable[[TrainPlan], Any],
    build_probe_trainer: Callable[[TrainPlan, Any, list], Any],
) -> tuple[TrainPlan, list[dict[str, Any]]]:
    """Select geometry using exact one-step production Trainer probes."""
    if not _cuda_available():
        observation = {
            "status": "SKIP_NO_CUDA",
            "batch": int(plan.per_device_batch_size),
            "grad_accum": int(plan.grad_accum_steps),
        }
        telemetry.event("geometry_admission_skipped", **observation)
        _discard_model(model)
        return plan, [observation]

    attempts: list[dict[str, Any]] = []
    current_model = model
    model = None
    for candidate_index, candidate in enumerate(_geometry_plans(plan)):
        observations = []
        passed = True
        for batch_index, (label, examples, identity) in enumerate(
            _memory_probe_rows(train_ex, batch_size=candidate.per_device_batch_size)
        ):
            if candidate_index or batch_index:
                current_model = rebuild_model(candidate)
            holder = [current_model]
            current_model = None
            measured = _measure_sft_geometry(
                model=holder.pop(),
                examples=examples,
                identity={"label": label, **identity},
                plan=candidate,
                build_probe_trainer=build_probe_trainer,
            )
            observations.append(measured)
            if measured["status"] != "PASS":
                passed = False
                break
        fractions = [
            item["peak_reserved_fraction"]
            for item in observations
            if "peak_reserved_fraction" in item
        ]
        attempt = {
            "batch": int(candidate.per_device_batch_size),
            "grad_accum": int(candidate.grad_accum_steps),
            "effective_batch": int(
                candidate.per_device_batch_size * candidate.grad_accum_steps
            ),
            "status": "PASS" if passed else observations[-1]["status"],
            "max_reserved_fraction": max(fractions) if fractions else None,
            "limit": _GEOMETRY_MAX_RESERVED_FRACTION,
            "batches": observations,
        }
        attempts.append(attempt)
        telemetry.event("geometry_admission_attempt", **attempt)
        if passed:
            telemetry.event(
                "geometry_admission_selected",
                batch=int(candidate.per_device_batch_size),
                grad_accum=int(candidate.grad_accum_steps),
                effective_batch=int(
                    candidate.per_device_batch_size * candidate.grad_accum_steps
                ),
                attempts=len(attempts),
            )
            return candidate, attempts
    telemetry.event("geometry_admission_failed", attempts=attempts)
    raise RuntimeError("no measured SFT geometry passed p99/worst memory admission")


def _make_sft_probe_trainer(
    *,
    spec: TaskSpec,
    plan: TrainPlan,
    model: Any,
    train_ex: list,
    collator: Any,
    is_kl: bool,
) -> Any:
    """Build the production Trainer stack for one throwaway optimizer step."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    kwargs["max_steps"] = 1
    args = TrainingArguments(
        **compatible_dataclass_kwargs(
            TrainingArguments,
            kwargs,
            allow_removed={"overwrite_output_dir"},
        )
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_ex),
        data_collator=collator,
        callbacks=[],
    )
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer

        return KLSFTTrainer(kl_coef=spec.kl_coef, **trainer_kwargs)
    return Trainer(**trainer_kwargs)


def _time_sft_schedule(
    *,
    plan: TrainPlan,
    model: Any,
    train_ex: list,
    collator: Any,
    deadline: Deadline,
    strategy: str,
    is_kl: bool,
    kl_coef: float,
    spec: TaskSpec,
) -> tuple[float | None, float | None]:
    """Consume a separate model generation for the existing timing probe."""
    from transformers import Trainer

    trainer_cls: Any = Trainer
    trainer_extra = None
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer

        trainer_cls = KLSFTTrainer
        trainer_extra = {"kl_coef": kl_coef}
    try:
        return time_aware_epochs(
            trainer_cls=trainer_cls,
            model=model,
            kwargs=build_training_kwargs(
                spec, plan, neftune_alpha=None if is_kl else 5.0
            ),
            train_ex=train_ex,
            collator=collator,
            deadline=deadline,
            eff_batch=plan.per_device_batch_size * plan.grad_accum_steps,
            strategy=strategy,
            trainer_extra=trainer_extra,
        )
    finally:
        holder = [model]
        model = None
        _discard_model(holder.pop())


def _make_sft_trainer(
    *,
    spec: TaskSpec,
    deadline: Deadline,
    plan: TrainPlan,
    model: Any,
    tokenizer: Any,
    train_ex: list,
    val_ex: list,
    collator: Any,
    is_kl: bool,
    strategy: str,
    n_gpus: int,
    scheduled_epochs: float | None,
) -> tuple[Any, BestTracker, Any]:
    """Construct the existing SFT Trainer/callback policy at one geometry."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    if scheduled_epochs is not None:
        kwargs["num_train_epochs"] = scheduled_epochs

    tracker = BestTracker()
    soup_route = eligible_qwen35_soup_route(
        spec,
        model,
        strategy=strategy,
        n_gpus=n_gpus,
        capture_root=os.path.join(workdir(spec), "qwen35-r4-r2-captures"),
    )
    if val_ex:
        steps_per_epoch = max(1, len(train_ex) // eff_batch)
        kwargs.update(
            eval_strategy="steps",
            eval_steps=max(1, steps_per_epoch // 4),
            per_device_eval_batch_size=max(1, plan.per_device_batch_size),
        )
    args = TrainingArguments(
        **compatible_dataclass_kwargs(
            TrainingArguments,
            kwargs,
            allow_removed={"overwrite_output_dir"},
        )
    )
    mirror_every = 100 if strategy == "full" else 25
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(
            spec,
            tokenizer,
            every=mirror_every,
            tracker=tracker,
            label_artifact_truth=True,
        ),
        telemetry.make_trainer_callback(spec.output_dir),
    ]
    if val_ex:
        # Existing always-best policy: annotate it, do not replace it.
        callbacks.append(
            _make_best_checkpoint_callback(
                spec, tokenizer, tracker, label_artifact_truth=True
            )
        )
        if soup_route is not None:
            callbacks.append(make_qwen35_soup_capture_callback(soup_route, tokenizer))
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_ex),
        eval_dataset=Dataset.from_list(val_ex) if val_ex else None,
        data_collator=collator,
        callbacks=callbacks,
    )
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer

        trainer = KLSFTTrainer(kl_coef=spec.kl_coef, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)
    return trainer, tracker, soup_route


def _discard_trainer(trainer: Any) -> None:
    """Release all model/Accelerator references before pristine reconstruction."""
    try:
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "free_memory"):
            accelerator.free_memory()
    except Exception:
        pass
    for name in ("optimizer", "lr_scheduler", "model_wrapped", "model"):
        try:
            setattr(trainer, name, None)
        except Exception:
            pass
    del trainer
    gc.collect()
    _free_cuda()


def _preserve_progressed_oom(
    *, trainer: Any, tracker: BestTracker, spec: TaskSpec, tokenizer: Any, step: int
) -> None:
    """Keep the persisted eval-best, or save progressed weights before floor."""
    preserved = False
    if tracker.persisted_best is not None:
        preserved = write_artifact_truth(
            spec.output_dir,
            ARTIFACT_PARTIAL_TRAINED_BEST,
            optimizer_step=int(tracker.persisted_best_step or 0),
            reason=f"progressed_oom_at_step_{step}_retained_eval_minimum",
        )
    else:
        existing = read_artifact_truth(spec.output_dir)
        if (
            existing is not None
            and existing["truth"] == ARTIFACT_PARTIAL_TRAINED_BEST
            and int(existing["optimizer_step"]) > 0
        ):
            preserved = write_artifact_truth(
                spec.output_dir,
                ARTIFACT_PARTIAL_TRAINED_BEST,
                optimizer_step=int(existing["optimizer_step"]),
                reason=f"progressed_oom_at_step_{step}_retained_periodic_export",
            )
        model = getattr(trainer, "model", None)
        if not preserved and model is not None:
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            try:
                setattr(trainer, "optimizer", None)
                setattr(trainer, "lr_scheduler", None)
            except Exception:
                pass
            _free_cuda()
            try:
                save_adapter(
                    model,
                    tokenizer,
                    spec.output_dir,
                    artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST,
                    optimizer_step=step,
                    truth_reason="progressed_oom_saved_latest_valid_weights",
                )
                preserved = True
            except Exception as exc:
                telemetry.event(
                    "progressed_oom_export_failed",
                    step=step,
                    error=f"{type(exc).__name__}: {exc}",
                )
    telemetry.event(
        "progressed_oom_preserved",
        step=step,
        preserved=preserved,
        floor_last_resort=not preserved,
    )


def _train_sft_geometry_ladder(
    *,
    initial_plan: TrainPlan,
    initial_model: Any,
    build_trainer: Callable[[TrainPlan, Any], tuple[Any, BestTracker, Any]],
    rebuild_model: Callable[[TrainPlan], Any],
    spec: TaskSpec,
    tokenizer: Any,
) -> tuple[Any, BestTracker, Any, TrainPlan]:
    """Train, rebuilding only after a genuine zero-progress OOM."""
    plans = _geometry_plans(initial_plan)
    current_model = initial_model
    initial_model = None
    for index, candidate in enumerate(plans):
        trainer = None
        tracker = None
        soup_route = None
        build_oom = False
        try:
            trainer, tracker, soup_route = build_trainer(candidate, current_model)
            current_model = None
        except Exception as exc:
            if not _is_oom(exc):
                raise
            build_oom = True

        if not build_oom:
            failure: tuple[int, str] | None = None
            try:
                trainer.train()
                return trainer, tracker, soup_route, candidate
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
                failure = (step, f"{type(exc).__name__}: {exc}")
            assert failure is not None
            step, error = failure
            if step > 0:
                _preserve_progressed_oom(
                    trainer=trainer,
                    tracker=tracker,
                    spec=spec,
                    tokenizer=tokenizer,
                    step=step,
                )
                raise RuntimeError(
                    f"training OOM after optimizer progress at step {step}; "
                    "preserved last valid artifact"
                ) from None
            _discard_trainer(trainer)
        else:
            step, error = 0, "OOM during Trainer construction"
            discard_holder = [current_model]
            current_model = None
            _discard_model(discard_holder.pop())

        if index + 1 >= len(plans):
            telemetry.event(
                "zero_progress_oom_exhausted",
                batch=int(candidate.per_device_batch_size),
                grad_accum=int(candidate.grad_accum_steps),
                error=error,
            )
            raise RuntimeError(
                "zero-progress OOM exhausted measured 4->2->1 geometry ladder"
            ) from None
        next_plan = plans[index + 1]
        telemetry.event(
            "zero_progress_oom_rebuild",
            from_batch=int(candidate.per_device_batch_size),
            to_batch=int(next_plan.per_device_batch_size),
            to_grad_accum=int(next_plan.grad_accum_steps),
            effective_batch=int(
                next_plan.per_device_batch_size * next_plan.grad_accum_steps
            ),
            error=error,
        )
        current_model = rebuild_model(next_plan)
    raise AssertionError("unreachable geometry ladder exit")


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
