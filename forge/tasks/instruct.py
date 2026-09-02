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
    # On CUDA the measured real-batch admission below supersedes the old
    # Qwen-only formula. CPU has no training geometry to measure, so retain the
    # historical conservative route there for compatibility.
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
    # `model` now owns the trainable wrapper/base. Drop the LoadedModel's second
    # reference so a failed memory probe can genuinely release this generation.
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

    # Give admission sole ownership of the model so failed geometry really
    # releases it before the next pristine reconstruction.
    model_holder = [model]
    del model
    plan, model, geometry_attempts = _admit_sft_geometry(
        plan=plan,
        model=model_holder.pop(),
        train_ex=train_ex,
        collator=collator,
        rebuild_model=rebuild_trainable,
        is_kl=is_kl,
        kl_coef=spec.kl_coef,
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
        )

    # The actual train path uses the admitted pristine reconstruction. A rare
    # step-zero OOM discards the complete Trainer/model and rebuilds at the next
    # measured ladder rung; it never mutates Accelerator geometry in place.
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
    if _completed_artifact_truth(final_step) == ARTIFACT_FLOOR:
        # A deadline can stop cleanly during the first accumulation window.
        # That is not trained progress: keep the already-valid floor generation.
        write_artifact_truth(
            spec.output_dir,
            ARTIFACT_FLOOR,
            optimizer_step=0,
            reason="normal_return_without_optimizer_progress",
        )
        telemetry.event("training_returned_without_optimizer_progress")
        return
    selected_artifact_step = final_step
    if should_final_save(tracker, final_step=final_step):
        save_adapter(
            model,
            tokenizer,
            spec.output_dir,
            artifact_truth=ARTIFACT_COMPLETE_BEST,
            optimizer_step=final_step,
            truth_reason="completed_schedule_final_is_best",
        )
    else:
        # The exported best checkpoint is strictly better than final weights;
        # leave it in place and record why.
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
            ARTIFACT_COMPLETE_BEST,
            optimizer_step=int(tracker.best_step or 0),
            reason="completed_schedule_retained_eval_minimum",
        )
        selected_artifact_step = int(tracker.best_step or 0)
    apply_qwen35_soup_override(
        soup_route,
        tracker=tracker,
        model=model,
        tokenizer=tokenizer,
    )
    # The optional fixed soup promotion replaces a directory generation. State
    # the terminal truth again after it, without changing its existing policy.
    write_artifact_truth(
        spec.output_dir,
        ARTIFACT_COMPLETE_BEST,
        optimizer_step=selected_artifact_step,
        reason="completed_schedule_selected_export",
    )


def _completed_artifact_truth(final_step: int) -> str:
    """A normal Trainer return is complete only after optimizer progress."""
    return ARTIFACT_COMPLETE_BEST if int(final_step) > 0 else ARTIFACT_FLOOR


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
    """Return the 4 -> 2 -> 1 ladder while preserving effective batch >=16."""
    start = max(1, int(plan.per_device_batch_size))
    original_effective = start * max(1, int(plan.grad_accum_steps))
    target = max(_GEOMETRY_TARGET_EFFECTIVE_BATCH, original_effective)
    batches = [batch for batch in _GEOMETRY_BATCH_LADDER if batch <= start]
    if start not in batches:
        batches.insert(0, start)
    return tuple(
        replace(
            plan,
            per_device_batch_size=batch,
            grad_accum_steps=max(
                int(plan.grad_accum_steps), math.ceil(target / batch)
            ),
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


def _model_input_device(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        device = getattr(parameter, "device", None)
        if device is not None and device.type == "cuda":
            return device
    return torch.device("cuda", torch.cuda.current_device())


def _model_cuda_devices(model: Any) -> tuple[Any, ...]:
    """All CUDA devices carrying parameters, in stable index order."""
    import torch

    indices = {
        int(parameter.device.index)
        if parameter.device.index is not None
        else int(torch.cuda.current_device())
        for parameter in model.parameters()
        if getattr(parameter, "device", None) is not None
        and parameter.device.type == "cuda"
    }
    if not indices:
        indices.add(int(torch.cuda.current_device()))
    return tuple(torch.device("cuda", index) for index in sorted(indices))


def _enable_probe_gradient_checkpointing(model: Any, plan: TrainPlan) -> None:
    """Mirror Trainer's checkpointing mode before measuring activations."""
    if not plan.gradient_checkpointing:
        return
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        return
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        # Older/custom model implementations expose the no-argument form.
        enable()


def _probe_sft_loss(
    model: Any,
    batch: dict[str, Any],
    output: Any,
    *,
    is_kl: bool,
    kl_coef: float,
) -> Any:
    """Use the real policy/reference KL memory path when the task requires it."""
    loss = getattr(output, "loss", None)
    if loss is None or not is_kl:
        return loss
    # Reuse the exact existing implementation. Its method is state-free: the
    # Trainer instance is not needed for the adapter-disabled reference forward
    # or chunked KL graph.
    from forge.tuning.kl import KLSFTTrainer

    kl_sum, kl_tokens = KLSFTTrainer._completion_kl_sum(
        None, model, batch, output.logits, batch["labels"]
    )
    if kl_tokens:
        loss = loss + float(kl_coef) * (kl_sum / float(kl_tokens))
    return loss


def _measure_sft_geometry(
    *,
    model: Any,
    train_ex: list,
    collator: Any,
    plan: TrainPlan,
    is_kl: bool = False,
    kl_coef: float = 0.0,
) -> dict[str, Any]:
    """Measure real p99/worst train steps, including optimizer-state creation."""
    import torch

    if not torch.cuda.is_available():
        return {"status": "SKIP_NO_CUDA", "batches": []}
    input_device = _model_input_device(model)
    devices = _model_cuda_devices(model)
    observations: list[dict[str, Any]] = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("memory admission found no trainable parameters")
    optimizer_kwargs: dict[str, Any] = {"lr": 0.0, "weight_decay": 0.0}
    if plan.optimizer == "adamw_torch_fused":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    _enable_probe_gradient_checkpointing(model, plan)
    model.train()
    for label, examples, identity in _memory_probe_rows(
        train_ex, batch_size=plan.per_device_batch_size
    ):
        batch = None
        output = None
        loss = None
        try:
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            batch = collator(examples)
            padded_tokens = int(batch["input_ids"].numel())
            batch = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            if plan.bf16:
                autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif plan.fp16:
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                from contextlib import nullcontext

                autocast = nullcontext()
            with autocast:
                output = model(**batch)
                loss = _probe_sft_loss(
                    model,
                    batch,
                    output,
                    is_kl=is_kl,
                    kl_coef=kl_coef,
                )
            if loss is None or not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError("memory admission produced no finite loss")
            loss.backward()
            # lr=0 keeps weights pristine while the exact AdamW state footprint
            # is allocated. The whole measured model is discarded afterward.
            optimizer.step()
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
            observations.append(
                {
                    "label": label,
                    **identity,
                    "padded_tokens": padded_tokens,
                    "optimizer": str(plan.optimizer),
                    "peak_allocated_bytes": peak["peak_allocated_bytes"],
                    "peak_reserved_bytes": peak["peak_reserved_bytes"],
                    "total_memory_bytes": peak["total_memory_bytes"],
                    "peak_reserved_fraction": peak["peak_reserved_fraction"],
                    "device_peaks": device_peaks,
                }
            )
        except Exception as exc:
            if not _is_oom(exc):
                raise
            observations.append(
                {
                    "label": label,
                    **identity,
                    "status": "OOM",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return {"status": "OOM", "batches": observations}
        finally:
            try:
                model.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            del loss, output, batch
            _free_cuda()
    del optimizer, trainable
    _free_cuda()
    highest = max(item["peak_reserved_fraction"] for item in observations)
    status = (
        "PASS"
        if highest <= _GEOMETRY_MAX_RESERVED_FRACTION
        else "HEADROOM_EXCEEDED"
    )
    return {
        "status": status,
        "max_reserved_fraction": round(highest, 6),
        "limit": _GEOMETRY_MAX_RESERVED_FRACTION,
        "batches": observations,
    }


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
    collator: Any,
    rebuild_model: Callable[[TrainPlan], Any],
    is_kl: bool = False,
    kl_coef: float = 0.0,
) -> tuple[TrainPlan, Any, list[dict[str, Any]]]:
    """Measure before Trainer construction and return a pristine selected model."""
    if not _cuda_available():
        observation = {
            "status": "SKIP_NO_CUDA",
            "batch": int(plan.per_device_batch_size),
            "grad_accum": int(plan.grad_accum_steps),
        }
        telemetry.event("geometry_admission_skipped", **observation)
        return plan, model, [observation]

    attempts: list[dict[str, Any]] = []
    current_model = model
    # The parameter itself is a strong reference. Clear it before any discard
    # so a failed probe can actually release this generation before rebuilding.
    model = None
    for index, candidate in enumerate(_geometry_plans(plan)):
        if index > 0:
            current_model = rebuild_model(candidate)
        try:
            measured = _measure_sft_geometry(
                model=current_model,
                train_ex=train_ex,
                collator=collator,
                plan=candidate,
                is_kl=is_kl,
                kl_coef=kl_coef,
            )
        finally:
            # Remove the caller alias before the discard helper collects CUDA
            # storage; otherwise its own gc runs while this frame still pins it.
            discard_holder = [current_model]
            current_model = None
            _discard_model(discard_holder.pop())
        attempt = {
            "batch": int(candidate.per_device_batch_size),
            "grad_accum": int(candidate.grad_accum_steps),
            "effective_batch": int(
                candidate.per_device_batch_size * candidate.grad_accum_steps
            ),
            **measured,
        }
        attempts.append(attempt)
        telemetry.event("geometry_admission_attempt", **attempt)
        if measured["status"] == "PASS":
            # The measured model has seen backward passes. Reopen base bytes and
            # reproduce the original trainable initialization for real training.
            selected_model = rebuild_model(candidate)
            telemetry.event(
                "geometry_admission_selected",
                batch=int(candidate.per_device_batch_size),
                grad_accum=int(candidate.grad_accum_steps),
                effective_batch=int(
                    candidate.per_device_batch_size * candidate.grad_accum_steps
                ),
                attempts=len(attempts),
            )
            return candidate, selected_model, attempts
    telemetry.event("geometry_admission_failed", attempts=attempts)
    raise RuntimeError("no measured SFT geometry passed p99/worst memory admission")


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
) -> tuple[Any, BestTracker, Any]:
    """Construct the existing SFT Trainer/callback policy at one geometry."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer as probe_cls

        probe_extra = {"kl_coef": spec.kl_coef}
    else:
        probe_cls = Trainer
        probe_extra = None
    ta_epochs, probe_per_step = time_aware_epochs(
        trainer_cls=probe_cls,
        model=model,
        kwargs=kwargs,
        train_ex=train_ex,
        collator=collator,
        deadline=deadline,
        eff_batch=eff_batch,
        strategy=strategy,
        trainer_extra=probe_extra,
    )
    if ta_epochs is not None:
        kwargs["num_train_epochs"] = ta_epochs
        telemetry.event(
            "time_aware_epochs",
            epochs=ta_epochs,
            planned=plan.num_epochs,
            probe_per_step_s=round(probe_per_step, 4),
            batch=int(plan.per_device_batch_size),
            grad_accum=int(plan.grad_accum_steps),
        )

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
    # As in admission, do not let the function parameter pin a discarded base
    # while the next geometry reloads another complete generation.
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
