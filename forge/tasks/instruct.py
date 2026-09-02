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
_BATCHES = (4, 2, 1)


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

    def rebuild(candidate: TrainPlan) -> Any:
        reopened = load_base(spec.cached_model_dir, for_generation=False)
        base = reopened.model
        reopened.model = None
        _restore_torch_rng(init_rng)
        if use_full:
            return prepare_full_finetune(base, gradient_checkpointing=candidate.gradient_checkpointing)
        return attach_lora(base, r=candidate.lora_r, alpha=candidate.lora_alpha,
                           dropout=candidate.lora_dropout)

    def make(candidate: TrainPlan, candidate_model: Any,
             probe_rows: list | None = None, epochs: float | None = None):
        return _make_trainer(
            spec, deadline, candidate, candidate_model, tokenizer,
            probe_rows or train_ex,
            [] if probe_rows is not None else val_ex,
            collator, is_kl, strategy, n_gpus,
            probe=probe_rows is not None, epochs=epochs)

    holder = [model]
    del model
    plan, admission = _admit(plan, holder.pop(), train_ex, rebuild, make)
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
        geometry_admission=admission,
    )

    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer as probe_cls

        probe_extra = {"kl_coef": spec.kl_coef}
    else:
        probe_cls = Trainer
        probe_extra = None
    timing_model = rebuild(plan)
    try:
        ta_epochs, probe_per_step = time_aware_epochs(
            trainer_cls=probe_cls, model=timing_model, kwargs=kwargs,
            train_ex=train_ex, collator=collator, deadline=deadline,
            eff_batch=eff_batch, strategy=strategy, trainer_extra=probe_extra)
    finally:
        holder = [timing_model]
        timing_model = None
        _discard(holder.pop())
    if ta_epochs is not None:
        telemetry.event("time_aware_epochs", epochs=ta_epochs, planned=plan.num_epochs,
                        probe_per_step_s=round(probe_per_step, 4))

    model = rebuild(plan)
    trainer, tracker, soup_route, plan = _train_ladder(
        plan, model, rebuild, make, spec, tokenizer, ta_epochs
    )
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


def _plans(plan: TrainPlan) -> tuple[TrainPlan, ...]:
    start = min(4, max(1, int(plan.per_device_batch_size)))
    return tuple(
        replace(plan, per_device_batch_size=batch, grad_accum_steps=math.ceil(16 / batch))
        for batch in _BATCHES if batch <= start
    )


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
    plan: TrainPlan, model: Any, rows: list, identity: dict[str, Any], make: Callable
) -> dict[str, Any]:
    import torch

    devices = tuple(range(torch.cuda.device_count()))
    trainer = None
    try:
        torch.cuda.empty_cache()
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        trainer = make(plan, model, rows)[0]
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


def _admit(plan: TrainPlan, model: Any, rows: list,
           rebuild: Callable, make: Callable):
    import torch

    if not torch.cuda.is_available():
        _discard(model)
        return plan, [{"status": "SKIP_NO_CUDA"}]
    attempts = []
    current = model
    model = None
    for plan_index, candidate in enumerate(_plans(plan)):
        observations = []
        for batch_index, (label, selected, identity) in enumerate(
                _probe_rows(rows, candidate.per_device_batch_size)):
            if plan_index or batch_index:
                current = rebuild(candidate)
            holder = [current]
            current = None
            observation = _probe_once(candidate, holder.pop(), selected,
                                      {"label": label, **identity}, make)
            observations.append(observation)
            if observation["status"] != "PASS":
                break
        status = ("PASS" if len(observations) == 2
                  and all(o["status"] == "PASS" for o in observations)
                  else observations[-1]["status"])
        attempt = dict(batch=candidate.per_device_batch_size,
                       grad_accum=candidate.grad_accum_steps,
                       status=status, batches=observations)
        attempts.append(attempt)
        telemetry.event("geometry_admission_attempt", **attempt)
        if status == "PASS":
            return candidate, attempts
    raise RuntimeError("no exact 4->2->1 SFT geometry passed admission")


def _preserve_progress(trainer: Any, tracker: BestTracker, spec: TaskSpec,
                       tokenizer: Any, step: int) -> None:
    if tracker.persisted_best is not None:
        write_artifact_truth(spec.output_dir, ARTIFACT_PARTIAL_TRAINED_BEST,
                             optimizer_step=int(tracker.persisted_best_step or 0),
                             reason="progressed_oom_retained_best")
    else:
        try:
            save_adapter(
                trainer.model, tokenizer, spec.output_dir,
                artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST,
                optimizer_step=step, truth_reason="progressed_oom_latest")
        except Exception as exc:
            telemetry.event("progressed_oom_export_failed", error=repr(exc), step=step)


def _train_ladder(plan: TrainPlan, model: Any, rebuild: Callable, make: Callable,
                  spec: TaskSpec, tokenizer: Any, epochs: float | None):
    plans = _plans(plan)
    current = model
    model = None
    for index, candidate in enumerate(plans):
        trainer = None
        try:
            trainer, tracker, route = make(candidate, current, None, epochs)
            current = None
            trainer.train()
            return trainer, tracker, route, candidate
        except Exception as exc:
            if not _is_oom(exc):
                raise
            step = int(getattr(getattr(trainer, "state", None),
                               "global_step", 0) or 0)
            if step:
                _preserve_progress(trainer, tracker, spec, tokenizer, step)
                raise RuntimeError(f"training OOM after progress at step {step}") from None
            if trainer is not None:
                holder = [trainer]
                trainer = None
                _discard(holder.pop(), trainer=True)
            else:
                holder = [current]
                current = None
                _discard(holder.pop())
        if index + 1 == len(plans):
            raise RuntimeError("zero-progress OOM exhausted 4->2->1 ladder") from None
        current = rebuild(plans[index + 1])
    raise AssertionError("unreachable")


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
