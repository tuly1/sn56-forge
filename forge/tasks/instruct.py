"""Supervised fine-tuning for InstructTextTask and ChatTask.

Strategy defaults to the validated LoRA path.  Experimental full fine-tuning is
fail-closed behind an explicit environment opt-in and remains subject to KL,
topology, and memory-fit gates; it must be GPU-certified before deployment.

Loss is completion-only, matching the evaluator. We also hold out a small slice
and durably export the best measured checkpoint; later unevaluated weights never
replace that measured minimum.
"""

from __future__ import annotations

import random

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
    conservative_quasar_plan,
    decide_full_finetune,
    effective_sft_seq_len,
    gpu_topology,
    load_base,
    model_param_billions,
    prepare_full_finetune,
)
from forge.tasks.common import (
    BestTracker,
    _make_best_checkpoint_callback,
    _make_periodic_save_callback,
    build_training_kwargs,
    compatible_dataclass_kwargs,
    safe_train,
    save_adapter,
    should_final_save,
    time_aware_epochs,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_sft_plan

# Hold out a small fixed validation slice for measured checkpoint selection.
# Skip it on datasets too small to spare rows.
_EVAL_VAL_ROWS = 256
_EVAL_MIN_DATASET = 1000


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

    if use_full:
        model = prepare_full_finetune(
            loaded.model, gradient_checkpointing=plan.gradient_checkpointing
        )
    else:
        model = attach_lora(
            loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
        )

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
        save_adapter(model, tokenizer, spec.output_dir)

    # Baseline stats are provenance-only for SFT length. Shrinking below G.O.D's
    # evaluator ladder changes which rows retain supervised completion tokens.
    initial_seq_len = effective_sft_seq_len(loaded.model, plan.max_seq_len)
    seq_candidates = tokenize.sft_sequence_len_candidates(
        loaded.model, tokenizer, initial_seq_len
    )
    if spec.chat is not None:
        conversations = prompts.build_chat_conversations(rows, spec.chat)
        tokenized, seq_len = tokenize.first_nonempty_tokenization(
            seq_candidates,
            lambda candidate: tokenize.tokenize_chat(
                conversations,
                tokenizer,
                candidate,
                chat_template=spec.chat.chat_template,
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
    )

    # NEFTune (embedding noise) regularises SFT, but only on non-KL: on a KL task
    # the noise would leak into the disable_adapter base forward and corrupt the
    # reference. (It applies in training mode only, so eval stays clean.)
    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)

    # Size the schedule to the wall clock (zero-LR throughput probe) so the
    # cosine cooldown completes at the deadline. Must use the same trainer class
    # as the real run — a plain-Trainer probe would understate KL's double
    # forward and over-plan the schedule.
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer as _probe_cls

        _probe_extra = {"kl_coef": spec.kl_coef}
    else:
        from transformers import Trainer as _probe_cls

        _probe_extra = None
    ta_epochs, probe_per_step = time_aware_epochs(
        trainer_cls=_probe_cls,
        model=model,
        kwargs=kwargs,
        train_ex=train_ex,
        collator=tokenize.PadCollator(tokenizer.pad_token_id),
        deadline=deadline,
        eff_batch=eff_batch,
        strategy=strategy,
        trainer_extra=_probe_extra,
    )
    if ta_epochs is not None:
        kwargs["num_train_epochs"] = ta_epochs
        telemetry.event(
            "time_aware_epochs",
            epochs=ta_epochs,
            planned=plan.num_epochs,
            probe_per_step_s=round(probe_per_step, 4),
        )

    tracker = BestTracker()
    if val_ex:
        steps_per_epoch = max(1, len(train_ex) // eff_batch)
        kwargs.update(
            eval_strategy="steps",
            # ~4 evals/epoch — the week-3 curves put the eval minimum near
            # 1 epoch with meaningful movement each half-epoch, so 2/epoch was
            # too sparse to land near it. Relative cadence only: an absolute
            # floor is the small-task bug that sank the week-1 version.
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

    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    # Mirror less often when full weights are large to write (kill-safety is still
    # covered by the eager floor + final save + the DeadlineCallback margin).
    mirror_every = 100 if strategy == "full" else 25
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(spec, tokenizer, every=mirror_every, tracker=tracker),
        telemetry.make_trainer_callback(spec.output_dir),
    ]
    if val_ex:
        # Best-checkpoint selection: ship the eval-minimum artifact, not the
        # final one. Week-3 measured final-vs-best at +0.17 (LoRA) / +0.77
        # (full-FT) eval loss — the single largest lever we have.
        callbacks.append(_make_best_checkpoint_callback(spec, tokenizer, tracker))

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

    safe_train(trainer)
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if should_final_save(tracker, final_step=final_step):
        save_adapter(model, tokenizer, spec.output_dir)
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
