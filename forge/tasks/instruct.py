"""Supervised fine-tuning for InstructTextTask and ChatTask.

Strategy: on small non-KL models we FULL fine-tune (the week-1 postmortem showed
every advancer full-fine-tuned and no LoRA miner advanced); on KL tasks and
large/multi-GPU models we keep the validated LoRA path (the KL trainer needs the
adapter, and LoRA is the safe multi-GPU route). The choice is made after the
model loads, from its real parameter count and the GPU it landed on.

Loss is completion-only, matching the evaluator. We also hold out a small slice
and log an eval loss purely for post-tournament learning — it drives no decision,
so it can't repeat the eval-cadence regression that got best-checkpoint reverted.
"""

from __future__ import annotations

import random

from forge import telemetry
from forge.clock import Deadline
from forge.data import loader, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.model import (
    attach_lora,
    decide_full_finetune,
    effective_seq_len,
    gpu_topology,
    load_base,
    model_param_billions,
    prepare_full_finetune,
)
from forge.tasks.common import (
    _make_periodic_save_callback,
    build_training_kwargs,
    safe_train,
    save_adapter,
    time_aware_epochs,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_sft_plan

# Hold out a small fixed val slice for eval-loss LOGGING only (never gates a
# checkpoint). Skip it on datasets too small to spare rows.
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

    seq_len = effective_seq_len(loaded.model, plan.max_seq_len)
    if spec.chat is not None:
        conversations = prompts.build_chat_conversations(rows, spec.chat)
        tokenized = tokenize.tokenize_chat(conversations, tokenizer, seq_len)
    else:
        assert spec.instruct is not None, "instruct task missing instruct columns"
        examples = prompts.build_instruct_examples(rows, spec.instruct)
        tokenized = tokenize.tokenize_instruct(examples, tokenizer, seq_len)

    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")

    # Eval-loss logging holds out a small slice — but NOT on KL tasks, where each
    # eval reruns the full KL double-forward and would eat the time budget.
    train_ex, val_ex = (tokenized, []) if is_kl else _split_for_eval(tokenized)
    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        handler="chat" if spec.chat is not None else "instruct",
        strategy=strategy,
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        gpu_gb=per_gpu_gb,
        seq_len=seq_len,
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

    if val_ex:
        steps_per_epoch = max(1, len(train_ex) // eff_batch)
        kwargs.update(
            eval_strategy="steps",
            eval_steps=max(1, steps_per_epoch // 2),  # ~2 evals/epoch, no floor bug
            per_device_eval_batch_size=max(1, plan.per_device_batch_size),
        )
    args = TrainingArguments(**kwargs)

    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    # Mirror less often when full weights are large to write (kill-safety is still
    # covered by the eager floor + final save + the DeadlineCallback margin).
    mirror_every = 100 if strategy == "full" else 25
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(spec, tokenizer, every=mirror_every),
        telemetry.make_trainer_callback(spec.output_dir),
    ]

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
    save_adapter(model, tokenizer, spec.output_dir)


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
