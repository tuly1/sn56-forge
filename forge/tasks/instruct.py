"""Supervised fine-tuning for InstructTextTask and ChatTask.

Loss is computed only on completion tokens, matching the evaluator. On
KL-regularised tasks we swap in a trainer that adds the scored KL(model || base)
term. This one handler serves both task types: it branches on whether the spec
carries instruct columns or chat columns.

Checkpoint policy: we hold out a small validation slice and submit the
lowest-eval-loss checkpoint, not the last one — with the floored cosine the
model keeps moving right up to the deadline, so the final step is often past its
best. On KL tasks the eval loss includes the KL term, i.e. selection uses the
exact quantity the grader ranks.
"""

from __future__ import annotations

import math

from forge import telemetry
from forge.clock import Deadline
from forge.data import loader, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.data.split import split_train_val
from forge.model import attach_lora, effective_seq_len, load_base
from forge.tasks.common import (
    _make_periodic_save_callback,
    build_training_kwargs,
    safe_train,
    save_adapter,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_sft_plan

# Only run the end-of-training eval when at least this much soft-stop time is
# left; otherwise keep the mirrored best and let the export reserve do its job.
_FINAL_EVAL_MIN_REMAINING_S = 240


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
    telemetry.event("model_loaded", rows=len(rows))

    plan = make_sft_plan(use_kl=spec.use_kl and spec.kl_coef > 0)
    model = attach_lora(
        loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
    )

    # Floor first, before the minutes-long tokenization of a large dataset: write
    # a valid (untrained) adapter so a kill anywhere in setup still leaves a
    # scoreable model at the output path. Training overwrites it.
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

    is_kl = spec.use_kl and spec.kl_coef > 0
    train_examples, val_examples = split_train_val(tokenized)
    telemetry.set_meta(
        handler="chat" if spec.chat is not None else "instruct",
        seq_len=seq_len,
        tokenized=len(tokenized),
        train_n=len(train_examples),
        val_n=len(val_examples),
        lora_r=plan.lora_r,
        lr=plan.learning_rate,
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        epochs=plan.num_epochs,
        neftune=not is_kl,
    )

    # NEFTune (embedding noise) regularises SFT, but only on the plain path: on a
    # KL task the noise would leak into the disable_adapter base forward and
    # corrupt the reference the KL is measured against. (During evaluation the
    # hook is inert — noise applies in training mode only — so eval stays clean.)
    kwargs = build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    if val_examples:
        steps_per_epoch = max(
            1,
            len(train_examples)
            // (plan.per_device_batch_size * plan.grad_accum_steps),
        )
        kwargs.update(
            eval_strategy="steps",
            eval_steps=max(50, steps_per_epoch // 3),
            per_device_eval_batch_size=plan.per_device_batch_size,
        )
    args = TrainingArguments(**kwargs)

    from forge.tuning.best import BestCheckpointCallback, BestTracker

    tracker = BestTracker()
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(spec, tokenizer, every=25, tracker=tracker),
        telemetry.make_trainer_callback(spec.output_dir),
    ]
    if val_examples:
        callbacks.append(BestCheckpointCallback(spec, tokenizer, tracker))

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples) if val_examples else None,
        data_collator=collator,
        callbacks=callbacks,
    )
    if is_kl:
        from forge.tuning.kl import KLSFTTrainer

        trainer = KLSFTTrainer(kl_coef=spec.kl_coef, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    safe_train(trainer)
    _finalize(trainer, tracker, spec, tokenizer, model, deadline)


def _finalize(trainer, tracker, spec: TaskSpec, tokenizer, model, deadline: Deadline) -> None:
    """Decide what the output path holds: the best-eval checkpoint or the final.

    No best recorded (tiny dataset, or eval never fired) -> save the final model,
    as before. Best recorded -> it is already mirrored at the output path; only
    overwrite it if a final evaluation, time permitting, shows the final model is
    actually better.
    """
    if not tracker.saved_best:
        telemetry.event("final_saved", reason="no_best_recorded")
        save_adapter(model, tokenizer, spec.output_dir)
        return

    if deadline.remaining() <= _FINAL_EVAL_MIN_REMAINING_S:
        telemetry.event("best_kept", reason="no_time_for_final_eval",
                        best_loss=tracker.best_loss)
        telemetry.write_into(spec.output_dir)
        return  # keep the mirrored best; no time to fairly judge the final

    try:
        final_loss = trainer.evaluate().get("eval_loss")
    except Exception:
        telemetry.event("best_kept", reason="final_eval_failed",
                        best_loss=tracker.best_loss)
        telemetry.write_into(spec.output_dir)
        return  # keep the mirrored best
    if (
        final_loss is not None
        and not (isinstance(final_loss, float) and math.isnan(final_loss))
        and tracker.best_loss is not None
        and final_loss < tracker.best_loss
    ):
        telemetry.event("final_saved", reason="final_beats_best",
                        final_loss=float(final_loss), best_loss=tracker.best_loss)
        save_adapter(model, tokenizer, spec.output_dir)
    else:
        telemetry.event("best_kept", reason="best_beats_final",
                        final_loss=final_loss, best_loss=tracker.best_loss)
        telemetry.write_into(spec.output_dir)
