"""Supervised fine-tuning for InstructTextTask and ChatTask.

Loss is computed only on completion tokens, matching the evaluator. On
KL-regularised tasks we swap in a trainer that adds the scored KL(model || base)
term. This one handler serves both task types: it branches on whether the spec
carries instruct columns or chat columns.
"""

from __future__ import annotations

from forge.clock import Deadline
from forge.data import loader, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.model import attach_lora, load_base
from forge.tasks.common import (
    _make_periodic_save_callback,
    build_training_kwargs,
    safe_train,
    save_adapter,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_sft_plan


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

    if spec.chat is not None:
        conversations = prompts.build_chat_conversations(rows, spec.chat)
        tokenized = tokenize.tokenize_chat(
            conversations, tokenizer, _plan_seq_len(spec)
        )
    else:
        assert spec.instruct is not None, "instruct task missing instruct columns"
        examples = prompts.build_instruct_examples(rows, spec.instruct)
        tokenized = tokenize.tokenize_instruct(examples, tokenizer, _plan_seq_len(spec))

    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")

    plan = make_sft_plan(use_kl=spec.use_kl and spec.kl_coef > 0)
    model = attach_lora(
        loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
    )

    # Floor first: write a valid (untrained) adapter before training so a kill
    # during setup or before the first periodic save still leaves a scoreable
    # model at the output path. Training overwrites it.
    save_adapter(model, tokenizer, spec.output_dir)

    dataset = Dataset.from_list(tokenized)
    args = TrainingArguments(**build_training_kwargs(spec, plan))
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(spec, tokenizer, every=25),
    ]

    if spec.use_kl and spec.kl_coef > 0:
        from forge.tuning.kl import KLSFTTrainer

        trainer = KLSFTTrainer(
            kl_coef=spec.kl_coef,
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

    safe_train(trainer)
    save_adapter(model, tokenizer, spec.output_dir)


def _plan_seq_len(spec: TaskSpec) -> int:
    return make_sft_plan(use_kl=spec.use_kl and spec.kl_coef > 0).max_seq_len
