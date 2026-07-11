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
from forge.model import attach_lora, effective_seq_len, load_base
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
    dataset = Dataset.from_list(tokenized)
    # NEFTune (embedding noise) regularises SFT, but only on the plain path: on a
    # KL task the noise would leak into the disable_adapter base forward and
    # corrupt the reference the KL is measured against.
    args = TrainingArguments(
        **build_training_kwargs(spec, plan, neftune_alpha=None if is_kl else 5.0)
    )
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(spec, tokenizer, every=25),
    ]

    if is_kl:
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
