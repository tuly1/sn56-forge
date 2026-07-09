"""Direct Preference Optimization for DpoTask.

We use TRL's DPOTrainer — the same class the evaluator scores with — at beta=0.1
to match its `BETA_DPO`. The reference model is the LoRA-disabled base (ref_model
left None on a PEFT model), which is exactly the base the evaluator compares
against.
"""

from __future__ import annotations

from forge.clock import Deadline
from forge.data import loader, prompts
from forge.data.schema import TaskSpec
from forge.model import attach_lora, load_base
from forge.tasks.common import (
    _make_periodic_save_callback,
    build_training_kwargs,
    safe_train,
    save_adapter,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_dpo_plan

_EVAL_BETA = 0.1  # matches the evaluator's BETA_DPO


def run(spec: TaskSpec, deadline: Deadline) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    assert spec.dpo is not None, "DPO task missing dpo columns"

    rows = loader.load_rows(
        spec.cached_dataset_path, dataset_arg=spec.dataset, file_format=spec.file_format
    )
    examples = prompts.build_dpo_examples(rows, spec.dpo)
    if not examples:
        raise RuntimeError("no trainable DPO pairs")

    loaded = load_base(spec.cached_model_dir, for_generation=False)
    tokenizer = loaded.tokenizer
    plan = make_dpo_plan()
    model = attach_lora(
        loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
    )

    save_adapter(model, tokenizer, spec.output_dir)  # floor before training

    dataset = Dataset.from_list(examples)
    config = DPOConfig(
        **build_training_kwargs(spec, plan),
        beta=_EVAL_BETA,
        max_length=plan.max_seq_len,
        max_prompt_length=plan.max_seq_len // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference is the adapter-disabled base
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            DeadlineCallback(deadline),
            _make_periodic_save_callback(spec, tokenizer, every=25),
        ],
    )

    safe_train(trainer)
    save_adapter(model, tokenizer, spec.output_dir)
