"""Group Relative Policy Optimization for GrpoTask.

The validator ships reward functions as Python source inside `--dataset-type`.
We materialise them into callables and hand them to TRL's GRPOTrainer (the same
class the evaluator uses). GRPO is generation-heavy and the riskiest path, so
anything that goes wrong here still degrades to the fallback via the CLI wrapper.
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
from forge.tasks.rewards import materialise_rewards
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_grpo_plan

_NUM_GENERATIONS = 2


def run(spec: TaskSpec, deadline: Deadline) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    assert spec.grpo is not None, "GRPO task missing grpo spec"

    reward_funcs, weights = materialise_rewards(
        spec.grpo.reward_functions, spec.grpo.reward_weights
    )
    if not reward_funcs:
        raise RuntimeError("no usable reward functions")

    rows = loader.load_rows(
        spec.cached_dataset_path, dataset_arg=spec.dataset, file_format=spec.file_format
    )
    examples = prompts.build_grpo_examples(rows, spec.grpo)
    if not examples:
        raise RuntimeError("no GRPO prompts")

    loaded = load_base(spec.cached_model_dir, for_generation=True)
    tokenizer = loaded.tokenizer
    from forge import telemetry

    telemetry.collect_env()
    telemetry.set_meta(
        handler="grpo",
        rows=len(rows),
        prompts_n=len(examples),
        reward_fns=len(reward_funcs),
    )
    plan = make_grpo_plan()
    model = attach_lora(
        loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
    )

    save_adapter(model, tokenizer, spec.output_dir)  # floor before training

    base_kwargs = build_training_kwargs(spec, plan)
    # GRPO requires the global batch to be divisible by num_generations.
    base_kwargs["per_device_train_batch_size"] = max(
        _NUM_GENERATIONS, base_kwargs["per_device_train_batch_size"]
    )
    config = GRPOConfig(
        **base_kwargs,
        num_generations=_NUM_GENERATIONS,
        max_prompt_length=plan.max_seq_len // 2,
        max_completion_length=256,
        reward_weights=weights,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
        callbacks=[
            DeadlineCallback(deadline),
            _make_periodic_save_callback(spec, tokenizer, every=10),
            telemetry.make_trainer_callback(spec.output_dir),
        ],
    )

    # Floor the OOM-retry batch at num_generations so a retry can't break TRL's
    # global-batch-divisible-by-num_generations requirement.
    safe_train(trainer, min_batch=_NUM_GENERATIONS)
    save_adapter(model, tokenizer, spec.output_dir)
