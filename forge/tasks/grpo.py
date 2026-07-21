"""Group Relative Policy Optimization for GrpoTask.

The validator ships reward functions as Python source inside `--dataset-type`.
We materialise them into callables and hand them to TRL's GRPOTrainer (the same
class the evaluator uses). GRPO is generation-heavy and the riskiest path, so
anything that goes wrong here still degrades to the fallback via the CLI wrapper.
"""

from __future__ import annotations

from forge.baseline import (
    load_baseline_summary,
    suggested_sequence_length,
    telemetry_fields,
)
from forge.clock import Deadline
from forge.data import loader, prompts
from forge.data.schema import TaskSpec
from forge.model import attach_lora, load_base
from forge.tasks.common import (
    _make_periodic_save_callback,
    build_training_kwargs,
    compatible_dataclass_kwargs,
    safe_train,
    save_adapter,
)
from forge.tasks.rewards import EVAL_BETA_GRPO, materialise_rewards
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_grpo_plan
from forge.tasks.trl_compat import (
    generation_kwargs_for_model,
    prompt_capped_grpo_trainer,
)

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
    baseline = None
    try:
        baseline = load_baseline_summary(
            spec.baseline_stats_path, expected_task_type=spec.task_type
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        telemetry.event("baseline_stats_invalid", error=str(exc))

    baseline_meta = telemetry_fields(baseline) if baseline is not None else {}
    telemetry.set_meta(
        handler="grpo",
        rows=len(rows),
        prompts_n=len(examples),
        reward_fns=len(reward_funcs),
        **baseline_meta,
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
    # Baseline GRPO lengths describe prompts, not generated completions.  Keep a
    # fixed 256-token completion reserve when converting p99 into a total cap.
    # With no valid baseline, preserve the existing static 512-token prompt cap.
    if baseline is None:
        max_prompt_length = plan.max_seq_len // 2
    else:
        total_sequence_cap = suggested_sequence_length(
            baseline,
            default=plan.max_seq_len,
            completion_reserve=256,
        )
        max_prompt_length = max(1, total_sequence_cap - 256)
    telemetry.set_meta(
        max_prompt_length=max_prompt_length, max_completion_length=256
    )
    # Quasar's staged DynamicCache is incompatible with TF5.12's causal-mask
    # call shape. model.config.use_cache=False is insufficient because TRL
    # builds a separate GenerationConfig from generation_config.json.
    generation_kwargs = generation_kwargs_for_model(model)
    config_values = dict(
        base_kwargs,
        num_generations=_NUM_GENERATIONS,
        max_prompt_length=max_prompt_length,
        max_completion_length=256,
        beta=EVAL_BETA_GRPO,
        reward_weights=weights,
    )
    if generation_kwargs is not None:
        config_values["generation_kwargs"] = generation_kwargs
    config_kwargs = compatible_dataclass_kwargs(
        GRPOConfig,
        config_values,
        allow_removed={"overwrite_output_dir", "max_prompt_length"},
    )
    telemetry.set_meta(
        grpo_prompt_cap_applied=(
            max_prompt_length
        ),
        grpo_prompt_cap_source=(
            "trl_0_24_config"
            if "max_prompt_length" in config_kwargs
            else "trl_1_5_trainer_override"
        ),
        grpo_generation_cache=("disabled_quasar" if generation_kwargs else "native"),
    )
    config = GRPOConfig(**config_kwargs)

    trainer_class = GRPOTrainer
    if "max_prompt_length" not in config_kwargs:
        trainer_class = prompt_capped_grpo_trainer(GRPOTrainer, max_prompt_length)

    trainer = trainer_class(
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
