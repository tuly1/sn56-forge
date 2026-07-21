"""Direct Preference Optimization for DpoTask.

We use TRL's DPOTrainer — the same class the evaluator scores with — at beta=0.1
to match its `BETA_DPO`. The reference model is the LoRA-disabled base (ref_model
left None on a PEFT model), which is exactly the base the evaluator compares
against.
"""

from __future__ import annotations

from forge.baseline import (
    load_baseline_summary,
    telemetry_fields,
)
from forge.clock import Deadline
from forge.data import loader, prompts
from forge.data.schema import TaskSpec
from forge.model import attach_lora, effective_seq_len, load_base
from forge.tasks.common import (
    BestTracker,
    _make_best_checkpoint_callback,
    _make_periodic_save_callback,
    build_training_kwargs,
    compatible_dataclass_kwargs,
    safe_train,
    save_adapter,
    should_final_save,
)
from forge.tuning.callbacks import DeadlineCallback
from forge.tuning.plan import make_dpo_plan
from forge.tasks.trl_compat import PromptCappedPreferenceCollator

_EVAL_BETA = 0.1  # matches the evaluator's BETA_DPO
_EVAL_MIN_PAIRS = 256
_EVAL_MAX_PAIRS = 64


def _dpo_v15_collator(config_kwargs, tokenizer, seq_len, prompt_cap):
    """Return a prompt-capping collator only when TRL removed its config field."""
    if "max_prompt_length" in config_kwargs:
        return None
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    delegate = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        max_length=seq_len,
        truncation_mode="keep_start",
    )
    return PromptCappedPreferenceCollator(delegate, prompt_cap)


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
    from forge import telemetry

    telemetry.collect_env()
    baseline = None
    try:
        baseline = load_baseline_summary(
            spec.baseline_stats_path, expected_task_type=spec.task_type
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        # Baseline statistics are untrusted task input.  A malformed/mismatched
        # file must not forfeit training or change the static safe plan.
        telemetry.event("baseline_stats_invalid", error=str(exc))

    train_examples, eval_examples = prompts.split_for_eval(
        examples,
        min_size=_EVAL_MIN_PAIRS,
        max_eval_rows=_EVAL_MAX_PAIRS,
    )
    baseline_meta = telemetry_fields(baseline) if baseline is not None else {}
    telemetry.set_meta(
        handler="dpo",
        rows=len(rows),
        pairs=len(examples),
        train_pairs=len(train_examples),
        eval_pairs=len(eval_examples),
        baseline_seq_policy="provenance_only",
        **baseline_meta,
    )
    plan = make_dpo_plan()
    model = attach_lora(
        loaded.model, r=plan.lora_r, alpha=plan.lora_alpha, dropout=plan.lora_dropout
    )

    save_adapter(model, tokenizer, spec.output_dir)  # floor before training

    dataset = Dataset.from_list(train_examples)
    # Current baseline DPO stats do not expose prompt length independently:
    # sequence p99 is prompt+chosen while rejected p99 is rejected-only.  No safe
    # prompt+rejected cap can be derived, so record the provenance but retain the
    # static/model positional cap rather than silently truncating long pairs.
    seq_len = effective_seq_len(loaded.model, plan.max_seq_len)
    prompt_cap = min(512, max(1, seq_len - 1))
    telemetry.set_meta(seq_len=seq_len)
    training_kwargs = build_training_kwargs(spec, plan)
    tracker = BestTracker()
    if eval_examples:
        effective_batch = plan.per_device_batch_size * plan.grad_accum_steps
        steps_per_epoch = max(1, len(train_examples) // effective_batch)
        training_kwargs.update(
            eval_strategy="steps",
            eval_steps=max(1, steps_per_epoch // 3),
            per_device_eval_batch_size=max(1, plan.per_device_batch_size),
        )
    config_values = dict(
        training_kwargs,
        beta=_EVAL_BETA,
        max_length=seq_len,
        # TRL 0.24 exposed this scorer default as a field; TRL 1.5 removed the
        # field entirely, matching current G.O.D, so it is omitted there.
        max_prompt_length=prompt_cap,
        truncation_mode="keep_start",
    )
    config_kwargs = compatible_dataclass_kwargs(
        DPOConfig,
        config_values,
        allow_removed={"overwrite_output_dir", "max_prompt_length"},
    )
    telemetry.set_meta(
        dpo_prompt_cap=(
            config_kwargs.get("max_prompt_length")
            if "max_prompt_length" in config_kwargs
            else prompt_cap
        ),
        dpo_prompt_cap_source=(
            "trl_0_24_config"
            if "max_prompt_length" in config_kwargs
            else "trl_1_5_collator"
        ),
    )
    config = DPOConfig(**config_kwargs)
    data_collator = _dpo_v15_collator(
        config_kwargs, tokenizer, seq_len, prompt_cap
    )

    callbacks = [
        DeadlineCallback(deadline),
        _make_periodic_save_callback(
            spec, tokenizer, every=25, tracker=tracker
        ),
        telemetry.make_trainer_callback(spec.output_dir),
    ]
    if eval_examples:
        callbacks.append(_make_best_checkpoint_callback(spec, tokenizer, tracker))

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference is the adapter-disabled base
        args=config,
        train_dataset=dataset,
        eval_dataset=Dataset.from_list(eval_examples) if eval_examples else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    safe_train(trainer)
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if should_final_save(tracker, final_step=final_step):
        save_adapter(model, tokenizer, spec.output_dir)
    else:
        telemetry.event(
            "kept_best_checkpoint",
            best=round(tracker.best, 5),
            best_step=tracker.best_step,
            last_eval=round(tracker.last, 5),
            last_eval_step=tracker.last_step,
            final_step=final_step,
        )
