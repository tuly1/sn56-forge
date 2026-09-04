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
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, replace
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
    is_quasar_model,
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
from forge.tuning.overrides import (
    RecipeOverrides,
    apply_plan_overrides,
    apply_training_kwargs_overrides,
    cap_epochs,
    load_recipe_overrides,
    long_rows_policy,
    plan_override_diff,
)
from forge.tuning.overrides import neftune_alpha as recipe_neftune_alpha
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
_CAP_QUANTUM = 256

# --- Admission headroom -------------------------------------------------------
# Calibrated 2026-09-03 on a real H100 80 GB (survival smoke, lease
# 20260903T105152Z, exact axolotl image, --network none; nine cells, host
# memory.used sampled every 5 s).  `_probe_once` measures max_memory_reserved /
# total for ONE fresh optimizer step; real training then grows the reserved pool
# (variable-length batches fragment the caching allocator, optimizer state and
# eval passes settle in).  Admitted rung, its worst-batch probe, then the host
# fraction seen during training (peak / plateau):
#   bloomz-560m   b4/ga4/gc   0.831 -> 0.916 / 0.771
#   qwen3.5-9b    b1/ga16     0.857 -> 0.919 / 0.908
#   granite-4.1   b4/ga4/gc   0.629 -> 0.948 / 0.948
#   gemma-4-e4b   b2/ga8/gc   0.629 -> 0.957 / 0.701
#   lfm2.5-8b     b4/ga4/gc   0.604 -> 0.922 / 0.589
#   minicpm5-1b   b4/ga4/gc   0.413 -> 0.750 / 0.359
#   falcon-rw-1b  b4/ga4      0.176 -> 0.289 / 0.288   (p99 probe 0.213)
# A rung is admitted only when every measured probe batch plus this headroom
# still fits under the ceiling (worst <= 0.78).  That steps the two rungs that
# went on to train above 0.90 from a high probe (bloomz gc rung, qwen3.5-9b)
# down exactly one rung, and leaves Falcon's first rung trivially admitted
# (0.213 + 0.12 = 0.333).  Growth from a moderate probe (granite: 0.629 ->
# 0.948) is allocator fragmentation that no fixed headroom predicts; it is
# recorded, not modelled, here.
_ADMISSION_CEILING = 0.90
_STEADY_STATE_HEADROOM = 0.12

# --- Wall-budget step planning -------------------------------------------------
# Same smoke.  Four of the six 2026-pool cells hit `deadline_stop` and exported
# PARTIAL_TRAINED_BEST: the epoch plan carried no evaluation cost, and the
# zero-LR timing probe was cut by its own budget cap on Qwen3.5 (4.4-20 s per
# step) so the default two epochs were scheduled.  Measured, net of evaluation:
#   cell          schedule       step s     eval s (256 rows)   stopped at
#   granite-4.1   2.51 ep / 211  5.7        23.7 (eval batch 4)  206
#   gemma-4-e4b   2.40 ep / 202  6.05       20.5 (eval batch 2)  192
#   qwen3.5-9b    2 ep / 168     4.4-20     27 (batch 1; 83 first) 118
#   qwen3.5-4b    2 ep / 168     ~13        26 (batch 1; 82 first)  66
#   lfm2.5-8b     4.0 ep / 336   1.65       7.2                  completed
#   bloomz-560m   2 ep / 168     1.5        6.3                  completed
#   falcon-rw-1b  2 ep / 110     0.72       2.7                  completed
# eval_s / (step_s * val_rows / eff_batch) measured 0.21-0.38 (mean 0.26); the
# probe's per-step drifted -5%..+10% over a run; the first Qwen3.5 eval ran 3x
# its steady cost.  The plan discounts the remaining soft budget (which already
# excludes the 180 s export reserve) by a setup allowance and this margin.
_WALL_PLAN_MARGIN = 0.90
_WALL_PLAN_EVAL_ROW_FACTOR = 0.30
_WALL_PLAN_SETUP_S = 60.0

# --- Cold-only planning (no zero-LR timing probe) ---------------------------
# 646d0b70 smoke (VM 1022601, torch 2.9.1; Mac mirror lease
# 20260903T214700Z).  On Qwen3.5 the zero-LR probe is cut by its own 12 %
# budget cap (10-25 s per step), leaving only the admitted rung's worst-batch
# admission timing -- which is ONE micro-batch of the longest rows plus any
# cold start, not an optimizer step, and misled in both directions:
#   qwen3.5-4b  b1/ga16     cold 21.26 s -> planned 29 of 168; real steps
#               13-25 s (train_curve 596.6 s @10, 802.7 s @20); 29 steps
#               COMPLETE at 1004 s of a 1260 s soft budget with the host at
#               0.65 -- below the 50-step certification floor.
#   qwen3.5-9b  b1/ga16/gc  warm micro-batch 1.786 s -> "schedule fits" 168;
#               real steps 7.5-23 s -> deadline_stop at 77 (per_step 7.56 s).
# So whenever the zero-LR probe declined or was cut: time three real optimizer
# steps on the still-alive timing model if three cold-priced steps fit in 5 %
# of the remaining budget (median of steps 2-3, self-limited to that
# allowance); otherwise plan with cold / 3 when the cold step would cap the
# run; and never plan below max(50, 30 % of the schedule) while the discounted
# estimate affords it.  The deadline callback stays the backstop, so an
# optimistic plan degrades to an honest PARTIAL_TRAINED_BEST, never a floor.
_COLD_PROBE_DISCOUNT = 3.0
_MIN_REAL_STEPS = 50
_FLOOR_SCHEDULE_FRACTION = 0.30
_WARM_PROBE_STEPS = 3
_WARM_PROBE_BUDGET_FRACTION = 0.05


@dataclass(frozen=True)
class _DataView:
    """One sequence-cap-specific dataset, split, and ordering authority."""

    cap: int
    cap_basis: str
    source_rows: int
    retained_source_rows: int
    retained_rows: int
    retained_fraction: float
    ordering_sha256: str
    train: list
    validation: list
    authorized: bool = True
    required_retained_fraction: float = 0.0

    def identity(self) -> dict[str, Any]:
        return {
            "sequence_cap": self.cap,
            "sequence_cap_basis": self.cap_basis,
            "source_rows": self.source_rows,
            "retained_source_rows": self.retained_source_rows,
            "rows_retained": self.retained_source_rows,
            "retained_rows": self.retained_rows,
            "tokenized_examples": self.retained_rows,
            "retained_fraction": self.retained_fraction,
            "train_rows": len(self.train),
            "train_n": len(self.train),
            "validation_rows": len(self.validation),
            "val_n": len(self.validation),
            "ordering_sha256": self.ordering_sha256,
            "cap_authorized": self.authorized,
            "required_retained_fraction": self.required_retained_fraction,
        }


def run(spec: TaskSpec, deadline: Deadline) -> None:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    rows = loader.load_rows(
        spec.cached_dataset_path, dataset_arg=spec.dataset, file_format=spec.file_format
    )
    if not rows:
        raise RuntimeError("empty dataset")

    # Study-only recipe overrides (FORGE_RECIPE_OVERRIDES_JSON): read exactly
    # once here; inert and silent when the variable is unset.
    recipe = load_recipe_overrides()
    if recipe.present:
        telemetry.event("recipe_overrides", **recipe.record())
        telemetry.set_meta(recipe_overrides=recipe.record())

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
    checkpointing_supported = not is_quasar_model(loaded.model)
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
    plan_before_recipe = plan
    plan = apply_plan_overrides(plan, recipe)
    if recipe.present:
        telemetry.event(
            "recipe_plan_applied",
            applied=recipe.active,
            changes=plan_override_diff(plan_before_recipe, plan),
            neftune_alpha=recipe_neftune_alpha(is_kl, recipe),
            long_rows=long_rows_policy(recipe),
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

    # Start at G.O.D's evaluator-equivalent cap. If measured memory rejects every
    # full-cap geometry, later rungs may use smaller caps, but only when those
    # caps are derived from a validated task distribution and the exact rows are
    # retokenized, fingerprinted, and threaded through every downstream phase.
    initial_seq_len = effective_sft_seq_len(model, plan.max_seq_len)
    seq_candidates = tokenize.sft_sequence_len_candidates(
        model, tokenizer, initial_seq_len
    )
    if spec.chat is not None:
        source_items = prompts.build_chat_conversations(rows, spec.chat)
        template_resolution = tokenize.resolve_chat_template(
            spec.chat.chat_template, tokenizer
        )
        if template_resolution.degraded:
            # The valid floor already exists. Persist the compatibility choice
            # before processing rows so even a kill during tokenization explains
            # why native/ChatML semantics replaced the requested future name.
            telemetry.write_into(spec.output_dir)
        def tokenize_one(source_index: int, candidate: int) -> list:
            return tokenize.tokenize_chat_resolved(
                [source_items[source_index]], tokenizer, candidate,
                resolution=template_resolution,
            )
    else:
        assert spec.instruct is not None, "instruct task missing instruct columns"
        if spec.instruct.output is None:
            source_items = prompts.build_completion_documents(rows, spec.instruct)

            def tokenize_one(source_index: int, candidate: int) -> list:
                return tokenize.tokenize_completion(
                    [source_items[source_index]], tokenizer, candidate
                )
        else:
            source_items = prompts.build_instruct_examples(rows, spec.instruct)

            def tokenize_one(source_index: int, candidate: int) -> list:
                return tokenize.tokenize_instruct(
                    [source_items[source_index]], tokenizer, candidate,
                    on_overflow=long_rows_policy(recipe),
                )

    initial_view = None
    for candidate in seq_candidates:
        basis = ("evaluator_initial" if candidate == seq_candidates[0]
                 else "evaluator_nonempty_retry")
        view = _make_data_view(
            candidate, basis, len(source_items), tokenize_one, is_kl
        )
        if view.retained_rows:
            initial_view = view
            break
    if initial_view is None:
        raise RuntimeError("no trainable examples after tokenization")
    seq_len = initial_view.cap
    plan = replace(plan, max_seq_len=seq_len)
    reduced_cap_rows, cap_authority = _reduced_sequence_caps(
        baseline_summary, initial_view, expected_records=len(rows)
    )
    plans = _plans(
        plan,
        tuple(cap for cap, _basis, _minimum in reduced_cap_rows),
        checkpointing_supported=checkpointing_supported,
    )
    cap_basis = {
        seq_len: initial_view.cap_basis,
        **{cap: basis for cap, basis, _minimum in reduced_cap_rows},
    }
    cap_minimum = {
        cap: minimum for cap, _basis, minimum in reduced_cap_rows
    }
    views = {seq_len: initial_view}

    telemetry.event("sequence_cap_view", **initial_view.identity())
    telemetry.event("sequence_cap_authority", **cap_authority)
    initial_view = None
    view = None

    def data_for(candidate: TrainPlan) -> _DataView:
        cap = int(candidate.max_seq_len)
        if cap not in views:
            views.clear()
            gc.collect()
            views[cap] = _make_data_view(
                cap,
                cap_basis[cap],
                len(source_items),
                tokenize_one,
                is_kl,
                required_retained_fraction=cap_minimum.get(cap, 0.0),
            )
            telemetry.event("sequence_cap_view", **views[cap].identity())
        return views[cap]

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

    def make(candidate: TrainPlan, candidate_model: Any, view: _DataView,
             probe_rows: list | None = None, epochs: float | None = None,
             max_steps: int | None = None):
        return _make_trainer(
            spec, deadline, candidate, candidate_model, tokenizer,
            probe_rows if probe_rows is not None else view.train,
            [] if probe_rows is not None else view.validation,
            collator, is_kl, strategy, n_gpus,
            probe=probe_rows is not None, epochs=epochs, max_steps=max_steps,
            recipe=recipe)

    holder = [model]
    del model
    plan, admitted_view, admitted_index, admission, admitted = _admit(
        plans, holder.pop(), data_for, rebuild, make
    )
    views.clear()
    views[admitted_view.cap] = admitted_view
    gc.collect()
    eff_batch = plan.per_device_batch_size * plan.grad_accum_steps
    telemetry.set_meta(
        handler="chat" if spec.chat is not None else "instruct",
        strategy=strategy,
        params_b=round(params_b, 3),
        n_gpus=n_gpus,
        gpu_gb=per_gpu_gb,
        seq_len=plan.max_seq_len,
        sequence_cap_basis=admitted_view.cap_basis,
        seq_len_candidates=seq_candidates,
        survival_seq_caps=list(dict.fromkeys(
            candidate.max_seq_len for candidate in plans
        )),
        baseline_seq_policy="validated_distribution_survival_fallback_only",
        tokenized=admitted_view.retained_rows,
        source_rows=admitted_view.source_rows,
        rows_retained=admitted_view.retained_source_rows,
        retained_source_rows=admitted_view.retained_source_rows,
        retained_fraction=admitted_view.retained_fraction,
        dataset_ordering_sha256=admitted_view.ordering_sha256,
        train_n=len(admitted_view.train),
        val_n=len(admitted_view.validation),
        lora_r=plan.lora_r,
        lr=plan.learning_rate,
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=eff_batch,
        gradient_checkpointing=plan.gradient_checkpointing,
        epochs=plan.num_epochs,
        neftune=recipe_neftune_alpha(is_kl, recipe) is not None,
        tokens_per_step_cap=eff_batch * plan.max_seq_len,
        geometry_admission=admission,
        geometry_admitted=admitted,
        admission_ceiling=_ADMISSION_CEILING,
        admission_headroom=_STEADY_STATE_HEADROOM,
        admitted_predicted_steady_state_fraction=_admitted_prediction(
            admission, admitted
        ),
    )
    admitted_view = None
    admitted_plan = plan
    admitted_step_s = _admitted_step_s(admission, admitted)

    if is_kl:
        from forge.tuning.kl import KLSFTTrainer as probe_cls

        probe_extra = {"kl_coef": spec.kl_coef}
    else:
        probe_cls = Trainer
        probe_extra = None

    timing: dict[str, float | None] = {}

    def measure_epochs(candidate: TrainPlan, view: _DataView) -> float | None:
        kwargs = apply_training_kwargs_overrides(
            build_training_kwargs(
                spec, candidate,
                neftune_alpha=recipe_neftune_alpha(is_kl, recipe),
            ),
            recipe,
        )
        timing["probe_per_step"] = None
        timing["warm_step_s"] = None
        timing_model = rebuild(candidate)
        try:
            candidate_epochs, probe_per_step = time_aware_epochs(
                trainer_cls=probe_cls,
                model=timing_model,
                kwargs=kwargs,
                train_ex=view.train,
                collator=collator,
                deadline=deadline,
                eff_batch=(candidate.per_device_batch_size
                           * candidate.grad_accum_steps),
                strategy=strategy,
                trainer_extra=probe_extra,
            )
            if candidate_epochs is None:
                # The zero-LR probe declined or was cut by its own budget cap.
                # While the timing model is still alive, measure a few warm
                # real steps when the cold admission step would cap the run.
                timing["warm_step_s"] = warm_probe(
                    candidate, view, timing_model, kwargs
                )
        finally:
            holder = [timing_model]
            timing_model = None
            _discard(holder.pop())
        candidate_epochs = cap_epochs(candidate_epochs, recipe)
        if candidate_epochs is not None:
            timing["probe_per_step"] = probe_per_step
            telemetry.event(
                "time_aware_epochs",
                epochs=candidate_epochs,
                planned=candidate.num_epochs,
                probe_per_step_s=round(probe_per_step, 4),
                **_plan_identity(candidate),
                **view.identity(),
            )
        return candidate_epochs

    def is_admitted_geometry(candidate: TrainPlan) -> bool:
        return admitted_step_s is not None and (
            _plan_identity(candidate) == _plan_identity(admitted_plan)
        )

    def warm_probe(candidate: TrainPlan, view: _DataView, model: Any,
                   kwargs: dict[str, Any]) -> float | None:
        if not is_admitted_geometry(candidate):
            return None
        decision = _warm_probe_decision(
            budget_s=deadline.remaining(),
            cold_step_s=admitted_step_s,
            train_rows=len(view.train),
            val_rows=len(view.validation),
            per_device_batch=candidate.per_device_batch_size,
            grad_accum=candidate.grad_accum_steps,
            epochs=candidate.num_epochs,
            n_gpus=n_gpus,
        )
        telemetry.event(
            "warm_probe_decision", **{**_plan_identity(candidate), **decision}
        )
        if not decision["run"]:
            return None
        measured = _warm_probe_step_s(
            trainer_cls=probe_cls, model=model, kwargs=kwargs,
            train_ex=view.train, collator=collator, trainer_extra=probe_extra,
            stop_after_s=decision["stop_after_s"],
        )
        telemetry.event(
            "warm_probe",
            **{**view.identity(), **_plan_identity(candidate), **measured},
        )
        return measured["step_s"]

    def plan_steps(candidate: TrainPlan, view: _DataView,
                   epochs: float | None) -> int | None:
        # Measured seconds per optimizer step, best instrument first: the
        # zero-LR timing probe (steps 10->30 of the exact geometry) plans
        # exactly as before; otherwise the admitted rung's cold worst-batch
        # probe plans through `_plan_cold_only` (warm probe, discount, floor).
        # Without any measurement the schedule is left to the deadline
        # callback exactly as before.
        geometry = dict(
            train_rows=len(view.train),
            val_rows=len(view.validation),
            per_device_batch=candidate.per_device_batch_size,
            grad_accum=candidate.grad_accum_steps,
            epochs=candidate.num_epochs if epochs is None else epochs,
            n_gpus=n_gpus,
        )
        probe_step = timing.get("probe_per_step")
        if probe_step is not None:
            wall = _plan_wall_steps(
                budget_s=deadline.remaining(), step_s=probe_step,
                step_source="timing_probe", **geometry,
            )
        elif is_admitted_geometry(candidate):
            wall = _plan_cold_only(
                budget_s=deadline.remaining(), cold_step_s=admitted_step_s,
                warm_step_s=timing.get("warm_step_s"), **geometry,
            )
        else:
            wall = _plan_wall_steps(
                budget_s=deadline.remaining(), step_s=None,
                step_source="none", **geometry,
            )
        _record_wall_plan(wall, candidate, view)
        return int(wall["planned_steps"]) if wall["cap_applied"] else None

    trainer, tracker, soup_route, plan, training_view, training_outcome = _train_ladder(
        plans[admitted_index:], data_for, rebuild, make, measure_epochs,
        spec, tokenizer, plan_steps=plan_steps,
    )
    _record_training_selection(plan, training_view, training_outcome)
    if training_outcome == "progressed_oom_preserved":
        telemetry.event(
            "training_progressed_oom_preserved",
            **_plan_identity(plan),
            **training_view.identity(),
        )
        telemetry.write_into(spec.output_dir)
        return
    if training_outcome == "progressed_oom_export_failed":
        telemetry.event(
            "training_progressed_oom_export_failed",
            **_plan_identity(plan),
            **training_view.identity(),
        )
        telemetry.write_into(spec.output_dir)
        return
    if trainer is None or tracker is None or training_view is None:
        telemetry.event(
            "training_geometry_exhausted",
            terminal_artifact_truth=ARTIFACT_FLOOR,
            training_outcome=training_outcome,
            **_plan_identity(plan),
            **training_view.identity(),
        )
        write_artifact_truth(
            spec.output_dir,
            ARTIFACT_FLOOR,
            optimizer_step=0,
            reason="zero_progress_oom_ladder_exhausted",
        )
        telemetry.write_into(spec.output_dir)
        return
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


def _make_data_view(
    cap: int,
    cap_basis: str,
    source_rows: int,
    tokenize_one: Callable[[int, int], list],
    is_kl: bool,
    *,
    required_retained_fraction: float = 0.0,
) -> _DataView:
    """Retokenize exact source rows at ``cap`` and fingerprint their ordering."""
    tokenized = []
    retained_sources: set[int] = set()
    ordering = hashlib.sha256()
    for source_index in range(source_rows):
        chunks = tokenize_one(source_index, cap)
        if chunks:
            retained_sources.add(source_index)
        for chunk_index, row in enumerate(chunks):
            payload = json.dumps(
                {
                    "source_index": source_index,
                    "chunk_index": chunk_index,
                    "row": row,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            ordering.update(len(payload).to_bytes(8, "big"))
            ordering.update(payload)
            tokenized.append(row)
    train, validation = (tokenized, []) if is_kl else _split_for_eval(tokenized)
    retained = len(retained_sources)
    retained_fraction = round(retained / source_rows, 12) if source_rows else 0.0
    return _DataView(
        cap=int(cap),
        cap_basis=cap_basis,
        source_rows=source_rows,
        retained_source_rows=retained,
        retained_rows=len(tokenized),
        retained_fraction=retained_fraction,
        ordering_sha256=ordering.hexdigest(),
        train=train,
        validation=validation,
        authorized=(
            bool(train) and retained_fraction >= required_retained_fraction
        ),
        required_retained_fraction=required_retained_fraction,
    )


def _reduced_sequence_caps(
    summary: Any,
    initial: _DataView,
    *,
    expected_records: int,
):
    """Return only distribution-authorized caps below the evaluator cap.

    The validator summary is schema-checked before reaching this function.  p99
    is the first quality-preserving rescue and p95 is the final survival rung;
    both round upward so the reported quantile remains inside the cap.
    """
    authority = {
        "policy": "validated_p99_then_p95_round_up_256",
        "quantum": _CAP_QUANTUM,
        "initial_cap": initial.cap,
        "source": "unavailable",
        "current_task_records": int(expected_records),
        "caps": [],
    }
    if summary is None:
        return (), authority
    authority.update(
        source="validated_baseline_summary",
        baseline_stats_sha256=summary.sha256,
        baseline_num_records=summary.num_records,
        sequence_p99=summary.sequence_p99,
        sequence_p95=summary.sequence_p95,
        record_count_matches=(summary.num_records == int(expected_records)),
    )
    if summary.num_records != int(expected_records):
        authority["source"] = "rejected_record_count_mismatch"
        return (), authority
    rows: list[tuple[int, str, float]] = []
    for quantile, value, minimum in (
        ("p99", summary.sequence_p99, 0.99),
        ("p95", summary.sequence_p95, 0.95),
    ):
        if value <= 0:
            continue
        cap = min(
            initial.cap,
            math.ceil(int(value) / _CAP_QUANTUM) * _CAP_QUANTUM,
        )
        if cap >= initial.cap:
            continue
        duplicate = next(
            (index for index, (existing, _, _) in enumerate(rows)
             if existing == cap),
            None,
        )
        if duplicate is not None:
            _cap, old_basis, old_minimum = rows[duplicate]
            rows[duplicate] = (
                cap,
                old_basis.replace("_round_up_", f"_{quantile}_round_up_"),
                min(old_minimum, minimum),
            )
            continue
        rows.append(
            (
                cap,
                f"validated_baseline_{quantile}_round_up_{_CAP_QUANTUM}",
                minimum,
            )
        )
    authority["caps"] = [cap for cap, _, _ in rows]
    return tuple(rows), authority


def _plan_identity(plan: TrainPlan) -> dict[str, Any]:
    batch = int(plan.per_device_batch_size)
    accum = int(plan.grad_accum_steps)
    return {
        "batch": batch,
        "grad_accum": accum,
        "effective_batch": batch * accum,
        "gradient_checkpointing": bool(plan.gradient_checkpointing),
        "seq_cap": int(plan.max_seq_len),
    }


def _record_training_selection(
    plan: TrainPlan, view: _DataView, outcome: str
) -> None:
    """Refresh top-level telemetry to the rung that actually ended training."""
    telemetry.set_meta(
        seq_len=plan.max_seq_len,
        sequence_cap_basis=view.cap_basis,
        tokenized=view.retained_rows,
        rows_retained=view.retained_source_rows,
        retained_source_rows=view.retained_source_rows,
        retained_fraction=view.retained_fraction,
        dataset_ordering_sha256=view.ordering_sha256,
        train_n=len(view.train),
        val_n=len(view.validation),
        batch=plan.per_device_batch_size,
        grad_accum=plan.grad_accum_steps,
        eff_batch=plan.per_device_batch_size * plan.grad_accum_steps,
        gradient_checkpointing=plan.gradient_checkpointing,
        tokens_per_step_cap=(plan.per_device_batch_size
                             * plan.grad_accum_steps * plan.max_seq_len),
        training_outcome=outcome,
    )


def _plans(
    plan: TrainPlan,
    reduced_caps: tuple[int, ...] = (),
    *,
    checkpointing_supported: bool = True,
) -> tuple[TrainPlan, ...]:
    """Ordered survival rungs with one fixed effective batch of 16."""
    requested = min(4, max(1, int(plan.per_device_batch_size)))
    start = next(batch for batch in _BATCHES if batch <= requested)
    first = replace(
        plan,
        per_device_batch_size=start,
        grad_accum_steps=16 // start,
        gradient_checkpointing=(
            plan.gradient_checkpointing if checkpointing_supported else False
        ),
    )
    candidates = [first]
    rescue_gc = checkpointing_supported
    if checkpointing_supported and not first.gradient_checkpointing:
        candidates.append(replace(first, gradient_checkpointing=True))
    for batch in _BATCHES:
        if batch >= start:
            continue
        candidates.append(
            replace(
                first,
                per_device_batch_size=batch,
                grad_accum_steps=16 // batch,
                gradient_checkpointing=rescue_gc,
            )
        )
    for cap in reduced_caps:
        cap = int(cap)
        if cap <= 0 or cap >= first.max_seq_len:
            continue
        candidates.append(
            replace(
                first,
                per_device_batch_size=1,
                grad_accum_steps=16,
                gradient_checkpointing=rescue_gc,
                max_seq_len=cap,
            )
        )
    unique: list[TrainPlan] = []
    identities = set()
    for candidate in candidates:
        identity = (
            candidate.per_device_batch_size,
            candidate.grad_accum_steps,
            candidate.gradient_checkpointing,
            candidate.max_seq_len,
        )
        if identity not in identities:
            identities.add(identity)
            unique.append(candidate)
    return tuple(unique)


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
    max_steps: int | None = None,
    recipe: RecipeOverrides | None = None,
):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    recipe = recipe if recipe is not None else RecipeOverrides()
    kwargs = apply_training_kwargs_overrides(
        build_training_kwargs(
            spec, plan, neftune_alpha=recipe_neftune_alpha(is_kl, recipe)
        ),
        recipe,
    )
    if probe:
        kwargs["max_steps"] = 1
    else:
        if epochs is not None:
            kwargs["num_train_epochs"] = epochs
        if max_steps is not None:
            # Wall-budget cap: the Trainer builds its schedule (cosine floor,
            # warm-up) over exactly this many optimizer steps, so the anneal
            # completes inside the budget instead of being cut by the clock.
            kwargs["max_steps"] = int(max_steps)
    tracker = BestTracker()
    route = None if probe else eligible_qwen35_soup_route(
        spec, model, strategy=strategy, n_gpus=n_gpus,
        capture_root=os.path.join(workdir(spec), "qwen35-r4-r2-captures"))
    if val_ex:
        eff = plan.per_device_batch_size * plan.grad_accum_steps
        kwargs.update(eval_strategy="steps", eval_steps=_eval_every(len(train_ex), eff),
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


def _admission_verdict(peak: float) -> tuple[str, float]:
    """PASS only if the measured peak plus the calibrated headroom fits."""
    predicted = round(float(peak) + _STEADY_STATE_HEADROOM, 6)
    status = "PASS" if predicted <= _ADMISSION_CEILING else "HEADROOM_EXCEEDED"
    return status, predicted


def _admitted_attempt(admission: list, admitted: bool) -> dict[str, Any] | None:
    if not admitted or not admission:
        return None
    attempt = admission[-1]
    return attempt if attempt.get("status") == "PASS" else None


def _admitted_prediction(admission: list, admitted: bool) -> float | None:
    attempt = _admitted_attempt(admission, admitted)
    return None if attempt is None else attempt.get("predicted_steady_state_fraction")


def _admitted_step_s(admission: list, admitted: bool) -> float | None:
    """Wall seconds of the admitted rung's worst-batch single-step probe."""
    attempt = _admitted_attempt(admission, admitted)
    if attempt is None:
        return None
    worst = [
        batch for batch in attempt.get("batches") or []
        if batch.get("label") == "worst" and batch.get("step_wall_s") is not None
    ]
    if not worst:
        return None
    try:
        value = float(worst[-1]["step_wall_s"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _probe_once(
    plan: TrainPlan,
    model: Any,
    rows: list,
    identity: dict[str, Any],
    view: _DataView,
    make: Callable,
) -> dict[str, Any]:
    import torch

    devices = tuple(range(torch.cuda.device_count()))
    trainer = None
    try:
        torch.cuda.empty_cache()
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        trainer = make(plan, model, view, rows)[0]
        model = None
        started = time.monotonic()
        trainer.train()
        step_wall = time.monotonic() - started
        if int(getattr(trainer.state, "global_step", 0) or 0) != 1:
            raise RuntimeError("exact Trainer probe did not complete one optimizer step")
        for device in devices:
            torch.cuda.synchronize(device)
        peak = max(
            torch.cuda.max_memory_reserved(device)
            / torch.cuda.get_device_properties(device).total_memory
            for device in devices)
        status, predicted = _admission_verdict(peak)
        return {**identity, "status": status,
                "peak_reserved_fraction": round(peak, 6),
                "predicted_steady_state_fraction": predicted,
                "headroom": _STEADY_STATE_HEADROOM,
                "admission_ceiling": _ADMISSION_CEILING,
                "step_wall_s": round(step_wall, 3)}
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


def _admit(
    plans: tuple[TrainPlan, ...],
    model: Any,
    data_for: Callable[[TrainPlan], _DataView],
    rebuild: Callable,
    make: Callable,
):
    import torch

    if not torch.cuda.is_available():
        _discard(model)
        view = data_for(plans[0])
        attempt = {
            **_plan_identity(plans[0]),
            **view.identity(),
            "status": "SKIP_NO_CUDA",
        }
        return plans[0], view, 0, [attempt], False
    attempts = []
    current = model
    model = None
    fallback = None
    view = None
    current_cap = None
    for plan_index, candidate in enumerate(plans):
        if current_cap is not None and candidate.max_seq_len != current_cap:
            view = None
            gc.collect()
        view = data_for(candidate)
        current_cap = candidate.max_seq_len
        if not view.authorized:
            attempt = {
                **_plan_identity(candidate),
                **view.identity(),
                "status": "CAP_AUTHORITY_REJECTED",
                "batches": [],
            }
            attempts.append(attempt)
            telemetry.event("geometry_admission_attempt", **attempt)
            continue
        if not view.train:
            attempt = {
                **_plan_identity(candidate),
                **view.identity(),
                "status": "NO_RETAINED_ROWS",
                "batches": [],
            }
            attempts.append(attempt)
            telemetry.event("geometry_admission_attempt", **attempt)
            continue
        fallback = (candidate, plan_index)
        observations = []
        for batch_index, (label, selected, identity) in enumerate(
                _probe_rows(view.train, candidate.per_device_batch_size)):
            if plan_index or batch_index:
                current = rebuild(candidate)
            holder = [current]
            current = None
            observation = _probe_once(
                candidate,
                holder.pop(),
                selected,
                {
                    "label": label,
                    **_plan_identity(candidate),
                    **view.identity(),
                    **identity,
                },
                view,
                make,
            )
            observations.append(observation)
            if observation["status"] != "PASS":
                break
        status = ("PASS" if len(observations) == 2
                  and all(o["status"] == "PASS" for o in observations)
                  else observations[-1]["status"])
        predicted = [
            o["predicted_steady_state_fraction"] for o in observations
            if o.get("predicted_steady_state_fraction") is not None
        ]
        attempt = {
            **_plan_identity(candidate),
            **view.identity(),
            "status": status,
            "headroom": _STEADY_STATE_HEADROOM,
            "admission_ceiling": _ADMISSION_CEILING,
            "predicted_steady_state_fraction": max(predicted) if predicted else None,
            "batches": observations,
        }
        attempts.append(attempt)
        telemetry.event("geometry_admission_attempt", **attempt)
        if status == "PASS":
            return candidate, view, plan_index, attempts, True
    if fallback is None:
        raise RuntimeError("all distribution-authorized SFT geometries were empty")
    candidate, plan_index = fallback
    view = None
    gc.collect()
    view = data_for(candidate)
    telemetry.event(
        "geometry_admission_exhausted_fallback",
        attempts=len(attempts),
        admitted=False,
        **_plan_identity(candidate),
        **view.identity(),
    )
    return candidate, view, plan_index, attempts, False


def _truth_matches(output_dir: str, truth: str, step: int) -> bool:
    try:
        with open(
            os.path.join(output_dir, "forge_artifact_truth.json"),
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("truth") == truth
            and int(payload.get("optimizer_step", -1)) == int(step)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _preserve_progress(trainer: Any, tracker: BestTracker, spec: TaskSpec,
                       tokenizer: Any, step: int) -> bool:
    if tracker.persisted_best is not None:
        selected_step = int(tracker.persisted_best_step or 0)
        written = write_artifact_truth(
            spec.output_dir,
            ARTIFACT_PARTIAL_TRAINED_BEST,
            optimizer_step=selected_step,
            reason="progressed_oom_retained_best",
        )
        return written and _truth_matches(
            spec.output_dir, ARTIFACT_PARTIAL_TRAINED_BEST, selected_step
        )
    try:
        save_adapter(
            trainer.model, tokenizer, spec.output_dir,
            artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST,
            optimizer_step=step, truth_reason="progressed_oom_latest")
        return _truth_matches(
            spec.output_dir, ARTIFACT_PARTIAL_TRAINED_BEST, step
        )
    except Exception as exc:
        telemetry.event("progressed_oom_export_failed", error=repr(exc), step=step)
        return False


def _train_ladder(
    plans: tuple[TrainPlan, ...],
    data_for: Callable[[TrainPlan], _DataView],
    rebuild: Callable,
    make: Callable,
    measure_epochs: Callable[[TrainPlan, _DataView], float | None],
    spec: TaskSpec,
    tokenizer: Any,
    *,
    plan_steps: Callable[[TrainPlan, _DataView, float | None], int | None] | None = None,
):
    last_plan = None
    view = None
    current_cap = None
    for index, candidate in enumerate(plans):
        if current_cap is not None and candidate.max_seq_len != current_cap:
            view = None
            gc.collect()
        view = data_for(candidate)
        current_cap = candidate.max_seq_len
        if not view.authorized:
            telemetry.event(
                "training_geometry_attempt",
                status="CAP_AUTHORITY_REJECTED",
                **_plan_identity(candidate),
                **view.identity(),
            )
            continue
        if not view.train:
            telemetry.event(
                "training_geometry_attempt",
                status="NO_RETAINED_ROWS",
                **_plan_identity(candidate),
                **view.identity(),
            )
            continue
        last_plan = candidate
        trainer = None
        tracker = None
        route = None
        current = None
        try:
            epochs = measure_epochs(candidate, view)
            current = rebuild(candidate)
            step_cap = None if plan_steps is None else plan_steps(candidate, view, epochs)
            if step_cap is None:
                trainer, tracker, route = make(candidate, current, view, None, epochs)
            else:
                trainer, tracker, route = make(
                    candidate, current, view, None, epochs, max_steps=step_cap
                )
            current = None
            trainer.train()
            telemetry.event(
                "training_geometry_attempt",
                status="PASS",
                **_plan_identity(candidate),
                **view.identity(),
            )
            return trainer, tracker, route, candidate, view, "trained"
        except Exception as exc:
            if not _is_oom(exc):
                raise
            step = int(getattr(getattr(trainer, "state", None),
                               "global_step", 0) or 0)
            if step:
                preserved = _preserve_progress(
                    trainer, tracker, spec, tokenizer, step
                )
                status = (
                    "PROGRESSED_OOM_PRESERVED"
                    if preserved else "PROGRESSED_OOM_EXPORT_FAILED"
                )
                telemetry.event(
                    "training_geometry_attempt",
                    status=status,
                    optimizer_step=step,
                    **_plan_identity(candidate),
                    **view.identity(),
                )
                return (
                    None,
                    tracker,
                    route,
                    candidate,
                    view,
                    (
                        "progressed_oom_preserved"
                        if preserved else "progressed_oom_export_failed"
                    ),
                )
            telemetry.event(
                "training_geometry_attempt",
                status="ZERO_PROGRESS_OOM",
                **_plan_identity(candidate),
                **view.identity(),
            )
            if trainer is not None:
                holder = [trainer]
                trainer = None
                _discard(holder.pop(), trainer=True)
            else:
                holder = [current]
                current = None
                _discard(holder.pop())
    if last_plan is None:
        raise RuntimeError("training ladder contained no nonempty dataset view")
    view = None
    gc.collect()
    terminal_view = data_for(last_plan)
    return (
        None,
        None,
        None,
        last_plan,
        terminal_view,
        "zero_progress_exhausted",
    )


def _eval_every(train_rows: int, eff_batch: int) -> int:
    """The unchanged evaluation cadence: about four evaluations per epoch."""
    return max(1, (int(train_rows) // max(1, int(eff_batch))) // 4)


def _steps_per_epoch(
    train_rows: int, per_device_batch: int, grad_accum: int, n_gpus: int = 1
) -> int:
    """Optimizer steps per epoch exactly as the Trainer derives them."""
    sampler_batch = max(1, int(per_device_batch)) * max(1, int(n_gpus))
    batches = math.ceil(max(0, int(train_rows)) / sampler_batch)
    accum = max(1, int(grad_accum))
    return max(1, batches // accum + int(batches % accum > 0))


def _plan_wall_steps(
    *,
    budget_s: float,
    step_s: float | None,
    step_source: str,
    train_rows: int,
    val_rows: int,
    per_device_batch: int,
    grad_accum: int,
    epochs: float,
    n_gpus: int = 1,
) -> dict[str, Any]:
    """Cap the optimizer-step schedule so it completes inside the wall budget.

    ``budget_s`` is ``deadline.remaining()``: seconds to the soft stop, which
    already excludes the export reserve.  ``step_s`` is a measured seconds per
    optimizer step.  The schedule is the Trainer's own step count for
    ``epochs``; the cap only ever lowers it, so a run that fits keeps its epoch
    plan untouched, and an unmeasured run is left to the deadline callback.
    """
    batch = max(1, int(per_device_batch))
    accum = max(1, int(grad_accum))
    eff = batch * accum
    per_epoch = _steps_per_epoch(train_rows, batch, accum, n_gpus)
    schedule = max(1, math.ceil(float(epochs) * per_epoch))
    every = _eval_every(train_rows, eff)
    plan: dict[str, Any] = {
        "schedule_steps": schedule,
        "steps_per_epoch": per_epoch,
        "eval_every": every,
        "eval_rows": int(val_rows),
        "measured_step_s": None,
        "step_source": step_source,
        "budget_s": round(float(budget_s), 1),
        "setup_s": _WALL_PLAN_SETUP_S,
        "margin": _WALL_PLAN_MARGIN,
        "eval_row_factor": _WALL_PLAN_EVAL_ROW_FACTOR,
        "estimated_eval_s": None,
        "affordable_steps": None,
        "estimated_wall_s": None,
        "planned_steps": schedule,
        "cap_applied": False,
        "reason": "unmeasured_step_time",
    }
    try:
        step = float(step_s)
    except (TypeError, ValueError):
        return plan
    if not math.isfinite(step) or step <= 0:
        return plan
    plan["measured_step_s"] = round(step, 4)
    eval_s = (
        step * (int(val_rows) / eff) * _WALL_PLAN_EVAL_ROW_FACTOR
        if int(val_rows) > 0 else 0.0
    )
    plan["estimated_eval_s"] = round(eval_s, 2)
    usable = _usable_budget_s(budget_s)

    def cost(steps: int) -> float:
        return _wall_cost(steps, step_s=step, eval_s=eval_s, eval_every=every)

    if usable < cost(1):
        plan["affordable_steps"] = 0
        plan["reason"] = "budget_below_one_step"
        return plan
    affordable = max(1, int(usable // (step + eval_s / every)))
    while affordable > 1 and cost(affordable) > usable:
        affordable -= 1
    while cost(affordable + 1) <= usable:
        affordable += 1
    plan["affordable_steps"] = affordable
    if affordable >= schedule:
        plan["estimated_wall_s"] = round(cost(schedule), 1)
        plan["reason"] = "schedule_fits"
        return plan
    plan.update(
        planned_steps=affordable,
        cap_applied=True,
        estimated_wall_s=round(cost(affordable), 1),
        reason="wall_budget_cap",
    )
    return plan


def _usable_budget_s(budget_s: float) -> float:
    return max(0.0, float(budget_s) - _WALL_PLAN_SETUP_S) * _WALL_PLAN_MARGIN


def _wall_cost(steps: int, *, step_s: float, eval_s: float, eval_every: int) -> float:
    # Every `eval_every` steps evaluates, and the Trainer evaluates once more
    # at the final step unless that step is already an evaluation step.
    return steps * step_s + math.ceil(steps / max(1, eval_every)) * eval_s


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _step_floor(schedule_steps: int) -> int:
    """Never plan fewer real steps than this while the estimate affords them."""
    schedule = max(1, int(schedule_steps))
    return min(
        schedule,
        max(_MIN_REAL_STEPS, math.ceil(_FLOOR_SCHEDULE_FRACTION * schedule)),
    )


def _plan_cold_only(
    *,
    budget_s: float,
    cold_step_s: float,
    warm_step_s: float | None,
    train_rows: int,
    val_rows: int,
    per_device_batch: int,
    grad_accum: int,
    epochs: float,
    n_gpus: int = 1,
) -> dict[str, Any]:
    """Plan when the admission probe's cold worst-batch step is all we have.

    The admission timing is one micro-batch of the longest rows plus any cold
    start, so it is trusted only when no warmer measurement exists and the
    schedule fits under it anyway.  A warm-probe measurement always wins; when
    the cold step would cap the run without one, it is discounted; either way
    a capped plan never drops below the certification floor while the
    discounted estimate affords it.
    """
    geometry = dict(
        train_rows=train_rows, val_rows=val_rows,
        per_device_batch=per_device_batch, grad_accum=grad_accum,
        epochs=epochs, n_gpus=n_gpus,
    )
    cold = _plan_wall_steps(
        budget_s=budget_s, step_s=cold_step_s,
        step_source="admission_worst_batch", **geometry,
    )
    cold_fields = {
        "cold_step_s": round(float(cold_step_s), 4),
        "cold_planned_steps": cold["planned_steps"],
        "cold_reason": cold["reason"],
        "warm_step_s": (
            None if _finite_positive(warm_step_s) is None
            else round(_finite_positive(warm_step_s), 4)
        ),
        "floor_steps": None,
        "discounted_step_s": None,
        "discounted_affordable_steps": None,
    }
    warm = _finite_positive(warm_step_s)
    if warm is None and not cold["cap_applied"]:
        cold.update(cold_fields)
        return cold
    discounted_step = float(cold_step_s) / _COLD_PROBE_DISCOUNT
    discounted = _plan_wall_steps(
        budget_s=budget_s, step_s=discounted_step,
        step_source="admission_worst_batch_discounted", **geometry,
    )
    if warm is not None:
        # A measured optimizer step beats the admission micro-batch in either
        # direction (qwen3.5-9b: 1.786 s micro-batch, 7.5-23 s real steps).
        plan = _plan_wall_steps(
            budget_s=budget_s, step_s=warm, step_source="warm_probe", **geometry,
        )
    else:
        plan = discounted
    floor = _step_floor(plan["schedule_steps"])
    plan.update(
        cold_fields,
        floor_steps=floor,
        discounted_step_s=round(discounted_step, 4),
        discounted_affordable_steps=discounted["affordable_steps"],
    )
    affordable = discounted["affordable_steps"] or 0
    if plan["cap_applied"] and plan["planned_steps"] < floor and affordable >= floor:
        every = plan["eval_every"]
        plan["estimated_wall_s"] = round(_wall_cost(
            floor, step_s=discounted_step,
            eval_s=discounted["estimated_eval_s"] or 0.0, eval_every=every,
        ), 1)
        if floor >= plan["schedule_steps"]:
            plan.update(
                planned_steps=plan["schedule_steps"], cap_applied=False,
                reason="schedule_fits_by_floor",
            )
        else:
            plan.update(planned_steps=floor, reason="wall_budget_cap_floor")
    return plan


def _warm_probe_decision(
    *,
    budget_s: float,
    cold_step_s: float,
    train_rows: int,
    val_rows: int,
    per_device_batch: int,
    grad_accum: int,
    epochs: float,
    n_gpus: int = 1,
) -> dict[str, Any]:
    """Run the warm probe whenever a few cold-priced steps fit inside a small
    slice of the remaining budget; ``needed`` only records whether the cold
    step alone would have capped the run (it under-read 4-13x on qwen3.5-9b,
    so a fitting cold plan is not a reason to skip the measurement)."""
    cold = _plan_wall_steps(
        budget_s=budget_s, step_s=cold_step_s,
        step_source="admission_worst_batch", train_rows=train_rows,
        val_rows=val_rows, per_device_batch=per_device_batch,
        grad_accum=grad_accum, epochs=epochs, n_gpus=n_gpus,
    )
    needed = bool(cold["cap_applied"])
    cold_step = _finite_positive(cold_step_s) or 0.0
    cost_estimate = _WARM_PROBE_STEPS * cold_step
    allowance = _WARM_PROBE_BUDGET_FRACTION * max(0.0, float(budget_s))
    affordable = cold_step > 0 and cost_estimate <= allowance
    return {
        "run": bool(affordable),
        "needed": needed,
        "affordable": affordable,
        "probe_steps": _WARM_PROBE_STEPS,
        "cost_estimate_s": round(cost_estimate, 1),
        "allowance_s": round(allowance, 1),
        "stop_after_s": round(allowance, 1),
        "budget_s": round(float(budget_s), 1),
        "cold_step_s": round(float(cold_step_s), 4),
        "cold_planned_steps": cold["planned_steps"],
        "cold_reason": cold["reason"],
    }


def _warm_probe_median(durations: list[float]) -> float | None:
    """Median of the second and third steps; the first carries the warm-up."""
    warm = [
        float(d) for d in list(durations)[1:_WARM_PROBE_STEPS]
        if isinstance(d, (int, float)) and math.isfinite(float(d)) and float(d) > 0
    ]
    if not warm:
        return None
    return round(statistics.median(warm), 4)


def _warm_probe_step_s(
    *,
    trainer_cls: Any,
    model: Any,
    kwargs: dict[str, Any],
    train_ex: list,
    collator: Any,
    trainer_extra: dict[str, Any] | None = None,
    stop_after_s: float | None = None,
) -> dict[str, Any]:
    """Time a few real optimizer steps on an already-built model, then discard.

    Same construction as the zero-LR timing probe (exact production arguments,
    LR 0 and no NEFTune so nothing moves, no evaluation, no callbacks that
    write artifacts), stopped after ``_WARM_PROBE_STEPS`` optimizer steps or
    once ``stop_after_s`` of wall time has elapsed.  The caller releases the
    model afterwards; nothing here is reused.
    """
    result: dict[str, Any] = {
        "source": "warm_probe",
        "probe_steps": _WARM_PROBE_STEPS,
        "stop_after_s": None if stop_after_s is None else round(float(stop_after_s), 1),
        "steps_completed": 0,
        "step_durations_s": [],
        "elapsed_s": None,
        "step_s": None,
    }
    try:
        from datasets import Dataset
        from transformers import TrainerCallback, TrainingArguments
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    durations: list[float] = []
    opened = time.monotonic()

    def expired(now: float) -> bool:
        return stop_after_s is not None and now - opened >= float(stop_after_s)

    class _StepTimer(TrainerCallback):
        started: float | None = None

        def on_step_begin(self, args, state, control, **kw):  # noqa: ANN001
            self.started = time.monotonic()
            return control

        def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
            now = time.monotonic()
            if self.started is not None:
                durations.append(now - self.started)
                self.started = None
            if len(durations) >= _WARM_PROBE_STEPS or expired(now):
                control.should_training_stop = True
            return control

        def on_substep_end(self, args, state, control, **kw):  # noqa: ANN001
            if expired(time.monotonic()):
                control.should_training_stop = True
            return control

    probe_kwargs = dict(kwargs)
    probe_kwargs.update(
        max_steps=_WARM_PROBE_STEPS, learning_rate=0.0, neftune_noise_alpha=None
    )
    for key in ("eval_strategy", "eval_steps", "per_device_eval_batch_size"):
        probe_kwargs.pop(key, None)
    probe = None
    try:
        probe = trainer_cls(
            model=model,
            args=TrainingArguments(
                **compatible_dataclass_kwargs(
                    TrainingArguments, probe_kwargs,
                    allow_removed={"overwrite_output_dir"},
                )
            ),
            train_dataset=Dataset.from_list(train_ex),
            data_collator=collator,
            callbacks=[_StepTimer()],
            **(trainer_extra or {}),
        )
        probe.train()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if probe is not None:
            holder = [probe]
            probe = None
            _discard(holder.pop(), trainer=True)
        else:
            _free_cuda()
    result["elapsed_s"] = round(time.monotonic() - opened, 3)
    result["steps_completed"] = len(durations)
    result["step_durations_s"] = [round(d, 3) for d in durations]
    result["step_s"] = _warm_probe_median(durations)
    return result


def _record_wall_plan(wall: dict[str, Any], plan: TrainPlan, view: _DataView) -> None:
    # Merged explicitly: the plan's keys are disjoint from the identities (a
    # test pins that), and a diagnostic must never raise inside the ladder.
    telemetry.event(
        "wall_budget_plan", **{**view.identity(), **_plan_identity(plan), **wall}
    )
    telemetry.set_meta(
        planned_steps=wall["planned_steps"],
        measured_step_s=wall["measured_step_s"],
        budget_s=wall["budget_s"],
        plan_reason=wall["reason"],
        wall_plan=wall,
    )


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
