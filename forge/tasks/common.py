"""Shared trainer setup: argument construction, periodic saving, finalisation.

Kill-safety strategy: we disable the HF Trainer's own checkpointing and instead
mirror the current adapter into the mandated output path every few steps. That
keeps exactly one clean adapter at `spec.output_dir` at all times — so if the
wall-clock kill lands, the uploader finds the latest model and no stale
`checkpoint-*` subdirectory can shadow it.
"""

from __future__ import annotations

import os
from typing import Any

from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tuning.plan import TrainPlan


def workdir(spec: TaskSpec) -> str:
    d = f"/app/checkpoints/{spec.task_id}/_work"
    os.makedirs(d, exist_ok=True)
    return d


def build_training_kwargs(
    spec: TaskSpec, plan: TrainPlan, *, neftune_alpha: float | None = None
) -> dict[str, Any]:
    """Common TrainingArguments fields shared by SFT/DPO/GRPO configs.

    `neftune_alpha` enables NEFTune input-embedding noise (instruction-tuning
    regulariser). Left None on the KL path, where the noise would otherwise
    contaminate the disable_adapter base reference.
    """
    kwargs = dict(
        output_dir=workdir(spec),
        overwrite_output_dir=True,
        num_train_epochs=plan.num_epochs,
        per_device_train_batch_size=plan.per_device_batch_size,
        gradient_accumulation_steps=plan.grad_accum_steps,
        learning_rate=plan.learning_rate,
        lr_scheduler_type=plan.lr_scheduler,
        # Floor the cosine at 25% of peak when using the floored scheduler.
        lr_scheduler_kwargs=(
            {"min_lr_rate": 0.25}
            if plan.lr_scheduler == "cosine_with_min_lr"
            else {}
        ),
        neftune_noise_alpha=neftune_alpha,
        warmup_ratio=plan.warmup_ratio,
        weight_decay=plan.weight_decay,
        optim=plan.optimizer,
        max_grad_norm=1.0,
        bf16=plan.bf16,
        fp16=plan.fp16,
        gradient_checkpointing=plan.gradient_checkpointing,
        # PEFT + gradient checkpointing needs non-reentrant to keep adapter grads.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",  # we mirror to output_dir ourselves (see below)
        report_to=[],  # wandb is offline via env; don't import integrations
        remove_unused_columns=False,
        dataloader_num_workers=2,
        disable_tqdm=True,
        seed=7,
    )
    return kwargs


def safe_train(trainer: Any, *, min_batch: int = 1) -> None:
    """Run trainer.train(), retrying once at a smaller batch on CUDA OOM.

    `min_batch` floors the retry micro-batch: GRPO requires the batch to stay a
    multiple of num_generations, so it passes min_batch=num_generations. Pairs
    with the eager floor save the handlers do before training: if even the retry
    fails, the exception propagates to the CLI, but a valid (untrained) adapter
    already exists at the output path, so we get the floor rather than a forfeit.
    """
    try:
        trainer.train()
        return
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
    from forge import telemetry

    telemetry.event("oom_retry")
    _free_cuda()
    _clear_neftune_hook(trainer)  # a NEFTune hook from the aborted run persists
    args = trainer.args
    cur = getattr(args, "per_device_train_batch_size", 1)
    if cur > min_batch:
        # Preserve the effective batch and, for GRPO, num_generations divisibility.
        args.gradient_accumulation_steps = max(
            1, getattr(args, "gradient_accumulation_steps", 1) * (cur // min_batch)
        )
        args.per_device_train_batch_size = min_batch
    trainer.train()


def _clear_neftune_hook(trainer: Any) -> None:
    """Remove a NEFTune forward hook left attached when a run aborts mid-training
    (HF only detaches it on the normal return path, not on exception), so the
    retry doesn't stack a second noise hook on the embeddings.
    """
    try:
        handle = getattr(trainer, "neftune_hook_handle", None)
        if handle is not None:
            handle.remove()
            trainer.neftune_hook_handle = None
    except Exception:
        pass


def _is_oom(exc: Exception) -> bool:
    import torch

    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_adapter(model: Any, tokenizer: Any, output_dir: str) -> None:
    # Write to a sibling temp dir, then swap directories with atomic renames so
    # the path the uploader reads is always a complete adapter — never a
    # half-written one, even if a kill lands mid-save.
    final = output_dir.rstrip("/")
    tmp = final + ".tmp"
    old = final + ".old"
    _rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)

    # Carry the flight recorder INTO the staging dir *before* it goes live, so the
    # swapped-in directory already contains forge_run.json the instant it becomes
    # the adapter the uploader reads. This makes the log exactly as kill-safe as
    # the weights: there is no window where `final` has an adapter but no log,
    # even if a SIGKILL lands mid-swap.
    from forge import telemetry

    telemetry.write_into(tmp)

    _rmtree(old)
    if os.path.isdir(final):
        os.rename(final, old)  # atomic: move the complete old dir aside
    os.rename(tmp, final)  # atomic: the new complete dir (adapter + log) becomes live
    _rmtree(old)


def _rmtree(path: str) -> None:
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _make_periodic_save_callback(spec: TaskSpec, tokenizer: Any, *, every: int = 25):
    """Mirror the adapter into the output path every `every` optimizer steps.

    Built as a TrainerCallback subclass at call time to keep this module usable
    (for arg/save helpers) even where transformers isn't importable. Keeps the
    latest model at the mandated output path so a wall-clock kill always uploads
    the most recent trained adapter.
    """
    from transformers import TrainerCallback

    step = max(1, every)

    class PeriodicSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if state.global_step > 0 and state.global_step % step == 0:
                model = kwargs.get("model")
                if model is not None:
                    try:
                        save_adapter(model, tokenizer, spec.output_dir)
                    except Exception:
                        pass  # a failed mirror must never stop training
            return control

    return PeriodicSaveCallback()
