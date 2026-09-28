"""Legacy adapter fixes for the in-scope crash fallback and crash exceptions (release-fullft-default-20260929).

The legacy LoRA handler (forge.tasks.instruct) is now only the last-resort rung of the full-FT ladder and the route of
crash exceptions. The 09-28 R1 post-mortem (task 803a8f69, memory sn56-lfm-0928-postmortem) found that its schedule
stretched 2 -> 4 epochs on 15k rows (time_aware_epochs) and exported at ~65% of peak LR (the cosine floored at 0.25 and
cut by the clock). This module applies the four fixes by reusing existing pieces:

* anneal to ~0: plain cosine (the time-aware epoch plan already sizes the schedule to end at the deadline);
* epoch cap on small data: at most SMALL_DATA_MAX_EPOCHS when the train split has < SMALL_DATA_ROWS rows;
* early stop: stop after EARLY_STOP_PATIENCE evaluations without a new best (the best checkpoint is already exported
  the moment it is observed by common._make_best_checkpoint_callback);
* checkpoint averaging: forge.tuning.selection.SnapshotPool at every evaluation + greedy_soup (monotone: never worse
  than the best single snapshot on the held-out slice, under the validator statistic).

Every piece is best-effort: a failure logs an event and keeps the incumbent artifact.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

SMALL_DATA_ROWS = 20000
SMALL_DATA_MAX_EPOCHS = 2.0
EARLY_STOP_PATIENCE = 3
SOUP_POOL_K = 4
SOUP_MAX_EVAL_ROWS = 512


def apply_plan(plan: Any) -> Any:
    """Anneal to ~0: cosine without the 0.25 floor."""
    return replace(plan, lr_scheduler="cosine")


def cap_epochs(epochs: float | None, n_train: int, default_epochs: float) -> float | None:
    """Epochs for the legacy schedule. None (the time-aware probe was skipped) keeps the plan default, itself capped."""
    if n_train >= SMALL_DATA_ROWS:
        return epochs
    if epochs is None:
        return SMALL_DATA_MAX_EPOCHS if float(default_epochs) > SMALL_DATA_MAX_EPOCHS else None
    return min(float(epochs), SMALL_DATA_MAX_EPOCHS)


def make_callbacks(pool: Any) -> list:
    """Early stop + snapshot capture, both driven by the trainer's eval_loss."""
    from transformers import TrainerCallback

    from forge import telemetry

    class LegacyFixesCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best: float | None = None
            self.stale = 0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):  # noqa: ANN001
            try:
                loss = float((metrics or {}).get("eval_loss"))
            except (TypeError, ValueError):
                return control
            if not math.isfinite(loss):
                return control
            step = int(state.global_step)
            if model is not None:
                try:
                    pool.consider(model, loss, step)
                except Exception as exc:  # noqa: BLE001
                    telemetry.event("legacy_fixes_snapshot_failed", step=step, error=f"{type(exc).__name__}: {exc}")
            if self.best is None or loss < self.best - 1e-6:
                self.best, self.stale = loss, 0
            else:
                self.stale += 1
                if self.stale >= EARLY_STOP_PATIENCE:
                    control.should_training_stop = True
                    telemetry.event("legacy_fixes_early_stop", step=step, best=round(self.best, 6),
                                    stale_evals=self.stale)
            return control

    return [LegacyFixesCallback()]


def soup_and_save(model: Any, pool: Any, dev: list, collator: Any, *, save: Any, time_budget_s: float) -> bool:
    """Greedy soup over the captured snapshots; ``save(model)`` persists it when the soup has > 1 member and beat the
    best single snapshot. Returns True when a soup was saved."""
    from forge import telemetry
    from forge.tuning import selection

    if len(getattr(pool, "items", []) or []) < 2 or not dev:
        return False
    try:
        import torch

        autocast = bool(torch.cuda.is_available())
        rows = dev[:SOUP_MAX_EVAL_ROWS]

        def eval_fn(m: Any) -> float:
            m.eval()
            loss, _ = selection.per_example_dev_loss(m, rows, collator, batch_size=1, autocast_bf16=autocast)
            return loss

        seed = pool.best()
        loss, state, members = selection.greedy_soup(
            model, pool, eval_fn, time_budget_s=time_budget_s,
            log=lambda n, d: telemetry.event("legacy_fixes_" + n, **{k: v for k, v in d.items()
                                                                      if isinstance(v, (int, float, str, bool))}))
        telemetry.event("legacy_fixes_soup", members=members, soup_loss=round(float(loss), 6),
                        seed_step=int(seed["step"]) if seed else None)
        if members > 1 and state is not None:
            save(model)
            return True
    except Exception as exc:  # noqa: BLE001
        telemetry.event("legacy_fixes_soup_failed", error=f"{type(exc).__name__}: {exc}")
    return False
