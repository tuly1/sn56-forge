"""Best-checkpoint selection: submit the lowest-eval-loss model, not the last.

With the floored cosine schedule the model keeps moving at a real learning rate
until the deadline, so the final checkpoint is *more* likely to have drifted past
its best point. We evaluate periodically on a small held-out split and mirror the
best-so-far adapter straight into the mandated output path. That also upgrades
kill-safety: a wall-clock kill now uploads the best-known model.

On KL tasks the eval loss flows through KLSFTTrainer.compute_loss, i.e. the
checkpoint is selected by CE + kl_coef*KL — the exact quantity the grader ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback

from forge.data.schema import TaskSpec
from forge.tasks.common import save_adapter


@dataclass
class BestTracker:
    """Shared state between the best-checkpoint callback, the periodic mirror
    (which stands down once a best exists), and the end-of-training decision."""

    best_loss: float | None = None
    saved_best: bool = False


class BestCheckpointCallback(TrainerCallback):
    def __init__(self, spec: TaskSpec, tokenizer: Any, tracker: BestTracker) -> None:
        self._spec = spec
        self._tokenizer = tokenizer
        self._tracker = tracker

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        loss = (metrics or {}).get("eval_loss")
        if loss is None or (isinstance(loss, float) and math.isnan(loss)):
            return control
        if self._tracker.best_loss is not None and loss >= self._tracker.best_loss:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        try:
            save_adapter(model, self._tokenizer, self._spec.output_dir)
        except Exception:
            return control  # a failed mirror must never stop training
        self._tracker.best_loss = float(loss)
        self._tracker.saved_best = True
        return control
