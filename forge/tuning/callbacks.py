"""A training callback that paces against the wall clock.

The validator kills the container at `hours_to_complete` with no grace. We feed
every optimizer step's real duration into the Deadline and stop training once
there isn't enough time left to take another step *and* write the model. Because
the container is treated as a success on timeout and whatever is on disk is
uploaded, we also let the Trainer checkpoint periodically as a safety net.
"""

from __future__ import annotations

import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from forge.clock import Deadline


class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline: Deadline) -> None:
        self._deadline = deadline
        self._step_started: float | None = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._step_started = time.monotonic()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._step_started is not None:
            self._deadline.record_step(time.monotonic() - self._step_started)

        # Stop while at least one measured step plus a margin still fits before
        # the soft stop, so the final export lands inside the export reserve.
        per_step = self._deadline.per_step() or 0.0
        if self._deadline.remaining() <= per_step * 1.5:
            if not control.should_training_stop:
                from forge import telemetry

                telemetry.event(
                    "deadline_stop",
                    step=int(state.global_step),
                    remaining_s=round(self._deadline.remaining(), 1),
                    per_step_s=round(per_step, 2),
                )
            control.should_training_stop = True

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Gradient-accumulation sub-steps: bail hard if the soft stop has already
        # passed, so a large accumulation window can't run us into the kill.
        if self._deadline.should_stop():
            control.should_training_stop = True
