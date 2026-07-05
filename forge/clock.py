"""Wall-clock accounting.

The validator kills the container at `hours_to_complete`. Everything we do is
paced against that hard stop: we keep a running estimate of per-step cost, and
we always leave an export reserve so the final model is written and closed
before the kill can land. This module is pure bookkeeping — no ML imports — so
it is cheap to construct and easy to test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Deadline:
    """Tracks remaining time and answers pacing questions.

    All times are seconds on `time.monotonic()`'s clock. `hard_stop` is when the
    validator kills us; `soft_stop` is `hard_stop` minus the export reserve, and
    is the moment we must stop training and begin writing the final artifact.
    """

    hard_stop: float
    export_reserve_s: float
    _step_costs: list[float] = field(default_factory=list)

    @classmethod
    def from_hours(
        cls, hours: float, *, started_monotonic: float, export_reserve_s: float
    ) -> "Deadline":
        return cls(
            hard_stop=started_monotonic + hours * 3600.0,
            export_reserve_s=export_reserve_s,
        )

    @property
    def soft_stop(self) -> float:
        return self.hard_stop - self.export_reserve_s

    def remaining(self) -> float:
        """Seconds until we must stop training (soft stop), never negative."""
        return max(0.0, self.soft_stop - time.monotonic())

    def remaining_hard(self) -> float:
        return max(0.0, self.hard_stop - time.monotonic())

    def record_step(self, seconds: float) -> None:
        # Keep a short trailing window; early steps are warm-up-noisy and later
        # steps reflect true throughput once caches are hot.
        self._step_costs.append(seconds)
        if len(self._step_costs) > 50:
            self._step_costs.pop(0)

    def per_step(self) -> float | None:
        if not self._step_costs:
            return None
        # Median over the trailing window: robust to a stray slow step (eval,
        # checkpoint, GC) without waiting for a full recompute.
        s = sorted(self._step_costs)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

    def affordable_steps(self, *, safety: float = 0.85) -> int | None:
        """How many more optimizer steps fit before the soft stop, given
        measured throughput. `safety` discounts for eval/save overhead we
        haven't separately modelled yet. Returns None until we have timing.
        """
        cost = self.per_step()
        if cost is None or cost <= 0:
            return None
        return int(self.remaining() * safety / cost)

    def should_stop(self) -> bool:
        return self.remaining() <= 0.0
