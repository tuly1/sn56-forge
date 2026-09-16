"""Pure SFT learning-rate decisions; no training, imports of ML frameworks, or I/O.

Original implementation of two documented mathematical rules from the public
July 6 position-one trainer. See PROVENANCE.md for the immutable reference.
Complete-evidence checks are our proposed integration safeguards, not claims
about that trainer. No production caller imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

WARMUP_STEPS = 8
EDGE_TOLERANCE = 0.025


def _finite_number(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class CurvatureResult:
    status: str
    factor: float | None = None
    relative_drop: float | None = None
    start_mean: float | None = None
    end_mean: float | None = None
    recorded_steps: int = 0
    window_size: int = 0


def warmup_curvature(losses: Sequence[float], *, require_complete: bool = True) -> CurvatureResult:
    """Map at most eight ordered optimizer-step SFT losses to a bounded factor.

    On finite nonnegative observations the formula is the published rule:
    average the first/last floor(n/3) points, compute relative improvement using
    max(abs(start), 1e-6), then clip 1 + 4*(improvement - .04) to [.6, 1.6].

    Proposed safeguards: nonfinite/negative/nonnumeric values do not get silently
    dropped, and by default an incomplete eight-step warmup has no recommendation.
    require_complete=False permits the reference's minimum of three points for
    explicit arithmetic comparisons; it does not manufacture missing evidence.
    A finite rising loss still maps to the bounded rule, not an invented .3 floor.
    """
    points = tuple(losses)
    n = len(points)
    if n > WARMUP_STEPS:
        raise ValueError("Supply one warmup of at most eight optimizer steps")
    if any(not _finite_number(x) or x < 0 for x in points):
        return CurvatureResult("invalid_loss", recorded_steps=n)
    if n < 3 or (require_complete and n != WARMUP_STEPS):
        return CurvatureResult("insufficient_steps", recorded_steps=n)
    width = n // 3
    # Scale before summing: the mean stays finite even for very large finite CE.
    first = math.fsum(float(x) / width for x in points[:width])
    last = math.fsum(float(x) / width for x in points[-width:])
    denominator = max(abs(first), 1e-6)
    relative = (first - last) / denominator
    # Underflow-scale denominators and huge finite losses can overflow a ratio.
    # Its sign already determines the exact clipped result; never emit NaN/Inf.
    if not math.isfinite(relative):
        return CurvatureResult("numerical_overflow", recorded_steps=n, window_size=width)
    multiplier = min(1.6, max(0.6, 0.84 + 4.0 * relative))
    return CurvatureResult("ready", multiplier, relative, first, last, n, width)


@dataclass(frozen=True)
class ProbeSummary:
    """One SFT probe; loss is its finite tail-median training loss.

    comparison_id must bind the initial weights, data/order/mask, optimizer and
    geometry. It excludes LR, the variable being compared. This pure function
    cannot verify the caller's hashes or actual restoration. planned_steps is
    the same tier's total optimizer-step horizon, including its agreed ramp.
    """
    probe_id: str
    learning_rate: float
    loss: float | None
    planned_steps: int
    completed_steps: int
    comparison_id: str
    state_restored: bool
    status: str = "complete"  # complete, diverged, incomplete
    metric: str = "sft_tail_median"


@dataclass(frozen=True)
class EdgeResult:
    status: str
    selected_id: str | None = None
    learning_rate: float | None = None
    selected_loss: float | None = None
    best_loss: float | None = None
    threshold: float | None = None
    eligible_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()


def select_stability_edge(probes: Sequence[ProbeSummary], *, tolerance: float = EDGE_TOLERANCE) -> EdgeResult:
    """Highest LR within tolerance of the best comparable SFT score (<=4 probes).

    This reproduces the reference edge arithmetic on a valid comparison tier.
    Proposed safeguards reject corrupt/mixed evidence; explicitly diverged arms
    are excluded. At least two completed, finite arms are needed to recommend an
    edge. No hidden fallback LR is returned on failure. This rule is not for DPO
    or GRPO, for which the public reference uses different metrics/selection.
    """
    arms = tuple(probes)
    if len(arms) > 4:
        raise ValueError("At most four probe summaries are allowed")
    if not _finite_number(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be finite and between zero and one")
    if not arms:
        return EdgeResult("insufficient_probes")
    if any(not isinstance(p, ProbeSummary) for p in arms):
        raise TypeError("Every arm must be a ProbeSummary")
    if any(not isinstance(p.probe_id, str) or not p.probe_id.strip()
           or not isinstance(p.comparison_id, str) or not p.comparison_id.strip() for p in arms):
        return EdgeResult("missing_identity")
    if len({p.probe_id for p in arms}) != len(arms):
        return EdgeResult("duplicate_probe_id")
    if any(not _finite_number(p.learning_rate) or p.learning_rate <= 0 for p in arms):
        return EdgeResult("invalid_learning_rate")
    if len({p.learning_rate for p in arms}) != len(arms):
        return EdgeResult("duplicate_learning_rate")
    if any(p.metric != "sft_tail_median" for p in arms):
        return EdgeResult("unsupported_metric")
    if any(type(p.planned_steps) is not int or type(p.completed_steps) is not int
           or p.planned_steps < 3 or not 0 <= p.completed_steps <= p.planned_steps for p in arms):
        return EdgeResult("invalid_step_counts")
    if len({(p.comparison_id, p.planned_steps) for p in arms}) != 1:
        return EdgeResult("incomparable_probes")
    if any(p.state_restored is not True for p in arms):
        return EdgeResult("restoration_unverified")
    if any(p.status not in {"complete", "diverged", "incomplete"} for p in arms):
        return EdgeResult("invalid_probe_status")
    if any(p.status == "incomplete" or (p.status == "complete" and p.completed_steps != p.planned_steps) for p in arms):
        return EdgeResult("incomplete_probes")
    completed = tuple(p for p in arms if p.status == "complete")
    excluded = tuple(sorted(p.probe_id for p in arms if p.status == "diverged"))
    if any(not _finite_number(p.loss) or p.loss < 0 for p in completed):
        return EdgeResult("invalid_completed_loss", excluded_ids=excluded)
    if len(completed) < 2:
        return EdgeResult("insufficient_valid_probes", excluded_ids=excluded)
    best = min(float(p.loss) for p in completed)
    threshold = best * (1.0 + tolerance)
    if not math.isfinite(threshold):
        return EdgeResult("numerical_overflow", excluded_ids=excluded)
    eligible = tuple(p for p in completed if p.loss <= threshold)
    winner = max(eligible, key=lambda p: p.learning_rate)
    return EdgeResult("selected", winner.probe_id, float(winner.learning_rate),
                      float(winner.loss), best, threshold,
                      tuple(sorted(p.probe_id for p in eligible)), excluded)
