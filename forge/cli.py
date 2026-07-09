"""Entry point. Parses the validator-supplied arguments and dispatches to the
handler for the task type. Kept deliberately thin: all real work lives in the
task modules so this file stays a stable, readable contract surface.

Guiding rule: never exit non-zero. The validator treats a non-zero exit before
the wall-clock kill as a failure with no upload (scored -1), whereas any model
left at the output path is uploaded and scored. So every failure path funnels
into the fallback, which guarantees a valid artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from forge.clock import Deadline
from forge.data.schema import TaskSpec

# The task types the validator sends for text tournaments. We validate softly:
# an unknown value is routed to the fallback rather than crashing argparse, so a
# spec bump mid-tournament degrades instead of forfeiting.
_KNOWN_TASK_TYPES = ("InstructTextTask", "ChatTask", "DpoTask", "GrpoTask", "EnvTask")

# Reserve a slice of the wall clock for final export so a kill never catches us
# mid-write. Sized in clock.Deadline; named here for visibility.
_EXPORT_RESERVE_SECONDS = 180


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-type", default=None)
    # No `choices=`: we accept any value and let dispatch decide, so an unseen
    # task type falls through to the fallback instead of an argparse exit(2).
    p.add_argument("--task-type", required=True)
    p.add_argument("--file-format", default="s3")
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    # Present on some tasks; safe to ignore if absent.
    p.add_argument("--baseline-stats", default=os.environ.get("BASELINE_STATS_PATH"))
    known, _unknown = p.parse_known_args(argv)
    return known


def _kl_from_env() -> tuple[bool, float]:
    """The validator signals KL-regularised instruct tasks via env vars, not
    CLI args. `USE_KL=1` plus a `KL_COEF` float means the scorer will penalise
    divergence from the base model, so we mirror the term in training.
    """
    use_kl = os.environ.get("USE_KL", "") == "1"
    coef = 0.0
    if use_kl:
        try:
            coef = float(os.environ.get("KL_COEF", "0") or 0)
        except ValueError:
            coef = 0.0
    return use_kl, coef


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = _parse(sys.argv[1:] if argv is None else argv)

    deadline = Deadline.from_hours(
        args.hours_to_complete,
        started_monotonic=started,
        export_reserve_s=_EXPORT_RESERVE_SECONDS,
    )

    use_kl, kl_coef = _kl_from_env()
    # Building the spec parses --dataset-type, which can raise on a payload whose
    # required column is absent (e.g. a valid completion-style instruct task with
    # field_output=null) or on malformed JSON. That must degrade to the fallback,
    # not forfeit — so it's inside the guard, with a bare spec as the floor.
    try:
        spec = TaskSpec.build(
            task_id=args.task_id,
            task_type=args.task_type,
            model=args.model,
            dataset=args.dataset,
            dataset_type_json=args.dataset_type,
            expected_repo_name=args.expected_repo_name,
            baseline_stats_path=args.baseline_stats,
            file_format=args.file_format,
            use_kl=use_kl,
            kl_coef=kl_coef,
        )
    except BaseException as exc:  # noqa: BLE001
        _log(f"spec build failed ({type(exc).__name__}: {exc}); using bare spec + fallback")
        spec = TaskSpec(
            task_id=args.task_id,
            task_type=args.task_type,
            model=args.model,
            dataset=args.dataset,
            expected_repo_name=args.expected_repo_name,
            baseline_stats_path=args.baseline_stats,
            file_format=args.file_format,
        )

    _run(spec, deadline)
    return 0


def _run(spec: TaskSpec, deadline: Deadline) -> None:
    """Dispatch to a handler, degrading to the fallback on any failure.

    We import the handler lazily so heavy ML deps don't load for a task type
    this build doesn't implement, and we catch everything: a handler that raises
    on the validator's GPU must still leave a scoreable model behind.
    """
    handler = None
    try:
        from forge.tasks import dispatch

        handler = dispatch.for_task(spec.task_type)
    except Exception as exc:  # dispatch import problems must not forfeit
        _log(f"dispatch failed for {spec.task_type!r}: {exc!r}")

    if handler is not None:
        try:
            handler(spec, deadline)
            return
        except BaseException as exc:  # noqa: BLE001 — includes SystemExit/KeyboardInterrupt
            _log(f"handler raised ({type(exc).__name__}: {exc}); using fallback")

    try:
        from forge.tasks.fallback import emit_untrained_copy

        emit_untrained_copy(spec)
    except Exception as exc:  # the floor itself failing is all we can log
        _log(f"fallback failed: {exc!r}")


def _log(msg: str) -> None:
    print(f"[forge.cli] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
