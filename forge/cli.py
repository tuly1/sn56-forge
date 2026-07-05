"""Entry point. Parses the validator-supplied arguments and dispatches to the
handler for the task type. Kept deliberately thin: all real work lives in the
task modules so this file stays a stable, readable contract surface.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from forge.clock import Deadline
from forge.data.schema import TaskSpec

# The validator invokes exactly these; unknown extras are tolerated so a spec
# bump doesn't hard-fail a build mid-tournament.
_TASK_TYPES = ("InstructTextTask", "ChatTask", "DpoTask", "GrpoTask", "EnvTask")

# Reserve a slice of the wall clock for final export so a kill never catches us
# mid-write. Sized in clock.Deadline; named here for visibility.
_EXPORT_RESERVE_SECONDS = 180


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-type", default=None)
    p.add_argument("--task-type", required=True, choices=_TASK_TYPES)
    p.add_argument("--file-format", default="s3")
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    # Present on some tasks; safe to ignore if absent.
    p.add_argument("--baseline-stats", default=os.environ.get("BASELINE_STATS_PATH"))
    known, _unknown = p.parse_known_args(argv)
    return known


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = _parse(sys.argv[1:] if argv is None else argv)

    deadline = Deadline.from_hours(
        args.hours_to_complete,
        started_monotonic=started,
        export_reserve_s=_EXPORT_RESERVE_SECONDS,
    )
    spec = TaskSpec.build(
        task_id=args.task_id,
        task_type=args.task_type,
        model=args.model,
        dataset=args.dataset,
        dataset_type_json=args.dataset_type,
        expected_repo_name=args.expected_repo_name,
        baseline_stats_path=args.baseline_stats,
    )

    # Import the handler lazily: heavy ML deps shouldn't load for a task type
    # this build doesn't implement.
    from forge.tasks import dispatch

    handler = dispatch.for_task(spec.task_type)
    if handler is None:
        # No implementation for this type. Emit a valid-but-minimal model so the
        # entry is scoreable rather than a hard failure. Better a floor than a
        # forfeit.
        from forge.tasks.fallback import emit_untrained_copy

        emit_untrained_copy(spec)
        return 0

    handler(spec, deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
