#!/usr/bin/env python3
"""Fail-loud launcher for the explicit BloomZ experiment training route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

from forge import telemetry
from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tasks.common import workdir
from forge.tuning import bloomz


class TrainingRunError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise TrainingRunError(message)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-type", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--file-format", required=True)
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True, type=float)
    parser.add_argument("--baseline-stats")
    return parser.parse_args(argv)


def _accepted_inventory(spec: TaskSpec, request: bloomz.BloomzRequest) -> dict[str, Any]:
    path = Path(workdir(spec)) / "bloomz-checkpoint-inventory.json"
    require(path.is_file() and not path.is_symlink(), "accepted inventory is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategy = "full" if request.arm == "full" else "lora"
    require(
        isinstance(payload, dict)
        and payload.get("status") == "EXTERNAL_SCORE_READY"
        and payload.get("strategy") == strategy
        and payload.get("phase") == request.phase,
        "training did not produce an accepted arm inventory",
    )
    checkpoints = payload.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 4, "accepted inventory is not four-way")
    root = (Path(workdir(spec)) / "bloomz-decision-checkpoints").resolve()
    for item in checkpoints:
        require(isinstance(item, dict), "invalid checkpoint inventory entry")
        artifact = Path(str(item.get("path", ""))).resolve(strict=True)
        require(artifact.parent == root, "checkpoint escaped the experiment work directory")
        require(
            bloomz._tree_sha256(artifact) == item.get("tree_sha256"),
            "accepted checkpoint tree changed after training",
        )
    return {"path": str(path.resolve()), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(0 < args.hours_to_complete <= 3.0, "hours-to-complete must be in (0, 3]")
    require(args.task_type == "InstructTextTask", "only InstructTextTask is allowed")
    require(args.file_format == "json", "the frozen fixture requires JSON input")
    require(os.environ.get("USE_KL", "") != "1", "KL is forbidden in this experiment")
    request = bloomz.request_from_environment()
    require(request is not None, "explicit BloomZ arm environment is absent")
    spec = TaskSpec.build(
        task_id=args.task_id,
        task_type=args.task_type,
        model=args.model,
        dataset=args.dataset,
        dataset_type_json=args.dataset_type,
        expected_repo_name=args.expected_repo_name,
        baseline_stats_path=args.baseline_stats,
        file_format=args.file_format,
        use_kl=False,
        kl_coef=0.0,
    )
    bloomz.validate_task_contract(spec)
    output = Path(spec.output_dir)
    require(not output.exists(), "experiment output path must start absent")
    started = time.monotonic()
    deadline = Deadline.from_hours(
        args.hours_to_complete,
        started_monotonic=started,
        export_reserve_s=180,
    )
    telemetry.init(
        task_id=spec.task_id,
        task_type=spec.task_type,
        model_arg=spec.model,
        hours_to_complete=args.hours_to_complete,
        file_format=spec.file_format,
        use_kl=False,
        kl_coef=0.0,
    )
    bloomz.run_matched_training(spec, deadline)
    accepted = _accepted_inventory(spec, request)
    require(not output.exists(), "experiment route wrote an unaccepted visible fallback")
    print(
        json.dumps(
            {
                "status": "EXTERNAL_SCORE_READY",
                "arm": request.arm,
                "phase": request.phase,
                "inventory": accepted["path"],
                "checkpoint_count": len(accepted["payload"]["checkpoints"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
