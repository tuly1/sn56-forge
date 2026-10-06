"""Entry point. Parses the validator-supplied arguments and dispatches to the
handler for the task type. Kept deliberately thin: all real work lives in the
task modules so this file stays a stable, readable contract surface.

Guiding rule (owner decision 2026-10-05): the output path holds a trained model
or nothing. The validator uploads whatever is at the output path after a zero
exit or a wall-clock kill, and uploads nothing after a non-zero exit. A handler
failure therefore funnels into a real minimal fine-tune on the task's TRAIN
rows (forge.tasks.fallback); if that cannot train either, the output path is
cleared and the process exits non-zero (a failed task). No untrained artifact
is written at any point.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from forge import telemetry
from forge.clock import Deadline
from forge.data.schema import TaskSpec

# The task types the validator sends for text tournaments. We validate softly:
# an unknown value reaches dispatch rather than crashing argparse. A type with no
# handler and no supervised TRAIN schema fails honestly (non-zero exit).
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
    # task type is reported by our own telemetry instead of an argparse exit(2).
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
    telemetry.init(
        task_id=args.task_id,
        task_type=args.task_type,
        model_arg=args.model,
        hours_to_complete=args.hours_to_complete,
        file_format=args.file_format,
        use_kl=use_kl,
        kl_coef=kl_coef,
    )
    # Building the spec parses --dataset-type, which can raise on a payload whose
    # required column is absent (e.g. a valid completion-style instruct task with
    # field_output=null) or on malformed JSON. It is inside the guard so the
    # failure is recorded; a bare spec has no column schema, so no fine-tune can
    # run on it and the task then fails with a non-zero exit.
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
        _log(f"spec build failed ({type(exc).__name__}: {exc}); using bare spec")
        telemetry.event("spec_build_failed", error=f"{type(exc).__name__}: {exc}")
        spec = TaskSpec(
            task_id=args.task_id,
            task_type=args.task_type,
            model=args.model,
            dataset=args.dataset,
            expected_repo_name=args.expected_repo_name,
            baseline_stats_path=args.baseline_stats,
            file_format=args.file_format,
        )

    return _run(spec, deadline)


def _run(spec: TaskSpec, deadline: Deadline) -> int:
    """Dispatch to a handler. Returns the process exit code.

    0 only when a trained artifact is at the output path: the handler's own, or
    the minimal fine-tune's after a handler failure / a missing handler. 1 with
    an empty output path when neither could train.

    We import the handler lazily so heavy ML deps don't load for a task type
    this build doesn't implement, and we catch everything, so a handler that
    raises on the validator's GPU still gets the minimal fine-tune.
    """
    handler = None
    try:
        from forge.tasks import dispatch

        handler = dispatch.for_task(spec.task_type)
    except Exception as exc:
        _log(f"dispatch failed for {spec.task_type!r}: {exc!r}")
        telemetry.event("dispatch_failed", error=repr(exc))

    handler_ok = False
    if handler is not None:
        try:
            handler(spec, deadline)
            handler_ok = True
        except BaseException as exc:  # noqa: BLE001 — includes SystemExit/KeyboardInterrupt
            _log(f"handler raised ({type(exc).__name__}: {exc})")
            telemetry.event("handler_failed", error=f"{type(exc).__name__}: {exc}")
    else:
        telemetry.event("handler_missing", task_type=spec.task_type)

    from forge.tasks.fallback import clear_output, output_is_trained, run_mini_finetune

    trained = _safe_output_is_trained(output_is_trained, spec)
    if handler_ok and trained:
        # This is a separately reviewed, family/task-gated post-handler
        # operation. The handler's model/trainer locals are gone, so the
        # full export loads a fresh base and cannot destructively merge the
        # live training model. Export failures are diagnostic and must leave
        # the selected adapter for the uploader. LFM export keeps its existing
        # opt-out, while Gemma 4 E2B export is on by default and accepts
        # FORGE_GEMMA4_FULL_EXPORT=0 as its opt-out.
        try:
            from forge.tasks.lfm25_full_export import maybe_export

            maybe_export(spec, deadline)
        except BaseException as exc:  # noqa: BLE001
            _log(f"post-handler full export failed ({type(exc).__name__}: {exc})")
            telemetry.event(
                "lfm_moe_full_export_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            from forge.tasks.gemma4_full_export import maybe_export as maybe_export_gemma4

            maybe_export_gemma4(spec, deadline)
        except BaseException as exc:  # noqa: BLE001
            _log(f"Gemma 4 post-handler export failed ({type(exc).__name__}: {exc})")
            telemetry.event(
                "gemma4_full_export_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        trained = _safe_output_is_trained(output_is_trained, spec)

    if trained:
        # the handler's trained artifact (complete, or a trained checkpoint persisted before a late failure)
        telemetry.event("run_complete" if handler_ok else "handler_failed_kept_trained_artifact")
        _record_export_manifest(spec.output_dir)
        telemetry.write_into(spec.output_dir)
        return 0

    # Nothing trained is at the output path. Whatever is there (a partial write) is removed before anything else.
    telemetry.event("no_trained_artifact", handler_ok=handler_ok, handler_missing=handler is None)
    clear_output(spec.output_dir)
    try:
        receipt = run_mini_finetune(spec, deadline)
        _log(f"minimal fine-tune exported: {receipt}")
        telemetry.event("run_complete_mini_finetune", steps=int(receipt.get("steps", 0)))
        telemetry.write_into(spec.output_dir)
        return 0
    except BaseException as exc:  # noqa: BLE001
        clear_output(spec.output_dir)
        _log(f"minimal fine-tune failed ({type(exc).__name__}: {exc}); no artifact, failing the task")
        telemetry.event("task_failed_no_artifact", error=f"{type(exc).__name__}: {exc}")
        return 1


def _record_export_manifest(output_dir: str) -> None:
    """Independent custody record of what is exported (Codex RESP-136-1 C2): file names, sizes and sha256 of every file at
    the output path except the flight recorder itself, written into forge_run.json meta as ``export_manifest``. Taken here,
    after the handler has returned, by code that shares nothing with the geometry selector's own read-back. Only when a
    second training may have started in this task (the selector's record exists and says so): no other task pays for it.
    Never raises."""
    try:
        import hashlib

        rec = (telemetry._data.get("meta") or {}).get("geometry_select")
        if not isinstance(rec, dict) or not (rec.get("second_started") or rec.get("state") in ("second_run_started", "second_run_saving")):
            return
        files: dict[str, dict] = {}
        for name in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, name)
            if name == "forge_run.json" or name.endswith(".tmp") or not os.path.isfile(path):
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            files[name] = {"bytes": os.path.getsize(path), "sha256": h.hexdigest()}
        telemetry.set_meta(export_manifest={"captured_by": "forge.cli after the handler", "files": files})
    except BaseException as exc:  # noqa: BLE001
        try:
            telemetry.event("export_manifest_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        except BaseException:  # noqa: BLE001
            pass


def _safe_output_is_trained(check, spec: TaskSpec) -> bool:
    try:
        return bool(check(spec.output_dir))
    except Exception as exc:
        telemetry.event("output_check_failed", error=repr(exc))
        return False


def _log(msg: str) -> None:
    print(f"[forge.cli] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
