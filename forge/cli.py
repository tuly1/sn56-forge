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
import platform
import sys
import time

# --- CUDA caching-allocator configuration ------------------------------------
# Runs before any `forge` module that can reach torch is imported: the container
# entrypoint is `python -m forge.cli`, forge/__init__.py is empty, and
# telemetry/clock/schema import no ML library.  torch reads
# PYTORCH_CUDA_ALLOC_CONF once, when its CUDA allocator initialises, so this is
# the last moment the value can still take effect.  The 2026-09-03 H100 80 GB
# survival smoke (lease 20260903T105152Z) showed the reserved pool growing far
# past the admission probe through fragmentation of variable-length batches
# (granite 0.629 -> 0.948 of host memory sustained; early transients to
# 0.92-0.97 on bloomz/gemma/lfm; Qwen3.5 steps 5x slower while the allocator
# freed and retried) -- torch's own OOM message recommends expandable segments
# for exactly this.  An operator's value is never overridden, and non-Linux or
# GPU-less processes are left untouched.
_CUDA_ALLOC_CONF_KEY = "PYTORCH_CUDA_ALLOC_CONF"
_CUDA_ALLOC_CONF_DEFAULT = "expandable_segments:True"
_CUDA_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia0")


def _cuda_device_visible() -> bool:
    """A CUDA device node is mounted; decided without importing torch."""
    return any(os.path.exists(node) for node in _CUDA_DEVICE_NODES)


def _configure_cuda_allocator(
    environ, *, system: str, cuda_visible: bool, torch_preloaded: bool
) -> dict:
    """Set the allocator default when appropriate and report the decision."""
    state = {
        "key": _CUDA_ALLOC_CONF_KEY,
        "value": environ.get(_CUDA_ALLOC_CONF_KEY),
        "system": system,
        "cuda_device_visible": bool(cuda_visible),
        "torch_preloaded": bool(torch_preloaded),
    }
    if _CUDA_ALLOC_CONF_KEY in environ:
        state.update(source="environment", reason="preset_by_operator")
    elif system != "Linux":
        state.update(source="unset", reason="not_linux")
    elif not cuda_visible:
        state.update(source="unset", reason="no_cuda_device")
    else:
        environ[_CUDA_ALLOC_CONF_KEY] = _CUDA_ALLOC_CONF_DEFAULT
        state.update(
            value=_CUDA_ALLOC_CONF_DEFAULT,
            source="forge_default",
            reason="linux_cuda_device",
        )
    return state


_CUDA_ALLOC_CONF_STATE = _configure_cuda_allocator(
    os.environ,
    system=platform.system(),
    cuda_visible=_cuda_device_visible(),
    torch_preloaded="torch" in sys.modules,
)


def cuda_alloc_conf_state() -> dict:
    """What this process decided about PYTORCH_CUDA_ALLOC_CONF at import time."""
    return dict(_CUDA_ALLOC_CONF_STATE)


from forge import telemetry  # noqa: E402  (the allocator env must precede these)
from forge.clock import Deadline  # noqa: E402
from forge.data.schema import TaskSpec  # noqa: E402

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
    telemetry.init(
        task_id=args.task_id,
        task_type=args.task_type,
        model_arg=args.model,
        hours_to_complete=args.hours_to_complete,
        file_format=args.file_format,
        use_kl=use_kl,
        kl_coef=kl_coef,
    )
    _record_cuda_alloc_conf()
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
        telemetry.event("dispatch_failed", error=repr(exc))

    if handler is not None:
        try:
            handler(spec, deadline)
            _record_cuda_allocator_readback()
            telemetry.event("run_complete")
            telemetry.write_into(spec.output_dir)
            return
        except BaseException as exc:  # noqa: BLE001 — includes SystemExit/KeyboardInterrupt
            _log(f"handler raised ({type(exc).__name__}: {exc}); using fallback")
            telemetry.event("handler_failed", error=f"{type(exc).__name__}: {exc}")

    try:
        from forge.tasks.fallback import emit_untrained_copy

        emit_untrained_copy(spec)
    except Exception as exc:  # the floor itself failing is all we can log
        _log(f"fallback failed: {exc!r}")
        telemetry.event("fallback_failed", error=repr(exc))
    _record_cuda_allocator_readback()
    telemetry.write_into(spec.output_dir)


def _record_cuda_alloc_conf() -> None:
    """Flight-record the import-time allocator decision in the artifact."""
    state = cuda_alloc_conf_state()
    telemetry.set_meta(
        cuda_alloc_conf=state["value"],
        cuda_alloc_conf_source=state["source"],
        cuda_alloc_conf_reason=state["reason"],
    )
    telemetry.event("cuda_alloc_conf", **state)


def _record_cuda_allocator_readback() -> None:
    """Best-effort proof, read back from torch, of the allocator settings in force.

    Consulted once the handler has run (CUDA is initialised by then).  Never
    imports torch itself and never raises: a diagnostic must not cost a run.
    """
    try:
        torch = sys.modules.get("torch")
        fields = {"env_value": os.environ.get(_CUDA_ALLOC_CONF_KEY)}
        if torch is None:
            telemetry.event(
                "cuda_alloc_conf_readback", status="torch_not_imported", **fields
            )
            return
        if not torch.cuda.is_available():
            telemetry.event(
                "cuda_alloc_conf_readback", status="cuda_unavailable", **fields
            )
            return
        snapshot = torch.cuda.memory._snapshot()
        settings = (
            snapshot.get("allocator_settings") if isinstance(snapshot, dict) else None
        )
        if not isinstance(settings, dict):
            telemetry.event(
                "cuda_alloc_conf_readback",
                status="allocator_settings_absent",
                **fields,
            )
            return
        expandable = settings.get("expandable_segments")
        telemetry.event(
            "cuda_alloc_conf_readback",
            status="ok",
            expandable_segments=expandable,
            allocator_conf=settings.get(_CUDA_ALLOC_CONF_KEY),
            **fields,
        )
        telemetry.set_meta(cuda_alloc_conf_expandable_segments=expandable)
    except Exception as exc:  # noqa: BLE001
        try:
            telemetry.event(
                "cuda_alloc_conf_readback",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass


def _log(msg: str) -> None:
    print(f"[forge.cli] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
