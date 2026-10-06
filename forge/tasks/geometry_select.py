"""In-run geometry selector (release-candidate-v10b; FORGE_V2_GEOMETRY_SELECT, default ON).

"Let the run decide." A full-weight FIELD task first trains exactly as before (run A: the champion's rows-per-step bucket,
dev pass, export; the artifact is on disk and complete). Only then, and only when the measured time says a second training
fits, the base is loaded again and trained once more at about 32 rows per optimizer step (run S: the same code path the
research knob FORGE_V2_FIELD_EFF_BATCH drives). Run S replaces run A's artifact only when its INTERNAL DEV loss (the same
400 or so rows carved from TRAIN, verified by a fingerprint) is lower by at least MARGIN. Anything else keeps run A:
no time, a refusal, a crash, an OOM, a timeout, a lost comparison, a failed save.

Why (research/20261006-small-batch-separator/REPORT.md): always-S blew up on Qwen2.5-0.5B-I indic (+3.9 % DEV, board rank
7 -> 14 of 16) while it gained 2.7 to 6.1 % on three other sub-1B cells. No start-of-run observable separates them. On the
12 recorded A / S pairs where a second run fits, "keep the lower internal DEV" picked the externally better arm 12 / 12
(6 distinct cells: the 95 % upper bound on the per-cell error rate is still about 40 %). This module is NOT GPU-validated.

Scope is by observables, not by family or size: the full-weight FIELD route and the time gate. The time gate limits it to
runs that finish early (small or fast tasks).

Two time gates, both predeclared (SLOW_FACTOR 1.3, never risk the deadline, on any doubt keep run A):
  stage 1, before the base is loaded again (time_gate_stage1):
      load_s + 1.3 x core_A + tail + DEADLINE_MARGIN_S <= remaining
  stage 2, inside run S after its own 3-step timing probe and before its first optimizer step (time_gate_stage2):
      1.3 x (t_step_S x updates_S + evals_S x eval_s_A + soup_s_A) + tail + DEADLINE_MARGIN_S <= remaining
  with core_A   = run A's measured seconds from handler entry to the end of its soup (data, probes, training, evals, soup),
       updates_S = run S's optimizer steps per epoch x the epochs run A actually trained,
       evals_S   = the DEV evaluations that plan takes, eval_s_A = run A's measured seconds per DEV evaluation,
       tail      = max(1.3 x run A's measured dev-pass + export seconds, the run's own finish reserve),
       remaining = seconds to the hard stop minus DEADLINE_MARGIN_S (run S is planned against that earlier stop).
Run S is also stopped by its own clock guard (sft_v2._StopAt) against the earlier stop, like any run.

This module imports no ML library at import time; sft_v2 is imported lazily.
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import time
from typing import Any, Callable

ENV = "FORGE_V2_GEOMETRY_SELECT"
DEFAULT = "1"                    # FORGE_V2_GEOMETRY_SELECT=0: no second run, the release behaves as 9504136
TARGET_ROWS = 32                 # rows per optimizer step of run S (phase-bu S arm)
VALUES = (8, 96)                 # inclusive range of knob values the alternate geometry may pick (phase-bu geometry_cpu.py)
MARGIN = 0.005                   # run S must beat run A's internal DEV by at least 0.5 % (predeclared)
SLOW_FACTOR = 1.3                # predeclared factor on every predicted second-run time
DEADLINE_MARGIN_S = 60.0         # run S is planned against a hard stop this much earlier than the real one
LOAD_S_DEFAULT = 60.0            # used when the first base load was not timed
MIN_REMAINING_S = 900.0          # = sft_v2.CHEAP_FULL_MIN_REMAINING_S: a reload + tokenization + a few hundred steps
MIN_DEV_ROWS = 200               # below this an internal DEV difference of 0.5 % is not read
MIN_ROWS_RATIO = 1.5             # run A must take at least 1.5 x TARGET_ROWS rows per step, else there is no alternate
BATCH_ENVS = ("FORGE_V2_FIELD_EFF_BATCH", "FORGE_V2_EFF_BATCH", "FORGE_V2_FIELD_PACKED_EFF_BATCH")
META_KEY = "geometry_select"     # the decision record in forge_run.json meta (events are capped at 200 per run)


class SecondRunAbort(RuntimeError):
    """Run S refused before its first optimizer step (stage-2 time gate, same geometry, fingerprint, no timing)."""


class SecondRunRejected(RuntimeError):
    """Run S trained and lost the internal-DEV comparison. Nothing of run S was saved."""


def make_log(telemetry: Any) -> Callable[..., None]:
    """Record the event in the flight recorder and mirror it to stderr (container logs), like sft_v2's own events."""
    def log(name: str, **kv: Any) -> None:
        try:
            telemetry.event(name, **kv)
            import sys

            compact = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in kv.items()}
            print(f"[sft_v2] {name} {compact}", file=sys.stderr, flush=True)
        except Exception:
            pass
    return log


def enabled() -> bool:
    return os.environ.get(ENV, DEFAULT).strip() != "0"


def static_refusal(*, field_route: bool, strategy: str | None, rung: str = "full") -> str | None:
    """Why the selector does NOT arm for this handler call (None = it arms). Static: needs no data and no GPU.
    Used by the dispatcher (instruct.run) and by the route snapshot (tests/test_route_snapshot_v10b.py)."""
    if not enabled():
        return "switch_off"
    if not field_route or strategy != "full":
        return "not_field_full_weights"          # adapters, KL, multi-GPU, the legacy non-field full route
    if rung != "full":
        return "cheap_full_rung"                 # the first attempt crashed: no second training on top of a retry
    for name in BATCH_ENVS:
        if os.environ.get(name, "").strip() not in ("", "0"):
            return "explicit_batch_env"          # a research batch arm stays a pure arm
    return None


def armed(*, field_route: bool, strategy: str | None, rung: str = "full") -> bool:
    return static_refusal(field_route=field_route, strategy=strategy, rung=rung) is None


def dev_fingerprint(dev_rows: list) -> str:
    """sha256 over the DEV rows' token ids, in order. Equal fingerprints = the two runs were compared on the same rows."""
    h = hashlib.sha256()
    h.update(str(len(dev_rows)).encode())
    for ex in dev_rows:
        ids = ex["input_ids"] if isinstance(ex, dict) else ex
        h.update(b"|")
        h.update(",".join(str(int(t)) for t in ids).encode())
    return h.hexdigest()


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def first_run_refusal(first: dict[str, Any]) -> str | None:
    """Why no second run follows THIS first run (None = go to the time gate). Read from what run A measured."""
    if not first.get("completed"):
        return "first_run_incomplete"
    if not first.get("saved"):
        return "first_artifact_not_saved"
    if not first.get("field") or first.get("strategy") != "full":
        return "not_field_full_weights"          # e.g. a device_map (sharded) model: field is off inside run
    if first.get("cheap_full"):
        return "cheap_full_rung"
    if first.get("gemma2_champion"):
        return "gemma2_champion_geometry"        # that recipe fixes its own blocks per step
    if first.get("large_model_rule"):
        return "large_model_batch_rule"          # >= 3.0 B: the rule would grow run S's batch back
    if first.get("field_hygiene"):
        return "field_hygiene_experiment"        # its subsample depends on the remaining hours
    if first.get("lr_search"):
        return "field_lr_search_experiment"      # sweep / calibration experiments pick an LR per geometry
    if int(first.get("oom_retries") or 0) > 0:
        return "first_run_oom_retry"             # the memory ladder already changed the geometry
    if first.get("window_bound"):
        return "first_run_window_bound"          # the clock stopped run A: there is no time for another
    if int(first.get("dev_n") or 0) < MIN_DEV_ROWS:
        return "internal_dev_too_small"
    if not _finite(first.get("chosen_loss")) or float(first["chosen_loss"]) <= 0.0:
        return "no_internal_dev_loss"
    if not first.get("dev_fingerprint"):
        return "no_dev_fingerprint"
    if float(first.get("rows_per_step") or 0) < MIN_ROWS_RATIO * TARGET_ROWS:
        return "first_run_already_small_batch"
    if not _finite(first.get("core_s")) or float(first["core_s"]) <= 0 or not _finite(first.get("tail_s")):
        return "first_run_time_unmeasured"
    return None


def tail_seconds(first: dict[str, Any], finish_reserve_s: float | None = None) -> float:
    """Dev pass + export of the winner: the larger of 1.3 x run A's measured tail and the run's planned finish reserve."""
    reserve = float(finish_reserve_s if finish_reserve_s is not None else first.get("finish_reserve_s") or 0.0)
    return max(SLOW_FACTOR * float(first.get("tail_s") or 0.0), reserve)


def time_gate_stage1(first: dict[str, Any], *, remaining_s: float, load_s: float | None) -> dict[str, Any]:
    """Before the base is loaded again. remaining_s = seconds to the REAL hard stop."""
    load = float(load_s) if _finite(load_s) and float(load_s) > 0 else LOAD_S_DEFAULT
    predicted = SLOW_FACTOR * float(first["core_s"])
    tail = tail_seconds(first)
    need = load + predicted + tail + DEADLINE_MARGIN_S
    budget = float(remaining_s) - DEADLINE_MARGIN_S
    fits = bool(float(remaining_s) >= MIN_REMAINING_S and need <= budget)
    return {"stage": 1, "fits": fits, "need_s": round(need, 1), "budget_s": round(budget, 1), "load_s": round(load, 1),
            "predicted_second_s": round(predicted, 1), "tail_s": round(tail, 1), "first_core_s": round(float(first["core_s"]), 1),
            "slow_factor": SLOW_FACTOR, "remaining_s": round(float(remaining_s), 1)}


def planned_evals(steps_per_epoch: int, total_steps: int) -> int:
    """DEV evaluations sft_v2.run schedules for a phase (its eval_every rule, before the eval governor widens it)."""
    eval_every = max(10, min(max(1, steps_per_epoch // 8), max(1, total_steps // 8)))
    return int(math.ceil(total_steps / eval_every))


def time_gate_stage2(first: dict[str, Any], *, t_step_s: float | None, steps_per_epoch: int, finish_reserve_s: float,
                     remaining_s: float) -> dict[str, Any]:
    """Inside run S, after its timing probe. remaining_s = seconds to run S's (earlier) hard stop."""
    if not _finite(t_step_s) or float(t_step_s) <= 0:
        return {"stage": 2, "fits": False, "reason": "no_step_time", "remaining_s": round(float(remaining_s), 1)}
    epochs = max(float(first.get("epochs_done") or 0.0), 1.0)      # at least one pass
    updates = int(math.ceil(steps_per_epoch * epochs))
    per_eval = float(first.get("eval_s") or 0.0) / max(1, int(first.get("eval_count") or 0))
    evals = planned_evals(int(steps_per_epoch), updates)
    predicted = SLOW_FACTOR * (float(t_step_s) * updates + evals * per_eval + float(first.get("soup_s") or 0.0))
    tail = tail_seconds(first, finish_reserve_s)
    need = predicted + tail + DEADLINE_MARGIN_S
    fits = bool(need <= float(remaining_s))
    return {"stage": 2, "fits": fits, "need_s": round(need, 1), "budget_s": round(float(remaining_s), 1),
            "predicted_second_s": round(predicted, 1), "tail_s": round(tail, 1), "t_step_s": round(float(t_step_s), 4),
            "updates": updates, "epochs": round(epochs, 3), "evals": evals, "per_eval_s": round(per_eval, 2),
            "slow_factor": SLOW_FACTOR, "remaining_s": round(float(remaining_s), 1)}


def pick(first_dev: Any, second_dev: Any, *, margin: float = MARGIN) -> tuple[str, str]:
    """("S" | "A", reason). S only when both losses are finite and S <= A x (1 - margin)."""
    if not _finite(first_dev) or float(first_dev) <= 0:
        return "A", "first_dev_not_finite"
    if not _finite(second_dev) or float(second_dev) <= 0:
        return "A", "second_dev_not_finite"
    if float(second_dev) <= float(first_dev) * (1.0 - margin):
        return "S", "second_lower_by_margin"
    return "A", "second_not_lower_by_margin"


def alternate_value(*, params_b: float, unit_len: int, vocab: int, per_gpu_gb: float, bnb_ok: bool, layers: int, hidden: int,
                    liger: bool, pack: Any, recipe_bf16: bool) -> dict[str, Any]:
    """The value FORGE_V2_FIELD_EFF_BATCH would carry for run S and the geometry it realises BEFORE the memory probe, computed
    with the calls sft_v2.run makes for that knob: pack.eff_blocks -> choose_geometry -> the recipe's bf16 geometry -> rows
    per step. Selection = phase-bu geometry_cpu.py: closest realised rows / step to TARGET_ROWS, then the fewest accumulation
    steps, then TARGET_ROWS itself, then the smallest value."""
    from forge.tasks import sft_v2

    best: tuple | None = None
    for value in range(VALUES[0], VALUES[1] + 1):
        eff = pack.eff_blocks(value) if pack else value
        geo = sft_v2.choose_geometry(params_b=params_b, max_len=unit_len, vocab=vocab, per_gpu_gb=per_gpu_gb, bnb_ok=bnb_ok,
                                     layers=layers, hidden=hidden, liger=liger, eff_default=eff)
        if geo.fp32_master and recipe_bf16 and os.environ.get("FORGE_V2_FP32_MASTER", "").strip() != "1":
            micro, accum = sft_v2.field_bf16_micro_accum(unit_len=unit_len, vocab=vocab, eff_default=eff)
            geo = sft_v2.Geometry(micro, accum, geo.max_len, geo.gradient_checkpointing, False, geo.optim)
        rows = pack.rows_per_step(geo.eff_batch) if pack else geo.eff_batch
        key = (abs(rows - TARGET_ROWS), geo.grad_accum, value != TARGET_ROWS, value)
        if best is None or key < best[0]:
            best = (key, {"value": value, "rows_per_step": int(rows), "micro": geo.micro_batch, "grad_accum": geo.grad_accum,
                          "units_per_step": geo.eff_batch, "packed": bool(pack)})
    return best[1]


class SecondRun:
    """What sft_v2.run consults while it trains run S. One instance per second run."""

    def __init__(self, first: dict[str, Any], *, log: Callable[..., None]) -> None:
        self.first = first
        self.log = log
        self.defer_persist = True        # nothing of run S reaches the output path until it has won
        self.record: dict[str, Any] = {}

    @property
    def dev_n(self) -> int:
        """Run A's DEV size: sft_v2.dev_size_for depends on the remaining hours, which are fewer now."""
        return int(self.first["dev_n"])

    def eff_value(self, **kw: Any) -> dict[str, Any]:
        alt = alternate_value(**kw)
        self.record["alternate"] = alt
        self.log("sft_v2_geometry_select_alternate", target_rows=TARGET_ROWS, first_rows_per_step=self.first.get("rows_per_step"),
                 **alt)
        return alt

    def before_train(self, *, dev_fp: str, rows_per_step: int, t_step_s: float | None, steps_per_epoch: int,
                     finish_reserve_s: float, remaining_s: float) -> None:
        """Raises SecondRunAbort unless run S may take its first optimizer step."""
        if dev_fp != self.first.get("dev_fingerprint"):
            raise SecondRunAbort("dev_fingerprint_mismatch")
        self.record["second_rows_per_step"] = int(rows_per_step)
        if float(rows_per_step) * MIN_ROWS_RATIO > float(self.first.get("rows_per_step") or 0):
            raise SecondRunAbort("same_geometry")       # the memory probe or the recipe gave back about run A's batch
        gate = time_gate_stage2(self.first, t_step_s=t_step_s, steps_per_epoch=steps_per_epoch,
                                finish_reserve_s=finish_reserve_s, remaining_s=remaining_s)
        self.record["time_gate_stage2"] = gate
        self.log("sft_v2_geometry_select_time_gate", **gate)
        if not gate["fits"]:
            raise SecondRunAbort("time_gate_stage2" if "reason" not in gate else gate["reason"])

    def on_chosen(self, chosen_loss: Any) -> None:
        """Called once, after run S's soup and before its dev pass. Raises SecondRunRejected when run A stays."""
        choice, reason = pick(self.first.get("chosen_loss"), chosen_loss)
        self.record.update(second_dev=float(chosen_loss) if _finite(chosen_loss) else None, pick=choice, pick_reason=reason)
        if choice != "S":
            raise SecondRunRejected(reason)
        self.defer_persist = False


def _rel(first_dev: Any, second_dev: Any) -> float | None:
    if _finite(first_dev) and _finite(second_dev) and float(first_dev) > 0:
        return round(100.0 * (float(second_dev) / float(first_dev) - 1.0), 3)
    return None


def maybe_second_run(spec: Any, deadline: Any, rows: list, **kw: Any) -> dict[str, Any]:
    """Called after run A returned normally (its artifact is exported). Never raises: every failure keeps run A.
    Returns the decision record (also written to forge_run.json meta under META_KEY)."""
    try:
        return _second_run(spec, deadline, rows, **kw)
    except BaseException as exc:  # noqa: BLE001  nothing in the selector may cost the exported run A
        rec = {"ran": None, "pick": "A", "reason": "selector_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        try:
            kw["log"]("sft_v2_geometry_select", **rec)
        except BaseException:  # noqa: BLE001
            pass
        return rec


def _second_run(spec: Any, deadline: Any, rows: list, *, first: dict[str, Any], run_kwargs: dict[str, Any],
                run_fn: Callable[..., None], load_base: Callable[..., Any], free: Callable[[], None],
                artifact_present: Callable[[Any], bool], load_s: float | None = None,
                log: Callable[..., None], telemetry: Any = None) -> dict[str, Any]:
    t0 = time.monotonic()
    rec: dict[str, Any] = {"ran": False, "pick": "A", "margin": MARGIN, "target_rows": TARGET_ROWS,
                           "first_dev": first.get("chosen_loss"), "first_rows_per_step": first.get("rows_per_step"),
                           "first_core_s": first.get("core_s"), "first_tail_s": first.get("tail_s"),
                           "dev_fingerprint": first.get("dev_fingerprint"), "dev_n": first.get("dev_n")}

    def finish(reason: str, **kw: Any) -> dict[str, Any]:
        rec.update(reason=reason, seconds=round(time.monotonic() - t0, 1), **kw)
        try:
            log("sft_v2_geometry_select", **rec)
            if telemetry is not None:
                telemetry.set_meta(**{META_KEY: dict(rec)})
                telemetry.write_into(spec.output_dir)
        except Exception:
            pass
        return rec

    try:
        refusal = first_run_refusal(first)
        if refusal is None and not artifact_present(spec):
            refusal = "first_artifact_missing"
        if refusal is not None:
            return finish(refusal, skipped=True)
        gate = time_gate_stage1(first, remaining_s=deadline.remaining_hard(), load_s=load_s)
        rec["time_gate_stage1"] = gate
        if not gate["fits"]:
            return finish("time_gate_stage1", skipped=True)
    except Exception as exc:  # a broken gate must never cost the exported run A
        return finish("gate_error", skipped=True, error=f"{type(exc).__name__}: {exc}"[:200])

    snap = telemetry.snapshot() if telemetry is not None and hasattr(telemetry, "snapshot") else None
    second: SecondRun | None = None
    result: dict[str, Any] = {}
    outcome, error = "second_run_failed", None
    # Run A's model, pool and optimizer died with its frame; collect the reference cycles and the CUDA cache BEFORE the
    # base is loaded again (the reference-lifetime lesson of RESP-120-C1). Nothing below runs inside an except suite.
    gc.collect()
    free()
    rec["ran"] = True
    try:
        from forge.clock import Deadline

        t_load = time.monotonic()
        loaded = load_base(spec.cached_model_dir, for_generation=False)
        rec["second_load_s"] = round(time.monotonic() - t_load, 1)
        second = SecondRun(first, log=log)
        earlier = Deadline(hard_stop=float(deadline.hard_stop) - DEADLINE_MARGIN_S,
                           export_reserve_s=float(deadline.export_reserve_s))
        log("sft_v2_geometry_select_second_start", remaining_s=round(deadline.remaining_hard(), 1),
            second_load_s=rec["second_load_s"], **{f"gate_{k}": v for k, v in rec["time_gate_stage1"].items()})
        run_fn(spec, earlier, rows, loaded, loaded.tokenizer, **run_kwargs, geometry_select=second, result=result)
        outcome = "second_kept" if result.get("saved") else "second_save_failed"
    except SecondRunAbort as exc:
        outcome = str(exc)
    except SecondRunRejected as exc:
        outcome = str(exc)
    except BaseException as exc:  # noqa: BLE001  crash, OOM, clock, zero steps, interrupt: run A is on disk and stays
        outcome, error = "second_run_failed", f"{type(exc).__name__}: {exc}"[:300]
    loaded = None
    gc.collect()
    try:
        free()
    except Exception:
        pass
    if second is not None:
        rec.update(second.record)
    rec["second_dev"] = rec.get("second_dev", result.get("chosen_loss"))
    rec["internal_rel_pct"] = _rel(rec.get("first_dev"), rec.get("second_dev"))
    rec["second_core_s"] = result.get("core_s")
    rec["second_tail_s"] = result.get("tail_s")
    rec["second_epochs_done"] = result.get("epochs_done")
    rec["first_epochs_done"] = first.get("epochs_done")
    if error is not None:
        rec["error"] = error
    kept_second = outcome == "second_kept"
    rec["pick"] = "S" if kept_second else "A"
    present = False
    try:
        present = bool(artifact_present(spec))
    except Exception:
        pass
    rec["artifact_present"] = present
    if not kept_second and snap is not None:
        try:
            telemetry.restore(snap)     # the shipped artifact is run A's: its meta and curves describe run A again
        except Exception:
            pass
    return finish(outcome)
