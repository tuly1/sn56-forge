"""In-run geometry selector (release-candidate-v10b; FORGE_V2_GEOMETRY_SELECT, default ON).

"Let the run decide." A full-weight FIELD task first trains exactly as before (run A: the champion's rows-per-step bucket,
dev pass, export; the artifact is on disk and complete). Only then, and only when the measured time says a second training
fits, the base is loaded again and trained once more at about 32 rows per optimizer step (run S: the same code path the
research knob FORGE_V2_FIELD_EFF_BATCH drives). Run S replaces run A's artifact only when its INTERNAL DEV loss (the same
400 or so rows carved from TRAIN, verified by a fingerprint) is lower by at least MARGIN. A refusal, no time, a crash, an
OOM, a timeout or a lost comparison happen BEFORE run S writes anything, so run A stays.

Which run is exported is read back from disk, never inferred from a return value (Codex RESP-135-1 B1). Run S saves once,
through common.save_adapter: staged directory, validated, fsynced, ready marker, then ONE directory exchange / rename.
That exchange is the commit point. Run S's truth file carries a per-run generation marker (SecondRun.truth_reason).
After run S ends, however it ends, read_export() reads the truth file at the output path: the marker of this run = the
export is run S's (even when a later fsync failed or an interrupt arrived); run A's recorded truth and weight hash = the
export is run A's; anything else = UNKNOWN and is reported as such. The decision record and the telemetry follow that.

Exit paths (Codex RESP-136-1 B1). maybe_second_run has ONE place that emits a pick: _finalize. From the moment the second
run may start (state["second_started"]), every exit, whatever raised and wherever (run S, the cleanup after it, logging,
telemetry, the reconciliation itself, KeyboardInterrupt / SystemExit included), goes through _finalize, which determines the
pick by a disk read-back and retries when it is interrupted. If no read-back completes, the pick is UNKNOWN, never A.
Before that moment nothing of a second run exists, and pick A is a fact, not a guess.
Each export directory also carries its own record: run S writes no telemetry into the output path before its save, the
copy staged with run S's export says S, and the copy in run A's directory says A.

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
GENERATION_PREFIX = "sft_v2_final+geometry_select_S:"   # + a per-run nonce: the truth_reason of an export written by run S
TRUTH_FILE = "forge_artifact_truth.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin")
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
    """sha256 over the DEV rows in order: token ids AND labels (the label mask decides which tokens the DEV loss counts).
    Equal fingerprints = the two runs were scored on the same rows with the same loss positions."""
    h = hashlib.sha256()
    h.update(str(len(dev_rows)).encode())
    for ex in dev_rows:
        ids = ex["input_ids"] if isinstance(ex, dict) else ex
        labels = ex.get("labels") if isinstance(ex, dict) else None
        h.update(b"|")
        h.update(",".join(str(int(t)) for t in ids).encode())
        h.update(b"#")
        h.update(b"none" if labels is None else ",".join(str(int(t)) for t in labels).encode())
    return h.hexdigest()


def read_export(output_dir: str, *, hash_weights: bool = True) -> dict[str, Any]:
    """What is at the output path NOW: the truth file and (optionally) sha256 of the weight files. Any failure inside gives
    readable False (the caller reads again and, failing that, reports UNKNOWN)."""
    out: dict[str, Any] = {"readable": False}
    try:
        import json

        with open(os.path.join(output_dir, TRUTH_FILE), encoding="utf-8") as fh:
            truth = json.load(fh)
        out.update(readable=True, truth=truth.get("truth"), optimizer_step=truth.get("optimizer_step"), reason=truth.get("reason"))
        names = sorted(n for n in os.listdir(output_dir) if n.endswith(WEIGHT_SUFFIXES) and os.path.isfile(os.path.join(output_dir, n)))
        out["weight_files"] = names
        out["weight_bytes"] = sum(os.path.getsize(os.path.join(output_dir, n)) for n in names)
        if hash_weights:
            per_file: dict[str, str] = {}
            for n in names:
                h = hashlib.sha256()
                with open(os.path.join(output_dir, n), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                per_file[n] = h.hexdigest()
            out["weight_sha256"] = per_file          # per file: comparable with an independently captured manifest
            out["weights_sha256"] = hashlib.sha256("".join(f"{n}:{per_file[n]}\n" for n in names).encode()).hexdigest()
    except BaseException as exc:  # noqa: BLE001
        out.update(readable=False, error=f"{type(exc).__name__}: {exc}"[:200])
    return out


def generation_of(export: dict[str, Any], *, second_reason: str | None, first_export: dict[str, Any] | None) -> str:
    """"S": the export carries this second run's marker. "A": it is byte-for-byte what was there before the second run
    (truth, step, reason and weight hash). "UNKNOWN": unreadable or neither."""
    if not export.get("readable"):
        return "UNKNOWN"
    if second_reason and export.get("reason") == second_reason:
        return "S"
    if str(export.get("reason") or "").startswith(GENERATION_PREFIX):
        return "UNKNOWN"            # a second-run marker that is not this run's
    keys = ("truth", "optimizer_step", "reason", "weight_files", "weight_bytes", "weights_sha256")
    if first_export and first_export.get("readable") and all(export.get(k) == first_export.get(k) for k in keys):
        return "A"
    return "UNKNOWN"


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

    def __init__(self, first: dict[str, Any], *, log: Callable[..., None], telemetry: Any = None) -> None:
        self.first = first
        self.log = log
        self.telemetry = telemetry
        self.defer_persist = True        # nothing of run S reaches the output path until it has won
        self.record: dict[str, Any] = {}
        self.nonce = os.urandom(8).hex()

    @property
    def truth_reason(self) -> str:
        """The truth_reason run S's one save writes into the export: the generation marker read back after the run."""
        return GENERATION_PREFIX + self.nonce

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
        if self.telemetry is not None:
            # the telemetry copy that save_adapter stages INSIDE run S's export must name run S (a kill after the commit
            # point leaves that copy as the only record); _finalize replaces it with the full record. A non-final record
            # carries no "pick": only a record with state "final" claims one
            self.telemetry.set_meta(**{META_KEY: {"state": "second_run_saving", "provisional_generation": "S",
                                                  "second_marker": self.truth_reason, "first_dev": self.first.get("chosen_loss"),
                                                  "second_dev": self.record.get("second_dev"), "margin": MARGIN}})


def _rel(first_dev: Any, second_dev: Any) -> float | None:
    if _finite(first_dev) and _finite(second_dev) and float(first_dev) > 0:
        return round(100.0 * (float(second_dev) / float(first_dev) - 1.0), 3)
    return None


def maybe_second_run(spec: Any, deadline: Any, rows: list, **kw: Any) -> dict[str, Any]:
    """Called after run A returned normally (its artifact is exported). Returns the decision record (also written to
    forge_run.json meta under META_KEY). It handles every exception itself, BaseException included; the only way out by an
    exception is an interrupt that lands on one of the few statements no handler can cover (the return below, the loop head
    and the fallback of _finalize). Then NO pick is emitted, the CLI keeps the trained artifact that is on disk, and a
    record without state "final" claims nothing.

    The pick is emitted in ONE place, _finalize. Once the second run may have started, the pick is what a disk read-back
    finds at the output path (S, A, or UNKNOWN when no read-back completes). It is never inferred from an exception, a
    return value or the line an interrupt landed on."""
    state: dict[str, Any] = {"t0": time.monotonic(), "second_started": False, "rec": {"ran": False}, "kw": kw, "spec": spec,
                             "outcome": None, "error": None, "second": None, "result": {}, "first_export": None, "snap": None,
                             "generation": None, "skip_reason": None}
    try:
        _run_second(state, spec, deadline, rows, **kw)
    except BaseException as exc:  # noqa: BLE001  anything, from anywhere: the read-back in _finalize decides, not this handler
        state["error"] = state["error"] or f"{type(exc).__name__}: {exc}"[:300]
        state["outcome"] = state["outcome"] or ("second_run_failed" if state["second_started"] else "selector_error")
    return _finalize(state)


def _finalize(state: dict[str, Any]) -> dict[str, Any]:
    """The only place a pick is emitted. Retries the reconciliation when it is interrupted; a pick that could not be read
    back from disk after the second run may have started is UNKNOWN."""
    attempts: list[str] = []
    for _ in range(4):
        try:
            return _reconcile(state, attempts)
        except BaseException as exc:  # noqa: BLE001  e.g. KeyboardInterrupt between two statements of the reconciliation
            attempts.append(f"{type(exc).__name__}: {exc}"[:120])
    rec = dict(state.get("rec") or {})
    if state["second_started"]:
        gen = state.get("generation")                   # a read-back that completed in an interrupted attempt, else None
        rec.update(pick=gen if gen in ("A", "S") else "UNKNOWN", exported_generation=gen or "UNKNOWN",
                   reason="reconcile_interrupted_after_read_back" if gen in ("A", "S") else "exported_generation_unknown")
    else:
        rec.update(pick="A", exported_generation="A", reason=state.get("skip_reason") or "selector_error")
    rec.update(state="final", second_started=bool(state["second_started"]), finalize_interruptions=attempts)
    return rec


def _reconcile(state: dict[str, Any], attempts: list[str]) -> dict[str, Any]:
    """Idempotent: may be called again after an interruption. Order: (1) read the export back and fix the pick,
    (2) align the telemetry with it, (3) clean up, (4) log and write the record. Steps 2 to 4 cannot change the pick."""
    kw, spec = state["kw"], state["spec"]
    log, telemetry = kw.get("log"), kw.get("telemetry")
    rec = state["rec"]
    second, result = state["second"], state["result"]
    if not state["second_started"]:
        # nothing of a second run exists: the export is what run A wrote. Not a guess, and no read-back is needed.
        rec.update(pick="A", exported_generation="A", reason=state.get("skip_reason") or state.get("outcome") or "selector_error",
                   skipped=True)
    else:
        final_export: dict[str, Any] = {"readable": False}
        for _ in range(3):                                # a read that fails once (I/O hiccup, interrupt) is read again
            final_export = read_export(spec.output_dir)
            if final_export.get("readable"):
                break
        marker = second.truth_reason if second is not None else None
        generation = generation_of(final_export, second_reason=marker, first_export=state["first_export"])
        state["generation"] = generation                 # from here on an interrupted attempt still knows the read-back
        outcome = state.get("outcome") or "second_run_failed"
        if generation == "S":
            reason = "second_kept" if outcome == "second_kept" else "second_committed_despite_error"
        elif generation == "A":
            reason = "second_save_not_on_disk" if outcome == "second_kept" else outcome
        else:
            reason = "exported_generation_unknown"
        if second is not None:
            rec.update(second.record)
        rec.update(final_export=final_export, exported_generation=generation, artifact_present=bool(final_export.get("readable")), second_outcome=outcome, second_marker=marker,
                   pick=generation, reason=reason, second_dev=rec.get("second_dev", result.get("chosen_loss")),
                   second_core_s=result.get("core_s"), second_tail_s=result.get("tail_s"),
                   second_epochs_done=result.get("epochs_done"))
        rec["internal_rel_pct"] = _rel(rec.get("first_dev"), rec.get("second_dev"))
        # (2) telemetry follows the disk: run A exported -> run A's meta and curves again; S or UNKNOWN -> left as it is
        if generation == "A" and state.get("snap") is not None and telemetry is not None:
            try:
                telemetry.restore(state["snap"])
            except BaseException as exc:  # noqa: BLE001
                rec["telemetry_restore_error"] = f"{type(exc).__name__}: {exc}"[:120]
        # (3) cleanup: run S's model. Failures and interrupts here cannot change the pick (already read back above).
        try:
            state["loaded"] = None
            gc.collect()
            kw["free"]()
        except BaseException as exc:  # noqa: BLE001
            rec["cleanup_error"] = f"{type(exc).__name__}: {exc}"[:120]
    if state.get("error") is not None:
        rec["error"] = state["error"]
    rec.update(state="final", second_started=bool(state["second_started"]), finalize_interruptions=list(attempts),
               seconds=round(time.monotonic() - state["t0"], 1))
    # (4) log + record into the flight recorder. Each step on its own: a failure of one is noted, never raised.
    for step in ("log", "meta", "write"):
        try:
            if step == "log" and log is not None:
                log("sft_v2_geometry_select", **rec)
            elif step == "meta" and telemetry is not None:
                telemetry.set_meta(**{META_KEY: dict(rec)})
            elif step == "write" and telemetry is not None:
                telemetry.write_into(spec.output_dir)
        except BaseException as exc:  # noqa: BLE001
            rec.setdefault("record_errors", []).append(f"{step}: {type(exc).__name__}: {exc}"[:120])
    return rec


def _run_second(state: dict[str, Any], spec: Any, deadline: Any, rows: list, *, first: dict[str, Any], run_kwargs: dict[str, Any],
                run_fn: Callable[..., None], load_base: Callable[..., Any], free: Callable[[], None],
                artifact_present: Callable[[Any], bool], load_s: float | None = None,
                log: Callable[..., None], telemetry: Any = None) -> None:
    """Gates, then the second run. Emits NO pick and writes no final record: it only fills ``state``. May raise anything."""
    rec = state["rec"]
    rec.update(ran=False, margin=MARGIN, target_rows=TARGET_ROWS, first_dev=first.get("chosen_loss"),
               first_rows_per_step=first.get("rows_per_step"), first_core_s=first.get("core_s"),
               first_tail_s=first.get("tail_s"), dev_fingerprint=first.get("dev_fingerprint"), dev_n=first.get("dev_n"),
               first_epochs_done=first.get("epochs_done"))

    def skip(reason: str, **kw: Any) -> None:
        state["skip_reason"] = reason
        rec.update(kw)

    try:
        refusal = first_run_refusal(first)
        if refusal is None and not artifact_present(spec):
            refusal = "first_artifact_missing"
        if refusal is not None:
            return skip(refusal)
        gate = time_gate_stage1(first, remaining_s=deadline.remaining_hard(), load_s=load_s)
        rec["time_gate_stage1"] = gate
        if not gate["fits"]:
            return skip("time_gate_stage1")
    except Exception as exc:  # a broken gate must never cost the exported run A
        return skip("gate_error", error=f"{type(exc).__name__}: {exc}"[:200])

    # what is exported now (run A): truth file + weight hashes, read from disk; the reference for the read-back after run S
    first_export = read_export(spec.output_dir)
    rec["first_export"] = first_export
    if not first_export.get("readable") or str(first_export.get("reason") or "").startswith(GENERATION_PREFIX):
        return skip("first_export_unreadable")
    state["first_export"] = first_export
    state["snap"] = telemetry.snapshot() if telemetry is not None and hasattr(telemetry, "snapshot") else None
    if telemetry is not None:
        # run A's directory keeps a record that names run A while the second run trains (run S writes no telemetry into
        # the output path before its save): a kill before run S's commit point leaves this record with run A's weights
        telemetry.set_meta(**{META_KEY: {"state": "second_run_started", "provisional_generation": "A",
                                         "first_export": first_export, "first_dev": first.get("chosen_loss"), "margin": MARGIN}})
        telemetry.write_into(spec.output_dir)
    # Run A's model, pool and optimizer died with its frame; collect the reference cycles and the CUDA cache BEFORE the
    # base is loaded again (the reference-lifetime lesson of RESP-120-C1). Nothing below runs inside an except suite.
    gc.collect()
    free()
    # ---- from here on a second run may exist: every exit goes through the disk read-back in _finalize ----
    state["second_started"] = True
    rec["ran"] = True
    state["outcome"] = "second_run_failed"
    try:
        from forge.clock import Deadline

        t_load = time.monotonic()
        state["loaded"] = load_base(spec.cached_model_dir, for_generation=False)
        rec["second_load_s"] = round(time.monotonic() - t_load, 1)
        state["second"] = second = SecondRun(first, log=log, telemetry=telemetry)
        earlier = Deadline(hard_stop=float(deadline.hard_stop) - DEADLINE_MARGIN_S,
                           export_reserve_s=float(deadline.export_reserve_s))
        log("sft_v2_geometry_select_second_start", remaining_s=round(deadline.remaining_hard(), 1),
            second_load_s=rec["second_load_s"], **{f"gate_{k}": v for k, v in rec["time_gate_stage1"].items()})
        loaded, state["loaded"] = state["loaded"], None          # the run takes the model; this frame keeps no reference
        run_fn(spec, earlier, rows, loaded, loaded.tokenizer, **run_kwargs, geometry_select=second, result=state["result"])
        state["outcome"] = "second_kept" if state["result"].get("saved") else "second_save_failed"
    except SecondRunAbort as exc:
        state["outcome"] = str(exc)
    except SecondRunRejected as exc:
        state["outcome"] = str(exc)
    except BaseException as exc:  # noqa: BLE001  crash, OOM, clock, zero steps, interrupt: _finalize reads the disk
        state["outcome"], state["error"] = "second_run_failed", f"{type(exc).__name__}: {exc}"[:300]
