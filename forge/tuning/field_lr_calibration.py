"""Opt-in FIELD SFT LR experiment, pinned to the reviewed Trainer 5.12.1 API.

The published curvature/edge rules are in lr_calibration. This bounded runner,
complete-state recovery, fixed 25-step panel, prior fallback and fork guards are
our prospective experimental design, not a recovered historical trainer.
No quality or CUDA performance claim follows from its CPU tests.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
import random
import statistics
import time
from typing import Any, Callable

import numpy as np
import torch

from forge.tuning.lr_calibration import ProbeSummary, select_stability_edge, warmup_curvature

WARMUP_STEPS = 8
PROBE_STEPS = 25
RAMP_STEPS = 5
MAX_PROBES = 4
MAX_SECONDS = 360.0
BUDGET_FRACTION = 0.15
RESTORE_RESERVE_S = 30.0
_MISSING = object()


class StateRestoreError(RuntimeError):
    """Must propagate: continuing with unknown model state is forbidden."""


class ProbeTimeout(RuntimeError):
    pass


class Diverged(RuntimeError):
    pass


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _digest(entries):
    h = hashlib.sha256()
    for name, tensor in entries:
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        h.update(_bytes(tensor))
    return h.hexdigest()


class InitialState:
    """Own one CPU copy, including frozen parameters and nonpersistent buffers.

    Restore the original buffer objects even when forward replaces a registered
    buffer. Parameter topology changes fail closed. No optimizer used before
    this call is touched: every trial gets a newly created Trainer optimizer.
    """
    def __init__(self, trainer):
        self.trainer, self.model = trainer, trainer.model
        self.modules = list(self.model.modules())
        self.parameter_maps = [(m, dict(m._parameters)) for m in self.modules]
        self.buffer_maps = [(m, dict(m._buffers), set(m._non_persistent_buffers_set)) for m in self.modules]
        self.parameters = [(p, p.detach().cpu().clone(), p.requires_grad,
                            None if p.grad is None else p.grad.detach().cpu().clone()) for p in self.model.parameters()]
        seen = set()
        self.buffers = []
        for _, mapping, _ in self.buffer_maps:
            for b in mapping.values():
                if b is not None and id(b) not in seen:
                    seen.add(id(b)); self.buffers.append((b, b.detach().cpu().clone()))
        self.modes = [(m, m.training) for m in self.modules]
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        self.optimizer = trainer.optimizer
        self.learning_rate = trainer.args.learning_rate
        self.accum = getattr(trainer, "current_gradient_accumulation_steps", _MISSING)
        self.entries = [(f"p:{i}", saved) for i, (_, saved, _, _) in enumerate(self.parameters)]
        self.entries += [(f"b:{i}", saved) for i, (_, saved) in enumerate(self.buffers)]
        self.sha256 = _digest(self.entries)
        self.restores = 0

    def restore(self):
        try:
            if list(self.model.modules()) != self.modules:
                raise ValueError("Module topology changed during probe")
            for module, original in self.parameter_maps:
                if module._parameters.keys() != original.keys() or any(module._parameters[n] is not p for n, p in original.items()):
                    raise ValueError("Parameter topology changed during probe")
            with torch.no_grad():
                for param, saved, requires_grad, grad in self.parameters:
                    if param.shape != saved.shape or param.dtype != saved.dtype:
                        raise ValueError("Parameter shape/dtype changed during probe")
                    param.copy_(saved.to(param.device))
                    param.requires_grad_(requires_grad)
                    param.grad = None if grad is None else grad.to(param.device).clone()
                for module, mapping, nonpersistent in self.buffer_maps:
                    module._buffers.clear(); module._buffers.update(mapping)
                    module._non_persistent_buffers_set = set(nonpersistent)
                for buf, saved in self.buffers:
                    if buf.shape != saved.shape or buf.dtype != saved.dtype:
                        raise ValueError("Original buffer shape/dtype changed during probe")
                    buf.copy_(saved.to(buf.device))
            # Verify copies rather than assuming copy_ or recovery succeeded.
            current = [(f"p:{i}", p.detach().cpu()) for i, (p, _, _, _) in enumerate(self.parameters)]
            current += [(f"b:{i}", b.detach().cpu()) for i, (b, _) in enumerate(self.buffers)]
            if _digest(current) != self.sha256:
                raise ValueError("Restored model digest mismatch")
            for module, mode in self.modes:
                module.training = mode
            self.trainer.optimizer = self.optimizer
            self.trainer.args.learning_rate = self.learning_rate
            if self.accum is _MISSING:
                self.trainer.__dict__.pop("current_gradient_accumulation_steps", None)
            else:
                self.trainer.current_gradient_accumulation_steps = self.accum
            random.setstate(self.rng[0]); np.random.set_state(self.rng[1]); torch.set_rng_state(self.rng[2])
            if self.rng[3] is not None:
                torch.cuda.set_rng_state_all(self.rng[3])
            self.restores += 1
        except Exception as exc:
            raise StateRestoreError(f"FIELD LR probe restoration failed: {type(exc).__name__}: {exc}") from exc


def _conflict(trainer, prior_lr):
    import transformers
    if transformers.__version__ != "5.12.1":
        return "unreviewed_transformers_version"
    if os.environ.get("FORGE_V2_FIELD_LR_CALIBRATION", "0") != "1":
        return "flag_off"
    if os.environ.get("FORGE_V2_FIELD_SWEEP", "0") == "1":
        return "conflicting_field_sweep"
    if os.environ.get("FORGE_V2_ROW_LOSS", "0") == "1":
        return "unsupported_row_objective"
    try:
        if float(os.environ.get("FORGE_V2_LR", "0") or 0) != 0 or float(os.environ.get("FORGE_V2_FIELD_LR_MULT", "1") or 1) != 1:
            return "conflicting_fixed_lr"
    except ValueError:
        return "invalid_lr_override"
    if not math.isfinite(prior_lr) or not 2e-5 <= prior_lr <= 1e-4:
        return "prior_outside_field_envelope"
    if trainer.accelerator.num_processes != 1 or trainer.args.n_gpu > 1:
        return "unsupported_distributed_geometry"
    if trainer.accelerator.gradient_accumulation_steps != 1 or trainer.args.fp16:
        return "unsupported_accelerator_scaling"
    # FIELD_BF16's paged optimizer has global bitsandbytes registration state.
    # Do not claim to restore that unreviewed state by restoring tensors alone.
    if str(trainer.args.optim) not in {"OptimizerNames.ADAMW_TORCH", "adamw_torch", "OptimizerNames.ADAMW_TORCH_FUSED", "adamw_torch_fused"}:
        return "unsupported_optimizer"
    if getattr(trainer, "neftune_hook_handle", None) is not None:
        return "already_active_neftune"
    if getattr(trainer, "compute_loss_func", None) is not None or getattr(trainer, "label_smoother", None) is not None:
        return "unsupported_custom_objective"
    return None


def _optimizer_step(trainer, batches, *, learning_rate, before_micro):
    """Use the production Trainer's own objective, accumulation and backward.

    Accelerator.autocast supplies the forward policy normally installed by
    prepare_model during Trainer.train. Single-process BF16 requires no scaler.
    This is deliberately not lr_probe's unweighted mean of microbatch means.
    """
    trainer.current_gradient_accumulation_steps = len(batches)
    num_items = trainer._get_num_items_in_batch(batches, trainer.args.device)
    if num_items is not None and int(num_items) <= 0:
        raise ValueError("No supervised shifted target tokens in probe update")
    for group in trainer.optimizer.param_groups:
        group["lr"] = learning_rate
    trainer.optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for batch in batches:
        before_micro()
        # training_step owns tensor preparation and exact pinned loss scaling.
        with trainer.accelerator.autocast():
            loss = trainer.training_step(trainer.model, dict(batch), num_items_in_batch=num_items)
        x = float(loss.item())
        if not math.isfinite(x) or x < 0:
            raise Diverged("nonfinite/negative SFT loss")
        value += x
    before_micro()
    norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.args.max_grad_norm)
    if not math.isfinite(float(norm)):
        raise Diverged("nonfinite gradient norm")
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    return value


def _trial(trainer, batches, *, lr, steps, ramp, check, clock):
    from transformers.integrations.neftune import activate_neftune
    trainer.optimizer = None
    trainer.args.learning_rate = lr
    trainer.create_optimizer()  # same optimizer class/groups/settings as final training
    alpha = trainer.neftune_noise_alpha
    embedding = trainer.model.get_input_embeddings() if alpha is not None else None
    old_alpha = getattr(embedding, "neftune_noise_alpha", _MISSING) if embedding is not None else _MISSING
    handle = None
    losses, times = [], []
    try:
        if alpha is not None:
            handle = activate_neftune(trainer.model, alpha, trainer.accelerator)
        accum = trainer.args.gradient_accumulation_steps
        for i in range(steps):
            check()
            start = clock()
            chunk = batches[i * accum:(i + 1) * accum]
            value = _optimizer_step(trainer, chunk, learning_rate=lr * min(1.0, (i + 1) / (ramp + 1)), before_micro=check)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(clock() - start)
            losses.append(value)
        return losses, times
    except Exception as exc:
        exc.probe_completed_steps = len(losses)
        raise
    finally:
        if handle is not None:
            handle.remove()
            if old_alpha is _MISSING:
                delattr(embedding, "neftune_noise_alpha")
            else:
                embedding.neftune_noise_alpha = old_alpha
        # Release trial Adam states before recovery allocates GPU copy buffers.
        trainer.optimizer = None


def run_field_calibration(trainer, batches, *, prior_lr, deadline, finish_reserve_s,
                          log: Callable[[str, dict], None] | None = None,
                          clock: Callable[[], float] = time.monotonic):
    """Return LR/timing/telemetry, restoring the initial state before returning.

    A complete panel is required. Timeout/error falls back to the prior after
    verified recovery. StateRestoreError propagates. The wall bound is checked
    between microsteps; it cannot interrupt a running CUDA kernel or CPU copy.
    """
    started = clock()
    diag: dict[str, Any] = {"protocol": "field-lr-v1", "status": "prior", "prior_lr": prior_lr,
                            "selected_lr": prior_lr, "warmup_steps": WARMUP_STEPS, "probe_steps": PROBE_STEPS,
                            "ramp_steps": RAMP_STEPS, "max_probes": MAX_PROBES,
                            "scope": "experimental_field_sft_single_gpu", "restoration_verified": False}
    def finish(lr, timing):
        diag.update(selected_lr=lr, elapsed_s=clock() - started)
        if log:
            log("sft_v2_field_calibration", diag)
        return lr, timing, diag

    reason = _conflict(trainer, prior_lr)
    if reason:
        diag["reason"] = reason
        return finish(prior_lr, None)
    accum = trainer.args.gradient_accumulation_steps
    if len(batches) < PROBE_STEPS * accum:
        diag["reason"] = "insufficient_distinct_microbatches"
        return finish(prior_lr, None)
    # Only TRAIN batches already cached by the handler are used. Every arm sees
    # the same 25-update prefix; no wrapping, dev rows, or held-out scores.
    batches = batches[:PROBE_STEPS * accum]
    remaining = deadline.remaining_hard()
    reserve = max(float(finish_reserve_s), float(deadline.export_reserve_s))
    budget = min(MAX_SECONDS, BUDGET_FRACTION * remaining, max(0.0, remaining - reserve))
    diag.update(budget_s=budget, finish_reserve_s=reserve, micro=trainer.args.per_device_train_batch_size,
                grad_accum=accum, optim=str(trainer.args.optim), neftune_alpha=trainer.neftune_noise_alpha)
    if budget <= 2 * RESTORE_RESERVE_S:
        diag["reason"] = "insufficient_time_before_snapshot"
        return finish(prior_lr, None)
    expires = started + budget
    restore_allowance = RESTORE_RESERVE_S
    def check():
        if clock() + restore_allowance >= expires or deadline.remaining_hard() <= reserve + restore_allowance:
            raise ProbeTimeout("probe deadline/reserve")

    # Snapshot construction itself is read-only; fail explicitly if it fails.
    initial = InitialState(trainer)
    snapshot_seconds = clock() - started
    restore_allowance = max(RESTORE_RESERVE_S, 2 * snapshot_seconds)
    diag["initial_state_sha256"] = initial.sha256
    bh = []
    for i, batch in enumerate(batches):
        for key, value in sorted(batch.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError("Probe cache must contain only tensors")
            bh.append((f"{i}:{key}", value.detach().cpu()))
    batch_sha = _digest(bh)
    diag["train_batch_sha256"] = batch_sha
    comparison = hashlib.sha256(json.dumps({"state": initial.sha256, "batches": batch_sha,
        "micro": diag["micro"], "accum": accum, "optim": diag["optim"], "neftune": diag["neftune_alpha"],
        "args": {k: getattr(trainer.args, k) for k in ("adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm", "bf16")},
        "protocol": diag["protocol"]}, sort_keys=True).encode()).hexdigest()
    diag["comparison_id"] = comparison
    timing, chosen, summaries = None, prior_lr, []
    try:
        check()
        initial.restore()
        losses, times = _trial(trainer, batches, lr=prior_lr, steps=WARMUP_STEPS, ramp=0, check=check, clock=clock)
        restore_start = clock(); initial.restore(); restore_seconds = clock() - restore_start
        restore_allowance = max(restore_allowance, 2 * restore_seconds)
        timing = statistics.median(times)
        curvature = warmup_curvature(losses)
        diag.update(warmup_losses=losses, warmup_completed_steps=len(losses), curvature=asdict(curvature),
                    t_per_step=timing, restore_seconds=restore_seconds)
        if curvature.status != "ready":
            diag["reason"] = curvature.status
        else:
            center = prior_lr * curvature.factor
            # Prospective FIELD envelope; clipping/dedup is not a historical claim.
            rates = list(dict.fromkeys(max(2e-5, min(1e-4, center * 10**offset)) for offset in (0, -.3, .15, .3)))
            diag["planned_rates"] = rates
            forecast = len(rates) * (PROBE_STEPS * max(times) * 1.25 + 2 * max(restore_seconds, snapshot_seconds))
            diag["panel_forecast_s"] = forecast
            if len(rates) < 2:
                diag["reason"] = "insufficient_distinct_rates"
            elif clock() + forecast + restore_allowance >= expires or deadline.remaining_hard() <= reserve + forecast + restore_allowance:
                diag["reason"] = "complete_panel_does_not_fit"
            else:
                for i, rate in enumerate(rates):
                    check(); initial.restore()
                    status, score, completed = "complete", None, 0
                    try:
                        values, _ = _trial(trainer, batches, lr=rate, steps=PROBE_STEPS, ramp=RAMP_STEPS, check=check, clock=clock)
                        completed = len(values)
                        steady = values[RAMP_STEPS:]
                        score = statistics.median(steady[max(1, int(len(steady) * .8)):])
                    except Diverged as exc:
                        status = "diverged"
                        completed = exc.probe_completed_steps
                    finally:
                        initial.restore()
                    summaries.append(ProbeSummary(f"probe-{i}", rate, score, PROBE_STEPS, completed, comparison, True, status))
                edge = select_stability_edge(summaries)
                diag["edge"] = asdict(edge)
                if edge.status == "selected":
                    chosen = edge.learning_rate
                    diag["status"] = "selected"
                else:
                    diag["reason"] = edge.status
    except StateRestoreError:
        raise
    except Exception as exc:
        diag.update(status="prior", reason=type(exc).__name__, error=str(exc)[:240])
        chosen = prior_lr
    finally:
        initial.restore()
        diag.update(restoration_verified=True, restoration_count=initial.restores,
                    probes=[asdict(p) for p in summaries])
    return finish(chosen, timing)
