"""Learning-rate estimate + short in-process sweep.

1. Analytic prior: Adam's per-step relative update is ~lr / weight_rms, so
   lr = eta * median_weight_rms * sqrt(1e9 / N) (scaled down for gradient noise
   when a noise-scale estimate is available).
2. A few optimizer steps at the prior measure wall time per step and the loss
   trajectory's curvature; a fast drop means "far from a solution, push up".
3. Probe 3-4 candidate LRs on identical cached batches for a fixed number of
   steps; pick the highest LR whose tail-median loss is within tolerance of the
   best (edge-of-stability). Weights are restored afterwards; the sweep only
   produces a number.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

ETA_TARGET = 0.003
CLAMP = (1e-6, 1e-3)
EDGE_TOLERANCE = 0.025
WARMUP_STEPS = 8
PROBE_WARMUP = 4
HALF_RANGE = 0.30  # decades


def analytic_lr(
    *, weight_rms: float | None, params_b: float, eff_batch: int, gradient_noise_scale: float | None = None
) -> float:
    rms = weight_rms if weight_rms and weight_rms > 0 else 0.02
    lr = ETA_TARGET * rms
    if params_b > 0:
        lr *= math.sqrt(1.0 / params_b)
    if gradient_noise_scale and gradient_noise_scale > 0 and eff_batch > 0:
        lr *= math.sqrt(eff_batch / (eff_batch + gradient_noise_scale))
    return max(CLAMP[0], min(CLAMP[1], lr))


def curvature_factor(step_losses: list[float]) -> tuple[float, dict[str, Any]]:
    pts = [float(x) for x in step_losses if x is not None and math.isfinite(float(x))]
    if len(pts) < 3:
        return 1.0, {"reason": "too_few_points", "n": len(pts)}
    k = max(1, len(pts) // 3)
    start = sum(pts[:k]) / k
    end = sum(pts[-k:]) / k
    rel_drop = (start - end) / max(abs(start), 1e-6)
    factor = 1.0 + 4.0 * (rel_drop - 0.04)
    factor = max(0.6, min(1.6, factor))
    return factor, {"l_start": start, "l_end": end, "rel_drop": rel_drop, "n": len(pts)}


def _save_state(model: Any) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def _restore_state(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(p.device, non_blocking=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def run_trial(
    model: Any,
    batches: list[dict[str, Any]],
    *,
    lr: float,
    opt_steps: int,
    grad_accum: int,
    optimizer_factory: Callable[[list, float], Any],
    max_grad_norm: float = 1.0,
    autocast_bf16: bool = True,
    warmup_steps: int = 0,
    data_offset: int = 0,
    out_step_losses: list[float] | None = None,
    timing: dict[str, float] | None = None,
) -> float:
    """Train `opt_steps` optimizer steps at `lr`; return tail-median step loss.

    Prunes a diverging trial (rolling loss > 1.75x its best) early by returning inf.
    When `timing` is given, accumulates steady-state wall time (excluding the
    first optimizer step, which carries kernel compilation / cache warm-up)
    into timing["secs"] / timing["steps"].
    """
    import torch

    model.train()
    t_prev = None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_factory(params, lr)
    device = next(model.parameters()).device
    n_cached = len(batches)
    step_losses: list[float] = []
    micro_acc, micro_n, opt_idx = 0.0, 0, 0
    warm = min(warmup_steps, max(0, opt_steps // 3))
    best_seen = float("inf")
    for i in range(opt_steps * grad_accum):
        batch = _to_device(batches[(data_offset + i) % n_cached], device)
        if autocast_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / grad_accum
        else:
            loss = model(**batch).loss / grad_accum
        loss.backward()
        micro_acc += float(loss.item()) * grad_accum
        micro_n += 1
        if (i + 1) % grad_accum == 0:
            opt_idx += 1
            for g in optimizer.param_groups:
                g["lr"] = lr * min(1.0, opt_idx / (warm + 1))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if timing is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.perf_counter()
                if t_prev is not None and opt_idx >= 2:
                    timing["secs"] = timing.get("secs", 0.0) + (now - t_prev)
                    timing["steps"] = timing.get("steps", 0) + 1
                t_prev = now
            step_loss = micro_acc / max(1, micro_n)
            micro_acc, micro_n = 0.0, 0
            if opt_idx <= warm:
                continue
            step_losses.append(step_loss)
            if out_step_losses is not None:
                out_step_losses.append(step_loss)
            window = step_losses[-5:]
            rolling = sum(window) / len(window)
            if math.isfinite(rolling):
                best_seen = min(best_seen, rolling)
            if len(step_losses) >= 3 and (not math.isfinite(rolling) or rolling > best_seen * 1.75):
                del optimizer
                return float("inf")
    del optimizer
    if not step_losses:
        return float("inf")
    tail = sorted(step_losses[max(1, int(len(step_losses) * 0.8)) :])
    n = len(tail)
    return tail[n // 2] if n % 2 else 0.5 * (tail[n // 2 - 1] + tail[n // 2])


def select_edge(scores: dict[float, float], tolerance: float = EDGE_TOLERANCE) -> tuple[float, float]:
    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    if not finite:
        raise ValueError("no finite probe scores")
    best = min(finite.values())
    stable = {k: v for k, v in finite.items() if v <= best * (1.0 + tolerance)}
    edge = max(stable)
    return 10**edge, stable[edge]


def lr_search(
    model: Any,
    batches: list[dict[str, Any]],
    *,
    center_lr: float,
    budget_s: float,
    grad_accum: int,
    optimizer_factory: Callable[[list, float], Any],
    autocast_bf16: bool,
    log: Callable[[str, dict], None],
    max_grad_norm: float = 1.0,
) -> tuple[float, float | None, dict[str, Any]]:
    """Return (lr, seconds_per_optimizer_step, diagnostics). Never raises on OOM
    of the probes themselves; the caller decides what to do with the estimate."""
    import torch

    diag: dict[str, Any] = {"center_lr_prior": center_lr, "budget_s": round(budget_s, 1)}
    if not batches:
        return center_lr, None, {**diag, "mode": "no_batches"}
    initial = _save_state(model)
    try:
        warm_losses: list[float] = []
        timing: dict[str, float] = {"secs": 0.0, "steps": 0}
        t0 = time.perf_counter()
        # Two cheap steps first: if a step is slow, a long warmup would eat the
        # budget the sweep needs, so scale the warmup length to the budget.
        run_trial(
            model, batches, lr=center_lr, opt_steps=2, grad_accum=grad_accum,
            optimizer_factory=optimizer_factory, max_grad_norm=max_grad_norm,
            autocast_bf16=autocast_bf16, warmup_steps=0, out_step_losses=warm_losses, timing=timing,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        probe_step = (time.perf_counter() - t0) / 2.0
        extra = max(1, min(WARMUP_STEPS - 2, int(0.25 * budget_s / max(probe_step, 1e-3))))
        _restore_state(model, initial)
        run_trial(
            model, batches, lr=center_lr, opt_steps=extra + 1, grad_accum=grad_accum,
            optimizer_factory=optimizer_factory, max_grad_norm=max_grad_norm,
            autocast_bf16=autocast_bf16, warmup_steps=0, out_step_losses=warm_losses, timing=timing, data_offset=2 * grad_accum,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_per_step = (time.perf_counter() - t0) / (3 + extra)
        t_per_step = timing["secs"] / timing["steps"] if timing.get("steps", 0) >= 3 else wall_per_step
        diag.update(t_per_step_wall=round(wall_per_step, 4))
        _restore_state(model, initial)
        factor, cdiag = curvature_factor(warm_losses)
        center = center_lr * factor
        diag.update(t_per_step_warmup=round(t_per_step, 4), curvature_factor=round(factor, 3), curvature=cdiag,
                    center_lr_adjusted=center)
        remaining = budget_s - (time.perf_counter() - t0)
        # Sweep geometry: 3 probes (validate) if tight, else 4.
        n_probes = 4 if remaining >= 4 * 25 * t_per_step else 3
        steps = int(remaining / (n_probes * t_per_step))
        steps = max(15, min(100, steps))
        if n_probes * 15 * t_per_step > remaining:
            diag.update(mode="skip_no_budget", steps=steps, remaining_s=round(remaining, 1))
            log("lr_probe", diag)
            return center, t_per_step, diag
        offsets = [0.0, -HALF_RANGE, +HALF_RANGE] if n_probes == 3 else [0.0, -HALF_RANGE, +HALF_RANGE / 2, +HALF_RANGE]
        center_log = math.log10(center)
        scores: dict[float, float] = {}
        timed_steps, timed_secs = 0, 0.0
        for off in offsets:
            if (time.perf_counter() - t0) + steps * t_per_step > budget_s and scores:
                break
            lg = center_log + off
            ts = time.perf_counter()
            trial_timing: dict[str, float] = {"secs": 0.0, "steps": 0}
            loss = run_trial(
                model, batches, lr=10**lg, opt_steps=steps, grad_accum=grad_accum,
                optimizer_factory=optimizer_factory, max_grad_norm=max_grad_norm,
                autocast_bf16=autocast_bf16, warmup_steps=PROBE_WARMUP, timing=trial_timing,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            _restore_state(model, initial)
            scores[lg] = loss
            if trial_timing.get("steps", 0) >= 3:
                timed_steps += int(trial_timing["steps"])
                timed_secs += float(trial_timing["secs"])
            log("lr_probe_trial", {"lr": 10**lg, "loss": loss, "steps": steps, "secs": round(dt, 1)})
            if off == +HALF_RANGE / 2 and not math.isfinite(loss):
                break  # climbing diverged; no point going higher
        if timed_steps >= 10:
            t_per_step = timed_secs / timed_steps
        try:
            lr, best = select_edge(scores)
            diag.update(mode="sweep", steps=steps, scores={f"{10**k:.3e}": v for k, v in scores.items()},
                        selected_lr=lr, selected_loss=best, t_per_step=round(t_per_step, 4))
        except ValueError:
            lr = center
            diag.update(mode="sweep_all_diverged", selected_lr=lr)
        log("lr_probe", diag)
        return lr, t_per_step, diag
    finally:
        try:
            _restore_state(model, initial)
        except Exception:
            pass
        del initial
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
