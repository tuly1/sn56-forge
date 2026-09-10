"""Validator-style dev evaluation, best-checkpoint tracking, greedy soup.

The validator scores each held-out row as the token-mean cross entropy over
its completion tokens and then averages rows (batch size 1). We reproduce that
statistic exactly on the dev slice (batched, per-row means) so checkpoint
selection optimises the number the tournament ranks on, not a token-weighted
proxy.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Callable


def eval_batch_for(*, max_len: int, vocab: int, budget_bytes: float = 6e9) -> int:
    """Largest power-of-two batch whose fp32 logits (+bf16 copy) fit the budget."""
    per_row = max(1, max_len) * max(1, vocab) * 6.0
    b = int(budget_bytes // per_row)
    if b < 2:
        return 1
    return min(32, 1 << (b.bit_length() - 1))


def per_example_dev_loss(
    model: Any, dev: list[dict[str, list[int]]], collator: Any, *, batch_size: int = 1,
    autocast_bf16: bool = True, max_rows: int | None = None,
) -> tuple[float, list[float]]:
    """Mean over rows of per-row completion-token-mean CE (validator metric).

    batch_size 1: the model's own labelled loss is the per-example token mean
    (fused kernels never materialise the logits). batch_size > 1: per-example
    means from the logits, for models whose vocabulary keeps that cheap.
    """
    import torch
    import torch.nn.functional as F

    rows = dev if max_rows is None else dev[:max_rows]
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), max(1, batch_size)):
                chunk = rows[start : start + max(1, batch_size)]
                batch = collator(chunk)
                batch = {k: v.to(device) for k, v in batch.items()}
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (autocast_bf16 and device.type == "cuda") else _null()
                if len(chunk) == 1:
                    with ctx:
                        out = model(**batch)
                    loss = out.loss
                    losses.append(float(loss.item()) if loss is not None else float("nan"))
                    del out, loss
                    continue
                labels = batch.pop("labels")
                with ctx:
                    out = model(**batch)
                logits = out.logits[:, :-1, :].float()
                tgt = labels[:, 1:]
                mask = tgt != -100
                tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape)
                counts = mask.sum(dim=1)
                summed = (tok * mask).sum(dim=1)
                per = torch.where(counts > 0, summed / counts.clamp(min=1), torch.full_like(summed, float("nan")))
                losses.extend(per.tolist())
                del out, logits, tok
    finally:
        if was_training:
            model.train()
    finite = [x for x in losses if math.isfinite(x)]
    mean = sum(finite) / len(finite) if finite else float("inf")
    return mean, losses


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def snapshot_trainable(model: Any) -> dict[str, Any]:
    """CPU bf16 copy of the trainable parameters (what the soup averages)."""
    import torch

    with torch.no_grad():
        return {
            n: p.detach().to(dtype=torch.bfloat16).to("cpu", copy=True)
            for n, p in model.named_parameters()
            if p.requires_grad
        }


def load_state(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(device=p.device, dtype=p.dtype))


def available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


class SnapshotPool:
    """Up to `k` lowest-dev-loss snapshots, RAM-gated, sorted best first."""

    def __init__(self, k: int = 8) -> None:
        self.k = k
        self.items: list[dict[str, Any]] = []
        self._snap_bytes: int | None = None

    def _bytes(self, model: Any) -> int:
        if self._snap_bytes is None:
            self._snap_bytes = sum(p.numel() * 2 for p in model.parameters() if p.requires_grad)
        return self._snap_bytes

    def can_admit(self, model: Any) -> bool:
        snap = self._bytes(model)
        avail = available_ram()
        if avail is None:
            return len(self.items) < 2
        margin = max(2 * 1024**3, 3 * snap)
        return (avail - margin) >= snap

    def consider(self, model: Any, loss: float, step: int) -> bool:
        if loss is None or not math.isfinite(loss):
            return False
        if len(self.items) < self.k and self.can_admit(model):
            self.items.append({"loss": loss, "step": step, "state": snapshot_trainable(model)})
        elif self.items:
            worst = max(self.items, key=lambda e: e["loss"])
            if loss < worst["loss"]:
                self.items.remove(worst)
                worst["state"] = None
                del worst
                gc.collect()
                self.items.append({"loss": loss, "step": step, "state": snapshot_trainable(model)})
            else:
                return False
        else:
            return False
        self.items.sort(key=lambda e: e["loss"])
        return True

    def best(self) -> dict[str, Any] | None:
        return self.items[0] if self.items else None


def greedy_soup(
    model: Any, pool: SnapshotPool, eval_fn: Callable[[Any], float], *, time_budget_s: float,
    log: Callable[[str, dict], None],
) -> tuple[float, dict[str, Any] | None, int]:
    """Wortsman-style greedy soup over the pool; monotone (never worse than the
    best single). Leaves the winning weights in the model and returns
    (dev_loss, cpu_state_bf16, n_members)."""
    import torch

    cands = sorted(pool.items, key=lambda e: e["loss"])
    if not cands:
        return float("inf"), None, 0
    if len(cands) == 1:
        load_state(model, cands[0]["state"])
        return cands[0]["loss"], cands[0]["state"], 1
    t0 = time.perf_counter()
    load_state(model, cands[0]["state"])
    seed_loss = eval_fn(model)
    per_eval = max(1.0, time.perf_counter() - t0)
    if not math.isfinite(seed_loss):
        return cands[0]["loss"], cands[0]["state"], 1
    soup_sum = {n: t.float().clone() for n, t in cands[0]["state"].items()}
    best, n = seed_loss, 1
    accepted = [cands[0]["step"]]
    for i in range(1, len(cands)):
        if (time.perf_counter() - t0) + per_eval > time_budget_s:
            log("soup_time_exhausted", {"tried": i, "members": n})
            break
        st = cands[i]["state"]
        inv = 1.0 / (n + 1)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in soup_sum and name in st:
                    p.data.copy_(((soup_sum[name] + st[name].float()) * inv).to(p.dtype))
        cand_loss = eval_fn(model)
        if math.isfinite(cand_loss) and cand_loss < best - 1e-6:
            for name in soup_sum:
                if name in st:
                    soup_sum[name] += st[name].float()
            n += 1
            best = cand_loss
            accepted.append(cands[i]["step"])
        else:
            with torch.no_grad():
                inv = 1.0 / n
                for name, p in model.named_parameters():
                    if name in soup_sum:
                        p.data.copy_((soup_sum[name] * inv).to(p.dtype))
    with torch.no_grad():
        inv = 1.0 / n
        for name, p in model.named_parameters():
            if name in soup_sum:
                p.data.copy_((soup_sum[name] * inv).to(p.dtype))
    final_state = {name: (soup_sum[name] * (1.0 / n)).to(torch.bfloat16) for name in soup_sum}
    log("soup_result", {"seed_loss": seed_loss, "soup_loss": best, "members": n, "pool": len(cands), "steps": accepted})
    return best, final_state, n
