"""Dev-set construction for checkpoint selection.

The scored test set is the validator's, so the dev slice we hold out only has
to be *representative* and *not too large*. We stratify by total length and by
completion length so short and long, easy and hard rows are all present, and
we exact-deduplicate first so a row cannot sit on both sides of the split.
"""

from __future__ import annotations

import random
from typing import Any


def exact_dedup(examples: list[dict[str, list[int]]]) -> list[dict[str, list[int]]]:
    """Drop rows whose (input_ids, labels) pair has already been seen."""
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    out: list[dict[str, list[int]]] = []
    for ex in examples:
        key = (tuple(ex["input_ids"]), tuple(ex["labels"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def dev_size_for(n: int, *, hours: float, cap: int = 1000) -> int:
    """Champion-style rule (~3% of data, bounded) with a tighter cap on short
    tasks so evaluation never eats the training window."""
    if n < 60:
        return 0
    base = min(cap, max(min(400, n // 5), n // 33))
    if hours <= 1.0:
        base = min(base, 400)
    elif hours <= 2.0:
        base = min(base, 600)
    return max(0, min(base, n - 32))


def _quantile_bins(values: list[float], n_bins: int) -> list[int]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    bins = [0] * n
    for rank, idx in enumerate(order):
        bins[idx] = min(rank * n_bins // n, n_bins - 1)
    return bins


def stratified_split(
    examples: list[dict[str, Any]], dev_size: int, *, seed: int = 7, n_bins: int = 4
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train, dev). Dev is sampled proportionally from a total-length x
    completion-length grid so it mirrors the dataset's shape."""
    n = len(examples)
    if dev_size <= 0 or n < 2:
        return list(examples), []
    dev_size = min(dev_size, n - 1)
    total_len = [float(len(ex["input_ids"])) for ex in examples]
    comp_len = [float(sum(1 for t in ex["labels"] if t != -100)) for ex in examples]
    lb = _quantile_bins(total_len, n_bins)
    cb = _quantile_bins(comp_len, n_bins)
    strata: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        strata.setdefault((lb[i], cb[i]), []).append(i)
    rng = random.Random(seed)
    dev_idx: set[int] = set()
    for key, members in strata.items():
        rng.shuffle(members)
        take = max(1, round(dev_size * len(members) / n))
        for i in members[:take]:
            if len(dev_idx) >= dev_size:
                break
            dev_idx.add(i)
    if len(dev_idx) < dev_size:
        rest = [i for i in range(n) if i not in dev_idx]
        rng.shuffle(rest)
        for i in rest[: dev_size - len(dev_idx)]:
            dev_idx.add(i)
    train = [ex for i, ex in enumerate(examples) if i not in dev_idx]
    dev = [examples[i] for i in sorted(dev_idx)]
    return train, dev


def smart_truncate(ex: dict[str, list[int]], max_len: int) -> dict[str, list[int]]:
    """Trim from the prompt side so completion tokens (the scored part) survive."""
    ids = ex["input_ids"]
    if len(ids) <= max_len:
        return ex
    labels = ex["labels"]
    comp_start = len(labels)
    for i, lab in enumerate(labels):
        if lab != -100:
            comp_start = i
            break
    comp_len = len(ids) - comp_start
    if comp_len >= max_len:
        start = comp_start
    else:
        start = comp_start - (max_len - comp_len)
    return {
        "input_ids": ids[start : start + max_len],
        "labels": labels[start : start + max_len],
    }


def adaptive_max_len(lengths: list[int], *, ceiling: int, floor: int = 128, align: int = 64) -> int:
    """p99 of tokenized lengths with a 10% buffer, aligned, clamped to [floor, ceiling]."""
    if not lengths:
        return ceiling
    s = sorted(lengths)
    p99 = s[min(len(s) - 1, int(0.99 * len(s)))]
    target = int(p99 * 1.1)
    target = ((target + align - 1) // align) * align
    return max(floor, min(target, ceiling))


def random_subsample(examples: list[Any], target: int, *, seed: int = 7) -> list[Any]:
    if target >= len(examples):
        return list(examples)
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    keep = sorted(idx[:target])
    return [examples[i] for i in keep]
