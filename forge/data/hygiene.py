"""Field-route data hygiene (FORGE_V2_FIELD_HYGIENE=1, default off).

Four mechanisms the public reference stack applies before/while training, reimplemented
as pure functions over tokenized examples ({"input_ids", "labels"}) so they are testable
on CPU; sft_v2 wires them into the full-weight field route only:

* near-duplicate groups by MinHash-LSH over 5-character shingles of the rendered text
  (64 hashes, 16 bands of 4, estimated Jaccard >= 0.8, buckets above 200 members are
  common boilerplate and are skipped) -- removal keeps the first row of each group and
  is applied only when the validator reports near_duplicate_rate > 0.2 (or
  FORGE_V2_FIELD_DEDUP=1); the groups also keep near-duplicates on one side of the
  train/dev split (forge.data.split.grouped_stratified_split);
* a Gaussian "difficulty core" around the median completion length when the window
  covers less than half an epoch -- repeats buy little over fresh rows, so the kept set
  is sized to what the budget sees once (floor 2000 rows);
* a prompt-loss weight for prompt-dominated corpora (prompt/completion tokens > 5):
  prompt tokens enter the loss at plw = 0.05 / max(1, ratio / 5) through a per-token
  weight tensor from the collator and a weighted cross-entropy normalised by the sum of
  weights (labels stay masked, so dev evaluation is untouched);
* NEFTune escalation lives in sft_v2._neftune_alpha (alpha 5 when the validator's
  gradient_noise_scale exceeds 1.0).

No third-party near-duplicate library is available in the training image, so the MinHash
is numpy only.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import numpy as np

from forge.data.tokenize import PadCollator

ENV_KNOB = "FORGE_V2_FIELD_HYGIENE"
ENV_FORCE_DEDUP = "FORGE_V2_FIELD_DEDUP"

NEAR_DUP_RATE_THRESHOLD = 0.2  # validator near_duplicate_rate above which near-duplicates are removed
SHINGLE_K = 5
NUM_HASHES = 64
NUM_BANDS = 16
JACCARD_THRESHOLD = 0.8
MAX_BUCKET = 200

SUBSAMPLE_MAX_COVERAGE = 0.5  # subsample only when the window sees less than half an epoch
SUBSAMPLE_MIN_ROWS = 2000     # ... and the corpus is larger than this
SUBSAMPLE_FLOOR = 2000        # never keep fewer rows than this

PLW_RATIO_THRESHOLD = 5.0
PLW_BASE = 0.05

_PRIME = (1 << 31) - 1     # Mersenne prime: (a * x + b) % p with a, x < 2^31 stays inside int64
_SHINGLE_BASE = 1_000_003  # Horner base for the shingle hash; h * base + codepoint < 2^52


def enabled() -> bool:
    return os.environ.get(ENV_KNOB, "0") == "1"


def dedup_forced() -> bool:
    return os.environ.get(ENV_FORCE_DEDUP, "0") == "1"


# ---- near-duplicate grouping -------------------------------------------------------------

def _hash_coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=NUM_HASHES, dtype=np.int64)
    b = rng.integers(0, _PRIME, size=NUM_HASHES, dtype=np.int64)
    return a, b


def shingle_hashes(text: str, k: int = SHINGLE_K) -> np.ndarray:
    """Distinct hashes (< 2^31) of the k-character shingles of ``text``; a text shorter
    than k characters is one shingle. Vectorised over code points, so a 4k-token row
    costs well under a millisecond."""
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    if cps.size == 0:
        cps = np.zeros(1, dtype=np.int64)
    if cps.size < k:
        windows = cps.reshape(1, -1)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(cps, k)
    h = np.zeros(windows.shape[0], dtype=np.int64)
    for j in range(windows.shape[1]):
        h = (h * _SHINGLE_BASE + windows[:, j]) % _PRIME
    return np.unique(h)


def minhash_signature(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(NUM_HASHES,) minimum of each universal hash over the shingle set."""
    return ((a[:, None] * shingles[None, :] + b[:, None]) % _PRIME).min(axis=1)


def signature_agreement(text_a: str, text_b: str, *, seed: int = 42) -> float:
    """Fraction of agreeing MinHash values: an unbiased estimate of the shingle Jaccard."""
    a, b = _hash_coefficients(seed)
    sa = minhash_signature(shingle_hashes(text_a), a, b)
    sb = minhash_signature(shingle_hashes(text_b), a, b)
    return float(np.mean(sa == sb))


def near_duplicate_groups(
    texts: Sequence[str], *, threshold: float = JACCARD_THRESHOLD, max_bucket: int = MAX_BUCKET, seed: int = 42
) -> list[list[int]]:
    """Partition ``texts`` into near-duplicate groups (every index appears exactly once;
    members ascending, groups ordered by first member). Candidates come from rows that
    share one LSH band, are confirmed by signature agreement >= ``threshold`` and are
    merged transitively (union-find)."""
    n = len(texts)
    if n == 0:
        return []
    a, b = _hash_coefficients(seed)
    sigs = np.stack([minhash_signature(shingle_hashes(t), a, b) for t in texts])
    rows = NUM_HASHES // NUM_BANDS
    pairs: set[tuple[int, int]] = set()
    for band in range(NUM_BANDS):
        block = np.ascontiguousarray(sigs[:, band * rows : (band + 1) * rows])
        _, inverse, counts = np.unique(block, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse.reshape(-1), kind="stable")  # bucket members, ascending within a bucket
        start = 0
        for count in counts.tolist():
            members = order[start : start + count].tolist()
            start += count
            if 1 < count <= max_bucket:
                for x in range(count):
                    for y in range(x + 1, count):
                        pairs.add((members[x], members[y]))
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if pairs:
        idx = np.array(sorted(pairs), dtype=np.int64)
        agree = (sigs[idx[:, 0]] == sigs[idx[:, 1]]).mean(axis=1)
        for (i, j), ok in zip(idx.tolist(), (agree >= threshold).tolist()):
            if ok:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def render_texts(examples: Sequence[dict[str, Any]], tokenizer: Any) -> list[str]:
    """The rendered text of each tokenized row (prompt and completion, special tokens dropped)."""
    return [tokenizer.decode(list(ex["input_ids"]), skip_special_tokens=True) for ex in examples]


def dedup_triggered(summary: Any, *, forced: bool = False) -> bool:
    """Near-duplicate removal runs when forced or when the validator's near_duplicate_rate
    exceeds NEAR_DUP_RATE_THRESHOLD (absent statistics never trigger it)."""
    if forced:
        return True
    rate = getattr(summary, "near_duplicate_rate", None)
    try:
        return rate is not None and float(rate) > NEAR_DUP_RATE_THRESHOLD
    except (TypeError, ValueError):
        return False


def drop_near_duplicates(
    examples: Sequence[dict[str, Any]], groups: Sequence[Sequence[int]]
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Keep the first member of every group, in corpus order; returns the kept rows and
    their (now singleton) groups so the split path stays uniform."""
    keep = sorted(int(g[0]) for g in groups if len(g))
    kept = [examples[i] for i in keep]
    return kept, [[i] for i in range(len(kept))]


# ---- Gaussian difficulty core --------------------------------------------------------------

def coverage(*, rows: int, eff_batch: int, t_per_step: float | None, budget_s: float) -> float | None:
    """Fraction of one epoch the budget trains at the measured step time (rows the window
    sees once / rows). Computed on the FULL budget rather than the remaining window on
    purpose: the subsample target then runs ~20% generous (a breadth pad), so estimation
    error never turns into discarded fresh rows. None without a measurement."""
    if not t_per_step or t_per_step <= 0 or rows <= 0 or budget_s <= 0:
        return None
    steps_per_epoch = max(1, rows // max(1, eff_batch))
    achievable = budget_s * 0.85 / t_per_step
    return achievable / steps_per_epoch


def completion_lengths(examples: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.array([sum(1 for t in ex["labels"] if t != -100) for ex in examples], dtype=np.float64)


def gaussian_core(examples: Sequence[dict[str, Any]], target: int, *, seed: int = 7) -> list[dict[str, Any]]:
    """``target`` rows drawn without replacement with weights exp(-z^2/2), z the completion
    length's distance from the MEDIAN in standard deviations: the medium-difficulty core
    stays, trivial and noise-length extremes are thinned. Order is preserved."""
    n = len(examples)
    if target <= 0 or target >= n:
        return list(examples)
    lengths = completion_lengths(examples)
    median = float(np.median(lengths))
    std = max(float(np.std(lengths)), 1.0)
    z = np.clip((lengths - median) / std, -30.0, 30.0)
    weights = np.exp(-0.5 * z * z) + 1e-12  # the floor keeps every row eligible for the draw
    weights /= weights.sum()
    keep = np.sort(np.random.default_rng(seed).choice(n, size=target, replace=False, p=weights))
    return [examples[int(i)] for i in keep]


def difficulty_subsample(
    examples: Sequence[dict[str, Any]], *, coverage: float | None, seed: int = 7
) -> tuple[list[dict[str, Any]], int | None]:
    """Apply gaussian_core when measured coverage < SUBSAMPLE_MAX_COVERAGE and the corpus has
    more than SUBSAMPLE_MIN_ROWS rows; target = max(SUBSAMPLE_FLOOR, coverage * n).
    Returns (kept rows, target or None when nothing was done)."""
    n = len(examples)
    if coverage is None or not math.isfinite(coverage) or coverage >= SUBSAMPLE_MAX_COVERAGE or n <= SUBSAMPLE_MIN_ROWS:
        return list(examples), None
    target = max(SUBSAMPLE_FLOOR, int(coverage * n))
    if target >= n:
        return list(examples), None
    return gaussian_core(examples, target, seed=seed), target


# ---- prompt-loss weight --------------------------------------------------------------------

def prompt_completion_ratio(examples: Sequence[dict[str, Any]]) -> float:
    """Aggregate prompt tokens (masked labels, BOS included) over completion tokens."""
    prompt = 0
    total = 0
    for ex in examples:
        labels = ex["labels"]
        total += len(labels)
        prompt += sum(1 for t in labels if t == -100)
    return prompt / max(1, total - prompt)


def prompt_loss_weight(ratio: float) -> float:
    """0 up to a 5:1 prompt share; above it PLW_BASE shrinking with the ratio (a flat 5% on a
    28:1 corpus would put a third of the gradient on prompt tokens)."""
    if not math.isfinite(ratio) or ratio <= PLW_RATIO_THRESHOLD:
        return 0.0
    return PLW_BASE / max(1.0, ratio / PLW_RATIO_THRESHOLD)


class PromptWeightCollator(PadCollator):
    """PadCollator plus ``loss_weights`` (float32, batch x tokens): 1 on completion targets,
    ``plw`` on prompt tokens, 0 on padding and at position 0 (never a target after the
    causal shift). Labels are left masked so the same rows evaluate unchanged."""

    def __init__(self, pad_token_id: int | None, plw: float) -> None:
        super().__init__(pad_token_id)
        self.plw = float(plw)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        batch = super().__call__(features)
        labels = batch["labels"]
        weights = torch.full(labels.shape, self.plw, dtype=torch.float32)
        weights[labels != -100] = 1.0
        weights = weights * batch["attention_mask"].to(torch.float32)
        weights[:, 0] = 0.0
        batch["loss_weights"] = weights
        return batch


def weighted_token_loss(model: Any, inputs: dict[str, Any], *, use_fused: bool = True) -> Any:
    """Cross-entropy over the causally shifted targets, weighted per token by
    ``inputs["loss_weights"]`` and normalised by the weight sum. Masked (prompt) positions
    predict the next input token; zero-weight positions never touch the logits. With a
    fused linear-CE kernel available the logits are never materialised: one fused mean per
    distinct weight level (completion 1.0, prompt plw), rescaled to sums."""
    import torch
    import torch.nn.functional as F

    from forge.tuning.rowloss import _fused_loss_fn, unwrap_causal_lm

    base = unwrap_causal_lm(model)
    labels = inputs["labels"]
    kwargs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")}
    hidden = base.model(**kwargs, use_cache=False, return_dict=True).last_hidden_state
    targets = torch.where(labels != -100, labels, inputs["input_ids"])[:, 1:]
    weights = inputs["loss_weights"][:, 1:].to(torch.float32)
    active = weights > 0
    if not bool(active.any()):
        return hidden.sum() * 0.0
    denom = weights.sum()
    h = hidden[:, :-1]
    lm_weight = base.lm_head.weight
    softcap = getattr(getattr(base, "config", None), "final_logit_softcapping", None)
    fused = _fused_loss_fn(softcap) if use_fused else None
    if fused is not None:
        total = None
        for level in torch.unique(weights[active]):
            sel = weights == level
            part = fused(lm_weight, h[sel], targets[sel]) * sel.sum() * level
            total = part if total is None else total + part
        return total / denom
    logits = F.linear(h[active], lm_weight.to(h.dtype)).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    ce = F.cross_entropy(logits, targets[active], reduction="none")
    return (ce * weights[active]).sum() / denom
