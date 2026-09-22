"""Is the delivered base a validator-damaged copy of a public checkpoint? (2026-09-22)

The validator augments about half of the text-task bases (pinned public source: weight scaling 0.40, magnitude
pruning 0.25, gaussian noise 0.20, layer re-init 0.15; norms and embeddings are never touched). On damaged bases
adapter entries trailed the winner by a median 21% against 5.7% for full-weight entries, and 20 of 21 winners shipped
full weights (research/20260922-damaged-base-audit/FINDINGS.md).

``assess`` combines named detectors into one verdict and never raises:
  "clean"    the identity check found a byte-identical public checkpoint (overrides every statistic)
  "damaged"  otherwise, at least one enabled detector says damaged
  "unknown"  anything else (no table, oversized, error, a different checkpoint of the same shape, no detector)
Callers treat only "damaged" as a reason to change routes.

Detectors (``DETECTORS``; ``register`` adds more, e.g. the weight-statistic prototype in
research/20260922-damage-detector/):
  identity       the base_identity report matched a shipped table's config, every tensor is present, some tensors
                 differ, and every norm/embedding tensor is byte-identical. A different public fine-tune of the same
                 shape changes its norms too, so it reads "unknown", not "damaged". Needs a shipped table.
  zero_fraction  reference-free magnitude-pruning signature: sampled 2-D non-norm/non-embedding matrices carry a large
                 fraction of exact zeros (validator pruning zeroes 25-50%; trained bf16 weights carry almost none).
                 Opt-in: it detects pruning only.
"""
from __future__ import annotations

import glob
import json
import os
import struct
import time
from collections.abc import Callable, Sequence
from typing import Any

from forge.routing.base_identity import ALTERED, IDENTICAL, is_protected

DAMAGED, CLEAN, UNKNOWN = "damaged", "clean", "unknown"

def identity_signature(model_dir: str, identity: dict[str, Any] | None) -> dict[str, Any]:
    """Damage verdict from a forge.routing.base_identity.identify report (no extra I/O)."""
    if not identity:
        return {"verdict": UNKNOWN, "reason": "no_identity_report"}
    status = identity.get("status")
    if status == IDENTICAL:
        return {"verdict": CLEAN, "reason": "identical"}
    if status != ALTERED:
        return {"verdict": UNKNOWN, "reason": f"identity_{status}"}
    if int(identity.get("missing") or 0) or int(identity.get("extra") or 0):
        return {"verdict": UNKNOWN, "reason": "tensor_set_differs"}
    if int(identity.get("altered") or 0) <= 0:
        return {"verdict": UNKNOWN, "reason": "nothing_altered"}
    if identity.get("protected") is None or int(identity.get("protected") or 0) <= 0:
        return {"verdict": UNKNOWN, "reason": "no_protected_tensors"}
    if int(identity.get("protected_altered") or 0) > 0:
        return {"verdict": UNKNOWN, "reason": "norm_or_embedding_altered"}  # another checkpoint, not augmentation
    return {"verdict": DAMAGED, "reason": "weights_altered_norms_embeddings_intact"}


ZERO_FRACTION_MIN = 0.20      # validator pruning: 25-50% of entries per matrix
ZERO_FRACTION_SAMPLE = 32     # 2-D matrices read (<= 512 KiB each), spread over the file order
ZERO_FRACTION_MIN_HITS = 3    # sampled matrices that must clear ZERO_FRACTION_MIN (multi-layer pruning spares layers)
_ZERO_DTYPES = {"BF16": ("<u2", 0x7FFF), "F16": ("<u2", 0x7FFF), "F32": ("<u4", 0x7FFFFFFF)}  # +0.0 and -0.0


def _zero_fraction(fh: Any, base: int, meta: dict[str, Any], max_bytes: int) -> float | None:
    import numpy as np

    spec = _ZERO_DTYPES.get(meta.get("dtype"))
    if spec is None:
        return None
    width = int(spec[0][-1])
    s, e = meta["data_offsets"]
    n = min(e - s, max_bytes) // width * width
    if n <= 0:
        return None
    fh.seek(base + s)
    arr = np.frombuffer(fh.read(n), dtype=spec[0])
    return float(np.count_nonzero((arr & spec[1]) == 0)) / max(1, arr.size)


def zero_fraction(model_dir: str, identity: dict[str, Any] | None = None, *, sample: int = ZERO_FRACTION_SAMPLE,
                  max_bytes_per_tensor: int = 1 << 19) -> dict[str, Any]:
    """Pruning signature from exact-zero fractions of sampled 2-D weight matrices (reads <= sample / 2 MiB)."""
    cands: list[tuple[str, str, dict[str, Any], int]] = []
    for f in sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__" or is_protected(name) or len(meta.get("shape") or []) != 2:
                continue
            if "lm_head" in name or min(meta["shape"]) < 16:
                continue
            cands.append((f, name, meta, 8 + n))
    if len(cands) < 3:
        return {"verdict": UNKNOWN, "reason": "too_few_matrices", "n": len(cands)}
    step = max(1, len(cands) // sample)
    picked = cands[::step][:sample]
    fracs: list[float] = []
    for f, name, meta, base in picked:
        with open(f, "rb") as fh:
            z = _zero_fraction(fh, base, meta, max_bytes_per_tensor)
        if z is not None:
            fracs.append(z)
    if len(fracs) < 3:
        return {"verdict": UNKNOWN, "reason": "unsupported_dtype", "n": len(fracs)}
    hits = sum(1 for z in fracs if z >= ZERO_FRACTION_MIN)
    med = sorted(fracs)[len(fracs) // 2]
    out = {"n": len(fracs), "hits": hits, "median_zero_fraction": round(med, 4), "max_zero_fraction": round(max(fracs), 4)}
    if hits >= ZERO_FRACTION_MIN_HITS:
        return {"verdict": DAMAGED, "reason": "pruning_signature", **out}
    return {"verdict": UNKNOWN, "reason": "no_pruning_signature", **out}


DETECTORS: dict[str, Callable[..., dict[str, Any]]] = {"identity": identity_signature, "zero_fraction": zero_fraction}


def register(name: str, fn: Callable[..., dict[str, Any]]) -> None:
    """Hook for further detectors: fn(model_dir, identity_report) -> {"verdict": damaged|clean|unknown, ...}."""
    DETECTORS[name] = fn


def assess(model_dir: str, *, identity: dict[str, Any] | None, detectors: Sequence[str]) -> dict[str, Any]:
    """Combine the named detectors. Never raises; unknown detector names and detector errors read "unknown"."""
    t0 = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    for name in detectors:
        fn = DETECTORS.get(name)
        if fn is None:
            results[name] = {"verdict": UNKNOWN, "reason": "no_such_detector"}
            continue
        try:
            rep = fn(str(model_dir), identity)
            results[name] = rep if isinstance(rep, dict) else {"verdict": UNKNOWN, "reason": "bad_result"}
        except Exception as exc:  # noqa: BLE001 - detection is advisory; the caller keeps its route
            results[name] = {"verdict": UNKNOWN, "reason": f"{type(exc).__name__}: {exc}"}
    verdicts = [r.get("verdict") for r in results.values()]
    if CLEAN in verdicts:  # a byte-identical public checkpoint is proof; no statistic overrides it
        verdict = CLEAN
    elif DAMAGED in verdicts:
        verdict = DAMAGED
    else:
        verdict = UNKNOWN
    by = [n for n, r in results.items() if r.get("verdict") == verdict] if verdict != UNKNOWN else []
    return {"verdict": verdict, "by": by, "detectors": results, "seconds": round(time.monotonic() - t0, 2)}
