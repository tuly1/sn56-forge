"""Deterministic train/validation split for best-checkpoint selection.

The tournament scores held-out loss, so we hold out a small slice ourselves and
submit the checkpoint that does best on it. The split is deliberately small —
training rows are the scarcer resource — and skipped entirely for tiny datasets
where a val slice would be too noisy to trust and too costly to spare.
"""

from __future__ import annotations

import random
from typing import Any

# Below this many examples a val split is more noise than signal.
_MIN_ROWS_FOR_SPLIT = 256
# ~3% of the data, bounded to keep eval cheap and training rich.
_VAL_FLOOR = 32
_VAL_CAP = 500


def split_train_val(
    examples: list[dict[str, Any]], *, seed: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train, val); val is empty when the dataset is too small."""
    n = len(examples)
    if n < _MIN_ROWS_FOR_SPLIT:
        return examples, []
    val_n = min(_VAL_CAP, max(_VAL_FLOOR, n // 33))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:val_n])
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [examples[i] for i in sorted(val_idx)]
    return train, val
