"""Validated access to validator-provided model-prep baseline statistics.

The validator mounts a JSON file and exposes its path through
``BASELINE_STATS_PATH``.  These statistics are untrusted task input: consume a
small, explicitly validated summary rather than threading the raw mapping through
the trainer.  Policy decisions remain conservative and independently testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

_MAX_STATS_BYTES = 4 * 1024 * 1024
_MAX_SAFE_INTEGER = (1 << 63) - 1
_TASK_ALIASES = {
    "ChatTask": "instruct",
    "DpoTask": "dpo",
    "GrpoTask": "grpo",
    "InstructTextTask": "instruct",
}


@dataclass(frozen=True)
class BaselineSummary:
    task_type: str
    sha256: str
    sequence_p95: int
    sequence_p99: int
    sequence_max: int
    near_duplicate_rate: float
    total_tokens: int
    num_records: int
    tokens_per_sec: float | None


def load_baseline_summary(
    source: str | None, *, expected_task_type: str | None = None
) -> BaselineSummary | None:
    """Load a bounded JSON file/string and return its validated public summary.

    ``None`` means the validator supplied no statistics.  Malformed or mismatched
    data raises ``ValueError`` so the caller can record the failure and retain its
    static plan instead of silently acting on corrupt metadata.
    """
    if not source:
        return None
    payload = _read_payload(source)
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"invalid baseline-stats JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("baseline stats must be a JSON object")

    task_type = str(raw.get("task_type") or "").strip().lower()
    expected = _TASK_ALIASES.get(expected_task_type or "", expected_task_type or "")
    expected = str(expected).strip().lower()
    if task_type not in {"instruct", "dpo", "grpo"}:
        raise ValueError(f"unsupported baseline task_type {task_type!r}")
    if expected and task_type != expected:
        raise ValueError(
            f"baseline task_type {task_type!r} does not match expected {expected!r}"
        )

    dataset = _mapping(raw, "dataset")
    # Validate the complete top-level contract even though this first policy only
    # consumes dataset/throughput fields.  A partial payload must not be mistaken
    # for a production model-prep record.
    _mapping(raw, "weights")
    _mapping(raw, "training")
    seq = _mapping(dataset, "seq_length_distribution")
    p95 = _nonnegative_int(seq, "p95")
    p99 = _nonnegative_int(seq, "p99")
    maximum = _nonnegative_int(seq, "max")
    if not (p95 <= p99 <= maximum):
        raise ValueError("baseline sequence quantiles must satisfy p95 <= p99 <= max")

    duplicate_rate = _finite_float(dataset, "near_duplicate_rate")
    if not 0.0 <= duplicate_rate <= 1.0:
        raise ValueError("near_duplicate_rate must be in [0, 1]")

    total_tokens = _nonnegative_int(dataset, "total_tokens")
    num_records = _nonnegative_int(dataset, "num_records", default=0)
    throughput = raw.get("throughput")
    tokens_per_sec: float | None = None
    if throughput is not None:
        if not isinstance(throughput, dict):
            raise ValueError("baseline throughput must be an object or null")
        value = throughput.get("tokens_per_sec")
        if value is not None:
            tokens_per_sec = _as_finite_float(value, "throughput.tokens_per_sec")
            if tokens_per_sec <= 0:
                raise ValueError("throughput.tokens_per_sec must be positive")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return BaselineSummary(
        task_type=task_type,
        sha256=hashlib.sha256(canonical).hexdigest(),
        sequence_p95=p95,
        sequence_p99=p99,
        sequence_max=maximum,
        near_duplicate_rate=duplicate_rate,
        total_tokens=total_tokens,
        num_records=num_records,
        tokens_per_sec=tokens_per_sec,
    )


def suggested_sequence_length(
    summary: BaselineSummary | None,
    *,
    default: int,
    completion_reserve: int = 0,
    quantum: int = 256,
) -> int:
    """Return a conservative p99-based cap, never exceeding ``default``.

    The reserve is useful for prompt-only GRPO statistics, whose generated
    completion is absent from the observed prompt distribution.  Rounding to a
    stable quantum avoids overfitting the runtime plan to noisy point estimates.
    """
    default = max(1, int(default))
    if summary is None or summary.sequence_p99 <= 0:
        return default
    quantum = max(1, int(quantum))
    target = summary.sequence_p99 + max(0, int(completion_reserve))
    rounded = ((target + quantum - 1) // quantum) * quantum
    return max(1, min(default, rounded))


def telemetry_fields(summary: BaselineSummary) -> dict[str, Any]:
    """Small provenance/decision record safe to publish with the artifact."""
    return {
        "baseline_stats_sha256": summary.sha256,
        "baseline_task_type": summary.task_type,
        "baseline_seq_p99": summary.sequence_p99,
        "baseline_near_duplicate_rate": round(summary.near_duplicate_rate, 6),
        "baseline_num_records": summary.num_records,
        "baseline_total_tokens": summary.total_tokens,
        "baseline_tokens_per_sec": summary.tokens_per_sec,
    }


def _read_payload(source: str) -> str:
    if os.path.isfile(source):
        size = os.path.getsize(source)
        if size > _MAX_STATS_BYTES:
            raise ValueError(f"baseline-stats file exceeds {_MAX_STATS_BYTES} bytes")
        with open(source, encoding="utf-8") as fh:
            return fh.read(_MAX_STATS_BYTES + 1)
    if len(source.encode()) > _MAX_STATS_BYTES:
        raise ValueError(f"baseline-stats JSON exceeds {_MAX_STATS_BYTES} bytes")
    return source


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"baseline {key!r} must be an object")
    return value


def _nonnegative_int(
    mapping: dict[str, Any], key: str, *, default: int | None = None
) -> int:
    value = mapping.get(key, default)
    if value is None or isinstance(value, bool):
        raise ValueError(f"baseline {key!r} must be a nonnegative integer")
    try:
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError
            parsed = int(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped or not stripped.isdecimal():
                raise ValueError
            parsed = int(stripped)
        else:
            raise TypeError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"baseline {key!r} must be a nonnegative integer") from exc
    if parsed < 0 or parsed > _MAX_SAFE_INTEGER:
        raise ValueError(
            f"baseline {key!r} must be between 0 and {_MAX_SAFE_INTEGER}"
        )
    return parsed


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"baseline is missing {key!r}")
    return _as_finite_float(mapping[key], key)


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"baseline {label!r} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"baseline {label!r} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"baseline {label!r} must be finite")
    return parsed
