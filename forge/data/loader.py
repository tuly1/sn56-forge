"""Load the pre-staged training dataset from the read-only cache.

For tournament tasks the validator downloads the data before the container runs
and mounts it read-only at `/cache/datasets/{task_id}_train_data.json` (the
`--dataset` URL is a dead link inside the network-isolated container). We read
that file defensively — it may be a JSON array, a JSON-lines file, or an object
wrapping the rows — and never write back to `/cache`.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def load_rows(cached_path: str, *, dataset_arg: str | None, file_format: str) -> list[dict[str, Any]]:
    """Return the dataset as a list of row dicts.

    Resolution order: the standard cached path first (tournament case), then a
    local file named by `--dataset` (local test case). We deliberately do not
    attempt any network fetch — the container has no internet.
    """
    path = _resolve_path(cached_path, dataset_arg, file_format)
    if path is None:
        raise FileNotFoundError(
            f"no dataset found at {cached_path!r} or dataset arg {dataset_arg!r}"
        )

    if path.endswith(".csv") or file_format == "csv":
        return _read_csv(path)
    return _read_json_any(path)


def _resolve_path(cached_path: str, dataset_arg: str | None, file_format: str) -> str | None:
    if os.path.isfile(cached_path):
        return cached_path
    # HF-format datasets are staged as a directory; hand back a sentinel the
    # caller won't hit for s3 tasks but that keeps local `hf` tests working.
    if dataset_arg and os.path.isfile(dataset_arg):
        return dataset_arg
    if dataset_arg and os.path.isdir(dataset_arg):
        for name in ("train_data.json", "data.json", "dataset.json"):
            candidate = os.path.join(dataset_arg, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _read_json_any(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text_stripped = text.lstrip()
    if not text_stripped:
        return []

    # Whole-file JSON: array, or an object that wraps the rows under a key.
    if text_stripped[0] in "[{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            for key in ("data", "rows", "examples", "train"):
                val = obj.get(key)
                if isinstance(val, list):
                    return [r for r in val if isinstance(r, dict)]
            # A single record.
            return [obj]

    # JSON-lines.
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]
