"""Exact identity report for routing and damage detection (2026-09-23).

forge/routing/base_identity.py is BYTE-PINNED by the deployed Gemma small-batch admission
(gemma_small_batch_admission.EXPECTED_FIELDS["identity_helper_sha256"]): any edit to it makes that admission fall back
to the incumbent. It is therefore left exactly as released in c0dc300, and the extensions live here:

``identify`` does one streamed hash pass (base_identity.hash_safetensors) against the shipped tables and returns the
base_identity.identify report unchanged - same candidate choice, same ``status`` - so the identical-base admission sees
exactly what c0dc300 saw, plus the fields damage detection needs:
  protected / protected_altered  norm and embedding tensors (the validator's augmentation never touches them)
  structure_changed              shared tensors whose dtype or shape differs (a re-export, not augmentation)
  exact_status                   "identical" only when bytes, tensor set, dtypes and shapes all match; else "altered"
                                 (for reports that compared against a table), else the base status
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from forge.routing import base_identity as _bid

IDENTICAL, ALTERED, UNKNOWN, SKIPPED, ERROR = _bid.IDENTICAL, _bid.ALTERED, _bid.UNKNOWN, _bid.SKIPPED, _bid.ERROR


def is_protected(name: str) -> bool:
    """Tensors the validator's augmentation never touches (norms and embeddings; pinned public source,
    research/20260922-damaged-base-audit). A clean fine-tune of the same architecture changes them too."""
    n = name.lower()
    return "norm" in n or "embed" in n


def compare(delivered: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """base_identity.compare plus protected / structure fields and the strict ``exact_status``."""
    rep = _bid.compare(delivered, reference)
    altered = [k for k in reference if k in delivered and delivered[k]["sha256"] != reference[k]["sha256"]]
    rep["protected"] = sum(1 for k in reference if is_protected(k))
    rep["protected_altered"] = sum(1 for k in altered if is_protected(k))
    rep["structure_changed"] = sum(1 for k in reference if k in delivered and any(
        f in reference[k] and f in delivered[k] and _bid._norm(reference[k][f]) != _bid._norm(delivered[k][f])
        for f in ("dtype", "shape")))
    rep["exact_status"] = IDENTICAL if rep["status"] == IDENTICAL and not rep["structure_changed"] else ALTERED
    return rep


def identify(model_dir: str | os.PathLike, *, tables_dir: Path | None = None,
             max_bytes: int = _bid.MAX_BYTES_DEFAULT) -> dict[str, Any]:
    """base_identity.identify (same candidate selection and ``status``) plus the fields above. Never raises."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"status": UNKNOWN, "candidate": None, "n_tensors": 0, "altered": 0, "missing": 0,
                           "extra": 0, "seconds": 0.0, "reason": None}
    try:
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            out["reason"] = "no_config"
            return out
        config = _bid._read_json(cfg_path)
        keys = _bid.candidate_keys(config, tables_dir)
        if not keys:
            out["reason"] = "no_matching_table"
            return out
        size = _bid.safetensors_bytes(model_dir)
        if size <= 0:
            out["reason"] = "no_safetensors"
            return out
        if size > max_bytes:
            out.update(status=SKIPPED, reason=f"safetensors {size} bytes > {max_bytes}")
            return out
        delivered = _bid.hash_safetensors(model_dir)
        best: dict[str, Any] | None = None
        for key in keys:  # identical selection rule to base_identity.identify
            ref = _bid._read_json(Path(tables_dir or _bid.TABLES_DIR) / f"{key}.json")["tensors"]
            rep = compare(delivered, ref)
            rep["candidate"] = key
            if rep["status"] == IDENTICAL:
                best = rep
                break
            if best is None or rep["altered"] + rep["missing"] + rep["extra"] < best["altered"] + best["missing"] + best["extra"]:
                best = rep
        assert best is not None
        out.update({k: best[k] for k in ("status", "candidate", "n_tensors", "altered", "missing", "extra",
                                         "protected", "protected_altered", "structure_changed", "exact_status")})
        out["altered_names"] = best.get("altered_names", [])
        return out
    except Exception as exc:  # noqa: BLE001 - identity is advisory; the caller keeps its route
        out.update(status=ERROR, reason=f"{type(exc).__name__}: {exc}")
        return out
    finally:
        out["seconds"] = round(time.monotonic() - t0, 2)
