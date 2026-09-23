"""Exact identity of a delivered base model against known public checkpoints.

The validator hands miners an anonymised model directory; in about half of the tasks the
weights are augmented first (whole-matrix scaling, magnitude pruning, gaussian noise or
element re-initialisation). On such damaged bases the production adapter beat our
full-weight route on every damage type (2026-09-16 matrix on Qwen2.5-0.5B), while on
byte-identical public bases full weights were parity-to-better (Alpaca-20k, ru-AAQG-29k,
long-tail-29k). This module answers one question with no heuristics: is every tensor of
the delivered model byte-identical to a known public checkpoint?

Tables live next to this file in ``base_hashes/``:
  <key>.json         {"n_tensors": N, "tensors": {name: {"sha256", "shape", "dtype"}}}
  <key>.config.json  the public config.json, used only to shortlist candidate tables
Both are produced from the public safetensors files by ``build_table``; the hashes are of
the stored bytes (safetensors data ranges), which is what the validator serves.

Any doubt (unknown family, no table, hashing error, mismatch, oversized model) yields a
status other than "identical" and the caller keeps its existing route.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

TABLES_DIR = Path(__file__).resolve().parent / "base_hashes"
# config keys that must agree before a table is even considered
SHORTLIST_KEYS = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "intermediate_size",
                  "vocab_size", "num_attention_heads", "num_key_value_heads", "tie_word_embeddings")
MAX_BYTES_DEFAULT = 8 * 1024 ** 3  # hashing is streamed, but keep the check to the small models it was validated on
IDENTICAL, ALTERED, UNKNOWN, SKIPPED, ERROR = "identical", "altered", "unknown", "skipped", "error"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _norm(v: Any) -> Any:
    return list(v) if isinstance(v, (list, tuple)) else v


def candidate_keys(config: dict, tables_dir: Path | None = None) -> list[str]:
    """Keys of the shipped tables whose public config agrees with ``config`` on SHORTLIST_KEYS."""
    tables_dir = Path(tables_dir or TABLES_DIR)
    keys: list[str] = []
    for cfg_path in sorted(tables_dir.glob("*.config.json")):
        key = cfg_path.name[: -len(".config.json")]
        if not (tables_dir / f"{key}.json").exists():
            continue
        try:
            ref = _read_json(cfg_path)
        except Exception:  # noqa: BLE001 - a broken table is never a match
            continue
        if all(_norm(ref.get(k)) == _norm(config.get(k)) for k in SHORTLIST_KEYS):
            keys.append(key)
    return keys


def hash_safetensors(model_dir: str | os.PathLike, *, chunk: int = 1 << 24) -> dict[str, dict[str, Any]]:
    """Per-tensor SHA256 of the stored bytes of every ``*.safetensors`` file in ``model_dir``."""
    rec: dict[str, dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                fh.seek(base + s)
                h = hashlib.sha256()
                remaining = e - s
                while remaining > 0:
                    buf = fh.read(min(chunk, remaining))
                    if not buf:
                        raise IOError(f"short read in {f} for tensor {name}")
                    h.update(buf)
                    remaining -= len(buf)
                rec[name] = {"sha256": h.hexdigest(), "shape": list(meta["shape"]), "dtype": meta["dtype"],
                             "file": os.path.basename(f)}
    return rec


def build_table(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Table for a public checkpoint directory (safetensors + config.json)."""
    tensors = hash_safetensors(model_dir)
    return {"model_dir": str(model_dir), "n_tensors": len(tensors), "tensors": tensors}


def safetensors_bytes(model_dir: str | os.PathLike) -> int:
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(str(model_dir), "*.safetensors")))


def compare(delivered: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    altered = [k for k in reference if k in delivered and delivered[k]["sha256"] != reference[k]["sha256"]]
    missing = [k for k in reference if k not in delivered]
    extra = [k for k in delivered if k not in reference]
    status = IDENTICAL if not altered and not missing and not extra else ALTERED
    return {"status": status, "n_tensors": len(reference), "altered": len(altered), "missing": len(missing),
            "extra": len(extra), "altered_names": altered[:8]}


def identify(model_dir: str | os.PathLike, *, tables_dir: Path | None = None,
             max_bytes: int = MAX_BYTES_DEFAULT) -> dict[str, Any]:
    """Compare the delivered model with the shipped tables. Never raises."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"status": UNKNOWN, "candidate": None, "n_tensors": 0, "altered": 0, "missing": 0,
                           "extra": 0, "seconds": 0.0, "reason": None}
    try:
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            out["reason"] = "no_config"
            return out
        config = _read_json(cfg_path)
        keys = candidate_keys(config, tables_dir)
        if not keys:
            out["reason"] = "no_matching_table"
            return out
        size = safetensors_bytes(model_dir)
        if size <= 0:
            out["reason"] = "no_safetensors"
            return out
        if size > max_bytes:
            out.update(status=SKIPPED, reason=f"safetensors {size} bytes > {max_bytes}")
            return out
        delivered = hash_safetensors(model_dir)
        best: dict[str, Any] | None = None
        for key in keys:
            ref = _read_json(Path(tables_dir or TABLES_DIR) / f"{key}.json")["tensors"]
            rep = compare(delivered, ref)
            rep["candidate"] = key
            if rep["status"] == IDENTICAL:
                best = rep
                break
            if best is None or rep["altered"] + rep["missing"] + rep["extra"] < best["altered"] + best["missing"] + best["extra"]:
                best = rep
        assert best is not None
        out.update({k: best[k] for k in ("status", "candidate", "n_tensors", "altered", "missing", "extra")})
        out["altered_names"] = best.get("altered_names", [])
        return out
    except Exception as exc:  # noqa: BLE001 - identity is advisory; the caller keeps its route
        out.update(status=ERROR, reason=f"{type(exc).__name__}: {exc}")
        return out
    finally:
        out["seconds"] = round(time.monotonic() - t0, 2)
