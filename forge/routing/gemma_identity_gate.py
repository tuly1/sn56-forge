#!/usr/bin/env python3
"""Research-only clean Gemma2 identity gate.

The gate delegates tensor parsing and streaming hashing to Forge's existing
``base_identity`` helper. It is intentionally opt-in and fail-open: every
uncertain result keeps the incumbent route. A public reference table must be
supplied by the owner; this module does not contain one.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
from pathlib import Path
from typing import Any

DEFAULT_ENV = "FORGE_V2_GEMMA2_CLEAN_LORA"
GEMMA_ARCH = "Gemma2ForCausalLM"
IDENTICAL = "identical"
CONFIG_ANON_KEYS = frozenset({"_name_or_path", "name_or_path", "_commit_hash"})


def _helper():
    # The deployed source package supplies this module. Importing it here,
    # rather than copying its parser, keeps the gate tied to the reviewed Forge
    # implementation and lets the CPU tests inject that exact source tree.
    return importlib.import_module("forge.routing.base_identity")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_table(table_dir: Path, key: str) -> tuple[dict[str, Any] | None, str | None]:
    if not key or Path(key).name != key or not key.endswith(".json"):
        return None, "invalid_reference_key"
    table = table_dir / key
    if not table.is_file() or table.is_symlink():
        return None, "missing_reference_table"
    config = table_dir / (key.removesuffix(".json") + ".config.json")
    # The shortlist helper consumes this adjacent config; do not let a
    # symlinked or missing config silently alter candidate selection.
    if not config.is_file() or config.is_symlink():
        return None, "missing_reference_config"
    try:
        obj = json.loads(table.read_text(encoding="utf-8"))
        tensors = obj["tensors"]
        if not isinstance(tensors, dict) or not tensors:
            return None, "malformed_reference_table"
        return obj, None
    except Exception as exc:  # noqa: BLE001
        return None, f"reference_error:{type(exc).__name__}"


def _validate_safetensors_headers(model_dir: Path) -> tuple[bool, str]:
    """Reject parser ambiguities before delegating byte hashing to Forge."""
    names: set[str] = set()
    try:
        for path in sorted(model_dir.glob("*.safetensors")):
            ranges: list[tuple[int, int]] = []
            size = path.stat().st_size
            with path.open("rb") as fh:
                raw_n = fh.read(8)
                if len(raw_n) != 8:
                    return False, "short_safetensors_header_length"
                header_n = struct.unpack("<Q", raw_n)[0]
                if header_n > size - 8:
                    return False, "safetensors_header_out_of_range"
                raw_header = fh.read(header_n)
                pairs: list[tuple[str, Any]] = []
                def collect_top_level(xs):
                    obj = dict(xs)
                    # Nested tensor metadata has only dtype/shape/data_offsets;
                    # collect only the outer tensor-name mapping so duplicate
                    # names across the JSON header cannot collapse silently.
                    if ("__metadata__" in obj or
                            any(isinstance(v, dict) and "data_offsets" in v for _, v in xs)):
                        pairs.extend(xs)
                    return obj
                header = json.loads(raw_header, object_pairs_hook=collect_top_level)
                if not isinstance(header, dict):
                    return False, "safetensors_header_not_object"
                payload = size - 8 - header_n
                for name, meta in pairs:
                    if name == "__metadata__":
                        continue
                    if name in names:
                        return False, "duplicate_tensor_name"
                    names.add(name)
                    if not isinstance(meta, dict) or not isinstance(meta.get("dtype"), str):
                        return False, "malformed_tensor_metadata"
                    shape = meta.get("shape")
                    offsets = meta.get("data_offsets")
                    if (not isinstance(shape, list) or
                            any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in shape) or
                            not (isinstance(offsets, list) and len(offsets) == 2 and
                                 all(isinstance(v, int) and not isinstance(v, bool) for v in offsets))):
                        return False, "malformed_tensor_shape_or_offsets"
                    start, end = offsets
                    if start < 0 or end < start or end > payload:
                        return False, "tensor_offsets_out_of_range"
                    ranges.append((start, end))
            ranges.sort()
            if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
                return False, "overlapping_tensor_ranges"
        if not names:
            return False, "no_tensor_records"
        return True, "well_formed_safetensors_headers"
    except Exception as exc:  # noqa: BLE001
        return False, f"safetensors_parse_error:{type(exc).__name__}"


def _verify_reference_receipt(table_dir: Path, key: str, receipt_path: Path, helper: Any,
                              anchor: dict[str, Any] | None) -> tuple[bool, str]:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False, "missing_reference_receipt"
    table_path = table_dir / f"{key}.json"
    config_path = table_dir / f"{key}.config.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("public_key") != key or not isinstance(receipt.get("source_reference"), str) or not receipt["source_reference"]:
            return False, "reference_receipt_identity_missing"
        helper_path = Path(str(helper.__file__)).resolve()
        if receipt.get("identity_helper_sha256") != _sha(helper_path):
            return False, "reference_helper_binding_mismatch"
        if receipt.get("table_sha256") != _sha(table_path) or receipt.get("config_sha256") != _sha(config_path):
            return False, "reference_file_binding_mismatch"
        if not isinstance(anchor, dict):
            return False, "reference_anchor_required"
        for field in ("public_key", "source_reference", "identity_helper_sha256", "table_sha256", "config_sha256"):
            if receipt.get(field) != anchor.get(field):
                return False, f"reference_anchor_mismatch:{field}"
        return True, "reference_receipt_verified"
    except Exception as exc:  # noqa: BLE001
        return False, f"reference_receipt_error:{type(exc).__name__}"


def _strict_reference_match(model_dir: Path, table_dir: Path, key: str, helper: Any) -> tuple[bool, str]:
    """Use the existing streamed records, then require every identity field."""
    table, error = _safe_table(table_dir, f"{key}.json")
    if error:
        return False, error
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        ref_config = json.loads((table_dir / f"{key}.config.json").read_text(encoding="utf-8"))
        config = {k: v for k, v in config.items() if k not in CONFIG_ANON_KEYS}
        ref_config = {k: v for k, v in ref_config.items() if k not in CONFIG_ANON_KEYS}
        if config != ref_config:
            return False, "config_identity_mismatch"
        safe, reason = _validate_safetensors_headers(model_dir)
        if not safe:
            return False, reason
        delivered = helper.hash_safetensors(model_dir)
        reference = table["tensors"]
        if set(delivered) != set(reference):
            return False, "tensor_name_set_mismatch"
        for name in sorted(reference):
            got, want = delivered[name], reference[name]
            for field in ("sha256", "shape", "dtype"):
                if got.get(field) != want.get(field):
                    return False, f"tensor_{field}_mismatch:{name}"
        return True, "exact_public_tensor_identity"
    except Exception as exc:  # noqa: BLE001
        return False, f"identity_error:{type(exc).__name__}"


def choose_route(model_dir: str | os.PathLike, *, route: dict[str, Any],
                 expected_public_key: str | None, tables_dir: str | os.PathLike,
                 table_receipt: str | os.PathLike | None = None,
                 reference_anchor: dict[str, Any] | None = None,
                 enabled: bool | None = None, max_bytes: int | None = None) -> dict[str, Any]:
    """Return an admission receipt; uncertain inputs retain the incumbent.

    The clean-base result is only a qualification for an explicitly opted-in
    Gemma2 native LoRA route. FIELD/full, KL, generic routes, missing metadata,
    unknown/altered bases, malformed tables and hashing errors all preserve the
    incumbent route. `reference_anchor` must come from an independently
    reviewed owner receipt; the gate never treats a receipt generated beside a
    table as its own trust anchor.
    """
    out = {"action": "incumbent", "status": "disabled", "reason": "opt_in_required"}
    if enabled is None:
        enabled = os.environ.get(DEFAULT_ENV, "0") == "1"
    if not enabled:
        return out
    if not isinstance(route, dict):
        out.update(status="skipped", reason="route_metadata_missing")
        return out
    if route.get("model_type") != "gemma2" or route.get("strategy") != "lora":
        out.update(status="skipped", reason="not_gemma2_native_lora")
        return out
    kl_flag = route.get("is_kl", route.get("use_kl"))
    if route.get("field") is not False or kl_flag is not False:
        out.update(status="skipped", reason="field_or_kl_route")
        return out
    micro, accum, effective = route.get("micro_batch"), route.get("grad_accum"), route.get("effective_batch")
    if (route.get("cuda_available") is not True or route.get("device") not in ("cuda", "cuda:0") or
            not isinstance(route.get("n_gpus"), int) or isinstance(route.get("n_gpus"), bool) or route.get("n_gpus") != 1 or route.get("sharded") is not False or
            not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (micro, accum, effective)) or
            accum <= 1 or effective != micro * accum or
            ("expected_grad_accum" in route and route.get("expected_grad_accum") != accum) or
            ("actual_grad_accum" in route and route.get("actual_grad_accum") != accum)):
        out.update(status="skipped", reason="runtime_geometry_or_device_unqualified")
        return out
    if not isinstance(expected_public_key, str) or not expected_public_key:
        out.update(status="skipped", reason="public_reference_key_missing")
        return out
    if not Path(model_dir).is_dir():
        out.update(status="skipped", reason="model_directory_missing")
        return out
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    tensor_files = list(model_path.glob("*.safetensors"))
    if (not config_path.is_file() or config_path.is_symlink() or not tensor_files or
            any(not p.is_file() or p.is_symlink() for p in tensor_files)):
        out.update(status="error", reason="unsafe_model_file_set")
        return out
    try:
        helper = _helper()
    except Exception as exc:  # noqa: BLE001
        out.update(status="error", reason=f"identity_helper_unavailable:{type(exc).__name__}")
        return out
    receipt_ok, receipt_reason = _verify_reference_receipt(Path(tables_dir), expected_public_key,
                                                            Path(table_receipt) if table_receipt else Path(""), helper,
                                                            reference_anchor)
    if not receipt_ok:
        out.update(status="skipped", reason=receipt_reason)
        return out
    kwargs = {} if max_bytes is None else {"max_bytes": max_bytes}
    try:
        verdict = helper.identify(model_dir, tables_dir=Path(tables_dir), **kwargs)
    except Exception as exc:  # defensive: the reviewed helper itself is fail-open
        out.update(status="error", reason=f"identity_exception:{type(exc).__name__}")
        return out
    out["identity"] = verdict
    if verdict.get("status") != IDENTICAL or verdict.get("candidate") != expected_public_key:
        out.update(status=str(verdict.get("status", "unknown")), reason="public_tensor_identity_not_exact")
        return out
    exact, reason = _strict_reference_match(Path(model_dir), Path(tables_dir), expected_public_key, helper)
    if not exact:
        out.update(status="altered", reason=reason)
        return out
    out.update(action="qualified_gemma2_native_lora", status=IDENTICAL,
               reason="exact_public_tensor_identity", public_key=expected_public_key)
    return out
