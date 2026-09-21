"""Fail-open production adapter for the reviewed Gemma small-batch candidate.

This module only qualifies the already reviewed small-batch recipe. It does not
set experiment flags, change identity tables, or alter the incumbent route.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from forge.routing import gemma_identity_gate as gate

DEFAULT_FLAG = "FORGE_V2_GEMMA_ADAPTER_SMALL_BATCH"
BUNDLED_REFERENCE_DIR = Path(__file__).with_name("gemma_reference_v1")
EXPECTED_GATE_SHA256 = "ae987874281a8f723437f5168153503467d374a8948f329850f1a691b8472e64"
EXPECTED_ANCHOR_SHA256 = "bf099b1f56e2ce079e9f96cb70095fe2d5abbe9dccc8fa26850c869873e8ad50"
EXPECTED_REPLAY_SHA256 = "9bb28fb06586e9de4c794842410fc560a3319463147f424c65d6ba00bdae1381"
EXPECTED_FIELDS = {
    "public_key": "unsloth__gemma-2-2b",
    "source_reference": "unsloth/gemma-2-2b@25319945f7fd83b8b903e12081777b7eef2ba993;weight-sha256:40f7727761523db40b475358377c9a9b0f0d8fcf7ef8b869e71ae4f0ef12a555",
    "identity_helper_sha256": "012de809ea09ff4bfa656d1253deda23f05636b946b6a51b57fdebaf2b1c93ea",
    "table_sha256": "637879bc4e4a97913194dbb97ffd0e48afa7afd57b94cbebaa2c36985606512e",
    "config_sha256": "e7fd1f5621e3956cc2795c3e69d775da31b04a5aae436fa892ab2fe75285aa11",
    "root_proof_sha256": "22e94177817a29b51712a3b6eb18959b13d6f298d45d2193bc1d6b14eb8a1aab",
    "build_receipt_sha256": "d4bb2b57d25c94146251400e25e5eb109c7760c35edd0c42943b26d4b349b2b0",
}
EXPECTED_REFERENCE_TABLE = "unsloth__gemma-2-2b.json"
EXPECTED_REFERENCE_CONFIG = "unsloth__gemma-2-2b.config.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incumbent(reason: str, *, status: str = "skipped", **details: Any) -> dict[str, Any]:
    return {"action": "incumbent", "status": status, "reason": reason, **details}


def qualify(*, model_dir: str | os.PathLike, model_type: str, strategy: str,
            field: bool, is_kl: bool, cuda_available: bool, device: str,
            n_gpus: int, sharded: bool, micro_batch: int, grad_accum: int,
            effective_batch: int, enabled: bool | None = None,
            reference_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return a small-batch admission receipt; all uncertainty retains incumbent."""
    if enabled is None:
        enabled = os.environ.get(DEFAULT_FLAG, "1") != "0"
    if not enabled:
        return _incumbent("opt_in_required", status="disabled")
    if (str(model_type).lower() != "gemma2" or strategy != "lora" or field or is_kl or
            cuda_available is not True or device not in ("cuda", "cuda:0") or n_gpus != 1 or sharded):
        return _incumbent("route_not_qualified")
    # These are the two prospectively declared CAS/RAG control geometries.
    # Packaging this gate remains conditional on both fixtures' scored admission.
    if (micro_batch, grad_accum, effective_batch) not in ((5, 4, 20), (10, 2, 20)):
        return _incumbent("default_geometry_not_qualified")
    if os.environ.get("FORGE_V2_EXPERIMENT_INIT_SEED", ""):
        return _incumbent("initializer_experiment_flag_not_deployable")
    if any(os.environ.get(name, "0") == "1" for name in
           ("FORGE_V2_ADAPTER_SWEEP", "FORGE_V2_TWO_SEED")):
        return _incumbent("conflicting_experiment_flag")
    # Keep the deployable default scoped to the scored adapter recipe. Explicit
    # rank/alpha/dropout, LR, effective-batch, precision, or adapter-family
    # overrides describe a different experiment and retain the incumbent.
    if os.environ.get("FORGE_V2_LORA_R", "32") not in ("", "32") or os.environ.get("FORGE_V2_LORA_ALPHA", "64") not in ("", "64"):
        return _incumbent("lora_shape_override")
    try:
        if float(os.environ.get("FORGE_V2_LORA_DROPOUT", "0") or 0) != 0.0:
            return _incumbent("lora_dropout_override")
        if float(os.environ.get("FORGE_V2_LORA_PLUS", "0") or 0) > 1.0:
            return _incumbent("lora_plus_override")
        if float(os.environ.get("FORGE_V2_LR", "0") or 0) != 0.0:
            return _incumbent("lr_override")
    except ValueError:
        return _incumbent("malformed_recipe_override")
    if any(os.environ.get(name, "") not in ("", "0") for name in
           ("FORGE_V2_EFF_BATCH", "FORGE_V2_FIELD_EFF_BATCH", "FORGE_V2_FP32_MASTER", "FORGE_V2_DORA", "FORGE_V2_RSLORA")):
        return _incumbent("conflicting_recipe_override")
    if reference_dir is not None and Path(reference_dir).resolve() != BUNDLED_REFERENCE_DIR.resolve():
        return _incumbent("external_reference_not_allowed")
    # The production endpoint does not propagate arbitrary miner environment.
    # The reviewed public reference is shipped beside this module and is the
    # only identity anchor accepted by this source overlay.
    ref = BUNDLED_REFERENCE_DIR
    try:
        gate_path = Path(str(gate.__file__)).resolve()
        if _sha(gate_path) != EXPECTED_GATE_SHA256:
            return _incumbent("gate_source_hash_mismatch")
        anchor_path = ref / "ROOT-REFERENCE-ANCHOR.json"
        if not anchor_path.is_file() or anchor_path.is_symlink() or _sha(anchor_path) != EXPECTED_ANCHOR_SHA256:
            return _incumbent("reference_anchor_hash_mismatch")
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        if any(anchor.get(key) != value for key, value in EXPECTED_FIELDS.items()):
            return _incumbent("reference_anchor_fields_mismatch")
        replay = ref / "ROOT-ACTUAL-GATE-REPLAY.json"
        if not replay.is_file() or replay.is_symlink() or _sha(replay) != EXPECTED_REPLAY_SHA256:
            return _incumbent("actual_gate_replay_hash_mismatch")
        table = ref / EXPECTED_REFERENCE_TABLE
        config = ref / EXPECTED_REFERENCE_CONFIG
        receipt = ref / "BUILD-RECEIPT.json"
        if any((not path.is_file() or path.is_symlink()) for path in (table, config, receipt)):
            return _incumbent("reference_dependency_missing")
        if (_sha(table) != EXPECTED_FIELDS["table_sha256"] or
                _sha(config) != EXPECTED_FIELDS["config_sha256"] or
                _sha(receipt) != EXPECTED_FIELDS["build_receipt_sha256"]):
            return _incumbent("reference_dependency_hash_mismatch")
        result = gate.choose_route(
            model_dir,
            route={"model_type": "gemma2", "strategy": "lora", "field": False,
                   "is_kl": False, "cuda_available": True, "device": device,
                   "n_gpus": 1, "sharded": False, "micro_batch": micro_batch,
                   "grad_accum": grad_accum, "effective_batch": effective_batch},
            expected_public_key=EXPECTED_FIELDS["public_key"], tables_dir=ref,
            table_receipt=receipt, reference_anchor=anchor, enabled=True,
        )
        if result.get("action") != "qualified_gemma2_native_lora":
            return _incumbent("identity_gate_not_qualified", gate=result)
        return {"action": "enable_gemma_small_batch", "status": "qualified",
                "reason": "exact_public_identity_and_default_geometry", "gate": result,
                "dependencies": {"gate_sha256": EXPECTED_GATE_SHA256,
                                 "anchor_sha256": EXPECTED_ANCHOR_SHA256,
                                 "table_sha256": EXPECTED_FIELDS["table_sha256"],
                                 "config_sha256": EXPECTED_FIELDS["config_sha256"],
                                 "receipt_sha256": EXPECTED_FIELDS["build_receipt_sha256"],
                                 "replay_sha256": EXPECTED_REPLAY_SHA256}}
    except Exception as exc:  # fail-open: identity or packaging uncertainty is incumbent
        return _incumbent(f"integration_exception:{type(exc).__name__}", status="error")
