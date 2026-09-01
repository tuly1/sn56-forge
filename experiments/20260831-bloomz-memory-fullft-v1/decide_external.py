#!/usr/bin/env python3
"""Fail-closed BloomZ dev selection and one-shot confirmation verdict."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
SCORE_SCHEMA = "sn56.bloomz-external-score.v1"; DECISION_SCHEMA = "sn56.bloomz-dev-decision.v1"
VERDICT_SCHEMA = "sn56.bloomz-confirmation-verdict.v1"
AUTHORIZED = "AUTHORIZED_FOR_ONE_CONFIRMATION"
STOP = "STOP_NO_SCIENCE"
LOCAL_SCOPE = "matched_public_fixture_only_no_official_calibration"
IMAGE = "gradientsio/text-evaluator:basilica@sha256:860d49c7317a82b68d93b7e0e257091d810fdea12eee3013f373903092d279d0"
DRIVER_SHA256 = "6952bf4a9b365fa00387b87dd813eaf69d1ad8d0a555a668751990a673a1b0a3"
DEV = {"sha256": "f5548b1864a55c208f9f8061cb0e1d2471a6e58b976bb532ffdbb7a584bbfad6", "row_count": 1024, "vector_count": 1021}
CONFIRMATION = {"filename": "confirmation.jsonl", "sha256": "2b1a788ed12051688402d6709f75c7e1727d26711f4a52c9925d9eff5892c7ae", "row_count": 512, "expected_vector_count": 511}
AUTHORITY_KEYS = ("runtime_authority_sha256", "source_commit", "source_tree", "forge_child_tree", "experiment_child_tree", "runtime_source_inventory_sha256", "training_image_reference", "training_image_id", "experiment_config_sha256")
HEX40, HEX64, HEX32 = (re.compile(rf"^[0-9a-f]{{{width}}}$") for width in (40, 64, 32)); IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
class DecisionError(RuntimeError): pass
def need(condition: object, message: str) -> None:
    if not condition:
        raise DecisionError(message)
def canonical(value: Any, *, newline: bool = False) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + (b"\n" if newline else b"")
def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()
def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
def read_json(path: Path, label: str, limit: int = 64_000_000) -> tuple[Path, Any, str]:
    expanded = path.expanduser()
    need(not expanded.is_symlink(), f"{label} is a symlink")
    resolved = expanded.resolve(strict=True)
    need(resolved.is_file() and 0 < resolved.stat().st_size <= limit, f"bad {label}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecisionError(f"invalid {label}: {exc}") from exc
    return resolved, value, sha_file(resolved)
def hex_value(value: Any, width: int, label: str) -> str:
    need(isinstance(value, str) and (HEX40 if width == 40 else HEX64).fullmatch(value), f"invalid {label}")
    return value
def finite(value: Any, label: str) -> float:
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"non-finite {label}")
    return float(value)
def authority(value: Any) -> dict[str, str]:
    need(isinstance(value, Mapping) and set(value) == set(AUTHORITY_KEYS), "runtime authority shape drift")
    result = dict(value)
    for key in ("runtime_authority_sha256", "runtime_source_inventory_sha256", "experiment_config_sha256"):
        hex_value(result[key], 64, key)
    for key in ("source_commit", "source_tree", "forge_child_tree", "experiment_child_tree"):
        hex_value(result[key], 40, key)
    need(isinstance(result["training_image_reference"], str) and "@sha256:" in result["training_image_reference"], "training image reference is not pinned")
    need(isinstance(result["training_image_id"], str) and IMAGE_ID.fullmatch(result["training_image_id"]), "invalid training image ID")
    return result
def gpu_admission(value: Any, strategy: str, expected_authority: Mapping[str, str]) -> dict[str, Any]:
    keys = {"receipt_path", "receipt_sha256", "status", "proof", "arm", "strategy", "learning_rate", "geometry", "model", "gpu", "authority"}
    need(isinstance(value, Mapping) and set(value) == keys, "GPU admission shape drift")
    result = dict(value)
    probe_path = Path(result["receipt_path"])
    resolved, probe, probe_sha = read_json(probe_path, "GPU admission receipt", 4_000_000)
    need(str(resolved) == result["receipt_path"] and probe_sha == result["receipt_sha256"], "GPU admission receipt identity drift")
    need(isinstance(probe, Mapping), "GPU admission receipt is not an object")
    for key in ("status", "proof", "arm", "strategy", "learning_rate", "geometry", "model", "authority"):
        need(result[key] == probe.get(key), f"GPU admission projection drift: {key}")
    need(isinstance(probe.get("gpu"), Mapping) and result["gpu"] == {key: probe["gpu"].get(key) for key in result["gpu"]}, "GPU admission projection drift: gpu")
    need(result["status"] == "PASS" and result["proof"] == "actual_b1_s2048_forward_backward_fused_adamw_step_plus_optimizer_resident_b1_s4096_labeled_eval", "GPU admission did not pass measured proof")
    arm = "control" if strategy == "lora" else "full"
    need(result["arm"] == arm and result["strategy"] == strategy, "GPU admission arm drift")
    allowed_lr = {1.5e-4} if strategy == "lora" else {1e-4}
    need(result["learning_rate"] in allowed_lr, "GPU admission learning rate drift")
    geometry = result["geometry"]
    need(geometry == {"microbatch": 1, "gradient_accumulation_configured": 16, "optimizer_probe_microsteps_executed": 1, "configured_effective_batch": 16, "sequence_length": 2048, "packing": False, "gradient_checkpointing": True}, "GPU admission geometry drift")
    model = result["model"]
    need(isinstance(model, Mapping) and model == {
        "repo": "bigscience/bloomz-560m", "revision": "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05",
        "config_sha256": "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb",
        "weights_sha256": "365b2c5e9bd1057eb1e3f1a4fc3f89ae6584d20f24b682d2406bc7e90178ec13",
        "output_width": 250880, "params_b": 0.559214592,
    }, "GPU admission model drift")
    gpu = result["gpu"]
    gpu_keys = {"name", "card_bytes", "train_peak_reserved_bytes", "train_peak_reserved_ratio", "selection_peak_reserved_bytes", "selection_peak_reserved_ratio", "max_reserved_ratio"}
    need(isinstance(gpu, Mapping) and set(gpu) == gpu_keys and "H100" in gpu["name"], "GPU admission device drift")
    need(isinstance(gpu["card_bytes"], int) and gpu["card_bytes"] >= 70 * (1024**3), "GPU admission card capacity drift")
    for prefix in ("train", "selection"):
        ratio = finite(gpu[f"{prefix}_peak_reserved_ratio"], "GPU reserved ratio")
        need(isinstance(gpu[f"{prefix}_peak_reserved_bytes"], int) and 0 <= ratio <= gpu["max_reserved_ratio"] <= 0.70, "GPU peak exceeds admission limit")
    need(result["authority"] == expected_authority, "GPU admission authority drift")
    return result
def inventory(path: Path, strategy: str) -> dict[str, Any]:
    resolved, data, digest = read_json(path, f"{strategy} inventory", 2_000_000)
    need(isinstance(data, Mapping), "inventory is not an object")
    need(data.get("status") == "EXTERNAL_SCORE_READY", "inventory is not score-ready")
    need(data.get("strategy") == strategy, "inventory strategy drift")
    need(data.get("phase") == ("control" if strategy == "lora" else "candidate"), "inventory phase drift")
    need(data.get("schedule_completed") is True, "training schedule is incomplete")
    need(data.get("planned_steps") == data.get("final_step") == 256, "matched schedule length drift")
    need(data.get("required_external_scores") == 4, "external-score count policy drift")
    need(data.get("completed_scheduled_evals", 0) >= 4, "fewer than four evaluations")
    need(data.get("capture_errors") == [], "checkpoint capture errors are present")
    checkpoints = data.get("checkpoints")
    need(isinstance(checkpoints, list) and len(checkpoints) == 4, "inventory must contain four checkpoints")
    trees = [hex_value(item.get("tree_sha256") if isinstance(item, Mapping) else None, 64, "checkpoint tree") for item in checkpoints]
    need(len(set(trees)) == 4, "inventory checkpoint trees are not distinct")
    bound_authority = authority(data.get("authority"))
    return {"path": str(resolved), "sha256": digest, "trees": trees, "authority": bound_authority,
            "gpu_admission": gpu_admission(data.get("gpu_admission"), strategy, bound_authority)}
def score(path: Path, phase: str, role: str) -> dict[str, Any]:
    resolved, data, digest = read_json(path, "score receipt", 4_000_000)
    need(resolved.name == "receipt.json", "score receipt filename drift")
    _, pointer, _ = read_json(resolved.with_name("receipt.sha256.json"), "receipt pointer")
    need(pointer == {"filename": "receipt.json", "sha256": digest}, "receipt pointer mismatch")
    need(isinstance(data, Mapping), "score receipt is not an object")
    need(data.get("schema_version") == SCORE_SCHEMA and data.get("kind") == "bloomz_digest_pinned_external_score" and data.get("status") == "PASS", "score receipt did not pass")
    need(data.get("phase") == phase, "score phase drift")
    artifact, fixture, evaluator = data.get("artifact"), data.get("fixture"), data.get("evaluator")
    need(all(isinstance(value, Mapping) for value in (artifact, fixture, evaluator)), "score binding is absent")
    need(artifact.get("role") == role, "artifact role drift")
    tree = hex_value(artifact.get("tree_sha256"), 64, "artifact tree")
    need(isinstance(artifact.get("files"), list) and sha_value(artifact["files"]) == tree, "artifact inventory/tree drift")
    transport = "peft_adapter" if role == "control" else "full_model"
    need(artifact.get("expected_transport") == transport, "expected transport drift")
    expected = DEV if phase == "selection" else {
        "sha256": CONFIRMATION["sha256"], "row_count": CONFIRMATION["row_count"],
        "vector_count": CONFIRMATION["expected_vector_count"],
    }
    need(fixture.get("phase") == phase and fixture.get("sha256") == expected["sha256"] and fixture.get("row_count") == expected["row_count"], "fixture identity drift")
    driver = evaluator.get("score_driver")
    need(evaluator.get("image") == IMAGE and evaluator.get("network") == "none", "evaluator identity drift")
    need(isinstance(driver, Mapping) and driver.get("sha256") == DRIVER_SHA256, "score-driver drift")
    bound_authority = authority(data.get("authority"))
    result = data.get("result")
    need(isinstance(result, Mapping) and result.get("raw_result_filename") == "raw-result.json", "score result drift")
    _, raw, raw_sha = read_json(resolved.with_name("raw-result.json"), "raw result")
    need(raw_sha == result.get("raw_result_sha256"), "raw-result hash drift")
    need(isinstance(raw, Mapping) and list(raw) == ["/artifact"], "raw artifact key drift")
    raw_result = raw["/artifact"]
    need(isinstance(raw_result, Mapping), "raw result is not an object")
    need(raw_result.get("is_finetune") is True, "raw result is not a finetune")
    vector_raw = raw_result.get("per_example_losses")
    need(isinstance(vector_raw, list), "raw vector is absent")
    vector = [finite(value, "raw vector value") for value in vector_raw]
    need(len(vector) == result.get("vector_count") == expected["vector_count"], "raw vector count drift")
    need(result.get("vector_sha256") == sha_value(vector), "raw vector fingerprint drift")
    fingerprint = raw_result.get("eval_set_fingerprint")
    need(isinstance(fingerprint, str) and HEX32.fullmatch(fingerprint) and fingerprint == result.get("eval_set_fingerprint"), "eval fingerprint drift")
    need(raw_result.get("sn56_local_artifact_transport") == result.get("transport") == transport, "artifact transport drift")
    scalar = finite(raw_result.get("eval_loss"), "external scalar")
    need(scalar == finite(result.get("eval_loss"), "receipt scalar"), "external scalar drift")
    base = data.get("base")
    need(isinstance(base, Mapping), "base identity is absent")
    return {
        "path": str(resolved), "sha256": digest, "tree": tree, "transport": transport,
        "scalar": scalar, "vector": vector, "vector_sha256": sha_value(vector),
        "fingerprint": fingerprint, "authority": bound_authority,
        "base_identity": hex_value(base.get("identity_sha256"), 64, "base identity"), "data": data,
    }
def validation(path: Path, scored: Mapping[str, Any]) -> dict[str, str]:
    resolved, data, digest = read_json(path, "artifact validation receipt", 4_000_000)
    need(isinstance(data, Mapping) and data.get("schema_version") == "sn56.bloomz-artifact-validation.v1" and data.get("status") == "PASS", "artifact validation did not pass")
    need(data.get("offline") is True and data.get("trust_remote_code") is False, "artifact validation was not offline-safe")
    need(data.get("artifact_tree_sha256") == scored["tree"] and data.get("artifact_format") == scored["transport"], "artifact validation identity drift")
    need(data.get("authority") == scored["authority"], "artifact validation authority drift")
    fresh, summary = data.get("fresh_exact_evaluator_load"), data.get("serialized_state_summary")
    need(isinstance(fresh, Mapping) and fresh.get("receipt_sha256") == scored["sha256"], "artifact validation external-score digest drift")
    need(isinstance(summary, Mapping) and summary.get("all_tensors_finite") is True, "serialized artifact tensors are not finite")
    need(isinstance(summary.get("tensor_count"), int) and summary["tensor_count"] > 0, "serialized artifact tensor count is invalid")
    need(all(isinstance(summary.get(key), int) and summary[key] > 0 for key in ("total_numel", "total_bytes", "finite_scan_chunk_elements")), "serialized artifact state totals are invalid")
    need(isinstance(summary.get("dtype_tensor_counts"), Mapping) and sum(summary["dtype_tensor_counts"].values()) == summary["tensor_count"], "serialized artifact dtype counts drift")
    files = summary.get("tensor_files")
    need(isinstance(files, list) and files and all(isinstance(item, Mapping) and HEX64.fullmatch(str(item.get("sha256", ""))) for item in files), "serialized tensor-file identity drift")
    hex_value(summary.get("tensor_schema_sha256"), 64, "serialized tensor schema SHA")
    return {"path": str(resolved), "sha256": digest}
def validation_set(paths: Sequence[Path], scores: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    need(len(paths) == len(scores) == 4, "exactly four artifact validations per arm are required")
    by_score = {item["sha256"]: item for item in scores}
    facts: dict[str, dict[str, str]] = {}
    for path in paths:
        _, data, _ = read_json(path, "artifact validation receipt", 4_000_000)
        fresh = data.get("fresh_exact_evaluator_load") if isinstance(data, Mapping) else None
        score_sha = fresh.get("receipt_sha256") if isinstance(fresh, Mapping) else None
        need(score_sha in by_score and score_sha not in facts, "artifact validation does not map one-to-one to score receipts")
        facts[score_sha] = validation(path, by_score[score_sha])
    need(set(facts) == set(by_score), "artifact validation set is incomplete")
    return facts
def shared(scores: Sequence[Mapping[str, Any]], expected_authority: Mapping[str, str]) -> tuple[str, str]:
    fingerprint, base = scores[0]["fingerprint"], scores[0]["base_identity"]
    for item in scores:
        need(item["fingerprint"] == fingerprint, "eval fingerprints differ")
        need(item["base_identity"] == base, "base identities differ")
        need(item["authority"] == expected_authority, "receipt/runtime authority mismatch")
    return str(fingerprint), str(base)
def paired(control: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    need(len(control) == len(candidate) and control, "ordered vectors are not pairable")
    deltas = [right - left for left, right in zip(control, candidate, strict=True)]
    return {
        "direction": "candidate_minus_control_lower_is_better", "pair_count": len(deltas),
        "candidate_minus_control_mean": sum(deltas) / len(deltas),
        "candidate_win_rate": sum(value < 0 for value in deltas) / len(deltas),
        "tie_rate": sum(value == 0 for value in deltas) / len(deltas),
    }
def public(item: Mapping[str, Any], role: str, validations: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    result = {
        "role": role, "artifact_tree_sha256": item["tree"], "transport": item["transport"],
        "receipt_path": item["path"], "receipt_sha256": item["sha256"],
        "raw_vector_sha256": item["vector_sha256"], "external_scalar": item["scalar"],
    }
    if validations is not None:
        result["validation_receipt"] = validations[item["sha256"]]
    return result
def select(
    control_inventory: Path,
    full_inventory: Path,
    controls: Sequence[Path],
    candidates: Sequence[Path],
    control_validations: Sequence[Path],
    candidate_validations: Sequence[Path],
) -> dict[str, Any]:
    need(len(controls) == len(candidates) == 4, "exactly four receipts per arm are required")
    ci, fi = inventory(control_inventory, "lora"), inventory(full_inventory, "full")
    need(ci["authority"] == fi["authority"], "arm runtime authorities differ")
    cs = [score(path, "selection", "control") for path in controls]
    fs = [score(path, "selection", "candidate") for path in candidates]
    cv, fv = validation_set(control_validations, cs), validation_set(candidate_validations, fs)
    need(set(ci["trees"]) == {item["tree"] for item in cs}, "control receipts do not match inventory")
    need(set(fi["trees"]) == {item["tree"] for item in fs}, "candidate receipts do not match inventory")
    fingerprint, base = shared([*cs, *fs], ci["authority"])
    control = min(cs, key=lambda item: (item["scalar"], item["tree"]))
    candidate = min(fs, key=lambda item: (item["scalar"], item["tree"]))
    wins = candidate["scalar"] < control["scalar"]
    authorized = [
        {"role": role, "tree_sha256": item["tree"], "transport": item["transport"]}
        for role, item in (("control", control), ("candidate", candidate))
    ] if wins else []
    legacy_bindings = {
        "evaluator_image": IMAGE, "score_driver_sha256": DRIVER_SHA256,
        "base_identity_sha256": base,
        "source_tree_sha256": ci["authority"]["runtime_source_inventory_sha256"],
        "experiment_config_sha256": ci["authority"]["experiment_config_sha256"],
        "training_image_id": ci["authority"]["training_image_id"],
    }
    return {
        "schema_version": DECISION_SCHEMA, "kind": "bloomz_paired_external_dev_decision",
        "status": AUTHORIZED if wins else STOP, "local_scope": LOCAL_SCOPE,
        "official_calibration_claimed": False,
        "selection": {
            "fixture": {**DEV, "eval_set_fingerprint": fingerprint},
            "bindings": legacy_bindings, "authority": ci["authority"],
            "inventories": {
                "control": {"path": ci["path"], "sha256": ci["sha256"], "gpu_admission": ci["gpu_admission"]},
                "candidate": {"path": fi["path"], "sha256": fi["sha256"], "gpu_admission": fi["gpu_admission"]},
            },
            "validation_receipts": {
                "control": [{"score_receipt_sha256": key, **cv[key]} for key in sorted(cv)],
                "candidate": [{"score_receipt_sha256": key, **fv[key]} for key in sorted(fv)],
            },
            "control": public(control, "control", cv),
            "candidate": public(candidate, "candidate", fv),
            "paired": paired(control["vector"], candidate["vector"]), "candidate_strictly_lower": wins,
        },
        "confirmation": {**CONFIRMATION, "authorized_artifacts": authorized},
    }
def load_authorization(path: Path) -> tuple[Path, Mapping[str, Any], str]:
    resolved, data, digest = read_json(path, "dev authorization", 2_000_000)
    need(isinstance(data, Mapping) and data.get("schema_version") == DECISION_SCHEMA, "authorization schema drift")
    need(data.get("kind") == "bloomz_paired_external_dev_decision" and data.get("status") == AUTHORIZED, "confirmation is not authorized")
    need(data.get("local_scope") == LOCAL_SCOPE, "authorization scope drift")
    confirmation = data.get("confirmation")
    need(isinstance(confirmation, Mapping) and all(confirmation.get(k) == v for k, v in CONFIRMATION.items()), "confirmation fixture authorization drift")
    return resolved, data, digest
def confirm(authorization: Path, control_path: Path, candidate_path: Path) -> dict[str, Any]:
    auth_path, auth, auth_sha = load_authorization(authorization)
    cs, fs = score(control_path, "confirmation", "control"), score(candidate_path, "confirmation", "candidate")
    entries = auth["confirmation"].get("authorized_artifacts")
    need(isinstance(entries, list) and len(entries) == 2, "authorization does not bind two artifacts")
    for role, item, expected in zip(("control", "candidate"), (cs, fs), entries, strict=True):
        need(expected == {"role": role, "tree_sha256": item["tree"], "transport": item["transport"]}, f"unauthorized {role} artifact")
        binding = item["data"].get("authorization")
        need(isinstance(binding, Mapping), "confirmation receipt lacks authorization binding")
        need(binding.get("path") == str(auth_path) and binding.get("sha256") == auth_sha and binding.get("role") == role, "confirmation receipt authorization mismatch")
    selection = auth.get("selection")
    need(isinstance(selection, Mapping), "authorization selection is absent")
    bound_authority = authority(selection.get("authority"))
    inventory_facts = selection.get("inventories")
    need(isinstance(inventory_facts, Mapping), "authorization inventories are absent")
    for role, strategy in (("control", "lora"), ("candidate", "full")):
        expected = inventory_facts.get(role)
        need(isinstance(expected, Mapping), f"authorization {role} inventory is absent")
        current = inventory(Path(expected.get("path", "")), strategy)
        need(expected == {"path": current["path"], "sha256": current["sha256"], "gpu_admission": current["gpu_admission"]}, f"authorization {role} inventory changed")
        need(current["authority"] == bound_authority, f"authorization {role} authority changed")
    fingerprint, _ = shared([cs, fs], bound_authority)
    wins = fs["scalar"] < cs["scalar"]
    return {
        "schema_version": VERDICT_SCHEMA, "kind": "bloomz_paired_local_confirmation_verdict",
        "status": "LOCAL_PAIRED_CANDIDATE_WIN" if wins else STOP,
        "local_scope": LOCAL_SCOPE, "official_calibration_claimed": False,
        "authorization": {"path": str(auth_path), "sha256": auth_sha},
        "dev_evidence": inventory_facts,
        "confirmation": {
            **CONFIRMATION, "eval_set_fingerprint": fingerprint,
            "control": public(cs, "control"), "candidate": public(fs, "candidate"),
            "paired": paired(cs["vector"], fs["vector"]), "candidate_strictly_lower": wins,
        },
        "field_targets": {"scope": "future_official_or_nonlocal_only", "observed_here": False},
    }
def write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    output = path.expanduser().resolve(strict=False)
    need(not os.path.lexists(output), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value, newline=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    dev = modes.add_parser("select")
    dev.add_argument("--control-inventory", required=True, type=Path)
    dev.add_argument("--candidate-inventory", required=True, type=Path)
    dev.add_argument("--control-receipt", required=True, action="append", type=Path)
    dev.add_argument("--candidate-receipt", required=True, action="append", type=Path)
    dev.add_argument("--control-validation", required=True, action="append", type=Path)
    dev.add_argument("--candidate-validation", required=True, action="append", type=Path)
    dev.add_argument("--output", required=True, type=Path)
    held = modes.add_parser("confirm")
    held.add_argument("--authorization", required=True, type=Path)
    held.add_argument("--control-receipt", required=True, type=Path)
    held.add_argument("--candidate-receipt", required=True, type=Path)
    held.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = (select(args.control_inventory, args.candidate_inventory, args.control_receipt, args.candidate_receipt,
                    args.control_validation, args.candidate_validation)
             if args.mode == "select" else confirm(args.authorization, args.control_receipt, args.candidate_receipt))
    digest = write_exclusive(args.output, value)
    print(f"BLOOMZ_DECISION={value['status']} output={args.output.resolve()} sha256={digest}")
    return 0 if value["status"] in {AUTHORIZED, "LOCAL_PAIRED_CANDIDATE_WIN"} else 3

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOOMZ_DECISION=FAIL: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
