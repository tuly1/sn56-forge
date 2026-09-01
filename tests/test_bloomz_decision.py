from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / "experiments/20260831-bloomz-memory-fullft-v1/decide_external.py"
SPEC = importlib.util.spec_from_file_location("bloomz_decision", SCRIPT)
assert SPEC and SPEC.loader
d = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d
SPEC.loader.exec_module(d)

AUTHORITY = {
    "runtime_authority_sha256": "1" * 64,
    "source_commit": "2" * 40,
    "source_tree": "3" * 40,
    "forge_child_tree": "4" * 40,
    "experiment_child_tree": "5" * 40,
    "runtime_source_inventory_sha256": "6" * 64,
    "training_image_reference": "trainer@sha256:" + "7" * 64,
    "training_image_id": "sha256:" + "8" * 64,
    "experiment_config_sha256": "9" * 64,
    "provider_start_epoch": 4_102_444_800,
    "science_start_deadline_epoch": 4_102_446_000,
    "science_started_epoch": 4_102_445_400,
    "decision_deadline_epoch": 4_102_469_400,
    "provider_deadline_epoch": 4_102_475_400,
    "lease_budget_sha256": "b" * 64,
}
BASE = "a" * 64


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(d.canonical(value, newline=True))


def inventory(tmp: Path, strategy: str, trees: list[str], **changes) -> Path:
    probe = tmp / f"{strategy}-gpu-probe.json"
    admission = {
        "receipt_path": str(probe.resolve()), "receipt_sha256": "",
        "status": "PASS",
        "proof": "actual_b1_s2048_forward_backward_fused_adamw_step_plus_optimizer_resident_b1_s4096_labeled_eval",
        "arm": "control" if strategy == "lora" else "full", "strategy": strategy,
        "learning_rate": 1.5e-4 if strategy == "lora" else 1e-4,
        "geometry": {"microbatch": 1, "gradient_accumulation_configured": 16,
            "optimizer_probe_microsteps_executed": 1, "configured_effective_batch": 16,
            "sequence_length": 2048, "packing": False, "gradient_checkpointing": True},
        "model": {"repo": "bigscience/bloomz-560m",
            "revision": "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05",
            "config_sha256": "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb",
            "weights_sha256": "365b2c5e9bd1057eb1e3f1a4fc3f89ae6584d20f24b682d2406bc7e90178ec13",
            "output_width": 250880, "params_b": 0.559214592},
        "gpu": {"name": "NVIDIA H100 80GB HBM3", "card_bytes": 80 * 1024**3,
            "train_peak_reserved_bytes": 30 * 1024**3, "train_peak_reserved_ratio": 0.375,
            "selection_peak_reserved_bytes": 35 * 1024**3, "selection_peak_reserved_ratio": 0.4375,
            "max_reserved_ratio": 0.70},
        "authority": AUTHORITY,
    }
    dump(probe, {key: value for key, value in admission.items() if key not in {"receipt_path", "receipt_sha256"}})
    admission["receipt_sha256"] = d.sha_file(probe)
    value = {
        "status": "EXTERNAL_SCORE_READY",
        "strategy": strategy,
        "phase": "control" if strategy == "lora" else "candidate",
        "schedule_completed": True,
        "planned_steps": 256,
        "final_step": 256,
        "required_external_scores": 4,
        "completed_scheduled_evals": 4,
        "capture_errors": [],
        "authority": AUTHORITY,
        "gpu_admission": admission,
        "checkpoints": [{"tree_sha256": tree} for tree in trees],
        **changes,
    }
    path = tmp / f"{strategy}-inventory.json"
    dump(path, value)
    return path


def receipt(
    tmp: Path,
    name: str,
    role: str,
    loss: float,
    phase: str,
    *,
    tree: str | None = None,
    artifact_files: list[dict] | None = None,
    fingerprint: str = "f" * 32,
    auth: dict | None = None,
    bound_authority: dict = AUTHORITY,
) -> Path:
    folder = tmp / name
    folder.mkdir(parents=True)
    transport = "peft_adapter" if role == "control" else "full_model"
    count = d.DEV["vector_count"] if phase == "selection" else d.CONFIRMATION["expected_vector_count"]
    vector = [loss + (index % 3) * 0.001 for index in range(count)]
    raw = {"/artifact": {
        "eval_loss": loss,
        "per_example_losses": vector,
        "eval_set_fingerprint": fingerprint,
        "sn56_local_artifact_transport": transport,
        "is_finetune": True,
    }}
    raw_path = folder / "raw-result.json"
    dump(raw_path, raw)
    fixture = d.DEV if phase == "selection" else {
        "sha256": d.CONFIRMATION["sha256"], "row_count": d.CONFIRMATION["row_count"]
    }
    files = artifact_files or [{"path": name, "bytes": len(name), "sha256": d.sha_value(name)}]
    value = {
        "schema_version": d.SCORE_SCHEMA,
        "kind": "bloomz_digest_pinned_external_score",
        "status": "PASS",
        "phase": phase,
        "artifact": {
            "role": role,
            "expected_transport": transport,
            "files": files,
            "tree_sha256": tree or d.sha_value(files),
        },
        "base": {"identity_sha256": BASE},
        "fixture": {"phase": phase, "sha256": fixture["sha256"], "row_count": fixture["row_count"]},
        "authority": bound_authority,
        "evaluator": {
            "image": d.IMAGE,
            "network": "none",
            "score_driver": {"sha256": d.DRIVER_SHA256},
        },
        "result": {
            "raw_result_filename": "raw-result.json",
            "raw_result_sha256": d.sha_file(raw_path),
            "eval_loss": loss,
            "eval_set_fingerprint": fingerprint,
            "transport": transport,
            "vector_count": count,
            "vector_sha256": d.sha_value(vector),
            "vector_order": "evaluator_emission_order",
            "ordered_vector_sha256": d.sha_value(
                {"order": "evaluator_emission_order", "values": vector}
            ),
        },
    }
    if auth:
        value["authorization"] = auth
    path = folder / "receipt.json"
    dump(path, value)
    dump(folder / "receipt.sha256.json", {"filename": "receipt.json", "sha256": d.sha_file(path)})
    return path
def validations(tmp: Path, label: str, paths: list[Path], role: str) -> list[Path]:
    outputs = []
    for index, path in enumerate(paths):
        scored = d.score(path, "selection", role)
        output = tmp / f"{label}-validation-{index}.json"
        dump(output, {
            "schema_version": "sn56.bloomz-artifact-validation.v1", "status": "PASS",
            "offline": True, "trust_remote_code": False,
            "artifact_format": scored["transport"], "artifact_tree_sha256": scored["tree"],
            "authority": AUTHORITY,
            "fresh_exact_evaluator_load": {"receipt_sha256": scored["sha256"]},
            "serialized_state_summary": {"all_tensors_finite": True, "tensor_count": 123,
                "total_numel": 1000, "total_bytes": 2000, "finite_scan_chunk_elements": 100,
                "dtype_tensor_counts": {"torch.float32": 123},
                "tensor_files": [{"sha256": d.sha_value({"weights": scored["tree"]})}],
                "tensor_schema_sha256": d.sha_value({"artifact": scored["tree"]})},
        })
        outputs.append(output)
    return outputs


def setup_selection(tmp: Path, candidate_losses=(1.2, 1.0, 1.3, 1.4)):
    controls = [receipt(tmp, f"c{i}", "control", loss, "selection") for i, loss in enumerate((1.3, 1.1, 1.1, 1.4))]
    candidates = [receipt(tmp, f"f{i}", "candidate", loss, "selection") for i, loss in enumerate(candidate_losses)]
    ctrees = [d.score(path, "selection", "control")["tree"] for path in controls]
    ftrees = [d.score(path, "selection", "candidate")["tree"] for path in candidates]
    return (inventory(tmp, "lora", ctrees), inventory(tmp, "full", ftrees), controls, candidates,
            validations(tmp, "control", controls, "control"), validations(tmp, "candidate", candidates, "candidate"))


def test_select_is_inventory_exact_deterministic_and_confirmation_blind(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path)
    result = d.select(ci, fi, controls, candidates, cv, fv)
    assert result["status"] == d.AUTHORIZED
    assert result["selection"]["authority"] == AUTHORITY
    assert result["confirmation"]["expected_vector_count"] == 511
    tied = sorted(d.score(path, "selection", "control")["tree"] for path in controls[1:3])
    assert result["selection"]["control"]["artifact_tree_sha256"] == tied[0]
    assert result["selection"]["paired"]["pair_count"] == 1021
    assert result["selection"]["paired"]["bootstrap"] == {
        "seed": 20260808,
        "resamples": 10_000,
        "confidence": 0.99,
        "one_sided_tail": pytest.approx(0.01),
        "lower_bound_sorted_index": 99,
    }
    assert result["selection"]["paired"]["win_rate_lower_bound"] == 1.0
    assert result["selection"]["paired"]["mean_gap_lower_bound"] == pytest.approx(0.1)
    assert result["selection"]["paired"]["passed"] is True
    args = d.parse_args([
        "select", "--control-inventory", str(ci), "--candidate-inventory", str(fi),
        *sum((["--control-receipt", str(path)] for path in controls), []),
        *sum((["--candidate-receipt", str(path)] for path in candidates), []),
        *sum((["--control-validation", str(path)] for path in cv), []),
        *sum((["--candidate-validation", str(path)] for path in fv), []),
        "--output", str(tmp_path / "decision.json"),
    ])
    assert not hasattr(args, "confirmation")


def test_loser_and_invalid_inventory_do_not_authorize(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path, (1.4, 1.2, 1.3, 1.25))
    result = d.select(ci, fi, controls, candidates, cv, fv)
    assert result["status"] == d.STOP
    assert result["confirmation"]["authorized_artifacts"] == []
    broken = json.loads(ci.read_text())
    broken["schedule_completed"] = False
    dump(ci, broken)
    with pytest.raises(d.DecisionError, match="incomplete"):
        d.select(ci, fi, controls, candidates, cv, fv)


def test_scalar_win_without_owner_confidence_gate_stops(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(
        tmp_path, (1.095, 1.2, 1.3, 1.4)
    )
    result = d.select(ci, fi, controls, candidates, cv, fv)
    assert result["selection"]["candidate"]["external_scalar"] < result["selection"]["control"]["external_scalar"]
    assert result["selection"]["paired"]["mean_gap"] == pytest.approx(0.005)
    assert result["selection"]["paired"]["passed"] is False
    assert result["status"] == d.STOP
    assert result["confirmation"]["authorized_artifacts"] == []


def test_owner_paired_bootstrap_gate_synthetic_cases() -> None:
    control = [0.0] * 40 + [10.0] * 60
    winning = [-0.1] * 40 + [9.9] * 60
    gate = d.paired(control, winning)
    assert gate["passed"] is True
    assert gate["candidate_win_rate"] == 1.0
    assert gate["mean_gap"] == pytest.approx(0.1)
    assert gate["mean_gap_threshold"] == pytest.approx(0.06)

    null = d.paired([1.0] * 100, [1.0] * 100)
    assert null["passed"] is False
    assert null["mean_gap_lower_bound"] == 0.0
    assert null["win_rate_lower_bound"] == 0.0

    regression = d.paired([1.0] * 100, [1.2] * 100)
    assert regression["passed"] is False
    assert regression["mean_gap_lower_bound"] == pytest.approx(-0.2)

    reordered = d.paired(control, list(reversed(winning)))
    assert reordered["passed"] is False
    assert reordered["candidate_win_rate"] == 0.6
    assert reordered["win_rate_lower_bound"] < 0.55
    assert gate["bindings"]["ordered_pair_sha256"] != reordered["bindings"]["ordered_pair_sha256"]

    with pytest.raises(d.DecisionError, match="not pairable"):
        d.paired([1.0, 2.0], [1.0])


def test_receipts_must_match_inventory_raw_bytes_and_authority(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path)
    with pytest.raises(d.DecisionError, match="exactly four"):
        d.select(ci, fi, controls[:3], candidates, cv, fv)
    bad_validation = json.loads(cv[0].read_text())
    bad_validation["serialized_state_summary"]["all_tensors_finite"] = False
    dump(cv[0], bad_validation)
    with pytest.raises(d.DecisionError, match="not finite"):
        d.select(ci, fi, controls, candidates, cv, fv)
    bad_validation["serialized_state_summary"]["all_tensors_finite"] = True
    dump(cv[0], bad_validation)
    raw_path = candidates[0].with_name("raw-result.json")
    raw = json.loads(raw_path.read_text())
    raw["/artifact"]["per_example_losses"][0] += 1
    dump(raw_path, raw)
    with pytest.raises(d.DecisionError, match="raw-result hash"):
        d.select(ci, fi, controls, candidates, cv, fv)

    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path / "reorder")
    raw_path = candidates[0].with_name("raw-result.json")
    raw = json.loads(raw_path.read_text())
    raw["/artifact"]["per_example_losses"].reverse()
    dump(raw_path, raw)
    with pytest.raises(d.DecisionError, match="raw-result hash"):
        d.select(ci, fi, controls, candidates, cv, fv)

    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path / "authority")
    changed = dict(AUTHORITY)
    changed["source_commit"] = "0" * 40
    value = json.loads(candidates[0].read_text())
    value["authority"] = changed
    dump(candidates[0], value)
    dump(candidates[0].with_name("receipt.sha256.json"), {
        "filename": "receipt.json", "sha256": d.sha_file(candidates[0])
    })
    with pytest.raises(d.DecisionError, match="validation does not map|authority mismatch"):
        d.select(ci, fi, controls, candidates, cv, fv)


def test_authorized_confirmation_is_exact_and_paired_only(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path)
    authorization = d.select(ci, fi, controls, candidates, cv, fv)
    auth_path = tmp_path / "dev-decision.json"
    d.write_exclusive(auth_path, authorization)
    auth_sha = d.sha_file(auth_path)
    entries = authorization["confirmation"]["authorized_artifacts"]
    common = {"path": str(auth_path.resolve()), "sha256": auth_sha}
    selected_control = json.loads(Path(authorization["selection"]["control"]["receipt_path"]).read_text())
    selected_candidate = json.loads(Path(authorization["selection"]["candidate"]["receipt_path"]).read_text())
    control = receipt(tmp_path, "held-c", "control", 1.2, "confirmation", tree=entries[0]["tree_sha256"], artifact_files=selected_control["artifact"]["files"], auth={**common, "role": "control"})
    candidate = receipt(tmp_path, "held-f", "candidate", 1.1, "confirmation", tree=entries[1]["tree_sha256"], artifact_files=selected_candidate["artifact"]["files"], auth={**common, "role": "candidate"})
    verdict = d.confirm(auth_path, control, candidate)
    assert verdict["status"] == "LOCAL_PAIRED_CANDIDATE_WIN"
    assert verdict["confirmation"]["paired"]["pair_count"] == 511
    assert verdict["official_calibration_claimed"] is False
    assert verdict["field_targets"]["observed_here"] is False
    null_candidate = receipt(tmp_path, "held-null", "candidate", 1.195, "confirmation", tree=entries[1]["tree_sha256"], artifact_files=selected_candidate["artifact"]["files"], auth={**common, "role": "candidate"})
    null_verdict = d.confirm(auth_path, control, null_candidate)
    assert null_verdict["confirmation"]["paired"]["passed"] is False
    assert null_verdict["status"] == d.STOP
    value = json.loads(candidate.read_text())
    value["authorization"]["sha256"] = "c" * 64
    dump(candidate, value)
    dump(candidate.with_name("receipt.sha256.json"), {"filename": "receipt.json", "sha256": d.sha_file(candidate)})
    with pytest.raises(d.DecisionError, match="authorization mismatch"):
        d.confirm(auth_path, control, candidate)


def test_final_science_receipt_is_authority_bound_decision_only(tmp_path: Path) -> None:
    ci, fi, controls, candidates, cv, fv = setup_selection(tmp_path)
    authorization = d.select(ci, fi, controls, candidates, cv, fv)
    auth_path = tmp_path / "dev-decision.json"
    d.write_exclusive(auth_path, authorization)
    auth_sha = d.sha_file(auth_path)
    entries = authorization["confirmation"]["authorized_artifacts"]
    common = {"path": str(auth_path.resolve()), "sha256": auth_sha}
    selected_control = json.loads(Path(authorization["selection"]["control"]["receipt_path"]).read_text())
    selected_candidate = json.loads(Path(authorization["selection"]["candidate"]["receipt_path"]).read_text())
    control = receipt(tmp_path, "final-c", "control", 1.2, "confirmation", tree=entries[0]["tree_sha256"], artifact_files=selected_control["artifact"]["files"], auth={**common, "role": "control"})
    candidate = receipt(tmp_path, "final-f", "candidate", 1.1, "confirmation", tree=entries[1]["tree_sha256"], artifact_files=selected_candidate["artifact"]["files"], auth={**common, "role": "candidate"})
    verdict = d.confirm(auth_path, control, candidate)
    verdict_path = tmp_path / "confirmation-verdict.json"
    d.write_exclusive(verdict_path, verdict)
    completed = d.complete(
        verdict_path,
        completed_epoch=AUTHORITY["science_started_epoch"] + 100,
    )
    assert completed["status"] == "DECISION_COMPLETE"
    assert completed["authority"] == AUTHORITY
    assert set(completed) == {
        "schema_version", "kind", "status", "local_scope",
        "completed_epoch", "authority", "decision",
    }
    with pytest.raises(d.DecisionError, match="outside decision deadline"):
        d.complete(
            verdict_path,
            completed_epoch=AUTHORITY["decision_deadline_epoch"] + 1,
        )


def test_output_creation_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    d.write_exclusive(output, {"status": "first"})
    with pytest.raises(d.DecisionError, match="overwrite"):
        d.write_exclusive(output, {"status": "second"})
