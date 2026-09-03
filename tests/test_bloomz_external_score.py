from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "20260831-bloomz-memory-fullft-v1"
    / "score_external.py"
)
SPEC = importlib.util.spec_from_file_location("bloomz_external_score", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
score = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score
SPEC.loader.exec_module(score)

DRIVER = Path(
    "/Users/atulyashetty/Test/SN56-text/lanes/01-instruct-chat/runtime-smoke/"
    "local_artifact_score_driver.py"
)
FINGERPRINT = "1" * 32


def _authority(runtime_sha: str) -> dict[str, str]:
    return {
        "runtime_authority_sha256": runtime_sha,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "forge_child_tree": "3" * 40,
        "experiment_child_tree": "4" * 40,
        "runtime_source_inventory_sha256": "5" * 64,
        "training_image_reference": "image@sha256:" + "6" * 64,
        "training_image_id": "sha256:" + "7" * 64,
        "experiment_config_sha256": "8" * 64,
        "provider_start_epoch": 4_102_444_800,
        "science_start_deadline_epoch": 4_102_447_500,
        "science_started_epoch": 4_102_445_400,
        "decision_deadline_epoch": 4_102_469_400,
        "provider_deadline_epoch": 4_102_476_900,
        "lease_budget_sha256": "9" * 64,
    }


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _score_only_repositories(
    tmp_path: Path, monkeypatch
) -> tuple[Path, Path, dict[str, str]]:
    science = tmp_path / "science"
    science.mkdir(parents=True)
    _git(science, "init", "-q")
    _git(science, "config", "user.email", "science@example.invalid")
    _git(science, "config", "user.name", "Science Test")
    for relative in (*score.SCORE_ONLY_CHANGED_PATHS, "unrelated.txt"):
        path = science / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"science:{relative}\n", encoding="utf-8")
    _git(science, "add", ".")
    _git(science, "commit", "-qm", "science")
    scorer = tmp_path / "scorer"
    subprocess.run(["/usr/bin/git", "clone", "-q", str(science), str(scorer)], check=True)
    _git(scorer, "config", "user.email", "score@example.invalid")
    _git(scorer, "config", "user.name", "Score Test")
    for relative in score.SCORE_ONLY_CHANGED_PATHS:
        (scorer / relative).write_text(f"suffix-fix:{relative}\n", encoding="utf-8")
    _git(scorer, "add", ".")
    _git(scorer, "commit", "-qm", "suffix fix")
    intermediate = _git(scorer, "rev-parse", "HEAD")
    intermediate_tree = _git(scorer, "rev-parse", "HEAD^{tree}")
    for relative in score.SCORE_ONLY_CHANGED_PATHS:
        (scorer / relative).write_text(f"authority-bridge:{relative}\n", encoding="utf-8")
    _git(scorer, "add", ".")
    _git(scorer, "commit", "-qm", "authority bridge")
    science_commit = _git(science, "rev-parse", "HEAD")
    science_tree = _git(science, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(score, "REPO_ROOT", scorer)
    monkeypatch.setattr(score, "SCIENCE_SOURCE_COMMIT", science_commit)
    monkeypatch.setattr(score, "TARGET_SERIALIZATION_FIX_COMMIT", intermediate)
    monkeypatch.setattr(score, "TARGET_SERIALIZATION_FIX_TREE", intermediate_tree)
    binding = _authority("a" * 64)
    binding["source_commit"] = science_commit
    binding["source_tree"] = science_tree
    return science, scorer, binding


def _bound_inventory(
    tmp_path: Path, role: str, binding: dict[str, str]
) -> tuple[Path, list[str]]:
    trees = [f"{index:x}" * 64 for index in range(1 if role == "control" else 5, 5 if role == "control" else 9)]
    value = {
        "status": "EXTERNAL_SCORE_READY",
        "strategy": "lora" if role == "control" else "full",
        "phase": role,
        "schedule_completed": True,
        "planned_steps": 256,
        "final_step": 256,
        "authority": binding,
        "checkpoints": [{"tree_sha256": tree} for tree in trees],
    }
    path = tmp_path / f"{role}-inventory.json"
    path.write_bytes(score.canonical_bytes(value, newline=True))
    return path, trees


def _score_only_authority(
    tmp_path: Path,
    science: Path,
    binding: dict[str, str],
) -> tuple[Path, dict, list[str], list[str]]:
    control_path, control_trees = _bound_inventory(tmp_path, "control", binding)
    candidate_path, candidate_trees = _bound_inventory(tmp_path, "candidate", binding)
    document = {
        "schema_version": score.SCORE_ONLY_SCHEMA,
        "kind": "bloomz_score_only_successor_authority",
        "status": "AUTHORIZED",
        "scope": score.SCORE_ONLY_SCOPE,
        "science_authority": binding,
        "scorer": score.inspect_score_only_successor(science, binding),
        "inventories": {
            "control": {
                "path": str(control_path.resolve()),
                "sha256": score.file_sha256(control_path),
                "checkpoint_trees": control_trees,
            },
            "candidate": {
                "path": str(candidate_path.resolve()),
                "sha256": score.file_sha256(candidate_path),
                "checkpoint_trees": candidate_trees,
            },
        },
        "evaluation": {
            "evaluator_image": score.IMAGE,
            "score_driver_sha256": score.DRIVER_SHA256,
            "base": score.BASE,
            "selection_fixture_sha256": score.FIXTURES["selection"]["sha256"],
            "confirmation_fixture_sha256": score.FIXTURES["confirmation"]["sha256"],
            "decision_deadline_epoch": binding["decision_deadline_epoch"],
        },
        "approvals": {
            "owner_review_sha256": "a" * 64,
            "independent_audit_sha256": "b" * 64,
        },
    }
    path = tmp_path / "score-only-authority.json"
    path.write_bytes(score.canonical_bytes(document, newline=True))
    return path, document, control_trees, candidate_trees


def _inputs(tmp_path: Path) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weights")
    artifact, artifact_files, artifact_tree = score.inventory(artifact, "artifact")
    base = tmp_path / "base"
    base.mkdir()
    base, base_files, base_tree = score.inventory(base, "base")
    fixture = tmp_path / "dev.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")
    runtime_sha = score.file_sha256(runtime)
    return score.Inputs(
        phase="selection",
        role="control",
        transport="full_model",
        artifact=artifact,
        artifact_files=artifact_files,
        artifact_tree=artifact_tree,
        artifact_metadata={"transport": "full_model"},
        adapter_alias=None,
        base=base,
        base_files=base_files,
        base_tree=base_tree,
        fixture=fixture,
        fixture_facts={
            "phase": "selection",
            "row_count": 2,
            "prepared_row_count": 2,
            "sha256": score.file_sha256(fixture),
        },
        driver=DRIVER,
        runtime_authority=runtime,
        science_source_root=score.REPO_ROOT,
        authority_binding=_authority(runtime_sha),
        score_only_authority=None,
        score_only_authority_sha256=None,
        score_only_binding=None,
        expected_fingerprint=None,
        fingerprint_anchor=True,
        authorization=None,
        authorization_sha256=None,
        authorization_facts=None,
        output=tmp_path / "score-output",
        gpu="0",
        docker="/usr/bin/docker",
        timeout=123,
    )


def test_score_only_authority_binds_science_scorer_and_all_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    science, _, binding = _score_only_repositories(tmp_path, monkeypatch)
    path, document, control_trees, candidate_trees = _score_only_authority(
        tmp_path, science, binding
    )
    resolved, digest, projected = score.load_score_only_authority(
        path,
        score.file_sha256(path),
        science,
        binding,
        role="control",
        artifact_tree=control_trees[0],
        transport="peft_adapter",
    )
    assert resolved == path.resolve()
    assert digest == score.file_sha256(path)
    assert projected["authority"] == document
    assert projected["authority"]["science_authority"] == binding
    assert projected["authority"]["scorer"]["scorer_parent_commit"] == score.TARGET_SERIALIZATION_FIX_COMMIT
    score.load_score_only_authority(
        path,
        digest,
        science,
        binding,
        role="candidate",
        artifact_tree=candidate_trees[-1],
        transport="full_model",
    )
    with pytest.raises(score.ScoreError, match="checkpoint set"):
        score.load_score_only_authority(
            path,
            digest,
            science,
            binding,
            role="control",
            artifact_tree="f" * 64,
            transport="peft_adapter",
        )
    with pytest.raises(score.ScoreError, match="file SHA"):
        score.load_score_only_authority(
            path,
            "0" * 64,
            science,
            binding,
            role="control",
            artifact_tree=control_trees[0],
            transport="peft_adapter",
        )


@pytest.mark.parametrize("mutation", ["delete", "add", "mode", "dirty", "third_commit"])
def test_score_only_chain_rejects_nonminimal_or_mutable_source(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    science, scorer, binding = _score_only_repositories(tmp_path, monkeypatch)
    if mutation == "delete":
        (scorer / "unrelated.txt").unlink()
        _git(scorer, "add", "-A")
        _git(scorer, "commit", "--amend", "--no-edit", "-q")
    elif mutation == "add":
        (scorer / "extra.txt").write_text("extra\n", encoding="utf-8")
        _git(scorer, "add", ".")
        _git(scorer, "commit", "--amend", "--no-edit", "-q")
    elif mutation == "mode":
        changed = scorer / score.SCORE_ONLY_CHANGED_PATHS[0]
        os.chmod(changed, 0o755)
        _git(scorer, "add", str(changed))
        _git(scorer, "commit", "--amend", "--no-edit", "-q")
    elif mutation == "dirty":
        (scorer / score.SCORE_ONLY_CHANGED_PATHS[0]).write_text("dirty\n", encoding="utf-8")
    else:
        (scorer / score.SCORE_ONLY_CHANGED_PATHS[0]).write_text("third\n", encoding="utf-8")
        _git(scorer, "add", ".")
        _git(scorer, "commit", "-qm", "third scorer commit")
    with pytest.raises(score.ScoreError):
        score.inspect_score_only_successor(science, binding)


def _raw(**updates) -> dict:
    result = {
        "eval_loss": 1.25,
        "per_example_losses": [1.0, 1.5],
        "eval_set_fingerprint": FINGERPRINT,
        "is_finetune": True,
        "sn56_local_artifact_transport": "full_model",
    }
    result.update(updates)
    return {"/artifact": result}


def _lora_config() -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "peft_version": "0.19.1",
        "inference_mode": True,
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "use_qalora": False,
        "lora_bias": False,
        "init_lora_weights": True,
        "auto_mapping": None,
        "exclude_modules": None,
        "modules_to_save": None,
        "revision": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "layers_to_transform": None,
        "layers_pattern": None,
        "layer_replication": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "trainable_token_indices": None,
        "target_parameters": None,
        "loftq_config": {},
        "eva_config": None,
        "corda_config": None,
        "lora_ga_config": None,
        "alora_invocation_tokens": None,
        "qalora_group_size": 16,
        "use_bdlora": None,
        "arrow_config": None,
        "ensure_weight_tying": False,
        "base_model_name_or_path": "/cache/models/bigscience--bloomz-560m",
        # PEFT 0.19.1 serializes all-linear BLOOM targets as these four
        # suffixes; the adapter tensor keys carry the 24 layer prefixes.
        "target_modules": [
            "dense",
            "dense_4h_to_h",
            "dense_h_to_4h",
            "query_key_value",
        ],
    }


def _decision(inputs) -> dict:
    return {
        "schema_version": score.DECISION_SCHEMA,
        "kind": "bloomz_paired_external_dev_decision",
        "status": "AUTHORIZED_FOR_ONE_CONFIRMATION",
        "local_scope": score.LOCAL_SCOPE,
        "selection": {
            "fixture": {
                "sha256": score.FIXTURES["selection"]["sha256"],
                "row_count": 1024,
                "vector_count": 1021,
                "eval_set_fingerprint": FINGERPRINT,
            },
            "bindings": {
                "evaluator_image": score.IMAGE,
                "score_driver_sha256": score.DRIVER_SHA256,
                "base_identity_sha256": inputs.base_tree,
            },
            "authority": inputs.authority_binding,
        },
        "confirmation": {
            "filename": "confirmation.jsonl",
            "sha256": score.FIXTURES["confirmation"]["sha256"],
            "row_count": 512,
            "expected_vector_count": 511,
            "authorized_artifacts": [
                {
                    "role": "control",
                    "tree_sha256": inputs.artifact_tree,
                    "transport": inputs.transport,
                },
                {
                    "role": "candidate",
                    "tree_sha256": "9" * 64,
                    "transport": "full_model",
                },
            ],
        },
    }


def _stable_runtime(monkeypatch, inputs) -> None:
    monkeypatch.setattr(
        score,
        "load_runtime",
        lambda path, **_: (path, inputs.authority_binding),
    )


def test_argv_is_digest_pinned_and_adapter_gets_readonly_base_alias(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    argv = score.render_argv(inputs, tmp_path / "stage")
    assert argv[:8] == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
    ]
    assert argv[-3:] == [score.IMAGE, "-B", "/runner/local_artifact_score_driver.py"]
    assert argv[argv.index("--entrypoint") + 1] == score.ENTRYPOINT
    assert all("confirmation" not in value for value in argv)
    env = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
    assert "EMIT_PER_EXAMPLE_LOSSES=1" in env
    assert "TRANSFORMERS_OFFLINE=1" in env

    adapter = replace(
        inputs,
        transport="peft_adapter",
        adapter_alias="/cache/models/bigscience--bloomz-560m",
    )
    adapter_argv = score.render_argv(adapter, tmp_path / "other")
    mounts = [
        value for index, value in enumerate(adapter_argv) if index and adapter_argv[index - 1] == "--mount"
    ]
    assert any(
        "dst=/cache/models/bigscience--bloomz-560m,readonly" in value
        and f"src={inputs.base}" in value
        for value in mounts
    )


def test_selection_never_touches_confirmation_path(tmp_path: Path) -> None:
    dev = tmp_path / "dev.jsonl"
    absent = tmp_path / "confirmation-does-not-exist.jsonl"
    selected, facts = score.choose_fixture("selection", dev, absent)
    assert selected == dev
    assert facts == score.FIXTURES["selection"]
    selected, facts = score.choose_fixture("confirmation", dev, absent)
    assert selected == absent
    assert facts == score.FIXTURES["confirmation"]


def test_result_validation_fails_closed_on_vector_identity_and_transport(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = score.validate_result(_raw(), inputs)
    assert result["vector_count"] == 2
    assert result["vector_mean"] == 1.25

    broken = (
        (_raw(eval_loss=math.inf), "scalar"),
        (_raw(per_example_losses=[1.0, math.nan]), "vector"),
        (_raw(per_example_losses=[1.0]), "vector"),
        (_raw(eval_loss=9.0), "mean"),
        (_raw(eval_set_fingerprint="not-a-fingerprint"), "fingerprint"),
        (_raw(sn56_local_artifact_transport="peft_adapter"), "transport"),
    )
    for raw, message in broken:
        with pytest.raises(score.ScoreError, match=message):
            score.validate_result(raw, inputs)
    anchored = replace(inputs, expected_fingerprint="2" * 32, fingerprint_anchor=False)
    with pytest.raises(score.ScoreError, match="anchor"):
        score.validate_result(_raw(), anchored)


def test_full_and_lora_artifact_formats_are_strict(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"bloom"}\n', encoding="utf-8")
    full = tmp_path / "full"
    full.mkdir()
    (full / "config.json").write_bytes((base / "config.json").read_bytes())
    (full / "model.safetensors").write_bytes(b"weights")
    assert score.verify_artifact(full, base, "full_model")[3] is None
    (full / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(score.ScoreError, match="byte-identical"):
        score.verify_artifact(full, base, "full_model")
    (full / "optimizer.pt").write_bytes(b"pickle")
    with pytest.raises(score.ScoreError, match="pickle"):
        score.verify_artifact(full, base, "full_model")
    (full / "optimizer.pt").unlink()

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = _lora_config()
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    assert score.verify_artifact(adapter, base, "peft_adapter")[3] == config[
        "base_model_name_or_path"
    ]
    config["base_model_name_or_path"] = "bigscience/bloomz-560m"
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(score.ScoreError, match="normalized absolute"):
        score.verify_artifact(adapter, base, "peft_adapter")
    config = _lora_config()
    config["target_modules"] = config["target_modules"][:-1]
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(score.ScoreError, match="target modules"):
        score.verify_artifact(adapter, base, "peft_adapter")
    config = _lora_config()
    config["unexpected"] = None
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(score.ScoreError, match="shape"):
        score.verify_artifact(adapter, base, "peft_adapter")


def test_inventory_symlinks_and_output_overlap_fail_closed(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "weights").write_bytes(b"x")
    (tree / "link").symlink_to(tree / "weights")
    with pytest.raises(score.ScoreError, match="symlink"):
        score.inventory(tree, "artifact")
    with pytest.raises(score.ScoreError, match="overlaps"):
        score.disjoint(tree / "new-output", [tree])


@pytest.mark.parametrize("forbidden", ["added_tokens.json", "chat_template.jinja"])
def test_base_rejects_tokenizer_override_files(tmp_path: Path, monkeypatch, forbidden: str) -> None:
    files = [
        {"path": name, "bytes": size, "sha256": digest}
        for name, (size, digest) in score.BASE_FILES.items()
    ]
    files.append({"path": forbidden, "bytes": 1, "sha256": "0" * 64})
    monkeypatch.setattr(score, "inventory", lambda *_: (tmp_path, files, "tree"))
    with pytest.raises(score.ScoreError, match=forbidden):
        score.verify_base(tmp_path)


def test_confirmation_authorization_binds_fixture_artifact_and_runtime(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = tmp_path / "decision.json"
    decision = _decision(inputs)
    path.write_text(json.dumps(decision), encoding="utf-8")
    resolved, digest, facts = score.verify_authorization(
        path,
        role="control",
        artifact_tree=inputs.artifact_tree,
        transport=inputs.transport,
        base_tree=inputs.base_tree,
        authority_binding=inputs.authority_binding,
    )
    assert resolved == path.resolve()
    assert digest == score.file_sha256(path)
    assert facts["role"] == "control"

    for mutation, message in (
        (("status", "STOP_NO_SCIENCE"), "authorized"),
        (("authority", {}), "authority"),
        (("vector_count", 1024), "fixture"),
        (("tree_sha256", "0" * 64), "authorized"),
    ):
        broken = _decision(inputs)
        key, value = mutation
        if key == "authority":
            broken["selection"]["authority"] = value
        elif key == "vector_count":
            broken["selection"]["fixture"][key] = value
        elif key == "tree_sha256":
            broken["confirmation"]["authorized_artifacts"][0][key] = value
        else:
            broken[key] = value
        candidate = tmp_path / f"bad-{key}.json"
        candidate.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(score.ScoreError, match=message):
            score.verify_authorization(
                candidate,
                role="control",
                artifact_tree=inputs.artifact_tree,
                transport=inputs.transport,
                base_tree=inputs.base_tree,
                authority_binding=inputs.authority_binding,
            )


def test_confirmation_without_authorization_fails_before_fixture_open(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        score,
        "load_runtime",
        lambda _, **__: (inputs.runtime_authority, inputs.authority_binding),
    )
    monkeypatch.setattr(
        score, "verify_base", lambda _: (inputs.base, inputs.base_files, inputs.base_tree)
    )
    monkeypatch.setattr(
        score,
        "verify_artifact",
        lambda *_: (
            inputs.artifact,
            inputs.artifact_files,
            inputs.artifact_tree,
            None,
            inputs.artifact_metadata,
        ),
    )
    opened = False

    def fixture_spy(*_):
        nonlocal opened
        opened = True
        raise AssertionError("confirmation was opened")

    monkeypatch.setattr(score, "verify_fixture", fixture_spy)
    args = argparse.Namespace(
        output_dir=tmp_path / "new-output",
        gpu="0",
        timeout_seconds=60,
        docker="/usr/bin/docker",
        runtime_authority=inputs.runtime_authority,
        base=inputs.base,
        artifact=inputs.artifact,
        expected_transport="full_model",
        score_driver=DRIVER,
        expected_fingerprint=None,
        phase="confirmation",
        decision_authorization=None,
        selection_fingerprint_anchor=False,
        artifact_role="control",
        dry_run=False,
        dev=tmp_path / "absent-dev",
        confirmation=tmp_path / "protected-confirmation",
    )
    with pytest.raises(score.ScoreError, match="authorization"):
        score.prepare(args)
    assert opened is False


def test_success_receipt_keeps_ordered_raw_vector_logs_and_authority(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path)
    _stable_runtime(monkeypatch, inputs)
    observed_timeout = None

    def runner(argv, *, check, capture_output, text, timeout):
        nonlocal observed_timeout
        assert check is False and capture_output is True and text is False
        observed_timeout = timeout
        mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
        output_mount = next(value for value in mounts if "dst=/output" in value)
        host = Path(next(part[4:] for part in output_mount.split(",") if part.startswith("src=")))
        (host / "raw-result.json").write_text(json.dumps(_raw()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, b"evaluator stdout", b"evaluator stderr")

    output, receipt_sha = score.run_score(inputs, runner=runner)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert observed_timeout == 123
    assert receipt_sha == score.file_sha256(output / "receipt.json")
    assert receipt["authority"] == inputs.authority_binding
    assert receipt["result"]["vector_count"] == 2
    assert receipt["result"]["vector_sha256"] == score.canonical_sha256([1.0, 1.5])
    assert receipt["result"]["vector_order"] == "evaluator_emission_order"
    assert receipt["result"]["ordered_vector_sha256"] == score.canonical_sha256(
        {"order": "evaluator_emission_order", "values": [1.0, 1.5]}
    )
    assert (output / "raw-result.json").is_file()
    assert (output / "stdout.log").read_text() == "evaluator stdout"
    assert (output / "stderr.log").read_text() == "evaluator stderr"


def test_success_receipt_projects_science_authority_and_binds_scorer_authority(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    authority_path = tmp_path / "score-only-authority.json"
    authority_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "path": str(authority_path.resolve()),
        "sha256": score.file_sha256(authority_path),
        "authority": {"scorer": "exact-successor", "science": "unchanged"},
    }
    inputs = replace(
        inputs,
        score_only_authority=authority_path,
        score_only_authority_sha256=binding["sha256"],
        score_only_binding=binding,
    )
    _stable_runtime(monkeypatch, inputs)
    monkeypatch.setattr(
        score,
        "load_score_only_authority",
        lambda *_, **__: (authority_path.resolve(), binding["sha256"], binding),
    )

    def runner(argv, **_):
        mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
        output_mount = next(value for value in mounts if "dst=/output" in value)
        host = Path(next(part[4:] for part in output_mount.split(",") if part.startswith("src=")))
        (host / "raw-result.json").write_text(json.dumps(_raw()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    output, _ = score.run_score(inputs, runner=runner)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["authority"] == inputs.authority_binding
    assert receipt["score_only_authority"] == binding


def test_post_run_mutation_and_timeout_fail_without_publishing(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path)
    _stable_runtime(monkeypatch, inputs)

    def mutating_runner(argv, **_):
        (inputs.artifact / "model.safetensors").write_bytes(b"changed")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with pytest.raises(score.ScoreError, match="artifact changed"):
        score.run_score(inputs, runner=mutating_runner)
    assert not inputs.output.exists()

    fresh = _inputs(tmp_path / "fresh")
    _stable_runtime(monkeypatch, fresh)

    def timeout_runner(argv, **_):
        raise subprocess.TimeoutExpired(argv, fresh.timeout)

    with pytest.raises(score.ScoreError, match="deadline"):
        score.run_score(fresh, runner=timeout_runner)
    assert not fresh.output.exists()


def test_exact_score_driver_hash_is_still_bound() -> None:
    assert score.file_sha256(DRIVER) == score.DRIVER_SHA256
