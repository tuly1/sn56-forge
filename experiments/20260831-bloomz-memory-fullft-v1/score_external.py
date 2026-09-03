#!/usr/bin/env python3
"""Score one frozen BloomZ artifact with the digest-pinned external evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "sn56.bloomz-external-score.v2"
DECISION_SCHEMA = "sn56.bloomz-dev-decision.v2"
LOCAL_SCOPE = "matched_public_fixture_only_no_official_calibration"
IMAGE = (
    "gradientsio/text-evaluator:basilica@"
    "sha256:860d49c7317a82b68d93b7e0e257091d810fdea12eee3013f373903092d279d0"
)
ENTRYPOINT = "/workspace/axolotl-venv/bin/python"
DRIVER_SHA256 = "6952bf4a9b365fa00387b87dd813eaf69d1ad8d0a555a668751990a673a1b0a3"
DATASET_TYPE = dict(
    field_system="system", field_instruction="instruct", field_output="output",
    system_format="{system}", no_input_format="{instruction}",
)
FIXTURES = {
    "selection": {
        "filename": "dev.jsonl",
        "row_count": 1024,
        "prepared_row_count": 1021,
        "sha256": "f5548b1864a55c208f9f8061cb0e1d2471a6e58b976bb532ffdbb7a584bbfad6",
    },
    "confirmation": {
        "filename": "confirmation.jsonl",
        "row_count": 512,
        "prepared_row_count": 511,
        "sha256": "2b1a788ed12051688402d6709f75c7e1727d26711f4a52c9925d9eff5892c7ae",
    },
}
BASE = {
    "repo": "bigscience/bloomz-560m",
    "revision": "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05",
}
BASE_FILES = {
    "config.json": (715, "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb"),
    "model.safetensors": (1_118_459_450, "365b2c5e9bd1057eb1e3f1a4fc3f89ae6584d20f24b682d2406bc7e90178ec13"),
    "tokenizer.json": (14_500_438, "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3"),
    "tokenizer_config.json": (222, "ae85f7ec32efe4ba09f3914743b0187528eab0322fe90c4e077a9229d1de64a9"),
    "special_tokens_map.json": (85, "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"),
}
CONFIG_REQUIRED = dict(architectures=["BloomForCausalLM"], model_type="bloom",
                       vocab_size=250_880, seq_length=2048, use_cache=True)
TRANSPORTS = frozenset({"full_model", "peft_adapter"})
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{32}$")
GPU_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DEFAULT_TIMEOUT_SECONDS = 540
MAX_TIMEOUT_SECONDS = 540
REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_BASE_PREFIX = PurePosixPath("/cache/models")
EXPECTED_LORA_TARGETS = frozenset(
    {
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    }
)

class ScoreError(RuntimeError):
    pass
def require(condition: object, message: str) -> None:
    if not condition:
        raise ScoreError(message)

def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return payload + (b"\n" if newline else b"")

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def inventory(path: Path, label: str) -> tuple[Path, list[dict[str, Any]], str]:
    expanded = path.expanduser()
    require(not expanded.is_symlink(), f"{label} root is a symlink")
    root = expanded.resolve(strict=True)
    require(root.is_dir(), f"{label} is not a directory")
    files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        require(not item.is_symlink(), f"{label} contains symlink: {item}")
        require(item.is_dir() or item.is_file(), f"{label} contains special file: {item}")
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": file_sha256(item),
                }
            )
    return root, files, canonical_sha256(files)

def verify_base(path: Path) -> tuple[Path, list[dict[str, Any]], str]:
    root, files, tree = inventory(path, "pinned base")
    by_name = {item["path"]: item for item in files}
    for name, (size, digest) in BASE_FILES.items():
        actual = by_name.get(name)
        require(
            actual is not None
            and actual["bytes"] == size
            and actual["sha256"] == digest,
            f"pinned base file drift: {name}",
        )
    for name in ("added_tokens.json", "chat_template.jinja"):
        require(name not in by_name, f"pinned base unexpectedly contains {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    require(isinstance(config, dict), "base config is not an object")
    for key, expected in CONFIG_REQUIRED.items():
        require(config.get(key) == expected, f"base config drift: {key}")
    require("max_position_embeddings" not in config, "base invented max_position_embeddings")
    require("auto_map" not in config, "base enables remote code")
    return root, files, tree

def _adapter_alias(value: Any) -> str:
    require(isinstance(value, str) and value, "LoRA base_model_name_or_path is absent")
    require("," not in value and "\n" not in value, "unsafe LoRA base path")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value, "LoRA base path is not normalized absolute")
    require(".." not in path.parts, "LoRA base path contains parent traversal")
    require(
        path != ADAPTER_BASE_PREFIX and path.is_relative_to(ADAPTER_BASE_PREFIX),
        "LoRA base path must be below /cache/models",
    )
    return value

def _lora_metadata(config: Any) -> tuple[str, dict[str, Any]]:
    require(isinstance(config, dict), "adapter config is not an object")
    exact = {
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
        "auto_mapping": None, "exclude_modules": None, "modules_to_save": None,
        "revision": None, "rank_pattern": {}, "alpha_pattern": {},
        "layers_to_transform": None, "layers_pattern": None, "layer_replication": None,
        "megatron_config": None, "megatron_core": "megatron.core",
        "trainable_token_indices": None, "target_parameters": None,
        "loftq_config": {}, "eva_config": None, "corda_config": None,
        "lora_ga_config": None, "alora_invocation_tokens": None,
        "qalora_group_size": 16, "use_bdlora": None, "arrow_config": None,
        "ensure_weight_tying": False,
    }
    require(set(config) == set(exact) | {"base_model_name_or_path", "target_modules"}, "LoRA config shape drift")
    for key, expected in exact.items():
        require(config.get(key) == expected, f"LoRA metadata drift: {key}")
    targets = config.get("target_modules")
    require(
        isinstance(targets, list)
        and len(targets) == len(EXPECTED_LORA_TARGETS)
        and all(isinstance(item, str) for item in targets)
        and frozenset(targets) == EXPECTED_LORA_TARGETS,
        "LoRA target modules drift",
    )
    alias = _adapter_alias(config.get("base_model_name_or_path"))
    return alias, {
        "base_model_name_or_path": alias,
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_module_count": len(targets),
        "target_modules_sha256": canonical_sha256(sorted(targets)),
    }

def verify_artifact(
    path: Path,
    base: Path,
    transport: str,
) -> tuple[Path, list[dict[str, Any]], str, str | None, dict[str, Any]]:
    root, files, tree = inventory(path, "artifact")
    require(files, "artifact is empty")
    names = {item["path"] for item in files}
    full = {"model.safetensors", "model.safetensors.index.json"}
    adapter = {"adapter_config.json", "adapter_model.safetensors"}
    require(not any(name.endswith((".bin", ".pt", ".pth")) for name in names), "artifact has pickle weights")
    if transport == "full_model":
        require(not names.intersection(adapter), "full artifact contains adapter files")
        require(bool(names.intersection(full)), "full artifact lacks safetensors weights")
        require("pytorch_model.bin" not in names, "full artifact must use safetensors")
        config = root / "config.json"
        require(config.is_file(), "full artifact lacks config.json")
        require(
            config.read_bytes() == (base / "config.json").read_bytes(),
            "full artifact config is not byte-identical to base",
        )
        return root, files, tree, None, {
            "transport": transport,
            "config_sha256": file_sha256(config),
        }
    require(transport == "peft_adapter", "unknown artifact transport")
    require(adapter <= names, "LoRA artifact lacks exact config/safetensors files")
    lora_weights = {name for name in names if name.endswith(".safetensors")}
    require(lora_weights == {"adapter_model.safetensors"}, "LoRA artifact has unexpected weights")
    require(not names.intersection(full), "LoRA artifact contains full-model weights")
    require("adapter_model.bin" not in names, "LoRA artifact must use safetensors")
    config_path = root / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"invalid adapter config: {exc}") from exc
    alias, facts = _lora_metadata(config)
    return root, files, tree, alias, {
        "transport": transport,
        "adapter_config_sha256": file_sha256(config_path),
        **facts,
    }

def choose_fixture(phase: str, dev: Path, confirmation: Path) -> tuple[Path, dict[str, Any]]:
    require(phase in FIXTURES, "phase must be selection or confirmation")
    return (dev if phase == "selection" else confirmation), FIXTURES[phase]

def verify_fixture(
    phase: str, dev: Path, confirmation: Path
) -> tuple[Path, dict[str, Any]]:
    selected, expected = choose_fixture(phase, dev, confirmation)
    expanded = selected.expanduser()
    require(not expanded.is_symlink(), f"{phase} fixture is unsafe")
    selected = expanded.resolve(strict=True)
    require(selected.is_file(), f"{phase} fixture is unsafe")
    require(file_sha256(selected) == expected["sha256"], f"{phase} fixture hash drift")
    count = 0
    with selected.open(encoding="utf-8") as handle:
        for line in handle:
            require(line.endswith("\n"), f"unterminated {phase} row")
            row = json.loads(line)
            require(
                isinstance(row, dict)
                and set(row) == {"system", "instruct", "output"}
                and all(isinstance(value, str) for value in row.values()),
                f"{phase} schema drift at row {count}",
            )
            count += 1
    require(count == expected["row_count"], f"{phase} row-count drift")
    return selected, {**expected, "phase": phase}

def regular_json(path: Path, label: str) -> tuple[Path, Mapping[str, Any], str]:
    expanded = path.expanduser()
    require(not expanded.is_symlink(), f"{label} must not be a symlink")
    resolved = expanded.resolve(strict=True)
    require(resolved.is_file() and 0 < resolved.stat().st_size <= 2_000_000, f"unsafe {label}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"invalid {label}: {exc}") from exc
    require(isinstance(value, Mapping), f"{label} is not an object")
    return resolved, value, file_sha256(resolved)

def load_runtime(path: Path) -> tuple[Path, dict[str, Any]]:
    from forge.tuning import bloomz

    require(not path.expanduser().is_symlink(), "runtime authority must not be a symlink")
    try:
        resolved, authority, digest = bloomz.load_runtime_authority(
            path, source_root=REPO_ROOT
        )
        bloomz.require_science_stage(
            authority["lease"],
            stage_max_seconds=570,
            remaining_planned_seconds=570,
        )
    except Exception as exc:
        raise ScoreError(f"runtime authority rejected: {exc}") from exc
    return resolved, bloomz.authority_fields(authority, digest)

def verify_authorization(
    path: Path,
    *,
    role: str,
    artifact_tree: str,
    transport: str,
    base_tree: str,
    authority_binding: Mapping[str, str],
) -> tuple[Path, str, dict[str, str]]:
    resolved, decision, digest = regular_json(path, "dev decision authorization")
    require(decision.get("schema_version") == DECISION_SCHEMA, "decision schema drift")
    require(decision.get("kind") == "bloomz_paired_external_dev_decision", "decision kind drift")
    require(decision.get("status") == "AUTHORIZED_FOR_ONE_CONFIRMATION", "decision is not authorized")
    require(decision.get("local_scope") == LOCAL_SCOPE, "decision scope drift")
    selection = decision.get("selection")
    require(isinstance(selection, Mapping), "decision lacks selection")
    fixture = selection.get("fixture")
    require(
        isinstance(fixture, Mapping)
        and fixture.get("sha256") == FIXTURES["selection"]["sha256"]
        and fixture.get("row_count") == FIXTURES["selection"]["row_count"]
        and fixture.get("vector_count") == FIXTURES["selection"]["prepared_row_count"]
        and isinstance(fixture.get("eval_set_fingerprint"), str)
        and FINGERPRINT_RE.fullmatch(str(fixture["eval_set_fingerprint"])),
        "decision selection fixture drift",
    )
    bindings = selection.get("bindings")
    expected = {
        "evaluator_image": IMAGE,
        "score_driver_sha256": DRIVER_SHA256,
        "base_identity_sha256": base_tree,
    }
    require(isinstance(bindings, Mapping), "decision lacks selection bindings")
    for key, value in expected.items():
        require(bindings.get(key) == value, f"decision binding drift: {key}")
    require(selection.get("authority") == authority_binding, "decision runtime authority drift")
    confirmation = decision.get("confirmation")
    require(
        isinstance(confirmation, Mapping)
        and confirmation.get("filename") == FIXTURES["confirmation"]["filename"]
        and confirmation.get("sha256") == FIXTURES["confirmation"]["sha256"]
        and confirmation.get("row_count") == FIXTURES["confirmation"]["row_count"]
        and confirmation.get("expected_vector_count")
        == FIXTURES["confirmation"]["prepared_row_count"],
        "decision confirmation fixture drift",
    )
    authorized = confirmation.get("authorized_artifacts")
    require(
        isinstance(authorized, list)
        and len(authorized) == 2
        and [item.get("role") if isinstance(item, Mapping) else None for item in authorized]
        == ["control", "candidate"],
        "decision must authorize ordered control and candidate",
    )
    matches = [
        item
        for item in authorized
        if isinstance(item, Mapping)
        and item.get("role") == role
        and item.get("tree_sha256") == artifact_tree
        and item.get("transport") == transport
    ]
    require(len(matches) == 1, "exact artifact role/tree/transport is not authorized")
    return resolved, digest, {"path": str(resolved), "sha256": digest, "role": role}

@dataclass(frozen=True)
class Inputs:
    phase: str
    role: str
    transport: str
    artifact: Path
    artifact_files: list[dict[str, Any]]
    artifact_tree: str
    artifact_metadata: dict[str, Any]
    adapter_alias: str | None
    base: Path
    base_files: list[dict[str, Any]]
    base_tree: str
    fixture: Path
    fixture_facts: dict[str, Any]
    driver: Path
    runtime_authority: Path
    authority_binding: dict[str, Any]
    expected_fingerprint: str | None
    fingerprint_anchor: bool
    authorization: Path | None
    authorization_sha256: str | None
    authorization_facts: dict[str, str] | None
    output: Path
    gpu: str
    docker: str
    timeout: int

def disjoint(output: Path, paths: Sequence[Path]) -> None:
    for path in paths:
        require(
            output != path
            and not output.is_relative_to(path)
            and not path.is_relative_to(output),
            f"output overlaps input: {path}",
        )

def prepare(args: argparse.Namespace) -> Inputs:
    output = args.output_dir.expanduser().resolve(strict=False)
    require(not os.path.lexists(output), f"refusing to overwrite output: {output}")
    require(GPU_RE.fullmatch(args.gpu), "invalid GPU selector")
    require(1 <= args.timeout_seconds <= MAX_TIMEOUT_SECONDS, "invalid timeout")
    require(Path(args.docker).is_absolute(), "docker executable must be absolute")
    runtime_path, authority_binding = load_runtime(args.runtime_authority)
    base, base_files, base_tree = verify_base(args.base)
    artifact, artifact_files, artifact_tree, alias, metadata = verify_artifact(
        args.artifact, base, args.expected_transport
    )
    driver = args.score_driver.expanduser().resolve(strict=True)
    require(not args.score_driver.expanduser().is_symlink(), "score driver is a symlink")
    require(driver.is_file() and file_sha256(driver) == DRIVER_SHA256, "score driver drift")
    expected = args.expected_fingerprint
    if expected is not None:
        require(FINGERPRINT_RE.fullmatch(expected), "expected fingerprint must be 32 lowercase hex")
    if args.phase == "selection":
        require(args.decision_authorization is None, "selection must not accept decision authorization")
        require(bool(args.selection_fingerprint_anchor) != bool(expected), "selection needs anchor xor fingerprint")
        if args.selection_fingerprint_anchor:
            require(args.artifact_role == "control", "fingerprint anchor must be control")
    else:
        require(not args.dry_run, "confirmation dry-run is forbidden")
        require(expected is None, "confirmation cannot accept a selection fingerprint")
        require(not args.selection_fingerprint_anchor, "confirmation cannot establish selection anchor")
        require(args.decision_authorization is not None, "confirmation requires dev authorization")

    authorization = None
    authorization_sha = None
    authorization_facts = None
    if args.phase == "confirmation":
        authorization, authorization_sha, authorization_facts = verify_authorization(
            args.decision_authorization,
            role=args.artifact_role,
            artifact_tree=artifact_tree,
            transport=args.expected_transport,
            base_tree=base_tree,
            authority_binding=authority_binding,
        )
    # Only now may confirmation be opened. Selection never resolves its path.
    fixture, fixture_facts = verify_fixture(args.phase, args.dev, args.confirmation)
    paths = [artifact, base, fixture, driver, runtime_path]
    if authorization is not None:
        paths.append(authorization)
    disjoint(output, paths)
    return Inputs(
        phase=args.phase,
        role=args.artifact_role,
        transport=args.expected_transport,
        artifact=artifact,
        artifact_files=artifact_files,
        artifact_tree=artifact_tree,
        artifact_metadata=metadata,
        adapter_alias=alias,
        base=base,
        base_files=base_files,
        base_tree=base_tree,
        fixture=fixture,
        fixture_facts=fixture_facts,
        driver=driver,
        runtime_authority=runtime_path,
        authority_binding=authority_binding,
        expected_fingerprint=expected,
        fingerprint_anchor=bool(args.selection_fingerprint_anchor),
        authorization=authorization,
        authorization_sha256=authorization_sha,
        authorization_facts=authorization_facts,
        output=output,
        gpu=args.gpu,
        docker=args.docker,
        timeout=args.timeout_seconds,
    )

def mount(source: Path, destination: str, readonly: bool = True) -> str:
    src = str(source)
    dst = PurePosixPath(destination)
    require("," not in src and "\n" not in src, "unsafe mount source")
    require(dst.is_absolute() and str(dst) == destination and "," not in destination, "unsafe mount target")
    return f"type=bind,src={src},dst={destination}" + (",readonly" if readonly else "")

def render_argv(inputs: Inputs, output: Path) -> list[str]:
    argv = [
        inputs.docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        f"device={inputs.gpu}",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--shm-size",
        "16g",
        "--mount",
        mount(inputs.fixture, "/fixture/eval.jsonl"),
        "--mount",
        mount(inputs.artifact, "/artifact"),
        "--mount",
        mount(inputs.base, "/base"),
    ]
    if inputs.adapter_alias:
        argv.extend(("--mount", mount(inputs.base, inputs.adapter_alias)))
    argv.extend(
        (
            "--mount",
            mount(inputs.driver, "/runner/local_artifact_score_driver.py"),
            "--mount",
            mount(output, "/output", readonly=False),
        )
    )
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "7",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": "/app",
        "DATASET": "/fixture/eval.jsonl",
        "ORIGINAL_MODEL": "/base",
        "CONTINUOUS_SFT_TOKENIZER_REPO": "/base",
        "MODELS": "/artifact",
        "DATASET_TYPE": canonical_bytes(DATASET_TYPE).decode("utf-8"),
        "FILE_FORMAT": "json",
        "USE_KL": "0",
        "KL_COEF": "0",
        "EMIT_PER_EXAMPLE_LOSSES": "1",
        "SN56_LOCAL_SCORE_OUTPUT": "/output/raw-result.json",
    }
    for key, value in env.items():
        argv.extend(("--env", f"{key}={value}"))
    argv.extend(("--entrypoint", ENTRYPOINT, IMAGE, "-B", "/runner/local_artifact_score_driver.py"))
    return argv

def validate_result(raw: Any, inputs: Inputs) -> dict[str, Any]:
    require(isinstance(raw, dict) and list(raw) == ["/artifact"], "raw result artifact key drift")
    result = raw["/artifact"]
    require(isinstance(result, dict) and result.get("is_finetune") is True, "artifact is not a finetune")
    require(result.get("sn56_local_artifact_transport") == inputs.transport, "transport drift")
    scalar = result.get("eval_loss")
    vector = result.get("per_example_losses")
    fingerprint = result.get("eval_set_fingerprint")
    require(
        isinstance(scalar, (int, float))
        and not isinstance(scalar, bool)
        and math.isfinite(float(scalar))
        and float(scalar) >= 0,
        "invalid scalar loss",
    )
    require(
        isinstance(vector, list)
        and len(vector) == inputs.fixture_facts["prepared_row_count"]
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in vector
        ),
        "invalid per-example loss vector",
    )
    require(isinstance(fingerprint, str) and FINGERPRINT_RE.fullmatch(fingerprint), "invalid fingerprint")
    if inputs.expected_fingerprint is not None:
        require(fingerprint == inputs.expected_fingerprint, "fingerprint differs from anchor")
    values = [float(value) for value in vector]
    mean = sum(values) / len(values)
    require(math.isclose(mean, float(scalar), rel_tol=1e-3, abs_tol=1e-4), "vector mean differs from scalar")
    return {
        "eval_loss": float(scalar),
        "per_example_losses": values,
        "vector_mean": mean,
        "vector_count": len(values),
        "vector_sha256": canonical_sha256(values),
        "vector_order": "evaluator_emission_order",
        "ordered_vector_sha256": canonical_sha256(
            {"order": "evaluator_emission_order", "values": values}
        ),
        "eval_set_fingerprint": fingerprint,
        "transport": inputs.transport,
        "is_finetune": True,
    }

def unchanged(inputs: Inputs) -> None:
    _, files, tree = inventory(inputs.artifact, "artifact")
    require(files == inputs.artifact_files and tree == inputs.artifact_tree, "artifact changed during score")
    _, files, tree = inventory(inputs.base, "pinned base")
    require(files == inputs.base_files and tree == inputs.base_tree, "base changed during score")
    require(file_sha256(inputs.fixture) == inputs.fixture_facts["sha256"], "fixture changed during score")
    require(file_sha256(inputs.driver) == DRIVER_SHA256, "driver changed during score")
    _, authority_binding = load_runtime(inputs.runtime_authority)
    require(authority_binding == inputs.authority_binding, "runtime authority changed during score")
    if inputs.authorization is not None:
        require(file_sha256(inputs.authorization) == inputs.authorization_sha256, "authorization changed during score")

def write_exclusive(path: Path, payload: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()

def run_score(
    inputs: Inputs,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> tuple[Path, str]:
    inputs.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{inputs.output.name}.score.", dir=inputs.output.parent))
    try:
        argv = render_argv(inputs, staging)
        try:
            completed = runner(
                argv,
                check=False,
                capture_output=True,
                text=False,
                timeout=inputs.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            unchanged(inputs)
            raise ScoreError(f"evaluator exceeded {inputs.timeout}-second deadline") from exc
        unchanged(inputs)
        stdout = completed.stdout or b""
        stderr = completed.stderr or b""
        require(isinstance(stdout, bytes) and isinstance(stderr, bytes), "evaluator logs are not bytes")
        stdout_sha = write_exclusive(staging / "stdout.log", stdout)
        stderr_sha = write_exclusive(staging / "stderr.log", stderr)
        failure_tail = (stdout + b"\n" + stderr)[-4000:].decode("utf-8", "replace")
        require(completed.returncode == 0, f"evaluator failed: {failure_tail}")
        raw_path = staging / "raw-result.json"
        require(raw_path.is_file() and not raw_path.is_symlink(), "raw result is absent")
        raw_sha = file_sha256(raw_path)
        result = validate_result(json.loads(raw_path.read_text(encoding="utf-8")), inputs)
        argv_payload = canonical_bytes(argv, newline=True)
        argv_sha = write_exclusive(staging / "docker-argv.json", argv_payload)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bloomz_digest_pinned_external_score",
            "status": "PASS",
            "phase": inputs.phase,
            "artifact": {
                "role": inputs.role,
                "expected_transport": inputs.transport,
                "root": str(inputs.artifact),
                "files": inputs.artifact_files,
                "tree_sha256": inputs.artifact_tree,
                "metadata": inputs.artifact_metadata,
            },
            "base": {
                **BASE,
                "root": str(inputs.base),
                "files": inputs.base_files,
                "identity_sha256": inputs.base_tree,
            },
            "fixture": {"path": str(inputs.fixture), **inputs.fixture_facts},
            "authority": inputs.authority_binding,
            "fingerprint_policy": {
                "mode": "selection_anchor" if inputs.fingerprint_anchor else "expected"
                if inputs.expected_fingerprint
                else "authorized_confirmation",
                "expected_fingerprint": inputs.expected_fingerprint,
            },
            "evaluator": {
                "image": IMAGE,
                "entrypoint": ENTRYPOINT,
                "network": "none",
                "gpu_selector": inputs.gpu,
                "score_driver": {"path": str(inputs.driver), "sha256": DRIVER_SHA256},
                "dataset_type": DATASET_TYPE,
                "file_format": "json",
                "use_kl": False,
                "kl_coef": 0,
                "emit_per_example_losses": True,
                "timeout_seconds": inputs.timeout,
                "docker_argv_filename": "docker-argv.json",
                "docker_argv_sha256": argv_sha,
                "docker_argv": argv,
                "stdout": {"filename": "stdout.log", "bytes": len(stdout), "sha256": stdout_sha},
                "stderr": {"filename": "stderr.log", "bytes": len(stderr), "sha256": stderr_sha},
            },
            "result": {
                "raw_result_filename": "raw-result.json",
                "raw_result_sha256": raw_sha,
                **{key: value for key, value in result.items() if key != "per_example_losses"},
                "vector_order": "evaluator_emission_order",
            },
        }
        if inputs.authorization_facts is not None:
            receipt["authorization"] = inputs.authorization_facts
        receipt_payload = canonical_bytes(receipt, newline=True)
        receipt_sha = write_exclusive(staging / "receipt.json", receipt_payload)
        pointer = canonical_bytes({"filename": "receipt.json", "sha256": receipt_sha}, newline=True)
        write_exclusive(staging / "receipt.sha256.json", pointer)
        require(not os.path.lexists(inputs.output), f"refusing to overwrite output: {inputs.output}")
        os.replace(staging, inputs.output)
        return inputs.output, receipt_sha
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "candidate"))
    parser.add_argument("--expected-transport", required=True, choices=sorted(TRANSPORTS))
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--score-driver", required=True, type=Path)
    parser.add_argument("--runtime-authority", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(FIXTURES))
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--selection-fingerprint-anchor", action="store_true")
    parser.add_argument("--decision-authorization", type=Path)
    parser.add_argument("--docker", default="/usr/bin/docker")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = prepare(args)
    if args.dry_run:
        print(
            canonical_bytes(
                {
                    "mode": "DRY_RUN",
                    "phase": inputs.phase,
                    "artifact_tree_sha256": inputs.artifact_tree,
                    "fixture_sha256": inputs.fixture_facts["sha256"],
                    "prepared_row_count": inputs.fixture_facts["prepared_row_count"],
                    "expected_fingerprint": inputs.expected_fingerprint,
                    "runtime_authority_sha256": inputs.authority_binding["runtime_authority_sha256"],
                    "argv": render_argv(inputs, inputs.output),
                }
            ).decode("utf-8")
        )
        return 0
    output, receipt_sha = run_score(inputs)
    message = f"BLOOMZ_EXTERNAL_SCORE=PASS phase={inputs.phase} output={output} receipt_sha256={receipt_sha}"
    print(message, flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOOMZ_EXTERNAL_SCORE=FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
