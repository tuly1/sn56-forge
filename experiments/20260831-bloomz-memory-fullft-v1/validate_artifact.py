#!/usr/bin/env python3
"""Fresh offline load and finite-forward check for a completed BloomZ artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

from forge.tuning import bloomz


class ValidationError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    require(path.is_dir(), f"artifact directory missing: {path}")
    files: list[dict[str, Any]] = []
    for candidate in path.rglob("*"):
        require(not candidate.is_symlink(), f"artifact contains symlink: {candidate}")
        require(
            candidate.is_dir() or candidate.is_file(),
            f"artifact contains special file: {candidate}",
        )
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    payload = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--runtime-authority", required=True)
    parser.add_argument("--external-score-receipt", required=True)
    return parser.parse_args(argv)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite receipt: {path}") from exc


def validate_metadata(base: Path, artifact: Path, artifact_format: str) -> None:
    base_config = base / "config.json"
    require(base_config.is_file(), "base config missing")
    raw = json.loads(base_config.read_text(encoding="utf-8"))
    require(raw.get("architectures") == ["BloomForCausalLM"], "base architecture drift")
    require(raw.get("model_type") == "bloom", "base model_type drift")
    require(raw.get("vocab_size") == 250_880, "base vocab width drift")
    require("max_position_embeddings" not in raw, "base invented max_position_embeddings")
    require(file_sha256(base_config) == bloomz.MODEL_CONFIG_SHA256, "base config hash drift")
    if artifact_format == "full_model":
        artifact_config = artifact / "config.json"
        require(artifact_config.is_file(), "full artifact config missing")
        require(
            artifact_config.read_bytes() == base_config.read_bytes(),
            "full artifact config is not byte-identical to base",
        )
        require(
            not (artifact / "adapter_config.json").exists(),
            "full artifact unexpectedly contains adapter_config.json",
        )
    else:
        adapter_config = artifact / "adapter_config.json"
        adapter_weights = artifact / "adapter_model.safetensors"
        require(adapter_weights.is_file(), "LoRA artifact safetensors are missing")
        raw_adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
        require(isinstance(raw_adapter, dict), "adapter config is not an object")
        require(raw_adapter.get("peft_type") == "LORA", "adapter is not LoRA")
        require(
            raw_adapter.get("task_type") == "CAUSAL_LM",
            "adapter task type is not CAUSAL_LM",
        )
        require(
            isinstance(raw_adapter.get("base_model_name_or_path"), str)
            and bool(raw_adapter["base_model_name_or_path"]),
            "adapter base path is absent",
        )
        require(
            not (artifact / "model.safetensors").exists(),
            "LoRA artifact unexpectedly contains full weights",
        )


def load_external_score_receipt(
    path: Path,
    *,
    artifact_tree_sha256: str,
    artifact_format: str,
    authority: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    require(
        resolved.is_file() and not resolved.is_symlink() and resolved.stat().st_size < 5_000_000,
        "external score receipt is absent, unsafe, or too large",
    )
    raw = resolved.read_bytes()
    receipt = json.loads(raw)
    require(isinstance(receipt, dict), "external score receipt is not an object")
    require(
        receipt.get("schema_version") == "sn56.bloomz-external-score.v2"
        and receipt.get("status") == "PASS",
        "external score receipt did not pass",
    )
    require(receipt.get("authority") == authority, "external score authority drift")
    artifact = receipt.get("artifact")
    evaluator = receipt.get("evaluator")
    result = receipt.get("result")
    require(
        isinstance(artifact, dict)
        and artifact.get("tree_sha256") == artifact_tree_sha256,
        "external score receipt names a different artifact tree",
    )
    require(
        isinstance(evaluator, dict)
        and evaluator.get("image") == bloomz.EVALUATOR_IMAGE
        and isinstance(evaluator.get("score_driver"), dict)
        and evaluator["score_driver"].get("sha256") == bloomz.SCORE_DRIVER_SHA256,
        "external score runtime identity drift",
    )
    require(
        isinstance(result, dict)
        and result.get("transport") == artifact_format
        and isinstance(result.get("vector_count"), int)
        and result["vector_count"] > 0
        and isinstance(result.get("eval_set_fingerprint"), str),
        "external score result does not bind a fresh exact-evaluator load",
    )
    raw_result = resolved.parent / str(result.get("raw_result_filename", ""))
    require(
        raw_result.is_file()
        and file_sha256(raw_result) == result.get("raw_result_sha256"),
        "external raw result is missing or changed",
    )
    return receipt, hashlib.sha256(raw).hexdigest()


def finite_state_summary(model: Any, *, chunk_elements: int = 4_000_000) -> dict[str, Any]:
    """Scan every loaded parameter and buffer without a model-sized scratch tensor."""
    import torch

    require(chunk_elements > 0, "finite scan chunk must be positive")
    state = model.state_dict()
    require(bool(state), "loaded model state is empty")
    dtype_counts: dict[str, int] = {}
    tensor_schema: list[dict[str, Any]] = []
    floating_tensors = 0
    total_numel = 0
    total_bytes = 0
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor), f"state entry is not a tensor: {name}")
        value = tensor.detach()
        numel = int(value.numel())
        dtype = str(value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        total_numel += numel
        total_bytes += numel * int(value.element_size())
        tensor_schema.append(
            {"name": name, "shape": list(value.shape), "dtype": dtype, "numel": numel}
        )
        if value.is_floating_point() or value.is_complex():
            floating_tensors += 1
            flattened = value.reshape(-1)
            for start in range(0, numel, chunk_elements):
                require(
                    bool(torch.isfinite(flattened[start : start + chunk_elements]).all().item()),
                    f"nonfinite serialized tensor: {name}",
                )
    schema_sha = hashlib.sha256(
        json.dumps(
            tensor_schema,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "all_tensors_finite": True,
        "tensor_count": len(state),
        "floating_or_complex_tensor_count": floating_tensors,
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "tensor_schema_sha256": schema_sha,
        "finite_scan_chunk_elements": chunk_elements,
    }


def _serialized_files(artifact: Path, artifact_format: str) -> list[Path]:
    nested_files = sorted(
        path for path in artifact.rglob("*") if path.is_file() and path.parent != artifact
    )
    require(not nested_files, "unexpected nested artifact file")
    alternate_weights = sorted(
        path
        for path in artifact.iterdir()
        if path.is_file() and path.suffix.lower() in {".bin", ".pt", ".pth"}
    )
    require(not alternate_weights, "unexpected serialized tensor format")
    files = sorted(artifact.rglob("*.safetensors"))
    relative_names = [path.relative_to(artifact).as_posix() for path in files]
    if artifact_format == "peft_adapter":
        require(
            relative_names == ["adapter_model.safetensors"],
            "LoRA serialized tensor-file inventory drift",
        )
        return files
    index_path = artifact / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        require(isinstance(weight_map, dict) and weight_map, "invalid safetensors index")
        raw_names = list(weight_map.values())
        require(
            all(
                isinstance(name, str)
                and name
                and Path(name).name == name
                and "/" not in name
                and "\\" not in name
                for name in raw_names
            ),
            "unsafe safetensors shard name",
        )
        names = sorted(set(raw_names))
        require(relative_names == names, "full shard inventory drift")
    else:
        require(
            relative_names == ["model.safetensors"],
            "full serialized tensor-file inventory drift",
        )
    return files


def _serialized_keys(files: list[Path]) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                require(name not in keys, f"duplicate serialized tensor key: {name}")
                keys.add(name)
    require(bool(keys), "serialized tensor inventory is empty")
    return keys


def serialized_state_summary(
    artifact: Path,
    artifact_format: str,
    base: Path,
    *,
    chunk_elements: int = 4_000_000,
) -> dict[str, Any]:
    """Scan every on-disk tensor and reject missing, duplicate, or extra keys."""
    import torch
    from safetensors import safe_open

    require(chunk_elements > 0, "finite scan chunk must be positive")
    files = _serialized_files(artifact, artifact_format)
    actual_keys = _serialized_keys(files)
    index_path = artifact / "model.safetensors.index.json"
    if artifact_format == "full_model" and index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        require(set(weight_map) == actual_keys, "safetensors index key inventory drift")
        for path in files:
            expected_shard_keys = {
                name for name, shard in weight_map.items() if shard == path.name
            }
            require(
                _serialized_keys([path]) == expected_shard_keys,
                f"safetensors index shard mapping drift: {path.name}",
            )
    if artifact_format == "full_model":
        expected_keys = _serialized_keys([base / "model.safetensors"])
    else:
        expected_keys = {
            f"base_model.model.{target}.lora_{side}.weight"
            for target in bloomz.EXPECTED_LORA_TARGETS
            for side in ("A", "B")
        }
    require(actual_keys == expected_keys, "serialized tensor key inventory drift")

    tensor_schema: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    total_numel = 0
    total_bytes = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                require(isinstance(tensor, torch.Tensor), f"serialized entry is not tensor: {name}")
                numel = int(tensor.numel())
                dtype = str(tensor.dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                total_numel += numel
                total_bytes += numel * int(tensor.element_size())
                tensor_schema.append(
                    {
                        "file": path.name,
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": dtype,
                        "numel": numel,
                    }
                )
                if tensor.is_floating_point() or tensor.is_complex():
                    flat = tensor.reshape(-1)
                    for start in range(0, numel, chunk_elements):
                        require(
                            bool(torch.isfinite(flat[start : start + chunk_elements]).all().item()),
                            f"nonfinite serialized tensor: {name}",
                        )
    tensor_schema.sort(key=lambda item: (item["file"], item["name"]))
    return {
        "all_tensors_finite": True,
        "tensor_count": len(tensor_schema),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "tensor_schema_sha256": hashlib.sha256(
            json.dumps(
                tensor_schema,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "tensor_files": [
            {
                "name": path.relative_to(artifact).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ],
        "finite_scan_chunk_elements": chunk_elements,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = Path(args.base_dir).resolve(strict=True)
    artifact = Path(args.artifact).resolve(strict=True)
    receipt_path = Path(args.receipt).resolve(strict=False)
    require(
        receipt_path != artifact and artifact not in receipt_path.parents,
        "validation receipt must remain outside the artifact tree",
    )
    require(
        receipt_path != base and base not in receipt_path.parents,
        "validation receipt must not modify the base tree",
    )
    bloomz.validate_model_identity(
        SimpleModelIdentity.from_config(base / "config.json"), base
    )
    adapter = artifact / "adapter_config.json"
    artifact_format = "peft_adapter" if adapter.is_file() else "full_model"
    validate_metadata(base, artifact, artifact_format)
    serialized_summary = serialized_state_summary(artifact, artifact_format, base)
    artifact_tree_before = tree_sha256(artifact)
    authority_path, authority, authority_sha = bloomz.load_runtime_authority(
        args.runtime_authority
    )
    bloomz.require_science_stage(
        authority["lease"],
        stage_max_seconds=270,
        remaining_planned_seconds=270,
    )
    external_score, external_score_sha = load_external_score_receipt(
        Path(args.external_score_receipt),
        artifact_tree_sha256=artifact_tree_before,
        artifact_format=artifact_format,
        authority=bloomz.authority_fields(authority, authority_sha),
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
    )
    if artifact_format == "peft_adapter":
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        model = PeftModel.from_pretrained(
            base_model,
            artifact,
            local_files_only=True,
            is_trainable=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            artifact,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
    loaded_base = model.get_base_model() if hasattr(model, "get_base_model") else model
    require(
        type(loaded_base).__name__ == "BloomForCausalLM",
        "native BloomForCausalLM load failed",
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    encoded = tokenizer(
        "Give one concise scientific observation.",
        return_tensors="pt",
        add_special_tokens=True,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded, use_cache=False)
    logits = output.logits
    require(logits.ndim == 3 and logits.shape[-1] == 250_880, "logit width drift")
    require(bool(torch.isfinite(logits).all().item()), "finite-forward check failed")
    checksum = float(logits[0, -1, :64].float().sum().cpu())
    require(math.isfinite(checksum), "finite-forward checksum failed")
    state_summary = finite_state_summary(model)
    receipt = {
        "schema_version": "sn56.bloomz-artifact-validation.v1",
        "status": "PASS",
        "offline": True,
        "trust_remote_code": False,
        "artifact_format": artifact_format,
        "artifact_path": str(artifact),
        "artifact_tree_sha256": artifact_tree_before,
        "base_config_sha256": file_sha256(base / "config.json"),
        "model_class": type(loaded_base).__name__,
        "architecture": ["BloomForCausalLM"],
        "vocab_size": 250_880,
        "max_position_embeddings_present": False,
        "finite_forward_checksum": checksum,
        "serialized_state_summary": serialized_summary,
        "loaded_state_summary": state_summary,
        "runtime_authority_path": str(authority_path),
        "authority": bloomz.authority_fields(authority, authority_sha),
        "fresh_exact_evaluator_load": {
            "receipt_path": str(Path(args.external_score_receipt).resolve()),
            "receipt_sha256": external_score_sha,
            "phase": external_score["phase"],
            "evaluator_image": external_score["evaluator"]["image"],
            "raw_result_sha256": external_score["result"]["raw_result_sha256"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    require(
        tree_sha256(artifact) == artifact_tree_before,
        "artifact bytes changed during fresh offline validation",
    )
    return receipt


class SimpleModelIdentity:
    """Minimal loaded-config facade for the shared immutable-file verifier."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @classmethod
    def from_config(cls, path: Path) -> "SimpleModelIdentity":
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            path.parent,
            local_files_only=True,
            trust_remote_code=False,
        )
        return cls(config)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    receipt = run(args)
    _write_exclusive(Path(args.receipt), receipt)
    print(json.dumps({"status": "PASS", "receipt": args.receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
