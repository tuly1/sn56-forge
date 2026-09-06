"""Production Qwen3.5-4B fixed factor-midpoint route.

The exact production endpoint first completes its native time-aware trajectory,
then, only when the shared deadline retains the measured reserve, completes an
independently rebuilt one-epoch trajectory and exports their fixed 50:50 LoRA
factor midpoint.  Every failure restores the already selected native artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from forge.data.schema import TaskSpec

EXACT_MODEL_ID = "Qwen/Qwen3.5-4B"
EXACT_MODEL_PATH = "/cache/models/Qwen--Qwen3.5-4B"
EXACT_CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
EXACT_INDEX_SHA256 = "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
MIN_CAP1_SOFT_SECONDS = 2400.0


@dataclass(frozen=True)
class FixedAverageRoute:
    cap1_epochs: float = 1.0
    minimum_soft_seconds: float = MIN_CAP1_SOFT_SECONDS
    endpoint_mode: str = "named_one_gpu"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _endpoint_mode(spec: TaskSpec, n_gpus: int) -> str | None:
    cache = str(spec.cached_model_dir)
    if spec.model == EXACT_MODEL_ID and cache == EXACT_MODEL_PATH and n_gpus == 1:
        return "named_one_gpu"
    if (
        isinstance(spec.model, str)
        and re.fullmatch(r"[0-9a-f]{16}", spec.model) is not None
        and cache == f"/cache/models/{spec.model}"
        and isinstance(spec.baseline_stats_path, str)
        and bool(spec.baseline_stats_path)
        and n_gpus == 2
    ):
        return "anonymous_two_gpu"
    return None


def _exact_payload(cache: str) -> bool:
    root = Path(cache)
    try:
        return (
            _sha(root / "config.json") == EXACT_CONFIG_SHA256
            and _sha(root / "model.safetensors.index.json") == EXACT_INDEX_SHA256
        )
    except OSError:
        return False


def _exact_base(model: Any) -> bool:
    identity = model
    if type(model).__name__ == "PeftModelForCausalLM":
        getter = getattr(model, "get_base_model", None)
        if not callable(getter):
            return False
        identity = getter()
    config = getattr(identity, "config", None)
    return bool(
        type(identity).__name__ == "Qwen3_5ForCausalLM"
        and type(config).__name__ == "Qwen3_5TextConfig"
        and str(getattr(config, "model_type", "")).lower() == "qwen3_5_text"
        and int(getattr(config, "hidden_size", 0)) == 2560
        and int(getattr(config, "num_hidden_layers", 0)) == 32
        and int(getattr(config, "vocab_size", 0)) == 248320
    )


def eligible_route(
    spec: TaskSpec, model: Any, *, strategy: str, n_gpus: int
) -> FixedAverageRoute | None:
    """Return the production route only for the immutable Qwen endpoint."""
    mode = _endpoint_mode(spec, n_gpus)
    if mode is None:
        return None
    if (
        spec.task_type != "InstructTextTask"
        or spec.instruct is None
        or spec.instruct.output is None
        or spec.use_kl
        or strategy != "lora"
        or not _exact_payload(str(spec.cached_model_dir))
        or not _exact_base(model)
    ):
        raise ValueError("Qwen fixed-average production identity drift")
    return FixedAverageRoute(endpoint_mode=mode)


def cap1_admitted(route: FixedAverageRoute, soft_seconds_remaining: float) -> bool:
    return bool(
        math.isfinite(soft_seconds_remaining)
        and soft_seconds_remaining >= route.minimum_soft_seconds
    )


def adapter_factor_sha256(model: Any) -> str:
    rows = []
    factors = {}
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        tensor = parameter.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite initial Qwen LoRA factor")
        if ".lora_B." in name and torch.count_nonzero(tensor).item() != 0:
            raise ValueError("initial Qwen LoRA B factor is not zero")
        rows.append((name, tuple(tensor.shape), tensor.numpy().tobytes()))
        factors[name] = tensor
    _validate_factors(factors)
    digest = hashlib.sha256()
    for name, shape, payload in sorted(rows):
        header = f"{name}|{shape}|".encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return digest.hexdigest()


def snapshot_artifact(source: str, snapshot: str) -> None:
    from forge.tasks import common
    src, dst = Path(source), Path(snapshot)
    if not src.is_dir() or src.is_symlink() or any(p.is_symlink() for p in src.rglob("*")):
        raise ValueError("native artifact is not a real self-contained directory")
    temporary = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True); shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, temporary)
    common._validate_staged_artifact(str(temporary))
    if not common._has_ready_marker(str(temporary)):
        raise ValueError("native artifact is not durably ready")
    common._fsync_tree(str(temporary))
    os.replace(temporary, dst); common._fsync_dir(str(dst.parent))


def _promote(source: str, destination: str) -> None:
    from forge.tasks import common
    src, dst = Path(source), Path(destination)
    if not src.is_dir() or src.is_symlink() or any(p.is_symlink() for p in src.rglob("*")):
        raise ValueError("fixed-average artifact is unavailable")
    common._recover_artifact_dirs(str(dst))
    common._fsync_tree(str(src))
    common._promote_staged_dir(str(src), str(dst))


def restore_artifact(snapshot: str, destination: str) -> None:
    _promote(snapshot, destination)


def promote_artifact(source: str, destination: str) -> None:
    _promote(source, destination)


def _validate_factors(weights: dict[str, torch.Tensor]) -> None:
    groups = {
        side: {k.replace(f".lora_{side}.", ".lora_PAIR."): v for k, v in weights.items() if f".lora_{side}." in k}
        for side in ("A", "B")
    }
    if len(groups["A"]) != 248 or len(groups["B"]) != 248 or groups["A"].keys() != groups["B"].keys():
        raise ValueError("expected 248 complete Qwen A/B pairs")
    for module, a in groups["A"].items():
        b = groups["B"][module]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 32 or b.shape[1] != 32 or a.dtype != b.dtype:
            raise ValueError("incompatible rank32 Qwen pair: " + module)


def build_fixed_midpoint(native: str, cap1: str, output: str) -> dict[str, Any]:
    roots = [Path(native), Path(cap1)]; dst = Path(output)
    if dst.exists():
        raise ValueError("refusing to overwrite midpoint artifact")
    configs = [json.loads((root / "adapter_config.json").read_text()) for root in roots]
    if configs[0] != configs[1] or configs[0].get("r") != 32 or configs[0].get("lora_alpha") != 64:
        raise ValueError("endpoint adapter config drift")
    paths = [root / "adapter_model.safetensors" for root in roots]
    weights = [load_file(str(path), device="cpu") for path in paths]
    for arm in weights:
        _validate_factors(arm)
    if weights[0].keys() != weights[1].keys():
        raise ValueError("endpoint factor keys differ")
    averaged = {}
    for key in sorted(weights[0]):
        left, right = weights[0][key], weights[1][key]
        if left.shape != right.shape or left.dtype != right.dtype or not left.is_floating_point():
            raise ValueError("endpoint tensor mismatch: " + key)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("nonfinite endpoint: " + key)
        averaged[key] = ((left.float() + right.float()) * 0.5).to(left.dtype).contiguous()
    dst.mkdir(parents=True)
    for name in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        src = roots[0] / name
        if src.exists():
            other = roots[1] / name
            if not other.exists() or _sha(src) != _sha(other):
                raise ValueError("endpoint tokenizer/config provenance drift: " + name)
            shutil.copy2(src, dst / name)
    target = dst / "adapter_model.safetensors"
    save_file(averaged, str(target), metadata={"format": "pt"})
    receipt = {
        "schema": 1, "truth": "DERIVED_FIXED_FACTOR_SPACE_MIDPOINT",
        "native_adapter_sha256": _sha(paths[0]), "cap1_adapter_sha256": _sha(paths[1]),
        "output_adapter_sha256": _sha(target), "native_weight": 0.5, "cap1_weight": 0.5,
        "complete_ab_pairs": 248, "tensor_count": 496, "no_score_conditioned_selection": True,
    }
    (dst / "forge_artifact_truth.json").write_text(json.dumps({
        "schema": 1, "truth": receipt["truth"], "optimizer_step": None,
        "reason": "fixed midpoint of independently selected native and cap1 best checkpoints",
    }, sort_keys=True) + "\n")
    (dst / "FIXED-AVERAGE-RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    (dst / ".forge_artifact_ready").write_text("sn56-forge-artifact-ready-v1\n")
    return receipt
