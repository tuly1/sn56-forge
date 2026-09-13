"""Opt-in post-handler export of a selected LFM2-MoE LoRA artifact.

The validator's normal output is an adapter.  This module is deliberately
disabled unless ``FORGE_LFM_MOE_FULL_EXPORT=1`` is set.  When enabled, it runs
after the task handler has returned, backs up the selected adapter under the
training work directory, reconstructs a fresh PEFT model from the native base,
and atomically promotes a merged full model to the normal output path.  A
failure is diagnostic only: the original adapter remains the output and the
CLI must not invoke its fallback.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from forge import telemetry
from forge.clock import Deadline
from forge.model import pick_dtype, resolve_model_dir
from forge.tasks.common import (
    ARTIFACT_COMPLETE_BEST,
    ARTIFACT_PARTIAL_TRAINED_BEST,
    save_adapter,
    workdir,
)


EXPORT_ENV = "FORGE_LFM_MOE_FULL_EXPORT"
# The measured local conversion was about 45 s (including reload and a
# sharded save).  120 s leaves a bounded margin while staying below the CLI's
# 180 s hard export reserve; below that point leave the selected adapter alone.
MIN_REMAINING_HARD_SECONDS = 120.0
_TRAINED_TRUTHS = {ARTIFACT_COMPLETE_BEST, ARTIFACT_PARTIAL_TRAINED_BEST}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _base_model(model: Any) -> Any:
    candidate = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidate = getter()
        except Exception:
            return None
    return candidate


def is_native_lfm2_moe_config(config: Any) -> bool:
    """Check the native config identity without allocating model weights."""
    model_type = str(_config_value(config, "model_type", "") or "").casefold()
    if model_type != "lfm2_moe":
        return False
    architectures = _config_value(config, "architectures", ()) or ()
    declared = {str(item).casefold() for item in architectures if isinstance(item, str)}
    return "lfm2moeforcausallm" in declared


def is_native_lfm2_moe(model: Any) -> bool:
    """Recognize the native LFM2-MoE architecture without a config hash.

    Tournament bases may carry changed weights or nonidentity metadata, so the
    gate uses the native model type and architecture declaration/class.  It does
    not trust the public or anonymous model argument and does not bind the
    original campaign config digest.
    """
    base = _base_model(model)
    config = getattr(base, "config", None)
    if not is_native_lfm2_moe_config(config):
        return False
    class_names = {type(base).__name__.casefold()}
    if type(model) is not type(base):
        class_names.add(type(model).__name__.casefold())
    return "lfm2moeforcausallm" in class_names


def eligible(spec: Any, model: Any) -> bool:
    """Return whether this exact task shape may request the export."""
    return bool(
        getattr(spec, "task_type", None) == "InstructTextTask"
        and getattr(spec, "instruct", None) is not None
        and getattr(spec.instruct, "output", None) is not None
        and not bool(getattr(spec, "use_kl", False))
        and is_native_lfm2_moe(model)
    )


def internal_adapter_key(key: str) -> str:
    """Map a saved PEFT key to the live adapter's ``default`` parameter."""
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(".weight")] + ".default.weight"
    raise ValueError(f"unsupported adapter tensor key: {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trained_truth(adapter_dir: Path) -> tuple[str, int]:
    path = adapter_dir / "forge_artifact_truth.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    truth = payload.get("truth")
    step = int(payload.get("optimizer_step", -1))
    if truth not in _TRAINED_TRUTHS or step < 12:
        raise RuntimeError(f"selected artifact is not a trained-best export: {truth!r}, step={step}")
    return truth, step


def _copy_selected_adapter(output_dir: Path, spec: Any) -> Path:
    source = Path(workdir(spec)) / "lfm-moe-full-export-source"
    if source.exists():
        raise RuntimeError(f"refusing to reuse adapter backup: {source}")
    if not output_dir.is_dir():
        raise RuntimeError(f"selected output is missing: {output_dir}")
    if not (output_dir / "adapter_config.json").is_file():
        raise RuntimeError("selected output is not an adapter artifact")
    if not (output_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError("selected adapter weights are missing")
    shutil.copytree(output_dir, source, symlinks=True)
    return source


def _load_native_base(base_dir: Path) -> SimpleNamespace:
    """Load the native base once, with remote code explicitly disabled."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_dir), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    dtype = pick_dtype()
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    sharded = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if sharded:
        kwargs["device_map"] = "auto"
    failures: list[BaseException] = []
    model = None
    for attention in ("sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(base_dir), attn_implementation=attention, **kwargs
            )
            break
        except (OSError, ValueError, ImportError, RuntimeError, TypeError) as exc:
            # A failed allocation leaves the CUDA allocator under pressure;
            # retrying eager attention would attempt a second full model load.
            # This mirrors forge.model's native loader policy.
            if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            failures.append(exc)
    if model is None:
        raise RuntimeError(f"native LFM base load failed: {failures[-1]}")
    model.config.use_cache = False
    if torch.cuda.is_available() and not sharded:
        model = model.to("cuda")
    return SimpleNamespace(model=model, tokenizer=tokenizer)


def _promote_adapted_base_weights(
    model: Any, adapter_name: str = "default"
) -> list[tuple[Any, Any]]:
    """Promote only LoRA-targeted base weights before the native merge.

    PEFT 0.19 computes a vanilla delta in the adapter dtype, then casts it to
    the base dtype before adding it.  On the BF16 export path that rounds the
    delta before addition, which is especially consequential for LFM router
    gates.  Keeping the base weights of adapted layers in FP32 through
    ``merge_and_unload`` preserves the FP32 sum; the promoted adapted
    parameters are restored to their original dtype exactly once after the
    native merge.
    """
    import torch

    adapted: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    with torch.no_grad():
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            getter = getattr(module, "get_base_layer", None)
            if lora_a is None or not callable(getter):
                continue
            try:
                has_adapter = adapter_name in lora_a
            except TypeError:
                has_adapter = False
            if not has_adapter:
                continue
            base_layer = getter()
            for name in ("weight", "bias"):
                parameter = getattr(base_layer, name, None)
                if parameter is None or not parameter.is_floating_point():
                    continue
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                original_dtype = parameter.dtype
                adapted.append((parameter, original_dtype))
                if original_dtype != torch.float32:
                    parameter.data = parameter.data.float()
    if not adapted:
        raise RuntimeError("no adapted base weights were found")
    return adapted


def _restore_adapted_base_weights(
    promoted: list[tuple[Any, Any]], merged: Any
) -> int:
    """Restore only promoted adapted parameters after the native merge."""
    import torch

    merged_parameter_ids = {id(parameter) for parameter in merged.parameters()}
    cast = 0
    with torch.no_grad():
        for parameter, original_dtype in promoted:
            if id(parameter) not in merged_parameter_ids:
                raise RuntimeError("native merge replaced an adapted base parameter")
            if parameter.dtype != original_dtype:
                parameter.data = parameter.data.to(dtype=original_dtype)
                cast += 1
    return cast


def _reconstruct_and_promote(
    spec: Any, output_dir: Path, backup: Path, truth: str, step: int
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    base_dir = Path(resolve_model_dir(spec.cached_model_dir))
    base_model = _load_native_base(base_dir)
    if not is_native_lfm2_moe(base_model.model):
        raise RuntimeError("resolved base is not native LFM2-MoE")
    config = LoraConfig.from_pretrained(str(backup), local_files_only=True)
    if config.peft_type.value != "LORA" or config.bias != "none":
        raise RuntimeError("only bias-free LoRA adapters are supported")
    if bool(config.use_dora) or config.modules_to_save is not None:
        raise RuntimeError("DoRA/modules_to_save are outside the full export scope")

    peft_model = get_peft_model(base_model.model, config, adapter_name="default")
    saved = load_file(str(backup / "adapter_model.safetensors"), device="cpu")
    if not saved or not all(bool(value.isfinite().all()) for value in saved.values()):
        raise RuntimeError("adapter tensors are empty or nonfinite")
    if not all(".lora_A.weight" in key or ".lora_B.weight" in key for key in saved):
        raise RuntimeError("adapter contains unsupported non-LoRA tensors")
    nonzero_b = [
        key for key, value in saved.items()
        if ".lora_B.weight" in key and bool(value.ne(0).any())
    ]
    if not nonzero_b:
        raise RuntimeError("selected adapter has no nonzero LoRA-B tensor")
    parameters = dict(peft_model.named_parameters())
    expected = {internal_adapter_key(key) for key in saved}
    actual = {key for key in parameters if ".lora_" in key and ".default." in key}
    if expected != actual:
        raise RuntimeError(f"reconstructed LoRA parameter set differs: expected={len(expected)} actual={len(actual)}")
    with torch.no_grad():
        for source_key, value in saved.items():
            target = parameters[internal_adapter_key(source_key)]
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(f"shape mismatch for {source_key}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    roundtrip = get_peft_model_state_dict(peft_model, adapter_name="default")
    if set(roundtrip) != set(saved):
        raise RuntimeError("PEFT adapter key round trip differs")
    for key, value in saved.items():
        if not torch.equal(roundtrip[key].detach().cpu(), value):
            raise RuntimeError(f"PEFT adapter tensor round trip differs: {key}")

    promoted = _promote_adapted_base_weights(peft_model)
    merged = peft_model.merge_and_unload(safe_merge=True)
    cast_back = _restore_adapted_base_weights(promoted, merged)
    telemetry.event(
        "lfm_moe_full_export_prepared",
        adapter_config_sha256=_sha256(backup / "adapter_config.json"),
        adapter_model_sha256=_sha256(backup / "adapter_model.safetensors"),
        adapter_tensor_count=len(saved),
        nonzero_lora_B_count=len(nonzero_b),
        fp32_adapted_base_layers=sum(dtype != torch.float32 for _, dtype in promoted),
        adapted_base_parameter_count=len(promoted),
        merged_tensors_cast_back=cast_back,
        merge_precision="fp32_adapted_base_then_single_cast",
        source_adapter_backup=str(backup),
        production_qualified=False,
    )
    # save_adapter stages under output.tmp and performs its own atomic exchange.
    # The caller backed up the old adapter, so a failed stage leaves output_dir
    # untouched and a successful stage contains only the genuine full model.
    save_adapter(merged, base_model.tokenizer, str(output_dir),
                 artifact_truth=truth, optimizer_step=step,
                 truth_reason="lfm_moe_full_export_from_selected_adapter")


def maybe_export(spec: Any, deadline: Deadline) -> bool:
    """Best-effort opt-in export; never raises into CLI fallback handling."""
    if os.environ.get(EXPORT_ENV, "0") != "1":
        return False
    output_dir = Path(spec.output_dir)
    try:
        if deadline.remaining_hard() < MIN_REMAINING_HARD_SECONDS:
            telemetry.event("lfm_moe_full_export_skipped", reason="insufficient_deadline")
            return False
        # Validate trained-best provenance before allocating any model weights.
        truth, step = _trained_truth(output_dir)
        # Inspect only the native config before loading weights. This avoids a
        # second 8B model allocation during the subsequent reconstruction.
        base_dir = Path(resolve_model_dir(spec.cached_model_dir))
        config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        if not is_native_lfm2_moe_config(config):
            telemetry.event("lfm_moe_full_export_skipped", reason="native_identity_or_task_scope")
            return False
        if not (
            getattr(spec, "task_type", None) == "InstructTextTask"
            and getattr(spec, "instruct", None) is not None
            and getattr(spec.instruct, "output", None) is not None
            and not bool(getattr(spec, "use_kl", False))
        ):
            telemetry.event("lfm_moe_full_export_skipped", reason="native_identity_or_task_scope")
            return False
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        backup = _copy_selected_adapter(output_dir, spec)
        try:
            _reconstruct_and_promote(spec, output_dir, backup, truth, step)
        except BaseException:
            # Leave the selected adapter in place.  Keep the backup for custody;
            # it is outside the uploader-visible output tree.
            raise
        telemetry.event(
            "lfm_moe_full_export_complete",
            output_dir=str(output_dir),
            source_adapter_backup=str(backup),
            production_qualified=False,
        )
        return True
    except BaseException as exc:  # best effort must not trigger fallback
        telemetry.event(
            "lfm_moe_full_export_failed",
            error=f"{type(exc).__name__}: {exc}",
            output_preserved=output_dir.is_dir(),
        )
        return False
