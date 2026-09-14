"""Opt-in full-weight export for native Gemma 4 Instruct text tasks.

Gemma 4's multimodal config stores the text vocabulary under
``text_config.vocab_size`` while the unchanged evaluator currently reads the
top-level field.  This module reconstructs the selected LoRA adapter on a
fresh native base, merges it without changing any model buffers, and adds only
that truthful top-level alias in the staged config.  It is disabled unless
``FORGE_GEMMA4_FULL_EXPORT=1`` and failures leave the selected adapter intact.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from forge import telemetry
from forge.clock import Deadline
from forge.model import load_base, resolve_model_dir
from forge.tasks.common import (
    ARTIFACT_COMPLETE_BEST,
    ARTIFACT_PARTIAL_TRAINED_BEST,
    _fsync_tree,
    _fsync_dir,
    _promote_staged_dir,
    save_adapter,
    workdir,
)


EXPORT_ENV = "FORGE_GEMMA4_FULL_EXPORT"
# The existing post-handler leaves a 180 s CLI export reserve. Gemma 4's actual
# full-model export time is unmeasured; 120 s is a provisional admission gate
# that must be checked by the real-model qualification.
MIN_REMAINING_HARD_SECONDS = 120.0
SERIALIZATION_HEADROOM_BYTES = 1 * 1024 * 1024 * 1024
_TRAINED_TRUTHS = {ARTIFACT_COMPLETE_BEST, ARTIFACT_PARTIAL_TRAINED_BEST}
_CONTEXT_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
# The current paid qualification scope is E2B only.  These are identity
# fields from the pinned native E2B text config, rather than a model-size
# estimate; a later E4B qualification can add its own explicit tuple.
_E2B_TEXT_GEOMETRY = {
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_hidden_layers": 35,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_kv_shared_layers": 20,
    "use_double_wide_mlp": True,
    "enable_moe_block": False,
    "hidden_size_per_layer_input": 256,
    "vocab_size_per_layer_input": 262144,
    "tie_word_embeddings": True,
}
_E2B_LAYER_TYPES = tuple(
    "full_attention" if index % 5 == 4 else "sliding_attention"
    for index in range(_E2B_TEXT_GEOMETRY["num_hidden_layers"])
)


def _regular_bytes(path: Path) -> int:
    """Count regular-file bytes, including symlinked HF weight files.

    HF snapshots commonly use symlinks for blobs.  Follow symlinks only when
    they resolve to regular files, never to directories, and deduplicate inode
    aliases so a snapshot does not inflate its estimate.
    """
    seen: set[tuple[int, int]] = set()

    def add_file(filename: str) -> int:
        try:
            info = os.stat(filename, follow_symlinks=True)
        except OSError:
            return 0
        if not stat.S_ISREG(info.st_mode):
            return 0
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            return 0
        seen.add(identity)
        return info.st_size

    def walk(directory: str) -> int:
        total = 0
        try:
            entries = os.scandir(directory)
        except OSError:
            return 0
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        total += walk(entry.path)
                    elif entry.is_file(follow_symlinks=True):
                        total += add_file(entry.path)
                except OSError:
                    continue
        return total

    if path.is_dir() and not path.is_symlink():
        return walk(str(path))
    if path.is_file():
        return add_file(str(path))
    return 0


def _disk_preflight(spec: Any, base_dir: Path, output_dir: Path) -> dict[str, int]:
    """Ensure the full staged generation fits before loading model weights.

    The base already occupies its current bytes.  A merge needs one additional
    full generation plus the selected-adapter backup and staging copy.  Check
    every filesystem that can receive those writes; this remains metadata-only
    and does not create a directory or copy model data.
    """
    base_bytes = _regular_bytes(base_dir)
    adapter_bytes = _regular_bytes(output_dir)
    required_bytes = base_bytes + (2 * adapter_bytes) + SERIALIZATION_HEADROOM_BYTES
    work_root = Path(workdir(spec))
    locations = {output_dir.parent, work_root if work_root.exists() else work_root.parent}
    for location in locations:
        usage = shutil.disk_usage(str(location))
        if usage.free <= required_bytes:
            raise RuntimeError(
                f"insufficient disk for Gemma 4 full export at {location}: "
                f"free={usage.free} required>{required_bytes}"
            )
    return {"base_bytes": base_bytes, "selected_adapter_bytes": adapter_bytes,
            "required_free_bytes": required_bytes}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _base_model(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return model


def is_native_gemma4_config(config: Any) -> bool:
    """Recognize only the native Gemma 4 E2B architecture under qualification."""
    if str(_config_value(config, "model_type", "") or "").casefold() != "gemma4":
        return False
    architectures = _config_value(config, "architectures", ()) or ()
    if "gemma4forconditionalgeneration" not in {
        str(item).casefold() for item in architectures if isinstance(item, str)
    }:
        return False
    text_config = _config_value(config, "text_config")
    if str(_config_value(text_config, "model_type", "") or "").casefold() != "gemma4_text":
        return False
    vocab = _config_value(text_config, "vocab_size")
    if type(vocab) is not int or vocab <= 0:
        return False
    if any(_config_value(text_config, key) != expected
           for key, expected in _E2B_TEXT_GEOMETRY.items()):
        return False
    layer_types = _config_value(text_config, "layer_types")
    if not isinstance(layer_types, (list, tuple)):
        return False
    return tuple(layer_types) == _E2B_LAYER_TYPES


def is_native_gemma4(model: Any) -> bool:
    base = _base_model(model)
    config = getattr(base, "config", None)
    if not is_native_gemma4_config(config):
        return False
    names = {type(base).__name__.casefold(), type(model).__name__.casefold()}
    return "gemma4forconditionalgeneration" in names


def eligible(spec: Any, model: Any) -> bool:
    """Task/family gate; no weight allocation occurs here."""
    return bool(
        getattr(spec, "task_type", None) == "InstructTextTask"
        and getattr(spec, "instruct", None) is not None
        and getattr(spec.instruct, "output", None) is not None
        and not bool(getattr(spec, "use_kl", False))
        and is_native_gemma4(model)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trained_truth(adapter_dir: Path) -> tuple[str, int]:
    payload = json.loads((adapter_dir / "forge_artifact_truth.json").read_text(encoding="utf-8"))
    truth = payload.get("truth")
    step = int(payload.get("optimizer_step", -1))
    if truth not in _TRAINED_TRUTHS or step < 12:
        raise RuntimeError(f"selected artifact is not trained: truth={truth!r} step={step}")
    return truth, step


def _copy_selected_adapter(output_dir: Path, spec: Any) -> Path:
    source = Path(workdir(spec)) / "gemma4-full-export-source"
    if source.exists():
        raise RuntimeError(f"refusing to reuse adapter backup: {source}")
    if not output_dir.is_dir() or not (output_dir / "adapter_config.json").is_file():
        raise RuntimeError("selected output is not an adapter artifact")
    if not (output_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError("selected adapter weights are missing")
    shutil.copytree(output_dir, source, symlinks=True)
    return source


def _internal_adapter_key(key: str) -> str:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(".weight")] + ".default.weight"
    raise ValueError(f"unsupported adapter tensor key: {key}")


def _validate_native_vocab(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = _config_value(config, "text_config")
    text_vocab = _config_value(text_config, "vocab_size")
    if type(text_vocab) is not int or text_vocab <= 0:
        raise RuntimeError("native Gemma 4 text_config.vocab_size is not a positive integer")
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    dimensions = [int(input_embeddings.weight.shape[0]), int(output_embeddings.weight.shape[0])]
    if any(dimension != text_vocab for dimension in dimensions):
        raise RuntimeError(f"text vocabulary {text_vocab} disagrees with embedding rows {dimensions}")
    return text_vocab


def _embeddings_tied(model: Any) -> bool:
    """Return the native input/output embedding tie relation."""
    return model.get_input_embeddings().weight is model.get_output_embeddings().weight


def _alias_only_config(config: Any, text_vocab: int) -> tuple[bool, Any]:
    """Add only a missing top-level alias and return (was_present, old_value)."""
    present = hasattr(config, "vocab_size")
    old = getattr(config, "vocab_size", None)
    if present and old not in (None, text_vocab):
        raise RuntimeError(f"top-level vocab_size {old!r} disagrees with text_config {text_vocab}")
    if not present or old is None:
        config.vocab_size = text_vocab
    return present, old


def _restore_alias(config: Any, present: bool, old: Any) -> None:
    if present:
        config.vocab_size = old
    else:
        try:
            delattr(config, "vocab_size")
        except AttributeError:
            pass


def _config_with_source_alias(source: dict[str, Any], text_vocab: int) -> dict[str, Any]:
    """Return the supplied config plus the sole truthful top-level alias."""
    if not isinstance(source, dict):
        raise RuntimeError("base config is not a JSON object")
    old = source.get("vocab_size")
    if old not in (None, text_vocab):
        raise RuntimeError(f"top-level vocab_size {old!r} disagrees with text_config {text_vocab}")
    staged = dict(source)
    staged["vocab_size"] = text_vocab
    return staged


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write metadata atomically and durably within its existing directory."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(str(path.parent))
    finally:
        temporary.unlink(missing_ok=True)


def _validate_staged_alias(path: Path, text_vocab: int) -> None:
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if payload.get("vocab_size") != text_vocab:
        raise RuntimeError("staged config lacks the exact top-level text vocabulary alias")
    nested = payload.get("text_config")
    if not isinstance(nested, dict) or nested.get("vocab_size") != text_vocab:
        raise RuntimeError("staged nested text_config vocabulary changed")
    if payload.get("model_type") != "gemma4":
        raise RuntimeError("staged full artifact lost Gemma 4 model_type")
    if "Gemma4ForConditionalGeneration" not in payload.get("architectures", []):
        raise RuntimeError("staged full artifact lost native Gemma 4 architecture")


def _validate_full_keyset(base: Any, merged: Any, unchanged_keys: set[str] | None = None) -> None:
    import torch

    base_state, merged_state = base.state_dict(), merged.state_dict()
    if set(base_state) != set(merged_state):
        raise RuntimeError(f"merged full model key set changed: base={len(base_state)} merged={len(merged_state)}")
    for key in base_state:
        if tuple(base_state[key].shape) != tuple(merged_state[key].shape):
            raise RuntimeError(f"merged full model shape changed for {key}")
        if not bool(merged_state[key].isfinite().all().item()):
            raise RuntimeError(f"merged full model contains nonfinite tensor: {key}")
        if unchanged_keys is not None and key in unchanged_keys:
            if not torch.equal(base_state[key], merged_state[key].detach().cpu()):
                raise RuntimeError(f"unadapted full model tensor changed for {key}")


def _adapted_state_keys(model: Any, original_parameter_names: dict[int, str]) -> set[str]:
    """Map PEFT-wrapped adapted parameters back to original state keys."""
    names: set[str] = set()
    for module in model.modules():
        getter = getattr(module, "get_base_layer", None)
        if not callable(getter) or not hasattr(module, "lora_A"):
            continue
        base_layer = getter()
        for field in ("weight", "bias"):
            parameter = getattr(base_layer, field, None)
            name = original_parameter_names.get(id(parameter))
            if name is not None:
                names.add(name)
    return names


def _reconstruct_and_promote(spec: Any, output_dir: Path, backup: Path, truth: str, step: int) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    from forge.tasks.lfm25_full_export import _promote_adapted_base_weights, _restore_adapted_base_weights

    base_dir = Path(resolve_model_dir(spec.cached_model_dir))
    source_config_path = base_dir / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not is_native_gemma4_config(source_config):
        raise RuntimeError("supplied base config is not the pinned native Gemma 4 E2B shape")
    loaded = load_base(str(base_dir), for_generation=False)
    base_model, tokenizer = loaded.model, loaded.tokenizer
    if not is_native_gemma4(base_model):
        raise RuntimeError("resolved base is not native Gemma 4")
    text_vocab = _validate_native_vocab(base_model)
    if _config_value(_config_value(source_config, "text_config"), "vocab_size") != text_vocab:
        raise RuntimeError("supplied base config vocabulary disagrees with loaded native base")
    embeddings_tied = _embeddings_tied(base_model)
    base_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
    config = LoraConfig.from_pretrained(str(backup), local_files_only=True)
    peft_type = getattr(config.peft_type, "value", config.peft_type)
    if str(peft_type).upper() != "LORA" or config.bias != "none" or bool(getattr(config, "use_dora", False)):
        raise RuntimeError("only bias-free ordinary LoRA adapters are supported")
    if config.modules_to_save is not None:
        raise RuntimeError("modules_to_save is outside the Gemma 4 full-export scope")
    original_parameter_names = {id(parameter): name for name, parameter in base_model.named_parameters()}
    peft_model = get_peft_model(base_model, config, adapter_name="default")
    saved = load_file(str(backup / "adapter_model.safetensors"), device="cpu")
    if not saved or not all(bool(value.isfinite().all().item()) for value in saved.values()):
        raise RuntimeError("adapter tensors are empty or nonfinite")
    if not all(".lora_A.weight" in key or ".lora_B.weight" in key for key in saved):
        raise RuntimeError("adapter contains unsupported tensors")
    if not any(".lora_B.weight" in key and bool(value.ne(0).any()) for key, value in saved.items()):
        raise RuntimeError("selected adapter has no nonzero LoRA-B tensor")
    parameters = dict(peft_model.named_parameters())
    expected = {_internal_adapter_key(key) for key in saved}
    actual = {key for key in parameters if ".lora_" in key and ".default." in key}
    if expected != actual:
        raise RuntimeError(f"reconstructed LoRA key set differs: expected={len(expected)} actual={len(actual)}")
    with torch.no_grad():
        for source_key, value in saved.items():
            target = parameters[_internal_adapter_key(source_key)]
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(f"shape mismatch for {source_key}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    roundtrip = get_peft_model_state_dict(peft_model, adapter_name="default")
    if set(roundtrip) != set(saved) or any(not torch.equal(roundtrip[key].cpu(), value) for key, value in saved.items()):
        raise RuntimeError("PEFT adapter tensor round trip differs")

    promoted = _promote_adapted_base_weights(peft_model)
    adapted_keys = _adapted_state_keys(peft_model, original_parameter_names)
    if not adapted_keys:
        raise RuntimeError("could not identify adapted base state keys")
    merged = peft_model.merge_and_unload(safe_merge=True)
    _restore_adapted_base_weights(promoted, merged)
    if embeddings_tied and not _embeddings_tied(merged):
        raise RuntimeError("native tied input/output embeddings were lost during merge")
    unchanged_keys = set(base_state) - adapted_keys
    _validate_full_keyset(SimpleNamespace(state_dict=lambda: base_state), merged, unchanged_keys)
    alias_present, alias_old = _alias_only_config(merged.config, text_vocab)
    staged_config = _config_with_source_alias(source_config, text_vocab)
    stage = Path(workdir(spec)) / "gemma4-full-export-staged"
    if stage.exists():
        raise RuntimeError(f"refusing to reuse staged full export: {stage}")
    telemetry.event(
        "gemma4_full_export_prepared",
        adapter_config_sha256=_sha256(backup / "adapter_config.json"),
        adapter_model_sha256=_sha256(backup / "adapter_model.safetensors"),
        adapter_tensor_count=len(saved),
        text_vocab_size=text_vocab,
        alias_field="vocab_size",
        architecture="Gemma4ForConditionalGeneration",
        merge_precision="adapted_base_parameters_promoted_only",
        production_qualified=False,
    )
    try:
        save_adapter(merged, tokenizer, str(stage), artifact_truth=truth, optimizer_step=step,
                     truth_reason="gemma4_full_export_from_selected_adapter")
        for filename in _CONTEXT_FILES:
            source = base_dir / filename
            target = stage / filename
            if source.is_file():
                shutil.copy2(source, target)
                if _sha256(source) != _sha256(target):
                    raise RuntimeError(f"staged context file differs from supplied base: {filename}")
        _write_json_atomic(stage / "config.json", staged_config)
        telemetry.write_into(str(stage))
        _validate_staged_alias(stage, text_vocab)
        _fsync_tree(str(stage))
        _promote_staged_dir(str(stage), str(output_dir))
    finally:
        _restore_alias(merged.config, alias_present, alias_old)


def maybe_export(spec: Any, deadline: Deadline) -> bool:
    """Run only when explicitly enabled; failures preserve the adapter."""
    if os.environ.get(EXPORT_ENV, "0") != "1":
        return False
    started = time.monotonic()
    output_dir = Path(spec.output_dir)
    backup = None
    try:
        telemetry.event("gemma4_full_export_started")
        if deadline.remaining_hard() < MIN_REMAINING_HARD_SECONDS:
            telemetry.event("gemma4_full_export_skipped", reason="insufficient_deadline")
            return False
        truth, step = _trained_truth(output_dir)
        base_dir = Path(resolve_model_dir(spec.cached_model_dir))
        config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        if not is_native_gemma4_config(config) or not (
            getattr(spec, "task_type", None) == "InstructTextTask"
            and getattr(spec, "instruct", None) is not None
            and getattr(spec.instruct, "output", None) is not None
            and not bool(getattr(spec, "use_kl", False))
        ):
            telemetry.event("gemma4_full_export_skipped", reason="native_identity_or_task_scope")
            return False
        disk = _disk_preflight(spec, base_dir, output_dir)
        telemetry.event("gemma4_full_export_disk_preflight", **disk)
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        backup = _copy_selected_adapter(output_dir, spec)
        _reconstruct_and_promote(spec, output_dir, backup, truth, step)
        telemetry.event("gemma4_full_export_complete", output_dir=str(output_dir), source_adapter_backup=str(backup),
                        elapsed_seconds=time.monotonic() - started, production_qualified=False)
        return True
    except BaseException as exc:
        telemetry.event("gemma4_full_export_failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started, output_preserved=output_dir.is_dir(),
                        source_adapter_backup=str(backup) if backup else None)
        return False
