"""Fixed Qwen3.5 record-4/record-2 effective-delta release override.

This is deliberately *not* a checkpoint search policy.  It implements one
transparent ``CEO_EXPLORATORY_OVERRIDE`` supported by the Week-11 replication:
capture finite scheduled evaluation ordinals two and four, and (only when
ordinal four is also the durably persisted best checkpoint) represent the
equal mean of their effective LoRA deltas as concatenated standard LoRA
factors.  The immutable formal verdict remains
``NO_PRODUCTION_CHILD_I1_REPLICATION_FAILED``; this module does not reinterpret
or weaken that gate.

Every failure is a local fallback.  The record-four artifact remains at the
validator-visible output path until a fully validated, loadable, finite soup
has passed an inference smoke and is ready for atomic directory promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from forge.data.schema import TaskSpec


RELEASE_CLASSIFICATION = "CEO_EXPLORATORY_OVERRIDE"
FORMAL_VERDICT = "NO_PRODUCTION_CHILD_I1_REPLICATION_FAILED"
FORMAL_DECISION_FILE_SHA256 = (
    "046d48b5bf2916ad47d01746f2a482de58b6471709ae4258ef4352510c3952e6"
)
FORMAL_DECISION_RESULT_SHA256 = (
    "b58554da512cc26d6bff435310481091aae5cf0f5fa0d49515c8b6380e1018be"
)
METHOD = "FIXED_R4_R2_EQUAL_EFFECTIVE_MEAN_CONCAT_STANDARD_LORA_V1"
EXACT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
EXACT_BASE_MODEL = "/cache/models/Qwen--Qwen3.5-0.8B"
EXACT_SOURCE_RANK = 32
EXACT_SOURCE_ALPHA = 64
EXACT_SOURCE_DROPOUT = 0.05
EXACT_TENSOR_PAIR_COUNT = 186
EXACT_TARGET_MODULES = tuple(
    sorted(
        {
            "down_proj",
            "gate_proj",
            "in_proj_a",
            "in_proj_b",
            "in_proj_qkv",
            "in_proj_z",
            "k_proj",
            "o_proj",
            "out_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        }
    )
)
_EXACT_OUTER_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration",)
_EXACT_TEXT_DIMENSIONS = {
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
}
_EXACT_LAYER_TYPES = tuple(
    layer
    for _ in range(6)
    for layer in (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
)
_CAPTURE_ORDINALS = frozenset({2, 4})
_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"
_MODEL_FILE = "adapter_model.safetensors"
_CONFIG_FILE = "adapter_config.json"
_MARKER_FILE = "CEO_EXPLORATORY_OVERRIDE.json"


class SoupCompatibilityError(RuntimeError):
    """The fixed override cannot safely consume the observed artifacts."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SoupCompatibilityError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _base_model_config(model: Any) -> Any:
    candidate = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidate = getter()
        except Exception:
            return None
    return getattr(candidate, "config", getattr(model, "config", None))


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_exact_qwen35_08b(model: Any) -> bool:
    config = _base_model_config(model)
    text_config = _config_value(config, "text_config")
    if (
        str(_config_value(config, "model_type", "") or "").lower() != "qwen3_5"
        or tuple(_config_value(config, "architectures", ()) or ())
        != _EXACT_OUTER_ARCHITECTURES
        or str(_config_value(text_config, "model_type", "") or "").lower()
        != "qwen3_5_text"
        or tuple(_config_value(text_config, "layer_types", ()) or ())
        != _EXACT_LAYER_TYPES
    ):
        return False
    return all(
        _config_value(text_config, name) == expected
        for name, expected in _EXACT_TEXT_DIMENSIONS.items()
    )


def _supported_model_route(spec: TaskSpec) -> bool:
    cached_model_dir = str(spec.cached_model_dir)
    if spec.model == EXACT_MODEL_ID:
        return cached_model_dir == EXACT_BASE_MODEL
    return (
        isinstance(spec.model, str)
        and re.fullmatch(r"[0-9a-f]{16}", spec.model) is not None
        and isinstance(spec.baseline_stats_path, str)
        and bool(spec.baseline_stats_path)
        and cached_model_dir == f"/cache/models/{spec.model}"
    )


def _default_peft_config(model: Any) -> Any:
    configs = getattr(model, "peft_config", None)
    if isinstance(configs, Mapping):
        return configs.get("default")
    return None


def _target_modules(config: Any) -> tuple[str, ...]:
    raw = _config_value(config, "target_modules")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    if not all(isinstance(value, str) and value for value in raw):
        return ()
    return tuple(sorted(raw))


@dataclass
class Qwen35SoupRoute:
    """Run-local capture state for the single evidenced route."""

    spec: TaskSpec
    capture_root: Path
    expected_base_model: str
    expected_target_modules: tuple[str, ...]
    expected_tensor_pair_count: int
    scheduled_eval_count: int = 0
    record_steps: dict[int, int] = field(default_factory=dict)
    record_losses: dict[int, float] = field(default_factory=dict)
    capture_errors: dict[int, str] = field(default_factory=dict)

    def record_dir(self, ordinal: int) -> Path:
        return self.capture_root / f"record-{ordinal}"


def eligible_qwen35_soup_route(
    spec: TaskSpec,
    model: Any,
    *,
    strategy: str,
    n_gpus: int,
    capture_root: str | os.PathLike[str],
) -> Qwen35SoupRoute | None:
    """Return the override route only for the exact executed configuration."""
    if (
        strategy != "lora"
        or n_gpus != 1
        or spec.task_type != "InstructTextTask"
        or spec.instruct is None
        or spec.instruct.output is None
        or spec.use_kl
        or not _supported_model_route(spec)
        or not _is_exact_qwen35_08b(model)
    ):
        return None
    config = _default_peft_config(model)
    if config is None:
        return None
    try:
        rank = int(_config_value(config, "r"))
        alpha = int(_config_value(config, "lora_alpha"))
        dropout = float(_config_value(config, "lora_dropout"))
    except (TypeError, ValueError, OverflowError):
        return None
    base_model = _config_value(config, "base_model_name_or_path")
    targets = _target_modules(config)
    if (
        rank != EXACT_SOURCE_RANK
        or alpha != EXACT_SOURCE_ALPHA
        or not math.isclose(dropout, EXACT_SOURCE_DROPOUT, abs_tol=0.0, rel_tol=0.0)
        or not isinstance(base_model, str)
        or base_model != spec.cached_model_dir
        or targets != EXACT_TARGET_MODULES
        or bool(_config_value(config, "use_rslora", False))
        or bool(_config_value(config, "use_dora", False))
        or bool(_config_value(config, "use_qalora", False))
    ):
        return None
    return Qwen35SoupRoute(
        spec=spec,
        capture_root=Path(capture_root),
        expected_base_model=base_model,
        expected_target_modules=targets,
        expected_tensor_pair_count=EXACT_TENSOR_PAIR_COUNT,
    )


def make_qwen35_soup_capture_callback(
    route: Qwen35SoupRoute,
    tokenizer: Any,
) -> Any:
    """Capture scheduled evaluation ordinals two and four when each is finite."""
    from transformers import TrainerCallback

    class Qwen35SoupCaptureCallback(TrainerCallback):
        def on_evaluate(
            self, args, state, control, metrics=None, **kwargs
        ):  # noqa: ANN001
            route.scheduled_eval_count += 1
            ordinal = route.scheduled_eval_count
            if ordinal not in _CAPTURE_ORDINALS:
                return control
            try:
                loss = float((metrics or {}).get("eval_loss"))
            except (TypeError, ValueError, OverflowError):
                route.capture_errors[ordinal] = "eval_loss_missing_or_nonfinite"
                return control
            if not math.isfinite(loss):
                route.capture_errors[ordinal] = "eval_loss_missing_or_nonfinite"
                return control
            model = kwargs.get("model")
            if model is None:
                route.capture_errors[ordinal] = "model_missing"
                return control
            try:
                from forge.tasks.common import save_adapter

                save_adapter(model, tokenizer, str(route.record_dir(ordinal)))
                route.record_steps[ordinal] = int(state.global_step)
                route.record_losses[ordinal] = loss
            except Exception as exc:  # local evidence capture must not cost the run
                route.capture_errors[ordinal] = f"{type(exc).__name__}: {exc}"
                try:
                    from forge import telemetry

                    telemetry.event(
                        "qwen35_soup_capture_failed",
                        release_classification=RELEASE_CLASSIFICATION,
                        ordinal=ordinal,
                        step=int(getattr(state, "global_step", 0) or 0),
                        error=route.capture_errors[ordinal],
                    )
                except Exception:
                    pass
            return control

    return Qwen35SoupCaptureCallback()


def _read_adapter_config(path: Path) -> tuple[dict[str, Any], bytes]:
    config_path = path / _CONFIG_FILE
    _require(
        config_path.is_file() and not config_path.is_symlink(),
        "adapter config missing or linked",
    )
    raw = config_path.read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SoupCompatibilityError(f"invalid adapter config: {exc}") from exc
    _require(isinstance(config, dict), "adapter config root is not an object")
    return config, raw


def _validate_config(
    config: dict[str, Any],
    *,
    expected_base_model: str,
    expected_target_modules: tuple[str, ...],
) -> None:
    _require(config.get("peft_type") == "LORA", "adapter is not LoRA")
    _require(config.get("task_type") == "CAUSAL_LM", "adapter task type drift")
    _require(
        config.get("base_model_name_or_path") == expected_base_model,
        "base model identity drift",
    )
    _require(config.get("r") == EXACT_SOURCE_RANK, "source rank drift")
    _require(config.get("lora_alpha") == EXACT_SOURCE_ALPHA, "source alpha drift")
    _require(config.get("lora_dropout") == EXACT_SOURCE_DROPOUT, "source dropout drift")
    _require(config.get("bias") == "none", "adapter bias is unsupported")
    _require(config.get("fan_in_fan_out") is False, "fan_in_fan_out is unsupported")
    _require(config.get("use_rslora") is False, "RS-LoRA is unsupported")
    _require(config.get("use_dora") is False, "DoRA is unsupported")
    _require(config.get("use_qalora") is False, "QALoRA is unsupported")
    _require(config.get("modules_to_save") is None, "modules_to_save is unsupported")
    _require(config.get("rank_pattern") == {}, "rank_pattern is unsupported")
    _require(config.get("alpha_pattern") == {}, "alpha_pattern is unsupported")
    targets = config.get("target_modules")
    _require(isinstance(targets, list), "target_modules is not a list")
    _require(
        tuple(sorted(targets)) == expected_target_modules,
        "target module identity drift",
    )


def _tensor_pairs(tensors: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for key in tensors:
        if key.endswith(_A_SUFFIX):
            stem, factor = key[: -len(_A_SUFFIX)], "A"
        elif key.endswith(_B_SUFFIX):
            stem, factor = key[: -len(_B_SUFFIX)], "B"
        else:
            raise SoupCompatibilityError(f"non-LoRA tensor is unsupported: {key}")
        _require(
            stem and factor not in pairs.setdefault(stem, {}),
            f"duplicate LoRA factor: {key}",
        )
        pairs[stem][factor] = key
    _require(bool(pairs), "adapter has no LoRA tensors")
    result: dict[str, tuple[str, str]] = {}
    for stem, factors in pairs.items():
        _require(set(factors) == {"A", "B"}, f"incomplete LoRA pair: {stem}")
        result[stem] = (factors["A"], factors["B"])
    return result


def _load_tensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    model_path = path / _MODEL_FILE
    _require(
        model_path.is_file() and not model_path.is_symlink(),
        "adapter weights missing or linked",
    )
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    _require(metadata == {"format": "pt"}, "safetensors metadata drift")
    tensors = load_file(str(model_path), device="cpu")
    return dict(tensors), dict(metadata)


def _validate_tensor_contract(
    tensors: Mapping[str, Any],
    *,
    expected_target_modules: tuple[str, ...],
    expected_tensor_pair_count: int,
) -> dict[str, tuple[str, str]]:
    import torch

    pairs = _tensor_pairs(tensors)
    _require(
        len(pairs) == expected_tensor_pair_count,
        "LoRA tensor-pair count drift",
    )
    observed_targets: set[str] = set()
    for stem, (a_key, b_key) in pairs.items():
        a, b = tensors[a_key], tensors[b_key]
        _require(
            a.dtype == torch.float32 and b.dtype == torch.float32,
            f"non-F32 factors: {stem}",
        )
        _require(a.ndim == 2 and b.ndim == 2, f"non-matrix LoRA factors: {stem}")
        _require(a.shape[0] == EXACT_SOURCE_RANK, f"A rank drift: {stem}")
        _require(b.shape[1] == EXACT_SOURCE_RANK, f"B rank drift: {stem}")
        _require(a.shape[0] == b.shape[1], f"A/B inner-shape drift: {stem}")
        _require(
            bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            f"nonfinite factors: {stem}",
        )
        observed_targets.add(stem.rsplit(".", 1)[-1])
    _require(
        observed_targets == set(expected_target_modules),
        "configured/tensor target modules drift",
    )
    return pairs


def _materialize_fixed_mean(
    record4: Path,
    record2: Path,
    stage: Path,
    route: Qwen35SoupRoute,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    config4, raw4 = _read_adapter_config(record4)
    config2, raw2 = _read_adapter_config(record2)
    _require(
        raw4 == raw2 and config4 == config2, "record-4/record-2 config incompatibility"
    )
    _validate_config(
        config4,
        expected_base_model=route.expected_base_model,
        expected_target_modules=route.expected_target_modules,
    )
    tensors4, metadata4 = _load_tensors(record4)
    tensors2, metadata2 = _load_tensors(record2)
    _require(metadata4 == metadata2, "record safetensors metadata incompatibility")
    _require(set(tensors4) == set(tensors2), "record tensor-key incompatibility")
    pairs4 = _validate_tensor_contract(
        tensors4,
        expected_target_modules=route.expected_target_modules,
        expected_tensor_pair_count=route.expected_tensor_pair_count,
    )
    pairs2 = _validate_tensor_contract(
        tensors2,
        expected_target_modules=route.expected_target_modules,
        expected_tensor_pair_count=route.expected_tensor_pair_count,
    )
    _require(pairs4 == pairs2, "record LoRA pair incompatibility")
    for key in tensors4:
        _require(
            tensors4[key].dtype == tensors2[key].dtype
            and tuple(tensors4[key].shape) == tuple(tensors2[key].shape),
            f"record tensor shape/dtype incompatibility: {key}",
        )

    output: dict[str, Any] = {}
    for stem in sorted(pairs4):
        a_key, b_key = pairs4[stem]
        output[a_key] = torch.cat(
            (tensors4[a_key], tensors2[a_key]), dim=0
        ).contiguous()
        output[b_key] = torch.cat(
            (tensors4[b_key], tensors2[b_key]), dim=1
        ).contiguous()
        _require(
            bool(torch.isfinite(output[a_key]).all()), f"nonfinite output A: {stem}"
        )
        _require(
            bool(torch.isfinite(output[b_key]).all()), f"nonfinite output B: {stem}"
        )

    output_config = dict(config4)
    output_config["r"] = EXACT_SOURCE_RANK * 2
    model_tmp = stage / (_MODEL_FILE + ".tmp")
    config_tmp = stage / (_CONFIG_FILE + ".tmp")
    save_file(output, str(model_tmp), metadata=metadata4)
    config_tmp.write_bytes(_canonical_json(output_config))
    os.replace(model_tmp, stage / _MODEL_FILE)
    os.replace(config_tmp, stage / _CONFIG_FILE)

    reloaded, _ = _load_tensors(stage)
    _require(set(reloaded) == set(output), "reloaded soup tensor-key drift")
    for key in output:
        _require(
            torch.equal(reloaded[key], output[key]),
            f"reloaded soup tensor drift: {key}",
        )
    loaded_config, _ = _read_adapter_config(stage)
    _require(loaded_config == output_config, "reloaded soup config drift")
    return {
        "record4_adapter_sha256": _sha256(record4 / _MODEL_FILE),
        "record2_adapter_sha256": _sha256(record2 / _MODEL_FILE),
        "soup_adapter_sha256": _sha256(stage / _MODEL_FILE),
        "record4_step": route.record_steps[4],
        "record2_step": route.record_steps[2],
        "source_rank": EXACT_SOURCE_RANK,
        "output_rank": EXACT_SOURCE_RANK * 2,
        "lora_alpha": EXACT_SOURCE_ALPHA,
        "member_weights": ["1/2", "1/2"],
        "tensor_pair_count": route.expected_tensor_pair_count,
    }


def _require_regular_tree(path: Path) -> None:
    _require(
        path.is_dir() and not path.is_symlink(),
        f"artifact root is not a directory: {path}",
    )
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in (*directories, *files):
            child = root_path / name
            _require(not child.is_symlink(), f"artifact contains symlink: {child}")
        for name in files:
            child = root_path / name
            _require(child.is_file(), f"artifact contains non-regular file: {child}")


def _committed_output_matches(
    output: Path,
    *,
    adapter_sha256: str,
    marker_sha256: str,
) -> bool:
    """Recognize a completed promotion if post-swap cleanup reports an error."""
    try:
        from forge.tasks import common

        return (
            common._has_ready_marker(str(output))
            and (output / _MODEL_FILE).is_file()
            and not (output / _MODEL_FILE).is_symlink()
            and _sha256(output / _MODEL_FILE) == adapter_sha256
            and (output / _MARKER_FILE).is_file()
            and not (output / _MARKER_FILE).is_symlink()
            and _sha256(output / _MARKER_FILE) == marker_sha256
        )
    except Exception:
        return False


def _runtime_load_inference_smoke(model: Any, tokenizer: Any, stage: Path) -> None:
    """Load the staged adapter beside ``default`` and require finite logits."""
    import torch
    from peft import PeftConfig

    PeftConfig.from_pretrained(str(stage), local_files_only=True)
    adapter_name = "ceo_exploratory_override_smoke"
    configs = getattr(model, "peft_config", {})
    _require("default" in configs, "default adapter is unavailable for smoke rollback")
    if adapter_name in configs:
        model.delete_adapter(adapter_name)
    was_training = bool(getattr(model, "training", False))
    try:
        model.load_adapter(str(stage), adapter_name=adapter_name, is_trainable=False)
        model.set_adapter(adapter_name)
        model.eval()
        encoded = tokenizer("SN56 compatibility smoke", return_tensors="pt")
        device = next(model.parameters()).device
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        elif isinstance(encoded, Mapping):
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        with torch.inference_mode():
            result = model(**encoded)
        logits = getattr(result, "logits", None)
        _require(logits is not None and logits.numel() > 0, "smoke produced no logits")
        _require(bool(torch.isfinite(logits).all()), "smoke produced nonfinite logits")
    finally:
        try:
            model.set_adapter("default")
        finally:
            try:
                model.delete_adapter(adapter_name)
            except Exception:
                pass
        if was_training:
            model.train()


def apply_qwen35_soup_override(
    route: Qwen35SoupRoute | None,
    *,
    tracker: Any,
    model: Any,
    tokenizer: Any,
    smoke: Callable[[Any, Any, Path], None] | None = None,
) -> bool:
    """Atomically promote the fixed soup, or leave record four byte-identical."""
    if route is None:
        return False
    from forge import telemetry
    from forge.tasks import common

    output = Path(route.spec.output_dir)
    stage = Path(str(output) + ".ceo-override.tmp")
    try:
        _require(
            route.scheduled_eval_count >= 4, "fewer than four scheduled evaluations"
        )
        _require(
            set(route.record_steps) >= {2, 4}, "record-2 or record-4 capture missing"
        )
        _require(not ({2, 4} & set(route.capture_errors)), "record capture failed")
        _require(
            getattr(tracker, "persisted_best_step", None) == route.record_steps[4],
            "finite evaluation ordinal four is not the persisted best",
        )
        persisted = float(getattr(tracker, "persisted_best", math.nan))
        _require(
            math.isfinite(persisted) and persisted == route.record_losses[4],
            "record-four loss does not equal the persisted best",
        )
        record4, record2 = route.record_dir(4), route.record_dir(2)
        _require_regular_tree(record4)
        _require_regular_tree(record2)
        _require_regular_tree(output)
        _require(
            _sha256(output / _MODEL_FILE) == _sha256(record4 / _MODEL_FILE)
            and _sha256(output / _CONFIG_FILE) == _sha256(record4 / _CONFIG_FILE),
            "validator-visible output is not record four",
        )

        common._rmtree(str(stage))
        shutil.copytree(record4, stage, symlinks=False)
        for marker in (common._READY_MARKER, common._READY_MARKER_TMP):
            try:
                (stage / marker).unlink()
            except FileNotFoundError:
                pass
        materialization = _materialize_fixed_mean(record4, record2, stage, route)
        marker_payload = {
            "schema_version": 1,
            "kind": "sn56_qwen35_fixed_effective_delta_release_override",
            "release_classification": RELEASE_CLASSIFICATION,
            "formal_verdict_preserved": FORMAL_VERDICT,
            "formal_decision_file_sha256": FORMAL_DECISION_FILE_SHA256,
            "formal_decision_result_sha256": FORMAL_DECISION_RESULT_SHA256,
            "method": METHOD,
            "model_family": EXACT_MODEL_ID,
            "model_argument": route.spec.model,
            "model_argument_anonymized": route.spec.model != EXACT_MODEL_ID,
            "task_type": "InstructTextTask",
            "search_or_threshold_change": False,
            "materialization": materialization,
        }
        (stage / _MARKER_FILE).write_bytes(_canonical_json(marker_payload))
        (smoke or _runtime_load_inference_smoke)(model, tokenizer, stage)
        telemetry.event(
            "qwen35_soup_override_promoted",
            release_classification=RELEASE_CLASSIFICATION,
            formal_verdict=FORMAL_VERDICT,
            method=METHOD,
            record2_step=route.record_steps[2],
            record4_step=route.record_steps[4],
        )
        telemetry.write_into(str(stage))
        common._validate_staged_artifact(str(stage))
        common._fsync_tree(str(stage))
        common._write_ready_marker(str(stage))
        adapter_sha256 = _sha256(stage / _MODEL_FILE)
        marker_sha256 = _sha256(stage / _MARKER_FILE)
        try:
            common._promote_staged_dir(str(stage), str(output))
        except Exception:
            if _committed_output_matches(
                output,
                adapter_sha256=adapter_sha256,
                marker_sha256=marker_sha256,
            ):
                return True
            raise
        return True
    except Exception as exc:
        try:
            common._rmtree(str(stage))
        except Exception:
            pass
        try:
            telemetry.event(
                "qwen35_soup_override_fallback",
                release_classification=RELEASE_CLASSIFICATION,
                formal_verdict=FORMAL_VERDICT,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        try:
            telemetry.write_into(str(output))
        except Exception:
            pass
        return False
