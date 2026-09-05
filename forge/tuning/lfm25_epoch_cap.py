"""Production epoch cap for the evidenced LFM2.5-2.6B instruct LoRA route.

This module only recognizes the exact model and adapter geometry used by the
release study.  It does not select geometry, seed training, or alter the
time-aware probe.  The caller supplies the probe-derived epoch count and, for
the admitted route, receives the production policy ``min(native, 1.0)``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from forge.data.schema import TaskSpec


EXACT_MODEL_ID = "LiquidAI/LFM2.5-2.6B"
EXACT_BASE_MODEL = "/cache/models/LiquidAI--LFM2.5-2.6B"
EXACT_SOURCE_RANK = 32
EXACT_SOURCE_ALPHA = 64
EXACT_SOURCE_DROPOUT = 0.05
EXACT_TARGET_MODULES = tuple(
    sorted({"in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "w1", "w2", "w3"})
)


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _base_config(model: Any) -> Any:
    candidate = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidate = getter()
        except Exception:
            return None
    return getattr(candidate, "config", getattr(model, "config", None))


def _default_peft_config(model: Any) -> Any:
    configs = getattr(model, "peft_config", None)
    if isinstance(configs, Mapping):
        return configs.get("default")
    return None


def _targets(config: Any) -> tuple[str, ...]:
    raw = _value(config, "target_modules")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    if not all(isinstance(item, str) and item for item in raw):
        return ()
    return tuple(sorted(raw))


def eligible_lfm25_production_epoch_cap(
    spec: TaskSpec, model: Any, *, strategy: str, n_gpus: int
) -> bool:
    """Recognize only the measured LFM2.5-2.6B production LoRA route."""
    if (
        strategy != "lora"
        or n_gpus != 1
        or spec.model != EXACT_MODEL_ID
        or str(spec.cached_model_dir) != EXACT_BASE_MODEL
        or spec.task_type != "InstructTextTask"
        or spec.instruct is None
        or spec.instruct.output is None
        or spec.use_kl
    ):
        return False

    base = _base_config(model)
    if not (
        str(_value(base, "model_type", "") or "").lower() == "lfm2"
        and tuple(_value(base, "architectures", ()) or ()) == ("Lfm2ForCausalLM",)
        and _value(base, "hidden_size") == 2048
        and _value(base, "intermediate_size") == 10752
        and _value(base, "num_hidden_layers") == 30
        and _value(base, "num_attention_heads") == 32
        and _value(base, "num_key_value_heads") == 8
    ):
        return False

    config = _default_peft_config(model)
    try:
        return bool(
            config is not None
            and int(_value(config, "r")) == EXACT_SOURCE_RANK
            and int(_value(config, "lora_alpha")) == EXACT_SOURCE_ALPHA
            and math.isclose(
                float(_value(config, "lora_dropout")),
                EXACT_SOURCE_DROPOUT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and _value(config, "base_model_name_or_path") == spec.cached_model_dir
            and _targets(config) == EXACT_TARGET_MODULES
            and not bool(_value(config, "use_rslora", False))
            and not bool(_value(config, "use_dora", False))
            and not bool(_value(config, "use_qalora", False))
            and not bool(_value(config, "rank_pattern", {}))
            and not bool(_value(config, "alpha_pattern", {}))
        )
    except (TypeError, ValueError, OverflowError):
        return False


def cap_lfm25_production_epochs(
    spec: TaskSpec,
    model: Any,
    *,
    strategy: str,
    n_gpus: int,
    native_epochs: float,
) -> float | None:
    """Return the capped epochs for the exact route, otherwise ``None``."""
    if not eligible_lfm25_production_epoch_cap(
        spec, model, strategy=strategy, n_gpus=n_gpus
    ):
        return None
    try:
        observed = float(native_epochs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LFM2.5 production cap requires finite positive native epochs") from exc
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError("LFM2.5 production cap requires finite positive native epochs")
    return min(observed, 1.0)
