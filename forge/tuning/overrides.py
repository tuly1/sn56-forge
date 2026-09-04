"""Study-only recipe overrides (``FORGE_RECIPE_OVERRIDES_JSON``).

The SFT handler reads the environment variable exactly once at task setup.

* Unset: the hook is inert.  Every helper returns its input unchanged (the
  very same object), nothing is written to the flight recorder, and the run is
  byte-identical to the pinned trainer.
* Present: a strict JSON object of whitelisted *recipe* knobs.  Any parse or
  validation error disables the whole payload (nothing is applied) and is
  recorded in ``forge_run.json`` so a malformed study cell can never train a
  silently different recipe.

The hook deliberately cannot reach the survival machinery: batch geometry,
gradient checkpointing, the admission ladder, the wall-budget planner and the
artifact truth states are not overridable.  A payload naming any such key is
rejected as a whole.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

ENV_NAME = "FORGE_RECIPE_OVERRIDES_JSON"
_MAX_PAYLOAD_BYTES = 8192
_DEFAULT_NEFTUNE_ALPHA = 5.0
SCHEDULERS = (
    "cosine_with_min_lr",
    "cosine",
    "linear",
    "constant",
    "constant_with_warmup",
)
LONG_ROW_POLICIES = ("drop", "truncate")
# TrainPlan fields a payload may replace one-for-one.
_PLAN_FIELDS = (
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "num_epochs",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler",
)


def _number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not (low <= number <= high):
        raise ValueError(f"{name} must be within [{low}, {high}]")
    return number


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not (low <= value <= high):
        raise ValueError(f"{name} must be within [{low}, {high}]")
    return int(value)


def _choice(value: Any, choices: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")
    return value


def _neftune(value: Any) -> float | None:
    """``null``/``0`` switch NEFTune off; a positive alpha sets it."""
    if value is None:
        return None
    alpha = _number(value, 0.0, 50.0, "neftune_alpha")
    return None if alpha == 0.0 else alpha


_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "learning_rate": lambda v: _number(v, 1e-7, 1e-2, "learning_rate"),
    "lora_r": lambda v: _integer(v, 1, 1024, "lora_r"),
    "lora_alpha": lambda v: _integer(v, 1, 8192, "lora_alpha"),
    "lora_dropout": lambda v: _number(v, 0.0, 0.9, "lora_dropout"),
    "num_epochs": lambda v: _number(v, 0.05, 8.0, "num_epochs"),
    "epochs_cap": lambda v: _number(v, 0.05, 8.0, "epochs_cap"),
    "warmup_ratio": lambda v: _number(v, 0.0, 0.5, "warmup_ratio"),
    "weight_decay": lambda v: _number(v, 0.0, 1.0, "weight_decay"),
    "lr_scheduler": lambda v: _choice(v, SCHEDULERS, "lr_scheduler"),
    "min_lr_rate": lambda v: _number(v, 0.0, 1.0, "min_lr_rate"),
    "neftune_alpha": _neftune,
    "max_seq_len": lambda v: _integer(v, 64, 8192, "max_seq_len"),
    "long_rows": lambda v: _choice(v, LONG_ROW_POLICIES, "long_rows"),
}
ALLOWED_KEYS = tuple(sorted(_VALIDATORS))


@dataclass(frozen=True)
class RecipeOverrides:
    """What the environment asked for, and whether it was accepted."""

    source: str | None = None
    error: str | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        """The environment variable existed (accepted or not)."""
        return self.source is not None

    @property
    def active(self) -> bool:
        """At least one validated override will be applied."""
        return bool(self.values)

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "accepted": self.present and self.error is None,
            "error": self.error,
            "values": dict(self.values),
        }


def parse_recipe_overrides(raw: str) -> dict[str, Any]:
    """Validate one payload; raise ``ValueError`` on anything unexpected."""
    if not isinstance(raw, str):
        raise ValueError("payload must be a string")
    if len(raw.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {_MAX_PAYLOAD_BYTES} bytes")
    try:
        payload = json.loads(raw)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"invalid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    unknown = sorted(set(payload) - set(_VALIDATORS))
    if unknown:
        raise ValueError(f"unknown override keys: {unknown}")
    values: dict[str, Any] = {}
    for key in sorted(payload):
        values[key] = _VALIDATORS[key](payload[key])
    return values


def load_recipe_overrides(environ: Mapping[str, str] | None = None) -> RecipeOverrides:
    """Read the payload once.  Never raises: a bad payload yields an inert
    record carrying the error."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_NAME)
    if raw is None:
        return RecipeOverrides()
    try:
        values = parse_recipe_overrides(raw)
    except ValueError as exc:
        return RecipeOverrides(source=ENV_NAME, error=f"{type(exc).__name__}: {exc}")
    return RecipeOverrides(source=ENV_NAME, values=values)


def apply_plan_overrides(plan: Any, recipe: RecipeOverrides) -> Any:
    """Return the plan with the recipe's TrainPlan fields replaced.

    ``max_seq_len`` may only lower the plan's cap (the evaluator cap is the
    ceiling); ``epochs_cap`` lowers ``num_epochs`` when the plan (or the
    ``num_epochs`` override) exceeds it.  Inert payloads return ``plan`` itself.
    """
    if not recipe.active:
        return plan
    changes: dict[str, Any] = {
        key: recipe.values[key] for key in _PLAN_FIELDS if key in recipe.values
    }
    if "max_seq_len" in recipe.values:
        changes["max_seq_len"] = min(int(recipe.values["max_seq_len"]), int(plan.max_seq_len))
    if "epochs_cap" in recipe.values:
        cap = float(recipe.values["epochs_cap"])
        epochs = float(changes.get("num_epochs", plan.num_epochs))
        if epochs > cap:
            changes["num_epochs"] = cap
    changes = {
        key: value for key, value in changes.items() if getattr(plan, key) != value
    }
    if not changes:
        return plan
    return replace(plan, **changes)


def plan_override_diff(before: Any, after: Any) -> dict[str, list[Any]]:
    """``{field: [before, after]}`` for every TrainPlan field that changed."""
    fields = tuple(getattr(type(before), "__dataclass_fields__", {}))
    return {
        name: [getattr(before, name), getattr(after, name)]
        for name in fields
        if getattr(before, name) != getattr(after, name)
    }


def neftune_alpha(is_kl: bool, recipe: RecipeOverrides | None) -> float | None:
    """NEFTune alpha for the SFT path: always off on KL tasks (the adapter-
    disabled base reference must not see noise); otherwise the override or
    the production default."""
    if is_kl:
        return None
    if recipe is not None and recipe.active and "neftune_alpha" in recipe.values:
        return recipe.values["neftune_alpha"]
    return _DEFAULT_NEFTUNE_ALPHA


def apply_training_kwargs_overrides(
    kwargs: dict[str, Any], recipe: RecipeOverrides | None
) -> dict[str, Any]:
    """Post-process ``build_training_kwargs`` output; inert payloads return
    the same dict object."""
    if recipe is None or not recipe.active:
        return kwargs
    if "min_lr_rate" in recipe.values and kwargs.get("lr_scheduler_type") == "cosine_with_min_lr":
        out = dict(kwargs)
        out["lr_scheduler_kwargs"] = {"min_lr_rate": float(recipe.values["min_lr_rate"])}
        return out
    return kwargs


def cap_epochs(epochs: float | None, recipe: RecipeOverrides | None) -> float | None:
    """Cap a time-aware epoch plan at ``epochs_cap``; ``None`` passes through."""
    if epochs is None or recipe is None or not recipe.active:
        return epochs
    cap = recipe.values.get("epochs_cap")
    if cap is None:
        return epochs
    return min(float(epochs), float(cap))


def long_rows_policy(recipe: RecipeOverrides | None) -> str:
    """How rows longer than the sequence cap are handled at tokenization."""
    if recipe is None or not recipe.active:
        return "drop"
    return str(recipe.values.get("long_rows", "drop"))
