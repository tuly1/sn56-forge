from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.data.schema import InstructColumns
from forge.tuning import granite41_epoch_cap as policy


def _spec(
    *,
    model: str = policy.EXACT_MODEL_ID,
    use_kl: bool = False,
    baseline_stats_path: str | None = None,
    cached_model_dir: str | None = None,
):
    return SimpleNamespace(
        model=model,
        cached_model_dir=cached_model_dir
        or f"/cache/models/{model.replace('/', '--')}",
        baseline_stats_path=baseline_stats_path,
        task_type="InstructTextTask",
        instruct=InstructColumns("instruction", "output", "input"),
        use_kl=use_kl,
    )


class _Model:
    def __init__(
        self,
        *,
        hidden_size: int = 2560,
        rank: int = 32,
        base_model_name_or_path: str = policy.EXACT_BASE_MODEL,
    ):
        self._base = SimpleNamespace(
            config=SimpleNamespace(
                model_type="granite",
                architectures=["GraniteForCausalLM"],
                hidden_size=hidden_size,
                intermediate_size=8192,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=8,
            )
        )
        self.peft_config = {
            "default": SimpleNamespace(
                r=rank,
                lora_alpha=64,
                lora_dropout=0.05,
                base_model_name_or_path=base_model_name_or_path,
                target_modules=set(policy.EXACT_TARGET_MODULES),
                use_rslora=False,
                use_dora=False,
                use_qalora=False,
                rank_pattern={},
                alpha_pattern={},
            )
        }

    def get_base_model(self):
        return self._base


def test_exact_route_caps_only_native_epochs_above_one() -> None:
    spec = _spec()
    model = _Model()
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy="lora", n_gpus=1, native_epochs=3.76
    ) == 1.0
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy="lora", n_gpus=1, native_epochs=0.75
    ) == 0.75
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy="lora", n_gpus=1, native_epochs=1.0
    ) == 1.0


def test_official_anonymous_route_preserves_exact_cap() -> None:
    alias = "5e54c9ada1e3ee68"
    cached = f"/cache/models/{alias}"
    spec = _spec(
        model=alias,
        cached_model_dir=cached,
        baseline_stats_path="/cache/baseline_stats_task.json",
    )
    model = _Model(base_model_name_or_path=cached)
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy="lora", n_gpus=1, native_epochs=3.76
    ) == 1.0


@pytest.mark.parametrize(
    "spec",
    [
        _spec(model="5e54c9ada1e3ee68"),
        _spec(
            model="5E54C9ADA1E3EE68",
            baseline_stats_path="/cache/baseline_stats_task.json",
        ),
        _spec(
            model="5e54c9ada1e3ee68",
            baseline_stats_path="/cache/baseline_stats_task.json",
            cached_model_dir="/cache/models/ibm-granite--granite-4.1-3b",
        ),
    ],
)
def test_anonymous_route_fails_closed_on_contract_drift(spec) -> None:
    model = _Model(base_model_name_or_path=str(spec.cached_model_dir))
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy="lora", n_gpus=1, native_epochs=3.76
    ) is None


@pytest.mark.parametrize("native", [0.0, -1.0, math.inf, math.nan])
def test_exact_route_rejects_invalid_native_epochs(native: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        policy.cap_granite41_production_epochs(
            _spec(), _Model(), strategy="lora", n_gpus=1, native_epochs=native
        )


@pytest.mark.parametrize(
    ("spec", "model", "strategy", "n_gpus"),
    [
        (_spec(model="LiquidAI/LFM2.5-1.2B-Instruct"), _Model(), "lora", 1),
        (_spec(use_kl=True), _Model(), "lora", 1),
        (_spec(), _Model(), "full", 1),
        (_spec(), _Model(), "lora", 2),
        (_spec(), _Model(hidden_size=4096), "lora", 1),
        (_spec(), _Model(rank=64), "lora", 1),
    ],
)
def test_nonmatching_routes_are_unchanged(spec, model, strategy, n_gpus) -> None:
    assert policy.cap_granite41_production_epochs(
        spec, model, strategy=strategy, n_gpus=n_gpus, native_epochs=3.76
    ) is None


def test_cap_is_applied_before_training_arguments_and_keeps_native_cadence() -> None:
    source = (
        Path(__file__).parents[1] / "forge" / "tasks" / "instruct.py"
    ).read_text(encoding="utf-8")
    assign = source.index('kwargs["num_train_epochs"] = epochs')
    cap = source.index("cap_granite41_production_epochs(", assign)
    training_args = source.index("args = TrainingArguments(", cap)
    assert assign < cap < training_args

    # HelpSteer2: the existing production split and b4/ga4 geometry produce
    # 424 steps/epoch and eval every quarter epoch.  Capping 3.76 to 1.0
    # therefore yields the measured candidate scheduler horizon without
    # changing the existing evaluation cadence.
    train_rows = 6784
    effective_batch = 16
    steps_per_epoch = train_rows // effective_batch
    eval_steps = steps_per_epoch // 4
    applied = policy.cap_granite41_production_epochs(
        _spec(), _Model(), strategy="lora", n_gpus=1, native_epochs=3.76
    )
    assert steps_per_epoch == 424
    assert eval_steps == 106
    assert math.ceil(steps_per_epoch * applied) == 424


def test_patch_does_not_add_experiment_capture_or_seed_controls() -> None:
    source = (
        Path(__file__).parents[1] / "forge" / "tuning" / "granite41_epoch_cap.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "FORGE_",
        "set_seed",
        "factor_sha256",
        "capture_root",
        "record_best",
        "fixed_horizon",
    ):
        assert forbidden not in source
