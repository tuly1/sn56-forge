from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from forge.data.schema import InstructColumns
from forge.tuning import qwen35_soup as soup


def _spec(
    tmp_path: Path,
    *,
    model: str = soup.EXACT_MODEL_ID,
    use_kl: bool = False,
    baseline_stats_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_id="qwen-soup-test",
        task_type="InstructTextTask",
        model=model,
        instruct=InstructColumns("instruction", "output", "input"),
        use_kl=use_kl,
        baseline_stats_path=baseline_stats_path,
        cached_model_dir=f"/cache/models/{model.replace('/', '--')}",
        output_dir=str(tmp_path / "output"),
    )


def _qwen35_config(*, hidden_size: int = 1024, intermediate_size: int = 3584):
    dimensions = dict(soup._EXACT_TEXT_DIMENSIONS)
    dimensions.update(hidden_size=hidden_size, intermediate_size=intermediate_size)
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        layer_types=list(soup._EXACT_LAYER_TYPES),
        **dimensions,
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=list(soup._EXACT_OUTER_ARCHITECTURES),
        text_config=text_config,
    )


class _EligibleModel:
    def __init__(
        self,
        *,
        base_model: str = soup.EXACT_BASE_MODEL,
        hidden_size: int = 1024,
        intermediate_size: int = 3584,
    ) -> None:
        self._base = SimpleNamespace(
            config=_qwen35_config(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        )
        self.peft_config = {
            "default": SimpleNamespace(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                base_model_name_or_path=base_model,
                target_modules=set(soup.EXACT_TARGET_MODULES),
                use_rslora=False,
                use_dora=False,
                use_qalora=False,
            )
        }

    def get_base_model(self):
        return self._base


def _config(*, rank: int = 32, targets: list[str] | None = None) -> dict:
    return {
        "alpha_pattern": {},
        "base_model_name_or_path": "/cache/models/Qwen--Qwen3.5-0.8B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": targets or ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def _tensors(offset: float = 0.0) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for index, target in enumerate(("q_proj", "v_proj"), start=1):
        stem = f"base_model.model.layers.0.{target}"
        a = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
        b = torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)
        values[stem + ".lora_A.weight"] = a / 1000 + offset + index / 100
        values[stem + ".lora_B.weight"] = b / 1000 - offset - index / 100
    return values


def _artifact(path: Path, *, offset: float = 0.0, config: dict | None = None) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(config or _config(), sort_keys=True), encoding="utf-8"
    )
    save_file(
        _tensors(offset),
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


def _route(tmp_path: Path) -> soup.Qwen35SoupRoute:
    route = soup.Qwen35SoupRoute(
        spec=_spec(tmp_path),
        capture_root=tmp_path / "captures",
        expected_base_model=soup.EXACT_BASE_MODEL,
        expected_target_modules=("q_proj", "v_proj"),
        expected_tensor_pair_count=2,
        scheduled_eval_count=4,
        record_steps={2: 332, 4: 664},
        record_losses={2: 1.72, 4: 1.70},
    )
    _artifact(route.record_dir(2), offset=0.02)
    _artifact(route.record_dir(4), offset=0.04)
    _artifact(Path(route.spec.output_dir), offset=0.04)
    return route


def _tracker(*, step: int = 664, loss: float = 1.70):
    return SimpleNamespace(persisted_best_step=step, persisted_best=loss)


def _artifact_identity(path: Path) -> tuple[bytes, bytes]:
    return (
        (path / "adapter_config.json").read_bytes(),
        (path / "adapter_model.safetensors").read_bytes(),
    )


def test_route_is_exact_and_non_qwen_is_untouched(tmp_path: Path) -> None:
    model = _EligibleModel()
    route = soup.eligible_qwen35_soup_route(
        _spec(tmp_path),
        model,
        strategy="lora",
        n_gpus=1,
        capture_root=tmp_path / "captures",
    )
    assert route is not None
    assert route.expected_target_modules == soup.EXACT_TARGET_MODULES
    assert route.expected_tensor_pair_count == soup.EXACT_TENSOR_PAIR_COUNT

    anonymized = "a3f2c1098e7d6b54"
    anonymized_spec = _spec(
        tmp_path,
        model=anonymized,
        baseline_stats_path="/cache/baselines/task.json",
    )
    anonymized_route = soup.eligible_qwen35_soup_route(
        anonymized_spec,
        _EligibleModel(base_model=anonymized_spec.cached_model_dir),
        strategy="lora",
        n_gpus=1,
        capture_root=tmp_path / "anonymized",
    )
    assert anonymized_route is not None
    assert anonymized_route.expected_base_model == f"/cache/models/{anonymized}"

    # Public Qwen3.5-2B uses the same outer architecture but 2048/6144 text
    # dimensions; it must not enter the evidenced 0.8B-only route.
    assert (
        soup.eligible_qwen35_soup_route(
            anonymized_spec,
            _EligibleModel(
                base_model=anonymized_spec.cached_model_dir,
                hidden_size=2048,
                intermediate_size=6144,
            ),
            strategy="lora",
            n_gpus=1,
            capture_root=tmp_path / "qwen35-2b",
        )
        is None
    )
    assert (
        soup.eligible_qwen35_soup_route(
            _spec(tmp_path, model=anonymized),
            _EligibleModel(base_model=f"/cache/models/{anonymized}"),
            strategy="lora",
            n_gpus=1,
            capture_root=tmp_path / "anonymous-without-baseline",
        )
        is None
    )

    assert (
        soup.eligible_qwen35_soup_route(
            _spec(tmp_path, model="Qwen/Qwen3.5-2B"),
            model,
            strategy="lora",
            n_gpus=1,
            capture_root=tmp_path / "other",
        )
        is None
    )
    assert (
        soup.eligible_qwen35_soup_route(
            _spec(tmp_path, use_kl=True),
            model,
            strategy="lora",
            n_gpus=1,
            capture_root=tmp_path / "kl",
        )
        is None
    )
    assert (
        soup.eligible_qwen35_soup_route(
            _spec(tmp_path),
            model,
            strategy="lora",
            n_gpus=2,
            capture_root=tmp_path / "multi",
        )
        is None
    )


def test_capture_uses_absolute_scheduled_eval_ordinals(
    tmp_path: Path, monkeypatch
) -> None:
    route = _route(tmp_path)
    route.scheduled_eval_count = 0
    route.record_steps.clear()
    route.record_losses.clear()
    captured: list[tuple[int, str]] = []

    def fake_save(model, tokenizer, output):
        captured.append((model, output))

    monkeypatch.setattr("forge.tasks.common.save_adapter", fake_save)
    callback = soup.make_qwen35_soup_capture_callback(route, object())
    state = SimpleNamespace(global_step=0)
    control = object()
    for step, value in ((10, 2.0), (20, 1.9), (30, 1.8), (40, 1.7)):
        state.global_step = step
        callback.on_evaluate(
            None,
            state,
            control,
            metrics={} if value is None else {"eval_loss": value},
            model=17,
        )
    assert route.scheduled_eval_count == 4
    assert route.record_steps == {2: 20, 4: 40}
    assert route.record_losses == {2: 1.9, 4: 1.7}
    assert [Path(path).name for _, path in captured] == ["record-2", "record-4"]


@pytest.mark.parametrize(
    ("bad_ordinal", "bad_value"),
    ((2, None), (2, float("nan")), (4, None), (4, float("nan"))),
)
def test_missing_or_nonfinite_selected_ordinal_falls_back_without_shift(
    tmp_path: Path,
    monkeypatch,
    bad_ordinal: int,
    bad_value: float | None,
) -> None:
    route = _route(tmp_path)
    output = Path(route.spec.output_dir)
    before = _artifact_identity(output)
    route.scheduled_eval_count = 0
    route.record_steps.clear()
    route.record_losses.clear()

    monkeypatch.setattr("forge.tasks.common.save_adapter", lambda *args: None)
    callback = soup.make_qwen35_soup_capture_callback(route, object())
    state = SimpleNamespace(global_step=0)
    values: list[float | None] = [2.0, 1.9, 1.8, 1.7]
    values[bad_ordinal - 1] = bad_value
    for ordinal, value in enumerate(values, start=1):
        state.global_step = ordinal * 10
        callback.on_evaluate(
            None,
            state,
            object(),
            metrics={} if value is None else {"eval_loss": value},
            model=17,
        )

    assert route.scheduled_eval_count == 4
    assert route.capture_errors[bad_ordinal] == "eval_loss_missing_or_nonfinite"
    assert bad_ordinal not in route.record_steps
    assert not soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(),
        model=object(),
        tokenizer=object(),
        smoke=lambda *_: None,
    )
    assert _artifact_identity(output) == before


def test_fixed_r4_r2_effective_delta_promotes_atomically(tmp_path: Path) -> None:
    route = _route(tmp_path)
    record4_before = _artifact_identity(route.record_dir(4))
    output_before = _artifact_identity(Path(route.spec.output_dir))
    smoke_observations: list[int] = []

    def smoke(model, tokenizer, stage: Path) -> None:
        config = json.loads((stage / "adapter_config.json").read_text())
        smoke_observations.append(config["r"])

    assert soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(),
        model=object(),
        tokenizer=object(),
        smoke=smoke,
    )
    assert smoke_observations == [64]
    assert _artifact_identity(route.record_dir(4)) == record4_before
    assert _artifact_identity(Path(route.spec.output_dir)) != output_before
    marker = json.loads(
        (Path(route.spec.output_dir) / "CEO_EXPLORATORY_OVERRIDE.json").read_text()
    )
    assert marker["release_classification"] == "CEO_EXPLORATORY_OVERRIDE"
    assert marker["formal_verdict_preserved"] == soup.FORMAL_VERDICT
    assert marker["search_or_threshold_change"] is False

    out = load_file(str(Path(route.spec.output_dir) / "adapter_model.safetensors"))
    four = load_file(str(route.record_dir(4) / "adapter_model.safetensors"))
    two = load_file(str(route.record_dir(2) / "adapter_model.safetensors"))
    for stem in (
        "base_model.model.layers.0.q_proj",
        "base_model.model.layers.0.v_proj",
    ):
        a = out[stem + ".lora_A.weight"]
        b = out[stem + ".lora_B.weight"]
        assert a.shape[0] == b.shape[1] == 64
        probe = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float32)
        actual = b @ (a @ probe)
        expected = (
            2.0
            * four[stem + ".lora_B.weight"]
            @ (four[stem + ".lora_A.weight"] @ probe)
            + 2.0
            * two[stem + ".lora_B.weight"]
            @ (two[stem + ".lora_A.weight"] @ probe)
        ) / 2.0
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("failure", ["missing", "incompatible", "nonfinite", "smoke"])
def test_every_precommit_failure_preserves_record4(
    tmp_path: Path, failure: str
) -> None:
    route = _route(tmp_path)
    output = Path(route.spec.output_dir)
    before = _artifact_identity(output)
    if failure == "missing":
        (route.record_dir(2) / "adapter_model.safetensors").unlink()
    elif failure == "incompatible":
        bad = _config(targets=["q_proj"])
        (route.record_dir(2) / "adapter_config.json").write_text(
            json.dumps(bad, sort_keys=True), encoding="utf-8"
        )
    elif failure == "nonfinite":
        tensors = _tensors(0.02)
        tensors["base_model.model.layers.0.q_proj.lora_A.weight"][0, 0] = float("nan")
        save_file(
            tensors,
            str(route.record_dir(2) / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )

    def smoke(model, tokenizer, stage):
        if failure == "smoke":
            raise RuntimeError("injected load/inference smoke failure")

    assert not soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(),
        model=object(),
        tokenizer=object(),
        smoke=smoke,
    )
    assert _artifact_identity(output) == before
    assert not (output / "CEO_EXPLORATORY_OVERRIDE.json").exists()


def test_non_record4_best_preserves_existing_best(tmp_path: Path) -> None:
    route = _route(tmp_path)
    output = Path(route.spec.output_dir)
    before = _artifact_identity(output)
    assert not soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(step=500, loss=1.6),
        model=object(),
        tokenizer=object(),
        smoke=lambda *_: None,
    )
    assert _artifact_identity(output) == before


def test_post_commit_cleanup_error_reports_live_soup(
    tmp_path: Path, monkeypatch
) -> None:
    route = _route(tmp_path)
    output = Path(route.spec.output_dir)
    from forge.tasks import common

    original_promote = common._promote_staged_dir

    def promote_then_raise(stage: str, final: str) -> None:
        original_promote(stage, final)
        raise OSError("injected post-commit cleanup failure")

    monkeypatch.setattr(common, "_promote_staged_dir", promote_then_raise)
    assert soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(),
        model=object(),
        tokenizer=object(),
        smoke=lambda *_: None,
    )
    assert (output / "CEO_EXPLORATORY_OVERRIDE.json").is_file()
    assert json.loads((output / "adapter_config.json").read_text())["r"] == 64


def test_telemetry_failure_is_safe_record4_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    route = _route(tmp_path)
    output = Path(route.spec.output_dir)
    before = _artifact_identity(output)
    from forge import telemetry

    def fail(*args, **kwargs):
        raise OSError("injected telemetry failure")

    monkeypatch.setattr(telemetry, "event", fail)
    monkeypatch.setattr(telemetry, "write_into", fail)
    assert not soup.apply_qwen35_soup_override(
        route,
        tracker=_tracker(),
        model=object(),
        tokenizer=object(),
        smoke=lambda *_: None,
    )
    assert _artifact_identity(output) == before
    assert not (output / "CEO_EXPLORATORY_OVERRIDE.json").exists()
