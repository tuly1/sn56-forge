"""Focused contract for the pinned BloomZ production promotion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from forge.data.schema import InstructColumns, TaskSpec
from forge.tasks import instruct
from forge.tuning.plan import TrainPlan


def _spec(**updates) -> TaskSpec:
    values = dict(
        task_id="bloom",
        task_type="InstructTextTask",
        model=instruct._BLOOMZ_REPO,
        dataset="dataset",
        expected_repo_name="out",
        baseline_stats_path=None,
        use_kl=False,
        kl_coef=0.0,
        instruct=InstructColumns(instruction="prompt", output="answer"),
    )
    values.update(updates)
    return TaskSpec(**values)


def _plan(strategy: str = "full") -> TrainPlan:
    return TrainPlan(
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        learning_rate=5e-5,
        per_device_batch_size=4,
        grad_accum_steps=4,
        max_seq_len=4096,
        num_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.1,
        optimizer="adamw_torch",
        lr_scheduler="linear",
        gradient_checkpointing=False,
        bf16=False,
        fp16=True,
        strategy=strategy,
    )


def _padded_bytes(size: int, prefix: bytes) -> bytes:
    assert len(prefix) <= size
    return prefix + b" " * (size - len(prefix))


def _identity_fixture(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    root = tmp_path / "base"
    root.mkdir()
    config = {
        "architectures": ["BloomForCausalLM"],
        "model_type": "bloom",
        "n_embed": 1024,
        "n_layer": 24,
        "num_attention_heads": 16,
        "seq_length": 2048,
        "vocab_size": 250880,
        "unk_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "use_cache": True,
    }
    files = {
        "config.json": json.dumps(config, sort_keys=True).encode(),
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": _padded_bytes(222, b"{}"),
        "special_tokens_map.json": _padded_bytes(85, b"{}"),
        "model.safetensors": b"pinned-full-model",
    }
    constants = {
        "config.json": "_BLOOMZ_CONFIG_SHA256",
        "tokenizer.json": "_BLOOMZ_TOKENIZER_SHA256",
        "tokenizer_config.json": "_BLOOMZ_TOKENIZER_CONFIG_SHA256",
        "special_tokens_map.json": "_BLOOMZ_SPECIAL_TOKENS_SHA256",
        "model.safetensors": "_BLOOMZ_WEIGHTS_SHA256",
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
        monkeypatch.setattr(
            instruct, constants[name], hashlib.sha256(payload).hexdigest()
        )
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="bloom",
            architectures=["BloomForCausalLM"],
            vocab_size=250880,
        )
    )
    return SimpleNamespace(model=model, model_dir=str(root))


def test_exact_task_and_revision_identity_is_the_only_promoted_route(
    tmp_path: Path, monkeypatch
) -> None:
    loaded = _identity_fixture(tmp_path, monkeypatch)
    assert instruct._is_bloomz_promotion(
        _spec(), loaded, instruct._BLOOMZ_PARAMS_B
    )

    negatives = [
        _spec(task_type="ChatTask"),
        _spec(model="bigscience/bloom-560m"),
        _spec(use_kl=True, kl_coef=0.0),
        _spec(use_kl=True, kl_coef=0.1),
        _spec(instruct=InstructColumns(instruction="prompt", output=None)),
    ]
    for spec in negatives:
        assert not instruct._is_bloomz_promotion(
            spec, loaded, instruct._BLOOMZ_PARAMS_B
        )
    assert not instruct._is_bloomz_promotion(_spec(), loaded, 0.56)

    (Path(loaded.model_dir) / "model.safetensors").write_bytes(b"different")
    assert not instruct._is_bloomz_promotion(
        _spec(), loaded, instruct._BLOOMZ_PARAMS_B
    )


def test_bloomz_recipe_is_exact(tmp_path: Path, monkeypatch) -> None:
    from forge.tasks import common

    plan = instruct._bloomz_plan(_plan())
    assert plan.strategy == "full"
    assert (plan.lora_r, plan.lora_alpha, plan.lora_dropout) == (0, 0, 0.0)
    assert plan.learning_rate == 1e-4
    assert (plan.per_device_batch_size, plan.grad_accum_steps) == (1, 16)
    assert plan.max_seq_len == 2048
    assert plan.gradient_checkpointing is True
    assert (plan.bf16, plan.fp16) == (True, False)
    assert plan.optimizer == "adamw_torch_fused"
    assert plan.lr_scheduler == "cosine_with_min_lr"
    assert (plan.warmup_ratio, plan.weight_decay) == (0.03, 0.0)
    monkeypatch.setattr(common, "workdir", lambda spec: str(tmp_path / "work"))
    kwargs = common.build_training_kwargs(_spec(), plan, neftune_alpha=5.0)
    assert kwargs["max_grad_norm"] == 1.0
    assert kwargs["neftune_noise_alpha"] == 5.0
    assert kwargs["lr_scheduler_kwargs"] == {"min_lr_rate": 0.25}
    assert kwargs["seed"] == 7
    assert kwargs["save_strategy"] == "no"


def test_bloomz_full_finetune_requires_all_trainable_fp32() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 2).half()
    prepared = instruct._prepare_bloomz_full_finetune(model)
    assert all(
        parameter.requires_grad and parameter.dtype == torch.float32
        for parameter in prepared.parameters()
    )


@pytest.mark.parametrize("mode", ["failed", "partial"])
def test_bloomz_full_finetune_rejects_failed_or_partial_upcast(mode: str) -> None:
    torch = pytest.importorskip("torch")

    class BrokenUpcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
            self.second = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))

        def float(self):
            if mode == "failed":
                raise RuntimeError("injected float failure")
            self.first.data = self.first.data.float()
            return self

    with pytest.raises(instruct._BloomzPromotionError, match="trainable in fp32"):
        instruct._prepare_bloomz_full_finetune(BrokenUpcast())


def _install_training_stubs(monkeypatch, *, final_step: int = 256):
    captured = SimpleNamespace(trainer=None, args=None)

    datasets = ModuleType("datasets")

    class Dataset:
        @staticmethod
        def from_list(rows):
            return list(rows)

    datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    transformers = ModuleType("transformers")

    class TrainingArguments:
        def __init__(self, **kwargs):
            captured.args = kwargs
            self.__dict__.update(kwargs)

    class Trainer:
        def __init__(self, **kwargs):
            captured.trainer = self
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.args = kwargs["args"]
            self.state = SimpleNamespace(global_step=final_step)

        def train(self):
            return None

    class TrainerCallback:
        pass

    transformers.Trainer = Trainer
    transformers.TrainerCallback = TrainerCallback
    transformers.TrainingArguments = TrainingArguments
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return captured


def _stub_run_dependencies(monkeypatch, tmp_path: Path, *, promoted: bool):
    captured = _install_training_stubs(monkeypatch)
    rows = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(1100)]
    tokenizer = SimpleNamespace(pad_token_id=3)
    model = SimpleNamespace(config=SimpleNamespace())
    loaded = SimpleNamespace(model=model, tokenizer=tokenizer, model_dir=str(tmp_path))
    events = SimpleNamespace(
        floor=0,
        full_save=0,
        periodic=0,
        best=0,
        probe=0,
        truths=[],
    )

    monkeypatch.setattr(instruct.loader, "load_rows", lambda *a, **k: rows)
    monkeypatch.setattr(instruct, "load_base", lambda *a, **k: loaded)
    monkeypatch.setattr(instruct.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(instruct.telemetry, "event", lambda *a, **k: None)
    monkeypatch.setattr(instruct.telemetry, "set_meta", lambda **k: None)
    monkeypatch.setattr(
        instruct.telemetry, "make_trainer_callback", lambda output: "telemetry"
    )
    monkeypatch.setattr(instruct, "load_baseline_summary", lambda *a, **k: None)
    monkeypatch.setattr(instruct, "model_param_billions", lambda model: 0.559214592)
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (1, 80.0))
    monkeypatch.setattr(instruct, "_is_bloomz_promotion", lambda *a: promoted)
    monkeypatch.setattr(instruct, "decide_full_finetune", lambda **k: False)
    monkeypatch.setattr(instruct, "make_sft_plan", lambda **k: _plan(k["strategy"]))
    monkeypatch.setattr(
        instruct, "conservative_qwen35_plan", lambda model, plan: (plan, False)
    )
    monkeypatch.setattr(
        instruct, "conservative_quasar_plan", lambda model, plan: (plan, False)
    )
    monkeypatch.setattr(instruct, "prepare_full_finetune", lambda model, **k: model)
    monkeypatch.setattr(
        instruct, "_prepare_bloomz_full_finetune", lambda model: model
    )
    monkeypatch.setattr(instruct, "attach_lora", lambda model, **k: model)
    monkeypatch.setattr(
        instruct.prompts,
        "build_instruct_examples",
        lambda rows, columns: list(rows),
    )
    monkeypatch.setattr(
        instruct.tokenize,
        "tokenize_instruct",
        lambda examples, tokenizer, seq_len: [
            {"input_ids": [seq_len], "labels": [1]} for _ in examples
        ],
    )
    monkeypatch.setattr(instruct.tokenize, "PadCollator", lambda pad: ("pad", pad))
    monkeypatch.setattr(
        instruct,
        "build_training_kwargs",
        lambda spec, plan, neftune_alpha: {
            "num_train_epochs": plan.num_epochs,
            "neftune_noise_alpha": neftune_alpha,
            "lr_scheduler_kwargs": {"min_lr_rate": 0.25},
            "seed": 7,
        },
    )
    monkeypatch.setattr(
        instruct, "compatible_dataclass_kwargs", lambda cls, kwargs, **k: kwargs
    )
    monkeypatch.setattr(instruct, "DeadlineCallback", lambda deadline: "deadline")
    monkeypatch.setattr(
        instruct,
        "eligible_qwen35_soup_route",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(instruct, "apply_qwen35_soup_override", lambda *a, **k: None)
    monkeypatch.setattr(instruct, "safe_train", lambda trainer: trainer.train())
    monkeypatch.setattr(instruct, "workdir", lambda spec: str(tmp_path / "work"))
    monkeypatch.setattr(instruct, "save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(
        instruct,
        "write_artifact_truth",
        lambda output, truth, **kwargs: events.truths.append(
            (truth, kwargs["optimizer_step"], kwargs["reason"])
        ),
    )

    def periodic(*args, **kwargs):
        events.periodic += 1
        return "periodic"

    def best(*args, **kwargs):
        events.best += 1
        return "best"

    def probe(**kwargs):
        events.probe += 1
        return None, None

    def full_save(*args, **kwargs):
        events.full_save += 1
        if kwargs.get("artifact_truth") is not None:
            events.truths.append(
                (
                    kwargs["artifact_truth"],
                    kwargs["optimizer_step"],
                    kwargs["truth_reason"],
                )
            )

    monkeypatch.setattr(instruct, "_make_periodic_save_callback", periodic)
    monkeypatch.setattr(instruct, "_make_best_checkpoint_callback", best)
    monkeypatch.setattr(instruct, "time_aware_epochs", probe)
    monkeypatch.setattr(instruct, "_save_bloomz_full_model", full_save)
    from forge.tasks import fallback

    monkeypatch.setattr(
        fallback,
        "emit_untrained_copy",
        lambda spec: setattr(events, "floor", events.floor + 1),
    )
    return captured, events


def test_promoted_run_uses_all_rows_fixed_schedule_and_no_internal_best(
    tmp_path: Path, monkeypatch
) -> None:
    captured, events = _stub_run_dependencies(monkeypatch, tmp_path, promoted=True)
    instruct.run(_spec(), SimpleNamespace())

    assert len(captured.trainer.kwargs["train_dataset"]) == 1100
    assert captured.trainer.kwargs["eval_dataset"] is None
    assert captured.args["max_steps"] == 256
    assert "num_train_epochs" not in captured.args
    assert captured.args["neftune_noise_alpha"] == 5.0
    assert captured.args["lr_scheduler_kwargs"] == {"min_lr_rate": 0.25}
    assert captured.args["seed"] == 7
    callbacks = captured.trainer.kwargs["callbacks"]
    assert callbacks[0] == "deadline" and callbacks[-1] == "telemetry"
    assert len(callbacks) == 3
    assert (events.floor, events.full_save) == (1, 1)
    assert (events.periodic, events.best, events.probe) == (0, 0, 0)
    assert [(truth, step) for truth, step, _reason in events.truths] == [
        (instruct.ARTIFACT_FLOOR, 0),
        (instruct.ARTIFACT_COMPLETE_BEST, 256),
    ]


def test_non_bloom_run_retains_default_holdout_probe_and_callbacks(
    tmp_path: Path, monkeypatch
) -> None:
    captured, events = _stub_run_dependencies(monkeypatch, tmp_path, promoted=False)
    monkeypatch.setattr(
        instruct.tokenize,
        "sft_sequence_len_candidates",
        lambda model, tokenizer, initial: [4096],
    )
    monkeypatch.setattr(instruct, "effective_sft_seq_len", lambda model, cap: 4096)
    monkeypatch.setattr(
        instruct.tokenize,
        "first_nonempty_tokenization",
        lambda candidates, fn: (fn(candidates[0]), candidates[0]),
    )
    instruct.run(
        _spec(model="other/model"),
        SimpleNamespace(remaining=lambda: 1200.0),
    )

    assert len(captured.trainer.kwargs["train_dataset"]) == 844
    assert len(captured.trainer.kwargs["eval_dataset"]) == 256
    assert events.probe == 1
    assert events.periodic == 1
    assert events.best == 1
    assert events.floor == 0
    assert events.full_save == 0


def test_training_failure_leaves_bloomz_lora_floor(
    tmp_path: Path, monkeypatch
) -> None:
    _, events = _stub_run_dependencies(monkeypatch, tmp_path, promoted=True)

    def fail(_trainer):
        raise RuntimeError("training failed")

    monkeypatch.setattr(instruct, "safe_train", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        instruct.run(_spec(), SimpleNamespace())
    assert events.floor == 1
    assert events.full_save == 0
    assert [(truth, step) for truth, step, _reason in events.truths] == [
        (instruct.ARTIFACT_FLOOR, 0)
    ]


@pytest.mark.parametrize(
    "failure_step,expected_generation",
    [(63, "floor"), (65, "64"), (129, "128"), (193, "192"), (255, "192")],
)
def test_late_failure_retains_latest_science_cadence_generation(
    tmp_path: Path, monkeypatch, failure_step: int, expected_generation: str
) -> None:
    _install_training_stubs(monkeypatch)
    output = tmp_path / "visible"
    output.mkdir()
    generation = output / "generation"
    generation.write_text("floor")
    spec = SimpleNamespace(output_dir=str(output))

    def publish(model, tokenizer, output_dir, metadata_source_dir, **kwargs):
        Path(output_dir, "generation").write_text(str(model.step))

    monkeypatch.setattr(instruct, "_save_bloomz_full_model", publish)
    callback = instruct._make_bloomz_save_callback(
        spec, object(), str(tmp_path / "base")
    )
    model = SimpleNamespace(step=0)
    callback.on_step_end(None, SimpleNamespace(global_step=0), object(), model=model)
    assert generation.read_text() == "floor"

    with pytest.raises(RuntimeError, match="late training failure"):
        for step in range(1, failure_step + 1):
            model.step = step
            callback.on_step_end(
                None, SimpleNamespace(global_step=step), object(), model=model
            )
        raise RuntimeError("late training failure")
    assert generation.read_text() == expected_generation


def test_cadence_export_failure_preserves_prior_validated_generation(
    tmp_path: Path, monkeypatch
) -> None:
    _install_training_stubs(monkeypatch)
    output = tmp_path / "visible"
    output.mkdir()
    generation = output / "generation"
    generation.write_text("floor")
    spec = SimpleNamespace(output_dir=str(output))

    def publish(model, tokenizer, output_dir, metadata_source_dir, **kwargs):
        if model.step == 128:
            raise RuntimeError("injected export failure")
        Path(output_dir, "generation").write_text(str(model.step))

    monkeypatch.setattr(instruct, "_save_bloomz_full_model", publish)
    callback = instruct._make_bloomz_save_callback(
        spec, object(), str(tmp_path / "base")
    )
    model = SimpleNamespace(step=64)
    callback.on_step_end(None, SimpleNamespace(global_step=64), object(), model=model)
    model.step = 128
    callback.on_step_end(None, SimpleNamespace(global_step=128), object(), model=model)
    assert generation.read_text() == "64"
    assert callback.last_saved_step == 64


def test_bloomz_save_callback_publishes_exact_science_cadence(
    tmp_path: Path, monkeypatch
) -> None:
    _install_training_stubs(monkeypatch)
    saved = []
    truths = []
    def capture_save(model, *args, **kwargs):
        saved.append(model.step)
        truths.append((kwargs["artifact_truth"], kwargs["optimizer_step"]))

    monkeypatch.setattr(instruct, "_save_bloomz_full_model", capture_save)
    callback = instruct._make_bloomz_save_callback(
        SimpleNamespace(output_dir=str(tmp_path / "out")),
        object(),
        str(tmp_path / "base"),
    )
    model = SimpleNamespace(step=0)
    for step in (0, 1, 63, 64, 65, 127, 128, 191, 192, 255, 256):
        model.step = step
        callback.on_step_end(
            None, SimpleNamespace(global_step=step), object(), model=model
        )
    assert saved == [64, 128, 192, 256]
    assert callback.last_saved_step == 256
    assert truths == [
        (instruct.ARTIFACT_PARTIAL_TRAINED_BEST, 64),
        (instruct.ARTIFACT_PARTIAL_TRAINED_BEST, 128),
        (instruct.ARTIFACT_PARTIAL_TRAINED_BEST, 192),
        (instruct.ARTIFACT_COMPLETE_BEST, 256),
    ]


def test_bloomz_save_callback_deduplicates_successful_step(
    tmp_path: Path, monkeypatch
) -> None:
    _install_training_stubs(monkeypatch)
    saves = []
    monkeypatch.setattr(
        instruct,
        "_save_bloomz_full_model",
        lambda *args, **kwargs: saves.append(64),
    )
    callback = instruct._make_bloomz_save_callback(
        SimpleNamespace(output_dir=str(tmp_path / "out")), object(), "/base"
    )
    state = SimpleNamespace(global_step=64)
    callback.on_step_end(None, state, object(), model=object())
    callback.on_step_end(None, state, object(), model=object())
    assert saves == [64]
    assert callback.last_saved_step == 64


def test_bloomz_save_callback_retries_same_step_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _install_training_stubs(monkeypatch)
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(64)
        if len(attempts) == 1:
            raise RuntimeError("first save failed")

    monkeypatch.setattr(instruct, "_save_bloomz_full_model", fail_once)
    callback = instruct._make_bloomz_save_callback(
        SimpleNamespace(output_dir=str(tmp_path / "out")), object(), "/base"
    )
    state = SimpleNamespace(global_step=64)
    callback.on_step_end(None, state, object(), model=object())
    assert callback.last_saved_step == 0
    callback.on_step_end(None, state, object(), model=object())
    assert attempts == [64, 64]
    assert callback.last_saved_step == 64


def test_full_export_restores_metadata_has_no_adapter_and_native_reloads(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    tokenizers = pytest.importorskip("tokenizers")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    config = transformers.BloomConfig(
        hidden_size=8,
        n_layer=1,
        n_head=2,
        vocab_size=32,
        seq_length=16,
        use_cache=True,
    )
    model = transformers.BloomForCausalLM(config)
    source = tmp_path / "base"
    source.mkdir()
    model.config.save_pretrained(source)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<unk>": 0, "<pad>": 1, "hello": 2}, unk_token="<unk>"
        )
    )
    base_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
    )
    base_tokenizer.save_pretrained(source)
    (source / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>", "pad_token": "<pad>"}) + "\n"
    )
    base_config = (source / "config.json").read_bytes()
    model.config.use_cache = False
    base_tokenizer.padding_side = "left"
    base_tokenizer.chat_template = "generated"

    output = tmp_path / "out"
    instruct._save_bloomz_full_model(
        model, base_tokenizer, str(output), str(source)
    )
    assert (output / "config.json").read_bytes() == base_config
    assert (output / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert not (output / "chat_template.jinja").exists()
    assert not (output / "adapter_config.json").exists()
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True
    )
    assert isinstance(reloaded, transformers.BloomForCausalLM)
    assert reloaded.config.use_cache is True
    assert all(torch.isfinite(value).all() for value in reloaded.state_dict().values())


def test_failed_full_export_atomically_preserves_existing_floor(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text('{"use_cache":true}\n')
    output = tmp_path / "out"
    output.mkdir()
    safetensors.save_file(
        {"floor": torch.ones(1)}, output / "adapter_model.safetensors"
    )
    (output / "adapter_config.json").write_text("{}\n")
    floor_before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    class Model:
        def save_pretrained(self, path, **kwargs):
            target = Path(path)
            safetensors.save_file(
                {"candidate": torch.ones(1)}, target / "model.safetensors"
            )
            (target / "config.json").write_text('{"use_cache":false}\n')

    class FailingTokenizer:
        def save_pretrained(self, path, **kwargs):
            raise RuntimeError("tokenizer export failed")

    with pytest.raises(RuntimeError, match="tokenizer export failed"):
        instruct._save_bloomz_full_model(
            Model(), FailingTokenizer(), str(output), str(source)
        )
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == floor_before
    assert not Path(str(output) + ".tmp").exists()


def test_adapter_semantic_failure_is_rejected_before_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    source = tmp_path / "base"
    source.mkdir()
    for name, payload in {
        "config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "special_tokens_map.json": b"{}\n",
    }.items():
        (source / name).write_bytes(payload)
    output = tmp_path / "out"
    output.mkdir()
    safetensors.save_file(
        {"floor": torch.ones(1)}, output / "adapter_model.safetensors"
    )
    (output / "adapter_config.json").write_text("{}\n")
    floor_before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    class ContaminatedModel:
        def save_pretrained(self, path, **kwargs):
            target = Path(path)
            safetensors.save_file(
                {"candidate": torch.ones(1)}, target / "model.safetensors"
            )
            (target / "adapter_config.json").write_text("{}\n")

    class Tokenizer:
        def save_pretrained(self, path, **kwargs):
            return None

    with pytest.raises(instruct._BloomzPromotionError, match="adapter artifact"):
        instruct._save_bloomz_full_model(
            ContaminatedModel(), Tokenizer(), str(output), str(source)
        )
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == floor_before


def test_nested_adapter_path_is_rejected_before_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    source = tmp_path / "base"
    source.mkdir()
    for name in instruct._BLOOMZ_REQUIRED_METADATA:
        (source / name).write_text("{}\n")
    output = tmp_path / "out"
    output.mkdir()
    safetensors.save_file(
        {"floor": torch.ones(1)}, output / "adapter_model.safetensors"
    )
    (output / "adapter_config.json").write_text("{}\n")
    floor_before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    class NestedAdapterModel:
        def save_pretrained(self, path, **kwargs):
            target = Path(path)
            safetensors.save_file(
                {"candidate": torch.ones(1)}, target / "model.safetensors"
            )
            adapter = target / "adapter"
            adapter.mkdir()
            (adapter / "payload.bin").write_bytes(b"adapter payload")

    class Tokenizer:
        def save_pretrained(self, path, **kwargs):
            return None

    with pytest.raises(instruct._BloomzPromotionError, match="adapter artifact"):
        instruct._save_bloomz_full_model(
            NestedAdapterModel(), Tokenizer(), str(output), str(source)
        )
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == floor_before


def test_staged_config_symlink_cannot_overwrite_outside_target(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    source = tmp_path / "base"
    source.mkdir()
    for name in instruct._BLOOMZ_REQUIRED_METADATA:
        (source / name).write_text("{}\n")
    output = tmp_path / "out"
    output.mkdir()
    safetensors.save_file(
        {"floor": torch.ones(1)}, output / "adapter_model.safetensors"
    )
    (output / "adapter_config.json").write_text("{}\n")
    floor_before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    outside = tmp_path / "outside-config.json"
    outside.write_bytes(b"outside must not change\n")
    outside_before = outside.read_bytes()

    class SymlinkModel:
        def save_pretrained(self, path, **kwargs):
            target = Path(path)
            safetensors.save_file(
                {"candidate": torch.ones(1)}, target / "model.safetensors"
            )
            (target / "config.json").symlink_to(outside)

    class Tokenizer:
        def save_pretrained(self, path, **kwargs):
            return None

    with pytest.raises(instruct._BloomzPromotionError, match="symlink"):
        instruct._save_bloomz_full_model(
            SymlinkModel(), Tokenizer(), str(output), str(source)
        )
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == floor_before
    assert outside.read_bytes() == outside_before


def test_native_reload_failure_is_rejected_before_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    from forge import telemetry

    monkeypatch.setattr(telemetry, "write_into", lambda path: None)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    def fail_reload(*args, **kwargs):
        raise RuntimeError("injected native reload failure")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fail_reload
    )
    source = tmp_path / "base"
    source.mkdir()
    for name in instruct._BLOOMZ_REQUIRED_METADATA:
        (source / name).write_text("{}\n")
    output = tmp_path / "out"
    output.mkdir()
    safetensors.save_file(
        {"floor": torch.ones(1)}, output / "adapter_model.safetensors"
    )
    (output / "adapter_config.json").write_text("{}\n")
    floor_before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    class Model:
        def save_pretrained(self, path, **kwargs):
            safetensors.save_file(
                {"candidate": torch.ones(1)}, Path(path) / "model.safetensors"
            )

    class Tokenizer:
        def save_pretrained(self, path, **kwargs):
            return None

    with pytest.raises(RuntimeError, match="native reload failure"):
        instruct._save_bloomz_full_model(
            Model(), Tokenizer(), str(output), str(source)
        )
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == floor_before
