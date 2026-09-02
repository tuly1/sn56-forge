"""CPU contracts for the compact Sep-7 survival child."""

from __future__ import annotations

import json
import os
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest

from forge.tasks import common, instruct
from forge.tuning.plan import TrainPlan


def _plan(*, batch: int = 4, accum: int = 4) -> TrainPlan:
    return TrainPlan(
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        learning_rate=1.5e-4,
        per_device_batch_size=batch,
        grad_accum_steps=accum,
        max_seq_len=4096,
        num_epochs=2,
        warmup_ratio=0.03,
        weight_decay=0.0,
        optimizer="adamw_torch_fused",
        lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=False,
        bf16=True,
        fp16=False,
        strategy="lora",
    )


def test_geometry_ladder_preserves_falcon_recipe_and_effective_batch():
    source = _plan()
    plans = instruct._geometry_plans(source)
    assert [(p.per_device_batch_size, p.grad_accum_steps) for p in plans] == [
        (4, 4),
        (2, 8),
        (1, 16),
    ]
    # All science settings stay exact; only memory geometry may move.
    for candidate in plans:
        assert candidate.lora_r == source.lora_r == 32
        assert candidate.lora_alpha == source.lora_alpha == 64
        assert candidate.learning_rate == source.learning_rate
        assert candidate.num_epochs == source.num_epochs
        assert candidate.max_seq_len == source.max_seq_len
        assert candidate.per_device_batch_size * candidate.grad_accum_steps == 16


def test_probe_rows_are_actual_deterministic_p99_and_worst_batches():
    rows = [{"input_ids": list(range(length))} for length in range(1, 101)]
    selected = instruct._memory_probe_rows(rows, batch_size=4)
    assert [label for label, _rows, _identity in selected] == ["p99", "worst"]
    p99 = selected[0][2]
    worst = selected[1][2]
    assert len(p99["row_indices"]) == len(worst["row_indices"]) == 4
    assert 99 in p99["row_lengths"]
    assert worst["row_lengths"] == [100, 99, 98, 97]
    assert all(selected_row is rows[index] for index, selected_row in zip(
        p99["row_indices"], selected[0][1]
    ))


def test_probe_checkpointing_matches_plan_and_supports_legacy_models():
    calls: list[object] = []

    class Modern:
        def gradient_checkpointing_enable(self, **kwargs):
            calls.append(kwargs)

    class Legacy:
        def gradient_checkpointing_enable(self):
            calls.append("legacy")

    enabled = replace(_plan(), gradient_checkpointing=True)
    instruct._enable_probe_gradient_checkpointing(Modern(), enabled)
    instruct._enable_probe_gradient_checkpointing(Legacy(), enabled)
    instruct._enable_probe_gradient_checkpointing(
        Modern(), replace(_plan(), gradient_checkpointing=False)
    )
    assert calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        "legacy",
    ]


def test_kl_probe_uses_existing_adapter_disabled_reference_forward():
    torch = pytest.importorskip("torch")

    class Model:
        training = True
        reference_calls = 0

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def disable_adapter(self):
            from contextlib import nullcontext

            return nullcontext()

        def __call__(self, **_inputs):
            self.reference_calls += 1
            return SimpleNamespace(logits=torch.zeros(1, 2, 3))

    model = Model()
    policy_logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [1.0, -1.0, 0.0]]], requires_grad=True
    )
    ce = torch.tensor(1.0, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[1, 2]]),
    }
    loss = instruct._probe_sft_loss(
        model,
        batch,
        SimpleNamespace(loss=ce, logits=policy_logits),
        is_kl=True,
        kl_coef=0.5,
    )
    assert model.reference_calls == 1
    assert model.training is True
    assert loss > ce
    loss.backward()
    assert policy_logits.grad is not None


def test_admission_rebuilds_failed_and_selected_models_from_pristine(monkeypatch):
    measured: list[tuple[str, int]] = []
    rebuilt: list[int] = []
    discarded: list[str] = []

    monkeypatch.setattr(instruct, "_cuda_available", lambda: True)

    def measure(*, model, train_ex, collator, plan, **_kwargs):
        batch_size = plan.per_device_batch_size
        measured.append((model, batch_size))
        return {
            "status": "HEADROOM_EXCEEDED" if batch_size == 4 else "PASS",
            "batches": [{"label": "p99"}, {"label": "worst"}],
        }

    def rebuild(plan):
        rebuilt.append(plan.per_device_batch_size)
        return f"fresh-{plan.per_device_batch_size}-{len(rebuilt)}"

    monkeypatch.setattr(instruct, "_measure_sft_geometry", measure)
    monkeypatch.setattr(instruct, "_discard_model", discarded.append)
    plan, model, attempts = instruct._admit_sft_geometry(
        plan=_plan(),
        model="initial-4",
        train_ex=[{"input_ids": [1], "labels": [1]}],
        collator=object(),
        rebuild_model=rebuild,
    )
    assert (plan.per_device_batch_size, plan.grad_accum_steps) == (2, 8)
    assert measured == [("initial-4", 4), ("fresh-2-1", 2)]
    assert discarded == ["initial-4", "fresh-2-1"]
    assert rebuilt == [2, 2]
    assert model == "fresh-2-2", "training must not reuse the probed model"
    assert [attempt["status"] for attempt in attempts] == [
        "HEADROOM_EXCEEDED",
        "PASS",
    ]


def test_admission_releases_original_model_before_rebuild(monkeypatch):
    references: list[weakref.ReferenceType] = []

    class Model:
        def __init__(self):
            references.append(weakref.ref(self))

        def zero_grad(self, **_kwargs):
            pass

    monkeypatch.setattr(instruct, "_cuda_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    monkeypatch.setattr(
        instruct,
        "_measure_sft_geometry",
        lambda **kwargs: {
            "status": (
                "HEADROOM_EXCEEDED"
                if kwargs["plan"].per_device_batch_size == 4
                else "PASS"
            ),
            "batches": [],
        },
    )

    def rebuild(_plan):
        assert references[0]() is None, "discarded probe model remained pinned"
        return Model()

    _plan_out, selected, _attempts = instruct._admit_sft_geometry(
        plan=_plan(),
        model=Model(),
        train_ex=[{"input_ids": [1], "labels": [1]}],
        collator=object(),
        rebuild_model=rebuild,
    )
    assert selected is not None


class _Trainer:
    def __init__(self, model: str, *, oom_step: int | None):
        self.model = model
        self.state = SimpleNamespace(global_step=oom_step or 0)
        self.oom_step = oom_step

    def train(self):
        if self.oom_step is not None:
            raise RuntimeError("CUDA out of memory")


def test_zero_progress_oom_rebuilds_complete_trainer_at_next_geometry(monkeypatch):
    builds: list[tuple[int, str]] = []
    rebuilds: list[int] = []
    discarded: list[str] = []

    def build(plan, model):
        builds.append((plan.per_device_batch_size, model))
        oom = 0 if plan.per_device_batch_size == 4 else None
        return _Trainer(model, oom_step=oom), common.BestTracker(), None

    def rebuild(plan):
        rebuilds.append(plan.per_device_batch_size)
        return f"fresh-{plan.per_device_batch_size}"

    monkeypatch.setattr(
        instruct, "_discard_trainer", lambda trainer: discarded.append(trainer.model)
    )
    trainer, _tracker, _route, selected = instruct._train_sft_geometry_ladder(
        initial_plan=_plan(),
        initial_model="pristine-4",
        build_trainer=build,
        rebuild_model=rebuild,
        spec=SimpleNamespace(output_dir="/unused"),
        tokenizer=object(),
    )
    assert builds == [(4, "pristine-4"), (2, "fresh-2")]
    assert discarded == ["pristine-4"]
    assert rebuilds == [2]
    assert trainer.model == "fresh-2"
    assert (selected.per_device_batch_size, selected.grad_accum_steps) == (2, 8)


def test_zero_progress_retry_releases_original_model_before_rebuild(monkeypatch):
    references: list[weakref.ReferenceType] = []

    class Model:
        def __init__(self):
            references.append(weakref.ref(self))


    def build(plan, model):
        oom_step = 0 if plan.per_device_batch_size == 4 else None
        return _Trainer(model, oom_step=oom_step), common.BestTracker(), None

    def rebuild(_plan):
        assert references[0]() is None, "zero-step model remained pinned"
        return Model()

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    trainer, _tracker, _route, _selected = instruct._train_sft_geometry_ladder(
        initial_plan=_plan(),
        initial_model=Model(),
        build_trainer=build,
        rebuild_model=rebuild,
        spec=SimpleNamespace(output_dir="/unused"),
        tokenizer=object(),
    )
    assert trainer.model is not None


def test_normal_return_without_optimizer_progress_remains_floor():
    assert instruct._completed_artifact_truth(0) == common.ARTIFACT_FLOOR
    assert instruct._completed_artifact_truth(1) == common.ARTIFACT_COMPLETE_BEST


def test_progressed_oom_preserves_and_never_retries_geometry(monkeypatch):
    preserved: list[tuple[str, int]] = []
    rebuilds: list[int] = []

    def build(_plan, model):
        return _Trainer(model, oom_step=7), common.BestTracker(), None

    def preserve(**kwargs):
        preserved.append((kwargs["trainer"].model, kwargs["step"]))

    monkeypatch.setattr(instruct, "_preserve_progressed_oom", preserve)
    with pytest.raises(RuntimeError, match="preserved last valid artifact"):
        instruct._train_sft_geometry_ladder(
            initial_plan=_plan(),
            initial_model="trained-model",
            build_trainer=build,
            rebuild_model=lambda plan: rebuilds.append(plan.per_device_batch_size),
            spec=SimpleNamespace(output_dir="/unused"),
            tokenizer=object(),
        )
    assert preserved == [("trained-model", 7)]
    assert rebuilds == []


def test_progressed_oom_keeps_last_valid_periodic_export(tmp_path, monkeypatch):
    output = str(tmp_path / "artifact")
    _artifact(output)
    assert common.write_artifact_truth(
        output,
        common.ARTIFACT_PARTIAL_TRAINED_BEST,
        optimizer_step=25,
        reason="periodic_trained_recovery",
    )
    weights = (tmp_path / "artifact" / "model.safetensors").read_bytes()
    monkeypatch.setattr(
        instruct,
        "save_adapter",
        lambda *_args, **_kwargs: pytest.fail("valid periodic export was overwritten"),
    )
    instruct._preserve_progressed_oom(
        trainer=SimpleNamespace(model=object()),
        tracker=common.BestTracker(),
        spec=SimpleNamespace(output_dir=output),
        tokenizer=object(),
        step=31,
    )
    truth = common.read_artifact_truth(output)
    assert truth is not None
    assert truth["truth"] == common.ARTIFACT_PARTIAL_TRAINED_BEST
    assert truth["optimizer_step"] == 25
    assert (tmp_path / "artifact" / "model.safetensors").read_bytes() == weights


def _artifact(path, marker: bytes = b"best") -> None:
    os.makedirs(path, exist_ok=True)
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    with open(os.path.join(path, "model.safetensors"), "wb") as handle:
        handle.write(len(header).to_bytes(8, "little") + header + marker[:4])


def test_artifact_truth_transitions_do_not_change_weights(tmp_path):
    output = str(tmp_path / "artifact")
    _artifact(output)
    weights = (tmp_path / "artifact" / "model.safetensors").read_bytes()
    assert common.write_artifact_truth(
        output,
        common.ARTIFACT_FLOOR,
        optimizer_step=0,
        reason="test_floor",
    )
    truth_path = tmp_path / "artifact" / "forge_artifact_truth.json"
    assert json.loads(truth_path.read_text())["truth"] == "FLOOR"
    assert common.write_artifact_truth(
        output,
        common.ARTIFACT_COMPLETE_BEST,
        optimizer_step=55,
        reason="test_complete",
    )
    truth = json.loads(truth_path.read_text())
    assert truth == {
        "schema": 1,
        "truth": "COMPLETE_BEST",
        "optimizer_step": 55,
        "reason": "test_complete",
    }
    assert (tmp_path / "artifact" / "model.safetensors").read_bytes() == weights


def test_artifact_truth_commit_survives_telemetry_failure(tmp_path, monkeypatch):
    from forge import telemetry

    output = str(tmp_path / "artifact")
    _artifact(output)
    monkeypatch.setattr(
        telemetry,
        "write_into",
        lambda _path: (_ for _ in ()).throw(OSError("telemetry unavailable")),
    )
    assert common.write_artifact_truth(
        output,
        common.ARTIFACT_PARTIAL_TRAINED_BEST,
        optimizer_step=4,
        reason="trained_before_diagnostics",
    )
    assert common.read_artifact_truth(output)["optimizer_step"] == 4


def test_save_adapter_promotes_truth_with_the_weight_generation(tmp_path):
    output = str(tmp_path / "artifact")

    class Model:
        def save_pretrained(self, path, **_kwargs):
            _artifact(path, b"part")

    class Tokenizer:
        def save_pretrained(self, path):
            with open(os.path.join(path, "tokenizer_config.json"), "w") as handle:
                json.dump({}, handle)

    common.save_adapter(
        Model(),
        Tokenizer(),
        output,
        artifact_truth=common.ARTIFACT_PARTIAL_TRAINED_BEST,
        optimizer_step=25,
        truth_reason="periodic_trained_recovery",
    )
    truth = json.loads(
        (tmp_path / "artifact" / "forge_artifact_truth.json").read_text()
    )
    assert truth["truth"] == "PARTIAL_TRAINED_BEST"
    assert truth["optimizer_step"] == 25
    assert os.path.isfile(os.path.join(output, ".forge_artifact_ready"))
