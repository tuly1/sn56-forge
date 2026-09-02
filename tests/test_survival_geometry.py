"""Focused CPU contracts for the compact survival path."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import weakref
from types import SimpleNamespace

import pytest

from forge.tasks import common, instruct
from forge.tuning.plan import TrainPlan


def _plan(batch: int = 4, accum: int = 4) -> TrainPlan:
    return TrainPlan(
        lora_r=32, lora_alpha=64, lora_dropout=0.05,
        learning_rate=1.5e-4, per_device_batch_size=batch,
        grad_accum_steps=accum, max_seq_len=4096, num_epochs=2,
        warmup_ratio=0.03, weight_decay=0.0,
        optimizer="adamw_torch_fused", lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=False, bf16=True, fp16=False, strategy="lora",
    )


def test_only_approved_geometry_and_falcon_recipe():
    source = _plan()
    plans = instruct._plans(source)
    assert [(p.per_device_batch_size, p.grad_accum_steps) for p in plans] == [
        (4, 4), (2, 8), (1, 16)
    ]
    assert [(p.per_device_batch_size, p.grad_accum_steps)
            for p in instruct._plans(_plan(8, 8))] == [(4, 4), (2, 8), (1, 16)]
    for candidate in plans:
        assert candidate.lora_r == source.lora_r
        assert candidate.lora_alpha == source.lora_alpha
        assert candidate.learning_rate == source.learning_rate
        assert candidate.max_seq_len == source.max_seq_len
        assert candidate.per_device_batch_size * candidate.grad_accum_steps == 16


def test_real_p99_and_worst_rows_are_selected():
    rows = [{"input_ids": list(range(n))} for n in range(1, 101)]
    selected = instruct._probe_rows(rows, 4)
    assert [item[0] for item in selected] == ["p99", "worst"]
    assert 99 in selected[0][2]["row_lengths"]
    assert selected[1][2]["row_lengths"] == [100, 99, 98, 97]
    assert all(row is rows[index] for row, index in
               zip(selected[0][1], selected[0][2]["row_indices"]))


def test_probe_is_real_production_trainer_and_generations_are_separate():
    factory = inspect.getsource(instruct._make_trainer)
    assert "build_training_kwargs(" in factory
    assert 'kwargs["max_steps"] = 1' in factory
    assert "TrainingArguments(" in factory
    assert "KLSFTTrainer" in factory and "Trainer(**fields)" in factory
    probe = inspect.getsource(instruct._probe_once)
    assert "trainer.train()" in probe and "global_step" in probe
    run = inspect.getsource(instruct.run)
    first = run.index("timing_model = rebuild(plan)")
    discard = run.index("_discard(holder.pop())", first)
    real = run.index("holder = [rebuild(plan)]", discard)
    assert first < discard < real


def test_admission_discards_every_probe_and_selects_two(monkeypatch):
    import torch

    calls, rebuilds = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def probe(plan, model, rows, identity, make):
        calls.append((plan.per_device_batch_size, identity["label"], model))
        instruct._discard(model)
        status = "HEADROOM_EXCEEDED" if plan.per_device_batch_size == 4 else "PASS"
        return {**identity, "status": status}

    def rebuild(plan):
        rebuilds.append(plan.per_device_batch_size)
        return f"fresh-{plan.per_device_batch_size}-{len(rebuilds)}"

    monkeypatch.setattr(instruct, "_probe_once", probe)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    plan, attempts = instruct._admit(
        _plan(), "initial", [{"input_ids": [1]}], rebuild, object()
    )
    assert (plan.per_device_batch_size, plan.grad_accum_steps) == (2, 8)
    assert calls == [
        (4, "p99", "initial"),
        (2, "p99", "fresh-2-1"),
        (2, "worst", "fresh-2-2"),
    ]
    assert [item["status"] for item in attempts] == ["HEADROOM_EXCEEDED", "PASS"]


def test_discard_releases_trainer_model_optimizer_and_accelerator(monkeypatch):
    freed = []

    class Accelerator:
        def free_memory(self):
            freed.append(True)

    class Value:
        pass

    class Trainer:
        pass

    model, optimizer, trainer = Value(), Value(), Trainer()
    model_ref, optimizer_ref, trainer_ref = map(
        weakref.ref, (model, optimizer, trainer)
    )
    trainer.accelerator, trainer.model = Accelerator(), model
    trainer.model_wrapped, trainer.optimizer = model, optimizer
    trainer.lr_scheduler = Value()
    del model, optimizer
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    holder = [trainer]
    del trainer
    instruct._discard(holder.pop(), trainer=True)
    assert freed == [True]
    assert trainer_ref() is model_ref() is optimizer_ref() is None


def test_truth_requires_real_schedule_completion():
    assert instruct._truth(0, 10) == common.ARTIFACT_FLOOR
    assert instruct._truth(7, 10) == common.ARTIFACT_PARTIAL_TRAINED_BEST
    assert instruct._truth(10, 10) == common.ARTIFACT_COMPLETE_BEST
    assert instruct._truth(7, 0) == common.ARTIFACT_PARTIAL_TRAINED_BEST


class _Trainer:
    def __init__(self, model, oom_step):
        self.model = model
        self.state = SimpleNamespace(global_step=oom_step or 0)
        self.oom_step = oom_step
        self.accelerator = SimpleNamespace(free_memory=lambda: None)

    def train(self):
        if self.oom_step is not None:
            raise RuntimeError("CUDA out of memory")


def test_zero_progress_oom_rebuilds_next_trainer(monkeypatch):
    builds, rebuilds = [], []

    def make(plan, model, _rows, _epochs):
        builds.append((plan.per_device_batch_size, model))
        oom = 0 if plan.per_device_batch_size == 4 else None
        return _Trainer(model, oom), common.BestTracker(), None

    def rebuild(plan):
        rebuilds.append(plan.per_device_batch_size)
        return f"fresh-{plan.per_device_batch_size}"

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    trainer, _, _, selected = instruct._train_ladder(
        _plan(), "initial", rebuild, make,
        SimpleNamespace(output_dir="/unused"), object(), None
    )
    assert builds == [(4, "initial"), (2, "fresh-2")]
    assert rebuilds == [2]
    assert trainer.model == "fresh-2"
    assert selected.per_device_batch_size == 2


def test_zero_progress_model_is_released_before_lower_geometry_rebuild(monkeypatch):
    references = []

    class Model:
        def __init__(self):
            references.append(weakref.ref(self))

        def zero_grad(self, **_kwargs):
            pass

    def make(plan, model, _rows, _epochs):
        oom = 0 if plan.per_device_batch_size == 4 else None
        return _Trainer(model, oom), common.BestTracker(), None

    def rebuild(_plan):
        assert references[0]() is None
        return Model()

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    holder = [Model()]
    trainer, _, _, _ = instruct._train_ladder(
        _plan(), holder.pop(), rebuild, make,
        SimpleNamespace(output_dir="/unused"), object(), None
    )
    assert trainer.model is not None


def test_progressed_oom_preserves_and_never_retries(monkeypatch):
    preserved, rebuilds = [], []

    def make(_plan, model, _rows, _epochs):
        return _Trainer(model, 7), common.BestTracker(), None

    monkeypatch.setattr(
        instruct, "_preserve_progress",
        lambda trainer, tracker, spec, tokenizer, step:
            preserved.append((trainer.model, step)),
    )
    with pytest.raises(RuntimeError, match="after progress"):
        instruct._train_ladder(
            _plan(), "trained", lambda plan: rebuilds.append(plan),
            make, SimpleNamespace(output_dir="/unused"), object(), None
        )
    assert preserved == [("trained", 7)]
    assert rebuilds == []


def _artifact(path: str) -> None:
    os.makedirs(path)
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    with open(os.path.join(path, "model.safetensors"), "wb") as handle:
        handle.write(len(header).to_bytes(8, "little") + header + b"data")


def test_truth_relabel_is_atomic_and_does_not_change_weights(tmp_path):
    output = str(tmp_path / "artifact")
    _artifact(output)
    before = (tmp_path / "artifact" / "model.safetensors").read_bytes()
    assert common.write_artifact_truth(
        output, common.ARTIFACT_PARTIAL_TRAINED_BEST,
        optimizer_step=4, reason="trained"
    )
    truth = json.loads(
        (tmp_path / "artifact" / "forge_artifact_truth.json").read_text()
    )
    assert truth["truth"] == "PARTIAL_TRAINED_BEST"
    assert truth["optimizer_step"] == 4
    assert (tmp_path / "artifact" / "model.safetensors").read_bytes() == before


def test_frozen_recipe_files_are_unchanged():
    expected = {
        "forge/tuning/plan.py": "302b0c1da3de865ed08af186cf0b2fc8452173dc9a04ab88eee41a1de6578787",
        "forge/tasks/dpo.py": "25bb3d48c2ff1a568d434d3983d4384af1e060967072429cbab68a0b74e1ccd5",
        "forge/tasks/grpo.py": "16ecd365a984b807b7849eb86337092b789ea5bea1e2585768fd5f7cdc92897d",
    }
    for path, digest in expected.items():
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == digest
