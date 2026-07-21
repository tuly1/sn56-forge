"""Failure-path tests for trainer artifact selection and promotion.

These are CPU-only and intentionally use small stand-ins: the invariants under
test are filesystem/state transitions, not Transformers numerical behaviour.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from forge.tasks import common


class _TrainerCallback:
    pass


@pytest.fixture
def callback_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(TrainerCallback=_TrainerCallback)
    )


def test_best_is_persisted_only_after_success_and_nonfinite_is_rejected(
    tmp_path, monkeypatch, callback_module
):
    tracker = common.BestTracker()
    attempts: list[float] = []

    def flaky_save(_model, _tokenizer, _output):
        attempts.append(tracker.last)
        if len(attempts) == 1:
            raise OSError("disk full")

    monkeypatch.setattr(common, "save_adapter", flaky_save)
    spec = SimpleNamespace(output_dir=str(tmp_path / "out"))
    callback = common._make_best_checkpoint_callback(spec, object(), tracker)
    state = SimpleNamespace(global_step=10)
    control = SimpleNamespace()

    callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.0}, model=object())
    assert tracker.observed_best == 1.0
    assert tracker.persisted_best is None

    # Although 1.1 is worse than the observed minimum, it is the best checkpoint
    # we can actually persist and must get another export attempt.
    state.global_step = 20
    callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.1}, model=object())
    assert attempts == [1.0, 1.1]
    assert tracker.persisted_best == 1.1
    assert tracker.persisted_best_step == 20

    state.global_step = 30
    callback.on_evaluate(None, state, control, metrics={"eval_loss": float("nan")}, model=object())
    assert tracker.last == 1.1 and tracker.last_step == 20
    assert tracker.persisted_best == 1.1
    callback.on_evaluate(None, state, control, metrics={"eval_loss": "not-a-number"}, model=object())
    assert tracker.last == 1.1 and tracker.persisted_best == 1.1


def test_periodic_save_stops_only_after_a_persisted_best(monkeypatch, callback_module):
    tracker = common.BestTracker()
    calls: list[int] = []
    monkeypatch.setattr(
        common, "save_adapter", lambda *_args: calls.append(1)
    )
    callback = common._make_periodic_save_callback(
        SimpleNamespace(output_dir="/unused"), object(), every=5, tracker=tracker
    )
    state, control = SimpleNamespace(global_step=5), SimpleNamespace()
    tracker.observed_best = 1.0
    callback.on_step_end(None, state, control, model=object())
    assert len(calls) == 1
    tracker.persisted_best = 1.2
    callback.on_step_end(None, state, control, model=object())
    assert len(calls) == 1


def test_unevaluated_final_never_replaces_measured_best():
    tracker = common.BestTracker()
    tracker.persisted_best = 1.0
    tracker.persisted_best_step = 100
    tracker.last = 1.0
    tracker.last_step = 100
    assert common.should_final_save(tracker, final_step=100) is True
    assert common.should_final_save(tracker, final_step=125) is False
    assert common.should_final_save(tracker) is False


def _artifact(path, marker: bytes) -> None:
    os.makedirs(path, exist_ok=True)
    payload = marker[:4].ljust(4, b"\x00")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    with open(os.path.join(path, "model.safetensors"), "wb") as fh:
        fh.write(len(header).to_bytes(8, "little") + header + payload)


def _artifact_payload(path) -> bytes:
    with open(os.path.join(path, "model.safetensors"), "rb") as fh:
        return fh.read()[-4:]


def test_portable_promotion_rolls_back_if_second_rename_fails(tmp_path, monkeypatch):
    final, staged = str(tmp_path / "model"), str(tmp_path / "model.tmp")
    _artifact(final, b"old")
    _artifact(staged, b"new")
    common._write_ready_marker(staged)
    monkeypatch.setattr(
        common,
        "_rename_exchange",
        lambda *_args: (_ for _ in ()).throw(
            common._AtomicExchangeUnavailable(95, "unsupported")
        ),
    )
    real_replace = os.replace
    calls = 0

    def fail_new_generation(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        return real_replace(src, dst)

    monkeypatch.setattr(common.os, "replace", fail_new_generation)
    with pytest.raises(OSError, match="injected"):
        common._promote_staged_dir(staged, final)
    assert _artifact_payload(final) == b"old\x00"
    assert os.path.isdir(staged)


def test_startup_recovery_prefers_last_live_generation(tmp_path):
    final = str(tmp_path / "model")
    _artifact(final + ".old", b"old")
    _artifact(final + ".tmp", b"new-uncommitted")
    common._recover_artifact_dirs(final)
    assert _artifact_payload(final) == b"old\x00"
    assert not os.path.exists(final + ".old")
    assert not os.path.exists(final + ".tmp")


def test_partial_sharded_stage_is_never_recovered(tmp_path):
    final = str(tmp_path / "model")
    staged = final + ".tmp"
    os.makedirs(staged)
    # A shard written before save_pretrained failed is not a complete model when
    # its index never landed.
    with open(os.path.join(staged, "model-00001-of-00002.safetensors"), "wb") as fh:
        fh.write(b"partial")
    assert common._is_structurally_complete_artifact(staged) is False
    common._recover_artifact_dirs(final)
    assert not os.path.exists(final)
    assert not os.path.exists(staged)


def test_complete_but_unmarked_tmp_is_never_recovered(tmp_path):
    final = str(tmp_path / "model")
    staged = final + ".tmp"
    _artifact(staged, b"new")
    assert common._is_structurally_complete_artifact(staged) is True
    assert common._has_ready_marker(staged) is False
    common._recover_artifact_dirs(final)
    assert not os.path.exists(final)
    assert not os.path.exists(staged)


def test_complete_marked_tmp_is_recovered_and_marker_remains(tmp_path):
    final = str(tmp_path / "model")
    staged = final + ".tmp"
    _artifact(staged, b"new")
    common._write_ready_marker(staged)
    assert common._has_ready_marker(staged) is True
    common._recover_artifact_dirs(final)
    assert _artifact_payload(final) == b"new\x00"
    assert common._has_ready_marker(final) is True
    assert not os.path.exists(staged)


def test_save_failure_before_stage_is_ready_cannot_be_recovered(tmp_path):
    final = str(tmp_path / "model")
    _artifact(final, b"old")

    class Model:
        def save_pretrained(self, path, **_kwargs):
            _artifact(path, b"new-but-incomplete")

    class Tokenizer:
        def save_pretrained(self, _path):
            raise OSError("tokenizer write failed")

    with pytest.raises(OSError, match="tokenizer write failed"):
        common.save_adapter(Model(), Tokenizer(), final)
    assert _artifact_payload(final) == b"old\x00"
    assert not os.path.exists(final + ".tmp")


def test_save_adapter_exports_only_trained_default_peft_adapter(tmp_path):
    final = str(tmp_path / "model")
    observed = {}

    class Model:
        peft_config = {"default": object(), "ref": object()}

        def save_pretrained(self, path, **kwargs):
            observed.update(kwargs)
            _artifact(path, b"policy")
            if kwargs.get("selected_adapters") != ["default"]:
                _artifact(os.path.join(path, "ref"), b"reference")

    class Tokenizer:
        def save_pretrained(self, path):
            with open(os.path.join(path, "tokenizer_config.json"), "w") as fh:
                json.dump({}, fh)

    common.save_adapter(Model(), Tokenizer(), final)

    assert observed["safe_serialization"] is True
    assert observed["selected_adapters"] == ["default"]
    assert not os.path.exists(os.path.join(final, "ref"))


class _Resettable:
    def __init__(self):
        self.reset = False

    def zero_grad(self, **_kwargs):
        self.reset = True


class _FakeTrainer:
    def __init__(self, *, step=0, batch=4, derived_batch_geometry=False):
        self.args = SimpleNamespace(
            per_device_train_batch_size=batch, gradient_accumulation_steps=2
        )
        if derived_batch_geometry:
            self.args.generation_batch_size = batch * 2
            self.args.steps_per_generation = 1
        self.state = SimpleNamespace(global_step=step)
        self.control = SimpleNamespace()
        self.model = _Resettable()
        self.optimizer = _Resettable()
        self.lr_scheduler = object()
        self._train_batch_size = batch
        self.seen_train_batch_sizes = []
        self.calls = 0

    def train(self):
        self.calls += 1
        self.seen_train_batch_sizes.append(self._train_batch_size)
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")


def test_oom_retry_resets_zero_step_trainer_and_shrinks_batch(monkeypatch):
    monkeypatch.setattr(common, "_is_oom", lambda _exc: True)
    monkeypatch.setattr(common, "_free_cuda", lambda: None)
    trainer = _FakeTrainer()
    original_model = trainer.model
    common.safe_train(trainer)
    assert trainer.calls == 2
    assert trainer.args.per_device_train_batch_size == 2
    assert trainer._train_batch_size == 2
    assert trainer.seen_train_batch_sizes == [4, 2]
    assert trainer.args.gradient_accumulation_steps == 2
    assert original_model.reset is True
    assert trainer.optimizer is None and trainer.lr_scheduler is None


@pytest.mark.parametrize("step,batch,min_batch", [(1, 4, 1), (0, 2, 2)])
def test_oom_does_not_retry_progressed_or_unchanged_batch(
    step, batch, min_batch, monkeypatch
):
    monkeypatch.setattr(common, "_is_oom", lambda _exc: True)
    trainer = _FakeTrainer(step=step, batch=batch)
    with pytest.raises(RuntimeError, match="out of memory"):
        common.safe_train(trainer, min_batch=min_batch)
    assert trainer.calls == 1


def test_oom_does_not_mutate_trainer_with_derived_batch_geometry(monkeypatch):
    from forge import telemetry

    events = []
    monkeypatch.setattr(common, "_is_oom", lambda _exc: True)
    monkeypatch.setattr(
        telemetry, "event", lambda name, **fields: events.append((name, fields))
    )
    trainer = _FakeTrainer(batch=4, derived_batch_geometry=True)
    with pytest.raises(RuntimeError, match="out of memory"):
        common.safe_train(trainer)
    assert trainer.calls == 1
    assert trainer.args.per_device_train_batch_size == 4
    assert events[-1][0] == "oom_retry_skipped"
    assert events[-1][1]["reason"] == "trainer_requires_rebuild"
