from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from forge.data.schema import InstructColumns
from forge.tasks import lfm25_full_export as export


class Lfm2MoeForCausalLM:
    def __init__(self, *, model_type="lfm2_moe", architectures=None):
        self.config = SimpleNamespace(
            model_type=model_type,
            architectures=architectures or ["Lfm2MoeForCausalLM"],
        )


def _spec(*, use_kl=False, output="out", task_type="InstructTextTask"):
    return SimpleNamespace(
        task_type=task_type,
        instruct=InstructColumns("instruction", "output"),
        use_kl=use_kl,
        output_dir=output,
        cached_model_dir="/cache/models/anonymous",
    )


def test_native_identity_accepts_modified_weights_without_config_digest():
    model = Lfm2MoeForCausalLM()
    assert export.is_native_lfm2_moe(model)
    model.config.custom_weight_revision = "tournament-perturbed"
    assert export.eligible(_spec(), model)


@pytest.mark.parametrize(
    "model",
    [
        Lfm2MoeForCausalLM(model_type="lfm2"),
        Lfm2MoeForCausalLM(architectures=["Lfm2ForCausalLM"]),
        SimpleNamespace(config=SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])),
    ],
)
def test_non_lfm_native_models_are_rejected(model):
    assert not export.is_native_lfm2_moe(model)
    assert not export.eligible(_spec(), model)


def test_task_scope_rejects_kl_and_non_instruct():
    model = Lfm2MoeForCausalLM()
    assert not export.eligible(_spec(use_kl=True), model)
    assert not export.eligible(_spec(task_type="ChatTask"), model)


def test_adapter_key_inserts_default_before_weight():
    assert export.internal_adapter_key("base.model.lora_A.weight") == "base.model.lora_A.default.weight"
    assert export.internal_adapter_key("base.model.lora_B.weight") == "base.model.lora_B.default.weight"
    with pytest.raises(ValueError, match="unsupported"):
        export.internal_adapter_key("base.model.weight")


def test_export_is_off_by_default_and_does_not_load_or_touch_output(monkeypatch):
    monkeypatch.delenv(export.EXPORT_ENV, raising=False)
    def fail(*_args, **_kwargs):
        raise AssertionError("disabled export loaded a model")
    monkeypatch.setattr(export, "_load_native_base", fail)
    deadline = SimpleNamespace(remaining_hard=lambda: 9999.0)
    assert export.maybe_export(_spec(), deadline) is False


def _adapter_output(path, *, truth="COMPLETE_BEST"):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    (path / "forge_artifact_truth.json").write_text(
        json.dumps({"truth": truth, "optimizer_step": 32})
    )


def test_enabled_success_keeps_backup_and_promotes_via_isolated_stage(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]})
    )
    work = tmp_path / "work"
    spec = _spec(output=str(output))
    spec.cached_model_dir = str(base)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "workdir", lambda _spec: str(work))
    monkeypatch.setattr(export, "_reconstruct_and_promote", lambda *_args: None)
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    assert export.maybe_export(spec, SimpleNamespace(remaining_hard=lambda: 9999.0)) is True
    assert (work / "lfm-moe-full-export-source/adapter_model.safetensors").read_bytes() == b"adapter-bytes"
    assert (output / "adapter_model.safetensors").read_bytes() == b"adapter-bytes"


def test_stage_failure_preserves_selected_adapter_byte_for_byte(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]})
    )
    spec = _spec(output=str(output))
    spec.cached_model_dir = str(base)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "workdir", lambda _spec: str(tmp_path / "work"))
    monkeypatch.setattr(export, "_reconstruct_and_promote", lambda *_args: (_ for _ in ()).throw(RuntimeError("stage")))
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    assert export.maybe_export(spec, SimpleNamespace(remaining_hard=lambda: 9999.0)) is False
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert after == before


def test_insufficient_deadline_does_not_load_weights(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "_load_native_base", lambda *_a: pytest.fail("weight load"))
    monkeypatch.setattr(export, "resolve_model_dir", lambda *_a: pytest.fail("config load"))
    assert export.maybe_export(_spec(output=str(output)), SimpleNamespace(remaining_hard=lambda: 119.0)) is False


def test_untrained_truth_does_not_resolve_or_load_weights(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output, truth="FLOOR_UNTRAINED")
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "resolve_model_dir", lambda *_a: pytest.fail("model resolve"))
    monkeypatch.setattr(export, "_load_native_base", lambda *_a: pytest.fail("weight load"))
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    assert export.maybe_export(_spec(output=str(output)), SimpleNamespace(remaining_hard=lambda: 180.0)) is False


def test_native_loader_does_not_retry_after_oom(monkeypatch, tmp_path):
    calls = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "eos"
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append(_kwargs["attn_implementation"])
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeModel),
    )
    monkeypatch.setattr(export, "pick_dtype", lambda: "fake-dtype")
    with pytest.raises(RuntimeError, match="out of memory"):
        export._load_native_base(tmp_path)
    assert calls == ["sdpa"]


def test_non_lfm_config_does_not_load_weights(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    spec = _spec(output=str(output))
    spec.cached_model_dir = str(base)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "_load_native_base", lambda *_a: pytest.fail("weight load"))
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    assert export.maybe_export(spec, SimpleNamespace(remaining_hard=lambda: 9999.0)) is False


def test_cli_export_exception_does_not_enter_fallback(monkeypatch, tmp_path):
    from forge import cli
    from forge.tasks import dispatch
    from forge.tasks import fallback

    output = tmp_path / "output"
    output.mkdir()
    marker = output / "selected-adapter"
    spec = SimpleNamespace(output_dir=str(output), task_type="InstructTextTask")
    calls = {"fallback": 0}

    def handler(_spec, _deadline):
        marker.write_bytes(b"selected")

    def bad_export(*_args, **_kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(dispatch, "for_task", lambda _task: handler)
    monkeypatch.setattr(export, "maybe_export", bad_export)
    monkeypatch.setattr(fallback, "emit_untrained_copy", lambda *_a, **_k: calls.__setitem__("fallback", 1))
    monkeypatch.setattr(cli.telemetry, "event", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.telemetry, "write_into", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_log", lambda *_a, **_k: None)
    cli._run(spec, SimpleNamespace())
    assert marker.read_bytes() == b"selected"
    assert calls["fallback"] == 0
