from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.data.schema import InstructColumns
from forge.tasks import gemma4_full_export as export


class Gemma4ForConditionalGeneration:
    def __init__(self, *, model_type="gemma4", nested_type="gemma4_text", vocab=8):
        geometry = dict(export._E2B_TEXT_GEOMETRY)
        geometry["layer_types"] = list(export._E2B_LAYER_TYPES)
        self.config = SimpleNamespace(
            model_type=model_type,
            architectures=["Gemma4ForConditionalGeneration"],
            text_config=SimpleNamespace(model_type=nested_type, vocab_size=vocab, **geometry),
        )
        weight = SimpleNamespace(shape=(vocab, 4))
        self._input = SimpleNamespace(weight=weight)
        self._output = SimpleNamespace(weight=weight)

    def get_input_embeddings(self):
        return self._input

    def get_output_embeddings(self):
        return self._output


class LlamaForCausalLM:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="llama", architectures=["LlamaForCausalLM"], text_config=None
        )


def _spec(*, use_kl=False, output="out", task_type="InstructTextTask"):
    return SimpleNamespace(
        task_type=task_type,
        instruct=InstructColumns("instruction", "output"),
        use_kl=use_kl,
        output_dir=output,
        cached_model_dir="/cache/models/anonymous",
    )


def _adapter_output(path: Path, *, truth="COMPLETE_BEST"):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    (path / "forge_artifact_truth.json").write_text(
        json.dumps({"truth": truth, "optimizer_step": 32}), encoding="utf-8"
    )


def test_native_gate_derives_nested_text_vocab_and_embedding_rows():
    model = Gemma4ForConditionalGeneration()
    assert export.is_native_gemma4_config(model.config)
    assert export.is_native_gemma4(model)
    assert export._validate_native_vocab(model) == 8
    assert export._embeddings_tied(model)
    model.config.text_config.vocab_size = 9
    with pytest.raises(RuntimeError, match="disagrees"):
        export._validate_native_vocab(model)


@pytest.mark.parametrize(
    "model",
    [
        Gemma4ForConditionalGeneration(model_type="llama"),
        Gemma4ForConditionalGeneration(nested_type="gemma_text"),
        LlamaForCausalLM(),
    ],
)
def test_non_gemma4_families_are_rejected(model):
    assert not export.is_native_gemma4(model)
    assert not export.eligible(_spec(), model)


def test_task_scope_rejects_kl_and_non_instruct():
    model = Gemma4ForConditionalGeneration()
    assert not export.eligible(_spec(use_kl=True), model)
    assert not export.eligible(_spec(task_type="ChatTask"), model)
    assert not export.eligible(_spec(), SimpleNamespace(config=model.config))


def test_alias_changes_only_missing_top_level_field_and_restores():
    nested = {"model_type": "gemma4_text", "vocab_size": 8, "hidden_size": 64}
    config = SimpleNamespace(
        model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"], text_config=nested
    )
    before = dict(vars(config))
    present, old = export._alias_only_config(config, 8)
    assert not present and old is None and config.vocab_size == 8
    assert vars(config)["text_config"] == nested
    export._restore_alias(config, present, old)
    assert vars(config) == before


def test_source_config_replacement_changes_only_truthful_alias():
    source = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "text_config": {"model_type": "gemma4_text", "vocab_size": 8, "nested": {"keep": True}},
    }
    staged = export._config_with_source_alias(source, 8)
    assert staged["vocab_size"] == 8
    assert {key: value for key, value in staged.items() if key != "vocab_size"} == source
    assert staged["text_config"] == source["text_config"]
    with pytest.raises(RuntimeError, match="disagrees"):
        export._config_with_source_alias({**source, "vocab_size": 7}, 8)


def test_staged_alias_requires_nested_identity(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    config = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"model_type": "gemma4_text", "vocab_size": 8},
        "vocab_size": 8,
    }
    (stage / "config.json").write_text(json.dumps(config), encoding="utf-8")
    export._validate_staged_alias(stage, 8)
    config["vocab_size"] = 7
    (stage / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact top-level"):
        export._validate_staged_alias(stage, 8)


def test_e2b_gate_rejects_same_family_wrong_layer_geometry():
    model = Gemma4ForConditionalGeneration()
    model.config.text_config.layer_types[4] = "sliding_attention"
    assert not export.is_native_gemma4(model)


def test_disk_bytes_counts_symlinked_weight_once_and_skips_directory_links(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = snapshot / "blobs" / "weight.bin"
    blob.parent.mkdir()
    blob.write_bytes(b"12345")
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "duplicate.safetensors").symlink_to(blob)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "other.bin").write_bytes(b"not-followed")
    (snapshot / "linked-directory").symlink_to(outside, target_is_directory=True)
    assert export._regular_bytes(snapshot) == 5


def test_full_keyset_checks_unadapted_values():
    torch = pytest.importorskip("torch")

    class State:
        def __init__(self, values):
            self.values = values

        def state_dict(self):
            return self.values

    base = State({"adapted": torch.tensor([1.0]), "untouched": torch.tensor([2.0])})
    merged = State({"adapted": torch.tensor([3.0]), "untouched": torch.tensor([2.0])})
    export._validate_full_keyset(base, merged, {"untouched"})
    merged.values["untouched"] = torch.tensor([4.0])
    with pytest.raises(RuntimeError, match="unadapted"):
        export._validate_full_keyset(base, merged, {"untouched"})


def test_default_off_does_not_resolve_or_touch_output(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    monkeypatch.delenv(export.EXPORT_ENV, raising=False)
    monkeypatch.setattr(export, "resolve_model_dir", lambda *_a: pytest.fail("disabled load"))
    assert export.maybe_export(_spec(output=str(output)), SimpleNamespace(remaining_hard=lambda: 9999.0)) is False


def test_failed_export_preserves_selected_adapter_and_keeps_backup(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                    "model_type": "gemma4",
                    "architectures": ["Gemma4ForConditionalGeneration"],
                    "text_config": {
                        "model_type": "gemma4_text", "vocab_size": 8,
                        **export._E2B_TEXT_GEOMETRY,
                    "layer_types": list(export._E2B_LAYER_TYPES),
                    },
            }
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    spec = _spec(output=str(output))
    spec.cached_model_dir = str(base)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "workdir", lambda _spec: str(work))
    monkeypatch.setattr(
        export,
        "_reconstruct_and_promote",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("conversion")),
    )
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    assert export.maybe_export(spec, SimpleNamespace(remaining_hard=lambda: 9999.0)) is False
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert after == before
    assert (work / "gemma4-full-export-source/adapter_model.safetensors").read_bytes() == b"adapter-bytes"


def test_failed_export_does_not_cross_route(monkeypatch, tmp_path):
    output = tmp_path / "output"
    _adapter_output(output)
    monkeypatch.setenv(export.EXPORT_ENV, "1")
    monkeypatch.setattr(export, "resolve_model_dir", lambda *_a: pytest.fail("non-native route loaded"))
    monkeypatch.setattr(export, "telemetry", SimpleNamespace(event=lambda *_a, **_k: None))
    non_gemma = tmp_path / "base"
    non_gemma.mkdir()
    (non_gemma / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    spec = _spec(output=str(output))
    spec.cached_model_dir = str(non_gemma)
    # The config gate is deliberately checked after resolving its JSON but before
    # loading any model weights.
    monkeypatch.setattr(export, "resolve_model_dir", lambda *_a: str(non_gemma))
    assert export.maybe_export(spec, SimpleNamespace(remaining_hard=lambda: 9999.0)) is False
