"""CPU-only tests for offline model preflight and native-first resolution."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from forge import model


def _model_dir(tmp_path, *, config=None):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config or {"model_type": "qwen2"}))
    (path / "model.safetensors").write_bytes(b"placeholder")
    return path


def test_preflight_reports_missing_index_shard(tmp_path):
    path = _model_dir(tmp_path)
    (path / "model.safetensors").unlink()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model-00001-of-00002.safetensors"}})
    )
    with pytest.raises(model.LocalModelLoadError, match="model-00001"):
        model._preflight_model_dir(str(path))


def test_preflight_rejects_index_path_outside_model_dir(tmp_path):
    path = _model_dir(tmp_path)
    (path / "model.safetensors").unlink()
    (tmp_path / "outside.safetensors").write_bytes(b"not allowed")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with pytest.raises(model.LocalModelLoadError, match="outside the model directory"):
        model._preflight_model_dir(str(path))


def test_preflight_allows_hf_snapshot_symlink_to_blob_store(tmp_path):
    path = _model_dir(tmp_path)
    (path / "model.safetensors").unlink()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "weight").write_bytes(b"cached blob")
    (path / "model-00001-of-00001.safetensors").symlink_to(blobs / "weight")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model-00001-of-00001.safetensors"}})
    )
    assert model._preflight_model_dir(str(path))["model_type"] == "qwen2"


def test_tokenizer_uses_native_implementation_before_custom_code(tmp_path):
    path = _model_dir(tmp_path)

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            cls.calls.append(kwargs)
            return "native-tokenizer"

    result = model._load_tokenizer_native_first(Loader, str(path), {})
    assert result == "native-tokenizer"
    assert Loader.calls == [
        {"trust_remote_code": False, "use_fast": True, "local_files_only": True}
    ]


def test_tokenizer_custom_code_is_offline_last_resort(tmp_path):
    path = _model_dir(tmp_path, config={"auto_map": {"AutoTokenizer": "other--repo.X"}})

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            cls.calls.append(kwargs)
            if not kwargs["trust_remote_code"]:
                raise OSError("native class unavailable")
            return "custom-tokenizer"

    result = model._load_tokenizer_native_first(
        Loader, str(path), {"auto_map": {"AutoTokenizer": "other--repo.X"}}
    )
    assert result == "custom-tokenizer"
    assert [call["trust_remote_code"] for call in Loader.calls] == [False, False, True]
    assert all(call["local_files_only"] is True for call in Loader.calls)


def test_custom_code_retry_temporarily_exposes_local_sibling_imports(tmp_path):
    path = _model_dir(tmp_path, config={"auto_map": {"AutoModel": "modeling.X"}})

    class Loader:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            if not kwargs["trust_remote_code"]:
                raise ValueError("native class unavailable")
            assert sys.path[0] == str(path)
            return "custom-model"

    result = model._load_model_native_first(
        Loader,
        str(path),
        config_data={"auto_map": {"AutoModel": "modeling.X"}},
        common={"local_files_only": True},
    )

    assert result == "custom-model"
    assert str(path) not in sys.path


def test_model_loader_prefers_native_sdpa_then_native_eager(tmp_path):
    path = _model_dir(tmp_path)

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            cls.calls.append(kwargs)
            if kwargs["attn_implementation"] == "sdpa":
                raise RuntimeError("sdpa unsupported")
            return "eager-model"

    result = model._load_model_native_first(
        Loader,
        str(path),
        config_data={},
        common={"local_files_only": True},
    )
    assert result == "eager-model"
    assert [(c["trust_remote_code"], c["attn_implementation"]) for c in Loader.calls] == [
        (False, "sdpa"),
        (False, "eager"),
    ]


def test_model_loader_does_not_repeat_an_oom(tmp_path):
    path = _model_dir(tmp_path)

    class Loader:
        calls = 0

        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            cls.calls += 1
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        model._load_model_native_first(
            Loader,
            str(path),
            config_data={},
            common={"local_files_only": True},
        )
    assert Loader.calls == 1


def test_load_diagnostics_include_oserror_and_auto_map(tmp_path):
    path = _model_dir(tmp_path)

    class Loader:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            raise OSError("missing dynamic_module.py")

    with pytest.raises(model.LocalModelLoadError) as caught:
        model._load_tokenizer_native_first(
            Loader, str(path), {"auto_map": {"AutoTokenizer": "foreign.Repo"}}
        )
    message = str(caught.value)
    assert "OSError" in message and "missing dynamic_module.py" in message
    assert "auto_map" in message


def test_effective_sequence_length_respects_models_below_256():
    tiny = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128))
    assert model.effective_seq_len(tiny, 4096) == 128
    assert model.effective_seq_len(tiny, 64) == 64


@pytest.mark.parametrize(
    "positions,expected",
    [(128, 64), (4096, 2048), (8192, 4096)],
)
def test_sft_sequence_length_matches_evaluator_halving_rule(positions, expected):
    base = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=positions))
    assert model.effective_sft_seq_len(base, 4096) == expected


def test_sft_sequence_length_preserves_smaller_baseline_cap():
    base = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    assert model.effective_sft_seq_len(base, 1024) == 1024


def test_full_ft_is_explicit_opt_in_and_still_hardware_gated(monkeypatch):
    monkeypatch.delenv("FORGE_ENABLE_EXPERIMENTAL_FULL_FT", raising=False)
    args = dict(use_kl=False, params_b=1.0, n_gpus=1, per_gpu_gb=80.0)
    assert model.decide_full_finetune(**args) is False
    monkeypatch.setenv("FORGE_ENABLE_EXPERIMENTAL_FULL_FT", "1")
    assert model.decide_full_finetune(**args) is True
    assert model.decide_full_finetune(**{**args, "use_kl": True}) is False
    assert model.decide_full_finetune(**{**args, "n_gpus": 2}) is False
    monkeypatch.setenv("FORGE_ENABLE_EXPERIMENTAL_FULL_FT", "true")
    assert model.decide_full_finetune(**args) is False


def test_quasar_plan_uses_microbatch_one_without_fake_checkpointing():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Plan:
        per_device_batch_size: int = 4
        grad_accum_steps: int = 4
        gradient_checkpointing: bool = True

    quasar = SimpleNamespace(config=SimpleNamespace(model_type="quasar_text"))
    native = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))

    adjusted, changed = model.conservative_quasar_plan(quasar, Plan())
    untouched, native_changed = model.conservative_quasar_plan(native, Plan())

    assert changed is True
    assert adjusted.per_device_batch_size == 1
    assert adjusted.grad_accum_steps == 16
    assert adjusted.gradient_checkpointing is False
    assert native_changed is False and untouched == Plan()
