"""Structural fallback validation without loading large tensor payloads."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

from forge.tasks import fallback


def _safetensors(path, *, payload=b"\x00\x00\x00\x00"):
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


def _pytorch_zip(path):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", b"\x80\x04state-dict-metadata")
        archive.writestr("archive/data/0", b"tensor-bytes")
        archive.writestr("archive/version", b"3\n")


def test_valid_adapter_requires_config_and_tensor_header(tmp_path):
    out = tmp_path / "adapter"
    out.mkdir()
    _safetensors(out / "adapter_model.safetensors")
    assert fallback._has_weights(str(out)) is False
    (out / "adapter_config.json").write_text("{}")
    assert fallback._has_weights(str(out)) is True
    _safetensors(out / "adapter_model.safetensors", payload=b"\x00")
    assert fallback._has_weights(str(out)) is False  # F32[1] requires four bytes
    (out / "adapter_model.safetensors").write_bytes(b"truncated")
    assert fallback._has_weights(str(out)) is False


def test_sharded_model_rejects_missing_or_truncated_shards(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    (out / "config.json").write_text("{}")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    assert fallback._has_weights(str(out)) is False
    _safetensors(out / "model-00001-of-00001.safetensors")
    assert fallback._has_weights(str(out)) is True
    (out / "model-00001-of-00001.safetensors").write_bytes(b"")
    assert fallback._has_weights(str(out)) is False


def test_bin_validation_is_nonexecuting_and_rejects_arbitrary_bytes(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    (out / "config.json").write_text("{}")
    weights = out / "pytorch_model.bin"
    weights.write_bytes(b"not a torch model")
    assert fallback._has_weights(str(out)) is False
    weights.write_bytes(b"PK\x03\x04placeholder")
    assert fallback._has_weights(str(out)) is False
    weights.write_bytes(b"\x80\x04legacy-pickle")
    assert fallback._has_weights(str(out)) is False
    with zipfile.ZipFile(weights, "w") as archive:
        archive.writestr("unrelated.txt", b"not torch")
    assert fallback._has_weights(str(out)) is False
    _pytorch_zip(weights)
    assert fallback._has_weights(str(out)) is True
    raw = weights.read_bytes()
    weights.write_bytes(raw[:-8])  # no complete EOCD/central-directory trailer
    assert fallback._has_weights(str(out)) is False


def test_plain_copy_reports_only_a_structurally_valid_model(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"corrupt")
    assert fallback._emit_plain_copy(str(src), str(dst)) is False
    _pytorch_zip(src / "pytorch_model.bin")
    assert fallback._emit_plain_copy(str(src), str(dst)) is True


def test_fallback_recovers_interrupted_previous_generation(tmp_path):
    final = tmp_path / "output"
    old = tmp_path / "output.old"
    old.mkdir()
    (old / "config.json").write_text("{}")
    _safetensors(old / "model.safetensors")
    spec = SimpleNamespace(
        output_dir=str(final), cached_model_dir=str(tmp_path / "missing-base")
    )
    fallback.emit_untrained_copy(spec)
    assert fallback._has_weights(str(final)) is True
    assert not old.exists()
