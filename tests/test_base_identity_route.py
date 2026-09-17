"""Candidate D (2026-09-17): the field route admits Qwen2.5 only when the delivered base is
byte-identical to a shipped public checkpoint table, and only with FORGE_V2_FIELD_IDENTICAL_BASE=1.
Damaged bases (validator augmentation) keep the production adapter, where it won on every damage type."""
import json
import struct
from pathlib import Path

import pytest

from forge.routing import base_identity as bid
from forge.tasks import sft_v2

Q05_CONFIG = dict(model_type="qwen2", architectures=["Qwen2ForCausalLM"], hidden_size=896, intermediate_size=4864,
                  num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
                  tie_word_embeddings=True)


def _write_safetensors(path: Path, tensors: dict[str, bytes], shape=(2, 2), dtype="F32") -> None:
    header, blob, off = {}, b"", 0
    for name, data in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hb = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)) + hb + blob)


def _model_dir(tmp_path: Path, name: str, config: dict, tensors: dict[str, bytes]) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    _write_safetensors(d / "model.safetensors", tensors)
    return d


def _tables(tmp_path: Path, key: str, public_dir: Path, config: dict) -> Path:
    t = tmp_path / "tables"
    t.mkdir(exist_ok=True)
    (t / f"{key}.json").write_text(json.dumps(bid.build_table(public_dir)))
    (t / f"{key}.config.json").write_text(json.dumps(config))
    return t


def _f32(*vals: float) -> bytes:
    return struct.pack("<%df" % len(vals), *vals)


def test_identical_altered_and_unknown_bases(tmp_path):
    public = _model_dir(tmp_path, "public", Q05_CONFIG, {"a.weight": _f32(1, 2, 3, 4), "b.weight": _f32(5, 6, 7, 8)})
    tables = _tables(tmp_path, "Qwen__Qwen2.5-0.5B-Instruct", public, Q05_CONFIG)
    same = _model_dir(tmp_path, "delivered-same", Q05_CONFIG, {"a.weight": _f32(1, 2, 3, 4), "b.weight": _f32(5, 6, 7, 8)})
    rep = bid.identify(same, tables_dir=tables)
    assert rep["status"] == "identical" and rep["candidate"] == "Qwen__Qwen2.5-0.5B-Instruct"
    assert rep["n_tensors"] == 2 and rep["altered"] == 0 and rep["missing"] == 0 and rep["extra"] == 0
    # validator-style scaling of one matrix: the same shape and dtype, different bytes
    scaled = _model_dir(tmp_path, "delivered-scaled", Q05_CONFIG,
                        {"a.weight": _f32(0.63, 1.26, 1.89, 2.52), "b.weight": _f32(5, 6, 7, 8)})
    rep = bid.identify(scaled, tables_dir=tables)
    assert rep["status"] == "altered" and rep["altered"] == 1 and rep["altered_names"] == ["a.weight"]
    # a different architecture never matches a table
    other = _model_dir(tmp_path, "delivered-other", dict(Q05_CONFIG, hidden_size=1536),
                       {"a.weight": _f32(1, 2, 3, 4), "b.weight": _f32(5, 6, 7, 8)})
    rep = bid.identify(other, tables_dir=tables)
    assert rep["status"] == "unknown" and rep["reason"] == "no_matching_table"
    # missing / extra tensors are not identical either
    fewer = _model_dir(tmp_path, "delivered-fewer", Q05_CONFIG, {"a.weight": _f32(1, 2, 3, 4)})
    assert bid.identify(fewer, tables_dir=tables)["status"] == "altered"
    # oversized models are skipped, never hashed
    rep = bid.identify(same, tables_dir=tables, max_bytes=8)
    assert rep["status"] == "skipped"
    # never raises: a directory without config.json
    assert bid.identify(tmp_path / "nowhere", tables_dir=tables)["status"] == "unknown"


def test_shipped_qwen_tables_are_complete_and_keyed_by_public_configs():
    keys = {"Qwen__Qwen2.5-0.5B-Instruct": 290, "Qwen__Qwen2.5-1.5B-Instruct": 338, "Qwen__Qwen2.5-3B-Instruct": 434}
    for key, n in keys.items():
        table = json.load(open(bid.TABLES_DIR / f"{key}.json"))
        assert table["n_tensors"] == n == len(table["tensors"])
        assert all(len(v["sha256"]) == 64 for v in table["tensors"].values())
        cfg = json.load(open(bid.TABLES_DIR / f"{key}.config.json"))
        assert cfg["model_type"] == "qwen2" and cfg["vocab_size"] == 151936
        assert bid.candidate_keys(cfg) == [key]
    assert bid.candidate_keys(dict(Q05_CONFIG, model_type="llama")) == []


def _gate(**kw):
    base = dict(params_b=0.494, n_gpus=1, is_kl=False, model_type="qwen2", n_rows=29000, total_tokens=9_707_580,
                vocab_size=151936)
    base.update(kw)
    return sft_v2.field_full_route(**base)


def test_field_gate_admits_qwen2_only_on_an_identical_base_and_only_when_enabled(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FIELD", "1")
    for k in ("FORGE_V2_FIELD_TYPES", "FORGE_V2_FIELD_MIN_ROWS", "FORGE_V2_FIELD_MIN_TOKENS", "FORGE_V2_FIELD_IDENTICAL_BASE"):
        monkeypatch.delenv(k, raising=False)
    # release default ON: the validator runs our image with its own environment, so the switch lives in code;
    # FORGE_V2_FIELD_IDENTICAL_BASE=0 restores the pre-D behaviour (qwen2 keeps the adapter whatever the identity says)
    assert sft_v2._field_identical_base() is True
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_BASE", "0")
    assert _gate(base_identity="identical") is False
    assert _gate(base_identity=None) is False
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_BASE", "1")
    # ru-AAQG-29k shape (9.7M tokens) on a byte-identical Qwen2.5-0.5B -> full weights
    assert _gate(base_identity="identical") is True
    # any other identity verdict keeps the adapter (validator-damaged bases lost 1.3–2.9% on full weights)
    for status in ("altered", "unknown", "skipped", "error", None):
        assert _gate(base_identity=status) is False
    # the other field gates still apply on an identical base
    assert _gate(base_identity="identical", total_tokens=1_200_000) is False  # Alpaca-20k volume: parity, stays put
    assert _gate(base_identity="identical", n_rows=8000) is False
    assert _gate(base_identity="identical", vocab_size=32000) is False  # not a Qwen2.5 vocabulary
    assert _gate(base_identity="identical", params_b=7.0) is False  # above the field cap
    # validated at 0.5B only: Qwen2.5-1.5B/3B keep the adapter even on an identical base (3B lost −1.5% on ru-AAQG)
    assert _gate(base_identity="identical", params_b=1.54) is False
    assert _gate(base_identity="identical", params_b=3.09) is False
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B", "3.5")
    assert _gate(base_identity="identical", params_b=3.09) is True
    monkeypatch.delenv("FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B")
    # the experiment override (FORGE_V2_FIELD_TYPES=qwen2) still forces the family regardless of identity
    monkeypatch.setenv("FORGE_V2_FIELD_TYPES", "qwen2")
    assert _gate(base_identity=None) is True
    monkeypatch.delenv("FORGE_V2_FIELD_TYPES")
    # llama is untouched by the identity switch
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=128256) is True
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=128256, base_identity="altered") is True


def test_qwen2_field_defaults_reproduce_the_validated_arms(monkeypatch):
    for k in ("FORGE_V2_FIELD_BF16", "FORGE_V2_FIELD_BF16_MIN_B", "FORGE_V2_FIELD_LR_MULT", "FORGE_V2_FIELD_IDENTICAL_BASE"):
        monkeypatch.delenv(k, raising=False)
    # switch off (=0): the pre-D defaults are untouched for every family, qwen2 included
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_BASE", "0")
    assert sft_v2._field_bf16_enabled("qwen2") is False and sft_v2._field_bf16_min_b("qwen2") == 2.0
    assert sft_v2._field_lr_mult("qwen2") == 1.0
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_BASE", "1")
    # every validated Qwen2.5-0.5B full-weight arm ran bf16 + paged 8-bit AdamW from 0.3 B at half the raw law
    assert sft_v2._field_bf16_enabled("qwen2") is True and sft_v2._field_bf16_min_b("qwen2") == 0.3
    assert sft_v2._field_lr_mult("qwen2") == 0.5
    # other families keep their deployed defaults
    assert sft_v2._field_bf16_enabled("llama") is False and sft_v2._field_bf16_min_b("llama") == 2.0
    assert sft_v2._field_lr_mult("llama") == 1.0 and sft_v2._field_bf16_min_b() == 2.0
    # env still overrides everything
    monkeypatch.setenv("FORGE_V2_FIELD_LR_MULT", "0.25")
    monkeypatch.setenv("FORGE_V2_FIELD_BF16_MIN_B", "1.0")
    monkeypatch.setenv("FORGE_V2_FIELD_BF16", "0")
    assert sft_v2._field_lr_mult("qwen2") == 0.25 and sft_v2._field_bf16_min_b("qwen2") == 1.0
    assert sft_v2._field_bf16_enabled("qwen2") is False


def test_identical_base_admission_is_long_rows_only_by_default(monkeypatch):
    monkeypatch.delenv("FORGE_V2_FIELD_IDENTICAL_ROWS", raising=False)
    # long-row tasks (median >= 256 tokens) are where production runs the v2 full strategy that lost 5.2% to the field route
    assert sft_v2.identical_base_applies(True) is True
    # short-row tasks keep the legacy adapter (field route +1.3% worse on the Sept-14-shaped fixture)
    assert sft_v2.identical_base_applies(False) is False
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_ROWS", "all")
    assert sft_v2.identical_base_applies(False) is True
    monkeypatch.setenv("FORGE_V2_FIELD_IDENTICAL_ROWS", "bogus")
    assert sft_v2._field_identical_rows_policy() == "long" and sft_v2.identical_base_applies(False) is False


def test_only_the_tested_checkpoint_is_admitted():
    # the 1.5B / 3B tables ship for telemetry, but only the checkpoint whose field win was measured may switch routes
    assert sft_v2.FIELD_IDENTICAL_BASE_CANDIDATES == frozenset({"Qwen__Qwen2.5-0.5B-Instruct"})
    for key in ("Qwen__Qwen2.5-1.5B-Instruct", "Qwen__Qwen2.5-3B-Instruct"):
        assert (bid.TABLES_DIR / f"{key}.json").exists() and key not in sft_v2.FIELD_IDENTICAL_BASE_CANDIDATES
    # the dispatcher maps an identical-but-untested candidate to a non-admitting status
    src = (Path(__file__).resolve().parents[1] / "forge" / "tasks" / "instruct.py").read_text()
    assert "identical_untested" in src and "FIELD_IDENTICAL_BASE_CANDIDATES" in src
    assert _gate(base_identity="identical_untested") is False
