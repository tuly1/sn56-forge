"""Release defect D1 (2026-09-29): every full-FT family's export must pass the validator's OWN is_finetune logic.

For each model_type the full-FT default can route (research/20260929-release-fullft-default/ROUTE-TABLE.md, in-scope
rows), a tiny random base is saved with the base's architecture class, loaded the way forge trains it
(AutoModelForCausalLM), perturbed (a "trained" weight), exported through the production save path
(common.save_adapter), then loaded the way the validator loads a non-LoRA submission
(AutoModelForCausalLM.from_pretrained(repo, dtype=bfloat16, trust_remote_code=False); G.O.D a2b14db
evaluators/instruct_text.py:446-455) and checked with the VENDORED G.O.D model_is_a_finetune (tests/god_vendor).
The export must pass AND stay loadable with identical logits to the trained model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from forge.tasks import common, export_compat
from god_vendor.model_checks_a2b14db import model_is_a_finetune

T = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
         vocab_size=256, max_position_embeddings=512, head_dim=16, pad_token_id=0, bos_token_id=1, eos_token_id=2)

# model_type -> (causal-LM architecture of the public base, extra config kwargs)
FAMILIES: dict[str, tuple[str, dict]] = {
    "llama": ("LlamaForCausalLM", {}),               # Llama-3.2, TinyLlama, SmolLM/SmolLM2, open_llama, MiniCPM5
    "mistral": ("MistralForCausalLM", {}),           # unsloth/Phi-3-mini-4k-instruct is model_type mistral
    "qwen2": ("Qwen2ForCausalLM", {}),
    "qwen3": ("Qwen3ForCausalLM", {}),
    "lfm2": ("Lfm2ForCausalLM", {"layer_types": ["conv", "full_attention"]}),
    "gemma": ("GemmaForCausalLM", {}),
    "gemma2": ("Gemma2ForCausalLM", {}),
    "granite": ("GraniteForCausalLM", {}),
    "phi": ("PhiForCausalLM", {}),
    "phi3": ("Phi3ForCausalLM", {}),                 # NuExtract-v1.5
    "gpt_neox": ("GPTNeoXForCausalLM", {}),          # pythia
    "bloom": ("BloomForCausalLM", {"n_layer": 2, "n_head": 4}),
    "falcon": ("FalconForCausalLM", {}),             # quality exception (adapter), kept as a control
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", {}),
}


class _Tok:
    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer_config.json").write_text("{}")


def _tiny_config(model_type: str, arch: str, extra: dict):
    from transformers import CONFIG_MAPPING

    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

        text = dict(T, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
                    linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                    max_position_embeddings=262144, tie_word_embeddings=True)
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
        cfg = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)
    else:
        cls = CONFIG_MAPPING[model_type]
        kwargs = {**T, **extra}
        for _ in range(len(kwargs)):  # drop kwargs a class exposes only as read-only properties (falcon head_dim)
            try:
                cfg = cls(**kwargs)
                break
            except AttributeError as exc:
                bad = next((k for k in kwargs if f"'{k}'" in str(exc)), None)
                if bad is None:
                    raise
                kwargs.pop(bad)
    cfg.architectures = [arch]
    return cfg


def _make_base(tmp: Path, model_type: str) -> Path:
    import transformers

    arch, extra = FAMILIES[model_type]
    cfg = _tiny_config(model_type, arch, extra)
    torch.manual_seed(0)
    model = getattr(transformers, arch)(cfg)
    base = tmp / f"base-{model_type}"
    model.save_pretrained(base)
    return base


def _train_like_forge(base: Path):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32, trust_remote_code=False)
    with torch.no_grad():  # a "trained" update, so parity below is not trivially base == export
        next(p for n, p in m.named_parameters() if p.dim() == 2).add_(0.01)
    return m


def _validator_load(repo: Path, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(repo), dtype=dtype, trust_remote_code=False)


def _logits(m, ids):
    m.eval()
    with torch.no_grad():
        return m(input_ids=ids).logits.float()


@pytest.mark.parametrize("model_type", sorted(FAMILIES))
def test_full_export_passes_validator_is_finetune(tmp_path, model_type):
    base = _make_base(tmp_path, model_type)
    trained = _train_like_forge(base)
    out = tmp_path / "out"
    common.save_adapter(trained, _Tok(), str(out))
    submitted = _validator_load(out)
    assert model_is_a_finetune(str(base), submitted) is True, json.loads((out / "config.json").read_text())
    # loadable with the trained weights (exact logits at fp32)
    ids = torch.randint(3, 256, (2, 9))
    assert torch.equal(_logits(_validator_load(out, torch.float32), ids), _logits(trained, ids))


def test_qwen35_control_without_alignment_fails(tmp_path, monkeypatch):
    """The defect: the pre-D1 export (architectures restore only) and even the untouched base are non-finetunes."""
    base = _make_base(tmp_path, "qwen3_5")
    assert model_is_a_finetune(str(base), _validator_load(base)) is False  # field-wide: base weights uploaded as-is
    monkeypatch.setattr(common, "_align_export_with_base", lambda *a, **k: None)
    out = tmp_path / "out"
    common.save_adapter(_train_like_forge(base), _Tok(), str(out))
    assert json.loads((out / "config.json").read_text())["model_type"] == "qwen3_5_text"
    assert model_is_a_finetune(str(base), _validator_load(out)) is False


def test_qwen35_sharded_export_renames_index(tmp_path):
    base = _make_base(tmp_path, "qwen3_5")
    trained = _train_like_forge(base)
    out = tmp_path / "sharded"
    trained.save_pretrained(out, max_shard_size="40KB")
    assert (out / "model.safetensors.index.json").is_file()
    report = export_compat.align_export_with_base(str(out), str(base))
    assert report["ok"] and report["mode"] == "composite" and report["renamed_tensors"] > 0
    wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(k.startswith("model.language_model.") for k in wm)
    assert set(wm) == export_compat.checkpoint_keys(str(out))
    ids = torch.randint(3, 256, (1, 7))
    assert model_is_a_finetune(str(base), _validator_load(out))
    assert torch.equal(_logits(_validator_load(out, torch.float32), ids), _logits(trained, ids))


def test_context_window_raised_to_base(tmp_path):
    base = _make_base(tmp_path, "llama")
    out = tmp_path / "ctx"
    _train_like_forge(base).save_pretrained(out)
    cfg = json.loads((out / "config.json").read_text())
    cfg["max_position_embeddings"] = 128
    (out / "config.json").write_text(json.dumps(cfg))
    assert export_compat.emulated_is_finetune(str(base), str(out))[0] is False
    assert export_compat.align_export_with_base(str(out), str(base))["ok"]
    assert json.loads((out / "config.json").read_text())["max_position_embeddings"] == 512


def test_failed_alignment_restores_original_bytes(tmp_path, monkeypatch):
    base = _make_base(tmp_path, "qwen3_5")
    out = tmp_path / "bad"
    _train_like_forge(base).save_pretrained(out)
    before_cfg = (out / "config.json").read_bytes()
    before_keys = export_compat.checkpoint_keys(str(out))
    monkeypatch.setattr(export_compat, "keys_match", lambda d: (False, "forced"))
    report = export_compat.align_export_with_base(str(out), str(base))
    assert report["ok"] is False and "forced" in report["error"]
    assert (out / "config.json").read_bytes() == before_cfg
    assert export_compat.checkpoint_keys(str(out)) == before_keys


def test_adapter_export_untouched(tmp_path):
    out = tmp_path / "adapter"
    out.mkdir()
    (out / "adapter_config.json").write_text("{}")
    assert export_compat.align_export_with_base(str(out), str(tmp_path))["skipped"] == "no_config_json"


def test_emulated_view_matches_real_load(tmp_path):
    """loaded_view_config (no weights) must see what the validator's real load sees, or the check could drift."""
    for mt in ("qwen3_5", "llama", "bloom"):
        base = _make_base(tmp_path, mt)
        view = export_compat.loaded_view_config(str(base))
        real = _validator_load(base).config
        assert type(view) is type(real) and view.model_type == real.model_type
        assert view.architectures == real.architectures


# ---- RESP-072-1: the header/index rewrite is transactional -----------------------------------------------------------

def _sharded_q35(tmp_path):
    base = _make_base(tmp_path, "qwen3_5")
    trained = _train_like_forge(base)
    out = tmp_path / "sharded"
    trained.save_pretrained(out, max_shard_size="40KB")
    files, index = export_compat._weight_files(str(out))
    assert index and len(files) >= 2
    snap = {p.name: p.read_bytes() for p in out.iterdir()}
    return base, trained, out, snap


def _assert_pristine(out, snap, trained):
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snap  # every file byte-identical, no stray temp files
    ids = torch.randint(3, 256, (1, 7))
    assert torch.equal(_logits(_validator_load(out, torch.float32), ids), _logits(trained, ids))


def test_later_shard_failure_rolls_back_everything(tmp_path, monkeypatch):
    base, trained, out, snap = _sharded_q35(tmp_path)
    real, calls = export_compat._rename_in_place, []

    def flaky(path, rename):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("EIO on shard 2")
        return real(path, rename)

    monkeypatch.setattr(export_compat, "_rename_in_place", flaky)
    report = export_compat.align_export_with_base(str(out), str(base))
    assert report["ok"] is False and "EIO" in report["error"] and len(calls) == 2
    _assert_pristine(out, snap, trained)


def test_torn_header_write_rolls_back(tmp_path, monkeypatch):
    base, trained, out, snap = _sharded_q35(tmp_path)

    def torn(path, rename):  # half a header reaches the disk, then the write fails
        with open(path, "r+b") as fh:
            fh.seek(8)
            fh.write(b"{" * 64)
        raise OSError("ENOSPC mid-write")

    monkeypatch.setattr(export_compat, "_rename_in_place", torn)
    assert export_compat.align_export_with_base(str(out), str(base))["ok"] is False
    _assert_pristine(out, snap, trained)


def test_index_replacement_failure_rolls_back(tmp_path, monkeypatch):
    base, trained, out, snap = _sharded_q35(tmp_path)
    real, failed = export_compat._atomic_write_bytes, []

    def flaky(path, data):
        if path.endswith("model.safetensors.index.json") and not failed:
            failed.append(path)
            Path(path + ".align.tmp").write_bytes(data[:10])  # a partial temp file is left behind
            raise OSError("EIO replacing the index")
        return real(path, data)

    monkeypatch.setattr(export_compat, "_atomic_write_bytes", flaky)
    report = export_compat.align_export_with_base(str(out), str(base))
    assert report["ok"] is False and failed
    _assert_pristine(out, snap, trained)


def test_unverified_rollback_raises_and_save_adapter_keeps_promoted_artifact(tmp_path, monkeypatch):
    base = _make_base(tmp_path, "qwen3_5")
    trained = _train_like_forge(base)
    out = tmp_path / "out"
    common.save_adapter(trained, _Tok(), str(out))  # a good promoted artifact
    promoted = {p.name: p.read_bytes() for p in out.iterdir()}

    raw = tmp_path / "raw"
    trained.save_pretrained(raw)  # unaligned export: the alignment will run
    monkeypatch.setattr(export_compat, "keys_match", lambda d: (False, "forced"))
    monkeypatch.setattr(export_compat._Journal, "verify", lambda self: False)  # rollback cannot be verified
    with pytest.raises(export_compat.ExportAlignmentError):
        export_compat.align_export_with_base(str(raw), str(base))
    with pytest.raises(export_compat.ExportAlignmentError):
        common.save_adapter(trained, _Tok(), str(out))  # staging fails instead of promoting
    assert {p.name: p.read_bytes() for p in out.iterdir()} == promoted
    assert not (tmp_path / "out.tmp").exists()


def test_failed_rollback_write_raises(tmp_path, monkeypatch):
    base, trained, out, snap = _sharded_q35(tmp_path)
    real = export_compat._rename_in_place
    n = []

    def flaky(path, rename):
        n.append(path)
        if len(n) == 2:
            raise OSError("EIO on shard 2")
        return real(path, rename)

    monkeypatch.setattr(export_compat, "_rename_in_place", flaky)
    monkeypatch.setattr(export_compat, "_atomic_write_bytes", lambda p, d: (_ for _ in ()).throw(OSError("ro fs")))
    with pytest.raises(export_compat.ExportAlignmentError):
        export_compat.align_export_with_base(str(out), str(base))
