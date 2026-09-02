"""Regression tests for the proven 54fa0344 architecture restoration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from forge.tasks import common


class _PeftCfg:
    def __init__(self, base: str) -> None:
        self.base_model_name_or_path = base


class _Model:
    def __init__(self, base: str | None, *, via_peft: bool = True) -> None:
        if via_peft and base is not None:
            self.peft_config = {"default": _PeftCfg(base)}
        if base is not None and not via_peft:
            self.config = type("C", (), {"_name_or_path": base})()


def _base(tmp_path: Path, arch: list[str] | None, name: str = "base") -> str:
    path = tmp_path / name
    path.mkdir()
    config: dict = {"model_type": "mistral", "hidden_size": 4096}
    if arch is not None:
        config["architectures"] = arch
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _staged(tmp_path: Path, config: dict | str | None) -> str:
    path = tmp_path / "staged"
    path.mkdir()
    if config is not None:
        (path / "config.json").write_text(
            config if isinstance(config, str) else json.dumps(config)
        )
    return str(path)


def test_aliasing_case_is_restored(tmp_path):
    base = _base(tmp_path, ["MistralForCausalLM"])
    staged = _staged(
        tmp_path, {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}
    )
    common._restore_base_architectures(staged, _Model(base))
    output = json.loads(Path(staged, "config.json").read_text())
    assert output["architectures"] == ["MistralForCausalLM"]
    assert output["hidden_size"] == 4096


def test_identity_case_is_noop_and_lossless(tmp_path):
    base = _base(tmp_path, ["Qwen3ForCausalLM"])
    staged = _staged(tmp_path, {"architectures": ["Qwen3ForCausalLM"], "x": 1})
    before = Path(staged, "config.json").read_text()
    common._restore_base_architectures(staged, _Model(base))
    assert Path(staged, "config.json").read_text() == before


def test_resolver_falls_back_to_name_or_path(tmp_path):
    base = _base(tmp_path, ["MistralForCausalLM"])
    staged = _staged(tmp_path, {"architectures": ["LlamaForCausalLM"]})
    common._restore_base_architectures(staged, _Model(base, via_peft=False))
    output = json.loads(Path(staged, "config.json").read_text())
    assert output["architectures"] == ["MistralForCausalLM"]


def test_adapter_only_export_is_untouched(tmp_path):
    base = _base(tmp_path, ["MistralForCausalLM"])
    staged = _staged(tmp_path, None)
    common._restore_base_architectures(staged, _Model(base))
    assert not os.path.exists(os.path.join(staged, "config.json"))


def test_unresolvable_or_absent_base_does_not_modify(tmp_path):
    staged = _staged(tmp_path, {"architectures": ["LlamaForCausalLM"]})
    before = Path(staged, "config.json").read_text()
    common._restore_base_architectures(staged, _Model("/nonexistent/path/xyz"))
    common._restore_base_architectures(staged, _Model(None))
    common._restore_base_architectures(staged, object())
    assert Path(staged, "config.json").read_text() == before


def test_malformed_json_never_raises_or_modifies(tmp_path):
    base = _base(tmp_path, ["MistralForCausalLM"])
    staged = _staged(tmp_path, "{ this is not json")
    common._restore_base_architectures(staged, _Model(base))
    assert Path(staged, "config.json").read_text() == "{ this is not json"


def test_write_failure_is_swallowed(tmp_path, monkeypatch):
    base = _base(tmp_path, ["MistralForCausalLM"])
    staged = _staged(tmp_path, {"architectures": ["LlamaForCausalLM"]})
    real_open = open

    def boom(path, *args, **kwargs):
        if str(path).endswith("config.json.tmp"):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", boom)
    common._restore_base_architectures(staged, _Model(base))
    assert json.loads(Path(staged, "config.json").read_text())["architectures"] == [
        "LlamaForCausalLM"
    ]


def test_save_adapter_wires_restore_before_validation():
    source = Path(common.__file__).read_text()
    save = source.index("tokenizer.save_pretrained(tmp)")
    restore = source.index("_restore_base_architectures(tmp, model)")
    validate = source.index("_validate_staged_artifact(tmp)")
    assert save < restore < validate
