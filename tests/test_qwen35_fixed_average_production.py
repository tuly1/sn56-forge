import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from forge.tuning import qwen35_fixed_average as fixed


class Qwen3_5TextConfig:
    model_type = "qwen3_5_text"
    hidden_size = 2560
    num_hidden_layers = 32
    vocab_size = 248320


class Qwen3_5ForCausalLM:
    config = Qwen3_5TextConfig()


def spec(model="Qwen/Qwen3.5-4B", cache="/cache/models/Qwen--Qwen3.5-4B", baseline="/cache/baseline.json"):
    return SimpleNamespace(
        model=model, cached_model_dir=cache, baseline_stats_path=baseline,
        task_type="InstructTextTask", instruct=SimpleNamespace(output="output"),
        use_kl=False,
    )


def test_exact_named_and_anonymous_production_routes(monkeypatch):
    monkeypatch.setattr(fixed, "_exact_payload", lambda _path: True)
    named = fixed.eligible_route(spec(), Qwen3_5ForCausalLM(), strategy="lora", n_gpus=1)
    assert named.endpoint_mode == "named_one_gpu"
    alias = "0123456789abcdef"
    anonymous = spec(alias, f"/cache/models/{alias}")
    route = fixed.eligible_route(
        anonymous, Qwen3_5ForCausalLM(), strategy="lora", n_gpus=2
    )
    assert route.endpoint_mode == "anonymous_two_gpu"
    assert route.minimum_soft_seconds == 2400.0


def test_route_is_inert_for_other_models_and_rejects_provenance_drift(monkeypatch):
    monkeypatch.setattr(fixed, "_exact_payload", lambda _path: True)
    assert fixed.eligible_route(
        spec("tiiuae/falcon-7b", "/cache/models/tiiuae--falcon-7b"),
        Qwen3_5ForCausalLM(), strategy="lora", n_gpus=1,
    ) is None
    alias = "0123456789abcdef"
    assert fixed.eligible_route(
        spec(alias, f"/cache/models/{alias}", baseline=None),
        Qwen3_5ForCausalLM(), strategy="lora", n_gpus=2,
    ) is None
    with pytest.raises(ValueError, match="identity drift"):
        fixed.eligible_route(spec(), object(), strategy="lora", n_gpus=1)


def test_strict_peft_unwrap_and_deadline(monkeypatch):
    monkeypatch.setattr(fixed, "_exact_payload", lambda _path: True)

    class PeftModelForCausalLM:
        def get_base_model(self):
            return Qwen3_5ForCausalLM()

    assert fixed.eligible_route(
        spec(), PeftModelForCausalLM(), strategy="lora", n_gpus=1
    ) is not None
    route = fixed.FixedAverageRoute()
    assert fixed.cap1_admitted(route, 2400.0)
    assert not fixed.cap1_admitted(route, 2399.99)
    assert not fixed.cap1_admitted(route, float("nan"))


class Factors:
    def named_parameters(self):
        for index in range(248):
            yield f"base.m{index}.lora_A.default.weight", torch.nn.Parameter(torch.ones(32, 2))
            yield f"base.m{index}.lora_B.default.weight", torch.nn.Parameter(torch.zeros(3, 32))


def test_factor_hash_requires_complete_248_pair_topology():
    assert fixed.adapter_factor_sha256(Factors()) == fixed.adapter_factor_sha256(Factors())
    with pytest.raises(ValueError, match="248 complete"):
        fixed.adapter_factor_sha256(SimpleNamespace(named_parameters=lambda: []))

    class NonzeroB(Factors):
        def named_parameters(self):
            for name, parameter in super().named_parameters():
                if ".lora_B." in name:
                    parameter = torch.nn.Parameter(torch.ones_like(parameter))
                yield name, parameter

    with pytest.raises(ValueError, match="B factor is not zero"):
        fixed.adapter_factor_sha256(NonzeroB())


def _endpoint(path: Path, value: float) -> None:
    path.mkdir()
    weights = {}
    for index in range(248):
        weights[f"base.m{index}.lora_A.default.weight"] = torch.full((32, 2), value)
        weights[f"base.m{index}.lora_B.default.weight"] = torch.full((3, 32), value)
    save_file(weights, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    (path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))


def test_fixed_midpoint_is_numeric_for_every_factor(tmp_path):
    native, cap1, output = tmp_path / "native", tmp_path / "cap1", tmp_path / "midpoint"
    _endpoint(native, 1.0); _endpoint(cap1, 3.0)
    receipt = fixed.build_fixed_midpoint(str(native), str(cap1), str(output))
    weights = load_file(str(output / "adapter_model.safetensors"))
    assert len(weights) == 496
    assert all(torch.equal(value, torch.full_like(value, 2.0)) for value in weights.values())
    assert receipt["complete_ab_pairs"] == 248
    assert receipt["no_score_conditioned_selection"] is True


def test_instruct_integration_keeps_native_and_cap1_semantics():
    import inspect
    from forge.tasks import instruct

    source = inspect.getsource(instruct)
    assert "epochs=fixed_average_route.cap1_epochs" in source
    assert "qwen35_fixed_average_native_fallback" in source
    assert "before_train=capture_native_initial" in source
    assert "MIN_CAP1_SOFT_SECONDS = 2400.0" in inspect.getsource(fixed)
    assert "FORGE_QWEN35_4B_FIXED_AVERAGE" not in inspect.getsource(fixed)
