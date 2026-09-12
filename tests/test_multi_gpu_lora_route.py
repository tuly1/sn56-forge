"""Gate logic of the sharded multi-GPU adapter route (Round-2 Qwen3-32B)."""
import os
from types import SimpleNamespace

import pytest

from forge.tasks import sft_v2


def _model(model_type="qwen3"):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


def _spec():
    return SimpleNamespace(task_type="InstructTextTask", instruct=object())


@pytest.fixture(autouse=True)
def _cpu_env(monkeypatch):
    monkeypatch.setenv("FORGE_SFT_V2_ALLOW_CPU", "1")
    monkeypatch.delenv("FORGE_V2_MULTI_GPU_LORA", raising=False)
    monkeypatch.delenv("FORGE_V2_STRATEGY", raising=False)
    monkeypatch.setattr(sft_v2, "is_quasar_model", lambda m: False, raising=False)


def test_32b_on_four_gpus_is_eligible_by_default():
    assert sft_v2.strategy_for(32.8, "qwen3") == "lora"
    assert sft_v2.eligible(_spec(), is_kl=False, params_b=32.8, n_gpus=4, model=_model()) is True


def test_route_can_be_disabled():
    os.environ["FORGE_V2_MULTI_GPU_LORA"] = "0"
    try:
        assert sft_v2.strategy_for(32.8, "qwen3") == ""
        assert sft_v2.eligible(_spec(), is_kl=False, params_b=32.8, n_gpus=4, model=_model()) is False
    finally:
        del os.environ["FORGE_V2_MULTI_GPU_LORA"]


def test_mid_size_on_two_gpus_stays_on_production():
    assert sft_v2.eligible(_spec(), is_kl=False, params_b=8.0, n_gpus=2, model=_model()) is False


def test_single_gpu_gates_unchanged():
    assert sft_v2.eligible(_spec(), is_kl=False, params_b=3.1, n_gpus=1, model=_model()) is True
    assert sft_v2.eligible(_spec(), is_kl=False, params_b=32.8, n_gpus=1, model=_model()) is False
    assert sft_v2.eligible(_spec(), is_kl=True, params_b=3.1, n_gpus=1, model=_model()) is False
