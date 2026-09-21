"""Production activation and rollback contract; real weights are checked separately."""
import pytest

from forge.routing import gemma_small_batch_admission as admission


def route(**changes):
    return dict(model_dir="/closed-fixture", model_type="gemma2", strategy="lora",
                field=False, is_kl=False, cuda_available=True, device="cuda:0",
                n_gpus=1, sharded=False, micro_batch=5, grad_accum=4,
                effective_batch=20) | changes


@pytest.fixture
def exact_identity(monkeypatch):
    for name in list(admission.os.environ):
        if name.startswith("FORGE_V2_"):
            monkeypatch.delenv(name)
    calls = []
    def checked_identity(*args, **kwargs):
        calls.append(kwargs)
        return {"action": "qualified_gemma2_native_lora", "status": "identical"}
    monkeypatch.setattr(admission.gate, "choose_route", checked_identity)
    return calls


@pytest.mark.parametrize("micro,accum", [(5, 4), (10, 2)])
def test_bundled_gate_activates_without_environment(micro, accum, exact_identity):
    result = admission.qualify(**route(micro_batch=micro, grad_accum=accum))
    assert result["action"] == "enable_gemma_small_batch"
    assert len(exact_identity) == 1
    assert exact_identity[0]["tables_dir"] == admission.BUNDLED_REFERENCE_DIR
    assert exact_identity[0]["route"]["micro_batch"] == micro


def test_explicit_zero_is_immediate_rollback(monkeypatch, exact_identity):
    monkeypatch.setenv(admission.DEFAULT_FLAG, "0")
    assert admission.qualify(**route())["status"] == "disabled"
    assert not exact_identity


@pytest.mark.parametrize("change", [
    {"model_type": "llama"}, {"strategy": "full"}, {"field": True},
    {"is_kl": True}, {"n_gpus": 2}, {"sharded": True},
    {"cuda_available": False}, {"micro_batch": 11},
])
def test_other_routes_do_not_reach_identity_check(change, exact_identity):
    assert admission.qualify(**route(**change))["action"] == "incumbent"
    assert not exact_identity


def test_altered_identity_keeps_incumbent(monkeypatch, exact_identity):
    monkeypatch.setattr(admission.gate, "choose_route", lambda *a, **k: {
        "action": "incumbent", "status": "altered"})
    assert admission.qualify(**route())["action"] == "incumbent"


@pytest.mark.parametrize("name,value", [
    ("FORGE_V2_LORA_R", "64"), ("FORGE_V2_LORA_ALPHA", "128"),
    ("FORGE_V2_LORA_DROPOUT", "0.05"), ("FORGE_V2_LORA_PLUS", "2"),
    ("FORGE_V2_FP32_MASTER", "1"), ("FORGE_V2_LR", "0.001"),
    ("FORGE_V2_EXPERIMENT_INIT_SEED", "7"),
])
def test_recipe_overrides_keep_incumbent(name, value, monkeypatch, exact_identity):
    monkeypatch.setenv(name, value)
    assert admission.qualify(**route())["action"] == "incumbent"
    assert not exact_identity
