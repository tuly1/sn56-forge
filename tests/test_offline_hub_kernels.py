"""disable_hub_kernels_offline: under HF_HUB_OFFLINE=1 the Hub-kernel loader is switched to its local branch (phase-bf Falcon-H1)."""
import types

from forge import model


def _fake_hub(monkeypatch, available=True):
    hk = types.SimpleNamespace(_kernels_available=available, _KERNEL_MODULE_MAPPING={"causal-conv1d": None, "x": types})
    import transformers.integrations as ti
    monkeypatch.setattr(ti, "hub_kernels", hk, raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", hk)
    return hk


def test_offline_switches_hub_kernels_off(monkeypatch):
    hk = _fake_hub(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert model.disable_hub_kernels_offline() is True
    assert hk._kernels_available is False and "causal-conv1d" not in hk._KERNEL_MODULE_MAPPING and "x" in hk._KERNEL_MODULE_MAPPING


def test_online_or_no_kernels_is_untouched(monkeypatch):
    hk = _fake_hub(monkeypatch)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert model.disable_hub_kernels_offline() is False and hk._kernels_available is True
    hk2 = _fake_hub(monkeypatch, available=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert model.disable_hub_kernels_offline() is False and hk2._kernels_available is False


# ---- route snapshot declaration (tests/test_route_snapshot_v9.py collects V9_ROUTE_CHANGES; release-candidate-v9) ----------

def _every_cell(key, base):
    """forge.model.load_base runs the guard for every task (process level)."""
    return True


V9_ROUTE_CHANGES = {"offline_hub_kernels_guard": (_every_cell, ("rules", "hub_kernels_guard"), True)}
