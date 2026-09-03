"""PYTORCH_CUDA_ALLOC_CONF is set before torch loads and never overrides an operator.

The 2026-09-03 H100 survival smoke traced the reserved-memory growth past the
admission probe to caching-allocator fragmentation; `forge/cli.py` now sets
`expandable_segments:True` at process start on Linux with a CUDA device, only
when the variable is absent, and records the decision in the artifact.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from forge import cli, telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs in a fresh interpreter: fakes the platform and the CUDA device nodes
# BEFORE `forge.cli` is imported, then reports the environment, the recorded
# decision, and whether torch was in sys.modules around the import.
_PROBE = r"""
import json, os, platform, sys
system = os.environ.pop("FORGE_PROBE_SYSTEM")
nodes = os.environ.pop("FORGE_PROBE_CUDA_NODES") == "1"
_real_exists = os.path.exists
_device_nodes = ("/dev/nvidiactl", "/dev/nvidia0")
platform.system = lambda: system
os.path.exists = lambda path: nodes if path in _device_nodes else _real_exists(path)
torch_before = "torch" in sys.modules
import forge.cli as cli
print(json.dumps({
    "cli_file": cli.__file__,
    "torch_before_import": torch_before,
    "torch_after_import": "torch" in sys.modules,
    "env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "env_present": "PYTORCH_CUDA_ALLOC_CONF" in os.environ,
    "state": cli.cuda_alloc_conf_state(),
}))
"""


def _fresh_process(system: str, cuda_nodes: bool, preset: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env["PYTHONPATH"] = ROOT + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["FORGE_PROBE_SYSTEM"] = system
    env["FORGE_PROBE_CUDA_NODES"] = "1" if cuda_nodes else "0"
    if preset is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = preset
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
        timeout=120, check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["cli_file"].startswith(ROOT), report["cli_file"]
    return report


def test_linux_cuda_process_sets_default_before_torch_is_imported():
    report = _fresh_process("Linux", cuda_nodes=True)
    assert report["torch_before_import"] is False
    assert report["torch_after_import"] is False  # cli.py itself never imports torch
    assert report["env"] == "expandable_segments:True"
    state = report["state"]
    assert state["value"] == "expandable_segments:True"
    assert state["source"] == "forge_default"
    assert state["reason"] == "linux_cuda_device"
    assert state["torch_preloaded"] is False
    assert state["cuda_device_visible"] is True and state["system"] == "Linux"


def test_preset_value_is_preserved_in_a_fresh_process():
    for preset in ("max_split_size_mb:512", "expandable_segments:False", ""):
        report = _fresh_process("Linux", cuda_nodes=True, preset=preset)
        assert report["env_present"] is True
        assert report["env"] == preset
        assert report["state"]["value"] == preset
        assert report["state"]["source"] == "environment"
        assert report["state"]["reason"] == "preset_by_operator"


def test_non_linux_or_gpu_less_process_leaves_environment_untouched():
    darwin = _fresh_process("Darwin", cuda_nodes=True)
    assert darwin["env_present"] is False and darwin["env"] is None
    assert darwin["state"]["source"] == "unset"
    assert darwin["state"]["reason"] == "not_linux"
    headless = _fresh_process("Linux", cuda_nodes=False)
    assert headless["env_present"] is False
    assert headless["state"]["source"] == "unset"
    assert headless["state"]["reason"] == "no_cuda_device"
    assert headless["state"]["cuda_device_visible"] is False


def test_configure_is_pure_and_reports_every_branch():
    env = {}
    state = cli._configure_cuda_allocator(
        env, system="Linux", cuda_visible=True, torch_preloaded=False
    )
    assert env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert state["source"] == "forge_default" and state["torch_preloaded"] is False
    # Setting twice is a no-op: the first decision becomes an operator value.
    again = cli._configure_cuda_allocator(
        env, system="Linux", cuda_visible=True, torch_preloaded=True
    )
    assert env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert again["source"] == "environment" and again["torch_preloaded"] is True
    preset = {"PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    kept = cli._configure_cuda_allocator(
        preset, system="Linux", cuda_visible=True, torch_preloaded=False
    )
    assert preset == {"PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    assert kept["value"] == "garbage_collection_threshold:0.8"
    for system, visible, reason in (
        ("Darwin", True, "not_linux"), ("Windows", False, "not_linux"),
        ("Linux", False, "no_cuda_device"),
    ):
        untouched = {}
        state = cli._configure_cuda_allocator(
            untouched, system=system, cuda_visible=visible, torch_preloaded=False
        )
        assert untouched == {} and state["source"] == "unset"
        assert state["reason"] == reason and state["value"] is None
    # The recorded state honestly says whether torch had already been loaded.
    late = {}
    state = cli._configure_cuda_allocator(
        late, system="Linux", cuda_visible=True, torch_preloaded=True
    )
    assert late["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert state["torch_preloaded"] is True


def test_import_order_contract_keeps_the_setter_ahead_of_any_torch_import():
    source = inspect.getsource(cli)
    setter = source.index("_CUDA_ALLOC_CONF_STATE = _configure_cuda_allocator(")
    assert setter < source.index("from forge import telemetry")
    assert setter < source.index("from forge.clock import Deadline")
    assert setter < source.index("from forge.data.schema import TaskSpec")
    assert "import torch" not in source
    # Everything cli.py imports at module level is torch-free, and the package
    # itself imports nothing, so `python -m forge.cli` sets the variable before
    # torch can be loaded by the lazily imported task handlers.
    for relative in ("forge/__init__.py", "forge/telemetry.py", "forge/clock.py",
                     "forge/data/schema.py", "forge/data/__init__.py"):
        with open(os.path.join(ROOT, relative), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        assert not [line for line in lines
                    if line.startswith(("import torch", "from torch", "import transformers",
                                        "from transformers"))], relative
    with open(os.path.join(ROOT, "forge/__init__.py"), encoding="utf-8") as handle:
        assert handle.read().strip() == ""
    with open(os.path.join(ROOT, "ops/docker/standalone-text-trainer.dockerfile"),
              encoding="utf-8") as handle:
        assert '"-m", "forge.cli"' in handle.read()


def _capture(monkeypatch):
    events, metas = [], []
    monkeypatch.setattr(telemetry, "event", lambda name, **kv: events.append((name, kv)))
    monkeypatch.setattr(telemetry, "set_meta", lambda **kv: metas.append(kv))
    return events, metas


def test_main_records_the_decision_in_meta_and_as_an_event(monkeypatch):
    events, metas = _capture(monkeypatch)
    runs = []
    monkeypatch.setattr(cli, "_run", lambda spec, deadline: runs.append(spec.task_id))
    rc = cli.main([
        "--task-id", "t", "--model", "m", "--task-type", "InstructTextTask",
        "--dataset-type", json.dumps({"field_instruction": "q", "field_output": "a"}),
        "--expected-repo-name", "r", "--hours-to-complete", "0.1",
    ])
    assert rc == 0 and runs == ["t"]
    state = cli.cuda_alloc_conf_state()
    assert metas == [{
        "cuda_alloc_conf": state["value"],
        "cuda_alloc_conf_source": state["source"],
        "cuda_alloc_conf_reason": state["reason"],
    }]
    # Recorded before anything else can happen in the run.
    assert [name for name, _ in events] == ["cuda_alloc_conf"]
    recorded = events[0][1]
    assert recorded["key"] == "PYTORCH_CUDA_ALLOC_CONF"
    assert recorded["source"] == state["source"]
    assert recorded["torch_preloaded"] == state["torch_preloaded"]
    assert set(recorded) == {
        "key", "value", "system", "cuda_device_visible", "torch_preloaded",
        "source", "reason",
    }


def test_run_paths_call_the_torch_readback():
    source = inspect.getsource(cli._run)
    handler_ok = source.index("handler(spec, deadline)")
    complete = source.index('telemetry.event("run_complete")')
    assert handler_ok < source.index("_record_cuda_allocator_readback()") < complete
    fallback_write = source.rindex("telemetry.write_into(spec.output_dir)")
    assert source.rindex("_record_cuda_allocator_readback()") < fallback_write


def test_readback_reports_every_status_and_never_raises(monkeypatch):
    events, metas = _capture(monkeypatch)

    monkeypatch.setitem(sys.modules, "torch", None)
    cli._record_cuda_allocator_readback()
    assert events[-1][0] == "cuda_alloc_conf_readback"
    assert events[-1][1]["status"] == "torch_not_imported"

    def fake_torch(available, snapshot):
        return SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: available,
            memory=SimpleNamespace(_snapshot=snapshot),
        ))

    monkeypatch.setitem(sys.modules, "torch", fake_torch(False, lambda: {}))
    cli._record_cuda_allocator_readback()
    assert events[-1][1]["status"] == "cuda_unavailable"

    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, lambda: {"segments": []}))
    cli._record_cuda_allocator_readback()
    assert events[-1][1]["status"] == "allocator_settings_absent"

    def boom():
        raise RuntimeError("no snapshot")

    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, boom))
    cli._record_cuda_allocator_readback()
    assert events[-1][1]["status"] == "error"
    assert "RuntimeError: no snapshot" in events[-1][1]["error"]
    assert metas == []

    settings = {"expandable_segments": True,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    monkeypatch.setitem(
        sys.modules, "torch",
        fake_torch(True, lambda: {"allocator_settings": settings}),
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cli._record_cuda_allocator_readback()
    assert events[-1][1] == {
        "status": "ok",
        "expandable_segments": True,
        "allocator_conf": "expandable_segments:True",
        "env_value": "expandable_segments:True",
    }
    assert metas == [{"cuda_alloc_conf_expandable_segments": True}]


def test_real_torch_readback_on_this_host_is_recorded_not_raised(monkeypatch):
    import torch  # noqa: F401  (ensures the real module is what the readback sees)

    events, _ = _capture(monkeypatch)
    cli._record_cuda_allocator_readback()
    assert events[-1][0] == "cuda_alloc_conf_readback"
    assert events[-1][1]["status"] in {
        "cuda_unavailable", "ok", "allocator_settings_absent", "error"
    }
