"""Allocator env names are set before torch loads and never override an operator.

The 2026-09-03 H100 survival smoke traced the reserved-memory growth past the
admission probe to caching-allocator fragmentation; `forge/cli.py` sets
`expandable_segments:True` at process start on Linux with a CUDA device.  The
646d0b70 smoke (torch 2.9.1) showed the legacy name PYTORCH_CUDA_ALLOC_CONF is
deprecated in favour of PYTORCH_ALLOC_CONF, so both names are set together when
the operator set neither, and both are preserved when the operator set either.
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
NEW, LEGACY = "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"
DEFAULT = "expandable_segments:True"

# Runs in a fresh interpreter: fakes the platform and the CUDA device nodes
# BEFORE `forge.cli` is imported, then reports both environment names, the
# recorded decision, and whether torch was in sys.modules around the import.
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
    "env_new": os.environ.get("PYTORCH_ALLOC_CONF"),
    "env_legacy": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "new_present": "PYTORCH_ALLOC_CONF" in os.environ,
    "legacy_present": "PYTORCH_CUDA_ALLOC_CONF" in os.environ,
    "state": cli.cuda_alloc_conf_state(),
}))
"""


def _fresh_process(
    system: str, cuda_nodes: bool, *, new: str | None = None, legacy: str | None = None
) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (NEW, LEGACY)}
    env["PYTHONPATH"] = ROOT + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["FORGE_PROBE_SYSTEM"] = system
    env["FORGE_PROBE_CUDA_NODES"] = "1" if cuda_nodes else "0"
    if new is not None:
        env[NEW] = new
    if legacy is not None:
        env[LEGACY] = legacy
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
        timeout=120, check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["cli_file"].startswith(ROOT), report["cli_file"]
    return report


def test_linux_cuda_process_sets_both_names_before_torch_is_imported():
    report = _fresh_process("Linux", cuda_nodes=True)
    assert report["torch_before_import"] is False
    assert report["torch_after_import"] is False  # cli.py itself never imports torch
    assert report["env_new"] == DEFAULT and report["env_legacy"] == DEFAULT
    state = report["state"]
    assert state["alloc_conf_new"] == DEFAULT
    assert state["alloc_conf_legacy"] == DEFAULT
    assert state["preset_keys"] == []
    assert state["source"] == "forge_default"
    assert state["reason"] == "linux_cuda_device"
    assert state["torch_preloaded"] is False
    assert state["cuda_device_visible"] is True and state["system"] == "Linux"
    assert state["new_key"] == NEW and state["legacy_key"] == LEGACY


def test_preset_combinations_are_preserved_in_a_fresh_process():
    # Legacy name only: kept verbatim, the new name is NOT added.
    legacy_only = _fresh_process("Linux", cuda_nodes=True, legacy="max_split_size_mb:512")
    assert legacy_only["env_legacy"] == "max_split_size_mb:512"
    assert legacy_only["new_present"] is False and legacy_only["env_new"] is None
    assert legacy_only["state"]["preset_keys"] == [LEGACY]
    assert legacy_only["state"]["alloc_conf_legacy"] == "max_split_size_mb:512"
    assert legacy_only["state"]["alloc_conf_new"] is None
    assert legacy_only["state"]["source"] == "environment"
    assert legacy_only["state"]["reason"] == "preset_by_operator"

    # New name only: kept verbatim, the legacy name is NOT added.
    new_only = _fresh_process("Linux", cuda_nodes=True, new="garbage_collection_threshold:0.8")
    assert new_only["env_new"] == "garbage_collection_threshold:0.8"
    assert new_only["legacy_present"] is False and new_only["env_legacy"] is None
    assert new_only["state"]["preset_keys"] == [NEW]
    assert new_only["state"]["alloc_conf_new"] == "garbage_collection_threshold:0.8"
    assert new_only["state"]["alloc_conf_legacy"] is None
    assert new_only["state"]["source"] == "environment"

    # Both names, different values: both kept exactly as found.
    both = _fresh_process(
        "Linux", cuda_nodes=True, new="expandable_segments:False", legacy="max_split_size_mb:64",
    )
    assert both["env_new"] == "expandable_segments:False"
    assert both["env_legacy"] == "max_split_size_mb:64"
    assert both["state"]["preset_keys"] == [NEW, LEGACY]
    assert both["state"]["source"] == "environment"

    # An empty operator value is still an operator value.
    for kwargs in ({"legacy": ""}, {"new": ""}, {"new": "", "legacy": ""}):
        empty = _fresh_process("Linux", cuda_nodes=True, **kwargs)
        assert empty["new_present"] == ("new" in kwargs)
        assert empty["legacy_present"] == ("legacy" in kwargs)
        assert empty["env_new"] == ("" if "new" in kwargs else None)
        assert empty["env_legacy"] == ("" if "legacy" in kwargs else None)
        assert empty["state"]["source"] == "environment"


def test_non_linux_or_gpu_less_process_leaves_both_names_untouched():
    darwin = _fresh_process("Darwin", cuda_nodes=True)
    assert darwin["new_present"] is False and darwin["legacy_present"] is False
    assert darwin["state"]["source"] == "unset"
    assert darwin["state"]["reason"] == "not_linux"
    assert darwin["state"]["preset_keys"] == []
    headless = _fresh_process("Linux", cuda_nodes=False)
    assert headless["new_present"] is False and headless["legacy_present"] is False
    assert headless["state"]["source"] == "unset"
    assert headless["state"]["reason"] == "no_cuda_device"
    assert headless["state"]["cuda_device_visible"] is False


def test_configure_is_pure_and_reports_every_branch():
    env = {}
    state = cli._configure_cuda_allocator(
        env, system="Linux", cuda_visible=True, torch_preloaded=False
    )
    assert env == {NEW: DEFAULT, LEGACY: DEFAULT}
    assert state["source"] == "forge_default" and state["torch_preloaded"] is False
    assert state["preset_keys"] == []
    # Setting twice is a no-op: the first decision now reads as operator values.
    again = cli._configure_cuda_allocator(
        env, system="Linux", cuda_visible=True, torch_preloaded=True
    )
    assert env == {NEW: DEFAULT, LEGACY: DEFAULT}
    assert again["source"] == "environment" and again["torch_preloaded"] is True
    assert again["preset_keys"] == [NEW, LEGACY]

    # The four operator combinations.
    for preset, expected_keys in (
        ({}, []),
        ({LEGACY: "max_split_size_mb:512"}, [LEGACY]),
        ({NEW: "garbage_collection_threshold:0.8"}, [NEW]),
        ({NEW: "expandable_segments:False", LEGACY: "max_split_size_mb:64"}, [NEW, LEGACY]),
    ):
        environ = dict(preset)
        state = cli._configure_cuda_allocator(
            environ, system="Linux", cuda_visible=True, torch_preloaded=False
        )
        if preset:
            assert environ == preset, preset  # nothing added, nothing changed
            assert state["source"] == "environment"
            assert state["reason"] == "preset_by_operator"
        else:
            assert environ == {NEW: DEFAULT, LEGACY: DEFAULT}
            assert state["source"] == "forge_default"
        assert state["preset_keys"] == expected_keys
        assert state["alloc_conf_new"] == environ.get(NEW)
        assert state["alloc_conf_legacy"] == environ.get(LEGACY)

    for preset in ({LEGACY: ""}, {NEW: ""}):
        environ = dict(preset)
        state = cli._configure_cuda_allocator(
            environ, system="Linux", cuda_visible=True, torch_preloaded=False
        )
        assert environ == preset and state["source"] == "environment"

    for system, visible, reason in (
        ("Darwin", True, "not_linux"), ("Windows", False, "not_linux"),
        ("Linux", False, "no_cuda_device"),
    ):
        untouched = {}
        state = cli._configure_cuda_allocator(
            untouched, system=system, cuda_visible=visible, torch_preloaded=False
        )
        assert untouched == {} and state["source"] == "unset"
        assert state["reason"] == reason
        assert state["alloc_conf_new"] is None and state["alloc_conf_legacy"] is None
    # The recorded state honestly says whether torch had already been loaded.
    late = {}
    state = cli._configure_cuda_allocator(
        late, system="Linux", cuda_visible=True, torch_preloaded=True
    )
    assert late == {NEW: DEFAULT, LEGACY: DEFAULT}
    assert state["torch_preloaded"] is True


def test_import_order_contract_keeps_the_setter_ahead_of_any_torch_import():
    source = inspect.getsource(cli)
    setter = source.index("_CUDA_ALLOC_CONF_STATE = _configure_cuda_allocator(")
    assert setter < source.index("from forge import telemetry")
    assert setter < source.index("from forge.clock import Deadline")
    assert setter < source.index("from forge.data.schema import TaskSpec")
    assert "import torch" not in source
    assert cli._ALLOC_CONF_KEYS == (NEW, LEGACY)
    # Everything cli.py imports at module level is torch-free, and the package
    # itself imports nothing, so `python -m forge.cli` sets the variables before
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


def test_main_records_both_names_in_meta_and_as_an_event(monkeypatch):
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
        "cuda_alloc_conf": state["alloc_conf_legacy"],
        "alloc_conf_new": state["alloc_conf_new"],
        "alloc_conf_legacy": state["alloc_conf_legacy"],
        "cuda_alloc_conf_source": state["source"],
        "cuda_alloc_conf_reason": state["reason"],
        "cuda_alloc_conf_preset_keys": state["preset_keys"],
    }]
    # Recorded before anything else can happen in the run.
    assert [name for name, _ in events] == ["cuda_alloc_conf"]
    recorded = events[0][1]
    assert recorded["new_key"] == NEW and recorded["legacy_key"] == LEGACY
    assert recorded["source"] == state["source"]
    assert recorded["torch_preloaded"] == state["torch_preloaded"]
    assert set(recorded) == {
        "new_key", "legacy_key", "alloc_conf_new", "alloc_conf_legacy",
        "preset_keys", "system", "cuda_device_visible", "torch_preloaded",
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

    settings = {"expandable_segments": True, LEGACY: DEFAULT}
    monkeypatch.setitem(
        sys.modules, "torch",
        fake_torch(True, lambda: {"allocator_settings": settings}),
    )
    monkeypatch.setenv(LEGACY, DEFAULT)
    cli._record_cuda_allocator_readback()
    assert events[-1][1] == {
        "status": "ok",
        "expandable_segments": True,
        "allocator_conf": DEFAULT,
        "env_value": DEFAULT,
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
