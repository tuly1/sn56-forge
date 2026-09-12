"""The v2→production fallback must not reload the base while the failed
handler's frames are still alive (2026-09-12 review P2).

Inside an active ``except`` suite the exception's traceback owns the failed
handler's frames — model, trainer, optimizer — so clearing locals and calling
gc.collect() there cannot free them; a sharded 32B left resident starves the
production geometry probe. The reload must happen after the suite exits.
"""
from __future__ import annotations

import gc
import time
import weakref
from types import SimpleNamespace

import pytest

from forge.clock import Deadline
from forge.data.schema import InstructColumns, TaskSpec
from forge.tasks import instruct, sft_v2


class _Stop(Exception):
    pass


def _spec(tmp_path):
    return TaskSpec(
        task_id="fallback-release", task_type="InstructTextTask", model="0123456789abcdef",
        dataset=None, expected_repo_name="cpu-only", baseline_stats_path=None,
        instruct=InstructColumns("instruction", "output"),
    )


def test_failed_v2_model_is_collectible_before_fallback_reload(monkeypatch, tmp_path):
    class _HandlerModel:  # stands in for the sharded model held by sft_v2.run's frame
        pass

    holder: dict[str, weakref.ref] = {}
    calls = {"load_base": 0}
    rows = [{"instruction": f"p{i}", "output": f"a{i}"} for i in range(50)]
    base = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(model_type="gemma2")),
                           tokenizer=SimpleNamespace(pad_token_id=0), model_dir=str(tmp_path))

    def fake_load_base(*a, **k):
        calls["load_base"] += 1
        if calls["load_base"] == 2:  # the fallback reload
            gc.collect()
            assert holder["model"]() is None, "failed v2 model still alive at fallback reload"
            raise _Stop()
        return base

    def failing_v2_run(*a, **k):
        model = _HandlerModel()  # local of the failing frame, like the real handler's model
        holder["model"] = weakref.ref(model)
        raise RuntimeError("simulated v2 failure")

    monkeypatch.setattr(instruct.loader, "load_rows", lambda *a, **k: rows)
    monkeypatch.setattr(instruct, "load_base", fake_load_base)
    monkeypatch.setattr(instruct.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(instruct.telemetry, "event", lambda *a, **k: None)
    monkeypatch.setattr(instruct.telemetry, "set_meta", lambda **k: None)
    monkeypatch.setattr(instruct, "load_baseline_summary", lambda *a, **k: None)
    monkeypatch.setattr(instruct, "model_param_billions", lambda model: 2.6)
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (1, 80.0))
    monkeypatch.setattr(instruct, "_is_bloomz_promotion", lambda *a: False)
    monkeypatch.setattr(instruct, "is_qwen35_model", lambda model: False)
    monkeypatch.setattr(sft_v2, "strategy_for", lambda params_b, mt=None: "lora")
    monkeypatch.setattr(sft_v2, "long_row_task", lambda rows, spec, tok: (True, 700))
    monkeypatch.setattr(sft_v2, "bigdata_full_route", lambda **k: False)
    monkeypatch.setattr(sft_v2, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(sft_v2, "trained_artifact_present", lambda spec: False)
    monkeypatch.setattr(sft_v2, "run", failing_v2_run)
    monkeypatch.setenv("FORGE_V2_LEARNABILITY_GATE", "0")

    deadline = Deadline.from_hours(1.0, started_monotonic=time.monotonic(), export_reserve_s=30)
    with pytest.raises(_Stop):
        instruct.run(_spec(tmp_path), deadline)
    assert calls["load_base"] == 2
