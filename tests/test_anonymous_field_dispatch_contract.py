"""Actual anonymous CLI/baseline/dispatch decisions; stop before training.

Only model loading, reported hardware, and training/export terminals are doubles.
Schema parsing, baseline JSON validation, row loading, family/size/length gates,
strategy resolution and the dispatcher are the deployed implementation.
"""
import json
import os
import time
from types import SimpleNamespace as NS

import pytest
import torch

from forge import cli
from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tasks import dispatch, instruct, sft_v2
from forge.tasks import fallback, gemma4_full_export, lfm25_full_export

ANONYMOUS_ID = "0123456789abcdef"


class TokenizerDouble:
    """Predictable synthetic word IDs; not a model-tokenizer parity test."""
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens=True, **kwargs):
        return {"input_ids": ([1] if add_special_tokens else []) + [10] * len(text.split())}


class StopAtLegacyPlan(BaseException):
    pass


def prepare(monkeypatch, tmp_path, model_type, *, n_rows=20000, long_rows=True):
    for name in list(os.environ):
        if name == "FORGE_V2_FULLFT_DEFAULT":
            continue  # set by the caller: "0" = the cab930c-equivalence suites, unset = the release default
        if name.startswith("FORGE_") or name in {"USE_KL", "KL_COEF", "BASELINE_STATS_PATH"}:
            monkeypatch.delenv(name, raising=False)
    # Mock the deployment's reported one-GPU topology; no CUDA allocation occurs.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (1, 80.0))
    count = 1235814400 if model_type == "llama" else 3085938688
    model = NS(config=NS(model_type=model_type), parameters=lambda: iter([NS(numel=lambda: count)]))
    loaded = NS(model=model, tokenizer=TokenizerDouble(), model_dir=str(tmp_path / ANONYMOUS_ID))
    observed = {"model_paths": [], "routes": [], "events": [], "meta": []}

    def load_base(path, **kwargs):
        observed["model_paths"].append(path)
        return loaded

    monkeypatch.setattr(instruct, "load_base", load_base)
    for name in ("init", "collect_env", "write_into"):
        monkeypatch.setattr(cli.telemetry, name, lambda *a, **k: None)
    monkeypatch.setattr(cli.telemetry, "event", lambda name, **kw: observed["events"].append((name, kw)))
    monkeypatch.setattr(cli.telemetry, "set_meta", lambda **kw: observed["meta"].append(kw))
    monkeypatch.setattr(lfm25_full_export, "maybe_export", lambda *a: None)
    monkeypatch.setattr(gemma4_full_export, "maybe_export", lambda *a: None)

    def unexpected_fallback(*args):
        observed["events"].append(("UNEXPECTED_FALLBACK", {}))
        return {"steps": 1}

    # v9.2: the CLI's crash path is the minimal fine-tune. The stubbed handlers write no artifact, so the output check
    # is stubbed "trained" and any entry into the minimal fine-tune is recorded as unexpected.
    monkeypatch.setattr(fallback, "run_mini_finetune", unexpected_fallback)
    monkeypatch.setattr(fallback, "output_is_trained", lambda output_dir: True)
    monkeypatch.setattr(fallback, "clear_output", lambda output_dir: None)

    def stop_v2(spec, deadline, rows, loaded_arg, tokenizer, **kw):
        observed["routes"].append({
            "handler": "sft_v2", "field": kw["field_route"],
            "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], model_type),
            "rows": len(rows), "baseline": kw["baseline_summary"], "spec": spec,
        })

    monkeypatch.setattr(sft_v2, "run", stop_v2)

    def stop_legacy(**kw):
        observed["routes"].append({"handler": "legacy", "field": False, "strategy": kw["strategy"]})
        raise StopAtLegacyPlan()

    monkeypatch.setattr(instruct, "make_sft_plan", stop_legacy)
    data = tmp_path / "train_data.json"
    answer = "x " * (280 if long_rows else 4)
    data.write_text(json.dumps([{"instruction": f"question {i}", "output": answer} for i in range(n_rows)]))
    stats = tmp_path / "baseline_stats.json"
    stats.write_text(json.dumps({
        "task_type": "instruct", "dataset": {"num_records": n_rows,
            "total_tokens": n_rows * (600 if long_rows else 8), "near_duplicate_rate": 0.0,  # >=8M tokens on the long case (token gate)
            "seq_length_distribution": {"p95": 300, "p99": 300, "max": 400}},
        "weights": {}, "training": {}, "throughput": None,
    }))
    monkeypatch.setenv("BASELINE_STATS_PATH", str(stats))
    args = ["--task-id", "cpu-anonymous-contract", "--model", ANONYMOUS_ID,
            "--dataset", str(data), "--dataset-type", json.dumps({"field_instruction": "instruction", "field_output": "output"}),
            "--task-type", "InstructTextTask", "--file-format", "s3",
            "--expected-repo-name", "cpu-contract", "--hours-to-complete", "1"]
    return args, observed, stats


@pytest.fixture(params=["release", "cab930c"])
def fullft(request, monkeypatch):
    """release = the full-FT default (no env); cab930c = the research switch FORGE_V2_FULLFT_DEFAULT=0."""
    if request.param == "cab930c":
        monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")
    else:
        monkeypatch.delenv("FORGE_V2_FULLFT_DEFAULT", raising=False)
    return request.param


@pytest.mark.parametrize("model_type,long_rows,cab930c_field,cab930c_strategy", [
    ("llama", True, True, "full"),
    ("qwen2", True, False, "lora"),
])
def test_actual_cli_dispatch_with_anonymous_id_and_stats(monkeypatch, tmp_path, fullft, model_type,
                                                        long_rows, cab930c_field, cab930c_strategy):
    expected_field, expected_strategy = (True, "full") if fullft == "release" else (cab930c_field, cab930c_strategy)
    args, seen, stats = prepare(monkeypatch, tmp_path, model_type, long_rows=long_rows)
    assert cli.main(args) == 0
    assert len(seen["routes"]) == 1
    route = seen["routes"][0]
    assert route["handler"] == "sft_v2"
    assert (route["field"], route["strategy"], route["rows"]) == (expected_field, expected_strategy, 20000)
    assert route["baseline"].num_records == 20000  # real file was parsed and accepted
    assert route["baseline"].task_type == "instruct"
    assert route["spec"].model == ANONYMOUS_ID
    assert route["spec"].baseline_stats_path == str(stats)
    assert seen["model_paths"] == [f"/cache/models/{ANONYMOUS_ID}"]
    events = {name for name, _ in seen["events"]}
    assert not events & {"baseline_stats_invalid", "sft_v2_pinned_route_skip", "sft_v2_failed",
                         "handler_failed", "UNEXPECTED_FALLBACK"}
    assert ("sft_v2_field_full_route" in events) is expected_field


def _spec_from(args):
    parsed = cli._parse(args)
    return TaskSpec.build(task_id=parsed.task_id, task_type=parsed.task_type,
                          model=parsed.model, dataset=parsed.dataset,
                          dataset_type_json=parsed.dataset_type,
                          expected_repo_name=parsed.expected_repo_name,
                          baseline_stats_path=parsed.baseline_stats, file_format=parsed.file_format)


def _dispatch_until_plan(seen, args):
    spec = _spec_from(args)
    try:
        dispatch.for_task(spec.task_type)(spec, Deadline.from_hours(
            1, started_monotonic=time.monotonic(), export_reserve_s=180))
    except StopAtLegacyPlan:
        pass
    return seen["routes"]


def test_short_qwen_anonymous_contract(monkeypatch, tmp_path, fullft):
    """cab930c: short rows -> legacy adapter. Release: full weights (no row-length gate)."""
    args, seen, stats = prepare(monkeypatch, tmp_path, "qwen2", long_rows=False)
    routes = _dispatch_until_plan(seen, args)
    events = {name for name, _ in seen["events"]}
    if fullft == "release":
        assert [(r["handler"], r["field"], r["strategy"]) for r in routes] == [("sft_v2", True, "full")]
        assert "sft_v2_field_full_route" in events
    else:
        assert routes == [{"handler": "legacy", "field": False, "strategy": "lora"}]
        assert "sft_v2_field_full_route" not in events
    assert any(meta.get("baseline_num_records") == 20000 for meta in seen["meta"])
    assert "baseline_stats_invalid" not in events


def test_below_threshold_llama(monkeypatch, tmp_path, fullft):
    """9,999 rows: below cab930c's llama row gate (10k) -> v2 adapter there; the release has no floor -> full."""
    args, seen, stats = prepare(monkeypatch, tmp_path, "llama", n_rows=9999)
    assert cli.main(args) == 0
    assert len(seen["routes"]) == 1
    want = (True, "full") if fullft == "release" else (False, "lora")
    assert (seen["routes"][0]["field"], seen["routes"][0]["strategy"]) == want


def test_short_llama_below_token_gate(monkeypatch, tmp_path, fullft):
    """20k short rows = 160k tokens: cab930c's 5M-token llama gate + short-row rule -> legacy; release -> full."""
    args, seen, stats = prepare(monkeypatch, tmp_path, "llama", long_rows=False)
    routes = _dispatch_until_plan(seen, args)
    if fullft == "release":
        assert [(r["handler"], r["field"], r["strategy"]) for r in routes] == [("sft_v2", True, "full")]
    else:
        assert routes == [{"handler": "legacy", "field": False, "strategy": "lora"}]
        assert not {name for name, _ in seen["events"]} & {"sft_v2_field_full_route"}
