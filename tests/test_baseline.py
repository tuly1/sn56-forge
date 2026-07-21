import json

import pytest

from forge.baseline import (
    load_baseline_summary,
    suggested_sequence_length,
    telemetry_fields,
)


def _payload(task_type="instruct"):
    return {
        "task_type": task_type,
        "dataset": {
            "total_tokens": 123_456,
            "num_records": 8_000,
            "seq_length_distribution": {
                "mean": 500.0,
                "p50": 384,
                "p95": 900,
                "p99": 1100,
                "max": 2048,
            },
            "near_duplicate_rate": 0.125,
        },
        "weights": {"by_group": {}},
        "training": {"init_loss": 2.0},
        "throughput": {"tokens_per_sec": 20_000.0},
    }


def test_load_baseline_summary_from_json_and_hash_is_canonical():
    first = load_baseline_summary(
        json.dumps(_payload(), sort_keys=False), expected_task_type="InstructTextTask"
    )
    second = load_baseline_summary(
        json.dumps(_payload(), sort_keys=True), expected_task_type="instruct"
    )

    assert first == second
    assert first.sequence_p99 == 1100
    assert first.tokens_per_sec == 20_000.0
    assert telemetry_fields(first)["baseline_stats_sha256"] == first.sha256


def test_load_baseline_summary_from_file(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_payload("dpo")))

    summary = load_baseline_summary(str(path), expected_task_type="DpoTask")

    assert summary.task_type == "dpo"
    assert summary.num_records == 8_000


def test_load_baseline_summary_rejects_task_mismatch_and_bad_quantiles():
    with pytest.raises(ValueError, match="does not match"):
        load_baseline_summary(json.dumps(_payload("grpo")), expected_task_type="DpoTask")

    bad = _payload()
    bad["dataset"]["seq_length_distribution"]["p99"] = 3000
    with pytest.raises(ValueError, match="p95 <= p99 <= max"):
        load_baseline_summary(json.dumps(bad), expected_task_type="instruct")


def test_load_baseline_summary_rejects_nonfinite_or_out_of_range_values():
    bad = _payload()
    bad["dataset"]["near_duplicate_rate"] = 1.1
    with pytest.raises(ValueError, match="near_duplicate_rate"):
        load_baseline_summary(json.dumps(bad), expected_task_type="instruct")

    bad = _payload()
    bad["throughput"]["tokens_per_sec"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        load_baseline_summary(json.dumps(bad), expected_task_type="instruct")

    bad = _payload()
    bad["dataset"]["total_tokens"] = 10**1_000
    with pytest.raises(ValueError, match="between 0"):
        load_baseline_summary(json.dumps(bad), expected_task_type="instruct")


def test_load_baseline_summary_normalizes_deep_json_parser_failures(monkeypatch):
    def _raise_recursion(_payload):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(json, "loads", _raise_recursion)
    with pytest.raises(ValueError, match="invalid baseline-stats JSON"):
        load_baseline_summary("{}", expected_task_type="instruct")


def test_suggested_sequence_length_rounds_p99_and_never_exceeds_default():
    summary = load_baseline_summary(json.dumps(_payload()), expected_task_type="instruct")

    assert suggested_sequence_length(summary, default=4096) == 1280
    assert suggested_sequence_length(summary, default=1024) == 1024
    assert suggested_sequence_length(
        summary, default=4096, completion_reserve=256
    ) == 1536
    assert suggested_sequence_length(None, default=4096) == 4096
