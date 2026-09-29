"""LFM2.5 joins the field route (2026-09-16): allowlist + vocabulary, bf16 default for the
family, and the production pin yielding to the field route only when the field gates pass."""
import time
from types import SimpleNamespace as NS

import pytest

from forge import cli
from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tasks import dispatch, instruct, sft_v2
from forge.tuning import lfm25_epoch_cap as _lfm25
from tests.test_anonymous_field_dispatch_contract import StopAtLegacyPlan, prepare

LFM_PARAMS = 2_697_000_000  # LFM2.5-2.6B as measured on the harness (params_b 2.697)
LFM_CONFIG = dict(model_type="lfm2", architectures=["Lfm2ForCausalLM"], hidden_size=2048, intermediate_size=10752,
                  num_hidden_layers=30, num_attention_heads=32, num_key_value_heads=8, vocab_size=128000)


def test_lfm2_field_gate_matches_the_replayed_win(monkeypatch):
    monkeypatch.setenv("FORGE_V2_FIELD", "1")
    for k in ("FORGE_V2_FIELD_TYPES", "FORGE_V2_FIELD_MIN_ROWS", "FORGE_V2_FIELD_MIN_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    # MathFusionQA replay: 58,892 rows, 37.6M tokens, vocab 128000 -> field route
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128000) is True
    # a 20k-row / 2.5M-token LFM task stays below the token gate
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=20000,
                                   total_tokens=2_500_000, vocab_size=128000) is False
    # an lfm2 model with a different vocabulary (not LFM2.5) is not released
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=65536) is False
    # vocabularies are family-specific: llama with LFM's 128000 and lfm2 with Llama's 128256 are not the validated models
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=128000) is False
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128256) is False
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=128256) is True
    # an unknown family with a vocabulary given keeps its route; without model_type the older size-only contract holds
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="mistral", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=128256) is False
    # smaller lfm2 models (LFM2-1.2B: fp32 path, unvalidated) stay off; the 2.6B qualified configuration is admitted
    assert sft_v2.field_full_route(params_b=1.17, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128000) is False
    assert sft_v2.field_full_route(params_b=1.999, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128000) is False
    # llama has no size floor beyond the handler's (1B validated)
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=20000,
                                   total_tokens=12_000_000, vocab_size=None) is True
    # KL / multi-GPU / above the size cap stay off, as for llama
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=2, is_kl=False, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128000) is False
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=True, model_type="lfm2", n_rows=58892,
                                   total_tokens=37_592_948, vocab_size=128000) is False


@pytest.mark.parametrize("model_type,env,expected", [
    ("lfm2", None, True), ("llama", None, False), ("qwen2", None, True), ("gemma2", None, False),  # qwen2: identical-base release default ON (2026-09-17)
    ("lfm2", "0", False), ("llama", "1", True), ("LFM2", None, True), (None, None, False),
])
def test_field_bf16_default_is_family_scoped(monkeypatch, model_type, env, expected):
    if env is None:
        monkeypatch.delenv("FORGE_V2_FIELD_BF16", raising=False)
    else:
        monkeypatch.setenv("FORGE_V2_FIELD_BF16", env)
    assert sft_v2._field_bf16_enabled(model_type) is expected


def _lfm_model():
    return NS(config=NS(**LFM_CONFIG), parameters=lambda: iter([NS(numel=lambda: LFM_PARAMS)]))


def _run_dispatch(args):
    parsed = cli._parse(args)
    spec = TaskSpec.build(task_id=parsed.task_id, task_type=parsed.task_type, model=parsed.model, dataset=parsed.dataset,
                          dataset_type_json=parsed.dataset_type, expected_repo_name=parsed.expected_repo_name,
                          baseline_stats_path=parsed.baseline_stats, file_format=parsed.file_format)
    dispatch.for_task(spec.task_type)(spec, Deadline.from_hours(1, started_monotonic=time.monotonic(), export_reserve_s=180))


def _pin_lfm(monkeypatch, loaded_model):
    """The production pin matches the measured architecture and the anonymous cache form;
    the CPU contract cannot present /cache/models, so the route check is the only mocked part."""
    monkeypatch.setattr(_lfm25, "_supported_model_route", lambda spec: True)
    assert _lfm25._matches_base_model(loaded_model) is True
    monkeypatch.delenv("FORGE_V2_TAKES_LFM25", raising=False)


def test_lfm_pin_yields_to_field_route_when_gates_pass(monkeypatch, tmp_path):
    args, seen, stats = prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=True)  # 12M tokens, 20k rows
    loaded = instruct.load_base("x")
    loaded.model = _lfm_model()
    _pin_lfm(monkeypatch, loaded.model)
    cli.main(args)
    events = {name for name, _ in seen["events"]}
    assert "sft_v2_pin_released_to_field" in events
    assert "sft_v2_pinned_route_skip" not in events
    assert "UNEXPECTED_FALLBACK" not in events
    assert seen["routes"] == [seen["routes"][0]] and seen["routes"][0]["handler"] == "sft_v2"
    assert seen["routes"][0]["field"] is True and seen["routes"][0]["strategy"] == "full"


def test_lfm_pin_stands_below_the_token_gate(monkeypatch, tmp_path):
    # cab930c semantics (FORGE_V2_FULLFT_DEFAULT=0): the pin stands below the token gate. The release sends this in-scope
    # short-row task to full FT (the LFM2.5-2.6B short-row exception was removed 2026-09-29; see the test below).
    monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")
    args, seen, stats = prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=False)  # 160k tokens
    loaded = instruct.load_base("x")
    loaded.model = _lfm_model()
    _pin_lfm(monkeypatch, loaded.model)
    with pytest.raises(StopAtLegacyPlan):
        _run_dispatch(args)
    events = {name for name, _ in seen["events"]}
    assert "sft_v2_pinned_route_skip" in events
    assert "sft_v2_pin_released_to_field" not in events
    assert not {"sft_v2_field_full_route", "sft_v2_row_length_gate", "sft_v2_bigdata_full_route"} & events
    assert seen["routes"] == [{"handler": "legacy", "field": False, "strategy": "lora"}]


def test_lfm_pin_stands_when_field_disabled(monkeypatch, tmp_path):
    args, seen, stats = prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=True)
    loaded = instruct.load_base("x")
    loaded.model = _lfm_model()
    _pin_lfm(monkeypatch, loaded.model)
    monkeypatch.setenv("FORGE_V2_FIELD", "0")
    with pytest.raises(StopAtLegacyPlan):
        _run_dispatch(args)
    events = {name for name, _ in seen["events"]}
    assert "sft_v2_pinned_route_skip" in events and "sft_v2_pin_released_to_field" not in events


@pytest.mark.parametrize("fullft", ["release", "cab930c"])
def test_lfm_pin_without_a_known_token_count(monkeypatch, tmp_path, fullft):
    """No baseline total_tokens and no usable row-length estimate. cab930c: the protected pin is not released on
    guesswork. Release: a long-row LFM2.5-2.6B task is outside the short-row exception and has no token gate -> full."""
    if fullft == "cab930c":
        monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", "0")
    args, seen, stats = prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=True)
    import json as _json
    d = _json.loads(stats.read_text()); d["dataset"].pop("total_tokens", None); stats.write_text(_json.dumps(d))
    loaded = instruct.load_base("x")
    loaded.model = _lfm_model()
    _pin_lfm(monkeypatch, loaded.model)
    monkeypatch.setattr(sft_v2, "long_row_task", lambda rows, spec, tok: (True, None))
    if fullft == "release":
        _run_dispatch(args)
        events = {name for name, _ in seen["events"]}
        assert "sft_v2_pin_release_refused" not in events and "sft_v2_pin_released_to_field" in events
        assert [(r["handler"], r["field"], r["strategy"]) for r in seen["routes"]] == [("sft_v2", True, "full")]
        return
    with pytest.raises(StopAtLegacyPlan):
        _run_dispatch(args)
    events = {name for name, _ in seen["events"]}
    assert "sft_v2_pin_release_refused" in events and "sft_v2_pinned_route_skip" in events
    assert "sft_v2_pin_released_to_field" not in events and "sft_v2_field_full_route" not in events
    assert seen["routes"] == [{"handler": "legacy", "field": False, "strategy": "lora"}]


def test_release_lfm25_26b_short_rows_train_full(monkeypatch, tmp_path):
    """Release default: the in-scope LFM2.5-2.6B short-row task below the old token gate trains full weights (phase-an)."""
    monkeypatch.delenv("FORGE_V2_FULLFT_DEFAULT", raising=False)
    args, seen, stats = prepare(monkeypatch, tmp_path, "lfm2", n_rows=20000, long_rows=False)
    loaded = instruct.load_base("x")
    loaded.model = _lfm_model()
    _pin_lfm(monkeypatch, loaded.model)
    cli.main(args)
    assert seen["routes"] and seen["routes"][0]["handler"] == "sft_v2"
    assert seen["routes"][0]["field"] is True and seen["routes"][0]["strategy"] == "full"
