"""CPU dispatch contract for the Falcon-RW-1B legacy-route veto."""

from types import SimpleNamespace as NS
import time
from functools import lru_cache
import os
from pathlib import Path

import pytest

from forge.clock import Deadline
from forge.data.schema import InstructColumns, TaskSpec
from forge.tasks import instruct, sft_v2
from forge.tuning import plan as tuning_plan


class _StopAfterPlan(RuntimeError):
    pass


def _config(**changes):
    values = dict(
        model_type="falcon",
        architectures=["FalconForCausalLM"],
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        vocab_size=50304,
        bos_token_id=1,
        eos_token_id=2,
        multi_query=False,
        new_decoder_architecture=False,
        parallel_attn=False,
        alibi=True,
        max_position_embeddings=2048,
    )
    values.update(changes)
    return NS(**values)


def _spec(*, output="output", model="0123456789abcdef", baseline=None):
    return TaskSpec(
        task_id="falcon-policy-cpu",
        task_type="InstructTextTask",
        model=model,
        dataset=None,
        expected_repo_name="cpu-only",
        baseline_stats_path=baseline,
        instruct=InstructColumns("instruction", output),
    )


def _native(config):
    from transformers.models.falcon.modeling_falcon import FalconForCausalLM

    model = FalconForCausalLM.__new__(FalconForCausalLM)
    model.config = config
    return model


class _SyntheticTokenizer:
    def __init__(self, *, bos_id=50256, eos_id=50256, bos_token="<|endoftext|>", eos_token="<|endoftext|>"):
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.bos_token = bos_token
        self.eos_token = eos_token

    def get_vocab(self):
        return {self.bos_token: self.bos_token_id, self.eos_token: self.eos_token_id}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token)


def _tokenizer(**kwargs):
    return _SyntheticTokenizer(**kwargs)


@lru_cache(maxsize=1)
def _retained_tokenizer():
    from transformers import AutoTokenizer

    relative = Path(
        "experiments/20260902-survival-geometry/calibration/tokenizers/"
        "tiiuae-falcon-rw-1b"
    )
    candidates = []
    configured = os.environ.get("SN56_FALCON_TOKENIZER_DIR")
    if configured:
        if not (Path(configured) / "tokenizer_config.json").is_file():
            pytest.fail("SN56_FALCON_TOKENIZER_DIR does not contain the retained tokenizer")
        candidates.append(Path(configured))
    candidates.extend(parent / relative for parent in Path(__file__).resolve().parents)
    tokenizer_dir = next(
        (path for path in candidates if (path / "tokenizer_config.json").is_file()),
        None,
    )
    if tokenizer_dir is None:
        pytest.skip(
            "retained Falcon tokenizer unavailable; set SN56_FALCON_TOKENIZER_DIR"
        )
    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
    )


def _run_with_falcon(
    monkeypatch, *, long_rows, config=None, model_id="0123456789abcdef",
    baseline=False, expect_v2=False, tmp_path=None,
):
    config = config or _config()
    model = _native(config)
    loaded = NS(
        model=model,
        tokenizer=_retained_tokenizer(),
        model_dir="/cache/models/arbitrary-anonymous-cache",
    )
    events = []
    v2_calls = {"strategy": 0, "length": 0, "eligible": 0, "run": 0, "median": None}
    legacy_plan = {}
    rows = [
        {"instruction": "Answer in detail", "output": "word " * (400 if long_rows else 20)}
        for _ in range(512)
    ]

    monkeypatch.setattr(
        instruct.loader,
        "load_rows",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(instruct, "load_base", lambda *args, **kwargs: loaded)
    baseline_path = None
    if baseline:
        assert tmp_path is not None
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            '{"task_type":"instruct","dataset":{"seq_length_distribution":'
            '{"p95":800,"p99":1000,"max":1200},"near_duplicate_rate":0.0,'
            '"total_tokens":1000,"num_records":512},"throughput":null,'
            '"weights":{},"training":{}}',
            encoding="utf-8",
        )
    monkeypatch.setattr(instruct, "model_param_billions", lambda model: 1.312)
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (1, 80.0))
    monkeypatch.setattr(instruct, "is_qwen35_model", lambda model: False)
    monkeypatch.setattr(instruct, "_is_bloomz_promotion", lambda *args: False)
    monkeypatch.setenv("FORGE_SFT_V2_ALLOW_CPU", "1")
    monkeypatch.setattr(instruct.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(instruct.telemetry, "set_meta", lambda **kwargs: None)
    monkeypatch.setattr(instruct.telemetry, "event", lambda name, **kwargs: events.append((name, kwargs)))
    monkeypatch.setattr(instruct.telemetry, "write_into", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_plan, "_hardware", lambda: (True, True))
    for name in ("FORGE_LORA_R", "FORGE_LORA_ALPHA", "FORGE_LORA_DROPOUT", "FORGE_LORA_LR"):
        monkeypatch.delenv(name, raising=False)
    real_make_sft_plan = instruct.make_sft_plan

    def make_sft_plan(**kwargs):
        plan = real_make_sft_plan(**kwargs)
        legacy_plan["value"] = plan
        return plan

    monkeypatch.setattr(instruct, "make_sft_plan", make_sft_plan)

    real_strategy = sft_v2.strategy_for

    def strategy(*args, **kwargs):
        v2_calls["strategy"] += 1
        return real_strategy(*args, **kwargs)

    real_long_row_task = sft_v2.long_row_task

    def length(*args, **kwargs):
        v2_calls["length"] += 1
        result = real_long_row_task(*args, **kwargs)
        v2_calls["median"] = result[1]
        return result

    real_eligible = sft_v2.eligible

    def eligible(*args, **kwargs):
        v2_calls["eligible"] += 1
        return real_eligible(*args, **kwargs)

    monkeypatch.setattr(sft_v2, "strategy_for", lambda *args, **kwargs: strategy(*args, **kwargs))
    monkeypatch.setattr(sft_v2, "long_row_task", length)
    monkeypatch.setattr(sft_v2, "eligible", eligible)
    monkeypatch.setattr(sft_v2, "run", lambda *args, **kwargs: v2_calls.__setitem__("run", v2_calls["run"] + 1))
    monkeypatch.setattr(instruct, "decide_full_finetune", lambda **kwargs: False)
    monkeypatch.setattr(instruct, "attach_lora", lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterPlan()))

    run_spec = _spec(model=model_id, baseline=str(baseline_path) if baseline_path else None)
    if expect_v2:
        instruct.run(
            run_spec,
            Deadline.from_hours(
                1.0, started_monotonic=time.monotonic(), export_reserve_s=30
            ),
        )
    else:
        with pytest.raises(_StopAfterPlan):
            instruct.run(
                run_spec,
                Deadline.from_hours(
                    1.0, started_monotonic=time.monotonic(), export_reserve_s=30
                ),
            )
    return v2_calls, events, legacy_plan.get("value")


def test_structural_identity_and_contract_are_strict():
    base = _config()
    assert instruct.eligible_falcon_rw1b_legacy_route(
        _spec(model="anonymous", baseline=None), _native(base), _tokenizer(), is_kl=False, n_gpus=1
    )
    assert instruct.eligible_falcon_rw1b_legacy_route(
        _spec(model="tiiuae/falcon-rw-1b", baseline="/arbitrary"), _native(base), _tokenizer(), is_kl=False, n_gpus=1
    )
    for changes in (
        {"hidden_size": 4544},
        {"hidden_size": "2048"},
        {"new_decoder_architecture": True},
        {"new_decoder_architecture": None},
        {"alibi": "true"},
        {"alibi": None},
        {"vocab_size": "50304"},
        {"vocab_size": True},
        {"model_type": None},
        {"architectures": None},
        {"num_hidden_layers": None},
        {"num_hidden_layers": "24"},
        {"num_attention_heads": None},
        {"num_attention_heads": True},
        {"bos_token_id": None},
        {"bos_token_id": False},
        {"eos_token_id": "2"},
        {"parallel_attn": None},
        {"multi_query": None},
        {"parallel_attn": "false"},
    ):
        assert not instruct.eligible_falcon_rw1b_legacy_route(
            _spec(), _native(_config(**changes)), _tokenizer(), is_kl=False, n_gpus=1
        )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), NS(config=base), _tokenizer(), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(output=None), _native(base), _tokenizer(), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), _tokenizer(bos_id=50304), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), _tokenizer(eos_token="<bad>"), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), NS(
            bos_token_id=50256, eos_token_id=50256,
            bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        ), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), _tokenizer(eos_id=50255), is_kl=False, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), _tokenizer(), is_kl=True, n_gpus=1
    )
    assert not instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), _native(base), _tokenizer(), is_kl=False, n_gpus=2
    )


@pytest.mark.parametrize("long_rows", [True, False])
@pytest.mark.parametrize("model_id", ["0123456789abcdef", "tiiuae/falcon-rw-1b"])
@pytest.mark.parametrize("baseline", [False, True])
def test_matching_falcon_preserves_preprocessing_but_vetoes_final_v2(
    monkeypatch, long_rows, model_id, baseline, tmp_path
):
    calls, events, plan = _run_with_falcon(
        monkeypatch, long_rows=long_rows, model_id=model_id, baseline=baseline,
        tmp_path=tmp_path,
    )
    assert calls["strategy"] == 1
    assert calls["length"] == 1
    assert calls["eligible"] == 0
    assert calls["run"] == 0
    assert (calls["median"] >= 256) is long_rows
    assert plan is not None
    assert (plan.lora_r, plan.lora_alpha, plan.lora_dropout) == (32, 64, 0.05)
    assert plan.learning_rate == 1.5e-4
    assert (plan.per_device_batch_size, plan.grad_accum_steps) == (4, 4)
    assert any(name == "falcon_rw1b_legacy_route" for name, _ in events)


def test_near_miss_reaches_current_generic_v2_gate(monkeypatch):
    calls, events, plan = _run_with_falcon(
        monkeypatch, long_rows=True, config=_config(alibi=False), expect_v2=True
    )
    assert calls["strategy"] == 2
    assert calls["length"] == 1
    assert calls["eligible"] == 1
    assert calls["run"] == 1
    assert calls["median"] >= 256
    assert plan is None
    assert not any(name == "falcon_rw1b_legacy_route" for name, _ in events)


def test_current_legacy_plan_values_remain_unchanged(monkeypatch):
    for name in ("FORGE_LORA_R", "FORGE_LORA_ALPHA", "FORGE_LORA_DROPOUT", "FORGE_LORA_LR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tuning_plan, "_hardware", lambda: (True, True))
    plan = tuning_plan.make_sft_plan(
        use_kl=False, strategy="lora", params_b=1.312, n_gpus=1, per_gpu_gb=80.0
    )
    assert (plan.lora_r, plan.lora_alpha, plan.lora_dropout) == (32, 64, 0.05)
    assert plan.learning_rate == 1.5e-4
    assert (plan.per_device_batch_size, plan.grad_accum_steps) == (4, 4)
    assert plan.max_seq_len == 4096
    assert instruct.effective_sft_seq_len(NS(config=_config()), plan.max_seq_len) == 1024


def test_retained_falcon_tokenizer_preserves_real_length_gate():
    from transformers.models.falcon.modeling_falcon import FalconForCausalLM

    tokenizer = _retained_tokenizer()
    native = FalconForCausalLM.__new__(FalconForCausalLM)
    native.config = _config()
    assert instruct.eligible_falcon_rw1b_legacy_route(
        _spec(), native, tokenizer, is_kl=False, n_gpus=1
    )
    short_rows = [{"instruction": "Answer briefly", "output": "word " * 20} for _ in range(512)]
    long_rows = [{"instruction": "Answer in detail", "output": "word " * 400} for _ in range(512)]
    short, short_median = sft_v2.long_row_task(short_rows, _spec(), tokenizer)
    long, long_median = sft_v2.long_row_task(long_rows, _spec(), tokenizer)
    assert short is False and short_median < 256
    assert long is True and long_median >= 256
