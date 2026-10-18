"""Route-table release gate for the full-FT default (release-fullft-default-20260929; owner directives 1-4).

The REAL dispatcher (forge.tasks.instruct.run up to the handler call) runs on CPU for every pool model x small /
medium / large data x short / long rows x clean / damaged / unknown detector verdicts. Only model loading, row
loading, the baseline file, tokenization inside long_row_task, the identity / damage detectors and the handlers
(sft_v2.run, the legacy plan) are doubles; the scope check, exception matching, pin / falcon predicates, damage
detection gating, field-route flags and strategy resolution are the deployed code.

FAILS if any in-scope task (single GPU, non-KL InstructTextTask, params_b <= 4.25) lands on an adapter without a listed
exception, or if an out-of-scope task's route differs from cab930c (FORGE_V2_FULLFT_DEFAULT=0).

Pool = every model of the shipped fingerprint table (forge/routing/data/base_fingerprints.json, 102 validator-pool
bases incl. the 2026-09-27 addendum; the 7 out-of-scope 8 B past-winner repos were removed 2026-10-18) + the CURRENT-CAMPAIGN pool members missing from it (NuExtract-v1.5 / phi3,
Qwen1.5-0.5B-Chat, pythia-410m) + the loaded sizes of Qwen3.5 (text-only load 4.2058 B, receipt
research/20260925-release-lfm-q35drift/CPU-PRECHECK.json).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from forge.clock import Deadline
from forge.data.schema import InstructColumns, TaskSpec
from forge.routing import damage as damage_mod
from forge.routing import identity_report as idr
from forge.tasks import instruct, sft_v2

ROOT = Path(__file__).resolve().parents[1]
ANON = "0123456789abcdef"

# ---- pool ------------------------------------------------------------------------------------------------------------
# exact architecture fields of the two pinned production routes (forge.tuning.{lfm25,granite41}_epoch_cap)
LFM26_SHAPE = dict(hidden_size=2048, intermediate_size=10752, num_hidden_layers=30, num_attention_heads=32,
                   num_key_value_heads=8)
GRANITE41_3B_SHAPE = dict(hidden_size=2560, intermediate_size=8192, num_hidden_layers=40, num_attention_heads=40,
                          num_key_value_heads=8)
FALCON_RW1B = dict(hidden_size=2048, num_hidden_layers=24, num_attention_heads=32, vocab_size=50304, bos_token_id=1,
                   eos_token_id=2, multi_query=False, new_decoder_architecture=False, parallel_attn=False, alibi=True,
                   max_position_embeddings=2048)
QWEN35_VOCAB = 248320
QWEN35_LOADED_B = {"Qwen/Qwen3.5-0.8B": 0.873, "Qwen/Qwen3.5-0.8B-Base": 0.873, "Qwen/Qwen3.5-2B": 2.274,
                   "Qwen/Qwen3.5-2B-Base": 2.274, "Qwen/Qwen3.5-4B": 4.2058, "Qwen/Qwen3.5-4B-Base": 4.2058}
EXTRA_POOL = [  # CURRENT-CAMPAIGN pool members without a fingerprint entry
    dict(model="numind/NuExtract-v1.5", model_type="phi3", vocab=32064, params_b=3.821, arch="Phi3ForCausalLM"),
    dict(model="Qwen/Qwen1.5-0.5B-Chat", model_type="qwen2", vocab=151936, params_b=0.620, arch="Qwen2ForCausalLM"),
    dict(model="EleutherAI/pythia-410m", model_type="gpt_neox", vocab=50304, params_b=0.405, arch="GPTNeoXForCausalLM"),
    dict(model="Qwen/Qwen3.5-4B (vision tower loaded)", model_type="qwen3_5", vocab=QWEN35_VOCAB, params_b=4.66,
         arch="Qwen3_5ForConditionalGeneration"),
]


def pool() -> list[dict]:
    table = json.loads((ROOT / "forge/routing/data/base_fingerprints.json").read_text())["models"]
    out = []
    for mid, v in sorted(table.items()):
        c = v["config"]
        mt = str(c.get("model_type"))
        params_b = QWEN35_LOADED_B.get(mid) or sum(math.prod(t[1]) for t in v["tensors"].values()) / 1e9
        vocab = c.get("vocab_size") or (QWEN35_VOCAB if mt == "qwen3_5" else None)
        out.append(dict(model=mid, model_type=mt, vocab=vocab, params_b=round(params_b, 4),
                        arch=(c.get("architectures") or [""])[0], layers=c.get("num_hidden_layers"),
                        hidden=c.get("hidden_size")))
    return out + EXTRA_POOL


def in_scope_size(params_b: float) -> bool:
    return 0 < params_b <= sft_v2.FULLFT_DEFAULT_MAX_PARAMS_B


# ---- dispatcher harness ----------------------------------------------------------------------------------------------
DATA = {  # rows, tokens: small / medium (= 803a8f69) / large
    "small": (600, 90_000),
    "medium": (14_959, 1_377_813),
    "large": (80_000, 20_000_000),
}
ROWS = {"short": 89, "long": 614}  # sampled tokenized medians (803a8f69 = 89; 90d361cb = 614)


class StopAtLegacyPlan(BaseException):
    pass


class Tok:
    pad_token_id = 0
    bos_token = eos_token = "<|endoftext|>"
    bos_token_id = eos_token_id = 50256

    def get_vocab(self):
        return {"<|endoftext|>": 50256}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token)


def model_double(entry: dict):
    mt = entry["model_type"]
    cfg = dict(model_type=mt, architectures=[entry["arch"]], vocab_size=entry["vocab"],
               num_hidden_layers=entry.get("layers"), hidden_size=entry.get("hidden"))
    if mt == "lfm2" and entry.get("layers") == 30 and entry.get("hidden") == 2048:
        cfg.update(LFM26_SHAPE)
    if mt == "granite" and entry.get("layers") == 40 and entry.get("hidden") == 2560:
        cfg.update(GRANITE41_3B_SHAPE)
    if mt == "falcon" and entry["model"] == "tiiuae/falcon-rw-1b":
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM

        cfg.update(FALCON_RW1B)
        m = FalconForCausalLM.__new__(FalconForCausalLM)
        m.config = NS(**cfg)
        return m
    if mt == "qwen3_5":
        cfg["text_config"] = NS(model_type="qwen3_5_text", vocab_size=QWEN35_VOCAB)
    return NS(config=NS(**cfg))


def dispatch(monkeypatch, entry: dict, *, data: str, rows: str, verdict: str, n_gpus: int = 1, use_kl: bool = False,
             switch: str | None = None, hours: float = 1.0, v2_handler=None, legacy_handler=None,
             events_out: list | None = None, env: dict | None = None) -> dict:
    """Run instruct.run to the handler call. Returns {"handler", "field", "strategy", "recipe", "exception", ...}."""
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    if switch is not None:
        monkeypatch.setenv("FORGE_V2_FULLFT_DEFAULT", switch)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    n_rows, tokens = DATA[data]
    median = ROWS[rows]
    events: list = [] if events_out is None else events_out
    routes: list = []
    model = model_double(entry)
    loaded = NS(model=model, tokenizer=Tok(), model_dir=f"/cache/models/{ANON}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (n_gpus, 80.0))
    monkeypatch.setattr(instruct, "load_base", lambda path, **kw: loaded)
    monkeypatch.setattr(instruct, "model_param_billions", lambda m: entry["params_b"])
    monkeypatch.setattr(instruct.loader, "load_rows", lambda *a, **k: [{"instruction": "q", "output": "a"}] * n_rows)
    monkeypatch.setattr(instruct, "load_baseline_summary",
                        lambda *a, **k: NS(total_tokens=tokens, num_records=n_rows, task_type="instruct"))
    monkeypatch.setattr(instruct, "telemetry_fields", lambda s: {})
    for name in ("collect_env", "set_meta", "write_into"):
        monkeypatch.setattr(instruct.telemetry, name, lambda *a, **k: None)
    monkeypatch.setattr(instruct.telemetry, "event", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda picked, cols: list(picked))
    monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct", lambda ex, tok, cap: [{"input_ids": [5] * median} for _ in ex])
    monkeypatch.setattr(idr, "identify", lambda model_dir, **kw: {"status": "unknown", "reason": "double"})
    ref = {"lfm2": "LiquidAI/LFM2.5-1.2B-Instruct"}.get(entry["model_type"], entry["model"])
    detectors = {} if verdict == "unknown" else {"fingerprint": {"verdict": verdict, "reason": "double", "source": ref}}
    monkeypatch.setattr(damage_mod, "assess", lambda *a, **k: {"verdict": verdict, "by": "double",
                                                              "detectors": detectors, "seconds": 0.0})

    def stop_v2(spec, deadline, rows_, loaded_arg, tokenizer, **kw):
        routes.append({"handler": "sft_v2", "field": kw["field_route"], "recipe": kw.get("recipe_key"),
                       "strategy": kw["strategy_override"] or sft_v2.strategy_for(kw["params_b"], entry["model_type"]),
                       "damaged": kw["damaged_base_route"], "cheap": kw.get("cheap_full"),
                       "kw": {k: kw.get(k) for k in ("params_b", "per_gpu_gb", "strategy_override", "route_lr",
                                                     "field_route", "damaged_base_route", "clean_small_qwen_route",
                                                     "clean_small_lfm2_route", "recipe_key", "cheap_full")}})

    def stop_legacy(**kw):
        routes.append({"handler": "legacy", "field": False, "strategy": kw["strategy"], "recipe": None})
        raise StopAtLegacyPlan()

    if v2_handler is not None:
        def stop_v2(spec, deadline, rows_, loaded_arg, tokenizer, _record=stop_v2, **kw):  # noqa: F811
            _record(spec, deadline, rows_, loaded_arg, tokenizer, **kw)
            v2_handler(**kw)
    if legacy_handler is not None:
        def stop_legacy(_record=stop_legacy, **kw):  # noqa: F811
            try:
                _record(**kw)
            except StopAtLegacyPlan:
                pass
            return legacy_handler(**kw)
    monkeypatch.setattr(sft_v2, "run", stop_v2)
    monkeypatch.setattr(instruct, "make_sft_plan", stop_legacy)
    spec = TaskSpec(task_id="route-table", task_type="InstructTextTask", model=ANON, dataset=None,
                    expected_repo_name="cpu", baseline_stats_path="/cache/baseline_stats/t.json",
                    instruct=InstructColumns("instruction", "output"), use_kl=use_kl, kl_coef=0.1 if use_kl else 0.0)
    try:
        instruct.run(spec, Deadline.from_hours(hours, started_monotonic=time.monotonic(), export_reserve_s=180))
    except StopAtLegacyPlan:
        pass
    if v2_handler is not None or legacy_handler is not None:
        return {"routes": routes, "events": events}
    assert len(routes) == 1, (entry["model"], routes)
    ev = {n: kw for n, kw in events}
    r = dict(routes[0])
    r["exception"] = (ev.get("sft_v2_adapter_exception") or {}).get("exception")
    r["decision"] = ev.get("sft_v2_fullft_default")
    r["scope"] = ev.get("sft_v2_fullft_scope")
    return r


def is_full(r: dict) -> bool:
    return r["handler"] == "sft_v2" and r["field"] is True and r["strategy"] == "full"


def expected_exception(entry: dict, rows: str) -> str | None:
    if entry["model"] == "tiiuae/falcon-rw-1b":
        return "quality:falcon-rw-1b"
    # "quality:lfm2.5-2.6b-short-rows" was removed 2026-09-29 (phase-an: full FT beat the pinned adapter)
    return None


EXPECTED_RECIPE = {"qwen2": "family", "qwen3": "family", "gemma2": "family", "lfm2": "family", "llama": "llama"}


def expected_recipe(entry: dict) -> str:
    if entry["model_type"] == "llama" and entry["vocab"] == 130560:
        return "minicpm5"
    return EXPECTED_RECIPE.get(entry["model_type"], "generic")


POOL = pool()
IN_SCOPE = [e for e in POOL if in_scope_size(e["params_b"])]
OUT_OF_SCOPE = [e for e in POOL if not in_scope_size(e["params_b"])]


def test_pool_covers_the_named_families():
    names = {e["model"] for e in IN_SCOPE}
    for must in ("numind/NuExtract-v1.5", "Qwen/Qwen1.5-0.5B-Chat", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B",
                 "ibm-granite/granite-4.1-3b", "unsloth/SmolLM2-1.7B", "unsloth/Llama-3.2-1B", "unsloth/Llama-3.2-3B",
                 "LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B", "LiquidAI/LFM2.5-1.2B-Instruct",
                 "LiquidAI/LFM2.5-2.6B", "openbmb/MiniCPM5-1B", "unsloth/gemma-2-2b", "Qwen/Qwen2-0.5B",
                 "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-4B", "tiiuae/falcon-rw-1b", "unsloth/Phi-3-mini-4k-instruct",
                 "unsloth/codegemma-2b", "microsoft/phi-1_5", "bigscience/bloomz-560m", "EleutherAI/pythia-410m"):
        assert must in names, must
    assert len(IN_SCOPE) >= 70 and len(POOL) >= 106  # 2026-10-18: -7 past-winner 8 B repos (all out of scope)


@pytest.mark.parametrize("entry", IN_SCOPE, ids=[e["model"] for e in IN_SCOPE])
def test_in_scope_route_table(monkeypatch, entry):
    """Every in-scope pool model x 3 data sizes x 2 row lengths x 3 verdicts: full FT unless a listed exception."""
    for data in DATA:
        for rows in ROWS:
            want_exc = expected_exception(entry, rows)
            for verdict in ("clean", "damaged", "unknown"):
                r = dispatch(monkeypatch, entry, data=data, rows=rows, verdict=verdict)
                cell = (entry["model"], data, rows, verdict, r)
                if want_exc is None:
                    assert is_full(r), cell  # the release gate: no in-scope adapter without an exception
                    assert r["exception"] is None and r["recipe"] == expected_recipe(entry), cell
                    assert r["decision"]["route"] == "full", cell
                else:
                    assert r["handler"] == "legacy" and r["exception"] == want_exc, cell


@pytest.mark.parametrize("entry", OUT_OF_SCOPE, ids=[e["model"] for e in OUT_OF_SCOPE])
def test_out_of_scope_is_cab930c(monkeypatch, entry):
    """> 4.25 B: the scope check refuses ("size") and the route equals FORGE_V2_FULLFT_DEFAULT=0 (cab930c)."""
    for data in ("small", "large"):
        for rows in ROWS:
            new = dispatch(monkeypatch, entry, data=data, rows=rows, verdict="unknown")
            old = dispatch(monkeypatch, entry, data=data, rows=rows, verdict="unknown", switch="0")
            assert new["scope"]["refusal"] in ("size", "quasar_remote_code")
            strip = lambda r: {k: v for k, v in r.items() if k not in ("decision", "scope")}  # noqa: E731
            assert strip(new) == strip(old), (entry["model"], data, rows, new, old)


@pytest.mark.parametrize("name,kw,refusal", [
    ("multi_gpu", dict(n_gpus=2), "not_single_gpu"),
    ("kl", dict(use_kl=True), "kl"),
])
def test_multi_gpu_and_kl_are_cab930c(monkeypatch, name, kw, refusal):
    for entry in [e for e in IN_SCOPE if e["model"] in ("unsloth/Llama-3.2-1B", "Qwen/Qwen2.5-3B-Instruct",
                                                          "LiquidAI/LFM2.5-1.2B-Instruct", "unsloth/gemma-2-2b")]:
        new = dispatch(monkeypatch, entry, data="large", rows="long", verdict="damaged", **kw)
        old = dispatch(monkeypatch, entry, data="large", rows="long", verdict="damaged", switch="0", **kw)
        assert new["scope"]["refusal"] == refusal
        strip = lambda r: {k: v for k, v in r.items() if k not in ("decision", "scope")}  # noqa: E731
        assert strip(new) == strip(old)


def test_803a8f69_shape_is_full_ft(monkeypatch):
    """09-28 R1: clean LFM2.5-1.2B-Instruct, 14,959 rows / 1.38M tokens, median 89 (13th of 13 on the legacy adapter)."""
    entry = next(e for e in IN_SCOPE if e["model"] == "LiquidAI/LFM2.5-1.2B-Instruct")
    r = dispatch(monkeypatch, entry, data="medium", rows="short", verdict="clean")
    assert is_full(r) and r["recipe"] == "family"
    assert r["decision"]["rows"] == 14_959 and r["decision"]["total_tokens"] == 1_377_813
    rec = sft_v2.field_recipe("lfm2", entry["params_b"], damaged_base=True)
    assert rec["bf16"] is False and rec["lr_mult"] == 2.0  # fp32 master, LR x2 (the Phase AJ / Phase H recipe)
    old = dispatch(monkeypatch, entry, data="medium", rows="short", verdict="clean", switch="0")
    assert old["handler"] == "legacy"  # what cab930c shipped (the floors refused it)


def test_lfm25_26b_trains_full_at_every_row_length_equal_to_phase_an_arm_f(monkeypatch):
    """The LFM2.5-2.6B short-row exception is gone (phase-an REDUCE-20260929: full FT DEV -0.0408, 99% CI < 0). Every
    cell trains the family full-FT route, and the handler receives exactly what arm F ran (FORGE_V2_TAKES_LFM25=1)."""
    entry = next(e for e in IN_SCOPE if e["model"] == "LiquidAI/LFM2.5-2.6B")
    for verdict in ("clean", "damaged", "unknown"):
        for data in DATA:
            for rows in ROWS:
                r = dispatch(monkeypatch, entry, data=data, rows=rows, verdict=verdict)
                f = dispatch(monkeypatch, entry, data=data, rows=rows, verdict=verdict, env={"FORGE_V2_TAKES_LFM25": "1"})
                assert is_full(r) and r["recipe"] == "family" and r["exception"] is None, (data, rows, verdict, r)
                assert r["kw"] == f["kw"], (data, rows, verdict, r["kw"], f["kw"])
    lfm2_26 = next(e for e in IN_SCOPE if e["model"] == "LiquidAI/LFM2-2.6B")  # same architecture, vocab 65536
    assert is_full(dispatch(monkeypatch, lfm2_26, data="small", rows="short", verdict="clean"))
    assert sft_v2.QUALITY_EXCEPTIONS == ("falcon-rw-1b",)


def test_lfm25_26b_out_of_scope_keeps_the_pin(monkeypatch):
    entry = next(e for e in IN_SCOPE if e["model"] == "LiquidAI/LFM2.5-2.6B")
    for kw in (dict(n_gpus=2), dict(use_kl=True)):
        new = dispatch(monkeypatch, entry, data="large", rows="short", verdict="clean", **kw)
        old = dispatch(monkeypatch, entry, data="large", rows="short", verdict="clean", switch="0", **kw)
        strip = lambda r: {k: v for k, v in r.items() if k not in ("decision", "scope")}  # noqa: E731
        assert strip(new) == strip(old)


def test_lfm25_26b_median_never_refuses():
    for median in (0, 1, 155, 255, 256, 257, None):
        for long_rows in (False, True, None):
            d = sft_v2.fullft_default_route(model_type="lfm2", params_b=2.697, vocab_size=128000, lfm25_26b=True,
                                            median_tokens=median, long_rows=long_rows)
            assert d.route == "full" and d.recipe == "family", (median, long_rows, d)


def test_crash_exception_hook(monkeypatch):
    assert sft_v2.CRASH_EXCEPTIONS == {}  # none yet: the crash sweep has not run
    entry = next(e for e in IN_SCOPE if e["model"] == "numind/NuExtract-v1.5")
    monkeypatch.setitem(sft_v2.CRASH_EXCEPTIONS, "phi3", {"since": "2026-10-01", "test": "crash-sweep/phi3"})
    r = dispatch(monkeypatch, entry, data="large", rows="long", verdict="unknown")
    # dispatch() clears FORGE_ env but not the table: the family trains the (fixed) legacy adapter
    assert r["handler"] == "legacy" and r["exception"] == "crash:phi3:2026-10-01:crash-sweep/phi3"
    monkeypatch.setitem(sft_v2.CRASH_EXCEPTIONS, "phi3", {"since": "x", "test": "y", "max_params_b": 3.0})
    assert sft_v2.crash_exception("phi3", 3.8) is None and sft_v2.crash_exception("phi3", 2.0)


def test_size_cap_includes_qwen35_4b_and_nothing_else_between():
    assert sft_v2.FULLFT_DEFAULT_MAX_PARAMS_B == 4.25
    assert in_scope_size(4.2058) and not in_scope_size(4.66)
    assert not [e for e in POOL if 4.2058 < e["params_b"] <= 4.25]


def test_detector_verdict_never_changes_full_vs_adapter(monkeypatch):
    """Directive 1: identity / damage verdicts choose hyperparameters only."""
    for entry in IN_SCOPE:
        if entry["model_type"] not in ("qwen2", "qwen3", "gemma2", "lfm2") and entry["vocab"] != 130560:
            continue
        got = {v: is_full(dispatch(monkeypatch, entry, data="small", rows="long", verdict=v))
               for v in ("clean", "damaged", "unknown")}
        assert len(set(got.values())) == 1, (entry["model"], got)


# ---- historical R1 coverage ------------------------------------------------------------------------------------------
# 34 past R1 instruct draws (07-13 .. 09-28), from the review session's route-floor audit
# (scratchpad lfm0928/wf/route_floor_audit.py, sources EXEC / RECOMP / EST / CODE per row there). Row length p50 None =
# not recorded: both row lengths are evaluated and must agree unless the model is LFM2.5-2.6B.
HISTORICAL_R1 = [
    ("5131a786", "0713", "unsloth/SmolLM2-1.7B", None), ("75742018", "0713", "Qwen/Qwen2.5-1.5B", 73),
    ("8ef92b0f", "0713", "EleutherAI/pythia-410m", None), ("97520f56", "0720", "unsloth/tinyllama-chat", None),
    ("037e2970", "0720", "Qwen/Qwen2.5-0.5B", 698), ("f644ed68", "0720", "Qwen/Qwen3-0.6B-Base", 345),
    ("f6782145", "0727", "zake7749/gemma-2-2b-it-chinese-kyara-dpo", 147),
    ("ce763cd5", "0727", "Qwen/Qwen2.5-3B-Instruct", 1004), ("e1d461c8", "0727", "Qwen/Qwen2-1.5B-Instruct", 3400),
    ("27840c46", "0803", "TinyLlama/TinyLlama_v1.1", None), ("0bf165c5", "0803", "unsloth/SmolLM2-1.7B", None),
    ("865ff4ab", "0803", "unsloth/SmolLM-135M-Instruct", None), ("6d8b2af6", "0810", "Qwen/Qwen2-1.5B-Instruct", 239),
    ("1f57709d", "0810", "TinyLlama/TinyLlama-1.1B-Chat-v0.6", None),
    ("2065add7", "0810", "Qwen/Qwen2.5-Math-1.5B-Instruct", 900), ("45f9a179", "0817", "LiquidAI/LFM2.5-2.6B", 856),
    ("3edc57af", "0817", "unsloth/tinyllama", None), ("05266233", "0817", "microsoft/phi-1_5", None),
    ("28c9f3bb", "0824", "unsloth/Llama-3.2-1B", 120), ("3a6658eb", "0824", "Qwen/Qwen2-1.5B-Instruct", 1982),
    ("10b5cdcc", "0824", "Qwen/Qwen3-4B-Base", 493), ("d99ce208", "0831", "tiiuae/falcon-rw-1b", None),
    ("bc347f11", "0831", "Qwen/Qwen2.5-3B-Instruct", 302), ("597dc05e", "0831", "bigscience/bloomz-560m", None),
    ("59469903", "0907", "unsloth/gemma-2-2b", 731), ("29fc1a10", "0907", "LiquidAI/LFM2.5-2.6B", 155),
    ("ef9824ae", "0907", "unsloth/SmolLM2-360M", None), ("90d361cb", "0914", "LiquidAI/LFM2.5-2.6B", 614),
    ("067761fe", "0914", "unsloth/Llama-3.2-1B-Instruct", 425),
    ("87c51c39", "0914", "Qwen/Qwen2.5-0.5B-Instruct", 206), ("522e0294", "0921", "Qwen/Qwen2.5-3B", 449),
    ("6bffc021", "0921", "Qwen/Qwen2-0.5B", 262), ("cc4550ab", "0921", "LiquidAI/LFM2.5-1.2B-Instruct", 178),
    ("803a8f69", "0928", "LiquidAI/LFM2.5-1.2B-Instruct", 89),
]


def historical_routes(monkeypatch) -> list[tuple[str, str, str, bool]]:
    by_name = {e["model"]: e for e in POOL}
    out = []
    for tid, date, model, p50 in HISTORICAL_R1:
        entry = by_name[model]
        lengths = ["long" if p50 >= 256 else "short"] if p50 is not None else ["short", "long"]
        fulls = {is_full(dispatch(monkeypatch, entry, data="medium", rows=rl, verdict="unknown")) for rl in lengths}
        assert len(fulls) == 1, (tid, model)
        out.append((tid, date, model, fulls.pop()))
    return out


def test_historical_r1_coverage(monkeypatch):
    routes = historical_routes(monkeypatch)
    assert len(routes) == 34
    misses = sorted((tid, model) for tid, _d, model, full in routes if not full)
    assert misses == [("d99ce208", "tiiuae/falcon-rw-1b")]  # the one remaining quality exception
    assert sum(full for *_, full in routes) == 33


def test_gate_catches_adapters_under_cab930c(monkeypatch):
    """Negative control: the same harness under FORGE_V2_FULLFT_DEFAULT=0 (cab930c) finds in-scope adapter cells, so the
    release gate above is able to fail."""
    adapters = []
    for model in ("Qwen/Qwen2.5-3B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "numind/NuExtract-v1.5",
                  "unsloth/gemma-2-2b", "Qwen/Qwen3.5-4B"):
        entry = next(e for e in IN_SCOPE if e["model"] == model)
        r = dispatch(monkeypatch, entry, data="medium", rows="short", verdict="clean", switch="0")
        if not is_full(r):
            adapters.append(model)
    assert adapters == ["Qwen/Qwen2.5-3B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "numind/NuExtract-v1.5",
                        "unsloth/gemma-2-2b", "Qwen/Qwen3.5-4B"]
