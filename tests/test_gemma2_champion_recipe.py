"""Damaged Gemma2 champion recipe (research candidate 2026-09-26, branch research-gemma-champion-port-20260926).

A run the dispatcher put on the damaged-base FIELD route with model_type gemma2, vocabulary 256000 and 2.0 < params_b <= 3.0
(gemma-2-2b, 2.614 B) trains with the public champion's EXECUTED recipe on replay 59469903
(research/20260923-replay-bench/phase-u2/results-extra/pu2-r-gemma2-ctxsplit-champ/EXECUTED-RECIPE.md): row-isolated
packing, 48 x 2048 / pack_len blocks per optimizer step (32 at 3072), the fixed peak LR 2.9757061485479407e-05 and the
1.3-epoch planner floor; the shared pieces (bf16 + paged 8-bit AdamW, cosine to 0.25, NEFTune alpha 1, warmup 3%) are
asserted unchanged. FORGE_V2_GEMMA2_CHAMPION_RECIPE=0 restores the incumbent route; every other family / verdict / size is
unchanged. The harness only uses sft_v2.run arguments that also exist on 52d319d, so this file runs on the cherry-pick too.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch
import transformers

from forge.data import packing
from forge.tasks import sft_v2
from forge.tuning import lr_probe

SWITCH = "FORGE_V2_GEMMA2_CHAMPION_RECIPE"
GEMMA_VOCAB = 256000
QWEN_VOCAB = 151936
PARAMS_B = 2.614341888  # gemma-2-2b (CONTRACT-PRECHECK num_params 2,614,341,888)
EXECUTED_LR = 2.9757061485479407e-05  # W&B learning_rate of the champion run on 59469903
qualified = pytest.mark.skipif(transformers.__version__ != packing.QUALIFIED_TRANSFORMERS,
                               reason="packing is qualified on the image's transformers only")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    torch.set_num_threads(2)


# ---- constants and the executed values --------------------------------------------------------------------------

def test_constants_pin_the_executed_recipe():
    assert sft_v2.GEMMA2_CHAMPION_RECIPE_DEFAULT == "1"
    assert sft_v2.GEMMA2_CHAMPION_PEAK_LR == EXECUTED_LR
    assert sft_v2.GEMMA2_CHAMPION_MIN_EPOCHS == 1.3
    assert (sft_v2.GEMMA2_CHAMPION_BUCKET_BLOCKS, sft_v2.GEMMA2_CHAMPION_BUCKET_LEN) == (48, 2048)
    assert sft_v2.GEMMA2_CHAMPION_MICRO_TOKENS == 36864
    assert (sft_v2.GEMMA2_CHAMPION_MIN_PARAMS_B, sft_v2.GEMMA2_CHAMPION_MAX_PARAMS_B) == (2.0, 3.0)
    assert sft_v2.GEMMA2_CHAMPION_VOCAB == GEMMA_VOCAB
    assert packing.GEMMA2_ROUTE_ATTR == "_forge_gemma2_champion_packing"
    # pieces the incumbent FIELD route already shares with the executed recipe (unchanged by this candidate)
    assert sft_v2._min_lr_rate(0.25) == 0.25  # cosine_with_min_lr, min_lr_rate 0.25
    assert sft_v2._neftune_alpha(None, "full") == 1.0  # NEFTune alpha 1
    assert sft_v2.field_eff_batch(PARAMS_B) == 48  # the 2_4_b bucket the block rule scales
    assert sft_v2.field_recipe("gemma2", PARAMS_B, damaged_base=True) == {
        "bf16": True, "bf16_min_b": 2.0, "lr_mult": 1.0, "eff_batch": 48, "strategy": "full"}
    fam = sft_v2.DAMAGED_BASE_FAMILIES["gemma2"]
    assert fam["enabled"] and fam["vocabs"] == frozenset({GEMMA_VOCAB}) and fam["tested_params_b"] == (2.61, 2.61)


@pytest.mark.parametrize("pack_len,blocks", [(3072, 32), (3136, 31), (4096, 24), (2049, 47), (2048, 48), (1024, 48)])
def test_block_rule_is_the_champion_bucket_rule(pack_len, blocks):
    # instruct_config.get_training_json: batch_size 48 (2_4_b) -> max(1, int(48 * 2048 / max_length)) above 2048
    assert sft_v2.gemma2_champion_blocks(pack_len) == blocks


@pytest.mark.parametrize("pack_len,blocks,expected", [(3072, 32, (8, 4)), (3136, 31, (8, 4)), (4096, 24, (8, 3)),
                                                      (2048, 48, (16, 3)), (1024, 48, (24, 2)), (36865, 1, (1, 1))])
def test_micro_accum_keeps_the_block_count(pack_len, blocks, expected):
    micro, accum = sft_v2.gemma2_champion_micro_accum(pack_len, blocks)
    assert (micro, accum) == expected
    assert micro * pack_len <= max(pack_len, sft_v2.GEMMA2_CHAMPION_MICRO_TOKENS)
    assert abs(micro * accum - blocks) <= 1


def test_executed_geometry_on_the_replay_cell():
    # executed: 32 blocks x 3072 per step. Our max_len on the same TRAIN is 3136 (Phase I sft_v2_data), where the rule
    # gives 31 and the micro/accum split lands on 32 blocks again (8 x 4)
    assert sft_v2.gemma2_champion_blocks(3072) == 32
    assert sft_v2.gemma2_champion_micro_accum(3072, 32) == (8, 4)
    assert sft_v2.gemma2_champion_micro_accum(3136, sft_v2.gemma2_champion_blocks(3136)) == (8, 4)


# ---- admission --------------------------------------------------------------------------------------------------

def _admits(mt="gemma2", params_b=PARAMS_B, vocab=GEMMA_VOCAB, damaged=True):
    return sft_v2.gemma2_champion_admits(mt, params_b, vocab, damaged_base_route=damaged)


def test_admission_default_and_switch(monkeypatch):
    assert _admits() is True
    assert _admits(mt="Gemma2") is True
    monkeypatch.setenv(SWITCH, "0")
    assert _admits() is False
    monkeypatch.setenv(SWITCH, "1")
    assert _admits() is True
    monkeypatch.setattr(sft_v2, "GEMMA2_CHAMPION_RECIPE_DEFAULT", "0")
    monkeypatch.delenv(SWITCH)
    assert _admits() is False  # the source default decides when the env is unset (the validator's case)


@pytest.mark.parametrize("params_b,ok", [(2.0, False), (2.000000001, True), (2.614, True), (3.0, True),
                                         (3.000000001, False), (9.241705984, False), (27.2, False), (1.5, False)])
def test_admission_size_class(params_b, ok):
    assert _admits(params_b=params_b) is ok


@pytest.mark.parametrize("mt,vocab", [("qwen2", QWEN_VOCAB), ("qwen3", QWEN_VOCAB), ("llama", 128256), ("lfm2", 128000),
                                      ("gemma", GEMMA_VOCAB), ("gemma3", 262144), ("gemma3_text", 262144),
                                      ("gemma2", 256128), ("gemma2", None), (None, GEMMA_VOCAB)])
def test_admission_other_families_and_vocabularies(mt, vocab):
    assert _admits(mt=mt, vocab=vocab) is False


def test_admission_needs_the_damaged_base_route():
    # clean / unknown gemma2 never reach it (the dispatcher's damaged_base_route flag is False for them)
    assert _admits(damaged=False) is False


@pytest.mark.parametrize("verdict", ["clean", "unknown", None])
def test_dispatcher_flag_is_false_for_clean_and_unknown_gemma2(verdict):
    gate = dict(params_b=PARAMS_B, n_gpus=1, is_kl=False, model_type="gemma2", vocab_size=GEMMA_VOCAB,
                n_rows=26980, total_tokens=24_056_622)
    assert sft_v2.field_full_route(**gate, base_damage="damaged") is True
    assert sft_v2.damaged_base_admits("gemma2", "damaged", PARAMS_B, GEMMA_VOCAB) is True
    assert sft_v2.damaged_base_admits("gemma2", verdict, PARAMS_B, GEMMA_VOCAB) is False
    assert sft_v2.field_full_route(**gate, base_damage=verdict) is False  # clean / unknown gemma2: the adapter route


# ---- packing: route-scoped admission and row isolation for Gemma2 -----------------------------------------------

def _tiny_gemma2(vocab=GEMMA_VOCAB, sliding_window=4096, flag=True):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(17)
    m = Gemma2ForCausalLM(Gemma2Config(
        vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=8192, sliding_window=sliding_window,
        bos_token_id=2, eos_token_id=1, pad_token_id=0, use_cache=False, attn_implementation="sdpa")).eval()
    if flag:
        setattr(m, packing.GEMMA2_ROUTE_ATTR, True)
    return m


def _tiny(cls_name, vocab):
    import transformers as tf

    torch.manual_seed(17)
    cfg = getattr(tf, cls_name + "Config")(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                           num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                           bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False,
                                           attn_implementation="sdpa")
    return getattr(tf, cls_name + "ForCausalLM")(cfg).eval()


def _rows(n=8):
    result = []
    for i in range(n):
        ids = [2] + [5 + i] * (3 + i % 5) + [17, 1]
        result.append(dict(input_ids=ids, labels=([-100, -100] + ids[2:]) if i % 2 else ids.copy()))
    return result


@qualified
def test_gemma2_packing_is_admitted_only_with_the_route_flag(monkeypatch):
    monkeypatch.setenv(packing.ENV_MODE, "sdpa")
    assert packing._eligibility(_tiny_gemma2(flag=False), field_route=True, field_hygiene=False) == "not_native_llama"
    assert packing._eligibility(_tiny_gemma2(), field_route=True, field_hygiene=False) is None
    assert packing._eligibility(_tiny_gemma2(vocab=1000), field_route=True,
                                field_hygiene=False) == "llama_scope_unqualified"
    m = _tiny_gemma2()
    setattr(m, packing.GEMMA2_ROUTE_ATTR, "yes")  # only the exact True the route sets
    assert packing._eligibility(m, field_route=True, field_hygiene=False) == "not_native_llama"
    # the flag never changes another family's decision
    for name, vocab, want in (("Llama", 128256, None), ("Qwen2", QWEN_VOCAB, None), ("Qwen3", QWEN_VOCAB, "not_native_llama")):
        m = _tiny(name, vocab)
        before = packing._eligibility(m, field_route=True, field_hygiene=False)
        setattr(m, packing.GEMMA2_ROUTE_ATTR, True)
        assert packing._eligibility(m, field_route=True, field_hygiene=False) == before == want
    # the other gates still apply to the flagged Gemma2
    m = _tiny_gemma2()
    assert packing._eligibility(m, field_route=False, field_hygiene=False) == "not_field_route"
    assert packing._eligibility(m, field_route=True, field_hygiene=True) == "hygiene_not_supported"
    m.config.use_cache = True
    assert packing._eligibility(m, field_route=True, field_hygiene=False) == "cache_must_be_disabled"
    eager = _tiny_gemma2()
    eager.config._attn_implementation = "eager"
    for layer in eager.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    assert packing.plan_field_packing(eager, _rows(), max_len=32, pad_token_id=0, model_cap=256, field_route=True) is None


@qualified
@pytest.mark.parametrize("sliding_window", [4096, 4])  # 4: the sliding layers' window is shorter than the rows
def test_sdpa_gemma2_packing_matches_unpacked_token_losses_and_isolates_rows(sliding_window):
    m, source = _tiny_gemma2(sliding_window=sliding_window), _rows()
    p = packing.plan_field_packing(m, source, max_len=32, pad_token_id=0, model_cap=256, field_route=True)
    assert p is not None and p.mode == "sdpa" and p.stats["rows"] == len(source)
    blocks = packing.pack_rows(source, 32, seed=7)
    assert any(len(b.rows) > 1 for b in blocks)

    def ce(logits, labels):
        labels = torch.as_tensor(labels)[1:]
        losses = torch.nn.functional.cross_entropy(logits[:-1].float(), labels, reduction="none", ignore_index=-100)
        return losses[labels != -100]

    with torch.no_grad():
        expected = [ce(m(input_ids=torch.tensor([r["input_ids"]]), use_cache=False).logits[0], r["labels"]) for r in source]
        batch = p.collator(p.features)
        assert "attention_mask" not in batch  # sdpa: isolation comes from the reset position ids
        out = m(**batch)
        got = torch.cat([ce(out.logits[k], batch["labels"][k]) for k in range(len(blocks))])
        # label[0] of each packed row is ignored; unpacked, the shifted targets never include it either
        ref = torch.cat([expected[i] for b in blocks for i in b.rows])
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
        # mutate every token of the first row of a multi-row block: the later rows' logits stay fixed
        k = next(i for i, b in enumerate(blocks) if len(b.rows) > 1)
        boundary = len(source[blocks[k].rows[0]]["input_ids"])
        changed = {kk: v.clone() for kk, v in batch.items()}
        changed["input_ids"][k, :boundary] = 31
        later = m(**changed).logits
        torch.testing.assert_close(later[k, boundary:blocks[k].n_tokens], out.logits[k, boundary:blocks[k].n_tokens],
                                   atol=1e-6, rtol=1e-6)
        # the same mutation leaks with a plain causal mask (naive packing): the isolation above is real
        naive = packing.PackedPadCollator(0, attention_mask=True)(p.features)
        first = m(**naive).logits
        naive["input_ids"][k, :boundary] = 31
        second = m(**naive).logits
        assert float((first[k, boundary:blocks[k].n_tokens] - second[k, boundary:blocks[k].n_tokens]).abs().max()) > 1e-5


# ---- sft_v2.run on CPU: the real route, geometry, LR and planner code; training / timing / export are doubles ------

def _long_rows(n=420):
    # ~2.7 rows per 3072-token block, as on 59469903 (p50 758, mean ~890 tokens)
    out = []
    for i in range(n):
        length = 700 + (i * 37) % 700
        ids = [2] + [5 + i % 50] * (length - 2) + [1]
        labels = [-100] * (length // 2) + ids[length // 2:]
        out.append(dict(input_ids=ids, labels=labels))
    return out


def _run(monkeypatch, tmp_path, *, model=None, params_b=PARAMS_B, damaged=True, env=None, t_step=1000.0, rms=0.02,
         max_len=3072):
    """sft_v2.run with a tiny model; returns (events, trainers). Only arguments that exist on 52d319d are passed."""
    from forge.data.schema import InstructColumns
    from forge.tasks import fallback
    from forge.tuning import selection

    if transformers.__version__ != packing.QUALIFIED_TRANSFORMERS:
        pytest.skip("packing is qualified on the image's transformers only")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("FORGE_DEV_PASS", "0")
    events, trainers = [], []
    source = _long_rows()
    monkeypatch.setattr(sft_v2, "_liger_ok", lambda m: False)
    monkeypatch.setattr(sft_v2, "_bnb_available", lambda: True)
    monkeypatch.setattr(sft_v2, "workdir", lambda spec: str(tmp_path / "work"))
    monkeypatch.setattr(sft_v2, "_layer_weight_rms", lambda m: rms)
    monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda *a: [])
    monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct", lambda *a: copy.deepcopy(source))
    monkeypatch.setattr(sft_v2, "adaptive_max_len", lambda *a, **k: max_len)
    monkeypatch.setattr(sft_v2, "dev_size_for", lambda *a, **k: 2)
    monkeypatch.setattr(sft_v2, "stratified_split", lambda r, *a, **k: (r[:-2], r[-2:]))
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **d: events.append((n, d)))
    for name in ("set_meta", "eval_point", "write_into"):
        monkeypatch.setattr(sft_v2.telemetry, name, lambda *a, **k: None)
    monkeypatch.setattr(sft_v2.telemetry, "make_trainer_callback", lambda *a: object())
    monkeypatch.setattr(sft_v2, "write_artifact_truth", lambda *a, **k: None)
    monkeypatch.setattr(sft_v2, "save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(fallback, "emit_untrained_copy", lambda spec: None)
    monkeypatch.setattr(selection, "per_example_dev_loss", lambda model, rs, collator, **kw: (0.5, []))
    monkeypatch.setattr(selection, "available_ram", lambda: 10 ** 12)
    monkeypatch.setattr(sft_v2.lr_probe, "measure_step_time", lambda model, batches, **kw: (t_step, [1.0]))

    class Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setattr(sft_v2, "compatible_dataclass_kwargs", lambda cls, kw, **_: kw)
    monkeypatch.setattr(sys.modules["transformers"], "TrainingArguments", Args)

    class RecordingTrainer:
        def __init__(self, model, args, train_dataset, data_collator, callbacks=()):
            self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
            self.callbacks = callbacks
            self.state = NS(global_step=0, max_steps=100)
            self.optimizer = object()
            self.lr_scheduler = object()
            trainers.append(self)

        def get_train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, collate_fn=self.data_collator)

        def train(self):
            self.state.global_step = 1
            self.callbacks[0].on_step_end(self.args, self.state, NS(should_training_stop=False), model=self.model)

    monkeypatch.setattr(sys.modules["transformers"], "Trainer", RecordingTrainer)
    spec = NS(output_dir=str(tmp_path / "out"), task_id="synthetic", instruct=InstructColumns("instruction", "output"))
    deadline = NS(remaining_hard=lambda: 2700.0, export_reserve_s=180.0)
    sft_v2.run(spec, deadline, source, NS(model=model if model is not None else _tiny_gemma2(flag=False)),
               NS(pad_token_id=0), baseline_summary=None, params_b=params_b, per_gpu_gb=80.0, strategy_override="full",
               field_route=True, damaged_base_route=damaged)
    return events, trainers


def _ev(events, name):
    return [d for n, d in events if n == name]


def _stable(events):
    """Events without wall-clock readings (elapsed seconds differ between two identical runs)."""
    return [(n, {k: v for k, v in d.items() if k != "elapsed_s"}) for n, d in events]


def _trace(events, trainers):
    plan0 = _ev(events, "sft_v2_plan")[0]
    a = trainers[-1].args
    return {"field_lr": _ev(events, "sft_v2_field_lr"), "override": _ev(events, "sft_v2_lr_override"),
            "sweep_gate_lr": _ev(events, "sft_v2_sweep_gate")[0]["lr_override"],
            "train_start_lr": _ev(events, "sft_v2_train_start")[0]["lr"], "plan": plan0,
            "trainer_lr": a.learning_rate, "trainer_warmup": a.warmup_steps, "trainer_epochs": a.num_train_epochs,
            "micro": a.per_device_train_batch_size, "accum": a.gradient_accumulation_steps,
            "sched": (a.lr_scheduler_type, a.lr_scheduler_kwargs), "neftune": a.neftune_noise_alpha,
            "wd": a.weight_decay, "clip": a.max_grad_norm, "packing": _ev(events, "sft_v2_packing"),
            "skipped": _ev(events, "sft_v2_packing_skipped")}


def _analytic(rms, eff):
    return max(2.0e-5, min(1.0e-4, lr_probe.analytic_lr(weight_rms=rms, params_b=PARAMS_B, eff_batch=eff,
                                                          gradient_noise_scale=None)))


def test_run_default_is_the_executed_recipe(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path)
    t = _trace(ev, tr)
    route = _ev(ev, "sft_v2_gemma2_champion_route")
    assert len(route) == 1 and _ev(ev, "sft_v2_gemma2_champion_skipped") == []
    r = route[0]
    pk = t["packing"][0]
    assert pk["mode"] == "sdpa" and pk["pack_len"] == 3072 and t["skipped"] == []
    # pack-speed candidate (research-gemma-pack-speed-20260927): the 32 blocks run as one micro-batch (c292732 planned 8 x 4,
    # still reported as planned_*; FORGE_V2_GEMMA2_PACK_SPEED=0 restores it, tests/test_gemma2_pack_speed.py)
    assert (r["pack_len"], r["target_blocks"], r["micro"], r["grad_accum"], r["blocks_per_step"]) == (3072, 32, 32, 1, 32)
    assert (r["planned_micro"], r["planned_grad_accum"]) == (8, 4)
    assert r["env_geometry"] == [] and r["blocks"] == pk["blocks"] and r["mean_rows_per_block"] == pytest.approx(
        pk["rows"] / pk["blocks"], abs=1e-3)
    assert 2.0 < r["mean_rows_per_block"] < 3.5  # the replay cell's ~2.7 rows per block
    assert r["steps_per_epoch"] == math.ceil(pk["blocks"] / 32)
    assert r["lr"] == EXECUTED_LR and r["lr_source"] == "champion_fixed" and r["min_epochs"] == 1.3
    assert (t["micro"], t["accum"]) == (32, 1)
    # the fixed LR reaches every LR point (no clamp, multiplier, sweep or sft_v2_lr_override research event)
    assert t["override"] == []
    assert t["sweep_gate_lr"] == t["train_start_lr"] == t["plan"]["lr"] == t["trainer_lr"] == EXECUTED_LR
    assert r["prior_lr_before"] == t["field_lr"][0]["prior_lr"]  # the analytic prior it replaced, still reported
    # planner floor: 1.25 x achievable is below 1.3 epochs here, so the horizon is the floor (the executed 390 / 300)
    spe = r["steps_per_epoch"]
    achievable = t["plan"]["window_s"] * 0.85 / 1000.0
    assert 1.25 * achievable / spe < 1.3
    assert t["plan"]["epochs"] == t["trainer_epochs"] == 1.3
    assert t["plan"]["total_steps"] == math.ceil(spe * 1.3)
    # shared with the incumbent and equal to the executed recipe
    assert t["trainer_warmup"] == t["plan"]["warmup"] == min(200, max(3, int(0.03 * t["plan"]["total_steps"])))
    assert t["sched"] == ("cosine_with_min_lr", {"min_lr_rate": 0.25})
    assert (t["neftune"], t["wd"], t["clip"]) == (1.0, 0.0, 1.0)
    assert _ev(ev, "sft_v2_field_bf16")[0]["optim"] == "paged_adamw_8bit"


def test_run_executed_horizon_and_warmup_at_the_executed_scale():
    # at the executed 300 steps / epoch the floor gives 390 steps and our warmup rule gives the executed 11
    total = int(math.ceil(300 * sft_v2.GEMMA2_CHAMPION_MIN_EPOCHS))
    assert total == 390 and min(200, max(3, int(0.03 * total))) == 11


def test_run_floor_does_not_shorten_a_longer_plan(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path, t_step=1.0)  # 1.25 x achievable far above 1.3 epochs: the epoch cap as before
    t = _trace(ev, tr)
    assert t["plan"]["epochs"] == 4.0  # max_epochs_total: 4 below 10k training rows (418 here), 3 above
    assert t["trainer_lr"] == EXECUTED_LR


def test_off_switch_restores_the_incumbent_route(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path / "off", env={SWITCH: "0"})
    t = _trace(ev, tr)
    assert _ev(ev, "sft_v2_gemma2_champion_route") == [] == _ev(ev, "sft_v2_gemma2_champion_skipped")
    assert t["packing"] == [] and t["skipped"][0]["reason"] == "not_native_llama"
    assert not hasattr(tr[-1].model, packing.GEMMA2_ROUTE_ATTR)
    prior = _analytic(0.02, 48)
    assert t["field_lr"][0]["prior_lr"] == prior
    assert t["sweep_gate_lr"] == t["train_start_lr"] == t["trainer_lr"] == prior
    spe = max(1, 418 // 48)  # unpacked rows at the 48-row bucket
    assert (t["micro"], t["accum"]) == sft_v2.field_bf16_micro_accum(unit_len=3072, vocab=GEMMA_VOCAB, eff_default=48)
    achievable = t["plan"]["window_s"] * 0.85 / 1000.0
    assert t["plan"]["epochs"] == round(min(3.0, 1.25 * achievable / spe), 3) < 1.3
    # identical to a run with the source default off (the validator's no-env case): the switch is the only hook
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(sft_v2, "GEMMA2_CHAMPION_RECIPE_DEFAULT", "0")
    ev_src, tr_src = _run(monkeypatch, tmp_path / "src")
    assert _stable(ev_src) == _stable(ev)
    assert _trace(ev_src, tr_src) == t


def test_packing_unavailable_skips_the_whole_recipe(monkeypatch, tmp_path):
    ev_off, tr_off = _run(monkeypatch, tmp_path / "packoff", env={"FORGE_V2_FIELD_PACKING": "off"})
    assert _ev(ev_off, "sft_v2_gemma2_champion_skipped") == [{"reason": "packing_unavailable"}]
    assert _ev(ev_off, "sft_v2_gemma2_champion_route") == []
    ev_inc, tr_inc = _run(monkeypatch, tmp_path / "inc", env={"FORGE_V2_FIELD_PACKING": "off", SWITCH: "0"})
    strip = [(n, d) for n, d in ev_off if not n.startswith("sft_v2_gemma2_champion")]
    assert _stable(strip) == _stable(ev_inc)  # incumbent LR, geometry and planner
    assert _trace(ev_off, tr_off) == _trace(ev_inc, tr_inc)


def test_env_lr_and_env_geometry_keep_precedence(monkeypatch, tmp_path):
    ev, tr = _run(monkeypatch, tmp_path / "lr", env={"FORGE_V2_LR": "4e-05"})
    r = _ev(ev, "sft_v2_gemma2_champion_route")[0]
    assert r["lr"] == 4e-05 and r["lr_source"] == "env" and _trace(ev, tr)["trainer_lr"] == 4e-05
    assert _ev(ev, "sft_v2_lr_override") == [{"prior_lr": _ev(ev, "sft_v2_field_lr")[0]["prior_lr"], "override": 4e-05}]
    monkeypatch.delenv("FORGE_V2_LR")
    ev, tr = _run(monkeypatch, tmp_path / "geo", env={packing.ENV_EFF_BLOCKS: "12"})
    r = _ev(ev, "sft_v2_gemma2_champion_route")[0]
    assert r["env_geometry"] == [packing.ENV_EFF_BLOCKS] and r["target_blocks"] == 32
    assert r["blocks_per_step"] == _trace(ev, tr)["micro"] * _trace(ev, tr)["accum"] == 12
    assert r["lr"] == EXECUTED_LR and _ev(ev, "sft_v2_plan")[0]["epochs"] == 1.3


# ---- out of scope: unchanged runs --------------------------------------------------------------------------------

def _same_with_switch(monkeypatch, tmp_path, **kw):
    ev_on, tr_on = _run(monkeypatch, tmp_path / "on", **kw)
    ev_off, tr_off = _run(monkeypatch, tmp_path / "off", env={SWITCH: "0"}, **kw)
    assert _stable(ev_on) == _stable(ev_off)
    assert _trace(ev_on, tr_on) == _trace(ev_off, tr_off)
    assert _ev(ev_on, "sft_v2_gemma2_champion_route") == [] == _ev(ev_on, "sft_v2_gemma2_champion_skipped")
    return ev_on


def test_run_non_damaged_gemma2_is_unchanged(monkeypatch, tmp_path):
    ev = _same_with_switch(monkeypatch, tmp_path, damaged=False)
    assert _ev(ev, "sft_v2_packing_skipped")[0]["reason"] == "not_native_llama"


@pytest.mark.parametrize("params_b", [1.9, 3.2])
def test_run_other_gemma2_sizes_are_unchanged(monkeypatch, tmp_path, params_b):
    _same_with_switch(monkeypatch, tmp_path, params_b=params_b)


def test_run_foreign_vocabulary_gemma2_is_unchanged(monkeypatch, tmp_path):
    _same_with_switch(monkeypatch, tmp_path, model=_tiny_gemma2(vocab=1000, flag=False))


@pytest.mark.parametrize("name,vocab", [("Qwen2", QWEN_VOCAB), ("Llama", 128256), ("Qwen3", QWEN_VOCAB)])
def test_run_other_damaged_families_are_unchanged(monkeypatch, tmp_path, name, vocab):
    ev = _same_with_switch(monkeypatch, tmp_path, model=_tiny(name, vocab), params_b=2.5)
    assert not any(n.startswith("sft_v2_gemma2") for n, _ in ev)
