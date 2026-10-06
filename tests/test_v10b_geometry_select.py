"""v10b: the in-run geometry selector (forge.tasks.geometry_select; FORGE_V2_GEOMETRY_SELECT, default ON).

(a) static gate and first-run refusals, (b) the pick rule and its margin, (c) both time gates on numbers copied from the
recorded receipts (research/20261006-release-candidate-v10b/REPLAY-TIME-GATE.json), incl. the 1.3x slower node and the
indic-like run that needs 68 to 88 % of the window, (d) the alternate geometry reproduces the phase-bu S arms, (e) the real
sft_v2.run as run A and as run S on CPU (the trainer is a recorder), (f) the orchestrator: a second run that wins, loses,
is refused, crashes, runs out of memory or is interrupted, the base reload after the first model is released (weakref),
telemetry, and (g) the dispatcher wiring in instruct.run. All CPU. Nothing here is GPU evidence."""
from __future__ import annotations

import copy
import gc
import sys
import time
import weakref
from types import SimpleNamespace as NS

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from forge.clock import Deadline
from forge.data import packing
from forge.tasks import geometry_select as gs
from forge.tasks import instruct, sft_v2


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith("FORGE_"):
            monkeypatch.delenv(name)
    torch.set_num_threads(2)


# ---- (a) static gate and first-run refusals ---------------------------------------------------------------------------

def test_constants_are_the_predeclared_values():
    assert (gs.ENV, gs.DEFAULT, gs.TARGET_ROWS, gs.VALUES) == ("FORGE_V2_GEOMETRY_SELECT", "1", 32, (8, 96))
    assert (gs.MARGIN, gs.SLOW_FACTOR, gs.DEADLINE_MARGIN_S, gs.MIN_DEV_ROWS) == (0.005, 1.3, 60.0, 200)
    assert gs.MIN_REMAINING_S == sft_v2.CHEAP_FULL_MIN_REMAINING_S == 900.0


def test_armed_on_the_full_weight_field_route_at_any_size_and_family():
    assert gs.static_refusal(field_route=True, strategy="full") is None
    assert gs.armed(field_route=True, strategy="full", rung="full")


@pytest.mark.parametrize("kw,reason", [
    (dict(field_route=False, strategy="full"), "not_field_full_weights"),
    (dict(field_route=True, strategy="lora"), "not_field_full_weights"),
    (dict(field_route=True, strategy=None), "not_field_full_weights"),
    (dict(field_route=True, strategy="full", rung="cheap_full"), "cheap_full_rung"),
])
def test_not_armed(kw, reason):
    assert gs.static_refusal(**kw) == reason


@pytest.mark.parametrize("env,value,reason", [
    ("FORGE_V2_GEOMETRY_SELECT", "0", "switch_off"), ("FORGE_V2_FIELD_EFF_BATCH", "32", "explicit_batch_env"),
    ("FORGE_V2_EFF_BATCH", "64", "explicit_batch_env"), ("FORGE_V2_FIELD_PACKED_EFF_BATCH", "7", "explicit_batch_env"),
])
def test_switch_and_research_batch_arms_disarm_it(monkeypatch, env, value, reason):
    monkeypatch.setenv(env, value)
    assert gs.static_refusal(field_route=True, strategy="full") == reason


@pytest.mark.parametrize("env,value", [("FORGE_V2_GEOMETRY_SELECT", "1"), ("FORGE_V2_GEOMETRY_SELECT", ""),
                                       ("FORGE_V2_FIELD_EFF_BATCH", "0"), ("FORGE_V2_FIELD_EFF_BATCH", "")])
def test_neutral_env_values_leave_it_armed(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert gs.armed(field_route=True, strategy="full")


def first_ok(**over):
    """Run A of Qwen2.5-0.5B-I indic, seed 11, H100 (receipt numbers)."""
    d = dict(completed=True, saved=True, field=True, strategy="full", cheap_full=False, gemma2_champion=False,
             large_model_rule=False, field_hygiene=False, lr_search=False, oom_retries=0, window_bound=False, dev_n=400,
             chosen_loss=1.3437661168049089, dev_fingerprint="f" * 64, rows_per_step=204, core_s=1152.4, tail_s=19.9,
             finish_reserve_s=230.8, epochs_done=3.0, eval_s=108.4, eval_count=25, soup_s=47.6, steps_per_epoch=137)
    d.update(over)
    return d


def test_a_normal_first_run_goes_to_the_time_gate():
    assert gs.first_run_refusal(first_ok()) is None


@pytest.mark.parametrize("over,reason", [
    (dict(completed=False), "first_run_incomplete"), (dict(saved=False), "first_artifact_not_saved"),
    (dict(field=False), "not_field_full_weights"), (dict(strategy="lora"), "not_field_full_weights"),
    (dict(cheap_full=True), "cheap_full_rung"), (dict(gemma2_champion=True), "gemma2_champion_geometry"),
    (dict(large_model_rule=True), "large_model_batch_rule"), (dict(field_hygiene=True), "field_hygiene_experiment"),
    (dict(lr_search=True), "field_lr_search_experiment"), (dict(oom_retries=1), "first_run_oom_retry"),
    (dict(window_bound=True), "first_run_window_bound"), (dict(dev_n=199), "internal_dev_too_small"),
    (dict(dev_n=0), "internal_dev_too_small"), (dict(chosen_loss=float("inf")), "no_internal_dev_loss"),
    (dict(chosen_loss=float("nan")), "no_internal_dev_loss"), (dict(chosen_loss=None), "no_internal_dev_loss"),
    (dict(dev_fingerprint=None), "no_dev_fingerprint"), (dict(rows_per_step=47), "first_run_already_small_batch"),
    (dict(core_s=None), "first_run_time_unmeasured"), (dict(tail_s=None), "first_run_time_unmeasured"),
])
def test_first_run_refusals(over, reason):
    assert gs.first_run_refusal(first_ok(**over)) == reason


def test_an_empty_first_record_is_refused():
    assert gs.first_run_refusal({}) == "first_run_incomplete"


# ---- (b) the pick ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a,s,choice", [
    (2.5584, 2.4601, "S"),        # SmolLM-135M s11 (-3.84 %)
    (2.3878, 2.3106, "S"),        # llama-160m e81e65f2 s11
    (1.8985, 1.8508, "S"),        # Qwen2-0.5B trdpo s11
    (1.3438, 1.4001, "A"),        # Qwen2.5-0.5B-I indic s11 (+4.19 %): the blow-up cell keeps A
    (1.3405, 1.4113, "A"),        # indic s7
    (1.2259, 1.2466, "A"),        # LFM2.5-1.2B hko1aw s11
    (1.0, 1.0, "A"), (1.0, 0.9951, "A"), (1.0, 0.995, "S"), (1.0, 0.99, "S"), (1.0, 1.2, "A"),
])
def test_pick_needs_half_a_percent(a, s, choice):
    assert gs.pick(a, s)[0] == choice


@pytest.mark.parametrize("a,s,reason", [
    (float("nan"), 1.0, "first_dev_not_finite"), (None, 1.0, "first_dev_not_finite"), (0.0, 1.0, "first_dev_not_finite"),
    (1.0, float("nan"), "second_dev_not_finite"), (1.0, float("inf"), "second_dev_not_finite"), (1.0, None, "second_dev_not_finite"),
    (1.0, 0.0, "second_dev_not_finite"),
])
def test_pick_keeps_a_on_any_unreadable_loss(a, s, reason):
    assert gs.pick(a, s) == ("A", reason)


def test_dev_fingerprint_depends_on_rows_and_order():
    rows = [dict(input_ids=[1, 2, 3]), dict(input_ids=[4, 5])]
    fp = gs.dev_fingerprint(rows)
    assert len(fp) == 64 and fp == gs.dev_fingerprint(copy.deepcopy(rows))
    assert fp != gs.dev_fingerprint(rows[::-1]) and fp != gs.dev_fingerprint(rows[:1])
    assert fp != gs.dev_fingerprint([dict(input_ids=[1, 2]), dict(input_ids=[3, 4, 5])])   # same tokens, other row cut


# ---- (c) the time gates on receipt numbers -----------------------------------------------------------------------------

def test_stage1_formula():
    g = gs.time_gate_stage1(first_ok(), remaining_s=3600 - 1185.6, load_s=13.3)
    # load 13.3 + 1.3 x 1152.4 + max(1.3 x 19.9, 230.8) + 60 against (remaining - 60)
    assert g["fits"] and g["need_s"] == pytest.approx(13.3 + 1.3 * 1152.4 + 230.8 + 60, abs=0.2)
    assert g["budget_s"] == pytest.approx(3600 - 1185.6 - 60, abs=0.2) and g["slow_factor"] == 1.3


def test_indic_like_run_fits_at_measured_speed_and_is_refused_on_a_node_1_3x_slower():
    """Run A used 33 % of the hour; A + S + tail is 68 % of it, 88 % on a node 1.3x slower (REPORT.md 1.3)."""
    assert gs.time_gate_stage1(first_ok(), remaining_s=3600 - 1185.6, load_s=13.3)["fits"]
    slow = first_ok(core_s=1152.4 * 1.3, tail_s=19.9 * 1.3, eval_s=108.4 * 1.3, soup_s=47.6 * 1.3)
    g = gs.time_gate_stage1(slow, remaining_s=3600 - 1185.6 * 1.3, load_s=13.3 * 1.3)
    assert not g["fits"] and g["need_s"] > g["budget_s"]


def test_stage2_uses_the_second_runs_measured_step_time():
    remaining = 3600 - 1185.6 - 60 - 13.3 - (50.4 - 13.5)          # at run S's first optimizer step
    ok = gs.time_gate_stage2(first_ok(), t_step_s=0.400834959, steps_per_epoch=875, finish_reserve_s=241.6, remaining_s=remaining)
    assert ok["fits"] and ok["updates"] == 2625 and ok["epochs"] == 3.0
    assert ok["need_s"] == pytest.approx(1.3 * (0.400834959 * 2625 + ok["evals"] * 108.4 / 25 + 47.6) + 241.6 + 60, abs=0.5)
    # seed 7's probe measured 0.565 s per step on the same cell: the same gate refuses (and run A stays)
    no = gs.time_gate_stage2(first_ok(core_s=1143.8, eval_s=107.4, soup_s=40.7), t_step_s=0.565103008, steps_per_epoch=880,
                             finish_reserve_s=267.9, remaining_s=3600 - 1176.8 - 60 - 13.1 - (50.2 - 13.3))
    assert not no["fits"] and no["need_s"] > no["budget_s"]


@pytest.mark.parametrize("t_step", [None, 0.0, float("nan"), -1.0])
def test_stage2_refuses_without_a_step_time(t_step):
    g = gs.time_gate_stage2(first_ok(), t_step_s=t_step, steps_per_epoch=875, finish_reserve_s=241.6, remaining_s=3000.0)
    assert g == {"stage": 2, "fits": False, "reason": "no_step_time", "remaining_s": 3000.0}


def test_stage2_plans_at_least_one_epoch_and_the_tail_is_never_below_the_finish_reserve():
    g = gs.time_gate_stage2(first_ok(epochs_done=0.4, tail_s=400.0), t_step_s=1.0, steps_per_epoch=100, finish_reserve_s=150.0,
                            remaining_s=10 ** 6)
    assert g["updates"] == 100 and g["tail_s"] == 520.0
    assert gs.tail_seconds(first_ok(tail_s=1.0), 150.0) == 150.0


def test_a_window_bound_task_never_gets_a_second_run():
    """LFM2.5-1.2B Turkish: run A ends at 2,523 s of 2,700 s."""
    turkish = first_ok(core_s=2507.2, tail_s=3.7, finish_reserve_s=188.8, epochs_done=1.947, rows_per_step=116)
    g = gs.time_gate_stage1(turkish, remaining_s=2700 - 2523.5, load_s=12.6)
    assert not g["fits"] and g["remaining_s"] < gs.MIN_REMAINING_S


def test_less_than_fifteen_minutes_left_is_refused_whatever_the_prediction():
    tiny = first_ok(core_s=10.0, tail_s=1.0, finish_reserve_s=10.0)
    assert not gs.time_gate_stage1(tiny, remaining_s=899.0, load_s=1.0)["fits"]
    assert gs.time_gate_stage1(tiny, remaining_s=900.0, load_s=1.0)["fits"]


def test_unmeasured_load_time_uses_the_default():
    assert gs.time_gate_stage1(first_ok(), remaining_s=3000.0, load_s=None)["load_s"] == gs.LOAD_S_DEFAULT


@pytest.mark.parametrize("spe,total,evals", [(17, 140, 14), (35, 252, 26), (875, 2625, 25), (138, 414, 25)])
def test_planned_evals_is_the_runs_eval_rule(spe, total, evals):
    assert gs.planned_evals(spe, total) == evals


# ---- (d) the alternate geometry = the phase-bu S arms ------------------------------------------------------------------

def plan(mean_rows_per_block):
    if mean_rows_per_block is None:
        return None
    blocks = 10 ** 9
    stats = {"rows": int(round(mean_rows_per_block * blocks)), "blocks": blocks}
    return packing.PackingPlan("sdpa", "sdpa", 0, [], None, stats, attn_implementation="sdpa")


# cell -> shape (phase-bu GEOMETRY-CPU.json) and the screen's S arm (knob value, micro, accum, rows per step)
BU = {
    "llama160m-e81e65f2": (dict(params_b=0.162417792, unit_len=256, vocab=32000, rpb=None, recipe_bf16=False), (32, 32, 1, 32)),
    "smollm135-short": (dict(params_b=0.1345, unit_len=832, vocab=49152, rpb=None, recipe_bf16=False), (32, 19, 2, 38)),
    "q25-05bi-indic s7": (dict(params_b=0.494, unit_len=3072, vocab=151936, rpb=6.337352138307552, recipe_bf16=True), (26, 5, 1, 32)),
    "q25-05bi-indic s11": (dict(params_b=0.494, unit_len=3072, vocab=151936, rpb=6.370683741138806, recipe_bf16=True), (26, 5, 1, 32)),
    "q2-05b-trdpo s7": (dict(params_b=0.494, unit_len=1600, vocab=151936, rpb=4.8980263157894735, recipe_bf16=True), (32, 7, 1, 34)),
    "q2-05b-trdpo s11": (dict(params_b=0.494, unit_len=1600, vocab=151936, rpb=4.9114898295766904, recipe_bf16=True), (32, 7, 1, 34)),
}


@pytest.mark.parametrize("cell", sorted(BU))
def test_alternate_geometry_reproduces_the_phase_bu_s_arm(monkeypatch, cell):
    monkeypatch.setattr(sft_v2, "_bnb_available", lambda: True)
    shape, s_arm = BU[cell]
    alt = gs.alternate_value(params_b=shape["params_b"], unit_len=shape["unit_len"], vocab=shape["vocab"], per_gpu_gb=80.0,
                             bnb_ok=True, layers=32, hidden=2048, liger=False, pack=plan(shape["rpb"]),
                             recipe_bf16=shape["recipe_bf16"])
    assert (alt["value"], alt["micro"], alt["grad_accum"], alt["rows_per_step"]) == s_arm


def test_alternate_geometry_works_above_one_billion_params_too():
    alt = gs.alternate_value(params_b=1.17, unit_len=256, vocab=64400, per_gpu_gb=80.0, bnb_ok=True, layers=16, hidden=2048,
                             liger=False, pack=None, recipe_bf16=False)
    assert alt["rows_per_step"] == 32 and alt["value"] == 32


# ---- (e) the real handler as run A and as run S ------------------------------------------------------------------------

def tiny_llama():
    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                                        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)).eval()


def rows(n):
    out = []
    for i in range(n):
        ids = [1] + [5 + i % 50] * (3 + i % 5) + [100 + i, 17, 2]      # every row distinct (exact_dedup)
        out.append(dict(input_ids=ids, labels=ids.copy()))
    return out


class Harness:
    """sft_v2.run on CPU with the real geometry code; the trainer is a recorder that completes its plan in 3 steps."""

    def __init__(self, monkeypatch, tmp_path, *, params_b=0.162417792, train_error=None):
        from forge.data.schema import InstructColumns
        from forge.tuning import selection

        self.source, self.events, self.trainers, self.saves, self.meta = rows(600), [], [], [], {}
        self.dev_loss = {"value": 1.0}
        self.dev_sizes: list[int] = []
        self.train_error = train_error
        self.params_b = params_b
        h = self
        for name in ("FORGE_V2_FIELD_LR_CALIBRATION", "FORGE_V2_FIELD_SWEEP", "FORGE_V2_FIELD_BF16", "FORGE_V2_ROW_LOSS",
                     "FORGE_V2_FIELD_HYGIENE", "FORGE_DEV_PASS"):
            monkeypatch.setenv(name, "0")
        monkeypatch.setattr(sft_v2, "_liger_ok", lambda m: False)
        monkeypatch.setattr(sft_v2, "_bnb_available", lambda: False)
        monkeypatch.setattr(sft_v2, "workdir", lambda spec: str(tmp_path / "work"))
        monkeypatch.setattr(sft_v2, "_layer_weight_rms", lambda m: 1.0)
        monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda *a: [])
        monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct", lambda *a: copy.deepcopy(h.source))
        monkeypatch.setattr(sft_v2, "adaptive_max_len", lambda *a, **k: 32)
        # like the real rule, the DEV size depends on the remaining hours: 300 rows above 0.9 h, else 200
        monkeypatch.setattr(sft_v2, "dev_size_for", lambda n, *, hours, **k: 300 if hours > 0.9 else 200)

        def split(r, dev_n, *a, **k):
            h.dev_sizes.append(dev_n)
            return r[:-dev_n], r[-dev_n:]

        monkeypatch.setattr(sft_v2, "stratified_split", split)
        monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **d: h.events.append((n, d)))
        monkeypatch.setattr(sft_v2.telemetry, "set_meta", lambda **kw: h.meta.update(kw))
        for name in ("eval_point", "write_into"):
            monkeypatch.setattr(sft_v2.telemetry, name, lambda *a, **k: None)
        monkeypatch.setattr(sft_v2.telemetry, "make_trainer_callback", lambda *a: object())
        monkeypatch.setattr(sft_v2, "write_artifact_truth", lambda *a, **k: None)
        monkeypatch.setattr(sft_v2, "save_adapter", lambda m, tok, out, **k: h.saves.append(k.get("truth_reason")))
        monkeypatch.setattr(selection, "per_example_dev_loss", lambda model, rs, collator, **kw: (h.dev_loss["value"], []))
        monkeypatch.setattr(selection, "available_ram", lambda: 10 ** 12)
        monkeypatch.setattr(sft_v2.lr_probe, "measure_step_time", lambda model, batches, **kw: (0.01, [1.0]))

        class Args:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        monkeypatch.setattr(sft_v2, "compatible_dataclass_kwargs", lambda cls, kw, **_: kw)
        monkeypatch.setattr(sys.modules["transformers"], "TrainingArguments", Args)

        class RecordingTrainer:
            def __init__(self, model, args, train_dataset, data_collator, callbacks=()):
                self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
                self.callbacks = callbacks
                self.state = NS(global_step=0, max_steps=3)
                self.optimizer = object()
                self.lr_scheduler = object()
                # no reference to the model is kept by the harness
                h.trainers.append({"micro": args.per_device_train_batch_size, "accum": args.gradient_accumulation_steps,
                                   "lr": args.learning_rate, "trained": False})

            def get_train_dataloader(self):
                return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, collate_fn=self.data_collator)

            def train(self):
                if h.train_error is not None:
                    raise h.train_error
                h.trainers[-1]["trained"] = True
                self.state.global_step = 3
                self.callbacks[0].on_step_end(self.args, self.state, NS(should_training_stop=False), model=self.model)

        monkeypatch.setattr(sys.modules["transformers"], "Trainer", RecordingTrainer)
        self.spec = NS(output_dir=str(tmp_path / "out"), task_id="synthetic", instruct=InstructColumns("instruction", "output"),
                       cached_model_dir="/cache/models/synthetic")
        self.kwargs = dict(baseline_summary=None, params_b=params_b, per_gpu_gb=80.0, strategy_override="full", field_route=True)

    def ev(self, name):
        return [d for n, d in self.events if n == name]

    def deadline(self, hours=1.0):
        return Deadline.from_hours(hours, started_monotonic=time.monotonic(), export_reserve_s=180.0)

    def run(self, *, deadline=None, model=None, **kw):
        loaded = NS(model=model if model is not None else tiny_llama())
        sft_v2.run(self.spec, deadline or self.deadline(), self.source, loaded, NS(pad_token_id=0), **self.kwargs, **kw)
        assert loaded.model is None


def test_run_a_records_what_the_selector_needs_and_trains_as_before(monkeypatch, tmp_path):
    plain = Harness(monkeypatch, tmp_path / "plain")
    plain.run()
    h = Harness(monkeypatch, tmp_path / "armed")
    first: dict = {}
    h.run(result=first)
    # same geometry, LR, trainer arguments and saves as the call without a result dict (= 9504136)
    assert h.trainers == plain.trainers and h.saves == plain.saves and h.meta == plain.meta
    strip = lambda evs: [(n, {k: v for k, v in d.items() if not k.endswith("_s") and k not in ("eval_s", "available_ram")
                              and not k.startswith("cgroup")}) for n, d in evs if n != "sft_v2_dev_fingerprint"]  # noqa: E731
    assert [n for n, _ in strip(h.events)] == [n for n, _ in strip(plain.events)]
    assert plain.ev("sft_v2_dev_fingerprint") == [] and len(h.ev("sft_v2_dev_fingerprint")) == 1
    assert first["completed"] and first["saved"] and first["field"] and first["strategy"] == "full"
    assert first["dev_n"] == 300 and first["dev_fingerprint"] == gs.dev_fingerprint(h.source[-300:])
    assert first["rows_per_step"] == 192 and (first["micro"], first["grad_accum"]) == (64, 3)       # the 140 bucket here
    assert first["chosen_loss"] == 1.0 and first["steps"] == 3 and not first["window_bound"]
    assert first["core_s"] > 0 and first["tail_s"] >= 0 and first["finish_reserve_s"] > 0 and first["eval_count"] >= 1
    assert not (first["cheap_full"] or first["gemma2_champion"] or first["large_model_rule"] or first["lr_search"])
    assert gs.first_run_refusal(first) is None


def second(h, first, log=None):
    return gs.SecondRun(first, log=log or (lambda n, **d: h.events.append((n, d))))


def run_a(h):
    first: dict = {}
    h.run(result=first)
    return first


def test_run_s_trains_at_about_32_rows_on_the_same_dev_rows_and_saves_only_after_it_has_won(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    a_lr = h.ev("sft_v2_field_lr")[0]
    saves_a = list(h.saves)
    h.dev_loss["value"] = 0.9                                         # run S: 10 % lower internal DEV
    sel, res = second(h, first), {}
    h.run(deadline=h.deadline(0.5), geometry_select=sel, result=res)  # half the hours left: dev_size_for would say 200
    assert h.dev_sizes == [300, 300]                                  # ... but run S keeps run A's DEV size
    fps = h.ev("sft_v2_dev_fingerprint")
    assert fps[0]["sha256"] == fps[1]["sha256"] and fps[1]["second_run"] is True
    assert h.ev("sft_v2_geometry_select_alternate")[0]["rows_per_step"] == 32
    assert (h.trainers[-1]["micro"], h.trainers[-1]["accum"]) == (32, 1) and h.meta["eff_batch"] == 32
    s_lr = h.ev("sft_v2_field_lr")[1]
    assert s_lr["rows_per_step"] == 32 and (s_lr["prior_lr"], s_lr["raw"], s_lr["mult"]) == (a_lr["prior_lr"], a_lr["raw"], a_lr["mult"])
    assert h.ev("sft_v2_geometry_select_time_gate")[0]["fits"] is True
    cmp_ = h.ev("sft_v2_geometry_select_compare")[0]
    assert (cmp_["first_dev"], cmp_["second_dev"], cmp_["margin"]) == (1.0, 0.9, 0.005)
    assert sel.record["pick"] == "S" and sel.defer_persist is False
    assert h.saves == saves_a + ["sft_v2_final"]                      # exactly one save by run S: its final one
    assert res["completed"] and res["saved"] and res["rows_per_step"] == 32


@pytest.mark.parametrize("s_loss", [1.0, 0.996, 1.05, float("nan")])
def test_run_s_that_does_not_win_writes_nothing(monkeypatch, tmp_path, s_loss):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    saves_a = list(h.saves)
    h.dev_loss["value"] = s_loss
    sel, res = second(h, first), {}
    with pytest.raises(gs.SecondRunRejected):
        h.run(geometry_select=sel, result=res)
    assert h.saves == saves_a and sel.record["pick"] == "A" and sel.defer_persist is True
    assert res["completed"] is False and h.ev("sft_v2_dev_pass_done") == []
    assert len(h.ev("sft_v2_final")) == 1                             # run A's only


def test_stage2_refusal_stops_run_s_before_its_first_optimizer_step(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    n = len(h.trainers)
    with pytest.raises(gs.SecondRunAbort, match="time_gate_stage2"):
        h.run(geometry_select=second(h, dict(first, tail_s=10 ** 4)), result={})
    assert not any(t["trained"] for t in h.trainers[n:])
    assert h.ev("sft_v2_geometry_select_time_gate")[-1]["fits"] is False


def test_run_s_without_a_timing_probe_is_stopped_before_training(monkeypatch, tmp_path):
    """With 72 s left the run takes no timing probe: no measured step time, no second training."""
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    n = len(h.trainers)
    with pytest.raises(gs.SecondRunAbort, match="no_step_time"):
        h.run(deadline=h.deadline(0.02), geometry_select=second(h, first), result={})
    assert not any(t["trained"] for t in h.trainers[n:])


def test_a_different_dev_split_stops_run_s_before_training(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    n = len(h.trainers)
    with pytest.raises(gs.SecondRunAbort, match="dev_fingerprint_mismatch"):
        h.run(geometry_select=second(h, dict(first, dev_fingerprint="0" * 64)), result={})
    assert not any(t["trained"] for t in h.trainers[n:])


def test_run_s_with_about_the_same_rows_per_step_is_stopped(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    with pytest.raises(gs.SecondRunAbort, match="same_geometry"):
        h.run(geometry_select=second(h, dict(first, rows_per_step=47)), result={})


def test_run_s_refuses_a_non_field_call(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    h.kwargs["field_route"] = False
    with pytest.raises(gs.SecondRunAbort, match="not_field_full_weights"):
        h.run(geometry_select=second(h, first), result={})


# ---- (f) the orchestrator ----------------------------------------------------------------------------------------------

class Tele:
    def __init__(self):
        self.meta, self.written, self.restored, self.snaps = {}, [], [], 0

    def set_meta(self, **kw):
        self.meta.update(kw)

    def write_into(self, out):
        self.written.append(out)

    def snapshot(self):
        self.snaps += 1
        return {"meta": dict(self.meta)}

    def restore(self, snap):
        self.restored.append(snap)


def orchestrate(h, first, *, deadline=None, run_fn=None, load_base=None, present=True, load_s=1.0, tele=None):
    loads = []

    def default_load(path, **kw):
        loads.append(path)
        return NS(model=tiny_llama(), tokenizer=NS(pad_token_id=0))

    tele = tele or Tele()
    rec = gs.maybe_second_run(h.spec, deadline or h.deadline(), h.source, first=first, run_kwargs=h.kwargs,
                              run_fn=run_fn or sft_v2.run, load_base=load_base or default_load, free=lambda: None,
                              artifact_present=lambda spec: present, load_s=load_s,
                              log=lambda n, **d: h.events.append((n, d)), telemetry=tele)
    return rec, loads, tele


def test_second_run_wins_and_replaces_the_artifact(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    h.dev_loss["value"] = 0.95
    rec, loads, tele = orchestrate(h, first)
    assert loads == ["/cache/models/synthetic"] and rec["ran"] and rec["pick"] == "S" and rec["reason"] == "second_kept"
    assert (rec["first_dev"], rec["second_dev"], rec["margin"], rec["internal_rel_pct"]) == (1.0, 0.95, 0.005, -5.0)
    assert rec["time_gate_stage1"]["fits"] and rec["time_gate_stage2"]["fits"] and rec["alternate"]["rows_per_step"] == 32
    assert rec["second_rows_per_step"] == 32 and rec["first_rows_per_step"] == 192 and rec["dev_fingerprint"] == first["dev_fingerprint"]
    assert h.saves[-1] == "sft_v2_final" and h.saves.count("sft_v2_final") == 2
    assert tele.meta[gs.META_KEY]["pick"] == "S" and tele.restored == [] and tele.written[-1] == h.spec.output_dir
    assert h.ev("sft_v2_geometry_select")[-1]["pick"] == "S" and rec["seconds"] >= 0


def test_second_run_loses_and_the_first_artifact_stays(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    saves_a = list(h.saves)
    h.dev_loss["value"] = 1.04                                        # the indic-like outcome: S is 4 % worse
    rec, loads, tele = orchestrate(h, first)
    assert rec["ran"] and rec["pick"] == "A" and rec["reason"] == "second_not_lower_by_margin"
    assert rec["second_dev"] == 1.04 and rec["internal_rel_pct"] == 4.0 and h.saves == saves_a
    assert tele.snaps == 1 and len(tele.restored) == 1 and tele.meta[gs.META_KEY]["pick"] == "A"


@pytest.mark.parametrize("error", [RuntimeError("boom"), RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
                                   MemoryError("host"), TimeoutError("clock"), KeyboardInterrupt(), SystemExit(1),
                                   sft_v2.ZeroTrainedStepsError("no step")])
def test_any_failure_in_the_second_run_ships_the_first(monkeypatch, tmp_path, error):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    saves_a = list(h.saves)
    h.train_error = error
    rec, loads, tele = orchestrate(h, first)                          # must not raise
    assert rec["ran"] and rec["pick"] == "A" and rec["artifact_present"] is True and h.saves == saves_a
    if isinstance(error, RuntimeError) and "out of memory" in str(error):
        assert rec["reason"] in ("second_run_failed", "second_not_lower_by_margin", "second_dev_not_finite")   # the OOM ladder may retry
    else:
        assert rec["reason"] == "second_run_failed" and type(error).__name__ in rec["error"]
    assert len(tele.restored) == 1


def test_a_failing_base_reload_ships_the_first(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)

    def broken(path, **kw):
        raise OSError("cache gone")

    rec, _, _ = orchestrate(h, first, load_base=broken)
    assert rec["pick"] == "A" and rec["reason"] == "second_run_failed" and "OSError" in rec["error"]


def test_a_failing_final_save_of_the_second_run_keeps_the_first(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    h.dev_loss["value"] = 0.9

    def failing_save(m, tok, out, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sft_v2, "save_adapter", failing_save)
    monkeypatch.setattr(sft_v2, "trained_artifact_present", lambda spec: True)   # run A's artifact is at the output path
    rec, _, _ = orchestrate(h, first)
    assert rec["pick"] == "A" and rec["reason"] == "second_save_failed"


@pytest.mark.parametrize("over,reason", [(dict(window_bound=True), "first_run_window_bound"),
                                         (dict(large_model_rule=True), "large_model_batch_rule"),
                                         (dict(saved=False), "first_artifact_not_saved")])
def test_a_refused_first_run_never_reloads_the_base(monkeypatch, tmp_path, over, reason):
    h = Harness(monkeypatch, tmp_path)
    first = dict(run_a(h), **over)
    n = len(h.trainers)
    rec, loads, tele = orchestrate(h, first)
    assert loads == [] and len(h.trainers) == n and tele.snaps == 0
    assert (rec["ran"], rec["skipped"], rec["pick"], rec["reason"]) == (False, True, "A", reason)
    assert h.ev("sft_v2_geometry_select")[-1]["reason"] == reason and tele.meta[gs.META_KEY]["reason"] == reason


def test_a_missing_first_artifact_never_starts_a_second_run(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    rec, loads, _ = orchestrate(h, run_a(h), present=False)
    assert loads == [] and rec["reason"] == "first_artifact_missing"


def test_stage1_refusal_is_identical_to_no_selector_after_run_a(monkeypatch, tmp_path):
    """When the gate refuses, nothing is loaded, trained or saved after run A: the task ends exactly as before."""
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    before = (list(h.trainers), list(h.saves), dict(h.meta))
    rec, loads, tele = orchestrate(h, first, deadline=h.deadline(0.2))            # 12 minutes left: below the 15-minute floor
    assert rec["reason"] == "time_gate_stage1" and rec["time_gate_stage1"]["fits"] is False and not rec["ran"]
    assert loads == [] and (h.trainers, h.saves, h.meta) == before and tele.restored == []


def test_a_broken_gate_keeps_the_first(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    monkeypatch.setattr(gs, "time_gate_stage1", lambda *a, **k: 1 / 0)
    rec, loads, _ = orchestrate(h, first)
    assert loads == [] and rec["reason"] == "gate_error" and rec["pick"] == "A"


def test_an_error_anywhere_in_the_selector_keeps_the_first_and_does_not_raise(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)

    def broken_free():
        raise RuntimeError("cuda gone")

    rec = gs.maybe_second_run(h.spec, h.deadline(), h.source, first=first, run_kwargs=h.kwargs, run_fn=sft_v2.run,
                              load_base=lambda *a, **k: 1 / 0, free=broken_free, artifact_present=lambda spec: True, load_s=1.0,
                              log=lambda n, **d: h.events.append((n, d)), telemetry=Tele())
    assert rec["pick"] == "A" and rec["reason"] == "selector_error" and "cuda gone" in rec["error"]
    assert h.ev("sft_v2_geometry_select")[-1]["reason"] == "selector_error"
    assert gs.maybe_second_run(h.spec, h.deadline(), h.source)["reason"] == "selector_error"       # even a bad call


def test_the_second_run_is_planned_against_an_earlier_hard_stop(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    real = h.deadline()
    seen = {}

    def fake_run(spec, deadline, rows_, loaded, tokenizer, **kw):
        seen.update(hard_stop=deadline.hard_stop, reserve=deadline.export_reserve_s, kw=kw)
        raise gs.SecondRunRejected("second_not_lower_by_margin")

    rec, _, _ = orchestrate(h, first, deadline=real, run_fn=fake_run)
    assert seen["hard_stop"] == real.hard_stop - 60.0 and seen["reserve"] == real.export_reserve_s
    assert isinstance(seen["kw"]["geometry_select"], gs.SecondRun) and seen["kw"]["result"] == {}
    assert {k: seen["kw"][k] for k in h.kwargs} == h.kwargs and rec["pick"] == "A"


def test_the_first_model_is_released_before_the_base_is_loaded_again(monkeypatch, tmp_path):
    """Reference lifetime (the RESP-120-C1 lesson): two bases must never be resident together."""
    h = Harness(monkeypatch, tmp_path)
    model_a = tiny_llama()
    ref_a = weakref.ref(model_a)
    first: dict = {}
    h.run(model=model_a, result=first)
    del model_a
    assert gs.first_run_refusal(first) is None
    alive_at_reload, second_refs = [], []

    def load(path, **kw):
        alive_at_reload.append(ref_a() is not None)
        m = tiny_llama()
        second_refs.append(weakref.ref(m))
        return NS(model=m, tokenizer=NS(pad_token_id=0))

    h.dev_loss["value"] = 1.2
    rec, _, _ = orchestrate(h, first, load_base=load)
    assert rec["ran"] and alive_at_reload == [False]
    gc.collect()
    assert second_refs[0]() is None                                   # and the losing second model is gone afterwards


def test_the_first_model_is_released_even_when_it_sits_in_a_reference_cycle(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)

    class Model:
        pass

    def cyclic_first_run():
        m = Model()
        m.me = m                                                      # only the cycle collector frees it
        return weakref.ref(m)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        ref = cyclic_first_run()
        assert ref() is not None
        alive = []

        def load(path, **kw):
            alive.append(ref() is not None)
            raise OSError("stop here")

        orchestrate(h, first, load_base=load)
    finally:
        if was_enabled:
            gc.enable()
    assert alive == [False]


def test_real_telemetry_snapshot_and_restore():
    from forge import telemetry

    saved = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in telemetry._data.items() if k in ("meta", "train_curve", "eval_curve", "events")}
    try:
        telemetry._data["meta"] = {"batch": 64}
        telemetry._data["train_curve"] = [(1.0, 1, 2.0, 1e-4)]
        telemetry._data["eval_curve"] = [(1.0, 1, 2.0)]
        snap = telemetry.snapshot()
        telemetry.set_meta(batch=32)
        telemetry.train_point(2, 1.0, 1e-4)
        telemetry.eval_point(2, 1.5)
        n_events = len(telemetry._data["events"])
        telemetry.restore(snap)
        assert telemetry._data["meta"]["batch"] == 64 and len(telemetry._data["train_curve"]) == 1
        assert len(telemetry._data["eval_curve"]) == 1 and len(telemetry._data["events"]) == n_events
        telemetry.restore({})                                         # a failed snapshot is a no-op
        assert telemetry._data["meta"]["batch"] == 64
    finally:
        telemetry._data.update(saved)


# ---- (g) the dispatcher wiring -----------------------------------------------------------------------------------------

def _route_table():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("route_table_v10b", Path(__file__).resolve().parent / "test_fullft_default_route_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wiring(monkeypatch, *, model="unsloth/Llama-3.2-1B", outcomes=("ok",), env=None):
    rt = _route_table()
    entry = next(e for e in rt.POOL if e["model"] == model)
    script, v2_calls, selector_calls = list(outcomes), [], []

    def v2(**kw):
        v2_calls.append(kw)
        out = script.pop(0)
        if out != "ok":
            raise out
        if kw.get("result") is not None:
            kw["result"].update(completed=True, marker=len(v2_calls))

    def fake_selector(spec, deadline, rows_, **kw):
        selector_calls.append(kw)
        return {}

    monkeypatch.setattr(sft_v2, "trained_artifact_present", lambda spec: False)

    class Stop(BaseException):
        pass

    def legacy(**kw):
        raise Stop()

    monkeypatch.setattr(gs, "maybe_second_run", fake_selector)
    events: list = []
    try:
        rt.dispatch(monkeypatch, entry, data="medium", rows="short", verdict="clean", v2_handler=v2, legacy_handler=legacy,
                    events_out=events, env=env or {})
    except Stop:
        pass
    return v2_calls, selector_calls, events


def test_dispatcher_calls_the_selector_after_a_successful_full_weight_field_run(monkeypatch):
    v2_calls, sel, _ = wiring(monkeypatch)
    assert len(v2_calls) == 1 and len(sel) == 1
    assert v2_calls[0]["field_route"] is True and v2_calls[0]["result"] is sel[0]["first"] and "geometry_select" not in v2_calls[0]
    assert sel[0]["first"] == {"completed": True, "marker": 1}
    assert sel[0]["run_fn"] is sft_v2.run and sel[0]["load_base"] is instruct.load_base
    assert sel[0]["artifact_present"] is sft_v2.trained_artifact_present and sel[0]["load_s"] >= 0
    assert sel[0]["run_kwargs"] == {k: v for k, v in v2_calls[0].items() if k != "result"}
    assert sel[0]["run_kwargs"]["cheap_full"] is False


def test_dispatcher_with_the_switch_off_is_the_9504136_call(monkeypatch):
    v2_calls, sel, _ = wiring(monkeypatch, env={"FORGE_V2_GEOMETRY_SELECT": "0"})
    assert len(v2_calls) == 1 and sel == [] and v2_calls[0]["result"] is None and "geometry_select" not in v2_calls[0]


def test_dispatcher_does_not_arm_a_research_batch_arm(monkeypatch):
    v2_calls, sel, _ = wiring(monkeypatch, env={"FORGE_V2_FIELD_EFF_BATCH": "32"})
    assert len(v2_calls) == 1 and sel == [] and v2_calls[0]["result"] is None


def test_dispatcher_does_not_arm_the_cheap_full_rung(monkeypatch):
    v2_calls, sel, _ = wiring(monkeypatch, outcomes=(RuntimeError("first attempt"), "ok"))
    assert [c["cheap_full"] for c in v2_calls] == [False, True] and sel == []
    assert v2_calls[0]["result"] == {} and v2_calls[1]["result"] is None


def test_dispatcher_keeps_the_ladder_when_the_first_run_fails(monkeypatch):
    v2_calls, sel, events = wiring(monkeypatch, outcomes=(RuntimeError("a"), RuntimeError("b")))
    assert len(v2_calls) == 2 and sel == []
    assert [n for n, _ in events if n in ("sft_v2_fallback_to_cheap_full", "sft_v2_fallback_to_lora")] == [
        "sft_v2_fallback_to_cheap_full", "sft_v2_fallback_to_lora"]
