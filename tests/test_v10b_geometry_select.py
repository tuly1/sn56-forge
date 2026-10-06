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
    # the label mask decides which tokens the loss counts: same ids with another mask is another DEV
    a = [dict(input_ids=[1, 2, 3], labels=[-100, 2, 3])]
    assert gs.dev_fingerprint(a) == gs.dev_fingerprint(copy.deepcopy(a))
    assert gs.dev_fingerprint(a) != gs.dev_fingerprint([dict(input_ids=[1, 2, 3], labels=[-100, -100, 3])])
    assert gs.dev_fingerprint(a) != gs.dev_fingerprint([dict(input_ids=[1, 2, 3])])


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


class Tok:
    pad_token_id = 0

    def save_pretrained(self, d):
        import os

        with open(os.path.join(d, "tokenizer.json"), "w") as fh:
            fh.write("{}")


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
        self.telemetry_writes, self.callback_dirs = [], []
        monkeypatch.setattr(sft_v2.telemetry, "eval_point", lambda *a, **k: None)
        monkeypatch.setattr(sft_v2.telemetry, "write_into", lambda out, *a, **k: h.telemetry_writes.append(out))
        monkeypatch.setattr(sft_v2.telemetry, "make_trainer_callback", lambda out, *a: h.callback_dirs.append(out) or object())
        monkeypatch.setattr(sft_v2, "write_artifact_truth", lambda *a, **k: None)
        from forge.tasks import common

        real_save = common.save_adapter                      # the REAL staged / validated / exchanged save, on disk

        def recording_save(m, tok, out, **k):
            h.saves.append(k.get("truth_reason"))
            return real_save(m, tok, out, **k)

        monkeypatch.setattr(sft_v2, "save_adapter", recording_save)
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
        sft_v2.run(self.spec, deadline or self.deadline(), self.source, loaded, Tok(), **self.kwargs, **kw)
        assert loaded.model is None


def test_run_a_records_what_the_selector_needs_and_trains_as_before(monkeypatch, tmp_path):
    plain = Harness(monkeypatch, tmp_path / "plain")
    plain.run()
    h = Harness(monkeypatch, tmp_path / "armed")
    first: dict = {}
    h.run(result=first)
    # same geometry, LR, trainer arguments and saves as the call without a result dict (= 9504136); not a claim of equal
    # wall-clock behaviour: anything the run decides from measured seconds can differ between two executions
    assert h.trainers == plain.trainers and h.saves == plain.saves and h.meta == plain.meta
    strip = lambda evs: [(n, {k: v for k, v in d.items() if not k.endswith("_s") and k not in ("eval_s", "available_ram")
                              and not k.startswith("cgroup")}) for n, d in evs
                         if n not in ("sft_v2_dev_fingerprint", "sft_v2_eval_governor", "sft_v2_eval_governor_slack")]  # noqa: E731
    # (the eval governor fires from measured eval / save seconds: clock-dependent, not a training difference)
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
    assert h.saves == saves_a + [sel.truth_reason]                    # exactly one save by run S: its final one, with its marker
    assert sel.truth_reason.startswith(gs.GENERATION_PREFIX) and len(sel.nonce) == 16
    assert gs.read_export(h.spec.output_dir)["reason"] == sel.truth_reason
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
        return NS(model=tiny_llama(), tokenizer=Tok())

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
    assert h.saves[-1] == rec["second_marker"] and h.saves[-1].startswith(gs.GENERATION_PREFIX) and h.saves.count("sft_v2_final") == 1
    assert rec["exported_generation"] == "S" and rec["final_export"]["reason"] == rec["second_marker"]
    assert rec["first_export"]["reason"] == "sft_v2_final" and len(rec["final_export"]["weights_sha256"]) == 64
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
    assert rec["exported_generation"] == "A" and rec["final_export"] == rec["first_export"] and rec["final_export"]["reason"] == "sft_v2_final"
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
        return NS(model=m, tokenizer=Tok())

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


# ---- (f2) the commit boundary of run S's save (Codex RESP-135-1 B1): the record follows the DISK ----------------------

def linux_exchange(monkeypatch, faults):
    """common._rename_exchange as on Linux (the two directory names swap in one step; macOS has no renameat2), with fault
    injection around the commit point. faults: {"before": exc} raises instead of exchanging; {"fsync": exc} raises from the
    first parent fsync AFTER the exchange; {"cleanup": exc} raises when the prior generation is removed after it."""
    import os

    from forge.tasks import common

    state = {"armed": False, "exchanged": False}
    real_fsync, real_rmtree = common._fsync_dir, common._rmtree

    def exchange(left, right):
        if state["armed"] and "before" in faults:
            raise faults["before"]
        os.rename(right, right + ".swap"); os.rename(left, right); os.rename(right + ".swap", left)
        state["exchanged"] = state["armed"]

    def fsync_dir(path):
        if state["exchanged"] and "fsync" in faults and not state.get("fsync_done"):
            state["fsync_done"] = True
            raise faults["fsync"]
        return real_fsync(path)

    def rmtree(path):
        if state["exchanged"] and "cleanup" in faults and path.endswith(".tmp") and not state.get("cleanup_done"):
            state["cleanup_done"] = True
            raise faults["cleanup"]
        return real_rmtree(path)

    monkeypatch.setattr(common, "_rename_exchange", exchange)
    monkeypatch.setattr(common, "_fsync_dir", fsync_dir)
    monkeypatch.setattr(common, "_rmtree", rmtree)
    return state


def io_error():
    import errno

    return OSError(errno.EIO, "injected")


def no_space():
    import errno

    return OSError(errno.ENOSPC, "injected")


def boundary(monkeypatch, tmp_path, faults, *, exchange=True):
    h = Harness(monkeypatch, tmp_path)
    state = linux_exchange(monkeypatch, faults) if exchange else None
    first = run_a(h)
    a_export = gs.read_export(h.spec.output_dir)
    assert a_export["reason"] == "sft_v2_final"
    h.dev_loss["value"] = 0.9                                         # run S wins the comparison and goes on to its save
    if state is not None:
        state["armed"] = True
    rec, _, tele = orchestrate(h, first)
    return h, rec, tele, a_export, gs.read_export(h.spec.output_dir)


@pytest.mark.parametrize("fault", [io_error, no_space, KeyboardInterrupt, lambda: SystemExit(1)])
def test_a_fault_before_the_exchange_leaves_run_a_exported_and_the_record_says_a(monkeypatch, tmp_path, fault):
    h, rec, tele, a_export, on_disk = boundary(monkeypatch, tmp_path, {"before": fault()})
    assert on_disk == a_export and rec["exported_generation"] == "A" and rec["pick"] == "A"
    assert rec["reason"] in ("second_save_failed", "second_run_failed") and len(tele.restored) == 1
    assert rec["final_export"] == a_export == rec["first_export"]


@pytest.mark.parametrize("fault", [io_error, no_space, KeyboardInterrupt, lambda: SystemExit(1)])
def test_a_fault_between_the_exchange_and_the_fsync_leaves_run_s_exported_and_the_record_says_s(monkeypatch, tmp_path, fault):
    """The RESP-135-1 reproduction: the save reports failure (or an interrupt lands) AFTER the commit point."""
    h, rec, tele, a_export, on_disk = boundary(monkeypatch, tmp_path, {"fsync": fault()})
    assert on_disk["reason"] == rec["second_marker"] and on_disk["reason"].startswith(gs.GENERATION_PREFIX)
    assert rec["exported_generation"] == "S" and rec["pick"] == "S" and rec["reason"] == "second_committed_despite_error"
    assert rec["second_outcome"] in ("second_save_failed", "second_run_failed")
    assert tele.restored == [] and tele.meta[gs.META_KEY]["pick"] == "S"         # telemetry describes the exported run S
    assert rec["final_export"] == on_disk and rec["first_export"] == a_export
    assert h.ev("sft_v2_geometry_select")[-1]["exported_generation"] == "S"


@pytest.mark.parametrize("fault", [io_error, KeyboardInterrupt])
def test_a_fault_after_the_exchange_during_cleanup_still_reports_run_s(monkeypatch, tmp_path, fault):
    h, rec, tele, a_export, on_disk = boundary(monkeypatch, tmp_path, {"cleanup": fault()})
    assert on_disk["reason"] == rec["second_marker"] and rec["pick"] == "S" and rec["exported_generation"] == "S"
    assert tele.restored == []


def test_no_fault_with_the_exchange_reports_run_s_kept(monkeypatch, tmp_path):
    h, rec, tele, a_export, on_disk = boundary(monkeypatch, tmp_path, {})
    assert rec["pick"] == "S" and rec["reason"] == "second_kept" and on_disk["reason"] == rec["second_marker"]


def test_portable_rename_path_rolls_back_before_its_commit_and_reports_what_is_on_disk(monkeypatch, tmp_path):
    """Without renameat2 (macOS; the default here): final -> .old, staged -> final. A failure of the second rename is rolled
    back by save_adapter, so run A is exported; a failure of the fsync after it leaves run S exported."""
    import os

    from forge.tasks import common

    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    a_export = gs.read_export(h.spec.output_dir)
    h.dev_loss["value"] = 0.9
    real_replace, n = os.replace, {"calls": 0}

    def replace(src, dst):
        if str(src).endswith(".tmp") and str(dst) == h.spec.output_dir:
            n["calls"] += 1
            raise io_error()
        return real_replace(src, dst)

    monkeypatch.setattr(common.os, "replace", replace)
    rec, _, tele = orchestrate(h, first)
    monkeypatch.setattr(common.os, "replace", real_replace)
    assert n["calls"] == 1 and rec["pick"] == "A" and rec["exported_generation"] == "A"
    assert gs.read_export(h.spec.output_dir) == a_export and len(tele.restored) == 1


def test_an_unreadable_export_after_the_second_run_is_reported_unknown_not_guessed(monkeypatch, tmp_path):
    import os

    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)

    def run_that_destroys_the_truth_file(spec, deadline, rows_, loaded, tokenizer, **kw):
        os.unlink(os.path.join(spec.output_dir, gs.TRUTH_FILE))
        raise RuntimeError("boom")

    rec, _, tele = orchestrate(h, first, run_fn=run_that_destroys_the_truth_file)
    assert rec["pick"] == "UNKNOWN" and rec["exported_generation"] == "UNKNOWN" and rec["reason"] == "exported_generation_unknown"
    assert tele.restored == [] and rec["final_export"]["readable"] is False


def test_changed_weights_under_run_a_truth_are_unknown_and_a_foreign_marker_is_not_s(tmp_path):
    a = {"readable": True, "truth": "COMPLETE_BEST", "optimizer_step": 3, "reason": "sft_v2_final", "weight_files": ["model.safetensors"],
         "weight_bytes": 10, "weights_sha256": "a" * 64}
    mine = gs.GENERATION_PREFIX + "0123456789abcdef"
    assert gs.generation_of(dict(a), second_reason=mine, first_export=a) == "A"
    assert gs.generation_of(dict(a, weights_sha256="b" * 64), second_reason=mine, first_export=a) == "UNKNOWN"
    assert gs.generation_of(dict(a, optimizer_step=4), second_reason=mine, first_export=a) == "UNKNOWN"
    assert gs.generation_of(dict(a, reason=mine), second_reason=mine, first_export=a) == "S"
    assert gs.generation_of(dict(a, reason=gs.GENERATION_PREFIX + "f" * 16), second_reason=mine, first_export=a) == "UNKNOWN"
    assert gs.generation_of({"readable": False}, second_reason=mine, first_export=a) == "UNKNOWN"
    assert gs.read_export(str(tmp_path / "missing"))["readable"] is False


def test_no_second_run_when_the_first_export_cannot_be_read_back(monkeypatch, tmp_path):
    import os

    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    os.unlink(os.path.join(h.spec.output_dir, gs.TRUTH_FILE))
    rec, loads, _ = orchestrate(h, first)
    assert loads == [] and rec["reason"] == "first_export_unreadable" and rec["pick"] == "A" and not rec["ran"]


# ---- (f3) every exit after the second run may have started ends in the disk read-back (Codex RESP-136-1 B1) -----------

class Raiser:
    """Raises `exc` from call number `first` on (1-based), for `times` calls; otherwise calls `real`."""

    def __init__(self, real, exc, *, first=1, times=10 ** 9, active=None):
        self.real, self.exc, self.first, self.times, self.n, self.raised = real, exc, first, times, 0, 0
        self.active = active if active is not None else {"on": True}

    def __call__(self, *a, **k):
        if self.active["on"]:
            self.n += 1
            if self.n >= self.first and self.raised < self.times:
                self.raised += 1
                raise self.exc
        return self.real(*a, **k)


def committed(monkeypatch, tmp_path, *, s_wins=True):
    """Run A, then arrange that run S wins and commits (the real save). Returns the harness and run A's result."""
    h = Harness(monkeypatch, tmp_path)
    first = run_a(h)
    h.dev_loss["value"] = 0.9 if s_wins else 1.1
    return h, first


def disk_generation(h):
    reason = str(gs.read_export(h.spec.output_dir, hash_weights=False).get("reason"))
    return "S" if reason.startswith(gs.GENERATION_PREFIX) else "A"


@pytest.mark.parametrize("exc", [KeyboardInterrupt("gc"), SystemExit(3), RuntimeError("gc"), MemoryError()])
def test_interrupt_in_the_cleanup_after_a_committed_second_run_still_reports_s(monkeypatch, tmp_path, exc):
    """Codex's reproduction: KeyboardInterrupt in the post-run gc.collect while the disk holds run S."""
    h, first = committed(monkeypatch, tmp_path)
    active = {"on": False}
    real_run = sft_v2.run

    def run_then_arm(*a, **k):
        try:
            return real_run(*a, **k)
        finally:
            active["on"] = True                                        # every gc.collect AFTER the second run raises

    monkeypatch.setattr(gs, "gc", NS(collect=Raiser(gc.collect, exc, active=active)))
    rec, _, tele = orchestrate(h, first, run_fn=run_then_arm)
    assert disk_generation(h) == "S" and rec["pick"] == "S" and rec["exported_generation"] == "S" and rec["state"] == "final"
    assert rec["reason"] == "second_kept" and type(exc).__name__ in rec["cleanup_error"] and tele.restored == []
    assert tele.meta[gs.META_KEY]["pick"] == "S"


@pytest.mark.parametrize("exc", [KeyboardInterrupt("free"), SystemExit(3), RuntimeError("cuda")])
def test_interrupt_in_free_after_a_committed_second_run_still_reports_s(monkeypatch, tmp_path, exc):
    h, first = committed(monkeypatch, tmp_path)
    calls = {"n": 0}

    def free():
        calls["n"] += 1
        if calls["n"] >= 2:                                           # the first call is before the reload
            raise exc

    tele = Tele()
    rec = gs.maybe_second_run(h.spec, h.deadline(), h.source, first=first, run_kwargs=h.kwargs, run_fn=sft_v2.run,
                              load_base=lambda path, **kw: NS(model=tiny_llama(), tokenizer=Tok()), free=free,
                              artifact_present=lambda spec: True, load_s=1.0, log=lambda n, **d: h.events.append((n, d)),
                              telemetry=tele)
    assert disk_generation(h) == "S" and rec["pick"] == "S" and rec["state"] == "final" and "cleanup_error" in rec


@pytest.mark.parametrize("where", ["log", "set_meta", "write_into"])
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("s_wins", [True, False])
def test_interrupt_in_the_final_logging_or_telemetry_does_not_change_the_pick(monkeypatch, tmp_path, where, exc_type, s_wins):
    h, first = committed(monkeypatch, tmp_path, s_wins=s_wins)
    tele = Tele()
    armed = {"on": False}
    real_run = sft_v2.run

    def run_then_arm(*a, **k):
        try:
            return real_run(*a, **k)
        finally:
            armed["on"] = True

    def log(name, **d):
        if where == "log" and armed["on"] and name == "sft_v2_geometry_select":
            raise exc_type("log")
        h.events.append((name, d))

    if where != "log":
        setattr(tele, where, Raiser(getattr(tele, where), exc_type(where), active=armed))
    rec = gs.maybe_second_run(h.spec, h.deadline(), h.source, first=first, run_kwargs=h.kwargs, run_fn=run_then_arm,
                              load_base=lambda path, **kw: NS(model=tiny_llama(), tokenizer=Tok()), free=lambda: None,
                              artifact_present=lambda spec: True, load_s=1.0, log=log, telemetry=tele)
    want = "S" if s_wins else "A"
    assert disk_generation(h) == want and rec["pick"] == want and rec["exported_generation"] == want and rec["state"] == "final"
    assert any(e.startswith({"log": "log", "set_meta": "meta", "write_into": "write"}[where]) for e in rec["record_errors"])


@pytest.mark.parametrize("s_wins", [True, False])
def test_an_interrupted_read_back_is_retried(monkeypatch, tmp_path, s_wins):
    h, first = committed(monkeypatch, tmp_path, s_wins=s_wins)
    armed = {"on": False}
    real_run = sft_v2.run

    def run_then_arm(*a, **k):
        try:
            return real_run(*a, **k)
        finally:
            armed["on"] = True

    monkeypatch.setattr(gs, "read_export", Raiser(gs.read_export, KeyboardInterrupt("read"), times=2, active=armed))
    rec, _, _ = orchestrate(h, first, run_fn=run_then_arm)
    want = "S" if s_wins else "A"
    assert rec["pick"] == want == disk_generation(h) and len(rec["finalize_interruptions"]) == 2 and rec["state"] == "final"


@pytest.mark.parametrize("s_wins", [True, False])
def test_a_read_back_that_never_completes_is_unknown_never_a(monkeypatch, tmp_path, s_wins):
    h, first = committed(monkeypatch, tmp_path, s_wins=s_wins)
    armed = {"on": False}
    real_run = sft_v2.run

    def run_then_arm(*a, **k):
        try:
            return real_run(*a, **k)
        finally:
            armed["on"] = True

    monkeypatch.setattr(gs, "read_export", Raiser(gs.read_export, KeyboardInterrupt("read"), active=armed))
    rec, _, _ = orchestrate(h, first, run_fn=run_then_arm)
    assert rec["pick"] == "UNKNOWN" and rec["exported_generation"] == "UNKNOWN" and rec["reason"] == "exported_generation_unknown"
    assert rec["second_started"] is True and len(rec["finalize_interruptions"]) == 4 and rec["state"] == "final"


def test_an_interrupt_before_the_second_run_can_start_is_a_fact_not_a_guess(monkeypatch, tmp_path):
    h, first = committed(monkeypatch, tmp_path)
    saves = list(h.saves)
    monkeypatch.setattr(gs, "time_gate_stage1", Raiser(gs.time_gate_stage1, KeyboardInterrupt("gate")))
    rec, loads, _ = orchestrate(h, first)
    assert loads == [] and h.saves == saves and rec["second_started"] is False and rec["pick"] == "A" and rec["reason"] == "selector_error"


def test_run_s_writes_no_telemetry_into_the_output_path_before_its_save(monkeypatch, tmp_path):
    """Each export directory carries a record that names its own run: while run A's directory is at the output path, run S
    writes nothing into it; the copy staged with run S's export says S; non-final records claim no pick."""
    h, first = committed(monkeypatch, tmp_path, s_wins=False)
    to_output = lambda: [w for w in h.telemetry_writes if w == h.spec.output_dir]  # noqa: E731  (staging writes go to out.tmp)
    assert h.callback_dirs == [h.spec.output_dir] and len(to_output()) == 1                          # run A, as before
    seen = []
    tele = Tele()
    real_set = tele.set_meta
    tele.set_meta = lambda **kw: (seen.append(dict(kw.get(gs.META_KEY) or {})), real_set(**kw))[1]
    rec, _, _ = orchestrate(h, first, tele=tele)
    assert rec["pick"] == "A" and h.callback_dirs[1:] and all(d != h.spec.output_dir for d in h.callback_dirs[1:])
    assert len(to_output()) == 1                                       # run S itself wrote nothing (it lost before its save)
    assert [r.get("state") for r in seen] == ["second_run_started", "final"]
    assert "pick" not in seen[0] and seen[0]["provisional_generation"] == "A" and seen[1]["pick"] == "A"


def test_the_record_staged_with_run_s_names_run_s_and_claims_no_pick(monkeypatch, tmp_path):
    h, first = committed(monkeypatch, tmp_path)
    seen = []
    tele = Tele()
    real_set = tele.set_meta
    tele.set_meta = lambda **kw: (seen.append(dict(kw.get(gs.META_KEY) or {})), real_set(**kw))[1]
    rec, _, _ = orchestrate(h, first, tele=tele)
    assert [r.get("state") for r in seen] == ["second_run_started", "second_run_saving", "final"]
    assert seen[1]["provisional_generation"] == "S" and "pick" not in seen[1] and seen[1]["second_marker"] == rec["second_marker"]
    # run S's own run() wrote nothing into the output path either: only into its staged directory, inside save_adapter
    assert rec["pick"] == "S" and [w for w in h.telemetry_writes if w == h.spec.output_dir] == [h.spec.output_dir]


# ---- property: a BaseException injected at EVERY statement boundary of the selector after the second run's save ----------

def _property_run(tmp_path, template, k, *, commit, exc_type):
    """One maybe_second_run with a fast second run (a real save_adapter commit, or a failure before it). A tracer raises
    exc_type at the k-th executed line of geometry_select.py after the save. Returns (record | None, raised, where, disk)."""
    import os
    import shutil
    import sys

    from forge.tasks import common

    root = tmp_path / f"{'c' if commit else 'n'}-{exc_type.__name__}-{k}"
    shutil.copytree(template, root)
    out = str(root / "out")
    spec = NS(output_dir=out, cached_model_dir="/cache/models/synthetic")
    first = first_ok(chosen_loss=1.0, core_s=1.0, tail_s=0.1, finish_reserve_s=1.0)
    armed = {"on": False, "n": 0, "where": None}
    model = tiny_llama()

    def run_fn(spec_, deadline, rows_, loaded, tokenizer, *, geometry_select, result, **kw):
        geometry_select.on_chosen(0.9)
        try:
            if not commit:
                raise OSError(5, "injected before the commit point")
            common.save_adapter(model, Tok(), spec_.output_dir, artifact_truth=common.ARTIFACT_COMPLETE_BEST, optimizer_step=7,
                                truth_reason=geometry_select.truth_reason)
            result.update(saved=True, chosen_loss=0.9)
        finally:
            armed["on"] = True

    target = gs.__file__

    def local(frame, event, arg):
        if event == "line" and armed["on"]:
            armed["n"] += 1
            if armed["n"] == k:
                armed["where"] = f"{frame.f_code.co_name}:{frame.f_lineno}"
                raise exc_type(f"injected at {armed['where']}")
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_filename == target else None

    tele = Tele()
    rec, raised = None, None
    sys.settrace(tracer)
    try:
        rec = gs.maybe_second_run(spec, Deadline.from_hours(1.0, started_monotonic=time.monotonic(), export_reserve_s=180.0), [],
                                  first=first, run_kwargs={}, run_fn=run_fn, load_base=lambda *a, **kw: NS(model=None, tokenizer=Tok()),
                                  free=lambda: None, artifact_present=lambda s_: True, load_s=1.0, log=lambda n, **d: None,
                                  telemetry=tele)
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    finally:
        sys.settrace(None)
    reason = str(gs.read_export(out, hash_weights=False).get("reason"))
    disk = "S" if reason.startswith(gs.GENERATION_PREFIX) else "A"
    shutil.rmtree(root, ignore_errors=True)
    return rec, raised, armed["where"], disk, tele, armed["n"]


@pytest.mark.parametrize("commit", [True, False])
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_property_no_statement_boundary_after_the_save_yields_a_wrong_pick(tmp_path, commit, exc_type):
    """Inject a BaseException at every executed line of the selector module after run S's save (committed, or failed before
    its commit point). A returned record must name the run that is on disk, or UNKNOWN. An escape is allowed only from the
    few statements that no handler can cover, and then no final record with another pick exists."""
    import os

    from forge.tasks import common

    template = tmp_path / "template"
    os.makedirs(template)
    common.save_adapter(tiny_llama(), Tok(), str(template / "out"), artifact_truth=common.ARTIFACT_COMPLETE_BEST,
                        optimizer_step=3, truth_reason="sft_v2_final")
    want = "S" if commit else "A"
    outcomes = {"correct": 0, "unknown": 0, "escaped": []}
    k = 0
    while True:
        k += 1
        rec, raised, where, disk, tele, lines = _property_run(tmp_path, template, k, commit=commit, exc_type=exc_type)
        assert disk == want, (k, where)                                # the injection never changes what is on disk
        if lines < k:                                                  # no line left to inject at: an uninterrupted run
            assert raised is None and rec["pick"] == want
            break
        final = tele.meta.get(gs.META_KEY) or {}
        if final.get("state") == "final":                              # a final record in the flight recorder never lies
            assert final["pick"] in (want, "UNKNOWN"), (k, where, final["pick"])
        else:
            assert "pick" not in final, (k, where)
        if raised is not None:
            assert where.split(":")[0] in ("maybe_second_run", "_finalize"), (k, where)
            outcomes["escaped"].append(where)
            continue
        assert rec["state"] == "final" and rec["pick"] in (want, "UNKNOWN"), (k, where, rec["pick"], rec.get("reason"))
        assert rec["exported_generation"] == rec["pick"]
        outcomes["correct" if rec["pick"] == want else "unknown"] += 1
        assert k < 2000
    assert k > 60 and outcomes["correct"] >= 0.9 * (k - 1), outcomes   # almost every boundary still ends with the right pick
    # the statements no handler can cover: the return of maybe_second_run and the loop head / fallback of _finalize
    assert len(outcomes["escaped"]) <= 6, outcomes["escaped"]


def test_cli_export_manifest_is_taken_only_when_a_second_run_may_have_started(tmp_path):
    import hashlib

    from forge import cli, telemetry

    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"weights")
    (out / "forge_artifact_truth.json").write_text("{}")
    (out / "forge_run.json").write_text("{}")
    (out / "model.safetensors.tmp").write_text("x")
    saved = dict(telemetry._data["meta"])
    try:
        telemetry._data["meta"] = {}
        cli._record_export_manifest(str(out))
        assert "export_manifest" not in telemetry._data["meta"]                       # no selector record: nothing is hashed
        telemetry._data["meta"] = {"geometry_select": {"state": "final", "second_started": False, "pick": "A"}}
        cli._record_export_manifest(str(out))
        assert "export_manifest" not in telemetry._data["meta"]
        for rec in ({"state": "final", "second_started": True, "pick": "S"}, {"state": "second_run_saving"}, {"state": "second_run_started"}):
            telemetry._data["meta"] = {"geometry_select": rec}
            cli._record_export_manifest(str(out))
            files = telemetry._data["meta"]["export_manifest"]["files"]
            assert sorted(files) == ["forge_artifact_truth.json", "model.safetensors"]
            assert files["model.safetensors"] == {"bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
        telemetry._data["meta"] = {"geometry_select": {"second_started": True}}
        cli._record_export_manifest(str(tmp_path / "missing"))                        # never raises
        assert "export_manifest" not in telemetry._data["meta"]
    finally:
        telemetry._data["meta"] = saved


def test_the_selector_read_back_and_the_cli_manifest_agree_on_the_weights(monkeypatch, tmp_path):
    from forge import cli, telemetry

    h, first = committed(monkeypatch, tmp_path)
    rec, _, _ = orchestrate(h, first)
    saved = dict(telemetry._data["meta"])
    try:
        telemetry._data["meta"] = {"geometry_select": rec}
        cli._record_export_manifest(h.spec.output_dir)
        files = h.meta["export_manifest"]["files"]                    # the harness routes telemetry.set_meta into h.meta
    finally:
        telemetry._data["meta"] = saved
    assert rec["pick"] == "S" and rec["final_export"]["weight_sha256"] == {n: files[n]["sha256"] for n in rec["final_export"]["weight_files"]}
    assert rec["final_export"]["weight_bytes"] == sum(files[n]["bytes"] for n in rec["final_export"]["weight_files"])


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
