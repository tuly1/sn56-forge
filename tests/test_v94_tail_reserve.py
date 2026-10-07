"""release-candidate-v9.4-20261007: the tail reserve (FORGE_V2_TAIL_RESERVE, default on; 0 = 9504136).

9504136 stops training at finish_reserve = export reserve + 1.5 x the estimated dev eval + the dev-pass estimate. The greedy
soup then gets half of what is left above export reserve + dev pass and is skipped below 2.5 measured evals, so a run that
trains to the clock never soups (audit: research/20261007-finish-failure-audit). v9.4 stops training earlier by the measured
time the final eval, the soup and the dev pass need.

Three layers:
1. pure rules (tail_reserve_plan, tail_soup_budget, tail_dev_pass_rows, tail_export_floor);
2. the REAL sft_v2.run driven by a fake clock (training, dev scoring, dev pass and export are doubles that advance the clock;
   route, recipe, geometry, planner, eval governor, stop guard, snapshot pool, greedy soup and the finish stages are the
   release code), replaying the two real 9504136 receipts in tests/data/v94 (task 01fdf13f on the delivered base, seeds 7, 11);
3. a grid over the same harness: never a later training stop, the 9504136 export floor, identical runs when there is margin.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from forge.data import packing
from forge.tasks import sft_v2
from forge.tuning import selection

HERE = Path(__file__).resolve().parent
RECEIPT = {s: json.loads((HERE / f"data/v94/receipt_ce_d0_01fdf13f_s{s}.json").read_text()) for s in (7, 11)}
TAIL_EVENTS = ("sft_v2_tail_reserve", "sft_v2_tail_soup_budget", "sft_v2_tail_dev_pass_shortened", "sft_v2_tail_dev_pass_skipped")


def _rev(seed, name):
    return [e for e in RECEIPT[seed]["events"] if e["name"] == name]


# ---- 1. pure rules ---------------------------------------------------------------------------------------------------

def _plan(**kw):
    base = dict(export_reserve_s=150.0, finish_reserve_s=212.9, eval_s=8.2, pool_n=8, trainable_params=2.5168e9,
                dev_pass_s=46.6, t_step=2.845, persist_max_s=5.0, window_s=3329.6)
    base.update(kw)
    return sft_v2.tail_reserve_plan(**base)


def test_switch_default_on_and_off(monkeypatch):
    monkeypatch.delenv("FORGE_V2_TAIL_RESERVE", raising=False)
    assert sft_v2.tail_reserve_enabled() is True and sft_v2.TAIL_RESERVE_DEFAULT == "1"
    monkeypatch.setenv("FORGE_V2_TAIL_RESERVE", "0")
    assert sft_v2.tail_reserve_enabled() is False


def test_plan_on_the_seed7_receipt_numbers():
    p = _plan()
    # trial = 8.2 + 3.5 x 2.5168 = 17.0 s (measured on seed 11: 122.9 s / 8 = 15.4 s)
    assert p["trial_s"] == pytest.approx(17.0, abs=0.05)
    extra = 1.3 * 8 * (8.2 + 3.5 * 2.5168) + 46.6 + 30 + 2.845
    assert (p["level"], p["soup_members"], p["dev_pass"]) == ("soup_and_dev_pass", 8, True)
    assert p["extra_s"] == pytest.approx(extra, abs=0.1) == pytest.approx(256.3, abs=0.2)
    assert p["extra_s"] < 0.15 * 3329.6  # 7.7 % of the training window


@pytest.mark.parametrize("kw", [dict(), dict(eval_s=300.0, pool_n=3), dict(pool_n=1), dict(pool_n=0, dev_pass_s=0.0),
                                dict(window_s=0.0), dict(persist_max_s=400.0), dict(finish_reserve_s=3695.5, eval_s=371.1)])
def test_extra_is_never_negative_and_never_above_the_cap(kw):
    p = _plan(**kw)
    assert p["extra_s"] >= 0.0 and p["extra_s"] <= sft_v2.TAIL_MAX_WINDOW_FRAC * max(0.0, kw.get("window_s", 3329.6)) + 0.05


def test_degrade_order_is_dev_pass_then_pool_then_nothing():
    seen = []
    for window in (4000, 1400, 1000, 500, 300, 100, 0):
        p = _plan(window_s=float(window))
        seen.append((p["level"], p["soup_members"]))
        if p["level"] == "none":
            assert p["extra_s"] == 0.0  # 9504136 behaviour
    levels = [s[0] for s in seen]
    order = ["soup_and_dev_pass", "soup_only", "soup_reduced", "none"]
    assert [order.index(x) for x in levels] == sorted(order.index(x) for x in levels)  # monotone in the window
    assert levels[0] == "soup_and_dev_pass" and levels[1] == "soup_only" and "soup_reduced" in levels and levels[-1] == "none"
    members = [m for lv, m in seen if lv == "soup_reduced"]
    assert members == sorted(members, reverse=True) and min(members) >= 2  # a smaller pool before no soup, never one member
    # one snapshot: nothing to soup, the dev pass alone is reserved; no dev pass either: nothing
    assert (_plan(pool_n=1)["level"], _plan(pool_n=1)["extra_s"]) == ("dev_pass_only", pytest.approx(46.6 + 30 + 2.845, abs=0.1))
    assert _plan(pool_n=1, dev_pass_s=0.0)["level"] == "none"


def test_slow_eval_cell_keeps_9504136(monkeypatch):
    # phase-ao Qwen3.5-4B short (0.5 h): dev eval 193 s, pool 3, window about 1,250 s: no reduced soup fits in 15 % of it
    p = _plan(eval_s=193.0, pool_n=3, trainable_params=4.2058e9, finish_reserve_s=553.9, dev_pass_s=100.0, t_step=20.0,
              window_s=1250.0)
    assert p == {**p, "level": "none", "extra_s": 0.0}


def test_export_floor_only_rises():
    assert sft_v2.tail_export_floor(150.0, 0.0) == 150.0 and sft_v2.tail_export_floor(150.0, 100.0) == 150.0
    assert sft_v2.tail_export_floor(150.0, 161.1) == pytest.approx(1.3 * 161.1)  # ba-nx15-0928-e-s7's 161 s save
    assert sft_v2.tail_export_floor(60.0, 1.0) == 60.0


def test_soup_budget_is_everything_above_the_floor_the_gate_margin_and_one_averaging():
    kw = dict(export_floor_s=150.0, eval_s=8.2, pool_n=8, trainable_params=2.5168e9)
    b = sft_v2.tail_soup_budget(remaining_s=420.0, **kw)
    assert b["budget_s"] == pytest.approx(420 - 150 - 30 - 8.81, abs=0.1) and b["pool_s"] == pytest.approx(8 * 17.0, abs=0.2)
    assert sft_v2.tail_soup_budget(remaining_s=100.0, **kw)["budget_s"] == 0.0
    for rem in range(0, 2000, 37):
        x = sft_v2.tail_soup_budget(remaining_s=float(rem), export_floor_s=150.0, eval_s=11.2, pool_n=3, trainable_params=3.8e9)
        assert x["budget_s"] <= max(0.0, rem - 150.0 - sft_v2.TAIL_GATE_MARGIN_S)


def test_shortened_dev_pass_rows():
    geo = sft_v2.Geometry(14, 4, 1728, True, True, "adamw_torch_fused")
    est = lambda n: sft_v2._dev_pass_estimate(n, 2.4806, geo)  # noqa: E731
    assert est(400) == pytest.approx(46.6, abs=0.1)
    assert sft_v2.tail_dev_pass_rows(remaining_s=150 + 46.6 + 31, export_floor_s=150.0, n_dev=400, estimate=est) == 400
    n = sft_v2.tail_dev_pass_rows(remaining_s=215.0, export_floor_s=150.0, n_dev=400, estimate=est)
    assert 100 <= n < 400 and 215.0 > 150 + est(n) + 30 and not 215.0 > 150 + est(n + 1) + 30
    assert sft_v2.tail_dev_pass_rows(remaining_s=205.0, export_floor_s=150.0, n_dev=400, estimate=est) == 0  # < 100 rows fit
    assert sft_v2.tail_dev_pass_rows(remaining_s=1e6, export_floor_s=150.0, n_dev=60, estimate=est) == 0   # gate minimum


def test_stop_guard_never_fires_later():
    d = NS(remaining_hard=lambda: 300.0)
    s = sft_v2._StopAt(d, 212.9)
    assert s.extra_s == 0.0 and s.should_stop() is False
    s.extra_s = 204.0
    assert s.should_stop() is True
    for rem in (0.0, 100.0, 212.9):  # whatever the extra, the 9504136 stop still stops
        d.remaining_hard = lambda rem=rem: rem
        s.extra_s = 0.0
        assert s.should_stop() is True


def test_no_model_name_or_size_threshold_in_the_rule():
    src = "".join(inspect.getsource(f) for f in (sft_v2.tail_reserve_plan, sft_v2.tail_soup_budget, sft_v2.tail_dev_pass_rows,
                                                 sft_v2.tail_export_floor, sft_v2.tail_trial_seconds))
    for word in ("model_type", "params_b", "recipe", "lfm", "qwen", "llama", "gemma", "minicpm", "falcon", "strategy"):
        assert word not in src.lower(), word


def test_run_wiring():
    src = inspect.getsource(sft_v2.run)
    assert "stop_at.extra_s = _tp[\"extra_s\"]" in src and "_tail_on = tail_reserve_enabled()" in src
    assert "export_floor = export_reserve\n" in src and "export_floor = tail_export_floor(export_reserve, state[\"persist_max\"])" in src
    assert "if _tb[\"budget_s\"] > soup_budget:" in src          # the soup budget only ever grows
    # the 9504136 expressions are still there, unchanged
    assert "soup_budget = max(0.0, 0.5 * (deadline.remaining_hard() - export_reserve - _dev_pass_estimate(len(dev_ex), t_per_step, geo)))" in src
    assert "len(pool.items) >= 2 and soup_budget < 2.5 * measured_eval" in src
    assert "finish_reserve = export_reserve + (1.5 + _soup_reserve_evals) * dev_eval_est + _dev_pass_estimate(len(dev_ex), t_per_step, geo)" in src
    assert "window = remaining - finish_reserve" in src          # the planner's window does not see the tail reserve


# ---- 2. the real sft_v2.run on a fake clock ---------------------------------------------------------------------------

P_B = 2.5168            # MiniCPM5-2B-Base as delivered (receipt sft_v2_fullft_scope)
HARD_T = 3600.0         # receipt clock: remaining_s + t = 3600.0 at every event
N_ROWS, N_DEV = 22925, 400


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(vocab_size=1024, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, max_position_embeddings=2048, bos_token_id=0, eos_token_id=1,
                                        pad_token_id=1, use_cache=False)).eval()


def _rows(n):
    return [dict(input_ids=[0, 5 + i % 1000, 5 + i // 1000, 17, 1], labels=[0, 5 + i % 1000, 5 + i // 1000, 17, 1]) for i in range(n)]


class Timeline:
    """What the doubles charge the fake clock, read from a receipt (seconds). ``slow`` scales the whole node (training
    and tail alike, as a slower validator GPU would); ``tail_slow`` scales only the finish stages (soup trials, dev pass,
    final save) relative to what the run measured while training."""

    PERSIST = {7: 4.9, 11: 4.4}   # from the receipts' eval-governor line: eval_every = ceil((eval_s + persist / 2) / (0.06 t_probe))

    def __init__(self, seed=7, *, slow=1.0, tail_slow=1.0, hard_t=HARD_T, eval_scale=1.0, mix_s=7.2, dev_pass_s=45.4,
                 final_persist_s=8.0):
        ph = _rev(seed, "sft_v2_phase_end")[0]
        self.seed = seed
        self.t_probe = _rev(seed, "sft_v2_timing")[0]["t_per_step"] * slow
        self.t_step = ph["t_per_step_measured"] * slow
        evs = _rev(seed, "sft_v2_eval")
        self.eval_s = [e["eval_s"] * slow * eval_scale for e in evs]
        self.loss_at = {e["step"]: e["dev_loss"] for e in evs}
        self.persist_s = self.PERSIST[seed] * slow
        # finish stages as measured on seed 11: soup 122.9 s over 8 trials of 8.2 s evals (7.2 s averaging each), dev pass 45.4 s
        self.tail_eval_s = 8.2 * slow * eval_scale * tail_slow
        self.mix_s, self.dev_pass_s = mix_s * slow * tail_slow, dev_pass_s * slow * tail_slow
        self.final_persist_s = final_persist_s * slow * tail_slow
        self.hard_t = hard_t
        self.start_t = _rev(seed, "sft_v2_train_start")[0]["t"]

    def loss(self, step: int) -> float:
        """The receipt's dev loss at ``step`` (linear between the receipt's eval steps, flat outside them)."""
        if step in self.loss_at:
            return self.loss_at[step]
        xs = sorted(self.loss_at)
        if step <= xs[0]:
            return self.loss_at[xs[0]]
        if step >= xs[-1]:
            return self.loss_at[xs[-1]]
        hi = next(x for x in xs if x > step)
        lo = xs[xs.index(hi) - 1]
        w = (step - lo) / (hi - lo)
        return self.loss_at[lo] * (1 - w) + self.loss_at[hi] * w


def _run(monkeypatch, tmp_path, tl: Timeline, *, tail: bool, rows=N_ROWS, n_dev=N_DEV, max_len=1728):
    """The real sft_v2.run with a fake clock. Returns NS(events, trainers, clock, dev_pass_rows, t_export_start, saved)."""
    import torch
    import transformers
    from forge.data.schema import InstructColumns

    if transformers.__version__ != packing.QUALIFIED_TRANSFORMERS:
        pytest.skip("route geometry is qualified on the image's transformers only")
    torch.set_num_threads(2)
    tiny = _tiny_llama()  # before the transformers doubles below: importing a model class re-binds the lazy module's names
    monkeypatch.setenv("FORGE_V2_TAIL_RESERVE", "1" if tail else "0")
    monkeypatch.setenv("FORGE_DEV_PASS", "1")
    clock = [tl.start_t]
    phase = {"training": False, "soup": False, "trained": False, "step": 0}
    out = NS(events=[], trainers=[], clock=clock, dev_pass_rows=[], t_export_start=None, saved=0, evals=0)
    fake_time = NS(monotonic=lambda: clock[0], perf_counter=lambda: clock[0])
    monkeypatch.setattr(sft_v2, "time", fake_time)
    monkeypatch.setattr(selection, "time", fake_time)
    source = _rows(rows)
    monkeypatch.setattr(sft_v2, "_liger_ok", lambda m: False)
    monkeypatch.setattr(sft_v2, "_bnb_available", lambda: False)
    monkeypatch.setattr(sft_v2, "workdir", lambda spec: str(tmp_path / "work"))
    monkeypatch.setattr(sft_v2, "_layer_weight_rms", lambda m: 1.0)
    monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", lambda *a: [])
    monkeypatch.setattr(sft_v2.tokenize, "tokenize_instruct", lambda *a: copy.deepcopy(source))
    monkeypatch.setattr(sft_v2, "adaptive_max_len", lambda *a, **k: max_len)
    monkeypatch.setattr(sft_v2, "dev_size_for", lambda *a, **k: n_dev)
    monkeypatch.setattr(sft_v2, "stratified_split", lambda r, *a, **k: (r[:-n_dev], r[-n_dev:]))
    monkeypatch.setattr(sft_v2, "_event_and_print", lambda n, **d: out.events.append((n, {**d, "_t": round(clock[0], 1)})))
    for name in ("set_meta", "eval_point", "write_into"):
        monkeypatch.setattr(sft_v2.telemetry, name, lambda *a, **k: None)
    monkeypatch.setattr(sft_v2.telemetry, "make_trainer_callback", lambda *a: object())
    monkeypatch.setattr(sft_v2, "write_artifact_truth", lambda *a, **k: None)
    monkeypatch.setattr(selection, "available_ram", lambda: 10 ** 12)
    # the snapshot pool believes the model has the receipt's trainable size (it only uses this for RAM gating and, in
    # v9.4, for the per-trial averaging time); the tensors themselves are the tiny model's
    monkeypatch.setattr(selection.SnapshotPool, "_bytes", lambda self, model: int(P_B * 1e9 * 2))
    monkeypatch.setattr(sft_v2.lr_probe, "measure_step_time", lambda model, batches, **kw: (tl.t_probe, [1.2254, 1.3209, 1.1684]))

    def save(*a, **k):
        if phase["trained"] and out.t_export_start is None and k.get("truth_reason") == "sft_v2_final":
            out.t_export_start = clock[0]
            clock[0] += tl.final_persist_s
        else:
            clock[0] += tl.persist_s
        out.saved += 1

    monkeypatch.setattr(sft_v2, "save_adapter", save)

    def dev_loss(model, rs, collator, **kw):
        out.evals += 1
        if phase["training"]:
            clock[0] += tl.eval_s[min(out.evals - 1, len(tl.eval_s) - 1)]
            return tl.loss(phase["step"]), []
        if phase["soup"]:
            phase["soup_n"] = phase.get("soup_n", 0) + 1
            clock[0] += tl.tail_eval_s + tl.mix_s
            return phase["soup_best"] * (1.0 - 0.004 * phase["soup_n"]), []  # every trial improves a little: all accepted
        clock[0] += tl.tail_eval_s  # the final eval of the last weights
        return tl.loss(phase["step"]), []

    monkeypatch.setattr(selection, "per_example_dev_loss", dev_loss)
    real_soup = selection.greedy_soup

    def soup(model, pool, eval_fn, *, time_budget_s, log):
        phase["soup"], phase["soup_best"], phase["soup_n"] = True, min(e["loss"] for e in pool.items), 0
        out.soup_budget = time_budget_s
        try:
            return real_soup(model, pool, eval_fn, time_budget_s=time_budget_s, log=log)
        finally:
            phase["soup"] = False

    monkeypatch.setattr(selection, "greedy_soup", soup)

    def dev_pass(model, dev, collator, **kw):
        out.dev_pass_rows.append(len(dev))
        clock[0] += tl.dev_pass_s * len(dev) / float(n_dev)

    monkeypatch.setattr(sft_v2, "_dev_pass", dev_pass)

    class Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setattr(sft_v2, "compatible_dataclass_kwargs", lambda cls, kw, **_: kw)
    monkeypatch.setattr(sys.modules["transformers"], "TrainingArguments", Args)

    class ClockTrainer:
        def __init__(self, model, args, train_dataset, data_collator, callbacks=()):
            self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
            self.callbacks = callbacks
            micro_batches = math.ceil(len(train_dataset) / args.per_device_train_batch_size)
            per_epoch = max(1, micro_batches // args.gradient_accumulation_steps + int(micro_batches % args.gradient_accumulation_steps > 0))
            self.state = NS(global_step=0, max_steps=math.ceil(args.num_train_epochs * per_epoch))  # as transformers computes it
            self.optimizer = object()
            self.lr_scheduler = object()
            out.trainers.append(self)

        def get_train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, collate_fn=self.data_collator)

        def train(self):
            phase["training"] = True
            try:
                for step in range(1, self.state.max_steps + 1):
                    clock[0] += tl.t_step
                    self.state.global_step = step
                    phase["step"] = step
                    control = NS(should_training_stop=False)
                    self.callbacks[0].on_step_end(self.args, self.state, control, model=self.model)
                    if control.should_training_stop:
                        break
            finally:
                phase["training"], phase["trained"] = False, True

    monkeypatch.setattr(sys.modules["transformers"], "Trainer", ClockTrainer)
    spec = NS(output_dir=str(tmp_path / "out"), task_id="synthetic", instruct=InstructColumns("instruction", "output"))
    deadline = NS(remaining_hard=lambda: max(0.0, tl.hard_t - clock[0]), export_reserve_s=180.0)
    sft_v2.run(spec, deadline, source, NS(model=tiny), NS(pad_token_id=1), baseline_summary=None,
               params_b=P_B, per_gpu_gb=80.0, strategy_override="full", field_route=True, damaged_base_route=False,
               recipe_key="llama")
    return out


def _ev(out, name):
    return [d for n, d in out.events if n == name]


def _names(out, drop=TAIL_EVENTS):
    return [n for n, _ in out.events if n not in drop]


def _args(out):
    return [{k: v for k, v in t.args.__dict__.items() if k != "output_dir"} for t in out.trainers]


def _steps(out):
    return _ev(out, "sft_v2_train_end")[0]["step"]


def _final_remaining(out):
    return _ev(out, "sft_v2_final")[0]["remaining_s"]


def _export_start_remaining(out, tl):
    return tl.hard_t - out.t_export_start


def test_seed7_receipt_replays_under_the_9504136_rule(monkeypatch, tmp_path):
    """Validates the harness: with the tail reserve off, the real run() on the receipt's timings ends as the real run did."""
    tl = Timeline(7)
    out = _run(monkeypatch, tmp_path, tl, tail=False)
    plan, rplan = _ev(out, "sft_v2_plan")[0], _rev(7, "sft_v2_plan")[0]
    assert (plan["total_steps"], plan["epochs"], plan["eval_every"], plan["warmup"]) == (1206, 3.0, 50, 36) == (
        rplan["total_steps"], rplan["epochs"], rplan["eval_every"], rplan["warmup"])
    assert plan["window_s"] == pytest.approx(rplan["window_s"], abs=0.2)
    assert _ev(out, "sft_v2_train_start")[0]["finish_reserve_s"] == pytest.approx(212.9, abs=0.05)
    assert _ev(out, "sft_v2_eval_governor")[0]["eval_every"] == _rev(7, "sft_v2_eval_governor")[0]["eval_every"] == 74
    assert [e["step"] for e in _ev(out, "sft_v2_eval")][:15] == [e["step"] for e in _rev(7, "sft_v2_eval")][:15]
    assert _ev(out, "sft_v2_early_stop") == []                                    # trained to the clock
    assert abs(_steps(out) - 1113) <= 5 and _ev(out, "sft_v2_phase_end")[0]["planned"] == 1209
    skipped = _ev(out, "sft_v2_soup_skipped")
    assert len(skipped) == 1 and skipped[0]["budget_s"] < 5.0 < 2.5 * skipped[0]["eval_s"]   # receipt: budget 0.9 s, eval 8.7 s
    assert _ev(out, "soup_result") == [] and out.dev_pass_rows == [] and _ev(out, "sft_v2_dev_pass_done") == []
    assert _final_remaining(out) == pytest.approx(190.1, abs=10.0)                # receipt: 190.1 s left, nothing souped
    assert not any(n in TAIL_EVENTS for n, _ in out.events)


def test_seed7_timeline_with_the_tail_reserve_soups_and_dev_passes(monkeypatch, tmp_path):
    tl = Timeline(7)
    off = _run(monkeypatch, tmp_path / "off", tl, tail=False)
    monkeypatch.undo()
    on = _run(monkeypatch, tmp_path / "on", tl, tail=True)
    # same plan
    assert _ev(on, "sft_v2_plan")[0] == _ev(off, "sft_v2_plan")[0] and _args(on) == _args(off)
    assert _ev(on, "sft_v2_train_start")[0] == _ev(off, "sft_v2_train_start")[0]
    # training stops earlier, by the measured tail and not much more
    lost = _steps(off) - _steps(on)
    assert 0 < lost <= 0.10 * _steps(off)
    tr = _ev(on, "sft_v2_tail_reserve")[-1]
    assert tr["level"] == "soup_and_dev_pass" and tr["soup_members"] == 8 and tr["extra_s"] == pytest.approx(256.2, abs=1.0)
    # the whole pool is tried, the dev pass runs on all 400 rows
    soup = _ev(on, "soup_result")
    assert len(soup) == 1 and soup[0]["pool"] == 8 and soup[0]["members"] >= 2
    assert _ev(on, "sft_v2_soup_skipped") == [] and _ev(on, "soup_time_exhausted") == []
    assert on.dev_pass_rows == [400] and _ev(on, "sft_v2_dev_pass_done")[0]["rows"] == 400
    assert _ev(on, "sft_v2_final")[0]["chosen_loss"] < _ev(off, "sft_v2_final")[0]["chosen_loss"]
    # and the export is not later than under 9504136
    assert on.t_export_start <= off.t_export_start
    assert _final_remaining(on) >= _final_remaining(off) and _final_remaining(on) > 180.0


def test_seed11_timeline_is_unchanged(monkeypatch, tmp_path):
    tl = Timeline(11)
    off = _run(monkeypatch, tmp_path / "off", tl, tail=False)
    monkeypatch.undo()
    on = _run(monkeypatch, tmp_path / "on", tl, tail=True)
    # the harness reproduces the receipt: overfit stop at step 1,022, soup over the pool of 8, dev pass on 400 rows
    assert _ev(off, "sft_v2_early_stop")[0]["step"] == _rev(11, "sft_v2_early_stop")[0]["step"] == 1022
    assert _ev(off, "sft_v2_train_end")[0]["remaining_s"] == pytest.approx(464.6, abs=10.0)
    assert _ev(off, "soup_result")[0]["pool"] == 8 and off.dev_pass_rows == [400]
    assert _final_remaining(off) == pytest.approx(288.9, abs=15.0)
    # v9.4 = 9504136 on every event (payload and fake-clock time), every trainer argument, every save
    drop = lambda o: [(n, d) for n, d in o.events if n not in TAIL_EVENTS]  # noqa: E731
    assert drop(on) == drop(off)
    assert _args(on) == _args(off)
    assert (on.saved, on.evals, on.dev_pass_rows, on.t_export_start, on.clock[0]) == (off.saved, off.evals, off.dev_pass_rows,
                                                                                    off.t_export_start, off.clock[0])
    assert _ev(on, "sft_v2_tail_reserve")  # the reserve was armed; the overfit stop came first


# ---- 3. grid on the same harness --------------------------------------------------------------------------------------

GRID = [dict(seed=7, slow=s, tail_slow=ts, hard_t=h, eval_scale=es)
        for s, ts, h, es in [(1.0, 1.0, 3600.0, 1.0), (1.3, 1.0, 3600.0, 1.0), (1.6, 1.0, 3600.0, 1.0), (1.0, 1.3, 3600.0, 1.0),
                             (1.0, 2.0, 3600.0, 1.0), (1.3, 1.3, 3600.0, 1.0), (1.0, 1.0, 2700.0, 1.0), (1.0, 1.0, 1800.0, 1.0),
                             (1.0, 1.0, 900.0, 1.0), (1.0, 1.0, 3600.0, 5.0), (1.0, 1.0, 1800.0, 12.0), (1.0, 1.3, 5400.0, 1.0)]]


@pytest.mark.parametrize("kw", GRID, ids=lambda kw: "s{slow}-t{tail_slow}-h{hard_t:.0f}-e{eval_scale}".format(**kw))
def test_grid_never_later_and_never_below_the_9504136_floor(monkeypatch, tmp_path, kw):
    tl = Timeline(**kw)
    off = _run(monkeypatch, tmp_path / "off", tl, tail=False)
    monkeypatch.undo()
    on = _run(monkeypatch, tmp_path / "on", tl, tail=True)
    # (a) the plan is the same and training never stops later
    assert _ev(on, "sft_v2_plan") == _ev(off, "sft_v2_plan") or len(_ev(on, "sft_v2_plan")) <= len(_ev(off, "sft_v2_plan"))
    assert _ev(on, "sft_v2_plan")[0] == _ev(off, "sft_v2_plan")[0]
    assert _steps(on) <= _steps(off)
    # (b) the export starts with at least the 9504136 gate floor left (export reserve + 30 s), less only what 9504136 is
    #     exposed to as well: a final eval or a dev pass that overran its own estimate
    geo = sft_v2.Geometry(14, 4, 1728, True, True, "adamw_torch_fused")
    dp_est = sft_v2._dev_pass_estimate(N_DEV, tl.t_probe, geo)
    eval_est = sft_v2._estimate_eval_seconds(N_DEV, tl.t_probe, geo)
    for o in (off, on):
        overrun = max(0.0, tl.dev_pass_s * (o.dev_pass_rows[0] / N_DEV) - dp_est) if o.dev_pass_rows else 0.0
        overrun += max(0.0, tl.tail_eval_s - eval_est)
        assert _export_start_remaining(o, tl) >= 150.0 + 30.0 - overrun - 1.0
    last = (_ev(on, "sft_v2_tail_reserve") or [{"level": "none", "soup_members": 0}])[-1]
    # (c) full reserve armed and the finish stages cost no more than slow-factor x what the run measured: the export
    #     starts no later than under 9504136 (to one optimizer step of stop granularity)
    if last["level"] == "soup_and_dev_pass" and kw["tail_slow"] <= sft_v2.TAIL_SLOW_FACTOR:
        assert on.t_export_start <= off.t_export_start + tl.t_step
    # (c') nothing armed: the run is 9504136's
    if last["level"] == "none":
        drop = lambda o: [(n, d) for n, d in o.events if n not in TAIL_EVENTS]  # noqa: E731
        assert _steps(on) == _steps(off) and on.t_export_start >= off.t_export_start - 1e-6
    # (d) never a bare checkpoint where a soup was affordable: if the reserve was armed for a soup, a soup was tried
    if last["soup_members"] >= 2 and _ev(off, "sft_v2_early_stop") == []:
        assert _ev(on, "soup_result") and _ev(on, "sft_v2_soup_skipped") == []
        assert _ev(on, "soup_result")[0]["members"] >= 2
    # (e) the reserve never exceeds its cap
    for e in _ev(on, "sft_v2_tail_reserve"):
        assert 0.0 <= e["extra_s"] <= e["cap_s"] + 0.05


@pytest.mark.parametrize("slow", [1.0, 1.1])
def test_grid_runs_with_margin_are_identical(monkeypatch, tmp_path, slow):
    tl = Timeline(11, slow=slow, hard_t=HARD_T * slow + 300.0)
    off = _run(monkeypatch, tmp_path / "off", tl, tail=False)
    monkeypatch.undo()
    on = _run(monkeypatch, tmp_path / "on", tl, tail=True)
    assert _ev(off, "soup_result") and _ev(off, "soup_time_exhausted") == [] and off.dev_pass_rows == [400]  # 9504136 finished properly
    drop = lambda o: [(n, d) for n, d in o.events if n not in TAIL_EVENTS]  # noqa: E731
    assert drop(on) == drop(off) and on.clock[0] == off.clock[0]
    assert _args(on) == _args(off)


def test_finish_stages_slower_than_reserved_degrade_in_the_declared_order(monkeypatch, tmp_path):
    """Seed 7's timeline with the soup trials and the dev pass 1.6 - 2.6 x slower than the run measured (beyond the 1.3
    factor). The soup keeps priority (its own clock check ends it), the dev pass is shortened to the rows that still fit
    or skipped, every degradation is logged, and the export starts above the 9504136 gate floor less what a stage in
    flight overran its estimate by (9504136 has the same exposure: neither interrupts a dev pass or a soup trial)."""
    seen = {"shortened": 0, "skipped_logged": 0, "soup": 0}
    geo = sft_v2.Geometry(14, 4, 1728, True, True, "adamw_torch_fused")
    for i, ts in enumerate((1.6, 1.9, 2.0, 2.2, 2.6)):
        tl = Timeline(7, tail_slow=ts)
        on = _run(monkeypatch, tmp_path / f"on{i}", tl, tail=True)
        monkeypatch.undo()
        soup = _ev(on, "soup_result")
        assert soup and soup[0]["members"] >= 2 and _ev(on, "sft_v2_soup_skipped") == []   # never the bare checkpoint
        seen["soup"] += 1
        rows = on.dev_pass_rows[0] if on.dev_pass_rows else 0
        overrun = max(0.0, tl.dev_pass_s * rows / N_DEV - sft_v2._dev_pass_estimate(rows, tl.t_probe, geo)) if rows else 0.0
        overrun += tl.tail_eval_s + tl.mix_s  # at most the one soup trial in flight when its budget ran out
        assert _export_start_remaining(on, tl) >= 180.0 - overrun
        short = _ev(on, "sft_v2_tail_dev_pass_shortened")
        if short:
            seen["shortened"] += 1
            assert 100 <= short[0]["rows"] < 400 and on.dev_pass_rows == [short[0]["rows"]]
            assert _ev(on, "sft_v2_dev_pass_done")[0]["rows"] == short[0]["rows"]
        if not on.dev_pass_rows:
            assert _ev(on, "sft_v2_tail_dev_pass_skipped")  # never silent
            seen["skipped_logged"] += 1
    assert seen["soup"] == 5 and seen["shortened"] >= 1 and seen["skipped_logged"] >= 1, seen


def test_overfit_stop_near_the_clock_is_preempted_by_the_reserve(monkeypatch, tmp_path):
    """Seed 11's curve with the deadline pulled in by 150 - 390 s: under 9504136 the overfit stop lands with too little
    left and the soup is cut short or skipped; under v9.4 at least as many members are souped and the floor holds."""
    more = 0
    for i, cut in enumerate(range(150, 420, 60)):
        tl = Timeline(11, hard_t=HARD_T - cut)
        off = _run(monkeypatch, tmp_path / f"off{i}", tl, tail=False)
        monkeypatch.undo()
        on = _run(monkeypatch, tmp_path / f"on{i}", tl, tail=True)
        monkeypatch.undo()
        members = lambda o: (_ev(o, "soup_result") or [{"members": 1}])[0]["members"]  # noqa: E731
        assert _steps(on) <= _steps(off) and members(on) >= members(off)
        assert _export_start_remaining(on, tl) >= 180.0 - 1.0 and on.dev_pass_rows == [400]
        more += members(on) > members(off)
    assert more >= 1


def test_off_switch_never_touches_the_stop_guard(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, Timeline(7, hard_t=1800.0), tail=False)
    assert not any(n in TAIL_EVENTS for n, _ in out.events)


def test_a_failing_reserve_never_fails_the_run(monkeypatch, tmp_path):
    """Any exception inside the new code leaves the 9504136 behaviour and is logged."""
    def boom(**kw):
        raise RuntimeError("synthetic")

    tl = Timeline(7, hard_t=1800.0)
    off = _run(monkeypatch, tmp_path / "off", tl, tail=False)
    monkeypatch.undo()
    monkeypatch.setattr(sft_v2, "tail_reserve_plan", boom)
    monkeypatch.setattr(sft_v2, "tail_soup_budget", boom)
    monkeypatch.setattr(sft_v2, "tail_dev_pass_rows", boom)
    on = _run(monkeypatch, tmp_path / "on", tl, tail=True)
    failed = _ev(on, "sft_v2_tail_reserve_failed")
    assert {e["stage"] for e in failed} >= {"plan", "soup_budget"}
    drop = lambda o: [(n, d) for n, d in o.events if n not in TAIL_EVENTS + ("sft_v2_tail_reserve_failed",)]  # noqa: E731
    assert drop(on) == drop(off) and on.t_export_start == off.t_export_start
