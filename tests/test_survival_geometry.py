"""Focused CPU contracts for the compact survival path."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import weakref
from types import SimpleNamespace

from forge.tasks import common, instruct
from forge.tuning.plan import TrainPlan


def _plan(
    batch: int = 4,
    accum: int = 4,
    *,
    cap: int = 4096,
    gc: bool = False,
) -> TrainPlan:
    return TrainPlan(
        lora_r=32, lora_alpha=64, lora_dropout=0.05,
        learning_rate=1.5e-4, per_device_batch_size=batch,
        grad_accum_steps=accum, max_seq_len=cap, num_epochs=2,
        warmup_ratio=0.03, weight_decay=0.0,
        optimizer="adamw_torch_fused", lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=gc, bf16=True, fp16=False, strategy="lora",
    )


def _view(cap: int, lengths: tuple[int, ...] = (1, 2, 3)) -> instruct._DataView:
    rows = [{"input_ids": list(range(length)), "labels": [1] * length}
            for length in lengths]
    return instruct._DataView(
        cap=cap,
        cap_basis=f"test_cap_{cap}",
        source_rows=len(rows),
        retained_source_rows=len(rows),
        retained_rows=len(rows),
        retained_fraction=1.0,
        ordering_sha256=f"sha-{cap}",
        train=rows,
        validation=[],
    )


def _geometry(plans):
    return [
        (
            item.per_device_batch_size,
            item.grad_accum_steps,
            item.gradient_checkpointing,
            item.max_seq_len,
        )
        for item in plans
    ]


def test_falcon_first_rung_is_exact_and_rescue_order_is_fixed():
    falcon_model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="falcon", max_position_embeddings=2048
        )
    )
    falcon = _plan(cap=instruct.effective_sft_seq_len(falcon_model, 4096))
    routed, changed = instruct.conservative_qwen35_plan(falcon_model, falcon)
    assert routed is falcon and changed is False
    plans = instruct._plans(falcon, (768, 512))
    assert plans[0] == falcon
    assert _geometry(plans) == [
        (4, 4, False, 1024),
        (4, 4, True, 1024),
        (2, 8, True, 1024),
        (1, 16, True, 1024),
        (1, 16, True, 768),
        (1, 16, True, 512),
    ]
    for candidate in plans:
        assert candidate.lora_r == falcon.lora_r
        assert candidate.lora_alpha == falcon.lora_alpha
        assert candidate.lora_dropout == falcon.lora_dropout
        assert candidate.learning_rate == falcon.learning_rate
        assert candidate.optimizer == falcon.optimizer
        assert candidate.lr_scheduler == falcon.lr_scheduler
        assert candidate.num_epochs == falcon.num_epochs
        assert candidate.per_device_batch_size * candidate.grad_accum_steps == 16


def test_qwen_granite_bloom_and_quasar_risk_orders():
    # Qwen3.5's proven route starts at 1x16; it never regresses to 4x4.
    assert _geometry(instruct._plans(_plan(1, 16), (1280,))) == [
        (1, 16, False, 4096),
        (1, 16, True, 4096),
        (1, 16, True, 1280),
    ]
    # Granite/default starts unchanged, then turns on GC before lowering batch.
    assert _geometry(instruct._plans(_plan(), (1280,))) == [
        (4, 4, False, 4096),
        (4, 4, True, 4096),
        (2, 8, True, 4096),
        (1, 16, True, 4096),
        (1, 16, True, 1280),
    ]
    # BLOOM may reach the distribution-derived cap. Quasar's GC no-op is never
    # represented as a rescue rung.
    assert _geometry(instruct._plans(
        _plan(1, 16, gc=True), (1280,), checkpointing_supported=False
    )) == [
        (1, 16, False, 4096),
        (1, 16, False, 1280),
    ]


def test_distribution_caps_are_quantile_derived_not_hardcoded():
    initial = _view(4096)
    first = SimpleNamespace(
        sha256="a" * 64, num_records=3, sequence_p99=1100, sequence_p95=620
    )
    second = SimpleNamespace(
        sha256="b" * 64, num_records=3, sequence_p99=1800, sequence_p95=900
    )
    first_caps, first_authority = instruct._reduced_sequence_caps(
        first, initial, expected_records=3
    )
    second_caps, _ = instruct._reduced_sequence_caps(
        second, initial, expected_records=3
    )
    assert first_caps == (
        (1280, "validated_baseline_p99_round_up_256", 0.99),
        (768, "validated_baseline_p95_round_up_256", 0.95),
    )
    assert second_caps == (
        (2048, "validated_baseline_p99_round_up_256", 0.99),
        (1024, "validated_baseline_p95_round_up_256", 0.95),
    )
    assert first_authority["source"] == "validated_baseline_summary"
    assert first_authority["baseline_stats_sha256"] == "a" * 64
    assert instruct._reduced_sequence_caps(
        None, initial, expected_records=3
    )[0] == ()
    rejected, authority = instruct._reduced_sequence_caps(
        first, initial, expected_records=4
    )
    assert rejected == ()
    assert authority["source"] == "rejected_record_count_mismatch"
    collision = SimpleNamespace(
        sha256="c" * 64, num_records=3, sequence_p99=700, sequence_p95=650
    )
    collision_caps, _ = instruct._reduced_sequence_caps(
        collision, initial, expected_records=3
    )
    assert collision_caps == (
        (768, "validated_baseline_p99_p95_round_up_256", 0.95),
    )


def test_cap_view_records_source_retention_order_and_retokenizes():
    calls = []
    lengths = (800, 1400, 2200)

    def tokenize_one(source_index, cap):
        calls.append((source_index, cap))
        length = lengths[source_index]
        if length > cap:
            return []
        return [{"input_ids": [source_index] * length, "labels": [1] * length}]

    first = instruct._make_data_view(2048, "p99", 3, tokenize_one, False)
    second = instruct._make_data_view(1024, "p95", 3, tokenize_one, False)
    repeat = instruct._make_data_view(1024, "p95", 3, tokenize_one, False)
    assert calls == [
        (0, 2048), (1, 2048), (2, 2048),
        (0, 1024), (1, 1024), (2, 1024),
        (0, 1024), (1, 1024), (2, 1024),
    ]
    assert (first.source_rows, first.retained_source_rows) == (3, 2)
    assert second.identity()["rows_retained"] == 1
    assert second.identity()["tokenized_examples"] == 1
    assert second.retained_fraction == round(1 / 3, 12)
    assert second.ordering_sha256 == repeat.ordering_sha256
    assert first.ordering_sha256 != second.ordering_sha256
    rejected = instruct._make_data_view(
        1024, "p95", 3, tokenize_one, False,
        required_retained_fraction=0.95,
    )
    assert rejected.authorized is False
    assert rejected.required_retained_fraction == 0.95


def test_completion_cap_rechunks_documents_instead_of_filtering():
    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(min(int(text), kwargs["max_length"])))}

    documents = ["10"]
    tokenizer = Tokenizer()

    def tokenize_one(source_index, cap):
        return instruct.tokenize.tokenize_completion(
            [documents[source_index]], tokenizer, cap
        )

    cap4 = instruct._make_data_view(4, "p99", 1, tokenize_one, False)
    cap3 = instruct._make_data_view(3, "p95", 1, tokenize_one, False)
    assert [len(row["input_ids"]) for row in cap4.train] == [4, 4, 2]
    assert [len(row["input_ids"]) for row in cap3.train] == [3, 3, 3, 1]
    assert cap4.retained_source_rows == cap3.retained_source_rows == 1
    assert cap4.ordering_sha256 != cap3.ordering_sha256


def test_real_p99_and_worst_rows_are_selected():
    rows = [{"input_ids": list(range(n))} for n in range(1, 101)]
    selected = instruct._probe_rows(rows, 4)
    assert [item[0] for item in selected] == ["p99", "worst"]
    assert 99 in selected[0][2]["row_lengths"]
    assert selected[1][2]["row_lengths"] == [100, 99, 98, 97]
    assert all(row is rows[index] for row, index in
               zip(selected[0][1], selected[0][2]["row_indices"]))


def test_probe_is_real_production_trainer_and_generations_are_separate():
    factory = inspect.getsource(instruct._make_trainer)
    assert "build_training_kwargs(" in factory
    assert 'kwargs["max_steps"] = 1' in factory
    assert "TrainingArguments(" in factory
    assert "KLSFTTrainer" in factory and "Trainer(**fields)" in factory
    probe = inspect.getsource(instruct._probe_once)
    assert "trainer.train()" in probe and "global_step" in probe
    run = inspect.getsource(instruct.run)
    assert "timing_model = rebuild(candidate)" in run
    assert "train_ex=view.train" in run
    assert "plans[admitted_index:]" in run
    ladder = inspect.getsource(instruct._train_ladder)
    assert ladder.index("measure_epochs(candidate, view)") < ladder.index(
        "current = rebuild(candidate)"
    ) < ladder.index("make(candidate, current, view")


def test_rng_authority_reconstructs_identical_generations():
    import torch

    torch.manual_seed(773)
    authority = instruct._torch_rng()
    first = torch.rand(16)
    _ = torch.rand(31)
    instruct._restore_torch_rng(authority)
    second = torch.rand(16)
    assert torch.equal(first, second)


def test_admission_discards_every_probe_and_selects_checkpointed_rung(monkeypatch):
    import torch

    calls, rebuilds = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def probe(plan, model, rows, identity, view, make):
        calls.append((
            plan.per_device_batch_size,
            plan.gradient_checkpointing,
            identity["label"],
            model,
            view.cap,
        ))
        instruct._discard(model)
        status = "PASS" if plan.gradient_checkpointing else "HEADROOM_EXCEEDED"
        return {**identity, "status": status}

    def rebuild(plan):
        rebuilds.append((plan.per_device_batch_size, plan.gradient_checkpointing))
        return f"fresh-{len(rebuilds)}"

    monkeypatch.setattr(instruct, "_probe_once", probe)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    plans = instruct._plans(_plan(), (1280,))
    full_view = _view(4096)
    cap_view = _view(1280)
    plan, view, index, attempts, admitted = instruct._admit(
        plans,
        "initial",
        lambda candidate: full_view if candidate.max_seq_len == 4096 else cap_view,
        rebuild,
        object(),
    )
    assert _geometry((plan,)) == [(4, 4, True, 4096)]
    assert view is full_view and index == 1 and admitted is True
    assert calls == [
        (4, False, "p99", "initial", 4096),
        (4, True, "p99", "fresh-1", 4096),
        (4, True, "worst", "fresh-2", 4096),
    ]
    assert [item["status"] for item in attempts] == ["HEADROOM_EXCEEDED", "PASS"]
    assert attempts[1]["effective_batch"] == 16
    assert attempts[1]["rows_retained"] == full_view.retained_source_rows


def test_admission_exhaustion_returns_last_nonempty_cap_without_raising(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    def probe(plan, model, rows, identity, view, make):
        calls.append((plan.max_seq_len, view.cap, rows is view.train))
        instruct._discard(model)
        return {**identity, "status": "OOM"}

    monkeypatch.setattr(instruct, "_probe_once", probe)
    plans = instruct._plans(_plan(), (1280, 768))
    rejected = instruct._make_data_view(
        768, "p95", 1, lambda _index, _cap: [], False,
        required_retained_fraction=0.95,
    )
    views = {4096: _view(4096), 1280: _view(1280), 768: rejected}
    selected, view, index, attempts, admitted = instruct._admit(
        plans,
        "initial",
        lambda candidate: views[candidate.max_seq_len],
        lambda candidate: f"fresh-{candidate.max_seq_len}",
        object(),
    )
    assert admitted is False
    assert selected.max_seq_len == view.cap == 1280
    assert index == len(plans) - 2
    assert attempts[-1]["status"] == "CAP_AUTHORITY_REJECTED"
    assert calls[-1][:2] == (1280, 1280)


def test_discard_releases_trainer_model_optimizer_and_accelerator(monkeypatch):
    freed = []

    class Accelerator:
        def free_memory(self):
            freed.append(True)

    class Value:
        pass

    class Trainer:
        pass

    model, optimizer, trainer = Value(), Value(), Trainer()
    model_ref, optimizer_ref, trainer_ref = map(
        weakref.ref, (model, optimizer, trainer)
    )
    trainer.accelerator, trainer.model = Accelerator(), model
    trainer.model_wrapped, trainer.optimizer = model, optimizer
    trainer.lr_scheduler = Value()
    del model, optimizer
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    holder = [trainer]
    del trainer
    instruct._discard(holder.pop(), trainer=True)
    assert freed == [True]
    assert trainer_ref() is model_ref() is optimizer_ref() is None


def test_truth_requires_real_schedule_completion():
    assert instruct._truth(0, 10) == common.ARTIFACT_FLOOR
    assert instruct._truth(7, 10) == common.ARTIFACT_PARTIAL_TRAINED_BEST
    assert instruct._truth(10, 10) == common.ARTIFACT_COMPLETE_BEST
    assert instruct._truth(7, 0) == common.ARTIFACT_PARTIAL_TRAINED_BEST


class _Trainer:
    def __init__(self, model, oom_step):
        self.model = model
        self.state = SimpleNamespace(global_step=oom_step or 0)
        self.oom_step = oom_step
        self.accelerator = SimpleNamespace(free_memory=lambda: None)

    def train(self):
        if self.oom_step is not None:
            raise RuntimeError("CUDA out of memory")


def test_bloom_default_admission_reaches_authorized_cap_and_trains(monkeypatch):
    import torch

    bloom = SimpleNamespace(config=SimpleNamespace(model_type="bloom"))
    plans = instruct._plans(
        _plan(), (1280,),
        checkpointing_supported=not instruct.is_quasar_model(bloom),
    )
    full, capped = _view(4096), _view(1280, (700, 1200))
    views = {4096: full, 1280: capped}
    probes = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    def probe(plan, model, rows, identity, view, make):
        probes.append((plan.max_seq_len, identity["label"], view))
        instruct._discard(model)
        return {
            **identity,
            "status": "PASS" if plan.max_seq_len == 1280 else "OOM",
        }

    monkeypatch.setattr(instruct, "_probe_once", probe)
    selected, selected_view, index, _, admitted = instruct._admit(
        plans,
        "initial-bloom",
        lambda candidate: views[candidate.max_seq_len],
        lambda candidate: f"fresh-{candidate.max_seq_len}",
        object(),
    )
    assert admitted is True
    assert selected.max_seq_len == selected_view.cap == 1280
    assert index == len(plans) - 1
    assert [label for cap, label, _ in probes if cap == 1280] == ["p99", "worst"]

    received = []

    def make(plan, model, view, probe_rows, epochs):
        received.append((plan.max_seq_len, model, view, probe_rows, epochs))
        return _Trainer(model, None), common.BestTracker(), None

    result = instruct._train_ladder(
        plans[index:],
        lambda candidate: views[candidate.max_seq_len],
        lambda _plan: "pristine-bloom",
        make,
        lambda plan, view: received.append(("timing", plan.max_seq_len, view)) or 0.5,
        SimpleNamespace(output_dir="/unused"),
        object(),
    )
    assert received == [
        ("timing", 1280, capped),
        (1280, "pristine-bloom", capped, None, 0.5),
    ]
    assert result[4] is capped and result[-1] == "trained"


def test_unauthorized_nonempty_cap_is_never_probed_or_trained(monkeypatch):
    import torch

    plans = instruct._plans(_plan(1, 16, gc=True), (1280,))
    full = _view(4096)
    capped = instruct._DataView(
        cap=1280,
        cap_basis="p99",
        source_rows=100,
        retained_source_rows=97,
        retained_rows=97,
        retained_fraction=0.97,
        ordering_sha256="unauthorized",
        train=[{"input_ids": [1], "labels": [1]}],
        validation=[],
        authorized=False,
        required_retained_fraction=0.99,
    )
    views = {4096: full, 1280: capped}
    probes = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    def probe(plan, model, rows, identity, view, make):
        probes.append(plan.max_seq_len)
        instruct._discard(model)
        return {**identity, "status": "OOM"}

    monkeypatch.setattr(instruct, "_probe_once", probe)
    selected, selected_view, index, attempts, admitted = instruct._admit(
        plans, "initial", lambda candidate: views[candidate.max_seq_len],
        lambda candidate: f"fresh-{candidate.max_seq_len}", object(),
    )
    assert admitted is False
    assert selected.max_seq_len == selected_view.cap == 4096
    assert attempts[-1]["status"] == "CAP_AUTHORITY_REJECTED"
    assert probes == [4096]

    builds = []

    def make(plan, model, view, _probe_rows, _epochs):
        builds.append((plan.max_seq_len, view))
        return _Trainer(model, 0), common.BestTracker(), None

    result = instruct._train_ladder(
        plans[index:], lambda candidate: views[candidate.max_seq_len],
        lambda _plan: "fresh", make, lambda _plan, _view: None,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert builds == [(4096, full)]
    assert result[3].max_seq_len == result[4].cap == 4096
    assert result[-1] == "zero_progress_exhausted"


def test_cap_transition_releases_previous_view_before_allocating_next(monkeypatch):
    import gc
    import torch

    plans = instruct._plans(_plan(1, 16, gc=True), (1280,))

    def view_cache():
        views = {}
        released = []

        def data_for(candidate):
            cap = candidate.max_seq_len
            if cap not in views:
                if views:
                    old = next(iter(views.values()))
                    reference = weakref.ref(old)
                    del old
                    views.clear()
                    gc.collect()
                    released.append(reference() is None)
                views[cap] = _view(cap)
            return views[cap]

        return data_for, released

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    def probe(plan, model, rows, identity, view, make):
        instruct._discard(model)
        return {
            **identity,
            "status": "PASS" if plan.max_seq_len == 1280 else "OOM",
        }

    monkeypatch.setattr(instruct, "_probe_once", probe)
    admission_data, admission_released = view_cache()
    selected, _, _, _, admitted = instruct._admit(
        plans, "initial", admission_data,
        lambda candidate: f"fresh-{candidate.max_seq_len}", object(),
    )
    assert admitted is True and selected.max_seq_len == 1280
    assert admission_released == [True]

    training_data, training_released = view_cache()

    def make(plan, model, view, _probe_rows, _epochs):
        oom = 0 if plan.max_seq_len == 4096 else None
        return _Trainer(model, oom), common.BestTracker(), None

    result = instruct._train_ladder(
        plans, training_data, lambda _plan: "fresh", make,
        lambda _plan, _view: None,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert result[-1] == "trained" and result[3].max_seq_len == 1280
    assert training_released == [True]


def test_zero_progress_oom_rebuilds_next_trainer(monkeypatch):
    builds, rebuilds, timings = [], [], []
    plans = instruct._plans(_plan())[:2]
    view = _view(4096)

    def make(plan, model, candidate_view, probe_rows, epochs):
        builds.append((
            plan.per_device_batch_size,
            plan.gradient_checkpointing,
            model,
            candidate_view,
            probe_rows,
            epochs,
        ))
        oom = 0 if not plan.gradient_checkpointing else None
        return _Trainer(model, oom), common.BestTracker(), None

    def rebuild(plan):
        rebuilds.append((plan.per_device_batch_size, plan.gradient_checkpointing))
        return f"fresh-{len(rebuilds)}"

    def measure(plan, candidate_view):
        timings.append((plan, candidate_view))
        return 1.25

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    trainer, _, _, selected, selected_view, outcome = instruct._train_ladder(
        plans, lambda _candidate: view, rebuild, make, measure,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert [(item[0], item[1], item[2]) for item in builds] == [
        (4, False, "fresh-1"),
        (4, True, "fresh-2"),
    ]
    assert all(item[3] is view and item[4] is None for item in builds)
    assert all(item[5] == 1.25 for item in builds)
    assert rebuilds == [(4, False), (4, True)]
    assert [item[1] for item in timings] == [view, view]
    assert trainer.model == "fresh-2"
    assert selected.gradient_checkpointing is True
    assert selected_view is view and outcome == "trained"


def test_zero_progress_model_is_released_before_lower_geometry_rebuild(monkeypatch):
    references = []

    class Model:
        def __init__(self):
            references.append(weakref.ref(self))

        def zero_grad(self, **_kwargs):
            pass

    def make(plan, model, _view, _probe_rows, _epochs):
        oom = 0 if not plan.gradient_checkpointing else None
        return _Trainer(model, oom), common.BestTracker(), None

    def rebuild(_plan):
        if references:
            assert references[-1]() is None
        return Model()

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    trainer, _, _, _, _, outcome = instruct._train_ladder(
        instruct._plans(_plan())[:2], lambda _candidate: _view(4096),
        rebuild, make, lambda _plan, _data: None,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert trainer.model is not None
    assert outcome == "trained"


def test_progressed_oom_preserves_and_never_retries(monkeypatch):
    preserved, rebuilds = [], []

    def make(_plan, model, _view, _probe_rows, _epochs):
        return _Trainer(model, 7), common.BestTracker(), None

    monkeypatch.setattr(
        instruct, "_preserve_progress",
        lambda trainer, tracker, spec, tokenizer, step:
            preserved.append((trainer.model, step)) or True,
    )
    def rebuild(plan):
        rebuilds.append(plan)
        return "trained"

    result = instruct._train_ladder(
        ( _plan(), ), lambda _candidate: _view(4096), rebuild,
        make, lambda _plan, _data: None,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert preserved == [("trained", 7)]
    assert len(rebuilds) == 1
    assert result[-1] == "progressed_oom_preserved"


def test_progressed_oom_export_failure_is_not_reported_as_preserved(monkeypatch):
    def make(_plan, model, _view, _probe_rows, _epochs):
        return _Trainer(model, 7), common.BestTracker(), None

    monkeypatch.setattr(instruct, "_preserve_progress", lambda *_args: False)
    result = instruct._train_ladder(
        (_plan(),), lambda _candidate: _view(4096), lambda _plan: "trained",
        make, lambda _plan, _data: None,
        SimpleNamespace(output_dir="/unused"), object(),
    )
    assert result[-1] == "progressed_oom_export_failed"


def test_stale_truth_cannot_mask_failed_structural_preservation(monkeypatch):
    tracker = common.BestTracker()
    tracker.persisted_best = 0.5
    tracker.persisted_best_step = 7
    monkeypatch.setattr(instruct, "_truth_matches", lambda *_args: True)
    monkeypatch.setattr(instruct, "write_artifact_truth", lambda *_args, **_kwargs: False)
    assert instruct._preserve_progress(
        SimpleNamespace(model=object()), tracker,
        SimpleNamespace(output_dir="/incomplete"), object(), 9,
    ) is False
    monkeypatch.setattr(instruct, "write_artifact_truth", lambda *_args, **_kwargs: True)
    assert instruct._preserve_progress(
        SimpleNamespace(model=object()), tracker,
        SimpleNamespace(output_dir="/complete"), object(), 9,
    ) is True


def test_zero_progress_exhaustion_returns_honest_degraded_outcome(monkeypatch):
    def make(_plan, model, _view, _probe_rows, _epochs):
        return _Trainer(model, 0), common.BestTracker(), None

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    plans = instruct._plans(_plan(), (1280,))
    views = {4096: _view(4096), 1280: _view(1280)}
    result = instruct._train_ladder(
        plans,
        lambda candidate: views[candidate.max_seq_len],
        lambda candidate: f"fresh-{candidate.max_seq_len}",
        make,
        lambda _plan, _data: None,
        SimpleNamespace(output_dir="/unused"),
        object(),
    )
    trainer, tracker, route, selected, selected_view, outcome = result
    assert trainer is tracker is route is None
    assert selected.max_seq_len == selected_view.cap == 1280
    assert outcome == "zero_progress_exhausted"


def test_terminal_rung_refreshes_top_level_telemetry(monkeypatch):
    captured = []
    selected = _plan(1, 16, cap=768, gc=True)
    view = _view(768, (500, 700))
    monkeypatch.setattr(
        instruct.telemetry, "set_meta", lambda **fields: captured.append(fields)
    )
    instruct._record_training_selection(
        selected, view, "zero_progress_exhausted"
    )
    assert captured == [{
        "seq_len": 768,
        "sequence_cap_basis": view.cap_basis,
        "tokenized": 2,
        "rows_retained": 2,
        "retained_source_rows": 2,
        "retained_fraction": 1.0,
        "dataset_ordering_sha256": "sha-768",
        "train_n": 2,
        "val_n": 0,
        "batch": 1,
        "grad_accum": 16,
        "eff_batch": 16,
        "gradient_checkpointing": True,
        "tokens_per_step_cap": 16 * 768,
        "training_outcome": "zero_progress_exhausted",
    }]


def test_cap_specific_view_reaches_timing_and_real_trainer(monkeypatch):
    selected = _plan(1, 16, cap=1280, gc=True)
    view = _view(1280, (500, 1200))
    received = []

    def measure(plan, candidate_view):
        received.append(("timing", plan, candidate_view))
        return 0.75

    def make(plan, model, candidate_view, probe_rows, epochs):
        received.append(
            ("trainer", plan, candidate_view, model, probe_rows, epochs)
        )
        return _Trainer(model, None), common.BestTracker(), None

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    result = instruct._train_ladder(
        (selected,), lambda _candidate: view, lambda _plan: "pristine",
        make, measure, SimpleNamespace(output_dir="/unused"), object(),
    )
    assert received[0] == ("timing", selected, view)
    assert received[1] == ("trainer", selected, view, "pristine", None, 0.75)
    assert result[4] is view and result[-1] == "trained"


def _artifact(path: str) -> None:
    os.makedirs(path)
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    with open(os.path.join(path, "model.safetensors"), "wb") as handle:
        handle.write(len(header).to_bytes(8, "little") + header + b"data")


def test_truth_relabel_is_atomic_and_does_not_change_weights(tmp_path):
    output = str(tmp_path / "artifact")
    _artifact(output)
    before = (tmp_path / "artifact" / "model.safetensors").read_bytes()
    assert common.write_artifact_truth(
        output, common.ARTIFACT_PARTIAL_TRAINED_BEST,
        optimizer_step=4, reason="trained"
    )
    truth = json.loads(
        (tmp_path / "artifact" / "forge_artifact_truth.json").read_text()
    )
    assert truth["truth"] == "PARTIAL_TRAINED_BEST"
    assert truth["optimizer_step"] == 4
    assert (tmp_path / "artifact" / "model.safetensors").read_bytes() == before


def test_frozen_recipe_files_are_unchanged():
    expected = {
        "forge/tuning/plan.py": "302b0c1da3de865ed08af186cf0b2fc8452173dc9a04ab88eee41a1de6578787",
        "forge/tasks/dpo.py": "25bb3d48c2ff1a568d434d3983d4384af1e060967072429cbab68a0b74e1ccd5",
        "forge/tasks/grpo.py": "16ecd365a984b807b7849eb86337092b789ea5bea1e2585768fd5f7cdc92897d",
    }
    for path, digest in expected.items():
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == digest


# ---------------------------------------------------------------------------
# Admission headroom and wall-budget planning, calibrated against the
# 2026-09-03 survival smoke (H100 80 GB, lease 20260903T105152Z).  The numbers
# below are the measured `peak_reserved_fraction`, probe `train_runtime`,
# per-step and per-eval wall costs recorded in that lease's forge_run.json and
# train.log files.
# ---------------------------------------------------------------------------


def test_admission_headroom_constants_and_verdict_boundary():
    assert instruct._ADMISSION_CEILING == 0.90
    assert instruct._STEADY_STATE_HEADROOM == 0.12
    assert instruct._admission_verdict(0.78) == ("PASS", 0.9)
    assert instruct._admission_verdict(0.780001) == ("HEADROOM_EXCEEDED", 0.900001)
    assert instruct._admission_verdict(0.0) == ("PASS", 0.12)
    # The old bare ceiling would have admitted these; the headroom does not.
    assert instruct._admission_verdict(0.85)[0] == "HEADROOM_EXCEEDED"
    assert instruct._admission_verdict(0.9)[0] == "HEADROOM_EXCEEDED"


# Measured admission probes (label, rung) -> (peak_reserved_fraction, verdict).
_MEASURED_PROBES = {
    "falcon-candidate b4/ga4 seq1024 p99": (0.213246, "PASS"),
    "falcon-candidate b4/ga4 seq1024 worst": (0.176127, "PASS"),
    "minicpm5-1b b4/ga4 gc-off worst": (0.903768, "HEADROOM_EXCEEDED"),
    "minicpm5-1b b4/ga4 gc worst": (0.412976, "PASS"),
    "bloomz-560m b4/ga4 gc p99": (0.491211, "PASS"),
    "bloomz-560m b4/ga4 gc worst": (0.831206, "HEADROOM_EXCEEDED"),
    "qwen3.5-9b b1/ga16 p99": (0.569618, "PASS"),
    "qwen3.5-9b b1/ga16 worst": (0.857276, "HEADROOM_EXCEEDED"),
    "granite-4.1-8b b4/ga4 gc worst": (0.629281, "PASS"),
    "gemma-4-e4b-it b4/ga4 gc worst": (0.933858, "HEADROOM_EXCEEDED"),
    "gemma-4-e4b-it b2/ga8 gc worst": (0.628541, "PASS"),
    "lfm2.5-8b-a1b b4/ga4 gc worst": (0.603828, "PASS"),
}


def test_admission_verdict_on_every_measured_probe():
    for label, (peak, expected) in _MEASURED_PROBES.items():
        status, predicted = instruct._admission_verdict(peak)
        assert status == expected, label
        assert predicted == round(peak + 0.12, 6), label
        assert (predicted <= 0.90) == (expected == "PASS"), label


def test_probe_once_records_prediction_headroom_and_step_wall(monkeypatch):
    import torch

    total = 81_559 * 1024 * 1024
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", lambda device=None: int(0.831206 * total)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device=None: SimpleNamespace(total_memory=total),
    )
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    class Probe:
        def __init__(self):
            self.state = SimpleNamespace(global_step=0)
            self.accelerator = SimpleNamespace(free_memory=lambda: None)

        def train(self):
            self.state.global_step = 1

    view = _view(4096)
    observation = instruct._probe_once(
        _plan(gc=True), object(), view.train[:1], {"label": "worst"}, view,
        lambda plan, model, candidate_view, rows: (Probe(), None, None),
    )
    assert observation["label"] == "worst"
    assert observation["status"] == "HEADROOM_EXCEEDED"
    assert observation["peak_reserved_fraction"] == 0.831206
    assert observation["predicted_steady_state_fraction"] == 0.951206
    assert observation["headroom"] == 0.12
    assert observation["admission_ceiling"] == 0.90
    assert observation["step_wall_s"] >= 0.0

    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", lambda device=None: int(0.176127 * total)
    )
    falcon = instruct._probe_once(
        _plan(cap=1024), object(), view.train[:1], {"label": "worst"}, view,
        lambda plan, model, candidate_view, rows: (Probe(), None, None),
    )
    assert falcon["status"] == "PASS"
    assert falcon["predicted_steady_state_fraction"] == 0.296127


def _probe_table(table, step_wall_s):
    def probe(plan, model, rows, identity, view, make):
        instruct._discard(model)
        key = (plan.per_device_batch_size, plan.gradient_checkpointing, identity["label"])
        peak = table[key]
        if peak is None:
            return {**identity, "status": "OOM",
                    "error": "OutOfMemoryError: CUDA out of memory"}
        status, predicted = instruct._admission_verdict(peak)
        return {**identity, "status": status, "peak_reserved_fraction": peak,
                "predicted_steady_state_fraction": predicted,
                "headroom": instruct._STEADY_STATE_HEADROOM,
                "admission_ceiling": instruct._ADMISSION_CEILING,
                "step_wall_s": step_wall_s}
    return probe


def test_falcon_measured_probes_admit_the_first_rung_unchanged(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    falcon_model = SimpleNamespace(
        config=SimpleNamespace(model_type="falcon", max_position_embeddings=2048)
    )
    falcon = _plan(cap=instruct.effective_sft_seq_len(falcon_model, 4096))
    plans = instruct._plans(falcon, (768,))
    # 2026-09-03 falcon-candidate: p99 0.213246 (cold first probe), worst 0.176127;
    # probe train_runtime 0.738 s and 0.344 s.
    table = {(4, False, "p99"): 0.213246, (4, False, "worst"): 0.176127}
    monkeypatch.setattr(instruct, "_probe_once", _probe_table(table, 0.344))
    views = {1024: _view(1024, (832, 944)), 768: _view(768)}
    selected, view, index, attempts, admitted = instruct._admit(
        plans, "initial", lambda c: views[c.max_seq_len],
        lambda c: f"fresh-{c.max_seq_len}", object(),
    )
    assert admitted is True and index == 0 and selected == falcon
    assert _geometry((selected,)) == [(4, 4, False, 1024)]
    assert [item["status"] for item in attempts] == ["PASS"]
    assert [b["label"] for b in attempts[0]["batches"]] == ["p99", "worst"]
    assert attempts[0]["predicted_steady_state_fraction"] == 0.333246
    assert attempts[0]["headroom"] == 0.12
    assert instruct._admitted_prediction(attempts, admitted) == 0.333246
    assert instruct._admitted_step_s(attempts, admitted) == 0.344


def test_measured_bloomz_and_qwen35_9b_step_down_exactly_one_rung(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)

    # bloomz-560m: gc-off p99 OOM; gc-on p99 0.491 / worst 0.831, then trained
    # at a 0.916 host peak.  The b2/ga8/gc rung was not probed that day; it is
    # assumed to fit here.
    bloomz = {
        (4, False, "p99"): None,
        (4, True, "p99"): 0.491211, (4, True, "worst"): 0.831206,
        (2, True, "p99"): 0.30, (2, True, "worst"): 0.50,
    }
    monkeypatch.setattr(instruct, "_probe_once", _probe_table(bloomz, 2.51))
    plans = instruct._plans(_plan(), (2560, 1536))
    views = {4096: _view(4096), 2560: _view(2560), 1536: _view(1536)}
    selected, view, index, attempts, admitted = instruct._admit(
        plans, "initial", lambda c: views[c.max_seq_len],
        lambda c: f"fresh-{c.max_seq_len}", object(),
    )
    assert admitted is True and index == 2 and view is views[4096]
    assert _geometry((selected,)) == [(2, 8, True, 4096)]
    assert [item["status"] for item in attempts] == [
        "OOM", "HEADROOM_EXCEEDED", "PASS"
    ]
    assert attempts[1]["predicted_steady_state_fraction"] == 0.951206
    assert attempts[1]["batches"][-1]["label"] == "worst"
    assert attempts[1]["batches"][-1]["status"] == "HEADROOM_EXCEEDED"
    assert attempts[2]["predicted_steady_state_fraction"] == 0.62
    assert instruct._admitted_prediction(attempts, admitted) == 0.62
    assert instruct._admitted_step_s(attempts, admitted) == 2.51

    # qwen3.5-9b on its proven b1/ga16 route: p99 0.570 / worst 0.857, then
    # trained at a 0.908 host plateau (65.8 GiB allocated).  The gc rescue rung
    # keeps b1/ga16 and is assumed to fit.
    qwen = {
        (1, False, "p99"): 0.569618, (1, False, "worst"): 0.857276,
        (1, True, "p99"): 0.35, (1, True, "worst"): 0.55,
    }
    monkeypatch.setattr(instruct, "_probe_once", _probe_table(qwen, 20.92))
    plans = instruct._plans(_plan(1, 16), (2816, 1536))
    views = {4096: _view(4096), 2816: _view(2816), 1536: _view(1536)}
    selected, view, index, attempts, admitted = instruct._admit(
        plans, "initial", lambda c: views[c.max_seq_len],
        lambda c: f"fresh-{c.max_seq_len}", object(),
    )
    assert admitted is True and index == 1
    assert _geometry((selected,)) == [(1, 16, True, 4096)]
    assert [item["status"] for item in attempts] == ["HEADROOM_EXCEEDED", "PASS"]
    assert attempts[0]["predicted_steady_state_fraction"] == 0.977276
    assert instruct._admitted_step_s(attempts, admitted) == 20.92


def test_admitted_extraction_is_none_without_a_passing_measured_rung():
    exhausted = [{"status": "OOM", "batches": []}]
    assert instruct._admitted_prediction(exhausted, False) is None
    assert instruct._admitted_step_s(exhausted, False) is None
    assert instruct._admitted_step_s(exhausted, True) is None
    unmeasured = [{"status": "PASS", "batches": [
        {"label": "p99", "status": "PASS", "step_wall_s": 0.7},
        {"label": "worst", "status": "PASS"},
    ]}]
    assert instruct._admitted_step_s(unmeasured, True) is None
    zero = [{"status": "PASS", "batches": [{"label": "worst", "step_wall_s": 0.0}]}]
    assert instruct._admitted_step_s(zero, True) is None
    assert instruct._admitted_step_s([{"status": "SKIP_NO_CUDA"}], False) is None


def test_steps_per_epoch_and_eval_cadence_match_the_trainer():
    # (train rows, batch, accum) -> steps/epoch, cross-checked against each
    # cell's train_end / train_steps_per_second in the smoke.
    assert instruct._steps_per_epoch(875, 4, 4) == 55      # falcon: 110 = 2 epochs
    assert instruct._steps_per_epoch(1339, 4, 4) == 84     # bloomz: 168
    assert instruct._steps_per_epoch(1338, 4, 4) == 84     # minicpm 168 / lfm 336
    assert instruct._steps_per_epoch(1337, 1, 16) == 84    # qwen3.5-9b: 168
    assert instruct._steps_per_epoch(1340, 4, 4) == 84     # granite: ceil(2.51*84)
    assert instruct._steps_per_epoch(1337, 2, 8) == 84     # gemma: ceil(2.4*84)
    assert math.ceil(2.51 * 84) == 211 and math.ceil(2.4 * 84) == 202
    assert instruct._steps_per_epoch(1337, 1, 16, n_gpus=2) == 42
    assert instruct._steps_per_epoch(0, 4, 4) == 1
    assert instruct._eval_every(875, 16) == 13   # falcon evaluated at 13, 26, ...
    assert instruct._eval_every(1337, 16) == 20  # 2026-pool cells: every 20
    assert instruct._eval_every(10, 16) == 1


def test_falcon_smoke_fixture_keeps_all_110_steps_with_measured_numbers():
    # falcon-candidate, 2026-09-03: hours 0.2 -> 540 s soft budget, admission
    # finished at 10.4 s; admission probes 0.738 s (p99, cold) and 0.344 s
    # (worst); real training 0.68-0.78 s/step net, 2.7 s per eval.
    budget = 540.0 - 10.4
    for step_s in (0.3436, 0.7381, 0.78):
        wall = instruct._plan_wall_steps(
            budget_s=budget, step_s=step_s, step_source="admission_worst_batch",
            train_rows=875, val_rows=256, per_device_batch=4, grad_accum=4,
            epochs=2,
        )
        assert wall["schedule_steps"] == 110
        assert wall["eval_every"] == 13
        assert wall["reason"] == "schedule_fits"
        assert wall["cap_applied"] is False
        assert wall["planned_steps"] == 110
        assert wall["affordable_steps"] >= 3 * 110
    # Replay of the real run: 110 steps + 9 evaluations fit with room to spare.
    replay = 110 * 0.78 + math.ceil(110 / 13) * 2.7 + instruct._WALL_PLAN_SETUP_S
    assert replay < 0.4 * budget


def test_wall_plan_caps_granite_and_gemma_inside_their_measured_budgets():
    # granite-4.1-8b: planned at t=206.7 s of a 1620 s soft budget; probe
    # 5.8429 s/step; 1340 rows b4/ga4; 2.51 epochs -> 211 steps; the deadline
    # stopped it at 206 with the anneal unfinished.
    budget = 1620.0 - 206.7
    granite = instruct._plan_wall_steps(
        budget_s=budget, step_s=5.8429, step_source="timing_probe",
        train_rows=1340, val_rows=256, per_device_batch=4, grad_accum=4,
        epochs=2.51,
    )
    assert granite["schedule_steps"] == 211 and granite["eval_every"] == 20
    assert granite["reason"] == "wall_budget_cap" and granite["cap_applied"] is True
    assert granite["planned_steps"] == 165
    assert granite["measured_step_s"] == 5.8429
    assert granite["estimated_wall_s"] <= (budget - 60.0) * 0.9
    # Replay with the measured run costs (5.7 s/step net, 23.7 s per eval,
    # final evaluation included): completes inside the budget, using most of it.
    replay = 165 * 5.7 + math.ceil(165 / 20) * 23.7 + instruct._WALL_PLAN_SETUP_S
    assert replay < budget
    assert replay > 0.75 * budget

    # gemma-4-e4b-it: planned at t=257.4 s; probe 5.8863 s/step; 1337 rows
    # b2/ga8; 2.4 epochs -> 202 steps; stopped at 192.
    budget = 1620.0 - 257.4
    gemma = instruct._plan_wall_steps(
        budget_s=budget, step_s=5.8863, step_source="timing_probe",
        train_rows=1337, val_rows=256, per_device_batch=2, grad_accum=8,
        epochs=2.4,
    )
    assert gemma["schedule_steps"] == 202
    assert gemma["reason"] == "wall_budget_cap"
    assert gemma["planned_steps"] == 160
    replay = 160 * 6.05 + math.ceil(160 / 20) * 20.5 + instruct._WALL_PLAN_SETUP_S
    assert replay < budget
    assert replay > 0.75 * budget


def test_wall_plan_leaves_completed_cells_uncapped():
    # lfm2.5-8b-a1b: planned at t=97.4 s; probe 1.5725 s/step; 1338 rows
    # b4/ga4; 4.0 epochs -> 336 steps; completed at 782 s.
    lfm = instruct._plan_wall_steps(
        budget_s=1620.0 - 97.4, step_s=1.5725, step_source="timing_probe",
        train_rows=1338, val_rows=256, per_device_batch=4, grad_accum=4,
        epochs=4.0,
    )
    assert lfm["schedule_steps"] == 336
    assert lfm["reason"] == "schedule_fits" and lfm["planned_steps"] == 336
    assert lfm["estimated_eval_s"] == 7.55  # measured 7.1-7.3 s

    # bloomz-560m and minicpm5-1b (0.3 h): the timing probe declines below
    # 900 s remaining, so the admission worst-batch probe (2.51 s / 1.70 s) is
    # the measurement; both two-epoch schedules of 168 steps fit.
    for step_s, rows in ((2.51, 1339), (1.697, 1338)):
        wall = instruct._plan_wall_steps(
            budget_s=900.0 - 22.0, step_s=step_s,
            step_source="admission_worst_batch", train_rows=rows,
            val_rows=256, per_device_batch=4, grad_accum=4, epochs=2,
        )
        assert wall["schedule_steps"] == 168
        assert wall["reason"] == "schedule_fits" and wall["cap_applied"] is False


def test_wall_plan_caps_qwen35_from_the_admission_probe_when_timing_probe_is_cut():
    # Raw planner pin (superseded in run() by `_plan_cold_only`, tested below):
    # qwen3.5-9b: the timing probe was cut at 180 s (per-step 4.4-20 s) and
    # returned nothing, so the default two epochs (168 steps) were scheduled
    # and the deadline stopped training at 118.  The admitted rung's worst-
    # batch probe took 20.92 s (a cold upper bound of the 4.4 s steady step).
    # Planning at ~t=310 s of the 1620 s soft budget:
    qwen9b = instruct._plan_wall_steps(
        budget_s=1620.0 - 310.0, step_s=20.92, step_source="admission_worst_batch",
        train_rows=1337, val_rows=256, per_device_batch=1, grad_accum=16,
        epochs=2,
    )
    assert qwen9b["schedule_steps"] == 168
    assert qwen9b["reason"] == "wall_budget_cap"
    assert qwen9b["planned_steps"] == 40
    # Replay with the measured costs (20 steps at 20.6 s warm-up, then 4.4 s;
    # first eval 83 s, later 27 s): the capped schedule completes early.
    replay = 20 * 20.6 + 20 * 4.4 + 83 + 27 + instruct._WALL_PLAN_SETUP_S
    assert replay < 1620.0 - 310.0

    # qwen3.5-4b (0.4 h): same shape; worst-batch probe 20.93 s; planning at
    # ~t=256 s of the 1260 s soft budget (66 steps of 168 ran before the stop).
    qwen4b = instruct._plan_wall_steps(
        budget_s=1260.0 - 256.0, step_s=20.93, step_source="admission_worst_batch",
        train_rows=1337, val_rows=256, per_device_batch=1, grad_accum=16,
        epochs=2,
    )
    assert qwen4b["reason"] == "wall_budget_cap"
    assert qwen4b["planned_steps"] == 30
    assert 30 * 13.0 + 2 * 26.2 + 82 + instruct._WALL_PLAN_SETUP_S < 1260.0 - 256.0


def test_wall_plan_without_measurement_or_budget_leaves_schedule_to_deadline():
    for step_s in (None, 0.0, -1.0, float("nan"), float("inf"), "n/a"):
        wall = instruct._plan_wall_steps(
            budget_s=1000.0, step_s=step_s, step_source="none",
            train_rows=1337, val_rows=256, per_device_batch=1, grad_accum=16,
            epochs=2,
        )
        assert wall["reason"] == "unmeasured_step_time"
        assert wall["cap_applied"] is False
        assert wall["planned_steps"] == 168
        assert wall["measured_step_s"] is None
    tiny = instruct._plan_wall_steps(
        budget_s=70.0, step_s=20.92, step_source="admission_worst_batch",
        train_rows=1337, val_rows=256, per_device_batch=1, grad_accum=16,
        epochs=2,
    )
    assert tiny["reason"] == "budget_below_one_step"
    assert tiny["cap_applied"] is False and tiny["affordable_steps"] == 0
    assert tiny["planned_steps"] == 168
    # No validation rows (KL or tiny datasets): no evaluation cost is charged.
    no_eval = instruct._plan_wall_steps(
        budget_s=1000.0, step_s=10.0, step_source="timing_probe",
        train_rows=1337, val_rows=0, per_device_batch=1, grad_accum=16,
        epochs=2,
    )
    assert no_eval["estimated_eval_s"] == 0.0
    assert no_eval["affordable_steps"] == int((1000.0 - 60.0) * 0.9 // 10.0)


def test_ladder_passes_the_cap_only_when_it_binds(monkeypatch):
    received = []

    def make(plan, model, view, probe_rows, epochs, **kwargs):
        received.append((epochs, kwargs))
        return _Trainer(model, None), common.BestTracker(), None

    monkeypatch.setattr(instruct, "_free_cuda", lambda: None)
    args = (
        (_plan(),), lambda _c: _view(4096), lambda _p: "fresh", make,
        lambda _p, _v: 2.51, SimpleNamespace(output_dir="/unused"), object(),
    )
    fits = instruct._train_ladder(*args, plan_steps=lambda plan, view, epochs: None)
    assert received == [(2.51, {})] and fits[-1] == "trained"
    received.clear()
    seen = []

    def plan_steps(plan, view, epochs):
        seen.append((plan, view.cap, epochs))
        return 165

    capped = instruct._train_ladder(*args, plan_steps=plan_steps)
    assert received == [(2.51, {"max_steps": 165})] and capped[-1] == "trained"
    assert seen == [(_plan(), 4096, 2.51)]
    received.clear()
    legacy = instruct._train_ladder(*args)
    assert received == [(2.51, {})] and legacy[-1] == "trained"


def test_wall_plan_is_recorded_in_event_and_meta(monkeypatch):
    events, metas = [], []
    monkeypatch.setattr(
        instruct.telemetry, "event", lambda name, **kv: events.append((name, kv))
    )
    monkeypatch.setattr(instruct.telemetry, "set_meta", lambda **kv: metas.append(kv))
    wall = instruct._plan_wall_steps(
        budget_s=1620.0 - 206.7, step_s=5.8429, step_source="timing_probe",
        train_rows=1340, val_rows=256, per_device_batch=4, grad_accum=4,
        epochs=2.51,
    )
    view = _view(4096, (700, 4084))
    instruct._record_wall_plan(wall, _plan(gc=True), view)
    assert [name for name, _ in events] == ["wall_budget_plan"]
    recorded = events[0][1]
    assert recorded["planned_steps"] == 165
    assert recorded["schedule_steps"] == 211
    assert recorded["reason"] == "wall_budget_cap"
    assert recorded["budget_s"] == 1413.3
    assert recorded["batch"] == 4 and recorded["gradient_checkpointing"] is True
    assert recorded["sequence_cap"] == 4096 and recorded["train_n"] == 2
    assert recorded["validation_rows"] == 0 and recorded["eval_rows"] == 256
    # A colliding key would have raised inside the ladder: keep them disjoint.
    assert not set(wall) & set(view.identity())
    assert not set(wall) & set(instruct._plan_identity(_plan()))
    assert metas == [{
        "planned_steps": 165,
        "measured_step_s": 5.8429,
        "budget_s": 1413.3,
        "plan_reason": "wall_budget_cap",
        "wall_plan": wall,
    }]


def test_step_cap_reaches_the_real_trainer_and_probe_stays_single_step():
    factory = inspect.getsource(instruct._make_trainer)
    assert 'kwargs["max_steps"] = 1' in factory
    assert 'kwargs["max_steps"] = int(max_steps)' in factory
    assert factory.index("if probe:") < factory.index('kwargs["max_steps"] = int(max_steps)')
    assert "eval_steps=_eval_every(len(train_ex), eff)" in factory
    run = inspect.getsource(instruct.run)
    assert "plan_steps=plan_steps" in run
    assert "_admitted_step_s(admission, admitted)" in run
    assert 'timing["probe_per_step"] = probe_per_step' in run
    assert "budget_s=deadline.remaining()" in run
    assert "admitted_predicted_steady_state_fraction=" in run
    ladder = inspect.getsource(instruct._train_ladder)
    assert ladder.index("current = rebuild(candidate)") < ladder.index(
        "plan_steps(candidate, view, epochs)"
    ) < ladder.index("max_steps=step_cap")


# ---------------------------------------------------------------------------
# Cold-only planning: warm probe, discount and floor, calibrated against the
# 646d0b70 smoke (VM 1022601, torch 2.9.1; Mac mirror lease
# 20260903T214700Z-sep7-smoke-v6-clean5-5p5h).  Numbers are the wall_budget_plan
# events, admission step_wall_s values and train_curve timings recorded there.
# ---------------------------------------------------------------------------

_QWEN_GEOMETRY = dict(
    train_rows=1337, val_rows=256, per_device_batch=1, grad_accum=16, epochs=2,
)


def test_cold_only_planning_constants():
    assert instruct._COLD_PROBE_DISCOUNT == 3.0
    assert instruct._MIN_REAL_STEPS == 50
    assert instruct._FLOOR_SCHEDULE_FRACTION == 0.30
    assert instruct._WARM_PROBE_STEPS == 3
    assert instruct._WARM_PROBE_BUDGET_FRACTION == 0.05
    assert instruct._step_floor(168) == 51      # max(50, ceil(0.3 * 168))
    assert instruct._step_floor(40) == 40       # never above the schedule
    assert instruct._step_floor(1000) == 300
    assert instruct._step_floor(0) == 1
    assert instruct._finite_positive("x") is None
    assert instruct._finite_positive(float("nan")) is None
    assert instruct._finite_positive(0) is None
    assert instruct._finite_positive("2.5") == 2.5


def test_qwen35_4b_plans_at_least_fifty_steps_from_the_cold_probe():
    # qwen3.5-4b (0.4 h): admitted b1/ga16, worst-batch admission timing
    # 21.259 s, plan at budget 990.4 s -> the raw planner allowed 29 of 168 and
    # the cell trained exactly 29 steps (COMPLETE at 1004 s, host 0.65).
    budget, cold = 990.4, 21.259
    decision = instruct._warm_probe_decision(budget_s=budget, cold_step_s=cold, **_QWEN_GEOMETRY)
    assert decision["needed"] is True and decision["cold_planned_steps"] == 29
    assert decision["cost_estimate_s"] == 63.8 and decision["allowance_s"] == 49.5
    assert decision["affordable"] is False and decision["run"] is False

    plan = instruct._plan_cold_only(budget_s=budget, cold_step_s=cold, warm_step_s=None, **_QWEN_GEOMETRY)
    assert plan["step_source"] == "admission_worst_batch_discounted"
    assert plan["measured_step_s"] == round(cold / 3, 4)
    assert plan["cold_step_s"] == cold and plan["cold_planned_steps"] == 29
    assert plan["floor_steps"] == 51
    assert plan["reason"] == "wall_budget_cap" and plan["cap_applied"] is True
    assert plan["planned_steps"] == 94
    assert 50 <= plan["planned_steps"] <= plan["schedule_steps"] == 168
    assert plan["estimated_wall_s"] <= instruct._usable_budget_s(budget)

    # Had a warm probe measured steps 2-3 still inside the warm-up (17 s), the
    # measured plan (39) would be lifted to the floor because the discounted
    # estimate affords it.
    warm = instruct._plan_cold_only(budget_s=budget, cold_step_s=cold, warm_step_s=17.0, **_QWEN_GEOMETRY)
    assert warm["step_source"] == "warm_probe" and warm["measured_step_s"] == 17.0
    assert warm["affordable_steps"] == 39
    assert warm["planned_steps"] == 51 and warm["reason"] == "wall_budget_cap_floor"
    assert warm["discounted_affordable_steps"] == 94
    assert warm["estimated_wall_s"] <= instruct._usable_budget_s(budget)


def test_qwen35_9b_warm_probe_runs_even_though_the_micro_batch_timing_fits():
    # qwen3.5-9b (0.5 h): admitted b1/ga16/gc after the no-gc rung was
    # rejected (worst 0.857 -> 0.977); its worst-batch admission timing was a
    # warm 1.786 s micro-batch, so the raw planner said the 168-step schedule
    # fits (affordable 499) while real steps ran 7.5-23 s and the deadline
    # stopped training at 77.  The warm probe must therefore run whenever it
    # is affordable, not only when the cold plan caps.
    budget, cold = 1288.4, 1.786
    decision = instruct._warm_probe_decision(budget_s=budget, cold_step_s=cold, **_QWEN_GEOMETRY)
    assert decision["needed"] is False and decision["cold_planned_steps"] == 168
    assert decision["cost_estimate_s"] == 5.4 and decision["allowance_s"] == 64.4
    assert decision["run"] is True and decision["stop_after_s"] == 64.4

    # ~45 s later the probe reports a 12 s warm step: a real cap, above 50.
    plan = instruct._plan_cold_only(budget_s=budget - 45.0, cold_step_s=cold, warm_step_s=12.0, **_QWEN_GEOMETRY)
    assert plan["step_source"] == "warm_probe"
    assert plan["reason"] == "wall_budget_cap" and plan["planned_steps"] == 69
    assert plan["cold_planned_steps"] == 168 and plan["floor_steps"] == 51
    assert plan["estimated_wall_s"] <= instruct._usable_budget_s(budget - 45.0)

    # Steps 2-3 measured deep in the warm-up (23 s): floor 51 applies.
    early = instruct._plan_cold_only(budget_s=budget - 45.0, cold_step_s=cold, warm_step_s=23.0, **_QWEN_GEOMETRY)
    assert early["affordable_steps"] == 36
    assert early["planned_steps"] == 51 and early["reason"] == "wall_budget_cap_floor"

    # Without any warm measurement the fitting cold plan is kept as it was.
    untouched = instruct._plan_cold_only(budget_s=budget, cold_step_s=cold, warm_step_s=None, **_QWEN_GEOMETRY)
    assert untouched["step_source"] == "admission_worst_batch"
    assert untouched["reason"] == "schedule_fits" and untouched["planned_steps"] == 168
    assert untouched["floor_steps"] is None


def test_cold_only_plan_keeps_fitting_schedules_untouched():
    # falcon-candidate (mirror): admission worst 0.707 s at budget 526.9 s ->
    # affordable 431, schedule 110 fits; bloomz / minicpm (0.3 h, timing probe
    # declines below 900 s): worst 2.51 s / 1.697 s, 168-step schedules fit.
    for cold, budget, rows, batch, accum, schedule in (
        (0.707, 526.9, 875, 4, 4, 110),
        (2.51, 878.1, 1339, 4, 4, 168),
        (1.697, 877.3, 1338, 4, 4, 168),
    ):
        plan = instruct._plan_cold_only(
            budget_s=budget, cold_step_s=cold, warm_step_s=None, train_rows=rows,
            val_rows=256, per_device_batch=batch, grad_accum=accum, epochs=2,
        )
        assert plan["schedule_steps"] == schedule
        assert plan["step_source"] == "admission_worst_batch"
        assert plan["reason"] == "schedule_fits" and plan["cap_applied"] is False
        assert plan["planned_steps"] == schedule and plan["floor_steps"] is None
        decision = instruct._warm_probe_decision(
            budget_s=budget, cold_step_s=cold, train_rows=rows, val_rows=256,
            per_device_batch=batch, grad_accum=accum, epochs=2,
        )
        assert decision["needed"] is False and decision["run"] is True
    # Falcon with its warm measurement (0.72 s real step): still 110, no cap.
    falcon = instruct._plan_cold_only(
        budget_s=526.9, cold_step_s=0.707, warm_step_s=0.72, train_rows=875,
        val_rows=256, per_device_batch=4, grad_accum=4, epochs=2,
    )
    assert falcon["step_source"] == "warm_probe"
    assert falcon["reason"] == "schedule_fits" and falcon["planned_steps"] == 110
    assert falcon["cap_applied"] is False


def test_floor_never_exceeds_schedule_and_needs_discounted_affordability():
    # A 40-step schedule (small dataset) that the warm step would cap at 27:
    # the floor is the schedule itself and the discounted estimate (10 s ->
    # 41 steps in the 414 s usable budget) affords it.
    small = instruct._plan_cold_only(
        budget_s=520.0, cold_step_s=30.0, warm_step_s=15.0, train_rows=320,
        val_rows=0, per_device_batch=4, grad_accum=4, epochs=2,
    )
    assert small["schedule_steps"] == 40 and small["floor_steps"] == 40
    assert small["affordable_steps"] == 27
    assert small["discounted_affordable_steps"] == 41
    assert small["planned_steps"] == 40 and small["cap_applied"] is False
    assert small["reason"] == "schedule_fits_by_floor"
    # With less budget the discounted estimate affords only 30 < 40: no bump.
    short = instruct._plan_cold_only(
        budget_s=400.0, cold_step_s=30.0, warm_step_s=15.0, train_rows=320,
        val_rows=0, per_device_batch=4, grad_accum=4, epochs=2,
    )
    assert short["discounted_affordable_steps"] == 30
    assert short["planned_steps"] == short["affordable_steps"] == 20
    assert short["reason"] == "wall_budget_cap"
    # When even the discounted estimate cannot afford the floor, the measured
    # cap stands (the deadline callback remains the backstop).
    tight = instruct._plan_cold_only(
        budget_s=200.0, cold_step_s=30.0, warm_step_s=15.0, train_rows=1337,
        val_rows=0, per_device_batch=1, grad_accum=16, epochs=2,
    )
    assert tight["floor_steps"] == 51 and tight["discounted_affordable_steps"] < 51
    assert tight["planned_steps"] == tight["affordable_steps"] == 8
    assert tight["reason"] == "wall_budget_cap"
    # Cold-only with an unusable warm value behaves as if unmeasured warm.
    for bad in ("n/a", float("inf"), 0.0, -3.0):
        plan = instruct._plan_cold_only(
            budget_s=990.4, cold_step_s=21.259, warm_step_s=bad, **_QWEN_GEOMETRY
        )
        assert plan["step_source"] == "admission_worst_batch_discounted"
        assert plan["planned_steps"] == 94


def test_warm_probe_median_uses_steps_two_and_three():
    assert instruct._warm_probe_median([20.9, 8.0, 6.0]) == 7.0
    assert instruct._warm_probe_median([20.9, 8.0]) == 8.0
    assert instruct._warm_probe_median([20.9]) is None
    assert instruct._warm_probe_median([]) is None
    assert instruct._warm_probe_median([20.9, float("nan"), 6.0]) == 6.0
    assert instruct._warm_probe_median([20.9, 8.0, 6.0, 1.0]) == 7.0  # a 4th step is ignored


class _FakeClock:
    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.now = 0.0

    def monotonic(self):
        if self.ticks:
            self.now = self.ticks.pop(0)
        return self.now


def test_warm_probe_times_real_steps_and_discards_state(monkeypatch, tmp_path):
    from transformers import TrainerControl, TrainerState

    built, discarded = [], []

    class Trainer:
        def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **extra):
            built.append((model, args, len(train_dataset), data_collator, callbacks, extra))
            self.args, self.callbacks = args, callbacks
            self.accelerator = SimpleNamespace(free_memory=lambda: None)
            self.model = model

        def train(self):
            # Mirrors the Trainer loop: a stop raised on a sub-step ends the
            # window before its optimizer step (no on_step_end for it).
            state, control = TrainerState(), TrainerControl()
            for _step in range(10):
                for callback in self.callbacks:
                    callback.on_step_begin(self.args, state, control)
                for callback in self.callbacks:
                    callback.on_substep_end(self.args, state, control)
                if control.should_training_stop:
                    break
                for callback in self.callbacks:
                    callback.on_step_end(self.args, state, control)
                if control.should_training_stop:
                    break

    # opened, then (begin, substep, end) per step: durations 21.0, 8.0, 6.0.
    clock = _FakeClock([0.0, 1.0, 5.0, 22.0, 23.0, 25.0, 31.0, 32.0, 34.0, 38.0, 40.0])
    monkeypatch.setattr(instruct, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        instruct, "_discard",
        lambda value, trainer=False: discarded.append((type(value).__name__, trainer)),
    )
    kwargs = dict(
        output_dir=str(tmp_path), per_device_train_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=1.5e-4,
        neftune_noise_alpha=5.0, eval_strategy="steps", eval_steps=20,
        per_device_eval_batch_size=1, report_to=[], disable_tqdm=True,
        save_strategy="no", logging_steps=10,
    )
    rows = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}] * 4
    result = instruct._warm_probe_step_s(
        trainer_cls=Trainer, model="warm-model", kwargs=kwargs, train_ex=rows,
        collator="collator", trainer_extra={"kl_coef": 0.1}, stop_after_s=64.4,
    )
    assert result["source"] == "warm_probe" and result["probe_steps"] == 3
    assert result["steps_completed"] == 3
    assert result["step_durations_s"] == [21.0, 8.0, 6.0]
    assert result["step_s"] == 7.0
    assert result["stop_after_s"] == 64.4 and result["elapsed_s"] == 40.0
    assert "error" not in result
    model, args, n_rows, collator, callbacks, extra = built[0]
    assert model == "warm-model" and n_rows == 4 and collator == "collator"
    assert extra == {"kl_coef": 0.1} and len(callbacks) == 1
    assert args.max_steps == 3 and args.learning_rate == 0.0
    assert args.neftune_noise_alpha is None
    assert str(args.eval_strategy) in ("no", "IntervalStrategy.NO")
    assert discarded == [("Trainer", True)]

    # Self-limit: the allowance expires during the second step, so only one
    # full step was timed and no estimate is produced.
    clock = _FakeClock([0.0, 1.0, 5.0, 22.0, 23.0, 70.0, 75.0])
    monkeypatch.setattr(instruct, "time", SimpleNamespace(monotonic=clock.monotonic))
    result = instruct._warm_probe_step_s(
        trainer_cls=Trainer, model="warm-model", kwargs=kwargs, train_ex=rows,
        collator="collator", stop_after_s=49.5,
    )
    assert result["steps_completed"] == 1 and result["step_s"] is None
    assert result["step_durations_s"] == [21.0]

    # A failing trainer is recorded, never raised.
    class Broken:
        def __init__(self, **_kwargs):
            raise RuntimeError("CUDA out of memory")

    freed = []
    monkeypatch.setattr(instruct, "_free_cuda", lambda: freed.append(True))
    monkeypatch.setattr(instruct, "time", SimpleNamespace(monotonic=_FakeClock([0.0, 1.0]).monotonic))
    result = instruct._warm_probe_step_s(
        trainer_cls=Broken, model="m", kwargs=kwargs, train_ex=rows, collator=None,
    )
    assert result["step_s"] is None and "RuntimeError: CUDA out of memory" in result["error"]
    assert freed == [True]


def test_run_wires_the_warm_probe_and_cold_only_planning():
    run = inspect.getsource(instruct.run)
    assert 'timing["warm_step_s"] = warm_probe(' in run
    assert run.index("candidate_epochs, probe_per_step = time_aware_epochs(") < run.index(
        'timing["warm_step_s"] = warm_probe('
    ) < run.index("_discard(holder.pop())")
    assert "if candidate_epochs is None:" in run
    assert "_warm_probe_decision(" in run and "_warm_probe_step_s(" in run
    assert 'stop_after_s=decision["stop_after_s"]' in run
    assert "_plan_cold_only(" in run
    assert run.index('step_source="timing_probe"') < run.index("_plan_cold_only(")
    assert 'warm_step_s=timing.get("warm_step_s")' in run
