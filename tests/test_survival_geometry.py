"""Focused CPU contracts for the compact survival path."""

from __future__ import annotations

import hashlib
import inspect
import json
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
