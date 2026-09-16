"""Field-route data hygiene (FORGE_V2_FIELD_HYGIENE=1): CPU tests of the pure mechanisms
and of the knob-off contract."""
from __future__ import annotations

import ast
import inspect
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from forge.baseline import load_baseline_summary, telemetry_fields
from forge.data import hygiene
from forge.data.split import grouped_stratified_split, stratified_split
from forge.data.tokenize import PadCollator
from forge.tasks import sft_v2


class _Tok:
    """Char-level stand-in with decode (ids > 2 are characters; 0/1/2 pad/bos/eos)."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        return "".join(chr(i) for i in ids if i > 2 or not skip_special_tokens)


def _ex(prompt: str, completion: str) -> dict[str, list[int]]:
    p = [ord(c) for c in prompt]
    c = [ord(c) for c in completion]
    return {"input_ids": [1] + p + c + [2], "labels": [-100] * (1 + len(p)) + c + [2]}


def _words(rng: random.Random, n: int) -> str:
    return " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9))) for _ in range(n))


# ---- A. MinHash-LSH near-duplicate groups ------------------------------------------------

def test_minhash_groups_exact_and_near_duplicates_but_not_unrelated():
    rng = random.Random(1)
    base = _words(rng, 110)  # ~600 characters
    near = base[: len(base) - 30] + " " + _words(rng, 5)  # ~95% character overlap (shingle Jaccard ~0.9)
    half = base[: len(base) // 2] + " " + _words(rng, 55)  # half the text replaced
    texts = [base, _words(rng, 110), base, near, _words(rng, 110), half, _words(rng, 3)]
    groups = hygiene.near_duplicate_groups(texts)
    assert sorted(i for g in groups for i in g) == list(range(len(texts)))  # a partition
    by_index = {i: tuple(g) for g in groups for i in g}
    assert by_index[0] == by_index[2] == by_index[3] == (0, 2, 3)
    for i in (1, 4, 5, 6):
        assert by_index[i] == (i,)


def test_signature_agreement_estimates_shingle_jaccard():
    rng = random.Random(2)
    a = _words(rng, 150)
    b = a[: int(len(a) * 0.8)] + " " + _words(rng, 30)
    sa, sb = set(map(int, hygiene.shingle_hashes(a))), set(map(int, hygiene.shingle_hashes(b)))
    jaccard = len(sa & sb) / len(sa | sb)
    assert abs(hygiene.signature_agreement(a, b) - jaccard) < 0.1
    assert hygiene.signature_agreement(a, a) == 1.0
    assert hygiene.signature_agreement(a, _words(rng, 150)) < 0.2


def test_short_and_empty_texts_are_handled():
    groups = hygiene.near_duplicate_groups(["", "ab", "ab", "", "abcdefgh"])
    by_index = {i: tuple(g) for g in groups for i in g}
    assert by_index[0] == (0, 3) and by_index[1] == (1, 2) and by_index[4] == (4,)
    assert hygiene.near_duplicate_groups([]) == []


def test_buckets_above_the_cap_are_skipped_as_boilerplate():
    text = _words(random.Random(3), 80)
    assert len(hygiene.near_duplicate_groups([text] * 150)) == 1
    assert len(hygiene.near_duplicate_groups([text] * (hygiene.MAX_BUCKET + 1))) == hygiene.MAX_BUCKET + 1


def test_drop_near_duplicates_keeps_the_first_of_each_group_in_corpus_order():
    rows = [{"input_ids": [i], "labels": [i]} for i in range(6)]
    kept, groups = hygiene.drop_near_duplicates(rows, [[0, 4], [1], [2, 3, 5]])
    assert [r["input_ids"][0] for r in kept] == [0, 1, 2]
    assert groups == [[0], [1], [2]]


def test_dedup_trigger_rule(monkeypatch):
    assert not hygiene.dedup_triggered(SimpleNamespace(near_duplicate_rate=0.2))
    assert hygiene.dedup_triggered(SimpleNamespace(near_duplicate_rate=0.21))
    assert not hygiene.dedup_triggered(None)
    assert hygiene.dedup_triggered(None, forced=True)
    monkeypatch.delenv("FORGE_V2_FIELD_DEDUP", raising=False)
    assert not hygiene.dedup_forced()
    monkeypatch.setenv("FORGE_V2_FIELD_DEDUP", "1")
    assert hygiene.dedup_forced()


def test_render_texts_drops_special_tokens():
    assert hygiene.render_texts([_ex("ab", "cd")], _Tok()) == ["abcd"]


# ---- A. group-aware dev split ---------------------------------------------------------------

def _corpus(n: int, seed: int = 5) -> list[dict[str, list[int]]]:
    rng = random.Random(seed)
    return [_ex("p" * rng.randint(1, 60), "c" * rng.randint(1, 80)) for _ in range(n)]


def test_grouped_split_keeps_every_group_on_one_side():
    rows = _corpus(400)
    groups = [[i, i + 1, i + 2] for i in range(0, 60, 3)] + [[i] for i in range(60, 400)]
    train, dev = grouped_stratified_split(rows, 50, groups, seed=7)
    ids = {id(r) for r in dev}
    assert 50 <= len(dev) <= 55 and len(train) + len(dev) == 400
    assert not ({id(r) for r in train} & ids)
    for g in groups[:20]:
        sides = {id(rows[i]) in ids for i in g}
        assert len(sides) == 1, "a near-duplicate group straddles train/dev"
    assert any(all(id(rows[i]) in ids for i in g) for g in groups[:20])  # groups do reach dev


def test_grouped_split_singletons_and_edge_cases():
    rows = _corpus(300)
    train, dev = grouped_stratified_split(rows, 30, None, seed=7)
    assert len(dev) == 30 and len(train) == 270
    assert grouped_stratified_split(rows, 0, None) == (rows, [])
    assert grouped_stratified_split(rows[:1], 5, [[0]]) == (rows[:1], [])
    # one giant group can never be dev: it is skipped and the quota comes from the rest
    giant = [list(range(0, 250))] + [[i] for i in range(250, 300)]
    train, dev = grouped_stratified_split(rows, 30, giant, seed=7)
    assert len(dev) == 30 and all(id(r) not in {id(rows[i]) for i in giant[0]} for r in dev)


# ---- B. Gaussian difficulty core -------------------------------------------------------------

def test_gaussian_core_size_centering_and_order():
    rows = [{"input_ids": [0] * ((i % 400) + 2), "labels": [-100] + [7] * ((i % 400) + 1), "i": i} for i in range(6000)]
    kept, target = hygiene.difficulty_subsample(rows, coverage=0.3, seed=7)
    assert target == 2000 and len(kept) == 2000  # floor 2000 > 0.3 * 6000
    order = [r["i"] for r in kept]
    assert order == sorted(order) and len(set(order)) == 2000
    pop = hygiene.completion_lengths(rows)
    core = hygiene.completion_lengths(kept)
    assert abs(float(np.median(core)) - float(np.median(pop))) < 25
    assert float(np.std(core)) < 0.9 * float(np.std(pop))
    kept2, target2 = hygiene.difficulty_subsample(rows, coverage=0.45, seed=7)
    assert target2 == 2700 and len(kept2) == 2700
    assert hygiene.gaussian_core(rows, 2000, seed=7) == kept


def test_difficulty_subsample_gates():
    rows = _corpus(2500)
    assert hygiene.difficulty_subsample(rows, coverage=None) == (rows, None)
    assert hygiene.difficulty_subsample(rows, coverage=0.5) == (rows, None)
    assert hygiene.difficulty_subsample(rows, coverage=0.9) == (rows, None)
    assert hygiene.difficulty_subsample(rows[:2000], coverage=0.1) == (rows[:2000], None)
    kept, target = hygiene.difficulty_subsample(rows, coverage=0.1)
    assert target == 2000 and len(kept) == 2000
    assert hygiene.difficulty_subsample(rows, coverage=float("nan")) == (rows, None)


def test_coverage_formula():
    assert hygiene.coverage(rows=10000, eff_batch=100, t_per_step=2.0, budget_s=3600.0) == pytest.approx(15.3)
    assert hygiene.coverage(rows=40000, eff_batch=100, t_per_step=6.0, budget_s=3600.0) == pytest.approx(1.275)
    assert hygiene.coverage(rows=40000, eff_batch=100, t_per_step=20.0, budget_s=3600.0) == pytest.approx(0.3825)
    assert hygiene.coverage(rows=40000, eff_batch=100, t_per_step=None, budget_s=3600.0) is None
    assert hygiene.coverage(rows=0, eff_batch=100, t_per_step=1.0, budget_s=3600.0) is None


# ---- C. prompt-loss weight -------------------------------------------------------------------

def test_prompt_loss_weight_formula_and_ratio():
    assert hygiene.prompt_loss_weight(3.0) == 0.0
    assert hygiene.prompt_loss_weight(5.0) == 0.0
    assert hygiene.prompt_loss_weight(6.0) == pytest.approx(0.05 / 1.2)
    assert hygiene.prompt_loss_weight(28.0) == pytest.approx(0.05 / 5.6)
    assert hygiene.prompt_loss_weight(float("inf")) == 0.0
    rows = [_ex("p" * 20, "c" * 4), _ex("p" * 30, "c" * 5)]  # prompt tokens include BOS: 21 + 31; completions 5 + 6
    assert hygiene.prompt_completion_ratio(rows) == pytest.approx(52 / 11)


def test_prompt_weight_collator_weights():
    batch = hygiene.PromptWeightCollator(0, 0.02)([_ex("ab", "cd"), _ex("a", "b")])
    w = batch["loss_weights"]
    assert w.dtype == torch.float32 and w.shape == (2, 6)
    assert w[0].tolist() == pytest.approx([0.0, 0.02, 0.02, 1.0, 1.0, 1.0])
    assert w[1].tolist() == pytest.approx([0.0, 0.02, 1.0, 1.0, 0.0, 0.0])
    plain = PadCollator(0)([_ex("ab", "cd")])
    assert "loss_weights" not in plain and plain["labels"].tolist() == [batch["labels"][0].tolist()]


class _Decoder(torch.nn.Module):
    def __init__(self, vocab: int, dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, **_):  # noqa: ANN001
        return SimpleNamespace(last_hidden_state=torch.tanh(self.proj(self.embed(input_ids))))


class TinyCausalLM(torch.nn.Module):
    """Exposes `model` (decoder) and `lm_head` like a HF causal LM; its own loss is the
    standard masked token-mean cross-entropy."""

    def __init__(self, vocab: int = 130, dim: int = 8) -> None:
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig()
        self.model = _Decoder(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.vocab = vocab

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, attention_mask=None, labels=None, **_):  # noqa: ANN001
        logits = self.lm_head(self.model(input_ids).last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def _reference_weighted_loss(model: TinyCausalLM, batch: dict) -> torch.Tensor:
    logits = model(batch["input_ids"]).logits[:, :-1]
    targets = torch.where(batch["labels"] != -100, batch["labels"], batch["input_ids"])[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, model.vocab), targets.reshape(-1), reduction="none")
    w = batch["loss_weights"][:, 1:].reshape(-1)
    return (ce * w).sum() / w.sum()


def test_weighted_loss_equals_plain_ce_when_prompt_weights_are_zero():
    torch.manual_seed(0)
    model = TinyCausalLM()
    rows = [_ex("abcd", "ef"), _ex("a", "bcdefg"), _ex("abc", "d")]
    batch = hygiene.PromptWeightCollator(0, 0.0)(rows)
    plain = model(**PadCollator(0)(rows)).loss
    weighted = hygiene.weighted_token_loss(model, batch, use_fused=False)
    assert torch.allclose(weighted, plain, atol=1e-6)
    assert torch.allclose(weighted, _reference_weighted_loss(model, batch), atol=1e-6)


def test_weighted_loss_matches_formula_with_prompt_weight():
    torch.manual_seed(1)
    model = TinyCausalLM()
    rows = [_ex("abcdefgh", "ij"), _ex("abcdefghijk", "l"), _ex("ab", "cdefgh")]
    batch = hygiene.PromptWeightCollator(0, 0.05 / 5.6)(rows)
    weighted = hygiene.weighted_token_loss(model, batch, use_fused=False)
    assert torch.allclose(weighted, _reference_weighted_loss(model, batch), atol=1e-6)
    assert not torch.allclose(weighted, model(**PadCollator(0)(rows)).loss, atol=1e-4)
    weighted.backward()
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
    empty = {**batch, "loss_weights": torch.zeros_like(batch["loss_weights"])}
    assert hygiene.weighted_token_loss(model, empty, use_fused=False).detach().item() == 0.0


def test_prompt_weight_trainer_runs_the_weighted_objective(tmp_path):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(2)
    model = TinyCausalLM()
    rows = [_ex("p" * (5 + i % 7), "c" * (1 + i % 3)) for i in range(12)]
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, max_steps=2, learning_rate=1e-3, optim="adamw_torch", weight_decay=0.0,
        report_to=[], remove_unused_columns=False, logging_steps=1, save_strategy="no", disable_tqdm=True,
    )
    collator = hygiene.PromptWeightCollator(0, 0.01)
    TrainerCls = sft_v2.make_prompt_weight_trainer(Trainer, use_fused=False)
    trainer = TrainerCls(model=model, args=args, train_dataset=Dataset.from_list(rows), data_collator=collator)
    assert trainer.model_accepts_loss_kwargs is False
    batch = collator(rows[:2])
    loss = trainer.compute_loss(model, dict(batch))
    assert torch.allclose(loss, _reference_weighted_loss(model, batch), atol=1e-6)
    trainer.train()
    assert trainer.state.global_step == 2


# ---- D. NEFTune escalation ----------------------------------------------------------------

def test_neftune_escalation_rule(monkeypatch):
    monkeypatch.delenv("FORGE_V2_NEFTUNE", raising=False)
    noisy = SimpleNamespace(gradient_noise_scale=1.7)
    assert sft_v2._neftune_alpha(noisy, "full") == 1.0  # knob off: unchanged
    assert sft_v2._neftune_alpha(noisy, "full", escalate=True) == 5.0
    assert sft_v2._neftune_alpha(SimpleNamespace(gradient_noise_scale=1.0), "full", escalate=True) == 1.0
    assert sft_v2._neftune_alpha(SimpleNamespace(gradient_noise_scale=None), "full", escalate=True) == 1.0
    assert sft_v2._neftune_alpha(None, "full", escalate=True) == 1.0
    assert sft_v2._neftune_alpha(noisy, "lora", escalate=True) is None
    monkeypatch.setenv("FORGE_V2_NEFTUNE", "0")
    assert sft_v2._neftune_alpha(noisy, "full", escalate=True) is None


def test_baseline_summary_parses_optional_gradient_noise_scale():
    payload = {
        "task_type": "instruct",
        "dataset": {"total_tokens": 1000, "num_records": 10, "near_duplicate_rate": 0.3,
                    "seq_length_distribution": {"p95": 100, "p99": 200, "max": 300}},
        "weights": {}, "training": {"init_loss": 2.0, "gradient_noise_scale": 1.7},
    }
    summary = load_baseline_summary(json.dumps(payload), expected_task_type="InstructTextTask")
    assert summary.gradient_noise_scale == 1.7
    assert telemetry_fields(summary)["baseline_gradient_noise_scale"] == 1.7
    assert hygiene.dedup_triggered(summary)
    del payload["training"]["gradient_noise_scale"]
    assert load_baseline_summary(json.dumps(payload), expected_task_type="instruct").gradient_noise_scale is None
    for bad in ("nan", -1.0, True, "x", None):
        payload["training"]["gradient_noise_scale"] = bad
        assert load_baseline_summary(json.dumps(payload), expected_task_type="instruct").gradient_noise_scale is None


# ---- knob off: existing behaviour ------------------------------------------------------------

def test_knob_off_by_default_and_gated_to_the_field_route(monkeypatch):
    monkeypatch.delenv("FORGE_V2_FIELD_HYGIENE", raising=False)
    assert not hygiene.enabled()
    tree = ast.parse(inspect.getsource(sft_v2.run))
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "field_hygiene" for t in n.targets))
    expr = compile(ast.Expression(gate.value), "<actual source gate>", "eval")
    never = SimpleNamespace(enabled=lambda: pytest.fail("hygiene evaluated off the field route"))
    monkeypatch.setenv("FORGE_V2_FIELD_HYGIENE", "1")
    assert eval(expr, {"field": False, "hygiene": never}) is False
    assert eval(expr, {"field": True, "hygiene": hygiene}) is True
    monkeypatch.delenv("FORGE_V2_FIELD_HYGIENE", raising=False)
    assert eval(expr, {"field": True, "hygiene": hygiene}) is False


def test_plain_split_and_collator_unchanged_when_off():
    rows = _corpus(300)
    train, dev = stratified_split(rows, 30, seed=7)
    assert len(dev) == 30 and len(train) == 270 and not ({id(r) for r in train} & {id(r) for r in dev})
    batch = PadCollator(0)(rows[:3])
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
