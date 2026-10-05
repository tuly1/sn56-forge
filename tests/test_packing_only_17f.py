"""CPU qualification of the scoped W1 extraction; no pretrained weights or data."""
from __future__ import annotations

import ast
import copy
import inspect
import sys
from types import SimpleNamespace as NS

import pytest
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

from forge.data import packing, tokenize
from forge.tasks import sft_v2


def llama(attn="sdpa", vocab=128256):
    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False,
        attn_implementation=attn)).eval()


def qwen2(attn="sdpa", vocab=151936):
    torch.manual_seed(17)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False,
        attn_implementation=attn)).eval()


def rows(n=8):
    result = []
    for i in range(n):
        ids = [1] + [5+i] * (3+i%5) + [17, 2]
        labels = ([-100, -100] + ids[2:]) if i % 2 else ids.copy()
        result.append(dict(input_ids=ids, labels=labels))
    return result


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os
    for name in list(os.environ):
        if name.startswith("FORGE_"):
            monkeypatch.delenv(name)
    torch.set_num_threads(2)


def plan(model, **kw):
    return packing.plan_field_packing(model, rows(), max_len=32, model_cap=256,
                                      pad_token_id=0, field_route=kw.pop("field_route", True), **kw)


def test_conditional_default_and_explicit_off(monkeypatch):
    m = llama()
    p = plan(m)
    assert p.mode == p.attn_implementation == "sdpa"
    assert p.stats["rows"] == 8 and p.n_blocks < 8
    assert "attention_mask" not in p.collator(p.features)
    monkeypatch.setenv(packing.ENV_MODE, "off")
    assert plan(m) is None


@pytest.mark.parametrize("case,reason", [
    ("nonfield", "not_field_route"), ("hygiene", "hygiene_not_supported"),
    ("rowloss", "row_loss_objective"), ("version", "transformers_version_unqualified"),
    ("vocab", "llama_scope_unqualified"), ("family", "llama_scope_unqualified"),
    ("cache", "cache_must_be_disabled"), ("backend", "sdpa_backend_required"),
    ("inner_backend", "sdpa_backend_required"),
    ("remote", "not_native_llama"), ("fa", "unsupported_mode"),
    ("invalid", "unsupported_mode"),
])
def test_unqualified_requests_skip_without_backend_or_naive_fallback(monkeypatch, case, reason):
    m = llama(); kw = {}; events = []
    monkeypatch.setenv(packing.ENV_MODE, "sdpa")
    monkeypatch.setenv("FORGE_V2_FIELD_TYPES", "llama,lfm2,qwen2")
    if case == "nonfield": kw["field_route"] = False
    if case == "hygiene": kw["field_hygiene"] = True
    if case == "rowloss": monkeypatch.setenv("FORGE_V2_ROW_LOSS", "1")
    if case == "version": monkeypatch.setattr(sys.modules["transformers"], "__version__", "5.12.0")
    if case == "vocab": m.config.vocab_size = 32000
    if case == "family": m.config.model_type = "lfm2"
    if case == "cache": m.config.use_cache = True
    if case == "backend": m.set_attn_implementation("eager")
    if case == "inner_backend":
        m.model.layers[0].self_attn.config = copy.deepcopy(m.config)
        m.model.layers[0].self_attn.config._attn_implementation = "eager"
    if case == "remote": m = NS(config=m.config, model=m.model)
    if case in {"fa", "invalid"}: monkeypatch.setenv(packing.ENV_MODE, case)
    before = m.config._attn_implementation
    assert plan(m, log=lambda n, **d: events.append((n,d)), **kw) is None
    assert m.config._attn_implementation == before
    assert events[-1][0] == "sft_v2_packing_skipped"
    assert events[-1][1]["mode"] == "off" and events[-1][1]["reason"].startswith(reason)


def test_naive_requires_explicit_opt_in(monkeypatch):
    m = llama(); assert plan(m).mode == "sdpa"
    monkeypatch.setenv(packing.ENV_MODE, "naive")
    p = plan(m)
    assert p.mode == "naive" and "attention_mask" in p.collator(p.features)
    assert m.config._attn_implementation == "sdpa"


def test_rows_and_supervised_targets_are_preserved_without_mutating_input():
    source = rows(19); before = copy.deepcopy(source)
    blocks = packing.pack_rows(source, 32, seed=7)
    assert sorted(i for b in blocks for i in b.rows) == list(range(19))
    assert blocks == packing.pack_rows(source, 32, seed=7)
    for b in blocks:
        assert b.n_tokens <= 32
        assert b.position_ids == [p for i in b.rows for p in range(len(source[i]["input_ids"]))]
        expected = [v for i in b.rows for v in source[i]["labels"][1:] if v != -100]
        assert [v for v in b.labels if v != -100] == expected
        assert sum(v == 2 for v in b.labels) == len(b.rows)
        assert all(v == -100 for p,v in zip(b.position_ids,b.labels) if p == 0)
    assert source == before
    batch = packing.PackedPadCollator(0, attention_mask=False)([b.feature() for b in blocks])
    for k,b in enumerate(blocks):
        assert (batch["labels"][k,b.n_tokens:] == -100).all()


def test_sdpa_native_llama_matches_unpacked_token_losses_and_isolates_prefix():
    m = llama(); source = rows(); p = plan(m)
    blocks = packing.pack_rows(source, 32, seed=7)
    def ce(logits, labels):
        labels = torch.as_tensor(labels)[1:]
        losses = torch.nn.functional.cross_entropy(logits[:-1].float(), labels, reduction="none", ignore_index=-100)
        return losses[labels != -100]
    with torch.no_grad():
        expected = [ce(m(input_ids=torch.tensor([r["input_ids"]]), use_cache=False).logits[0], r["labels"]) for r in source]
        batch = p.collator(p.features)
        out = m(**batch)
        got = torch.cat([ce(out.logits[k], batch["labels"][k]) for k in range(len(blocks))])
        ref = torch.cat([expected[i] for b in blocks for i in b.rows])
        torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)
        assert out.loss.item() == pytest.approx(ref.mean().item(), abs=2e-6)
        # Mutate every token of the preceding row; all later-row logits must stay fixed.
        k = next(i for i,b in enumerate(blocks) if len(b.rows)>1)
        boundary = len(source[blocks[k].rows[0]]["input_ids"])
        changed = {k:v.clone() for k,v in batch.items()}
        changed["input_ids"][k,:boundary] = 31
        later = m(**changed).logits
        torch.testing.assert_close(later[k,boundary:blocks[k].n_tokens],
                                   out.logits[k,boundary:blocks[k].n_tokens], atol=1e-7, rtol=1e-7)
        # The same perturbation is intentionally observable in explicitly naive mode.
        naive = packing.PackedPadCollator(0, attention_mask=True)(p.features)
        first = m(**naive).logits
        naive["input_ids"][k,:boundary] = 31
        second = m(**naive).logits
        assert float((first[k,boundary:blocks[k].n_tokens]-second[k,boundary:blocks[k].n_tokens]).abs().max()) > 1e-5


def test_real_memory_probe_uses_reset_positions_and_backpropagates():
    m = llama(vocab=257); seen=[]
    handle = m.register_forward_pre_hook(lambda mod,args,kw: seen.append(kw), with_kwargs=True)
    try:
        assert sft_v2.memory_probe(m, micro=2, max_len=16, vocab=257, autocast=False, packed_mode="sdpa")
    finally:
        handle.remove()
    batch = seen[0]
    assert "attention_mask" not in batch
    assert batch["position_ids"][0].tolist() == list(range(8))*2
    assert (batch["labels"][batch["position_ids"] == 0] == -100).all()
    assert all(p.grad is None for p in m.parameters())


def test_actual_trainer_counts_final_partial_accumulation(tmp_path):
    m=llama(vocab=257)
    # Five blocks / micro2 / accumulation2 => three batches => two updates, not one.
    features=[b.feature() for b in packing.pack_rows(rows(5), 10)]
    assert len(features) == 5
    args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=1,
        optim="adamw_torch", learning_rate=1e-4, report_to=[], save_strategy="no",
        remove_unused_columns=False, disable_tqdm=True)
    trainer=Trainer(model=m,args=args,train_dataset=features,
                    data_collator=packing.PackedPadCollator(0,attention_mask=False))
    trainer.train()
    assert trainer.state.global_step == packing.optimizer_steps(5,2,2) == 2


@pytest.mark.parametrize("packed", [True, False])
def test_actual_handler_training_timing_and_dev_keep_their_respective_views(monkeypatch,tmp_path,packed):
    from forge.tasks import fallback
    from forge.tuning import selection
    from forge.data.schema import InstructColumns
    m=llama(); source=rows(34); events=[]; creations=[]; dev_batches=[]; timing=[]
    if not packed: monkeypatch.setenv(packing.ENV_MODE,"off")
    for name in ("FORGE_V2_FIELD_LR_CALIBRATION","FORGE_V2_FIELD_SWEEP","FORGE_V2_FIELD_BF16",
                 "FORGE_V2_ROW_LOSS","FORGE_V2_FIELD_HYGIENE","FORGE_DEV_PASS"):
        monkeypatch.setenv(name,"0")
    monkeypatch.setattr(sft_v2,"_liger_ok",lambda m:False)
    monkeypatch.setattr(sft_v2,"_bnb_available",lambda:False)
    monkeypatch.setattr(sft_v2,"workdir",lambda spec:str(tmp_path/"work"))
    monkeypatch.setattr(sft_v2,"_layer_weight_rms",lambda m:1.)
    monkeypatch.setattr(sft_v2.prompts,"build_instruct_examples",lambda *a:[])
    monkeypatch.setattr(sft_v2.tokenize,"tokenize_instruct",lambda *a:copy.deepcopy(source))
    monkeypatch.setattr(sft_v2,"adaptive_max_len",lambda *a,**k:32)
    monkeypatch.setattr(sft_v2,"dev_size_for",lambda *a,**k:2)
    monkeypatch.setattr(sft_v2,"stratified_split",lambda r,*a,**k:(r[:-2],r[-2:]))
    monkeypatch.setattr(sft_v2,"choose_geometry",lambda **kw:sft_v2.Geometry(2,2,kw["max_len"],False,False,"adamw_torch"))
    monkeypatch.setattr(sft_v2,"_event_and_print",lambda n,**d:events.append((n,d)))
    for name in ("set_meta","eval_point","write_into"):
        monkeypatch.setattr(sft_v2.telemetry,name,lambda *a,**k:None)
    monkeypatch.setattr(sft_v2.telemetry,"make_trainer_callback",lambda *a:object())
    monkeypatch.setattr(sft_v2,"write_artifact_truth",lambda *a,**k:None)
    monkeypatch.setattr(sft_v2,"save_adapter",lambda *a,**k:None)
    def dev(model,rs,collator,**kw):
        dev_batches.append(collator(rs)); return .5,[]
    monkeypatch.setattr(selection,"per_example_dev_loss",dev)
    monkeypatch.setattr(selection,"available_ram",lambda:10**12)
    def measure(model,batches,**kw):
        timing.extend(batches);return 1.,[1.]
    monkeypatch.setattr(sft_v2.lr_probe,"measure_step_time",measure)
    monkeypatch.setattr(sft_v2,"make_prompt_weight_trainer",lambda *a,**k:pytest.fail("packing must not select W2"))
    class Args:
        def __init__(self,**kw):self.__dict__.update(kw)
    monkeypatch.setattr(sft_v2,"compatible_dataclass_kwargs",lambda cls,kw,**_:kw)
    monkeypatch.setattr(sys.modules["transformers"],"TrainingArguments",Args)
    class RecordingTrainer:
        def __init__(self,model,args,train_dataset,data_collator,callbacks=()):
            self.model,self.args,self.train_dataset,self.data_collator=model,args,train_dataset,data_collator
            self.callbacks=callbacks;self.state=NS(global_step=0,max_steps=100)
            self.optimizer=object();self.lr_scheduler=object();creations.append(self)
        def get_train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset,batch_size=2,collate_fn=self.data_collator)
        def train(self):
            self.state.global_step=1
            self.callbacks[0].on_step_end(self.args,self.state,NS(should_training_stop=False),model=self.model)
    monkeypatch.setattr(sys.modules["transformers"],"Trainer",RecordingTrainer)
    spec=NS(output_dir=str(tmp_path/"out"),task_id="synthetic",instruct=InstructColumns("instruction","output"))
    deadline=NS(remaining_hard=lambda:1800.,export_reserve_s=180.)
    sft_v2.run(spec,deadline,source,NS(model=m),NS(pad_token_id=0),baseline_summary=None,
               params_b=1.24,per_gpu_gb=80.,strategy_override="full",field_route=True)
    assert len(creations)==2 and timing and dev_batches
    probe,train=creations
    assert type(probe) is type(train) is RecordingTrainer
    assert probe.data_collator is train.data_collator
    assert ("position_ids" in timing[0]) is packed
    assert ("attention_mask" in timing[0]) is not packed
    assert all(set(b)=={"input_ids","labels","attention_mask"} for b in dev_batches)
    assert all(len(b["input_ids"])==2 for b in dev_batches)
    n=len(train.train_dataset)
    expected=packing.optimizer_steps(n,2,2) if packed else n//4
    start=next(d for name,d in events if name=="sft_v2_train_start")
    assert start["steps_per_epoch"]==expected
    assert (n<32) is packed
    assert any(name=="sft_v2_packing" for name,d in events) is packed


def test_no_extra_modules_or_changed_experimental_defaults():
    assert not sft_v2._field_lr_calibration()
    assert not sft_v2.hygiene.enabled()
    source=inspect.getsource(packing)
    assert "loss_weights" not in source and "PromptWeightWrapper" not in source
    assert "rebuild_plan" not in source and "field_survival" not in source
    tree=ast.parse(inspect.getsource(sft_v2.run))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                    and n.value.func.id=="make_prompt_weight_trainer")
    parent=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and assignment in n.body)
    assert ast.unparse(parent.test)=="field_hygiene and hyg['plw'] > 0"


def test_native_qwen2_qualifies_with_its_own_vocabulary(monkeypatch):
    """2026-09-17: packing extends to native Qwen2 (same position-id based block-diagonal masks); Qwen2 with a
    foreign vocabulary or a Llama with Qwen2's vocabulary stays out of scope."""
    monkeypatch.setenv(packing.ENV_MODE, "sdpa")
    assert packing._eligibility(qwen2(), field_route=True, field_hygiene=False) is None
    assert packing._eligibility(qwen2(vocab=128256), field_route=True, field_hygiene=False) == "llama_scope_unqualified"
    assert packing._eligibility(llama(vocab=151936), field_route=True, field_hygiene=False) == "llama_scope_unqualified"
    p = packing.plan_field_packing(qwen2(), rows(), max_len=64, pad_token_id=0, model_cap=256, field_route=True)
    assert p is not None and p.mode == "sdpa" and p.stats["rows"] == 8
