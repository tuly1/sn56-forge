"""Damaged Gemma2 pack-speed candidate (research-gemma-pack-speed-20260927, research/20260927-gemma-pack-speed/DESIGN.md).

On the damaged Gemma2 champion route (c292732) packed training batches take varlen attention (forge.data.packing: rows
from the reset position ids, PyTorch's FA2 kernel on CUDA, a per-row sdpa loop on CPU, no 4-D mask) and the 32 blocks of
a step run as one micro-batch. These CPU tests prove, with a tiny Gemma2:
- the row offsets equal transformers' packed-sequence detection;
- varlen attention equals the c292732 sdpa block-diagonal path on every real token (logits, per-token losses, parameter
  gradients), rows stay isolated, padded (dev) batches take sdpa unchanged, gradient checkpointing works;
- the switch-on gates, the CUDA self-check's comparison logic and config serialisation;
- the sft_v2.run wiring: default, off-switch (== c292732), the two component switches, env geometry precedence and
  unchanged out-of-scope runs.
The FA2 CUDA kernel itself cannot run here (GPU-only residual risk; the route verifies it on the GPU before use).
"""
from __future__ import annotations

import copy
import json
import os

import pytest
import torch
import transformers

from forge.data import packing
from forge.tasks import sft_v2

import test_gemma2_champion_recipe as base  # the c292732 harness (tiny Gemma2, sft_v2.run doubles)

qualified = pytest.mark.skipif(transformers.__version__ != packing.QUALIFIED_TRANSFORMERS,
                               reason="packing is qualified on the image's transformers only")
MASTER = "FORGE_V2_GEMMA2_PACK_SPEED"
VARLEN = "FORGE_V2_GEMMA2_VARLEN"
ONE_MICRO = "FORGE_V2_GEMMA2_ONE_MICRO"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_"):
            monkeypatch.delenv(k, raising=False)
    torch.set_num_threads(2)
    packing._SEGMENT_CACHE.clear()


def _model(sliding_window=4096, heads=4, kv_heads=2, layers=2):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(23)
    m = Gemma2ForCausalLM(Gemma2Config(
        vocab_size=base.GEMMA_VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=8, query_pre_attn_scalar=8,
        max_position_embeddings=8192, sliding_window=sliding_window, bos_token_id=2, eos_token_id=1, pad_token_id=0,
        use_cache=False, attn_implementation="sdpa")).eval()
    setattr(m, packing.GEMMA2_ROUTE_ATTR, True)
    return m


def _pair(**kw):
    """(c292732 sdpa model, the same weights switched to varlen)."""
    ref = _model(**kw)
    var = copy.deepcopy(ref)
    assert packing.enable_gemma2_varlen(var, pack_len=48) == "segment_loop"
    return ref, var


def _source(n=11):
    rows = []
    for i in range(n):
        length = 3 + (i * 5) % 13
        ids = [2] + [7 + (i * 3 + j) % 90 for j in range(length)] + [1]
        rows.append(dict(input_ids=ids, labels=([-100] * (length // 2) + ids[length // 2:])))
    return rows


def _ce(logits, labels):
    labels = torch.as_tensor(labels)[1:]
    losses = torch.nn.functional.cross_entropy(logits[:-1].float(), labels, reduction="none", ignore_index=-100)
    return losses[labels != -100]


# ---- row offsets --------------------------------------------------------------------------------------------------

def test_constants_and_default():
    assert sft_v2.GEMMA2_PACK_SPEED_DEFAULT == "1"
    assert sft_v2.GEMMA2_ONE_MICRO_MAX_TOKENS == 131072
    assert packing.VARLEN_ATTN == "forge_gemma2_varlen" and packing.VARLEN_BACKENDS == ("flash", "segment_loop")
    assert sft_v2.gemma2_one_micro(32, 3136) == (32, 1)  # phase-w: 32 x 3136 = 100,352 tokens
    assert sft_v2.gemma2_one_micro(32, 3072) == (32, 1)  # the champion's 32 x 3072
    assert sft_v2.gemma2_one_micro(48, 2048) == (48, 1)
    assert sft_v2.gemma2_one_micro(31, 4096) == (31, 1)
    assert sft_v2.gemma2_one_micro(40, 4096) == (32, 2)


def test_segment_offsets_match_transformers_packed_detection():
    from transformers.masking_utils import find_packed_sequence_indices

    g = torch.Generator().manual_seed(5)
    for trial in range(40):
        batch, width = 1 + trial % 4, 5 + trial
        rows = []
        for _ in range(batch):
            pos, left = [], width
            while left > 0:
                n = int(torch.randint(1, left + 1, (1,), generator=g))
                pos += list(range(n))
                left -= n
            if trial % 3 == 0:  # the c292732 collator's pad tail: one position-0 token per pad
                pos[-3:] = [0, 0, 0]
            rows.append(pos)
        pid = torch.tensor(rows)
        cu, max_len = packing.segment_offsets(pid, batch, width)
        seg = find_packed_sequence_indices(pid)
        seg = torch.zeros_like(pid) if seg is None else seg
        # transformers' sequence index + a new index per batch row == one segment per offset interval
        flat = (seg + torch.arange(batch).unsqueeze(1) * (width + 1)).reshape(-1)
        want = torch.nonzero(torch.cat([torch.ones(1, dtype=torch.bool), flat[1:] != flat[:-1]])).flatten()
        assert cu.dtype == torch.int32 and cu[0] == 0 and cu[-1] == batch * width
        assert cu[:-1].tolist() == want.tolist()
        assert max_len == int((cu[1:] - cu[:-1]).max())
    # 1-D and broadcast position ids
    cu, m = packing.segment_offsets(torch.tensor([0, 1, 2, 0, 1]), 2, 5)
    assert cu.tolist() == [0, 3, 5, 8, 10] and m == 3
    with pytest.raises(ValueError):
        packing.segment_offsets(torch.zeros(1, 4, dtype=torch.long), 1, 5)


def test_segment_cache_is_per_position_tensor():
    a = torch.tensor([[0, 1, 2, 0, 1]])
    first = packing._segments(a, 1, 5)
    assert packing._segments(a, 1, 5) is first  # same object: one computation per forward
    b = torch.tensor([[0, 1, 0, 1, 2]])  # a new batch is a new object
    assert packing._segments(b, 1, 5)[0].tolist() == [0, 2, 5]
    a[0, 3:] = torch.tensor([3, 4])  # in-place change bumps the version
    assert packing._segments(a, 1, 5)[0].tolist() == [0, 5]


def test_pad_as_row_collator_only_changes_pad_positions():
    feats = [b.feature() for b in packing.pack_rows(_source(), 48, seed=7)]
    old = packing.PackedPadCollator(0, attention_mask=False)(feats)
    new = packing.PackedPadCollator(0, attention_mask=False, pad_as_row=True)(feats)
    assert "attention_mask" not in new and set(new) == set(old)
    assert torch.equal(new["input_ids"], old["input_ids"]) and torch.equal(new["labels"], old["labels"])
    for k, f in enumerate(feats):
        n = len(f["input_ids"])
        assert new["position_ids"][k, :n].tolist() == old["position_ids"][k, :n].tolist() == f["position_ids"]
        assert new["position_ids"][k, n:].tolist() == list(range(new["input_ids"].shape[1] - n))
    assert packing.PackedPadCollator(0).pad_as_row is False  # every existing caller keeps one row per pad token


# ---- the attention function ---------------------------------------------------------------------------------------

@qualified
@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (4, 4), (4, 1)])
def test_varlen_attention_equals_the_sdpa_block_diagonal_mask(heads, kv_heads):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import (and_masks, causal_mask_function, find_packed_sequence_indices,
                                            packed_sequence_mask_function, sdpa_mask)

    g = torch.Generator().manual_seed(9)
    batch, width, dim = 3, 29, 8
    pos = torch.tensor([list(range(10)) + list(range(12)) + list(range(7)),
                        list(range(29)),
                        list(range(4)) + list(range(20)) + [0] * 5])
    q = torch.randn(batch, heads, width, dim, generator=g)
    k = torch.randn(batch, kv_heads, width, dim, generator=g)
    v = torch.randn(batch, kv_heads, width, dim, generator=g)
    module = torch.nn.Module()
    module.num_key_value_groups = heads // kv_heads
    module.is_causal = True
    seq = find_packed_sequence_indices(pos)
    mask = sdpa_mask(batch_size=batch, cache_position=torch.arange(width), kv_length=width, q_length=width,
                     q_offset=0, kv_offset=0,
                     mask_function=and_masks(causal_mask_function, packed_sequence_mask_function(seq)),
                     allow_is_causal_skip=False)
    want, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=0.3)
    got, _ = packing.varlen_attention_forward(module, q, k, v, None, scaling=0.3, position_ids=pos, softcap=50.0,
                                              sliding_window=4096)
    assert got.shape == want.shape == (batch, width, heads, dim)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    # a padded batch (dev evaluation) delegates to transformers' sdpa function exactly
    got_m, _ = packing.varlen_attention_forward(module, q, k, v, mask, scaling=0.3, position_ids=pos)
    assert torch.equal(got_m, want)
    # a window shorter than a row cannot be honoured by varlen: refuse rather than silently widen it
    with pytest.raises(RuntimeError, match="sliding window"):
        packing.varlen_attention_forward(module, q, k, v, None, scaling=0.3, position_ids=pos, sliding_window=8)


def test_varlen_mask_returns_none_without_padding_and_the_sdpa_mask_with_it():
    from transformers.masking_utils import causal_mask_function, sdpa_mask

    assert packing.varlen_mask(batch_size=1, q_length=4, kv_length=4, attention_mask=None) is None
    pad = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    kw = dict(batch_size=1, cache_position=torch.arange(4), q_length=4, kv_length=4, q_offset=0, kv_offset=0,
              mask_function=causal_mask_function)
    assert torch.equal(packing.varlen_mask(attention_mask=pad, **kw), sdpa_mask(attention_mask=pad, **kw))


# ---- model level ---------------------------------------------------------------------------------------------------

@qualified
def test_model_varlen_equals_c292732_sdpa_packing_logits_losses_and_gradients():
    ref, var = _pair()
    source = _source()
    blocks = packing.pack_rows(source, 48, seed=7)
    assert any(len(b.rows) > 2 for b in blocks)
    feats = [b.feature() for b in blocks]
    old = packing.PackedPadCollator(0, attention_mask=False)(feats)  # c292732's batch
    new = packing.PackedPadCollator(0, attention_mask=False, pad_as_row=True)(feats)  # the candidate's batch
    out_ref, out_var = ref(**old), var(**new)
    real = old["labels"] != -100
    for k, b in enumerate(blocks):  # every real token's logits (pads may differ: label -100, after the rows)
        n = b.n_tokens
        torch.testing.assert_close(out_var.logits[k, :n], out_ref.logits[k, :n], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(out_var.loss, out_ref.loss, atol=1e-6, rtol=1e-6)
    assert int(real.sum()) > 0
    # per-token losses equal the unpacked rows
    with torch.no_grad():
        expected = [_ce(ref(input_ids=torch.tensor([r["input_ids"]])).logits[0], r["labels"]) for r in source]
        got = torch.cat([_ce(out_var.logits[k].detach(), new["labels"][k]) for k in range(len(blocks))])
        torch.testing.assert_close(got, torch.cat([expected[i] for b in blocks for i in b.rows]), atol=2e-5, rtol=2e-5)
    # parameter gradients of the packed loss
    out_ref.loss.backward()
    out_var.loss.backward()
    for (name, a), (_, b) in zip(ref.named_parameters(), var.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, atol=2e-5, rtol=1e-4, msg=name)


@qualified
def test_model_varlen_isolates_rows_and_padded_batches_take_sdpa():
    ref, var = _pair()
    blocks = packing.pack_rows(_source(), 48, seed=7)
    batch = packing.PackedPadCollator(0, attention_mask=False, pad_as_row=True)([b.feature() for b in blocks])
    k = next(i for i, b in enumerate(blocks) if len(b.rows) > 1)
    boundary = len(_source()[blocks[k].rows[0]]["input_ids"])
    with torch.no_grad():
        first = var(**batch).logits
        changed = {kk: vv.clone() for kk, vv in batch.items()}
        changed["input_ids"][k, :boundary] = 31
        second = var(**changed).logits
        torch.testing.assert_close(second[k, boundary:blocks[k].n_tokens], first[k, boundary:blocks[k].n_tokens],
                                   atol=1e-6, rtol=1e-6)
        assert float((second[k, :boundary] - first[k, :boundary]).abs().max()) > 1e-4  # the mutation itself is seen
        # unpacked right-padded dev batch with a 2-D mask: the sdpa path, identical outputs
        ids = torch.tensor([[2, 9, 10, 11, 1, 0, 0], [2, 12, 13, 14, 15, 16, 1]])
        am = (ids != 0).long()
        am[1] = 1
        torch.testing.assert_close(var(input_ids=ids, attention_mask=am).logits, ref(input_ids=ids, attention_mask=am).logits,
                                   atol=0, rtol=0)
        # plain unpacked rows without a mask: varlen with one row per batch row == sdpa is_causal
        plain = torch.tensor([[2, 9, 10, 11, 1], [2, 12, 13, 14, 1]])
        torch.testing.assert_close(var(input_ids=plain).logits, ref(input_ids=plain).logits, atol=2e-5, rtol=2e-5)


@qualified
def test_model_varlen_under_gradient_checkpointing():
    ref, var = _pair()
    for m in (ref, var):
        m.train()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    feats = [b.feature() for b in packing.pack_rows(_source(), 48, seed=7)]
    ref(**packing.PackedPadCollator(0, attention_mask=False)(feats)).loss.backward()
    var(**packing.PackedPadCollator(0, attention_mask=False, pad_as_row=True)(feats)).loss.backward()
    for (name, a), (_, b) in zip(ref.named_parameters(), var.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, atol=2e-5, rtol=1e-4, msg=name)


@qualified
def test_enable_gates_serialisation_and_disable(tmp_path):
    logs = []
    m = _model()
    m_flagless = _model()
    delattr(m_flagless, packing.GEMMA2_ROUTE_ATTR)
    assert packing.enable_gemma2_varlen(m_flagless, pack_len=48, log=lambda n, **d: logs.append(d)) is None
    assert logs[-1]["reason"] == "not_gemma2_route" and m_flagless.config._attn_implementation == "sdpa"
    assert packing.enable_gemma2_varlen(_model(sliding_window=32), pack_len=48, log=lambda n, **d: logs.append(d)) is None
    assert logs[-1]["reason"] == "pack_len_exceeds_sliding_window:48>32"
    eager = _model()
    for layer in eager.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    assert packing.enable_gemma2_varlen(eager, pack_len=48) is None
    assert packing.enable_gemma2_varlen(m, pack_len=48, log=lambda n, **d: logs.append((n, d))) == "segment_loop"
    name, ev = logs[-1]
    assert name == "sft_v2_gemma2_varlen" and ev["enabled"] and ev["backend"] == "segment_loop"
    assert ev["attn_implementation"] == packing.VARLEN_ATTN and ev["softcap_applied"] is False
    assert all(l.self_attn.config._attn_implementation == packing.VARLEN_ATTN for l in m.model.layers)
    assert all(getattr(l.self_attn, packing.VARLEN_BACKEND_ATTR) == "segment_loop" for l in m.model.layers)
    # the exported config never names the research attention (the evaluator loads it with its own default, sdpa)
    m.save_pretrained(tmp_path / "exp")
    cfg = json.loads((tmp_path / "exp" / "config.json").read_text())
    assert packing.VARLEN_ATTN not in json.dumps(cfg) and "_attn_implementation" not in cfg
    packing.disable_gemma2_varlen(m)
    assert m.config._attn_implementation == "sdpa"
    assert not any(hasattr(l.self_attn, packing.VARLEN_BACKEND_ATTR) for l in m.model.layers)


def test_cuda_self_check_logic_on_cpu():
    # the comparison the CUDA route runs before using flash: the reference itself agrees ...
    ok = packing.flash_varlen_self_check("cpu", torch.float32, 4, 16, 0.25,
                                         impl=lambda q, k, v, cu, m, d, s: packing._loop_varlen(q, k, v, cu, d, s))
    assert ok is None

    # ... a kernel that attends across rows (plain causal over the whole stream) is caught ...
    def leaky(q, k, v, cu, m, d, s):
        o = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                             v.transpose(0, 1)[None], is_causal=True, scale=s)
        return o[0].transpose(0, 1)

    assert packing.flash_varlen_self_check("cpu", torch.float32, 4, 16, 0.25, impl=leaky).startswith("out_mismatch")

    # ... a wrong scale is caught, and a kernel error is a reason, not an exception
    def wrong_scale(q, k, v, cu, m, d, s):
        return packing._loop_varlen(q, k, v, cu, d, 2 * s)

    assert "mismatch" in packing.flash_varlen_self_check("cpu", torch.float32, 4, 16, 0.25, impl=wrong_scale)

    def boom(*a):
        raise RuntimeError("no kernel image")

    assert packing.flash_varlen_self_check("cpu", torch.float32, 4, 16, 0.25, impl=boom) == "RuntimeError: no kernel image"
    # the flash op the CUDA path calls exists with the argument layout used (torch's nested-tensor SDPA calls it so)
    schema = str(torch.ops.aten._flash_attention_forward.default._schema)
    assert schema.startswith("aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? cum_seq_q, "
                             "Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, "
                             "bool return_debug_mask, *, float? scale=None")


# ---- sft_v2.run wiring ---------------------------------------------------------------------------------------------

def _speed(events):
    return base._ev(events, "sft_v2_gemma2_pack_speed")


def test_run_default_takes_varlen_and_one_micro(monkeypatch, tmp_path):
    ev, tr = base._run(monkeypatch, tmp_path)
    v = base._ev(ev, "sft_v2_gemma2_varlen")
    assert len(v) == 1 and v[0]["enabled"] and v[0]["backend"] == "segment_loop" and v[0]["pack_len"] == 3072
    s = _speed(ev)
    assert s == [{"varlen_backend": "segment_loop", "one_micro": True, "micro": 32, "grad_accum": 1,
                  "blocks_per_step": 32, "planned_micro": 8, "planned_grad_accum": 4}]
    t = base._trace(ev, tr)
    assert (t["micro"], t["accum"]) == (32, 1)
    coll = tr[-1].data_collator
    assert isinstance(coll, packing.PackedPadCollator) and coll.pad_as_row and not coll.attention_mask
    assert tr[0].data_collator is coll  # the timing / LR probe measures the same batches
    model = tr[-1].model
    assert model.config._attn_implementation == packing.VARLEN_ATTN
    # everything else in the executed recipe is as c292732 (LR, planner floor, schedule, NEFTune)
    r = base._ev(ev, "sft_v2_gemma2_champion_route")[0]
    assert r["lr"] == base.EXECUTED_LR and r["blocks_per_step"] == 32 and t["plan"]["epochs"] == 1.3
    assert (t["neftune"], t["wd"], t["clip"]) == (1.0, 0.0, 1.0)


def _c292732_view(events):
    return [(n, d) for n, d in base._stable(events) if n not in ("sft_v2_gemma2_varlen", "sft_v2_gemma2_pack_speed")]


def test_off_switch_is_c292732(monkeypatch, tmp_path):
    ev_off, tr_off = base._run(monkeypatch, tmp_path / "off", env={MASTER: "0"})
    assert base._ev(ev_off, "sft_v2_gemma2_varlen") == [] == _speed(ev_off)
    t = base._trace(ev_off, tr_off)
    assert (t["micro"], t["accum"]) == (8, 4)
    assert tr_off[-1].model.config._attn_implementation == "sdpa"
    assert tr_off[-1].data_collator.pad_as_row is False
    r = base._ev(ev_off, "sft_v2_gemma2_champion_route")[0]
    assert (r["micro"], r["grad_accum"], r["blocks_per_step"]) == (8, 4, 32)
    # the validator's no-env case with the source default off is the same run
    monkeypatch.delenv(MASTER, raising=False)
    monkeypatch.setattr(sft_v2, "GEMMA2_PACK_SPEED_DEFAULT", "0")
    ev_src, tr_src = base._run(monkeypatch, tmp_path / "src")
    assert base._stable(ev_src) == base._stable(ev_off) and base._trace(ev_src, tr_src) == t
    # and a default run differs only in the pack-speed events, the geometry and the attention/collator
    monkeypatch.setattr(sft_v2, "GEMMA2_PACK_SPEED_DEFAULT", "1")
    ev_on, tr_on = base._run(monkeypatch, tmp_path / "on")
    on = base._trace(ev_on, tr_on)
    for key in ("field_lr", "override", "sweep_gate_lr", "train_start_lr", "trainer_lr", "trainer_warmup",
                "trainer_epochs", "sched", "neftune", "wd", "clip", "packing", "skipped"):
        assert on[key] == t[key], key
    assert [n for n, _ in _c292732_view(ev_on)] == [n for n, _ in base._stable(ev_off)]


@pytest.mark.parametrize("varlen,one_micro,want_backend,want_geo", [
    ("0", None, None, (8, 4)),              # varlen off: one-micro follows it (c292732 geometry)
    ("0", "1", None, (32, 1)),              # component arm: sdpa mask at one micro-batch
    ("1", "0", "segment_loop", (8, 4)),     # component arm: varlen at c292732's 8 x 4
    ("1", "1", "segment_loop", (32, 1)),
])
def test_component_switches(monkeypatch, tmp_path, varlen, one_micro, want_backend, want_geo):
    env = {VARLEN: varlen}
    if one_micro is not None:
        env[ONE_MICRO] = one_micro
    ev, tr = base._run(monkeypatch, tmp_path, env=env)
    s = _speed(ev)[0]
    assert s["varlen_backend"] == want_backend and (s["micro"], s["grad_accum"]) == want_geo
    t = base._trace(ev, tr)
    assert (t["micro"], t["accum"]) == want_geo
    assert (tr[-1].model.config._attn_implementation == packing.VARLEN_ATTN) == (want_backend is not None)
    assert tr[-1].data_collator.pad_as_row == (want_backend is not None)
    assert base._ev(ev, "sft_v2_gemma2_varlen") == ([] if varlen == "0" else base._ev(ev, "sft_v2_gemma2_varlen"))


def test_failed_gpu_self_check_keeps_c292732(monkeypatch, tmp_path):
    # enable_gemma2_varlen returns None when the CUDA self-check (or the switch) fails: the run is c292732's
    monkeypatch.setattr(packing, "enable_gemma2_varlen", lambda model, pack_len, log=None: None)
    ev, tr = base._run(monkeypatch, tmp_path / "fail")
    s = _speed(ev)[0]
    assert s["varlen_backend"] is None and s["one_micro"] is False and (s["micro"], s["grad_accum"]) == (8, 4)
    ev_off, tr_off = base._run(monkeypatch, tmp_path / "off", env={MASTER: "0"})
    assert _c292732_view(ev) == base._stable(ev_off) and base._trace(ev, tr) == base._trace(ev_off, tr_off)


def test_env_geometry_keeps_precedence(monkeypatch, tmp_path):
    ev, tr = base._run(monkeypatch, tmp_path, env={packing.ENV_EFF_BLOCKS: "12"})
    s = _speed(ev)[0]
    assert s["varlen_backend"] == "segment_loop" and s["one_micro"] is False
    t = base._trace(ev, tr)
    assert t["micro"] * t["accum"] == 12 and base._ev(ev, "sft_v2_gemma2_champion_route")[0]["blocks_per_step"] == 12


def _same_with_master(monkeypatch, tmp_path, **kw):
    ev_on, tr_on = base._run(monkeypatch, tmp_path / "on", **kw)
    ev_off, tr_off = base._run(monkeypatch, tmp_path / "off", env={MASTER: "0"}, **kw)
    assert base._stable(ev_on) == base._stable(ev_off)
    assert base._trace(ev_on, tr_on) == base._trace(ev_off, tr_off)
    assert base._ev(ev_on, "sft_v2_gemma2_varlen") == [] == _speed(ev_on)
    assert tr_on[-1].model.config._attn_implementation != packing.VARLEN_ATTN


def test_champion_recipe_off_is_untouched(monkeypatch, tmp_path):
    ev, tr = base._run(monkeypatch, tmp_path / "inc", env={base.SWITCH: "0"})
    assert base._ev(ev, "sft_v2_gemma2_varlen") == [] == _speed(ev)
    assert tr[-1].model.config._attn_implementation == "sdpa"


def test_packing_off_is_untouched(monkeypatch, tmp_path):
    ev, tr = base._run(monkeypatch, tmp_path / "po", env={"FORGE_V2_FIELD_PACKING": "off"})
    assert base._ev(ev, "sft_v2_gemma2_varlen") == [] == _speed(ev)


def test_run_non_damaged_gemma2_is_unchanged(monkeypatch, tmp_path):
    _same_with_master(monkeypatch, tmp_path, damaged=False)


@pytest.mark.parametrize("params_b", [1.9, 3.2])
def test_run_other_gemma2_sizes_are_unchanged(monkeypatch, tmp_path, params_b):
    _same_with_master(monkeypatch, tmp_path, params_b=params_b)


@pytest.mark.parametrize("name,vocab", [("Qwen2", base.QWEN_VOCAB), ("Llama", 128256), ("Qwen3", base.QWEN_VOCAB)])
def test_run_other_damaged_families_are_unchanged(monkeypatch, tmp_path, name, vocab):
    _same_with_master(monkeypatch, tmp_path, model=base._tiny(name, vocab), params_b=2.5)
