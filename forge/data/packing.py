"""Research-only FIELD Llama packing extracted from e2aad73 W1.

Default SDPA applies only to an already admitted FIELD native Llama (vocab128256)
with the pinned Transformers5.12.1 mask semantics and cache disabled. Unsupported
requests skip packing; no attention backend is switched, and no naive fallback
exists. Explicit ``naive`` retains cross-row attention and is opt-in only.

TRAIN rows stay whole, first labels are ignored at causal boundaries, position IDs
reset per source row, and EOS labels survive. Dev remains unpacked. Hygiene/row
loss are deliberately unsupported here: their integration belongs to other work.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

ENV_MODE = "FORGE_V2_FIELD_PACKING"
ENV_PACK_LEN = "FORGE_V2_FIELD_PACK_LEN"
ENV_EFF_BLOCKS = "FORGE_V2_FIELD_PACKED_EFF_BATCH"
# Set (True) on a model instance by sft_v2 only on the damaged Gemma2 champion route (sft_v2.GEMMA2_CHAMPION_RECIPE_DEFAULT):
# it admits native Gemma2 (vocab 256000) to the sdpa packing below for that run; every other Gemma2 stays unpacked.
GEMMA2_ROUTE_ATTR = "_forge_gemma2_champion_packing"
QUALIFIED_TRANSFORMERS = "5.12.1"
IGNORE = -100



@dataclass(frozen=True)
class Block:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    rows: tuple[int, ...]  # indices into the packed row list, in block order

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    def feature(self) -> dict[str, Any]:
        """The Trainer dataset row (``length`` feeds group_by_length)."""
        result = {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "position_ids": list(self.position_ids),
            "length": len(self.input_ids),
            "n_rows": len(self.rows),
        }
        return result


def _build_block(rows: list[dict[str, list[int]]], group: list[int]) -> Block:
    ids: list[int] = []
    labels: list[int] = []
    pos: list[int] = []
    for i in group:
        row_ids = list(rows[i]["input_ids"])
        row_labels = list(rows[i]["labels"])
        if len(row_labels) != len(row_ids):
            raise ValueError("Packed IDs/labels length mismatch")
        if row_labels:
            row_labels[0] = IGNORE
        ids.extend(row_ids)
        labels.extend(row_labels)
        pos.extend(range(len(row_ids)))
    return Block(ids, labels, pos, tuple(group))


def pack_rows(rows: list[dict[str, list[int]]], pack_len: int, *, seed: int = 7) -> list[Block]:
    """Greedy first-fit packing into blocks of at most ``pack_len`` tokens.

    Rows are visited in a seeded shuffled order (blocks mix lengths and the
    composition is reproducible) and never split; a row longer than the budget is
    a block of its own. The blocks are shuffled with the same seed.
    """
    if pack_len <= 0:
        raise ValueError("pack_len must be positive")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i in order:
        n = len(rows[i]["input_ids"])
        if current and used + n > pack_len:
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += n
    if current:
        groups.append(current)
    random.Random(seed + 1).shuffle(groups)
    return [_build_block(rows, g) for g in groups]


def block_stats(blocks: list[Block], pack_len: int) -> dict[str, Any]:
    n_blocks = len(blocks)
    n_rows = sum(len(b.rows) for b in blocks)
    n_tokens = sum(b.n_tokens for b in blocks)
    if n_blocks == 0:
        return {"blocks": 0, "rows": 0, "tokens": 0, "mean_rows_per_block": 0.0, "mean_tokens_per_block": 0.0,
                "max_rows_per_block": 0, "fill": 0.0, "pack_len": pack_len}
    return {
        "blocks": n_blocks,
        "rows": n_rows,
        "tokens": n_tokens,
        "mean_rows_per_block": round(n_rows / n_blocks, 3),
        "mean_tokens_per_block": round(n_tokens / n_blocks, 1),
        "max_rows_per_block": max(len(b.rows) for b in blocks),
        "fill": round(n_tokens / (n_blocks * pack_len), 4),
        "pack_len": pack_len,
    }


class PackedPadCollator:
    """Blocks -> ``[B, L]`` tensors right-padded to the longest block of the micro-batch.

    ``labels`` pad with -100 and ``position_ids`` with 0 (each pad token is then a
    sequence of its own under the position-id packed-sequence detection, so it
    attends to nothing real). ``attention_mask=True`` (naive) also emits the 2-D mask,
    which makes transformers use the plain causal+padding mask (rows attend across
    boundaries); ``attention_mask=False`` (sdpa) leaves the block-diagonal mask to
    transformers' position-id detection: rows are isolated.
    """

    def __init__(self, pad_token_id: int | None, *, attention_mask: bool = True, pad_as_row: bool = False) -> None:
        self._pad = pad_token_id if pad_token_id is not None else 0
        self.attention_mask = attention_mask
        # varlen attention (damaged Gemma2 pack-speed route only): the pad tail of a block is ONE extra row (positions
        # 0..pad-1) instead of one row per pad token, so the kernel sees ~4 segments per block, not hundreds; pads carry
        # label -100 and sit after the block's real rows, so real-token outputs are unchanged
        self.pad_as_row = pad_as_row

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        ids, labels, pos, attn = [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            ids.append(list(f["input_ids"]) + [self._pad] * pad)
            labels.append(list(f["labels"]) + [IGNORE] * pad)
            pos.append(list(f["position_ids"]) + (list(range(pad)) if self.pad_as_row else [0] * pad))
            attn.append([1] * n + [0] * pad)
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(pos, dtype=torch.long),
        }
        if self.attention_mask:
            out["attention_mask"] = torch.tensor(attn, dtype=torch.long)
        return out


def probe_batch(ids: Any, mode: str) -> dict[str, Any]:
    """Worst-case synthetic micro-batch for ``memory_probe`` in a packed mode.

    ``ids`` is ``[micro, pack_len]``. naive: 2-D mask plus position_ids (two rows per
    block); sdpa: no mask, two rows per block (materialises the block-diagonal mask
    the training path pays for). Boundary targets stay masked.
    """
    import torch

    micro, width = ids.shape
    head = max(1, width // 2)
    row = torch.cat([torch.arange(head), torch.arange(width - head)]).to(ids.device)
    pos = row.unsqueeze(0).expand(micro, -1).contiguous()
    labels = ids.clone()
    labels[pos == 0] = IGNORE
    batch = {"input_ids": ids, "position_ids": pos, "labels": labels}
    if mode != "sdpa":
        batch["attention_mask"] = torch.ones_like(ids)
    return batch


def resolve_pack_len(max_len: int, model_cap: int) -> int:
    """Block budget: the data max_len unless FORGE_V2_FIELD_PACK_LEN asks for more
    (clamped to the model/evaluator cap); never below max_len so no row is split."""
    try:
        forced = int(os.environ.get(ENV_PACK_LEN, "0") or 0)
    except ValueError:
        forced = 0
    if forced <= 0:
        return int(max_len)
    return max(int(max_len), min(forced, int(model_cap)))


def _seed() -> int:
    try:
        return int(os.environ.get("FORGE_V2_SEED", "7") or 7)
    except ValueError:
        return 7


@dataclass
class PackingPlan:
    requested: str
    mode: str
    pack_len: int
    features: list[dict[str, Any]]  # one Trainer row per block
    collator: Any
    stats: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None  # why ``mode`` differs from ``requested``
    attn_implementation: str | None = None

    @property
    def n_blocks(self) -> int:
        return len(self.features)

    @property
    def mean_rows_per_block(self) -> float:
        return float(self.stats["rows"] / self.stats["blocks"]) if self.stats["blocks"] else 1.0

    def eff_blocks(self, eff_rows: int) -> int:
        """Blocks per optimizer step that carry about ``eff_rows`` rows (the champion's
        rows-per-step bucket survives packing); FORGE_V2_FIELD_PACKED_EFF_BATCH forces it."""
        try:
            forced = int(os.environ.get(ENV_EFF_BLOCKS, "0") or 0)
        except ValueError:
            forced = 0
        if forced > 0:
            return forced
        return max(1, int(math.ceil(max(1, eff_rows) / max(self.mean_rows_per_block, 1e-9))))

    def rows_per_step(self, eff_blocks: int) -> int:
        """Rows an optimizer step of ``eff_blocks`` blocks sees on average."""
        return max(1, int(round(eff_blocks * self.mean_rows_per_block)))

    def telemetry(self) -> dict[str, Any]:
        out = {"mode": self.mode, "requested": self.requested, "attn_implementation": self.attn_implementation}
        if self.reason:
            out["reason"] = self.reason
        out.update(self.stats)
        return out


def _eligibility(model: Any, *, field_route: bool, field_hygiene: bool) -> str | None:
    if not field_route:
        return "not_field_route"
    if field_hygiene:
        return "hygiene_not_supported"
    if os.environ.get("FORGE_V2_ROW_LOSS", "0") == "1":
        return "row_loss_objective"
    import transformers
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2Attention

    if transformers.__version__ != QUALIFIED_TRANSFORMERS:
        return "transformers_version_unqualified:" + transformers.__version__
    cfg = getattr(model, "config", None)
    # native families: the block-diagonal isolation comes from transformers' position-id based packed-sequence
    # masks, which Llama and Qwen2 share; each family is scoped to its validated vocabulary
    native = {LlamaForCausalLM: (LlamaModel, LlamaAttention, "llama", 128256),
              Qwen2ForCausalLM: (Qwen2Model, Qwen2Attention, "qwen2", 151936)}
    if getattr(model, GEMMA2_ROUTE_ATTR, False) is True:
        # damaged Gemma2 champion route only: Gemma2 builds its full and sliding-window masks with the same position-id
        # packed-sequence detection (masking_utils), so rows stay isolated in both layer types
        from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention, Gemma2ForCausalLM, Gemma2Model

        native[Gemma2ForCausalLM] = (Gemma2Model, Gemma2Attention, "gemma2", 256000)
    spec = native.get(type(model))
    if (spec is None or type(getattr(model, "model", None)) is not spec[0]
            or any(type(layer.self_attn) is not spec[1] for layer in model.model.layers)):
        return "not_native_llama"  # historical reason string: "not a native Llama/Qwen2 implementation"
    if cfg.model_type != spec[2] or cfg.vocab_size != spec[3]:
        return "llama_scope_unqualified"  # historical reason string: family/vocabulary outside the validated scope
    if cfg.use_cache:
        return "cache_must_be_disabled"
    return None


def plan_field_packing(model: Any, rows: list[dict[str, list[int]]], *, max_len: int,
                       pad_token_id: int | None, model_cap: int, field_route: bool,
                       field_hygiene: bool = False, log: Callable[..., None] | None = None) -> PackingPlan | None:
    # Preserve explicit off. The conditional default is a research candidate, not
    # an admission override for other families or unqualified Llama implementations.
    requested = os.environ.get(ENV_MODE, "sdpa").strip().lower()
    if requested == "off":
        return None
    reason = _eligibility(model, field_route=field_route, field_hygiene=field_hygiene)
    attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if reason is None and requested not in {"sdpa", "naive"}:
        reason = "unsupported_mode:" + requested
    if reason is None and requested == "sdpa":
        backends = [attn, model.model.config._attn_implementation]
        backends += [layer.self_attn.config._attn_implementation for layer in model.model.layers]
        if any(backend != "sdpa" for backend in backends):
            reason = "sdpa_backend_required:" + repr(backends)
    if reason is None and not rows:
        reason = "empty_train"
    if reason is not None:
        if log is not None and (field_route or ENV_MODE in os.environ):
            log("sft_v2_packing_skipped", requested=requested, reason=reason,
                mode="off", attn_implementation=attn)
        return None
    pack_len = resolve_pack_len(max_len, model_cap)
    if any(not row["input_ids"] or len(row["input_ids"]) > pack_len for row in rows):
        raise ValueError("Packing requires nonempty rows within the existing truncation cap")
    blocks = pack_rows(rows, pack_len, seed=_seed())
    plan = PackingPlan(requested, requested, pack_len, [b.feature() for b in blocks],
                       PackedPadCollator(pad_token_id, attention_mask=requested == "naive"),
                       block_stats(blocks, pack_len), attn_implementation=attn)
    if log is not None:
        log("sft_v2_packing", **plan.telemetry())
    return plan


def optimizer_steps(n_units: int, micro: int, accum: int) -> int:
    """Trainer 5.12.1 counts the final partial batch/accumulation as an update."""
    if micro <= 0 or accum <= 0:
        raise ValueError("micro and accumulation must be positive")
    return max(1, math.ceil(n_units / (micro * accum)))


# ---- varlen attention for the damaged Gemma2 champion route (research-gemma-pack-speed-20260927) ------------------
#
# The sdpa route above isolates packed rows through a materialised 4-D block-diagonal mask. PyTorch's flash kernel
# takes no mask, so sdpa runs a masked kernel over every (query, key) pair of the block, including the other rows and
# the future half. Varlen attention runs the flash kernel per row instead, from cumulative row offsets (cu_seqlens)
# derived from the same reset position ids, the way the public champion's FA2 varlen packing does. Semantics are the
# sdpa route's exactly: causal inside each row, nothing across rows, NO attention-logit softcap (transformers 5.12.1
# sdpa drops it too, and the pinned evaluator scores with sdpa), and the sliding window never binds because every
# row fits inside it (checked). The kernel is PyTorch's own FA2 (torch.ops.aten._flash_attention_forward, the op
# torch's nested-tensor SDPA calls); the flash-attn package is not in the image. CPU has no flash kernel, so there
# the same function runs a per-row sdpa loop (the reference the CUDA self-check compares flash against).

VARLEN_ATTN = "forge_gemma2_varlen"
VARLEN_BACKEND_ATTR = "_forge_varlen_backend"  # set on each attention module: "flash" (CUDA) or "segment_loop"
VARLEN_BACKENDS = ("flash", "segment_loop")

_SEGMENT_CACHE: dict[str, Any] = {}


def segment_offsets(position_ids: Any, batch: int, length: int) -> tuple[Any, int]:
    """Row offsets over the flattened ``[batch * length]`` tokens: a row starts at every batch row's first token and
    wherever the position id does not advance by exactly 1 (transformers' ``find_packed_sequence_indices`` rule).
    Returns (int32 cumulative offsets incl. 0 and batch * length, the longest row)."""
    import torch

    pos = position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    if pos.shape[0] != batch:
        pos = pos.expand(batch, -1)
    if pos.shape[1] != length:
        raise ValueError(f"position_ids length {pos.shape[1]} != sequence length {length}")
    start = torch.ones((batch, length), dtype=torch.bool, device=pos.device)
    if length > 1:
        start[:, 1:] = (pos[:, 1:] - pos[:, :-1]) != 1
    idx = torch.nonzero(start.reshape(-1), as_tuple=False).flatten()
    cu = torch.cat([idx, idx.new_tensor([batch * length])]).to(torch.int32)
    return cu, int((cu[1:] - cu[:-1]).max().item())


def _segments(position_ids: Any, batch: int, length: int) -> tuple[Any, int]:
    """segment_offsets, computed once per forward: every layer (and the checkpointed recompute) receives the same
    position_ids object, so a weak reference to it keys the cache (a new batch is a new object)."""
    import weakref

    hit = _SEGMENT_CACHE.get("entry")
    key = (batch, length, int(position_ids._version), str(position_ids.device))
    if hit is not None and hit[0]() is position_ids and hit[1] == key:
        return hit[2]
    value = segment_offsets(position_ids, batch, length)
    _SEGMENT_CACHE["entry"] = (weakref.ref(position_ids), key, value)
    return value


def _flash_varlen(q: Any, k: Any, v: Any, cu: Any, max_len: int, dropout: float, scale: float | None) -> Any:
    """``[T, H, D]`` causal varlen attention with PyTorch's FA2 kernel (CUDA, fp16/bf16); autograd is registered."""
    import torch

    return torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, max_len, max_len, float(dropout), True, False,
                                                   scale=scale)[0]


def _loop_varlen(q: Any, k: Any, v: Any, cu: Any, dropout: float, scale: float | None) -> Any:
    """The reference: plain causal sdpa on each row separately (CPU path and the CUDA self-check's comparison)."""
    import torch

    bounds = cu.tolist()
    outs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o = torch.nn.functional.scaled_dot_product_attention(
            q[s:e].transpose(0, 1).unsqueeze(0), k[s:e].transpose(0, 1).unsqueeze(0),
            v[s:e].transpose(0, 1).unsqueeze(0), dropout_p=float(dropout), is_causal=True, scale=scale)
        outs.append(o[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


def varlen_attention_forward(module: Any, query: Any, key: Any, value: Any, attention_mask: Any, dropout: float = 0.0,
                             scaling: float | None = None, sliding_window: int | None = None, softcap: float | None = None,
                             position_ids: Any = None, **kwargs: Any) -> tuple[Any, None]:
    """transformers AttentionInterface entry. A batch that carries a padding mask (dev evaluation, the dev pass) or
    decodes against a cache takes transformers' sdpa function unchanged; packed / plain causal training batches take
    varlen attention. ``softcap`` is dropped exactly as sdpa drops it."""
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    if attention_mask is not None or position_ids is None or query.shape[2] != key.shape[2]:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      sliding_window=sliding_window, softcap=softcap, **kwargs)
    batch, heads, length, head_dim = query.shape
    n_rep = heads // key.shape[1]
    key, value = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    cu, max_len = _segments(position_ids, batch, length)
    if sliding_window is not None and max_len > int(sliding_window):
        raise RuntimeError(f"varlen row of {max_len} tokens exceeds the sliding window {sliding_window}")
    q = query.transpose(1, 2).reshape(batch * length, heads, head_dim)
    k = key.transpose(1, 2).reshape(batch * length, heads, head_dim)
    v = value.transpose(1, 2).reshape(batch * length, heads, value.shape[-1])
    backend = getattr(module, VARLEN_BACKEND_ATTR, "segment_loop")
    if backend == "flash":
        out = _flash_varlen(q, k, v, cu.to(q.device), max_len, dropout, scaling)
    else:
        out = _loop_varlen(q, k, v, cu, dropout, scaling)
    return out.view(batch, length, heads, value.shape[-1]), None


def varlen_mask(*args: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
    """AttentionMaskInterface entry: no mask for mask-free batches (varlen reads the rows from position_ids, like
    FA2); a padded batch gets exactly the mask sdpa would, for the sdpa fallback above."""
    if attention_mask is None:
        return None
    from transformers.masking_utils import sdpa_mask

    return sdpa_mask(*args, attention_mask=attention_mask, **kwargs)


def register_varlen() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(VARLEN_ATTN, varlen_attention_forward)
    AttentionMaskInterface.register(VARLEN_ATTN, varlen_mask)


def flash_varlen_self_check(device: Any, dtype: Any, heads: int, head_dim: int, scale: float | None,
                            *, impl: Callable[..., Any] | None = None, tol: float = 0.05) -> str | None:
    """Forward + backward of the flash varlen call against the per-row reference on ragged rows (lengths 1..90) at
    the model's real heads / head_dim / dtype. None = agree; otherwise the reason (the route then stays on sdpa)."""
    import torch

    impl = impl or _flash_varlen
    try:
        g = torch.Generator().manual_seed(20260927)
        lens = [37, 5, 1, 90, 3, 64]
        total = sum(lens)
        cu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], dtype=torch.int32, device=device)
        base = [torch.randn(total, heads, head_dim, generator=g).to(device=device, dtype=dtype) for _ in range(3)]
        w = torch.randn(total, heads, head_dim, generator=g).to(device=device, dtype=torch.float32)
        results = []
        for fn in (lambda q, k, v: impl(q, k, v, cu, max(lens), 0.0, scale),
                   lambda q, k, v: _loop_varlen(q, k, v, cu, 0.0, scale)):
            qkv = [t.clone().requires_grad_(True) for t in base]
            out = fn(*qkv)
            (out.float() * w).sum().backward()
            results.append([out.detach().float()] + [t.grad.detach().float() for t in qkv])
        for name, a, b in zip(("out", "dq", "dk", "dv"), results[0], results[1]):
            if a.shape != b.shape:
                return f"{name}_shape:{tuple(a.shape)}!={tuple(b.shape)}"
            err = float((a - b).abs().max())
            if not err <= tol * max(1.0, float(b.abs().max())):
                return f"{name}_mismatch:{err:.4g}"
        return None
    except Exception as exc:  # noqa: BLE001 - any kernel failure keeps the sdpa route
        return f"{type(exc).__name__}: {exc}"[:200]


def enable_gemma2_varlen(model: Any, *, pack_len: int, log: Callable[..., None] | None = None) -> str | None:
    """Switch an admitted damaged-Gemma2 packed model from sdpa to varlen attention. Returns the backend ("flash" on
    CUDA after a passing self-check, "segment_loop" on CPU) or None with the model left on sdpa. Only called for
    a model that already qualified for sdpa packing (plan_field_packing mode "sdpa")."""
    import torch

    reason = None
    cfg = getattr(model, "config", None)
    layers = list(getattr(getattr(model, "model", None), "layers", []) or [])
    sliding = getattr(cfg, "sliding_window", None)
    if getattr(model, GEMMA2_ROUTE_ATTR, False) is not True or getattr(cfg, "model_type", None) != "gemma2":
        reason = "not_gemma2_route"
    elif not layers or any(getattr(l.self_attn.config, "_attn_implementation", None) != "sdpa" for l in layers):
        reason = "sdpa_backend_required"
    elif sliding is not None and int(pack_len) > int(sliding):
        reason = f"pack_len_exceeds_sliding_window:{pack_len}>{sliding}"
    elif getattr(model, "hf_device_map", None) and len(set(map(str, model.hf_device_map.values()))) > 1:
        reason = "sharded"
    elif not hasattr(model, "set_attn_implementation"):
        reason = "no_set_attn_implementation"
    backend = None
    if reason is None:
        param = next(model.parameters())
        if param.device.type == "cuda":
            heads = int(cfg.num_attention_heads)
            head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // heads)
            scale = float(cfg.query_pre_attn_scalar) ** -0.5
            dtype = param.dtype if param.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            err = flash_varlen_self_check(param.device, dtype, heads, head_dim, scale)
            if err is not None:
                reason = "self_check_failed:" + err
            else:
                backend = "flash"
        else:
            backend = "segment_loop"
    if reason is None:
        try:
            register_varlen()
            model.set_attn_implementation(VARLEN_ATTN)
            if any(getattr(l.self_attn.config, "_attn_implementation", None) != VARLEN_ATTN for l in layers):
                raise RuntimeError("layers did not switch")
            for layer in layers:
                setattr(layer.self_attn, VARLEN_BACKEND_ATTR, backend)
        except Exception as exc:  # noqa: BLE001
            reason = f"switch_failed:{type(exc).__name__}: {exc}"[:200]
            disable_gemma2_varlen(model)
    if log is not None:
        log("sft_v2_gemma2_varlen", enabled=reason is None, backend=backend if reason is None else None,
            reason=reason, attn_implementation=getattr(cfg, "_attn_implementation", None), pack_len=int(pack_len),
            sliding_window=sliding, softcap_applied=False)
    return backend if reason is None else None


def disable_gemma2_varlen(model: Any) -> None:
    try:
        model.set_attn_implementation("sdpa")
    except Exception:  # noqa: BLE001
        model.config._attn_implementation = "sdpa"
    for layer in list(getattr(getattr(model, "model", None), "layers", []) or []):
        if hasattr(layer.self_attn, VARLEN_BACKEND_ATTR):
            delattr(layer.self_attn, VARLEN_BACKEND_ATTR)
