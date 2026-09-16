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

    def __init__(self, pad_token_id: int | None, *, attention_mask: bool = True) -> None:
        self._pad = pad_token_id if pad_token_id is not None else 0
        self.attention_mask = attention_mask

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        ids, labels, pos, attn = [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            ids.append(list(f["input_ids"]) + [self._pad] * pad)
            labels.append(list(f["labels"]) + [IGNORE] * pad)
            pos.append(list(f["position_ids"]) + [0] * pad)
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

    if transformers.__version__ != QUALIFIED_TRANSFORMERS:
        return "transformers_version_unqualified:" + transformers.__version__
    cfg = getattr(model, "config", None)
    if (type(model) is not LlamaForCausalLM or type(getattr(model, "model", None)) is not LlamaModel
            or any(type(layer.self_attn) is not LlamaAttention for layer in model.model.layers)):
        return "not_native_llama"
    if cfg.model_type != "llama" or cfg.vocab_size != 128256:
        return "llama_scope_unqualified"
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
