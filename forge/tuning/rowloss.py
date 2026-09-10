"""Per-example (validator-style) training objective.

The validator scores each held-out row by the mean cross-entropy over its
completion tokens and averages rows. The HF trainer instead averages over all
completion tokens of the accumulation group, which weights a 1,000-token
completion ten times a 100-token one. This Trainer mixin computes the loss as
the mean over rows of each row's completion-token mean, so the optimisation
target is the statistic the tournament ranks on.
"""

from __future__ import annotations

from typing import Any


def unwrap_causal_lm(model: Any) -> Any:
    """Innermost module exposing the decoder (`model`) and `lm_head` through
    accelerate / PEFT wrappers. PEFT's LoraModel forwards attribute access to
    the wrapped model, so wrappers are recognised by class before any
    duck-typed check."""
    m = model
    for _ in range(8):
        name = type(m).__name__
        if hasattr(m, "module") and "module" in getattr(m, "_modules", {}):  # DDP / accelerate
            m = m.module
            continue
        if name.startswith("PeftModel"):  # PeftModel -> LoraModel -> CausalLM
            m = m.base_model.model
            continue
        if name.endswith(("LoraModel", "Model")) and "model" in getattr(m, "_modules", {}) and not name.endswith("ForCausalLM"):
            inner = m._modules["model"]
            if type(inner).__name__.endswith("ForCausalLM"):
                m = inner
                continue
        if "lm_head" in getattr(m, "_modules", {}) and "model" in getattr(m, "_modules", {}):
            return m
        break
    raise TypeError(f"cannot find a causal LM with lm_head/model in {type(model).__name__}")


def _fused_loss_fn(softcap: float | None):
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss(reduction="mean", softcap=softcap)
    except Exception:
        return None


def row_mean_loss(model: Any, inputs: dict[str, Any], *, use_fused: bool = True) -> Any:
    """Mean over rows of the per-row completion-token cross-entropy."""
    import torch
    import torch.nn.functional as F

    base = unwrap_causal_lm(model)
    labels = inputs["labels"]
    kwargs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")}
    out = base.model(**kwargs, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    weight = base.lm_head.weight
    softcap = getattr(getattr(base, "config", None), "final_logit_softcapping", None)
    fused = _fused_loss_fn(softcap) if use_fused else None
    losses = []
    for i in range(hidden.size(0)):
        y = labels[i, 1:]
        mask = y != -100
        if int(mask.sum()) == 0:
            continue
        h = hidden[i, :-1][mask]
        t = y[mask]
        if fused is not None:
            losses.append(fused(weight, h, t))
        else:
            logits = F.linear(h, weight.to(h.dtype)).float()
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            losses.append(F.cross_entropy(logits, t))
    if not losses:
        return hidden.sum() * 0.0
    return torch.stack(losses).mean()
