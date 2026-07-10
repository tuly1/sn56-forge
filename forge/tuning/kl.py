"""KL-regularised SFT.

On KL tasks the scorer ranks by `eval_loss + kl_coef * KL(finetuned || base)`,
averaged over completion tokens. The reference trainer ignores this; we add the
matching term to the training objective so we optimise what we're graded on.

The base model's logits come from the *same* PEFT model with the LoRA adapter
disabled — no second copy in memory — which is exactly the "base" the evaluator
compares against (the pre-staged, already-merged base weights).

Two subtleties, both load-bearing:

* **Normalisation.** The base Trainer normalises CE by `num_items_in_batch` (the
  completion-token count across the whole gradient-accumulation group) and then
  skips its usual `/= grad_accum`. A per-micro-batch *mean* KL added on top would
  therefore be summed undivided across the group, inflating the effective
  coefficient by `grad_accum`. We normalise the KL by the same denominator.
* **Memory.** The KL needs full-vocab log-probs over every completion token. At
  4096 sequence length that is several multi-GB buffers, so we accumulate it in
  chunks that are recomputed during backward rather than held.
"""

from __future__ import annotations

import torch
from transformers import Trainer

# Completion tokens per KL chunk. Bounds the peak [chunk, vocab] buffers.
_KL_CHUNK_TOKENS = 1024


class KLSFTTrainer(Trainer):
    def __init__(self, *args, kl_coef: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._kl_coef = float(kl_coef)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Delegate the CE term to the base Trainer so its gradient-accumulation /
        # num_items_in_batch normalisation is applied exactly as on a normal run
        # (computing it ourselves would silently scale the effective LR). Then add
        # the KL term the scorer penalises, computed from the same forward's logits.
        ce_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )

        kl_sum, kl_tokens = self._completion_kl_sum(
            model, inputs, outputs.logits, inputs["labels"]
        )
        if kl_tokens == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Match the denominator the base class used for CE: the group-wide
        # completion-token count when it is supplied, else this micro-batch's.
        num_items = kwargs.get("num_items_in_batch")
        denom = float(num_items) if num_items is not None else float(kl_tokens)
        kl = kl_sum / denom

        loss = ce_loss + self._kl_coef * kl
        return (loss, outputs) if return_outputs else loss

    def _completion_kl_sum(self, model, inputs, ft_logits, labels):
        """Sum of per-token KL(ft || base) over completion tokens, and their count."""
        base_module = model.module if hasattr(model, "module") else model
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            with base_module.disable_adapter():
                base_logits = base_module(**model_inputs).logits

        # Align to the causal shift and the completion mask (labels != -100).
        shift_ft = ft_logits[:, :-1, :]
        shift_base = base_logits[:, :-1, :]
        mask = labels[:, 1:] != -100
        n_tokens = int(mask.sum())
        if n_tokens == 0:
            return ft_logits.new_zeros(()), 0

        ft = shift_ft[mask]
        base = shift_base[mask]

        total = ft.new_zeros((), dtype=torch.float32)
        for start in range(0, n_tokens, _KL_CHUNK_TOKENS):
            end = min(start + _KL_CHUNK_TOKENS, n_tokens)
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_kl_sum, ft[start:end], base[start:end], use_reentrant=False
            )
        return total, n_tokens


def _chunk_kl_sum(ft_chunk: torch.Tensor, base_chunk: torch.Tensor) -> torch.Tensor:
    """KL(ft || base) summed over a chunk of completion tokens.

    Recomputed during backward (see checkpoint above), so the full-vocab
    intermediates are never all resident at once.
    """
    log_p_ft = torch.log_softmax(ft_chunk.float(), dim=-1)
    log_p_base = torch.log_softmax(base_chunk.float(), dim=-1)
    return (log_p_ft.exp() * (log_p_ft - log_p_base)).sum(dim=-1).sum()
