"""KL-regularised SFT.

On KL tasks the scorer ranks by `eval_loss + kl_coef * KL(finetuned || base)`,
averaged over completion tokens. The reference trainer ignores this; we add the
matching term to the training objective so we optimise what we're graded on.

The base model's logits come from the *same* PEFT model with the LoRA adapter
disabled — no second copy in memory — which is exactly the "base" the evaluator
compares against (the pre-staged, already-merged base weights).
"""

from __future__ import annotations

import torch
from transformers import Trainer


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
        kl = self._completion_kl(model, inputs, outputs.logits, inputs["labels"])
        loss = ce_loss + self._kl_coef * kl
        return (loss, outputs) if return_outputs else loss

    def _completion_kl(self, model, inputs, ft_logits, labels) -> torch.Tensor:
        # Base logits: same model, adapter off, no gradient.
        base_module = model.module if hasattr(model, "module") else model
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            with base_module.disable_adapter():
                base_logits = base_module(**model_inputs).logits

        # Align to the causal shift and the completion mask (labels != -100).
        shift_ft = ft_logits[:, :-1, :]
        shift_base = base_logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if mask.sum() == 0:
            return ft_logits.new_zeros(())

        ft = shift_ft[mask].float()
        base = shift_base[mask].float()
        log_p_ft = torch.log_softmax(ft, dim=-1)
        log_p_base = torch.log_softmax(base, dim=-1)
        p_ft = log_p_ft.exp()
        # KL(ft || base), summed over vocab, averaged over completion tokens.
        kl_per_token = (p_ft * (log_p_ft - log_p_base)).sum(dim=-1)
        return kl_per_token.mean()
