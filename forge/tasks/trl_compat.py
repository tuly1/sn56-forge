"""Small dependency-free shims for the validator's TRL 1.5 transition."""

from __future__ import annotations

from forge.model import is_quasar_model


class PromptCappedPreferenceCollator:
    """Preserve DPO completion tokens under concatenated keep-start truncation."""

    def __init__(self, delegate, max_prompt_length: int):
        self.delegate = delegate
        self.max_prompt_length = max(1, int(max_prompt_length))

    def __call__(self, features):
        capped = []
        for feature in features:
            copied = dict(feature)
            prompt_ids = list(copied.get("prompt_ids") or [])
            copied["prompt_ids"] = prompt_ids[-self.max_prompt_length :]
            capped.append(copied)
        return self.delegate(capped)


def prompt_capped_grpo_trainer(base_class, max_prompt_length: int):
    """Version-gated TRL 1.5 prompt cap without patching installed packages."""
    cap = max(1, int(max_prompt_length))

    class PromptCappedGRPOTrainer(base_class):
        def _tokenize_prompts(self, prompts):
            prompt_ids, images, multimodal_fields = super()._tokenize_prompts(prompts)
            prompt_ids = [list(ids)[-cap:] for ids in prompt_ids]
            return prompt_ids, images, multimodal_fields

    PromptCappedGRPOTrainer.__name__ = "PromptCappedGRPOTrainer"
    return PromptCappedGRPOTrainer


def generation_kwargs_for_model(model):
    """Disable the known-broken Quasar KV cache, preserving native throughput."""
    return {"use_cache": False} if is_quasar_model(model) else None
