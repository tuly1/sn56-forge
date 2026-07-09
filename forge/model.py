"""Base-model resolution and loading.

The base model is pre-staged read-only under `/cache/models/...`; its identity is
often scrubbed (`config.json._name_or_path` removed) and any LoRA chain is
pre-merged by the validator's downloader. So we resolve it from the local path
only — never the network — and introspect the config/tokenizer rather than
trusting the `--model` string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def resolve_model_dir(cached_model_dir: str) -> str:
    """Return a directory containing `config.json`, tolerating both a flat
    `local_dir` layout and the HF hub `snapshots/<hash>/` layout.
    """
    if _has_config(cached_model_dir):
        return cached_model_dir
    # HF hub snapshot layout: models--org--name/snapshots/<rev>/
    snapshots = os.path.join(cached_model_dir, "snapshots")
    if os.path.isdir(snapshots):
        revs = [os.path.join(snapshots, d) for d in os.listdir(snapshots)]
        revs = [r for r in revs if _has_config(r)]
        if revs:
            return max(revs, key=os.path.getmtime)
    # Last resort: walk for the first config.json (bounded, cheap).
    for root, _dirs, files in os.walk(cached_model_dir):
        if "config.json" in files:
            return root
    raise FileNotFoundError(f"no config.json under {cached_model_dir!r}")


def _has_config(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_dir: str
    dtype: Any


def pick_dtype() -> Any:
    """bf16 on capable GPUs (matches the evaluator), fp16 on older GPUs, fp32 on
    CPU so smoke tests run without half-precision CPU kernels.
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_base(cached_model_dir: str, *, for_generation: bool = False) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = resolve_model_dir(cached_model_dir)
    dtype = pick_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    _fix_special_tokens(tokenizer)
    # Left padding for generation-time tasks (GRPO rollouts); right padding for
    # loss-only SFT/DPO so the last real token isn't buried behind pads.
    tokenizer.padding_side = "left" if for_generation else "right"

    # Flash-attn is intentionally not requested: the evaluator disables it and
    # not every base ships kernels for it. Prefer sdpa, fall back to eager for
    # architectures that don't implement it.
    common = dict(
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation="sdpa", **common
        )
    except (ValueError, ImportError, RuntimeError):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation="eager", **common
        )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
    return LoadedModel(model=model, tokenizer=tokenizer, model_dir=model_dir, dtype=dtype)


def _fix_special_tokens(tokenizer: Any) -> None:
    """Mirror the reference trainer's tokenizer fixups so masking/padding behave:
    a missing pad token defaults to eos, and a missing bos to eos as well.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None and tokenizer.eos_token is not None:
        tokenizer.bos_token = tokenizer.eos_token
    # Last resort: a model with neither pad nor eos. Reuse unk if present so we
    # never leave pad unset (the collator would otherwise pad with a bare 0).
    if tokenizer.pad_token_id is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token


def attach_lora(model: Any, *, r: int, alpha: int, dropout: float) -> Any:
    """Wrap the model in a LoRA adapter targeting all linear layers.

    LoRA is deliberate: it keeps the finetune close to the base model, which
    both fits the memory budget and naturally limits KL(model || base) — the
    exact quantity the KL-regularised tasks penalise.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    peft_model = get_peft_model(model, config)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    return peft_model
