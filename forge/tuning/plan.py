"""Choose hyperparameters and batch geometry for a run.

Two jobs: pick settings that beat the deliberately-weak reference baseline
(which trains ~10 steps with a rank-8 adapter), and shape the run so it fits the
wall clock. We don't try to predict throughput up front — the Deadline measures
it live and a callback stops training before the export reserve — so the plan
here is about *quality* settings and memory-safe geometry; the clock enforces
the *quantity*.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainPlan:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    per_device_batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    num_epochs: int
    warmup_ratio: float
    weight_decay: float
    optimizer: str
    lr_scheduler: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool


def _hardware() -> tuple[bool, bool]:
    """(cuda_available, bf16_supported)."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, bool(torch.cuda.is_bf16_supported())
    except Exception:
        pass
    return False, False


def _base(cuda: bool, bf16: bool) -> dict:
    return dict(
        per_device_batch_size=4 if cuda else 1,
        grad_accum_steps=4 if cuda else 1,
        warmup_ratio=0.03,
        weight_decay=0.0,
        # Fused AdamW is fast and, unlike the 8-bit optimizer, carries no
        # bitsandbytes version risk on the validator GPU.
        optimizer="adamw_torch_fused" if cuda else "adamw_torch",
        # Floored cosine (min-LR 0.25 of peak): keeps a useful LR if the deadline
        # cuts the schedule short, instead of annealing toward zero mid-run.
        lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=cuda,
        bf16=bf16,
        fp16=cuda and not bf16,
    )


def make_sft_plan(*, use_kl: bool) -> TrainPlan:
    cuda, bf16 = _hardware()
    b = _base(cuda, bf16)
    # Match the evaluator's 4096 sequence length so long completions aren't
    # truncated in training but kept at scoring. Padding is dynamic (per batch),
    # so a higher cap is nearly free on short data and OOM-retry guards big models.
    if use_kl:
        # KL tasks penalise divergence from base: a smaller adapter and gentler
        # LR keep us close while still improving eval loss. They also run a second
        # (base) forward, so halve the micro-batch and hold the effective batch
        # constant via accumulation.
        if cuda:
            b["per_device_batch_size"] = 2
            b["grad_accum_steps"] = 8
        return TrainPlan(
            lora_r=16, lora_alpha=32, lora_dropout=0.05,
            learning_rate=1.0e-4, max_seq_len=4096, num_epochs=2, **b,
        )
    return TrainPlan(
        lora_r=32, lora_alpha=64, lora_dropout=0.05,
        learning_rate=1.5e-4, max_seq_len=4096, num_epochs=2, **b,
    )


def make_dpo_plan() -> TrainPlan:
    cuda, bf16 = _hardware()
    b = _base(cuda, bf16)
    # DPO packs chosen+rejected into one batch (2x sequences) at 4096 and holds a
    # reference forward, so it is materially heavier than SFT. Halve the micro-
    # batch and double accumulation to keep the effective batch while staying
    # inside the memory the OOM-retry would otherwise have to rescue.
    if cuda:
        b["per_device_batch_size"] = 2
        b["grad_accum_steps"] = 8
    # DPO is sensitive; a too-hot LR silently collapses the policy log-probs
    # (loss falls while the model degrades). Champion DPO LRs sit in 5e-6..1.35e-5
    # with a 3e-5 hard ceiling, so 1e-5 is a safe, well-inside value. The evaluator
    # scores DPO pairs at the model's own max length, so we don't truncate tighter
    # than 4096 (clamped to the model's positional range at load time).
    return TrainPlan(
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=1.0e-5, max_seq_len=4096, num_epochs=2, **b,
    )


def make_grpo_plan() -> TrainPlan:
    cuda, bf16 = _hardware()
    b = _base(cuda, bf16)
    b["per_device_batch_size"] = 2 if cuda else 1
    # GRPO is also LR-sensitive; champion GRPO LRs are ~3e-6..8e-6 with a 1.5e-5
    # ceiling, so 8e-6 keeps us inside the safe band.
    return TrainPlan(
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=8.0e-6, max_seq_len=1024, num_epochs=1, **b,
    )
