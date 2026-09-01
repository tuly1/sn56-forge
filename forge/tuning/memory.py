"""Conservative causal-LM training memory admission.

Parameter-count-only gates miss the dominant term for models with a very wide
language-model head: the ``batch x sequence x vocabulary`` logits and loss
workspace.  The estimate here is deliberately an admission bound, not a claim
about exact allocator use.  It makes every term explicit and leaves 30% of the
card outside the admitted budget for CUDA kernels, fragmentation, and runtime
variance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_GB = 1_000_000_000.0


@dataclass(frozen=True)
class SFTMemoryEstimate:
    strategy: str
    params_b: float
    vocab_size: int
    sequence_length: int
    microbatch: int
    gradient_checkpointing: bool
    logit_loss_gb: float
    parameter_state_gb: float
    activation_gb: float
    runtime_gb: float
    needed_gb: float
    card_gb: float
    budget_ratio: float
    budget_gb: float
    admitted: bool

    def telemetry_fields(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in tuple(values.items()):
            if isinstance(value, float):
                values[key] = round(value, 4)
        return {f"memory_{key}": value for key, value in values.items()}


def infer_output_width(model: Any, tokenizer: Any | None = None) -> int:
    """Return the actual LM-head width, never just tokenizer vocabulary size.

    Some tokenizers intentionally expose fewer entries than the padded output
    projection.  CUDA allocates logits at the projection/config width, so using
    ``len(tokenizer)`` would under-admit exactly the models this guard protects.
    """
    widths: list[int] = []
    config_width = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(config_width, int) and config_width > 0:
        widths.append(config_width)

    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    for candidate in (
        getattr(head, "out_features", None),
        getattr(getattr(head, "weight", None), "shape", (None,))[0],
    ):
        if isinstance(candidate, int) and candidate > 0:
            widths.append(candidate)

    if tokenizer is not None:
        try:
            tokenizer_width = len(tokenizer)
        except Exception:
            tokenizer_width = None
        if isinstance(tokenizer_width, int) and tokenizer_width > 0:
            widths.append(tokenizer_width)
    if not widths:
        raise ValueError("cannot infer a positive causal-LM output width")
    return max(widths)


def estimate_sft_memory(
    *,
    params_b: float,
    vocab_size: int,
    sequence_length: int,
    microbatch: int,
    strategy: str,
    gradient_checkpointing: bool,
    card_gb: float,
    budget_ratio: float = 0.70,
) -> SFTMemoryEstimate:
    """Estimate and admit one unpacked SFT geometry.

    The loss term reserves 18 bytes per logit cell: bf16 forward logits, fp32
    loss/log-softmax materialization, gradient/backward staging, and a 1.5x
    allocator/workspace factor.  Full fine-tuning reserves the implementation's
    actual 16 B/parameter (fp32 trainable weights + gradients + AdamW states).
    LoRA reserves bf16 base weights plus 1.5 GB for adapter/optimizer state.
    Checkpointed activations get a 12 GB fixed reserve; non-checkpointed runs get
    24 GB.  A further 4 GB is kept for CUDA/runtime state.
    """
    if strategy not in {"lora", "full"}:
        raise ValueError(f"unsupported SFT strategy {strategy!r}")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not isinstance(microbatch, int) or microbatch <= 0:
        raise ValueError("microbatch must be positive")
    if params_b <= 0 or card_gb <= 0:
        raise ValueError("params_b and card_gb must be positive")
    if not 0 < budget_ratio < 1:
        raise ValueError("budget_ratio must be between zero and one")

    cells = microbatch * sequence_length * vocab_size
    logit_loss_gb = cells * 18.0 / _GB
    if strategy == "full":
        parameter_state_gb = params_b * 16.0
    else:
        parameter_state_gb = params_b * 2.0 + 1.5
    activation_gb = 12.0 if gradient_checkpointing else 24.0
    runtime_gb = 4.0
    needed_gb = logit_loss_gb + parameter_state_gb + activation_gb + runtime_gb
    budget_gb = card_gb * budget_ratio
    return SFTMemoryEstimate(
        strategy=strategy,
        params_b=params_b,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        microbatch=microbatch,
        gradient_checkpointing=gradient_checkpointing,
        logit_loss_gb=logit_loss_gb,
        parameter_state_gb=parameter_state_gb,
        activation_gb=activation_gb,
        runtime_gb=runtime_gb,
        needed_gb=needed_gb,
        card_gb=card_gb,
        budget_ratio=budget_ratio,
        budget_gb=budget_gb,
        admitted=needed_gb <= budget_gb,
    )


def require_sft_admission(**kwargs: Any) -> SFTMemoryEstimate:
    """Return the estimate or fail closed before optimizer construction."""
    estimate = estimate_sft_memory(**kwargs)
    if not estimate.admitted:
        raise RuntimeError(
            "SFT geometry rejected by logits-aware memory admission: "
            f"need {estimate.needed_gb:.2f} GB, budget {estimate.budget_gb:.2f} GB "
            f"for B={estimate.microbatch}, S={estimate.sequence_length}, "
            f"V={estimate.vocab_size}, strategy={estimate.strategy}"
        )
    return estimate
