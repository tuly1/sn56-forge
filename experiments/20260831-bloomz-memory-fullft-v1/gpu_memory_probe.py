#!/usr/bin/env python3
"""Measured one-H100 admission for the exact BloomZ training geometry.

This is a same-VM hardware proof, not a provider controller.  It executes an
actual max-length forward, backward, and fused-AdamW step so optimizer state is
allocated before the peak is admitted.  The analytic wide-head estimate is
recorded only as supporting evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import gc
import sys
from types import SimpleNamespace

from forge.model import (
    attach_lora,
    load_base,
    model_param_billions,
    prepare_full_finetune,
)
from forge.tuning import bloomz
from forge.tuning.memory import infer_output_width, require_sft_admission
from forge.tuning.plan import make_sft_plan


class ProbeError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--arm", choices=("control", "full"), required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--runtime-authority", required=True)
    parser.add_argument("--max-reserved-ratio", type=float, default=0.70)
    return parser.parse_args(argv)


def _frozen_lr(arm: str, supplied: float | None) -> float:
    if arm == "control":
        value = bloomz.CONTROL_LR if supplied is None else supplied
        require(value == bloomz.CONTROL_LR, "control LR must remain 1.5e-4")
        return value
    require(supplied in bloomz.ALLOWED_FULL_LRS, "full LR is outside the frozen probe set")
    return float(supplied)


def _write_exclusive(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProbeError(f"refusing to overwrite receipt: {path}") from exc


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import transformers
    import peft

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    require(torch.cuda.is_available(), "CUDA is required")
    require(torch.cuda.device_count() == 1, "exactly one GPU is required")
    device_name = str(torch.cuda.get_device_name(0))
    properties = torch.cuda.get_device_properties(0)
    card_bytes = int(properties.total_memory)
    card_gb = card_bytes / 1_000_000_000.0
    require("H100" in device_name.upper(), f"device is not H100: {device_name!r}")
    require(card_gb >= 70.0, f"H100 memory is below 70 GB: {card_gb:.2f}")
    require(
        0.50 <= args.max_reserved_ratio <= 0.80,
        "max-reserved-ratio must stay in [0.50, 0.80]",
    )

    lr = _frozen_lr(args.arm, args.learning_rate)
    authority_path, authority, authority_sha = bloomz.load_runtime_authority(
        args.runtime_authority
    )
    bloomz.require_science_stage(
        authority["lease"],
        stage_max_seconds=600,
        remaining_planned_seconds=600,
    )
    loaded = load_base(args.model_dir, for_generation=False)
    bloomz.validate_model_identity(loaded.model, loaded.model_dir)
    params_b = model_param_billions(loaded.model)
    strategy = "full" if args.arm == "full" else "lora"
    base_plan = make_sft_plan(
        use_kl=False,
        strategy=strategy,
        params_b=params_b,
        n_gpus=1,
        per_gpu_gb=card_gb,
    )
    plan = bloomz.apply_plan(
        base_plan,
        SimpleNamespace(arm=args.arm, learning_rate=lr),
    )
    require(
        (plan.per_device_batch_size, plan.grad_accum_steps, plan.max_seq_len)
        == (1, 16, 2048),
        "frozen geometry drift",
    )
    require(plan.gradient_checkpointing, "gradient checkpointing must be enabled")
    output_width = infer_output_width(loaded.model, loaded.tokenizer)
    analytic = require_sft_admission(
        params_b=params_b,
        vocab_size=output_width,
        sequence_length=plan.max_seq_len,
        microbatch=plan.per_device_batch_size,
        strategy=strategy,
        gradient_checkpointing=True,
        card_gb=card_gb,
    )
    analytic_selection = require_sft_admission(
        params_b=params_b,
        vocab_size=output_width,
        sequence_length=bloomz.SELECTION_SEQUENCE_LENGTH,
        microbatch=plan.per_device_batch_size,
        strategy=strategy,
        gradient_checkpointing=True,
        card_gb=card_gb,
    )

    if strategy == "full":
        model = prepare_full_finetune(
            loaded.model, gradient_checkpointing=True
        )
    else:
        model = attach_lora(
            loaded.model,
            r=plan.lora_r,
            alpha=plan.lora_alpha,
            dropout=plan.lora_dropout,
        )
    enable_gc = getattr(model, "gradient_checkpointing_enable", None)
    require(callable(enable_gc), "model cannot enable gradient checkpointing")
    enable_gc(gradient_checkpointing_kwargs={"use_reentrant": False})
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    model.config.use_cache = False
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    require(bool(trainable), "no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, fused=True)

    token_id = 42
    require(token_id < output_width, "synthetic token id exceeds output width")
    input_ids = torch.full(
        (1, bloomz.SEQUENCE_LENGTH), token_id, dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_before = torch.cuda.mem_get_info()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = output.loss
    require(loss is not None and bool(torch.isfinite(loss).item()), "forward loss is nonfinite")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_loss_value = float(loss.detach().float().cpu())
    torch.cuda.synchronize()
    train_peak_allocated = int(torch.cuda.max_memory_allocated())
    train_peak_reserved = int(torch.cuda.max_memory_reserved())
    train_free_after, train_total_after = torch.cuda.mem_get_info()

    # Reproduce the exact labeled 4096-token dev geometry while the just-created
    # AdamW state remains resident. This is the actual Trainer lifecycle at an
    # on-step evaluation, and closes the wide-head eval-memory gap separately
    # from the required B1/S2048 optimizer-step proof.
    del output, loss, input_ids, attention_mask, labels
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    selection_ids = torch.full(
        (1, bloomz.SELECTION_SEQUENCE_LENGTH),
        token_id,
        dtype=torch.long,
        device="cuda",
    )
    selection_mask = torch.ones_like(selection_ids)
    selection_labels = selection_ids.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        selection_output = model(
            input_ids=selection_ids,
            attention_mask=selection_mask,
            labels=selection_labels,
            use_cache=False,
        )
    selection_loss = selection_output.loss
    require(
        selection_loss is not None and bool(torch.isfinite(selection_loss).item()),
        "4096-token selection loss is nonfinite",
    )
    torch.cuda.synchronize()
    selection_peak_allocated = int(torch.cuda.max_memory_allocated())
    selection_peak_reserved = int(torch.cuda.max_memory_reserved())
    free_after, total_after = torch.cuda.mem_get_info()
    total_tolerance = 1024 * 1024
    require(
        abs(total_before - card_bytes) <= total_tolerance
        and abs(train_total_after - card_bytes) <= total_tolerance
        and abs(total_after - card_bytes) <= total_tolerance,
        "CUDA total-memory reading drifted beyond 1 MiB",
    )
    train_reserved_ratio = train_peak_reserved / card_bytes
    selection_reserved_ratio = selection_peak_reserved / card_bytes
    require(
        max(train_reserved_ratio, selection_reserved_ratio) <= args.max_reserved_ratio,
        "measured train/selection peak reserved ratio "
        f"{max(train_reserved_ratio, selection_reserved_ratio):.4f} exceeds "
        f"{args.max_reserved_ratio:.4f}",
    )

    optimizer_state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    optimizer_state_bytes = sum(
        int(value.numel() * value.element_size()) for value in optimizer_state_tensors
    )
    require(optimizer_state_bytes > 0, "optimizer step did not allocate tensor state")

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "proof": (
            "actual_b1_s2048_forward_backward_fused_adamw_step_plus_"
            "optimizer_resident_b1_s4096_labeled_eval"
        ),
        "analytic_estimate_role": "admission_support_only_not_hardware_proof",
        "arm": args.arm,
        "strategy": strategy,
        "learning_rate": lr,
        "geometry": {
            "microbatch": 1,
            "gradient_accumulation_configured": 16,
            "optimizer_probe_microsteps_executed": 1,
            "configured_effective_batch": 16,
            "sequence_length": 2048,
            "packing": False,
            "gradient_checkpointing": True,
        },
        "model": {
            "repo": bloomz.MODEL_REPO,
            "revision": bloomz.MODEL_REVISION,
            "config_sha256": file_sha256(Path(loaded.model_dir) / "config.json"),
            "weights_sha256": file_sha256(Path(loaded.model_dir) / "model.safetensors"),
            "output_width": output_width,
            "params_b": params_b,
        },
        "gpu": {
            "name": device_name,
            "card_bytes": card_bytes,
            "train_peak_allocated_bytes": train_peak_allocated,
            "train_peak_reserved_bytes": train_peak_reserved,
            "train_peak_reserved_ratio": train_reserved_ratio,
            "selection_peak_allocated_bytes": selection_peak_allocated,
            "selection_peak_reserved_bytes": selection_peak_reserved,
            "selection_peak_reserved_ratio": selection_reserved_ratio,
            "free_before_bytes": int(free_before),
            "train_free_after_bytes": int(train_free_after),
            "free_after_bytes": int(free_after),
            "max_reserved_ratio": args.max_reserved_ratio,
        },
        "loss": train_loss_value,
        "selection_loss": float(selection_loss.detach().float().cpu()),
        "optimizer_state_tensor_count": len(optimizer_state_tensors),
        "optimizer_state_bytes": optimizer_state_bytes,
        "analytic_estimate": {
            "train": analytic.telemetry_fields(),
            "selection": analytic_selection.telemetry_fields(),
        },
        "runtime_authority_path": str(authority_path),
        "authority": bloomz.authority_fields(authority, authority_sha),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    require(math.isfinite(float(receipt["loss"])), "recorded loss is nonfinite")
    require(
        math.isfinite(float(receipt["selection_loss"])),
        "recorded selection loss is nonfinite",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(args)
    _write_exclusive(Path(args.receipt), receipt)
    print(json.dumps({"status": "PASS", "receipt": args.receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
