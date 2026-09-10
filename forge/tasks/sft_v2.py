"""Full-weight supervised fine-tuning for small instruct tasks (v2 recipe).

Why a second SFT path: half of all tournament base models are deliberately
perturbed by the validator before miners see them (noise, pruning, scaling,
re-initialised layers), and a low-rank adapter cannot undo that. Full-weight
training can. This handler is used for non-KL InstructTextTask on one GPU when
the model is small enough; everything else stays on the validated LoRA path.

Pipeline: floor artifact -> tokenize at the evaluator cap -> exact dedup ->
adaptive max length + prompt-side truncation -> stratified dev split ->
learning-rate prior + short sweep -> time-aware epochs -> train with dev
evaluation every 1/8 epoch (validator statistic), best-checkpoint persistence,
early stop -> greedy soup of the best snapshots -> low-LR dev pass -> save.
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from forge import telemetry
from forge.clock import Deadline
from forge.data import prompts, tokenize
from forge.data.schema import TaskSpec
from forge.data.split import (
    adaptive_max_len,
    dev_size_for,
    exact_dedup,
    random_subsample,
    smart_truncate,
    stratified_split,
)
from forge.tasks.common import (
    ARTIFACT_COMPLETE_BEST,
    ARTIFACT_FLOOR,
    ARTIFACT_PARTIAL_TRAINED_BEST,
    _free_cuda,
    _is_oom,
    compatible_dataclass_kwargs,
    save_adapter,
    workdir,
    write_artifact_truth,
)
from forge.tuning import lr_probe, selection

EVAL_CAP = 4096
EFF_BATCH_TARGET = 64
MAX_PARAMS_B = 5.0  # hard ceiling of the handler (memory / throughput)
# Validated regime (2026-09-10 harness, official evaluator, paired rule): full-weight
# v2 beat production LoRA on SmolLM2-360M (-2.6%, win rate 0.76) and lost on every
# 2-3B model tried. Larger models stay on the production adapter path unless the
# operator widens the gate with FORGE_V2_MAX_PARAMS_B.
DEFAULT_MAX_PARAMS_B = 0.6
# Adapter (LoRA) strategy inside v2 for larger models: off by default until the
# batch-16 LoRA arms beat production (2026-09-10 evening decision).
DEFAULT_LORA_MAX_PARAMS_B = 0.0


def _gates() -> tuple[float, float]:
    try:
        full_b = float(os.environ.get("FORGE_V2_MAX_PARAMS_B", str(DEFAULT_MAX_PARAMS_B)))
    except ValueError:
        full_b = DEFAULT_MAX_PARAMS_B
    try:
        lora_b = float(os.environ.get("FORGE_V2_LORA_MAX_PARAMS_B", str(DEFAULT_LORA_MAX_PARAMS_B)))
    except ValueError:
        lora_b = DEFAULT_LORA_MAX_PARAMS_B
    return min(full_b, MAX_PARAMS_B), min(lora_b, MAX_PARAMS_B)


def strategy_for(params_b: float) -> str:
    """'full' inside the full-weight gate, 'lora' inside the adapter gate, else ''.
    FORGE_V2_STRATEGY=full|lora forces a strategy (harness use)."""
    forced = os.environ.get("FORGE_V2_STRATEGY", "auto").strip().lower()
    if forced in ("full", "lora"):
        return forced
    full_b, lora_b = _gates()
    if 0 < params_b <= full_b:
        return "full"
    if 0 < params_b <= lora_b:
        return "lora"
    return ""
FP32_MASTER_MAX_B = 3.3
EXPORT_RESERVE_S = 150.0
MIN_TASK_SECONDS_FOR_V2 = 20 * 60


def _log(name: str, payload: dict[str, Any] | None = None, **kw: Any) -> None:
    _event_and_print(name, **(payload or {}), **kw)


_orig_event = telemetry.event


def _event_and_print(name: str, **kv: Any) -> None:
    """Record the telemetry event and mirror v2 events to stderr so container
    logs show the recipe's decisions (the JSON flight recorder is only visible
    after the artifact is copied out)."""
    _orig_event(name, **kv)
    if name.startswith(("sft_v2", "lr_probe", "soup", "liger")):
        try:
            import sys

            compact = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in kv.items()}
            print(f"[sft_v2] {name} {compact}", file=sys.stderr, flush=True)
        except Exception:
            pass


def eligible(spec: TaskSpec, *, is_kl: bool, params_b: float, n_gpus: int, model: Any) -> bool:
    if os.environ.get("FORGE_SFT_V2", "1") == "0":
        return False
    if spec.task_type != "InstructTextTask" or spec.instruct is None or is_kl:
        return False
    allow_cpu = os.environ.get("FORGE_SFT_V2_ALLOW_CPU") == "1"
    if params_b <= 0 or params_b > MAX_PARAMS_B or not strategy_for(params_b):
        telemetry.event("sft_v2_size_gated", params_b=round(params_b, 3), gates=_gates())
        return False
    if n_gpus != 1 and not (allow_cpu and n_gpus == 0):
        return False
    try:
        import torch

        if not torch.cuda.is_available() and not allow_cpu:
            return False
    except Exception:
        return False
    from forge.model import is_quasar_model

    if is_quasar_model(model):
        return False
    return True


@dataclass
class Geometry:
    micro_batch: int
    grad_accum: int
    max_len: int
    gradient_checkpointing: bool
    fp32_master: bool
    optim: str

    @property
    def eff_batch(self) -> int:
        return self.micro_batch * self.grad_accum


def _vocab(model: Any) -> int:
    return int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)


def estimate_train_gb(*, params_b: float, tokens: int, layers: int, hidden: int, vocab: int, fp32_master: bool,
                      gradient_checkpointing: bool, liger: bool) -> float:
    """Rough peak-memory estimate (GB) for one micro-batch of `tokens` tokens."""
    weights = params_b * (16.0 if fp32_master else 8.0)  # weights+grads+Adam (8-bit Adam ~ bf16 sizes)
    per_token_layer = 34.0 * hidden * 2.0  # bf16 activations kept for backward (no checkpointing)
    if gradient_checkpointing:
        per_token_layer = 4.0 * hidden * 2.0 + per_token_layer / max(1, layers) * 2
    acts = tokens * layers * per_token_layer / 1e9
    logits = 0.0 if liger else tokens * vocab * 6.0 / 1e9
    return weights + acts + logits + 4.0


def choose_geometry(*, params_b: float, max_len: int, vocab: int, per_gpu_gb: float, bnb_ok: bool,
                    layers: int = 32, hidden: int = 2048, liger: bool = False) -> Geometry:
    """Geometry from measured behaviour (H100, Gemma-2-2B, 2026-09-10 benchmark):
    checkpointing with a large micro-batch (8 x 3136 tokens) ran at 5.7 samples/s
    while no-checkpointing at micro 2 crawled at 1.9 samples/s from memory
    pressure. So: models >= 0.8B train with checkpointing and a token budget per
    micro-batch of ~24k (fp32 master) / ~48k (bf16 + 8-bit Adam); tiny models skip
    checkpointing. The memory-shape probe below still shrinks anything that does
    not fit."""
    fp32_master = params_b <= FP32_MASTER_MAX_B
    gc_on = params_b >= 0.8
    if gc_on:
        tok_budget = 24576 if fp32_master else 49152
        if vocab >= 200_000 and not liger:
            tok_budget //= 3
        elif vocab >= 200_000:
            tok_budget = int(tok_budget * 0.75)
    else:
        tok_budget = 16384 if vocab < 100_000 else 8192
    if params_b > 4.0:
        tok_budget = int(tok_budget * 0.75)
    micro = max(1, min(64, tok_budget // max_len))
    try:
        eff_target = int(os.environ.get("FORGE_V2_EFF_BATCH", str(EFF_BATCH_TARGET)))
    except ValueError:
        eff_target = EFF_BATCH_TARGET
    if os.environ.get("FORGE_V2_GC") == "1":
        gc_on = True
    elif os.environ.get("FORGE_V2_GC") == "0":
        gc_on = False
    accum = max(1, math.ceil(max(1, eff_target) / micro))
    if fp32_master:
        optim = "adamw_torch_fused"
    else:
        optim = "paged_adamw_8bit" if bnb_ok else "adamw_torch"
    return Geometry(micro, accum, max_len, gc_on, fp32_master, optim)


def optimizer_state_bytes(model: Any, optim: str) -> int:
    """Bytes the optimizer will allocate lazily at its first step (not visible
    to a forward/backward probe): two fp32 moments per trainable parameter for
    torch AdamW, two 8-bit moments for bitsandbytes' 8-bit AdamW."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per = 2 if "8bit" in optim else 8
    return int(n * per)


def memory_probe(model: Any, *, micro: int, max_len: int, vocab: int, autocast: bool, reserve_bytes: int = 0) -> bool:
    """One synthetic forward/backward at the worst-case micro-batch shape. Fits
    only if the peak plus `reserve_bytes` (optimizer state) leaves 7% headroom."""
    import torch

    device = next(model.parameters()).device
    ids = torch.randint(5, max(6, vocab - 1), (micro, max_len), device=device)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    model.train()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        loss.backward()
        if device.type == "cuda" and reserve_bytes > 0:
            peak = torch.cuda.max_memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            if peak + reserve_bytes > 0.93 * total:
                _event_and_print("sft_v2_memory_probe_reserve", micro=micro, peak_gb=round(peak / 1e9, 1),
                                 reserve_gb=round(reserve_bytes / 1e9, 1), total_gb=round(total / 1e9, 1))
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            return False
        raise
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        _free_cuda()


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _liger_ok(model: Any) -> bool:
    if os.environ.get("FORGE_LIGER", "1") == "0":
        return False
    try:
        import liger_kernel  # noqa: F401
    except Exception:
        return False
    mt = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    return mt in {"llama", "gemma", "gemma2", "gemma3", "gemma3_text", "qwen2", "qwen3", "mistral", "mixtral", "phi3", "smollm3", "olmo2", "granite"}


class _StopAt:
    """Stops training when the remaining wall clock drops below the finish reserve."""

    def __init__(self, deadline: Deadline, finish_reserve_s: float) -> None:
        self.deadline = deadline
        self.finish_reserve_s = finish_reserve_s
        self.fired = False

    def should_stop(self) -> bool:
        return self.deadline.remaining_hard() <= self.finish_reserve_s


def run(
    spec: TaskSpec,
    deadline: Deadline,
    rows: list,
    loaded: Any,
    tokenizer: Any,
    *,
    baseline_summary: Any,
    params_b: float,
    per_gpu_gb: float,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from forge.tasks.fallback import emit_untrained_copy

    t_start = time.monotonic()
    model = loaded.model
    loaded.model = None
    _event_and_print("sft_v2_selected", params_b=round(params_b, 3), hours_remaining=round(deadline.remaining_hard() / 3600, 3))

    # ---- floor first: a valid untrained adapter at the output path ----
    emit_untrained_copy(spec)
    write_artifact_truth(spec.output_dir, ARTIFACT_FLOOR, optimizer_step=0, reason="pretraining_floor")

    # ---- data ----
    assert spec.instruct is not None
    if spec.instruct.output is None:
        docs = prompts.build_completion_documents(rows, spec.instruct)
        tokenized = tokenize.tokenize_completion(docs, tokenizer, EVAL_CAP)
    else:
        examples = prompts.build_instruct_examples(rows, spec.instruct)
        tokenized = tokenize.tokenize_instruct(examples, tokenizer, EVAL_CAP)
    n_raw = len(tokenized)
    tokenized = exact_dedup(tokenized)
    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")
    lengths = [len(ex["input_ids"]) for ex in tokenized]
    model_cap = int(getattr(model.config, "max_position_embeddings", EVAL_CAP) or EVAL_CAP)
    max_len = adaptive_max_len(lengths, ceiling=min(EVAL_CAP, max(256, model_cap)))
    hours_total = deadline.remaining_hard() / 3600.0
    dev_n = dev_size_for(len(tokenized), hours=hours_total)
    train_ex, dev_ex = stratified_split(tokenized, dev_n, seed=7)
    train_ex = [smart_truncate(ex, max_len) for ex in train_ex]
    dev_ex = [smart_truncate(ex, max_len) for ex in dev_ex]
    _event_and_print(
        "sft_v2_data", rows=len(rows), tokenized=n_raw, deduped=len(tokenized), train_n=len(train_ex), dev_n=len(dev_ex),
        max_len=max_len, p50=sorted(lengths)[len(lengths) // 2], p99=sorted(lengths)[min(len(lengths) - 1, int(0.99 * len(lengths)))],
    )

    # ---- model preparation (full weights) ----
    vocab = _vocab(model)
    cfg = getattr(model, "config", None)
    layers = int(getattr(cfg, "num_hidden_layers", 32) or 32)
    hidden = int(getattr(cfg, "hidden_size", 2048) or 2048)
    use_liger = _liger_ok(model)
    if use_liger:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)
        except Exception as exc:  # fused kernels are an optimisation, never a requirement
            _event_and_print("liger_apply_failed", error=f"{type(exc).__name__}: {exc}")
            use_liger = False
    strategy = strategy_for(params_b) or "full"
    geo = choose_geometry(params_b=params_b, max_len=max_len, vocab=vocab, per_gpu_gb=per_gpu_gb, bnb_ok=_bnb_available(),
                          layers=layers, hidden=hidden, liger=use_liger)
    if strategy == "lora":
        from forge.model import attach_lora

        model = attach_lora(model, r=32, alpha=64, dropout=0.05)
        # Adapter training is light: no fp32 master copy, plain fused AdamW,
        # checkpointing only when the probe says the geometry does not fit.
        geo = Geometry(geo.micro_batch, geo.grad_accum, geo.max_len, False, False, "adamw_torch_fused")
    else:
        if geo.fp32_master:
            model = model.float()
        for p in model.parameters():
            p.requires_grad_(True)
    model.config.use_cache = False
    autocast = geo.fp32_master and torch.cuda.is_available() and strategy == "full"
    if not torch.cuda.is_available():
        geo = Geometry(geo.micro_batch, geo.grad_accum, geo.max_len, False, geo.fp32_master, "adamw_torch")

    def _apply_gc(enabled: bool) -> None:
        if not hasattr(model, "gradient_checkpointing_enable"):
            return
        if enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    _apply_gc(geo.gradient_checkpointing)
    if torch.cuda.is_available():
        # Worst-case shape probe: shrink the micro-batch, then enable checkpointing, until it fits.
        ladder = [(geo.micro_batch, geo.gradient_checkpointing)]
        if not geo.gradient_checkpointing:
            ladder += [(geo.micro_batch, True)]
        ladder += [(max(1, geo.micro_batch // 2), True), (1, True)]
        seen = set()
        for micro_c, gc_c in ladder:
            if (micro_c, gc_c) in seen:
                continue
            seen.add((micro_c, gc_c))
            _apply_gc(gc_c)
            ok = memory_probe(model, micro=micro_c, max_len=geo.max_len, vocab=vocab, autocast=autocast,
                              reserve_bytes=optimizer_state_bytes(model, geo.optim))
            _event_and_print("sft_v2_memory_probe", micro=micro_c, gradient_checkpointing=gc_c, fits=ok)
            if ok:
                eff_target = geo.eff_batch
                geo = Geometry(micro_c, max(1, math.ceil(eff_target / micro_c)), geo.max_len, gc_c, geo.fp32_master, geo.optim)
                break
    telemetry.set_meta(
        handler="sft_v2", strategy=strategy, params_b=round(params_b, 3), seq_len=max_len, batch=geo.micro_batch,
        grad_accum=geo.grad_accum, eff_batch=geo.eff_batch, gradient_checkpointing=geo.gradient_checkpointing,
        fp32_master=geo.fp32_master, optim=geo.optim, liger=use_liger, vocab=vocab, train_n=len(train_ex), val_n=len(dev_ex),
    )

    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    from forge.model import median_weight_rms

    gns = None
    try:
        gns = getattr(baseline_summary, "gradient_noise_scale", None)
    except Exception:
        gns = None
    if strategy == "lora":
        # production trains the adapter at 1.5e-4 with effective batch 16; the prior
        # sits just above it and scales with sqrt(batch) (the Smol sweep chose
        # 4.4e-4 at batch 84, consistent with this law).
        prior_lr = 2.0e-4 * math.sqrt(max(1, geo.eff_batch) / 16.0)
    else:
        prior_lr = lr_probe.analytic_lr(weight_rms=_layer_weight_rms(model) or median_weight_rms(model), params_b=params_b,
                                        eff_batch=geo.eff_batch, gradient_noise_scale=gns)

    try:
        _lr_override = float(os.environ.get("FORGE_V2_LR", "") or 0.0)
    except ValueError:
        _lr_override = 0.0
    if _lr_override > 0:
        _event_and_print("sft_v2_lr_override", prior_lr=prior_lr, override=_lr_override)
        prior_lr = _lr_override

    def optimizer_factory(params: list, lr: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

    def make_args(num_epochs: float, learning_rate: float, warmup_steps: int, micro: int, accum: int, seed: int = 7) -> TrainingArguments:
        kwargs = dict(
            output_dir=workdir(spec), overwrite_output_dir=True, num_train_epochs=num_epochs,
            per_device_train_batch_size=micro, gradient_accumulation_steps=accum, learning_rate=learning_rate,
            lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1}, warmup_steps=warmup_steps,
            weight_decay=0.0, optim=geo.optim, max_grad_norm=1.0, bf16=True, fp16=False,
            gradient_checkpointing=False,  # enabled on the model directly above
            logging_steps=5, save_strategy="no", eval_strategy="no", report_to=[], remove_unused_columns=False,
            dataloader_num_workers=2, disable_tqdm=True, seed=seed, train_sampling_strategy="group_by_length", length_column_name="length",
            neftune_noise_alpha=_neftune_alpha(baseline_summary),
        )
        return TrainingArguments(**compatible_dataclass_kwargs(TrainingArguments, kwargs, allow_removed={"overwrite_output_dir"}))

    def with_lengths(exs: list[dict]) -> Dataset:
        return Dataset.from_list([{**ex, "length": len(ex["input_ids"])} for ex in exs])

    train_ds = with_lengths(train_ex)

    # ---- learning-rate probe on cached batches from the real dataloader ----
    probe_trainer = Trainer(model=model, args=make_args(1.0, prior_lr, 10, geo.micro_batch, geo.grad_accum), train_dataset=train_ds, data_collator=collator)
    steps_per_epoch = max(1, len(train_ex) // geo.eff_batch)
    remaining = deadline.remaining_hard()
    probe_budget = min(0.15 * remaining, 480.0) if _lr_override <= 0 else 0.0  # fixed LR: timing steps only
    batches: list[dict[str, Any]] = []
    # Cache enough distinct micro-batches that a 100-step probe never re-sees a
    # batch (re-seen batches are memorised and bias the sweep toward the fastest
    # memoriser rather than the best learner).
    n_probe_batches = min(400, max(100 * geo.grad_accum, 60))
    lr, t_per_step, diag = prior_lr, None, {}
    if remaining > MIN_TASK_SECONDS_FOR_V2:
        try:
            for i, b in enumerate(probe_trainer.get_train_dataloader()):
                if i >= n_probe_batches:
                    break
                batches.append({k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in b.items() if k != "length"})
            import random as _random

            _random.Random(11).shuffle(batches)  # the sampler front-loads the longest rows; timing needs a representative mix
            lr, t_per_step, diag = lr_probe.lr_search(
                model, batches, center_lr=prior_lr, budget_s=probe_budget, grad_accum=geo.grad_accum,
                optimizer_factory=optimizer_factory, autocast_bf16=autocast, log=lambda n, d: _event_and_print(n, **_flat(d)),
            )
        except Exception as exc:
            if _is_oom(exc):
                _event_and_print("lr_probe_oom", micro=geo.micro_batch)
                _free_cuda()
                geo = Geometry(max(1, geo.micro_batch // 2), geo.grad_accum * 2, geo.max_len, True, geo.fp32_master, geo.optim)
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                lr, t_per_step = prior_lr, None
            else:
                _event_and_print("lr_probe_failed", error=f"{type(exc).__name__}: {exc}")
                lr, t_per_step = prior_lr, None
    del probe_trainer, batches
    gc.collect()
    _free_cuda()

    # ---- adaptive effective batch: guarantee enough optimizer updates in short tasks ----
    if t_per_step:
        samples_per_s = geo.eff_batch / t_per_step
        window0 = deadline.remaining_hard() - (EXPORT_RESERVE_S + 2.0 * _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
                                                + _dev_pass_estimate(len(dev_ex), t_per_step, geo))
        affordable = max(0.0, window0) * 0.85 * samples_per_s
        try:
            min_updates = int(os.environ.get("FORGE_V2_MIN_UPDATES", "500"))
        except ValueError:
            min_updates = 500
        eff_new = int(affordable / max(1, min_updates))
        eff_new = max(16, min(geo.eff_batch, eff_new))
        eff_new = max(geo.micro_batch, (eff_new // geo.micro_batch) * geo.micro_batch)
        if eff_new != geo.eff_batch:
            scale = math.sqrt(eff_new / geo.eff_batch)
            _event_and_print("sft_v2_eff_batch", old=geo.eff_batch, new=eff_new, lr_old=lr, lr_new=lr * scale,
                             affordable_samples=int(affordable), samples_per_s=round(samples_per_s, 2))
            t_per_step = t_per_step * eff_new / geo.eff_batch
            lr = lr * scale
            geo = Geometry(geo.micro_batch, eff_new // geo.micro_batch, geo.max_len, geo.gradient_checkpointing, geo.fp32_master, geo.optim)

    # ---- planning constants ----
    dev_eval_est = _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
    export_reserve = 60.0 if strategy == "lora" else EXPORT_RESERVE_S
    finish_reserve = export_reserve + 1.5 * dev_eval_est + _dev_pass_estimate(len(dev_ex), t_per_step, geo)
    max_epochs_total = 4.0 if len(train_ex) < 10000 else 3.0
    steps_per_epoch = max(1, len(train_ex) // geo.eff_batch)

    # ---- training with validator-style dev selection ----
    pool = selection.SnapshotPool(k=8)
    stop_at = _StopAt(deadline, finish_reserve)
    state = {"best": float("inf"), "best_step": 0, "overfit": 0, "persisted_best": float("inf"), "persisted_step": 0,
             "eval_count": 0, "last_persist_at": 0.0, "step_offset": 0, "epochs_done": 0.0, "eval_secs": 0.0,
             "persist_secs": 0.0, "stopped_overfit": False, "last_phase_complete": False, "restarts": 0, "last_eval_step": -1}
    eval_bs = selection.eval_batch_for(max_len=max_len, vocab=vocab)
    dev_rows_for_selection = dev_ex

    def eval_dev(m: Any) -> float:
        t0 = time.perf_counter()
        loss, _ = selection.per_example_dev_loss(m, dev_rows_for_selection, collator, batch_size=eval_bs, autocast_bf16=autocast)
        state["eval_secs"] += time.perf_counter() - t0
        return loss

    def persist(m: Any, cpu_state: dict[str, Any] | None, step: int, truth: str, reason: str) -> bool:
        t0 = time.perf_counter()
        try:
            save_adapter(m, tokenizer, spec.output_dir, artifact_truth=truth, optimizer_step=step, truth_reason=reason,
                         state_dict=cpu_state)
            return True
        except Exception as exc:
            _event_and_print("sft_v2_persist_failed", step=step, error=f"{type(exc).__name__}: {exc}")
            return False
        finally:
            state["persist_secs"] += time.perf_counter() - t0

    class SelectCallback(TrainerCallback):
        def __init__(self, eval_every: int, total_steps: int) -> None:
            self.eval_every = eval_every
            self.total_steps = total_steps
            self.governed = False

        def on_step_end(self, args, st, control, **kw):  # noqa: ANN001
            local_step = int(st.global_step)
            step = state["step_offset"] + local_step
            m = kw.get("model")
            if stop_at.should_stop():
                control.should_training_stop = True
            end_step = int(getattr(st, "max_steps", 0) or self.total_steps)
            due = local_step > 0 and (local_step % self.eval_every == 0 or local_step >= end_step)
            if not due or m is None or not dev_rows_for_selection:
                return control
            t0 = time.perf_counter()
            loss = eval_dev(m)
            state["last_eval_step"] = step
            dt = time.perf_counter() - t0
            state["eval_count"] += 1
            telemetry.eval_point(step, loss)
            improved = loss < state["best"]
            state["phase_evals"] = state.get("phase_evals", 0) + 1
            if improved:
                state["best"], state["best_step"] = loss, step
                state["overfit"] = 0
            elif loss > state["best"] * 1.05 and state["phase_evals"] > 1:
                state["overfit"] += 1
                if state["overfit"] >= 3:
                    _event_and_print("sft_v2_early_stop", step=step, best=state["best"], best_step=state["best_step"])
                    state["stopped_overfit"] = True
                    control.should_training_stop = True
            else:
                state["overfit"] = 0
            admitted = pool.consider(m, loss, step)
            since_persist = time.monotonic() - state["last_persist_at"]
            if improved and (since_persist > 120.0 or loss < state["persisted_best"] * 0.995 or state["persisted_best"] == float("inf")):
                snap = pool.best()["state"] if (admitted and pool.best() and pool.best()["step"] == step) else selection.snapshot_trainable(m)
                if persist(m, snap, step, ARTIFACT_PARTIAL_TRAINED_BEST, "dev_minimum"):
                    state["persisted_best"], state["persisted_step"] = loss, step
                    state["last_persist_at"] = time.monotonic()
            _event_and_print("sft_v2_eval", step=step, dev_loss=round(loss, 6), best=round(state["best"], 6), eval_s=round(dt, 1),
                             pool=len(pool.items), improved=improved)
            if not self.governed and t_step:
                # keep evaluation (incl. persistence) under ~6% of training time
                self.governed = True
                per_eval = dt + 0.5 * (state["persist_secs"] / max(1, state["eval_count"]))
                widened = int(math.ceil(per_eval / (0.06 * max(t_step, 1e-3))))
                if widened > self.eval_every:
                    _event_and_print("sft_v2_eval_governor", eval_every=widened, eval_s=round(dt, 1))
                    self.eval_every = widened
            return control

    _event_and_print("sft_v2_train_start", elapsed_s=round(time.monotonic() - t_start, 1), lr=lr, prior_lr=prior_lr,
                     t_per_step=t_per_step, steps_per_epoch=steps_per_epoch, finish_reserve_s=round(finish_reserve, 1),
                     **_flat({"probe": diag}))
    trainer = None
    t_step = t_per_step
    phase = 0
    oom_retries = 0
    while phase < 4:
        remaining = deadline.remaining_hard()
        window = remaining - finish_reserve
        min_window = max(240.0, 25.0 * t_step) if t_step else 240.0
        if state["stopped_overfit"]:
            # One warm restart from the best snapshot at a lower learning rate
            # when plenty of budget remains; a second overfit stop ends training.
            if state["restarts"] >= 1 or window < max(600.0, 60.0 * (t_step or 10.0)) or pool.best() is None:
                break
            selection.load_state(model, pool.best()["state"])
            state["restarts"] += 1
            state["stopped_overfit"] = False
            state["overfit"] = 0
            lr = lr * 0.3
            _event_and_print("sft_v2_restart", lr=lr, from_step=pool.best()["step"], window_s=round(window, 1))
        if phase > 0 and window < min_window:
            break
        # Each phase is a fully annealed cosine cycle of at most one epoch (the
        # affordable fraction when less fits), so every phase ends on a low
        # learning rate: the phase endpoints are the checkpoints the soup wants.
        # Without timing (probe OOM) phase 0 is one epoch and the clock guard
        # stops it if the epoch does not fit.
        if os.environ.get("FORGE_V2_RICH_EPOCHS") == "1" and t_step and steps_per_epoch * t_step * 6.0 < window and max_epochs_total < 8.0:
            # an epoch costs < 1/6 of the window: keep training (annealed cycles at
            # decreasing peak LR); the overfit early stop ends it when dev turns.
            max_epochs_total = 8.0
            _event_and_print("sft_v2_epoch_cap_raised", max_epochs=max_epochs_total, epoch_s=round(steps_per_epoch * t_step, 1))
        epochs_left = max_epochs_total - state["epochs_done"]
        if t_step:
            achievable = window * 0.85 / t_step
            if phase == 0 or state["restarts"]:
                # first cycle: one fully annealed epoch (the strongest single
                # candidate on small data), also the timing measurement
                epochs = min(epochs_left, 1.0, achievable / steps_per_epoch)
            else:
                # then one continuous cosine over everything that still fits:
                # repeated one-epoch cycles restart at a high LR each time and
                # the dev loss spends most of each cycle recovering (Qwen2.5 run)
                epochs = min(epochs_left, achievable / steps_per_epoch)
        else:
            epochs = min(epochs_left, 1.0) if phase == 0 else 0.0
        epochs = round(max(0.0, epochs), 3)
        total_steps = int(math.ceil(steps_per_epoch * epochs))
        if total_steps < 1 or (phase > 0 and total_steps < max(10, int(0.15 * steps_per_epoch))):
            break
        warmup = min(200, max(3, int(0.03 * total_steps)))
        eval_every = max(10, min(max(1, steps_per_epoch // 8), max(1, total_steps // 8)))
        phase_lr = lr if (phase == 0 or state["restarts"]) else lr * max(0.1, 0.5 ** phase)
        state["overfit"] = 0
        state["phase_evals"] = 0
        _event_and_print("sft_v2_plan", phase=phase, lr=phase_lr, t_per_step=t_step, epochs=epochs, total_steps=total_steps,
                         warmup=warmup, eval_every=eval_every, window_s=round(window, 1), epochs_done=round(state["epochs_done"], 3))
        cb = SelectCallback(eval_every, total_steps)
        trainer = Trainer(
            model=model, args=make_args(epochs, phase_lr, warmup, geo.micro_batch, geo.grad_accum, seed=7 + phase),
            train_dataset=train_ds, data_collator=collator, callbacks=[cb, telemetry.make_trainer_callback(spec.output_dir)],
        )
        t_phase0 = time.monotonic()
        eval0, pers0 = state["eval_secs"], state["persist_secs"]
        try:
            trainer.train()
        except Exception as exc:
            if not _is_oom(exc):
                raise
            step = int(getattr(trainer.state, "global_step", 0) or 0)
            _event_and_print("sft_v2_train_oom", phase=phase, step=step, micro=geo.micro_batch,
                             gradient_checkpointing=geo.gradient_checkpointing)
            try:
                trainer.optimizer = None  # release optimizer state before retrying
            except Exception:
                pass
            del trainer
            trainer = None
            _free_cuda()
            # Keep whatever was trained; shrink the geometry and continue in a new phase.
            state["step_offset"] += step
            state["epochs_done"] += step / steps_per_epoch
            if step > 0:
                phase += 1
            if oom_retries >= 3:
                break
            oom_retries += 1
            new_micro = max(1, geo.micro_batch // 2) if (geo.gradient_checkpointing or geo.micro_batch > 1) else 1
            geo = Geometry(new_micro, max(1, geo.eff_batch // new_micro), geo.max_len, True, geo.fp32_master, geo.optim)
            _apply_gc(True)
            continue
        if trainer is None:
            continue
        steps_done = int(getattr(trainer.state, "global_step", 0) or 0)
        planned_steps = int(getattr(trainer.state, "max_steps", 0) or total_steps)
        state["step_offset"] += steps_done
        state["epochs_done"] += steps_done / steps_per_epoch
        state["last_phase_complete"] = steps_done >= planned_steps
        try:  # release the optimizer state (fp32 Adam moments) before the next phase / soup / dev pass
            trainer.optimizer = None
            trainer.lr_scheduler = None
        except Exception:
            pass
        del trainer
        trainer = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        _free_cuda()
        phase_wall = (time.monotonic() - t_phase0) - (state["eval_secs"] - eval0) - (state["persist_secs"] - pers0)
        if steps_done >= 5 and phase_wall > 0:
            t_step = phase_wall / steps_done
        _event_and_print("sft_v2_phase_end", phase=phase, steps=steps_done, planned=planned_steps, epochs_done=round(state["epochs_done"], 3),
                         t_per_step_measured=round(t_step, 3) if t_step else None, best=state["best"], best_step=state["best_step"],
                         remaining_s=round(deadline.remaining_hard(), 1))
        phase += 1
        if steps_done < planned_steps and not state["stopped_overfit"]:
            break  # stopped by the clock (or trainer); no time for another phase
    final_step = state["step_offset"]
    _event_and_print("sft_v2_train_end", step=final_step, best=state["best"], best_step=state["best_step"],
                     phases=phase, remaining_s=round(deadline.remaining_hard(), 1))
    if final_step <= 0:
        return

    # ---- final evaluation of the last weights (may be the best) ----
    if dev_rows_for_selection and state["last_eval_step"] != final_step and deadline.remaining_hard() > export_reserve + dev_eval_est:
        last_loss = eval_dev(model)
        telemetry.eval_point(final_step, last_loss)
        if last_loss < state["best"]:
            state["best"], state["best_step"] = last_loss, final_step
        pool.consider(model, last_loss, final_step)

    # ---- greedy soup ----
    chosen_state: dict[str, Any] | None = None
    chosen_loss = state["best"]
    chosen_step = state["best_step"]
    soup_budget = max(0.0, 0.5 * (deadline.remaining_hard() - export_reserve - _dev_pass_estimate(len(dev_ex), t_per_step, geo)))
    measured_eval = state["eval_secs"] / max(1, state["eval_count"]) if state["eval_count"] else dev_eval_est
    if pool.items and dev_rows_for_selection and len(pool.items) >= 2 and soup_budget < 2.5 * measured_eval:
        _event_and_print("sft_v2_soup_skipped", budget_s=round(soup_budget, 1), eval_s=round(measured_eval, 1))
        pool.items = pool.items[:1]  # keep the best only: the soup would not fit a single trial
    if pool.items and dev_rows_for_selection:
        try:
            soup_loss, soup_state, members = selection.greedy_soup(
                model, pool, eval_dev, time_budget_s=soup_budget, log=lambda n, d: _event_and_print(n, **_flat(d))
            )
            if soup_state is not None and soup_loss <= chosen_loss + 1e-9:
                chosen_state, chosen_loss, chosen_step = soup_state, soup_loss, final_step if members > 1 else pool.best()["step"]
            elif pool.best() is not None:
                chosen_state = pool.best()["state"]
                selection.load_state(model, chosen_state)
                chosen_loss, chosen_step = pool.best()["loss"], pool.best()["step"]
        except Exception as exc:
            _event_and_print("sft_v2_soup_failed", error=f"{type(exc).__name__}: {exc}")
            if pool.best() is not None:
                chosen_state = pool.best()["state"]
                selection.load_state(model, chosen_state)
    elif pool.best() is not None:
        chosen_state = pool.best()["state"]
        selection.load_state(model, chosen_state)

    # ---- dev pass: one low-LR epoch over the held-out slice from the chosen weights ----
    dev_pass_s = _dev_pass_estimate(len(dev_ex), t_per_step, geo)
    if dev_ex and len(dev_ex) >= 100 and deadline.remaining_hard() > export_reserve + dev_pass_s + 30 and os.environ.get("FORGE_DEV_PASS", "1") != "0":
        try:
            _dev_pass(model, dev_ex, collator, lr=lr * 0.25, geo=geo, autocast=autocast, optimizer_factory=optimizer_factory)
            chosen_state = None  # weights in the model are now the final ones
            _event_and_print("sft_v2_dev_pass_done", rows=len(dev_ex), lr=lr * 0.25)
        except Exception as exc:
            _event_and_print("sft_v2_dev_pass_failed", error=f"{type(exc).__name__}: {exc}")
            if pool.best() is not None:
                chosen_state = pool.best()["state"]
                selection.load_state(model, chosen_state)

    # ---- final save (bf16 weights) ----
    truth = ARTIFACT_COMPLETE_BEST if state["last_phase_complete"] else ARTIFACT_PARTIAL_TRAINED_BEST
    if final_step <= 0:
        return
    final_state = chosen_state if chosen_state is not None else selection.snapshot_trainable(model)
    ok = persist(model, final_state, chosen_step or final_step, truth, "sft_v2_final")
    _event_and_print("sft_v2_final", saved=ok, chosen_loss=chosen_loss, chosen_step=chosen_step, truth=truth,
                    remaining_s=round(deadline.remaining_hard(), 1))
    telemetry.write_into(spec.output_dir)


def trained_artifact_present(spec: TaskSpec) -> bool:
    """True when v2 already persisted a trained (non-floor) artifact."""
    import json

    try:
        with open(os.path.join(spec.output_dir, "forge_artifact_truth.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("truth") in (ARTIFACT_PARTIAL_TRAINED_BEST, ARTIFACT_COMPLETE_BEST)
    except Exception:
        return False


def _layer_weight_rms(model: Any) -> float | None:
    """Median RMS of 2-D weights inside transformer blocks (embeddings and the
    output head excluded, as in the public LR law)."""
    try:
        vals = []
        for name, p in model.named_parameters():
            if p.dim() != 2 or "embed" in name or "lm_head" in name or "layers." not in name:
                continue
            vals.append(p.detach().float().pow(2).mean().sqrt().item())
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]
    except Exception:
        return None


def _neftune_alpha(summary: Any) -> float | None:
    try:
        gns = getattr(summary, "gradient_noise_scale", None)
        if gns is not None and float(gns) > 1.0:
            return 5.0
    except Exception:
        pass
    return 1.0


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "_"))
        else:
            out[key] = v
    return out


def _estimate_eval_seconds(n_dev: int, t_per_step: float | None, geo: Geometry) -> float:
    if n_dev <= 0:
        return 0.0
    if t_per_step is None:
        return 60.0
    # forward-only per row ~ 1/3 of a training micro-row; batched eval is cheaper still
    per_row = (t_per_step / max(1, geo.eff_batch)) / 3.0
    return max(5.0, n_dev * per_row + 5.0)


def _dev_pass_estimate(n_dev: int, t_per_step: float | None, geo: Geometry) -> float:
    if n_dev <= 0:
        return 0.0
    if t_per_step is None:
        return 120.0
    return max(20.0, 1.5 * (n_dev / max(1, geo.eff_batch)) * t_per_step + 20.0)


def _dev_pass(model: Any, dev: list[dict], collator: Any, *, lr: float, geo: Geometry, autocast: bool, optimizer_factory: Callable) -> None:
    import torch

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer_factory(params, lr)
    device = next(model.parameters()).device
    rows = list(dev)
    micro = geo.micro_batch
    accum = geo.grad_accum
    n_micro = 0
    for start in range(0, len(rows), micro):
        batch = collator(rows[start : start + micro])
        batch = {k: v.to(device) for k, v in batch.items()}
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
        else:
            loss = model(**batch).loss / accum
        loss.backward()
        n_micro += 1
        if n_micro % accum == 0 or start + micro >= len(rows):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    del opt
    model.zero_grad(set_to_none=True)
    _free_cuda()
