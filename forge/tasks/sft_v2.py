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
MAX_PARAMS_B = 5.0
FP32_MASTER_MAX_B = 2.6
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
    if params_b <= 0 or params_b > MAX_PARAMS_B:
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


def choose_geometry(*, params_b: float, max_len: int, vocab: int, per_gpu_gb: float, bnb_ok: bool) -> Geometry:
    fp32_master = params_b <= FP32_MASTER_MAX_B
    # Tokens per micro-batch: activations + logits are the memory driver. Large
    # vocabularies (Gemma 256k) materialise big logit tensors, so budget fewer
    # tokens per step there. Adam state for fp32-master models also eats memory.
    if vocab >= 200_000:
        tok_budget = 6144 if fp32_master else 8192
    elif vocab >= 100_000:
        tok_budget = 12288
    else:
        tok_budget = 16384
    if params_b > 3.5:
        tok_budget //= 2
    micro = max(1, min(64, tok_budget // max_len))
    try:
        eff_target = int(os.environ.get("FORGE_V2_EFF_BATCH", str(EFF_BATCH_TARGET)))
    except ValueError:
        eff_target = EFF_BATCH_TARGET
    accum = max(1, math.ceil(max(1, eff_target) / micro))
    gc_on = params_b >= 1.2
    if fp32_master:
        optim = "adamw_torch_fused"
    else:
        optim = "paged_adamw_8bit" if bnb_ok else "adamw_torch"
    return Geometry(micro, accum, max_len, gc_on, fp32_master, optim)


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
    geo = choose_geometry(params_b=params_b, max_len=max_len, vocab=vocab, per_gpu_gb=per_gpu_gb, bnb_ok=_bnb_available())
    if geo.fp32_master:
        model = model.float()
    for p in model.parameters():
        p.requires_grad_(True)
    model.config.use_cache = False
    if geo.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    use_liger = _liger_ok(model)
    if use_liger:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)
        except Exception as exc:  # fused kernels are an optimisation, never a requirement
            _event_and_print("liger_apply_failed", error=f"{type(exc).__name__}: {exc}")
            use_liger = False
    autocast = geo.fp32_master and torch.cuda.is_available()
    if not torch.cuda.is_available():
        geo = Geometry(geo.micro_batch, geo.grad_accum, geo.max_len, False, geo.fp32_master, "adamw_torch")
    telemetry.set_meta(
        handler="sft_v2", strategy="full", params_b=round(params_b, 3), seq_len=max_len, batch=geo.micro_batch,
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
    prior_lr = lr_probe.analytic_lr(weight_rms=median_weight_rms(model), params_b=params_b, eff_batch=geo.eff_batch, gradient_noise_scale=gns)

    def optimizer_factory(params: list, lr: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

    def make_args(num_epochs: float, learning_rate: float, warmup_steps: int, micro: int, accum: int) -> TrainingArguments:
        kwargs = dict(
            output_dir=workdir(spec), overwrite_output_dir=True, num_train_epochs=num_epochs,
            per_device_train_batch_size=micro, gradient_accumulation_steps=accum, learning_rate=learning_rate,
            lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.25}, warmup_steps=warmup_steps,
            weight_decay=0.0, optim=geo.optim, max_grad_norm=1.0, bf16=True, fp16=False,
            gradient_checkpointing=False,  # enabled on the model directly above
            logging_steps=10, save_strategy="no", eval_strategy="no", report_to=[], remove_unused_columns=False,
            dataloader_num_workers=2, disable_tqdm=True, seed=7, train_sampling_strategy="group_by_length", length_column_name="length",
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
    probe_budget = min(0.15 * remaining, 600.0)
    batches: list[dict[str, Any]] = []
    n_probe_batches = min(60, 100 * geo.grad_accum)
    lr, t_per_step, diag = prior_lr, None, {}
    if remaining > MIN_TASK_SECONDS_FOR_V2:
        try:
            for i, b in enumerate(probe_trainer.get_train_dataloader()):
                if i >= n_probe_batches:
                    break
                batches.append({k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in b.items() if k != "length"})
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

    # ---- time-aware epoch planning ----
    remaining = deadline.remaining_hard()
    dev_eval_est = _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
    finish_reserve = EXPORT_RESERVE_S + 2.0 * dev_eval_est + _dev_pass_estimate(len(dev_ex), t_per_step, geo)
    train_window = max(60.0, remaining - finish_reserve)
    if t_per_step:
        achievable = train_window * 0.85 / t_per_step
        coverage = achievable / steps_per_epoch
        if coverage < 0.5 and len(train_ex) > 2000:
            target = max(2000, int(coverage * len(train_ex)))
            if target < len(train_ex):
                train_ex = random_subsample(train_ex, target, seed=7)
                train_ds = with_lengths(train_ex)
                steps_per_epoch = max(1, len(train_ex) // geo.eff_batch)
                _event_and_print("sft_v2_subsample", coverage=round(coverage, 3), kept=len(train_ex))
                achievable = train_window * 0.85 / t_per_step
        max_epochs = 4.0 if len(train_ex) < 10000 else 3.0
        epochs = round(max(1.0, min(max_epochs, 1.25 * achievable / steps_per_epoch)), 2)
    else:
        epochs = 2.0
    total_steps = int(steps_per_epoch * epochs)
    warmup = min(200, max(10, int(0.03 * total_steps)))
    eval_every = max(20, steps_per_epoch // 8)
    _event_and_print(
        "sft_v2_plan", lr=lr, prior_lr=prior_lr, t_per_step=t_per_step, epochs=epochs, steps_per_epoch=steps_per_epoch,
        total_steps=total_steps, warmup=warmup, eval_every=eval_every, train_window_s=round(train_window, 1),
        finish_reserve_s=round(finish_reserve, 1), **_flat({"probe": diag}),
    )

    # ---- training with validator-style dev selection ----
    pool = selection.SnapshotPool(k=8)
    stop_at = _StopAt(deadline, finish_reserve)
    state = {"best": float("inf"), "best_step": 0, "last_eval_at": time.monotonic(), "eval_every": eval_every,
             "overfit": 0, "persisted_best": float("inf"), "persisted_step": 0, "eval_count": 0, "governed": False,
             "last_persist_at": 0.0}
    eval_bs = max(1, min(16, (16384 // max(1, max_len))))
    dev_rows_for_selection = dev_ex

    def eval_dev(m: Any) -> float:
        loss, _ = selection.per_example_dev_loss(m, dev_rows_for_selection, collator, batch_size=eval_bs, autocast_bf16=autocast)
        return loss

    def persist(m: Any, cpu_state: dict[str, Any] | None, step: int, truth: str, reason: str) -> bool:
        try:
            save_adapter(m, tokenizer, spec.output_dir, artifact_truth=truth, optimizer_step=step, truth_reason=reason,
                         state_dict=cpu_state)
            return True
        except Exception as exc:
            _event_and_print("sft_v2_persist_failed", step=step, error=f"{type(exc).__name__}: {exc}")
            return False

    class SelectCallback(TrainerCallback):
        def on_step_end(self, args, st, control, **kw):  # noqa: ANN001
            step = int(st.global_step)
            m = kw.get("model")
            if stop_at.should_stop():
                control.should_training_stop = True
            due = step > 0 and step % state["eval_every"] == 0
            if not due or m is None or not dev_rows_for_selection:
                return control
            t0 = time.perf_counter()
            loss = eval_dev(m)
            dt = time.perf_counter() - t0
            state["eval_count"] += 1
            telemetry.eval_point(step, loss)
            improved = loss < state["best"]
            if improved:
                state["best"], state["best_step"] = loss, step
                state["overfit"] = 0
            elif loss > state["best"] * 1.05:
                state["overfit"] += 1
                if state["overfit"] >= 3:
                    _event_and_print("sft_v2_early_stop", step=step, best=state["best"], best_step=state["best_step"])
                    control.should_training_stop = True
            else:
                state["overfit"] = 0
            admitted = pool.consider(m, loss, step)
            since_persist = time.monotonic() - state["last_persist_at"]
            # Persist the new minimum unless we persisted very recently and the
            # gain is tiny; the final save always lands the best weights anyway.
            if improved and (since_persist > 120.0 or loss < state["persisted_best"] * 0.995 or state["persisted_best"] == float("inf")):
                snap = pool.best()["state"] if (admitted and pool.best() and pool.best()["step"] == step) else selection.snapshot_trainable(m)
                if persist(m, snap, step, ARTIFACT_PARTIAL_TRAINED_BEST, "dev_minimum"):
                    state["persisted_best"], state["persisted_step"] = loss, step
                    state["last_persist_at"] = time.monotonic()
            _event_and_print("sft_v2_eval", step=step, dev_loss=round(loss, 6), best=round(state["best"], 6), eval_s=round(dt, 1),
                            pool=len(pool.items), improved=improved)
            # eval-time governor: keep evaluation under ~10% of the remaining window
            if not state["governed"]:
                state["governed"] = True
                remaining_w = max(1.0, deadline.remaining_hard() - finish_reserve)
                max_evals = max(3, int(remaining_w * 0.10 / max(dt, 1e-3)))
                remaining_steps = max(1, total_steps - step)
                widened = max(state["eval_every"], remaining_steps // max_evals)
                if widened > state["eval_every"]:
                    _event_and_print("sft_v2_eval_governor", eval_every=widened, eval_s=round(dt, 1))
                    state["eval_every"] = widened
            return control

    trainer = Trainer(
        model=model, args=make_args(epochs, lr, warmup, geo.micro_batch, geo.grad_accum), train_dataset=train_ds,
        data_collator=collator, callbacks=[SelectCallback(), telemetry.make_trainer_callback(spec.output_dir)],
    )
    _event_and_print("sft_v2_train_start", elapsed_s=round(time.monotonic() - t_start, 1))
    try:
        trainer.train()
    except Exception as exc:
        if not _is_oom(exc):
            raise
        step = int(getattr(trainer.state, "global_step", 0) or 0)
        _event_and_print("sft_v2_train_oom", step=step, micro=geo.micro_batch)
        if step == 0 and geo.micro_batch > 1:
            _free_cuda()
            geo = Geometry(max(1, geo.micro_batch // 2), geo.grad_accum * 2, geo.max_len, True, geo.fp32_master, geo.optim)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            trainer = Trainer(
                model=model, args=make_args(epochs, lr, warmup, geo.micro_batch, geo.grad_accum), train_dataset=train_ds,
                data_collator=collator, callbacks=[SelectCallback(), telemetry.make_trainer_callback(spec.output_dir)],
            )
            trainer.train()
        elif step == 0:
            raise
    final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    _event_and_print("sft_v2_train_end", step=final_step, best=state["best"], best_step=state["best_step"],
                    remaining_s=round(deadline.remaining_hard(), 1))
    if final_step <= 0:
        return

    # ---- final evaluation of the last weights (may be the best) ----
    if dev_rows_for_selection and deadline.remaining_hard() > EXPORT_RESERVE_S + dev_eval_est:
        last_loss = eval_dev(model)
        telemetry.eval_point(final_step, last_loss)
        if last_loss < state["best"]:
            state["best"], state["best_step"] = last_loss, final_step
        pool.consider(model, last_loss, final_step)

    # ---- greedy soup ----
    chosen_state: dict[str, Any] | None = None
    chosen_loss = state["best"]
    chosen_step = state["best_step"]
    soup_budget = max(0.0, 0.5 * (deadline.remaining_hard() - EXPORT_RESERVE_S - _dev_pass_estimate(len(dev_ex), t_per_step, geo)))
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
    if dev_ex and len(dev_ex) >= 100 and deadline.remaining_hard() > EXPORT_RESERVE_S + dev_pass_s + 30 and os.environ.get("FORGE_DEV_PASS", "1") != "0":
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
    truth = ARTIFACT_COMPLETE_BEST if final_step >= int(getattr(trainer.state, "max_steps", 0) or 0) else ARTIFACT_PARTIAL_TRAINED_BEST
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
