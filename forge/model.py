"""Base-model resolution and loading.

The base model is pre-staged read-only under `/cache/models/...`; its identity is
often scrubbed (`config.json._name_or_path` removed) and any LoRA chain is
pre-merged by the validator's downloader. So we resolve it from the local path
only — never the network — and introspect the config/tokenizer rather than
trusting the `--model` string.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
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


class LocalModelLoadError(RuntimeError):
    """A staged model could not be loaded without network access."""


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
    config_data = _preflight_model_dir(model_dir)
    dtype = pick_dtype()

    tokenizer = _load_tokenizer_native_first(AutoTokenizer, model_dir, config_data)
    _fix_special_tokens(tokenizer)
    # Left padding for generation-time tasks (GRPO rollouts); right padding for
    # loss-only SFT/DPO so the last real token isn't buried behind pads.
    tokenizer.padding_side = "left" if for_generation else "right"

    # Flash-attn is intentionally not requested: the evaluator disables it and
    # not every base ships kernels for it. Prefer sdpa, fall back to eager for
    # architectures that don't implement it.
    # Transformers 5 renamed the loader keyword to ``dtype``.  Passing the old
    # alias still works today but emits on every task and is scheduled for
    # removal; the submission runtime is pinned to the v5 API.
    common = dict(local_files_only=True, dtype=dtype, low_cpu_mem_usage=True)
    # The validator allocates 2-8 GPUs for larger models (and multiplies the
    # effective size for DPO/GRPO/KL). Shard across them when there is more than
    # one, since a boss-round model will not fit on a single card.
    sharded = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if sharded:
        common["device_map"] = "auto"

    model = _load_model_native_first(
        AutoModelForCausalLM, model_dir, config_data=config_data, common=common
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available() and not sharded:
        model = model.to("cuda")
    return LoadedModel(model=model, tokenizer=tokenizer, model_dir=model_dir, dtype=dtype)


def _preflight_model_dir(model_dir: str) -> dict[str, Any]:
    """Validate definite local-cache failures before allocating model memory."""
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalModelLoadError(f"invalid local config {config_path!r}: {exc}") from exc
    if not isinstance(config, dict):
        raise LocalModelLoadError(f"local config {config_path!r} is not a JSON object")

    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    found_index = False
    for index_name in index_names:
        index_path = os.path.join(model_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        found_index = True
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("missing non-empty weight_map")
            shard_names = sorted({str(name) for name in weight_map.values()})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise LocalModelLoadError(f"invalid local weight index {index_path!r}: {exc}") from exc
        # Validate the index path lexically.  Hugging Face cache snapshots use
        # legitimate symlinks into a sibling ``blobs`` directory, so comparing
        # realpaths would reject normal staged caches.
        root_abs = os.path.abspath(model_dir)
        invalid_paths: list[str] = []
        missing: list[str] = []
        for name in shard_names:
            resolved = os.path.abspath(os.path.join(model_dir, name))
            try:
                inside_root = os.path.commonpath((root_abs, resolved)) == root_abs
            except ValueError:
                inside_root = False
            if not inside_root:
                invalid_paths.append(name)
            elif not _nonempty_file(resolved):
                missing.append(name)
        if invalid_paths:
            preview = ", ".join(invalid_paths[:5])
            raise LocalModelLoadError(
                f"local weight index {index_name!r} contains paths outside the "
                f"model directory: {preview}"
            )
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise LocalModelLoadError(
                f"local weight index {index_name!r} references missing/empty shards: "
                f"{preview}{suffix}"
            )
        # Transformers prefers safetensors.  Once its preferred index is valid,
        # an unused secondary serialization must not block an otherwise sound
        # local cache.
        break

    if not found_index:
        try:
            candidates = [
                name
                for name in os.listdir(model_dir)
                if name in ("model.safetensors", "pytorch_model.bin")
                and _nonempty_file(os.path.join(model_dir, name))
            ]
        except OSError as exc:
            raise LocalModelLoadError(f"cannot inspect local model directory: {exc}") from exc
        if not candidates:
            raise LocalModelLoadError(
                f"no non-empty .safetensors/.bin model weights under {model_dir!r}"
            )
    return config


def _nonempty_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _load_tokenizer_native_first(
    loader: Any, model_dir: str, config: dict[str, Any]
) -> Any:
    failures: list[tuple[str, BaseException]] = []
    # Prefer built-in Transformers implementations.  This avoids following a
    # scrubbed config's cross-repository auto_map when the architecture is
    # natively supported.  Custom code remains an offline-only last resort.
    for trust_remote_code in (False, True):
        for use_fast in (True, False):
            label = f"trust_remote_code={trust_remote_code},use_fast={use_fast}"
            try:
                context = (
                    _local_model_import_path(model_dir)
                    if trust_remote_code
                    else nullcontext()
                )
                with context:
                    return loader.from_pretrained(
                        model_dir,
                        trust_remote_code=trust_remote_code,
                        use_fast=use_fast,
                        local_files_only=True,
                    )
            except (OSError, ValueError, ImportError, RuntimeError, TypeError) as exc:
                if "out of memory" in str(exc).lower():
                    raise
                failures.append((label, exc))
    raise _local_load_error("tokenizer", model_dir, config, failures)


def _load_model_native_first(
    loader: Any,
    model_dir: str,
    *,
    config_data: dict[str, Any],
    common: dict[str, Any],
) -> Any:
    failures: list[tuple[str, BaseException]] = []
    for trust_remote_code in (False, True):
        for attention in ("sdpa", "eager"):
            label = f"trust_remote_code={trust_remote_code},attention={attention}"
            try:
                context = (
                    _local_model_import_path(model_dir)
                    if trust_remote_code
                    else nullcontext()
                )
                with context:
                    return loader.from_pretrained(
                        model_dir,
                        trust_remote_code=trust_remote_code,
                        attn_implementation=attention,
                        **common,
                    )
            except (OSError, ValueError, ImportError, RuntimeError, TypeError) as exc:
                if "out of memory" in str(exc).lower():
                    raise
                failures.append((label, exc))
    raise _local_load_error("model", model_dir, config_data, failures)


@contextmanager
def _local_model_import_path(model_dir: str):
    """Temporarily resolve custom model-code sibling imports offline.

    Quasar's staged remote code imports ``configuration_qwen3_5`` by absolute
    module name. Transformers' dynamic loader discovers relative imports, but
    cannot resolve that sibling unless the already-audited local model directory
    is on ``sys.path``. G.O.D applies the same narrow path context. The entry is
    always removed, including when custom-code import raises.
    """
    absolute = os.path.abspath(model_dir)
    sys.path.insert(0, absolute)
    try:
        yield
    finally:
        try:
            sys.path.remove(absolute)
        except ValueError:
            pass


def _local_load_error(
    component: str,
    model_dir: str,
    config: dict[str, Any],
    failures: list[tuple[str, BaseException]],
) -> LocalModelLoadError:
    auto_map = config.get("auto_map")
    auto_map_repr = repr(auto_map)
    if len(auto_map_repr) > 500:
        auto_map_repr = auto_map_repr[:497] + "..."
    auto_map_note = f"; config auto_map={auto_map_repr}" if auto_map else ""
    rendered: list[str] = []
    for label, exc in failures:
        message = " ".join(str(exc).split())
        if len(message) > 320:
            message = message[:317] + "..."
        rendered.append(f"{label}: {type(exc).__name__}: {message}")
    return LocalModelLoadError(
        f"offline {component} load failed for {model_dir!r}{auto_map_note}; attempts: "
        + " | ".join(rendered)
    )


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


def model_param_billions(model: Any) -> float:
    """Trainable-model size in billions of parameters (for strategy + LR sizing)."""
    try:
        return sum(p.numel() for p in model.parameters()) / 1e9
    except Exception:
        return 0.0


def median_weight_rms(model: Any) -> float | None:
    """Median RMS of the 2-D weight matrices — the scale term in the champion
    LR law. Cheap even on GPU (one reduction per matrix); None on failure so
    the LR falls back to its static table."""
    try:
        rms = [
            p.detach().float().pow(2).mean().sqrt().item()
            for p in model.parameters()
            if p.dim() == 2
        ]
        if not rms:
            return None
        rms.sort()
        return rms[len(rms) // 2]
    except Exception:
        return None


def gpu_topology() -> tuple[int, float]:
    """(gpu_count, per_gpu_total_GB). (0, 0.0) on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return n, round(gb, 1)
    except Exception:
        pass
    return 0, 0.0


def decide_full_finetune(
    *, use_kl: bool, params_b: float, n_gpus: int, per_gpu_gb: float
) -> bool:
    """Full fine-tune when it clearly beats LoRA and fits comfortably.

    Full-FT is the empirically-winning choice on the small models that dominate
    the group/knockout rounds (week-1 postmortem: every advancer full-FT'd, no
    LoRA miner advanced). We gate it conservatively for reliability:

    - Never on KL tasks: our KL trainer reads the base logits via the LoRA
      adapter's disable_adapter(); there is no adapter under full-FT. LoRA also
      naturally stays near base, which is what KL rewards.
    - Only single-GPU: multi-GPU full-FT training needs FSDP; our validated
      multi-GPU path is device_map + LoRA. When the validator hands us >1 GPU the
      model is large, so LoRA there is both safer and appropriate.
    - Only when full-FT (bf16 weights+grads + fp32 AdamW states ≈ 12 B/param,
      plus an activation budget) fits inside ~80% of one card's memory.
    """
    # Fail closed until a winner-geometry replica is GPU-certified.  The opt-in
    # is explicit rather than a hardcoded unreachable branch so experiments can
    # exercise the real hardware/fit checks without changing source.
    if os.environ.get("FORGE_ENABLE_EXPERIMENTAL_FULL_FT") != "1":
        return False

    # DISABLED by default after the Jul-20 tournament (week-3 rematch,
    # export-time evals,
    # 34k rows / Qwen2.5-1.5B): full-FT lost at ALL THREE tested LRs — 1e-4
    # diverges at our batch geometry, the champion-law 6.72e-5 NEVER beat the
    # base model (min 1.69 vs base 1.59), 2e-5 best 1.348 — while LoRA+best-
    # checkpoint exports 1.29. Jul-16 forensics: the winners' full-FT works at
    # per-device batch 100 with FA-varlen packing (~200k tok/step) plus
    # checkpoint-soup selection — machinery we don't have yet. Until a replica
    # of that geometry beats our LoRA on the replica evaluator, full-FT is a
    # measured regression, not a strategy.
    if use_kl or n_gpus != 1 or per_gpu_gb <= 0 or params_b <= 0:
        return False
    # 16 B/param = fp32 master weights (4) + fp32 grads (4) + fused-AdamW fp32
    # states (8); +20 GB for activations (the LM-head logits + fp32 CE dominate at
    # seq 4096 / large vocab). Held to 78% of the card so fragmentation + the
    # 20 GB activation reserve leave real headroom. This admits ~<=2.6B on 80 GB —
    # covering the small group/knockout models where full-FT wins — and defers
    # bigger models to LoRA. The OOM-retry is a backstop, not the plan.
    budget_gb = 0.78 * per_gpu_gb
    needed_gb = params_b * 16.0 + 20.0
    return needed_gb <= budget_gb


def prepare_full_finetune(model: Any, *, gradient_checkpointing: bool) -> Any:
    """Ready a raw (non-PEFT) model for full fine-tuning: every parameter trains,
    cache off, and (for gradient checkpointing) inputs require grad.

    Crucially, upcast the trainable weights to fp32 so the optimizer keeps fp32
    *master* weights while bf16 autocast (TrainingArguments bf16=True) does the
    compute. Pure-bf16 full fine-tuning silently stalls — at a 1e-4 LR the
    per-step update is at or below bf16's representable spacing and rounds to
    zero — which would defeat the whole point of full-FT and never show on a CPU
    (fp32) smoke test. This raises memory to ~16 bytes/param; the fit gate budgets
    for it.
    """
    try:
        model = model.float()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad_(True)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def effective_seq_len(model: Any, requested: int) -> int:
    """Clamp to the model's direct positional range (DPO/evaluator semantics)."""
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0:
        return max(1, min(requested, limit))
    return max(1, requested)


def is_quasar_model(model: Any) -> bool:
    """Identify the forced tournament custom architecture through PEFT wrappers."""
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    class_name = type(model).__name__.lower()
    base = getattr(model, "get_base_model", None)
    if callable(base):
        try:
            class_name += " " + type(base()).__name__.lower()
        except Exception:
            pass
    return "quasar" in model_type or "quasar" in class_name


def conservative_quasar_plan(model: Any, plan: Any) -> tuple[Any, bool]:
    """Use microbatch one when Quasar's advertised checkpointing is a no-op."""
    if not is_quasar_model(model) or int(plan.per_device_batch_size) <= 1:
        return plan, False
    effective = int(plan.per_device_batch_size) * int(plan.grad_accum_steps)
    return (
        replace(
            plan,
            per_device_batch_size=1,
            grad_accum_steps=max(1, effective),
            gradient_checkpointing=False,
        ),
        True,
    )


def effective_sft_seq_len(model: Any, requested: int) -> int:
    """Match G.O.D's SFT/Chat evaluation context cap.

    The evaluator starts at 4096 tokens.  When a model advertises fewer than
    8192 positions it uses ``ceil(max_position_embeddings / 2)`` instead, rather
    than the direct positional limit used by DPO.  Apply this *after* any
    baseline-statistics cap so training does not spend time on tokens SFT scoring
    will discard.
    """
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    evaluator_cap = 4096
    if isinstance(limit, int) and 0 < limit < 8192:
        evaluator_cap = (limit + 1) // 2
    return max(1, min(requested, evaluator_cap))


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
    trainable = sum(
        int(parameter.numel())
        for parameter in peft_model.parameters()
        if getattr(parameter, "requires_grad", False)
    )
    if trainable <= 0:
        raise RuntimeError("LoRA attachment produced no trainable parameters")
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    return peft_model
