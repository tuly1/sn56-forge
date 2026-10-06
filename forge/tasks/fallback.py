"""Crash fallback: a real minimal fine-tune, or an honest failure.

Owner decision, 2026-10-05 (history/owner-directives-20260928/OWNER-QUOTES.md): when a handler fails, or a supported text
task type has no handler, the task gets a genuine minimal fine-tune on its own TRAIN rows. If even that cannot run, the
process exits non-zero with nothing at the output path and takes the missed-task penalty. Nothing untrained is ever
written to the output path, at any point of any run.

The mini fine-tune:
  * loads the task's base from the validator-staged local directory (no network), attaches a small-rank LoRA adapter
    (the adapter is the crash fallback of last resort, owner directive 4);
  * tokenizes a seeded sample of the task's TRAIN rows with the same prompt / chat-template code as the main handlers;
  * takes real AdamW steps, one row per step, at a short sequence cap, bounded by MINI_MAX_STEPS and by the clock (the
    final save aims to keep MINI_SAVE_RESERVE_S before the validator's hard stop: best-effort, see _run_mini_finetune);
  * verifies that a finite loss was computed, that at least one optimizer step was applied, that the trainable weights
    changed and that they are finite;
  * exports through the production save path, labelled PARTIAL_TRAINED_BEST with the number of applied steps.

Any failure (no time, no base, no TRAIN rows, no supervised target, load error, OOM with no shorter row left, no
finite step, save error) raises MiniFinetuneError. The caller then clears the output path and exits non-zero.

Does NOT run (raises) for GrpoTask: its TRAIN file has prompts and reward functions but no target text, so there is no
supervised step to take; and for task types this build has no data schema for (EnvTask, unknown).
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tasks.common import _valid_pytorch_zip, _valid_safetensors_header

MINI_LORA_R = 8
MINI_LORA_ALPHA = 16
MINI_LR = 1e-4                  # linear decay to 0 over the planned steps
MINI_MAX_STEPS = 64             # optimizer steps, one TRAIN row each
MINI_GRAD_CLIP = 1.0
MINI_SEQ_CAPS = (512, 1024, 2048)  # the first cap that keeps at least one row is used; an OOM halves the row length
MINI_SCAN_ROWS = 2048           # TRAIN rows sampled (seed 7) to find MINI_MAX_STEPS usable examples
MINI_MIN_ENTRY_S = 60.0         # below this much wall clock the run is not started
MINI_SAVE_RESERVE_S = 45.0      # aimed at, before the hard stop, for the adapter save (best-effort, not guaranteed)
MINI_SUPPORTED_TASK_TYPES = ("InstructTextTask", "ChatTask", "DpoTask")

_TRUTH_FILE = "forge_artifact_truth.json"


class MiniFinetuneError(RuntimeError):
    """The minimal fine-tune could not train or could not save. The task must fail: no artifact, non-zero exit."""


def output_is_trained(output_dir: str) -> bool:
    """True when the output path holds a structurally complete artifact that is not labelled FLOOR (untrained).

    No code path writes an untrained artifact: every pre-training floor is removed, save_adapter and
    write_artifact_truth refuse the FLOOR label, and each handler saves only after an optimizer step. So a complete
    artifact here was written after training. The structural check is the one save_adapter applies before it promotes
    an artifact, so anything a handler promoted passes it. The FLOOR label is still read, so an untrained artifact from
    any older writer is never kept; other labels (PARTIAL_TRAINED_BEST, COMPLETE_BEST, the Qwen3.5 fixed-average
    midpoint label) and unlabelled trained saves (DPO / GRPO) are trained artifacts."""
    from forge.tasks.common import _is_structurally_complete_artifact, _recover_artifact_dirs

    try:
        _recover_artifact_dirs(output_dir.rstrip("/"))
    except Exception:
        pass
    if not _is_structurally_complete_artifact(output_dir):
        return False
    try:
        with open(os.path.join(output_dir, _TRUTH_FILE), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return True  # no (readable) label: a trained save that does not label, see above
    return not (isinstance(payload, dict) and payload.get("truth") == "FLOOR")


def clear_output(output_dir: str) -> None:
    """Remove the output directory and its staging siblings so a failed task leaves nothing to upload."""
    import shutil

    final = output_dir.rstrip("/")
    for path in (final, final + ".tmp", final + ".old"):
        try:
            if os.path.islink(path) or os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _examples(spec: TaskSpec, tokenizer: Any, model: Any) -> tuple[list[dict[str, list[int]]], int, str]:
    """Tokenized TRAIN examples for the mini fine-tune: (examples, sequence cap, objective)."""
    from forge.data import loader, prompts, tokenize
    from forge.model import effective_sft_seq_len

    if spec.task_type not in MINI_SUPPORTED_TASK_TYPES:
        raise MiniFinetuneError(f"no supervised mini fine-tune for task type {spec.task_type!r}")
    rows = loader.load_rows(spec.cached_dataset_path, dataset_arg=spec.dataset, file_format=spec.file_format)
    if not rows:
        raise MiniFinetuneError("no TRAIN rows")
    if len(rows) > MINI_SCAN_ROWS:
        picked = sorted(random.Random(7).sample(range(len(rows)), MINI_SCAN_ROWS))
        rows = [rows[i] for i in picked]

    if spec.chat is not None:
        objective = "chat_sft"
        items = prompts.build_chat_conversations(rows, spec.chat)
        resolution = tokenize.resolve_chat_template(spec.chat.chat_template, tokenizer)

        def tok(cap: int) -> list:
            return tokenize.tokenize_chat_resolved(items, tokenizer, cap, resolution=resolution)
    elif spec.instruct is not None and spec.instruct.output is None:
        objective = "completion_sft"
        items = prompts.build_completion_documents(rows, spec.instruct)

        def tok(cap: int) -> list:
            return tokenize.tokenize_completion(items[:MINI_MAX_STEPS], tokenizer, cap)
    elif spec.instruct is not None:
        objective = "instruct_sft"
        items = prompts.build_instruct_examples(rows, spec.instruct)

        def tok(cap: int) -> list:
            return tokenize.tokenize_instruct(items, tokenizer, cap)
    elif spec.dpo is not None:
        # supervised proxy for a preference task: the prompt with the CHOSEN response as the target
        objective = "dpo_chosen_sft"
        items = [{"prompt_text": ex["prompt"], "completion_text": ex["chosen"]}
                 for ex in prompts.build_dpo_examples(rows, spec.dpo)]

        def tok(cap: int) -> list:
            return tokenize.tokenize_instruct(items, tokenizer, cap)
    else:
        raise MiniFinetuneError("task has no column schema (spec build failed)")
    if not items:
        raise MiniFinetuneError("no trainable TRAIN rows")

    for requested in MINI_SEQ_CAPS:
        cap = effective_sft_seq_len(model, requested)
        examples = tok(cap)
        if examples:
            return examples[:MINI_MAX_STEPS], cap, objective
        if cap < requested:
            break  # the model's own context limit already binds
    raise MiniFinetuneError("no TRAIN row fits the mini fine-tune sequence caps")


def run_mini_finetune(spec: TaskSpec, deadline: Deadline) -> dict[str, Any]:
    """Train a small LoRA adapter on the task's TRAIN rows and export it. Returns the receipt; raises MiniFinetuneError."""
    from forge import telemetry

    try:
        return _run_mini_finetune(spec, deadline)
    except MiniFinetuneError as exc:
        telemetry.event("mini_finetune_failed", error=str(exc)[:400])
        raise
    except BaseException as exc:  # noqa: BLE001 - OOM, load errors, interrupts: all mean "could not train"
        telemetry.event("mini_finetune_failed", error=f"{type(exc).__name__}: {exc}"[:400])
        raise MiniFinetuneError(f"{type(exc).__name__}: {exc}") from exc


def _run_mini_finetune(spec: TaskSpec, deadline: Deadline) -> dict[str, Any]:
    """The save reserve is BEST-EFFORT. Nothing here is preemptible: the load, the tokenization, one attempt (forward,
    backward, optimizer step, or an OOM / non-finite attempt), the verification and the save each run to the end once
    started. The first attempt is admitted with no timing at all, so it can start just before train_end and run past it.
    Later attempts are admitted against the slowest attempt seen so far (of any kind), which is an estimate. An overrun
    that still ends before the hard stop exports normally. Only a hard-stop kill before the adapter is promoted leaves
    an empty output path (failed task); after promotion the trained adapter stays. The receipt records the clock facts."""
    from forge import telemetry

    if spec.task_type not in MINI_SUPPORTED_TASK_TYPES:
        raise MiniFinetuneError(f"no supervised mini fine-tune for task type {spec.task_type!r}")
    entry_remaining = deadline.remaining_hard()
    if entry_remaining < MINI_MIN_ENTRY_S:
        raise MiniFinetuneError(f"only {entry_remaining:.0f}s of wall clock left (< {MINI_MIN_ENTRY_S:.0f}s)")
    # steps stop at the soft stop when it is still ahead (the export reserve stays whole); a handler that failed inside
    # the export reserve leaves only the reserve itself, of which the adapter save aims to keep MINI_SAVE_RESERVE_S
    train_end = deadline.hard_stop - MINI_SAVE_RESERVE_S
    if deadline.remaining() > 0.0:
        train_end = min(train_end, deadline.soft_stop)
    telemetry.event("mini_finetune_start", remaining_hard_s=round(entry_remaining, 1),
                    train_window_s=round(train_end - time.monotonic(), 1))

    import torch

    from forge.data import tokenize
    from forge.model import attach_lora, load_base
    from forge.tasks.common import ARTIFACT_PARTIAL_TRAINED_BEST, _free_cuda, _is_oom, save_adapter

    _free_cuda()
    t_load = time.monotonic()
    loaded = load_base(spec.cached_model_dir, for_generation=False)
    tokenizer = loaded.tokenizer
    examples, seq_cap, objective = _examples(spec, tokenizer, loaded.model)
    model, lora_targets = _attach_mini_lora(spec, loaded, attach_lora, load_base)  # all-linear, or a safe list (Mamba)
    loaded.model = None
    load_s = time.monotonic() - t_load
    if time.monotonic() >= train_end:
        raise MiniFinetuneError(f"no time left to train after a {load_s:.0f}s load")

    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=MINI_LR, weight_decay=0.0)
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = params[0].device
    model.train()

    planned = len(examples)
    steps = 0
    skipped = 0
    losses: list[float] = []
    slowest_s = 0.0    # the slowest attempt so far of ANY kind: applied step, non-finite skip, or OOM and its cleanup
    first_attempt_s = None
    t_step = None
    index = 0
    while index < len(examples) and steps < MINI_MAX_STEPS:
        now = time.monotonic()
        if t_step is not None:
            slowest_s = max(slowest_s, now - t_step)
            if first_attempt_s is None:
                first_attempt_s = now - t_step
        # with nothing trained yet the save reserve protects nothing, so an attempt needs only now < train_end; once a
        # step is applied, another attempt starts only if twice the slowest attempt so far still fits before train_end
        if now >= train_end or (steps >= 1 and now + 2.0 * slowest_s >= train_end):
            break
        t_step = now
        example = examples[index]
        index += 1
        try:
            batch = {k: v.to(device) for k, v in collator([example]).items()}
            loss = model(**batch).loss
            if loss is None or not bool(torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, MINI_GRAD_CLIP)
            if not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            for group in optimizer.param_groups:
                group["lr"] = MINI_LR * max(0.0, 1.0 - steps / float(planned))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            # continue on the remaining rows that are at most half as long; with none left the attempt ends
            optimizer.zero_grad(set_to_none=True)
            batch = loss = None
            _free_cuda()
            seq_cap = len(example["input_ids"]) // 2
            examples = [ex for ex in examples[index:] if len(ex["input_ids"]) <= seq_cap]
            index = 0
            telemetry.event("mini_finetune_oom", new_seq_cap=seq_cap, steps=steps, rows_left=len(examples))
            if not examples:
                if steps >= 1:
                    break  # keep the steps already applied
                raise MiniFinetuneError("out of memory and no shorter TRAIN row left") from exc
            continue
        steps += 1
        losses.append(float(loss.detach()))

    t_train_done = time.monotonic()
    if t_step is not None:
        slowest_s = max(slowest_s, t_train_done - t_step)
        if first_attempt_s is None:
            first_attempt_s = t_train_done - t_step
    train_end_margin_s = train_end - t_train_done    # negative: the last attempt ran into the save reserve
    if steps < 1 or not losses:
        raise MiniFinetuneError(f"no optimizer step was applied ({skipped} non-finite, {index} rows tried)")
    changed = any(bool((p.detach() != b).any()) for p, b in zip(params, before))
    finite = all(bool(torch.isfinite(p.detach()).all()) for p in params)
    del before
    if not changed:
        raise MiniFinetuneError("the trainable weights did not change")
    if not finite:
        raise MiniFinetuneError("non-finite trainable weights after training")

    model.eval()
    t_save = time.monotonic()
    save_adapter(model, tokenizer, spec.output_dir, artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST,
                 optimizer_step=steps, truth_reason="mini_finetune")
    if not output_is_trained(spec.output_dir):
        raise MiniFinetuneError("the saved adapter failed validation")
    receipt = dict(steps=steps, skipped_nonfinite=skipped, first_loss=round(losses[0], 5),
                   last_loss=round(losses[-1], 5), mean_loss=round(sum(losses) / len(losses), 5),
                   seq_cap=seq_cap, objective=objective, lora_r=MINI_LORA_R, lr=MINI_LR, lora_targets=lora_targets,
                   load_s=round(load_s, 1), first_attempt_s=round(first_attempt_s or 0.0, 3),
                   slowest_attempt_s=round(slowest_s, 3), train_end_margin_s=round(train_end_margin_s, 1),
                   save_s=round(time.monotonic() - t_save, 2), save_reserve_s=MINI_SAVE_RESERVE_S,
                   remaining_hard_s=round(deadline.remaining_hard(), 1))
    telemetry.event("mini_finetune_done", **receipt)
    telemetry.write_into(spec.output_dir)
    return receipt


_MAMBA_MODEL_TYPES = ("falcon_h1", "mamba", "mamba2", "falcon_mamba")  # PEFT 0.19.1 _check_lora_target_modules_mamba
_MAMBA_LORA_INCOMPATIBLE = ("out_proj", "conv1d")
_LORA_HEADS = ("lm_head", "score")


def _linear_leaf_names(model: Any) -> list[str]:
    import torch

    return sorted({name.split(".")[-1] for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.split(".")[-1] not in _LORA_HEADS})


def _lora_incompatible_names(model: Any, names: list[str]) -> list[str]:
    """Linear leaf names PEFT refuses as LoRA targets on this architecture (Mamba hybrids: out_proj, conv1d).

    PEFT's own check is asked first; the pinned table above covers a PEFT build without that helper."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    banned = {n for n in names if model_type in _MAMBA_MODEL_TYPES and n in _MAMBA_LORA_INCOMPATIBLE}
    try:
        from peft import LoraConfig
        from peft.tuners.tuners_utils import _check_lora_target_modules_mamba as check

        probe = LoraConfig(r=MINI_LORA_R, lora_alpha=MINI_LORA_ALPHA, task_type="CAUSAL_LM", target_modules=["probe"])
        for n in names:
            try:
                check(probe, model, n)
            except ValueError:
                banned.add(n)
    except Exception:  # noqa: BLE001 - helper absent or changed: the pinned table stands
        pass
    return sorted(banned)


def _names_in_error(exc: BaseException) -> set[str]:
    """Module names a PEFT 'incompatible' error mentions (quoted identifiers), for an architecture the table misses."""
    import re

    text = str(exc)
    return set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", text)) if "ncompatible" in text else set()


def _attach_listed_lora(model: Any, names: list[str]) -> Any:
    """LoRA on exactly these linear leaf names (same rank / alpha / no dropout / no bias as forge.model.attach_lora)."""
    from peft import LoraConfig, get_peft_model

    if not names:
        raise MiniFinetuneError("no LoRA-compatible linear layer on this architecture")
    peft_model = get_peft_model(model, LoraConfig(r=MINI_LORA_R, lora_alpha=MINI_LORA_ALPHA, lora_dropout=0.0, bias="none",
                                                  task_type="CAUSAL_LM", target_modules=list(names)))
    if sum(int(p.numel()) for p in peft_model.parameters() if getattr(p, "requires_grad", False)) <= 0:
        raise MiniFinetuneError("LoRA attachment produced no trainable parameters")
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    return peft_model


def _attach_mini_lora(spec: TaskSpec, loaded: Any, attach_lora: Any, load_base: Any) -> tuple[Any, str]:
    """Attach the mini fine-tune's adapter. Returns (peft model, how the targets were chosen).

    * No linear layer PEFT refuses (every non-Mamba family): forge.model.attach_lora, exactly as before -> "all-linear".
    * PEFT refuses some linear leaf names on this architecture (Falcon-H1 and other Mamba hybrids: out_proj, conv1d): the
      adapter targets every other linear layer (attention and MLP projections, the Mamba in_proj), chosen BEFORE the first
      attach, because a refused attach leaves half-injected LoRA layers in the model -> "safe-list:-out_proj".
    * An 'incompatible' refusal the table did not predict: the base is loaded again (clean) and the adapter targets the
      linear layers the error does not name -> "safe-list-reloaded:-<names>". Any other error propagates (the task fails).
    The steps, the verification (weights changed, finite) and the export are the same in all three cases."""
    from forge import telemetry

    names = _linear_leaf_names(loaded.model)
    banned = _lora_incompatible_names(loaded.model, names)
    if banned:
        how = "safe-list:-" + ",-".join(banned)
        model = _attach_listed_lora(loaded.model, [n for n in names if n not in banned])
    else:
        try:
            return attach_lora(loaded.model, r=MINI_LORA_R, alpha=MINI_LORA_ALPHA, dropout=0.0), "all-linear"
        except ValueError as exc:
            refused = sorted(_names_in_error(exc) & set(names))
            if not refused:
                raise
            from forge.tasks.common import _free_cuda

            loaded.model = None
            _free_cuda()
            how = "safe-list-reloaded:-" + ",-".join(refused)
            model = _attach_listed_lora(load_base(spec.cached_model_dir, for_generation=False).model,
                                        [n for n in names if n not in refused])
    telemetry.event("mini_finetune_lora_targets", how=how, linear_leaf_names=names)
    return model, how


def _has_weights(path: str) -> bool:
    """Cheap structural validity gate for an uploadable model/adapter.

    This checks configs, sharded indexes and tensor-container headers without
    loading multi-gigabyte tensors into RAM.  Full Transformers loadability is a
    later GPU certification gate, but truncated/empty/misindexed files can no
    longer suppress a valid fallback.
    """
    if not os.path.isdir(path):
        return False

    adapter_cfg = os.path.join(path, "adapter_config.json")
    adapter_files = [
        os.path.join(path, name)
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if os.path.isfile(os.path.join(path, name))
    ]
    if os.path.isfile(adapter_cfg) or adapter_files:
        if not _valid_json_object(adapter_cfg) or not adapter_files:
            return False
        # Transformers prefers safetensors when both formats exist; validate the
        # preferred container rather than letting an unused stale .bin suppress
        # a sound adapter.
        return _valid_weight_container(adapter_files[0])

    if not _valid_json_object(os.path.join(path, "config.json")):
        return False

    indexes = [
        os.path.join(path, name)
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if os.path.isfile(os.path.join(path, name))
    ]
    if indexes:
        return _valid_weight_index(path, indexes[0])

    candidates = [
        os.path.join(path, name)
        for name in ("model.safetensors", "pytorch_model.bin")
        if os.path.isfile(os.path.join(path, name))
    ]
    return bool(candidates) and _valid_weight_container(candidates[0])


def _valid_json_object(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return isinstance(json.load(fh), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _valid_weight_index(root: str, index_path: str) -> bool:
    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        # Lexical containment permits normal HF snapshot symlinks into its
        # sibling blob store while still rejecting ``../`` in an index.
        root_abs = os.path.abspath(root)
        shards = {str(value) for value in weight_map.values()}
        for shard in shards:
            resolved = os.path.abspath(os.path.join(root, shard))
            if os.path.commonpath((root_abs, resolved)) != root_abs:
                return False
            if not _valid_weight_container(resolved):
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _valid_weight_container(path: str) -> bool:
    if path.endswith(".safetensors"):
        return _valid_safetensors_header(path)
    if path.endswith(".bin"):
        return _valid_pytorch_zip(path)
    return False
