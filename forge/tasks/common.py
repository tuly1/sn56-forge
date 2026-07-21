"""Shared trainer setup: argument construction, periodic saving, finalisation.

Kill-safety strategy: we disable the HF Trainer's own checkpointing and instead
mirror the current adapter into the mandated output path every few steps. That
keeps exactly one clean adapter at `spec.output_dir` at all times — so if the
wall-clock kill lands, the uploader finds the latest model and no stale
`checkpoint-*` subdirectory can shadow it.
"""

from __future__ import annotations

import errno
import json
import math
import os
import zipfile
from collections.abc import Mapping
from typing import Any

from forge.clock import Deadline
from forge.data.schema import TaskSpec
from forge.tuning.plan import TrainPlan


def workdir(spec: TaskSpec) -> str:
    d = f"/app/checkpoints/{spec.task_id}/_work"
    os.makedirs(d, exist_ok=True)
    return d


def build_training_kwargs(
    spec: TaskSpec, plan: TrainPlan, *, neftune_alpha: float | None = None
) -> dict[str, Any]:
    """Common TrainingArguments fields shared by SFT/DPO/GRPO configs.

    `neftune_alpha` enables NEFTune input-embedding noise (instruction-tuning
    regulariser). Left None on the KL path, where the noise would otherwise
    contaminate the disable_adapter base reference.
    """
    kwargs = dict(
        output_dir=workdir(spec),
        overwrite_output_dir=True,
        num_train_epochs=plan.num_epochs,
        per_device_train_batch_size=plan.per_device_batch_size,
        gradient_accumulation_steps=plan.grad_accum_steps,
        learning_rate=plan.learning_rate,
        lr_scheduler_type=plan.lr_scheduler,
        # Floor the cosine at 25% of peak when using the floored scheduler.
        lr_scheduler_kwargs=(
            {"min_lr_rate": 0.25}
            if plan.lr_scheduler == "cosine_with_min_lr"
            else {}
        ),
        neftune_noise_alpha=neftune_alpha,
        warmup_ratio=plan.warmup_ratio,
        weight_decay=plan.weight_decay,
        optim=plan.optimizer,
        max_grad_norm=1.0,
        bf16=plan.bf16,
        fp16=plan.fp16,
        gradient_checkpointing=plan.gradient_checkpointing,
        # PEFT + gradient checkpointing needs non-reentrant to keep adapter grads.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",  # we mirror to output_dir ourselves (see below)
        report_to=[],  # wandb is offline via env; don't import integrations
        remove_unused_columns=False,
        dataloader_num_workers=2,
        disable_tqdm=True,
        seed=7,
    )
    return kwargs


def compatible_dataclass_kwargs(
    config_class: Any,
    kwargs: dict[str, Any],
    *,
    allow_removed: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Filter only audited cross-version dataclass removals.

    The validator-aligned Transformers-v5/TRL-1.5 runtime removed a small set
    of fields that existed in the former v4/TRL-0.24 stack.  Silently filtering
    every unknown key would hide ordinary spelling/configuration bugs, so this
    helper fails on anything outside the caller's explicit allowlist.
    """
    fields = getattr(config_class, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        raise TypeError(f"{config_class!r} does not expose dataclass fields")
    unknown = set(kwargs) - set(fields)
    unexpected = unknown - set(allow_removed)
    if unexpected:
        rendered = ", ".join(sorted(unexpected))
        raise TypeError(
            f"unsupported {getattr(config_class, '__name__', config_class)!s} "
            f"arguments: {rendered}"
        )
    return {key: value for key, value in kwargs.items() if key in fields}


def time_aware_epochs(
    *,
    trainer_cls: Any,
    model: Any,
    kwargs: dict[str, Any],
    train_ex: list,
    collator: Any,
    deadline: Deadline,
    eff_batch: int,
    strategy: str,
    trainer_extra: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Plan exactly the epochs that fit the wall clock, so the cosine cooldown
    COMPLETES at the deadline instead of being cut mid-anneal.

    Week-2's decisive recipe flaw: schedules were built against a fixed epoch
    count, every run was deadline-cut at ~42% of its schedule, and the LR never
    left ~74% of peak. Here we measure real per-step cost with a ~35-step
    zero-LR probe (no weight movement, steps 10->30 timed to skip CUDA warm-up)
    and size the schedule to what the budget actually holds.

    Returns (epochs, per_step_s); (None, None) means "keep the plan default"
    (probe skipped: no CUDA, tiny dataset, tight budget, or probe failure).
    """
    import time as _time

    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
    except Exception:
        return None, None
    # Tight budget: probing would eat training time we can't spare. Tiny data:
    # the whole schedule finishes early regardless.
    if deadline.remaining() < 900 or len(train_ex) < eff_batch * 60:
        return None, None

    from datasets import Dataset
    from transformers import TrainerCallback, TrainingArguments

    # A slow model must not spend an unbounded fraction of its useful window on
    # a probe.  Thirty optimizer steps used to be unconditional once the probe
    # started, which could consume most of a short task on an unexpectedly slow
    # architecture.  Stop the probe after at most 12% of the current budget
    # (capped at three minutes); a partial probe is deliberately discarded.
    probe_stop_at = _time.monotonic() + min(
        180.0, max(60.0, deadline.remaining() * 0.12)
    )

    class _Timer(TrainerCallback):
        t10: float | None = None
        t30: float | None = None

        def _out_of_probe_time(self) -> bool:
            return _time.monotonic() >= probe_stop_at or deadline.remaining() <= 600.0

        def on_step_end(self, args, state, control, **kw):
            if state.global_step == 10:
                self.t10 = _time.monotonic()
            elif state.global_step >= 30 and self.t30 is None:
                self.t30 = _time.monotonic()
                control.should_training_stop = True
            elif self._out_of_probe_time():
                control.should_training_stop = True
            return control

        def on_substep_end(self, args, state, control, **kw):
            if self._out_of_probe_time():
                control.should_training_stop = True
            return control

    probe_kwargs = dict(kwargs)
    probe_kwargs.update(max_steps=35, learning_rate=0.0, neftune_noise_alpha=None)
    for k in ("eval_strategy", "eval_steps", "per_device_eval_batch_size"):
        probe_kwargs.pop(k, None)
    timer = _Timer()
    try:
        probe = trainer_cls(
            model=model,
            args=TrainingArguments(
                **compatible_dataclass_kwargs(
                    TrainingArguments,
                    probe_kwargs,
                    allow_removed={"overwrite_output_dir"},
                )
            ),
            train_dataset=Dataset.from_list(train_ex[: eff_batch * 40]),
            data_collator=collator,
            callbacks=[timer],
            **(trainer_extra or {}),
        )
        probe.train()
        del probe
        torch.cuda.empty_cache()
    except Exception:
        return None, None
    if timer.t10 is None or timer.t30 is None:
        return None, None

    per_step = (timer.t30 - timer.t10) / 20.0
    if not math.isfinite(per_step) or per_step <= 0:
        return None, None
    # Margin covers what the probe doesn't see: periodic mirror saves (a full
    # fp32 model every 100 steps is far heavier than an adapter every 25) and
    # the eval-logging passes. Erring LOW finishes the anneal early and idles;
    # erring high cuts the cooldown — the asymmetry we're here to remove.
    margin = 0.82 if strategy == "full" else 0.90
    steps_per_epoch = max(1, len(train_ex) // eff_batch)
    window = max(0.0, deadline.remaining() - 60.0)  # trainer re-init slack
    achievable = window / per_step * margin
    # Cap 4.0: small tasks (week-1 wasted 63% of a KL task's hour on a fixed
    # 2-epoch plan) get to fill their budget; the floor only guards division
    # pathology — a small planned-and-completed schedule is fine.
    epochs = max(0.05, min(4.0, achievable / steps_per_epoch))
    return round(epochs, 2), per_step


def safe_train(trainer: Any, *, min_batch: int = 1) -> None:
    """Run ``trainer.train()``, with one conservative zero-progress OOM retry.

    Reusing a Trainer after it has taken an optimizer step silently mixes a
    partially progressed model with a fresh optimizer/schedule.  We therefore
    retry only when ``global_step == 0``, after explicitly clearing gradients,
    optimizer/scheduler state and callback control.  The retry micro-batch must
    also be strictly smaller. Trainers such as GRPO that derive sampler geometry
    from the construction-time batch are never mutated in place; they require a
    future trainer-factory retry. Otherwise the original OOM propagates and the
    already-written floor/best artifact remains available to the CLI fallback.
    """
    try:
        trainer.train()
        return
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        oom_exc = exc
        oom_traceback = exc.__traceback__
    from forge import telemetry

    state = getattr(trainer, "state", None)
    progressed = int(getattr(state, "global_step", 0) or 0)
    args = trainer.args
    cur = max(1, int(getattr(args, "per_device_train_batch_size", 1) or 1))
    floor = max(1, int(min_batch or 1))
    reduced = max(floor, cur // 2)
    derived_batch_geometry = any(
        hasattr(args, name)
        for name in ("generation_batch_size", "steps_per_generation")
    )
    if progressed != 0 or derived_batch_geometry or reduced >= cur:
        if progressed:
            reason = "progressed"
        elif derived_batch_geometry:
            reason = "trainer_requires_rebuild"
        else:
            reason = "batch_cannot_shrink"
        telemetry.event(
            "oom_retry_skipped",
            reason=reason,
            step=progressed,
            batch=cur,
            min_batch=floor,
        )
        raise oom_exc.with_traceback(oom_traceback)

    telemetry.event("oom_retry", from_batch=cur, to_batch=reduced)
    _reset_trainer_after_zero_step_oom(trainer)
    _free_cuda()
    # Do not mutate gradient_accumulation_steps on an existing Trainer: its
    # Accelerator captures that value during construction.  The retry therefore
    # uses a genuinely smaller effective batch rather than claiming to preserve
    # it while the runtime keeps stale accumulation state.
    args.per_device_train_batch_size = reduced
    # Transformers 5 caches ``args.train_batch_size`` on the Trainer at
    # construction time and reads that cache when it builds the dataloader.
    # Updating only TrainingArguments would therefore repeat the OOM with the
    # original batch.  Keep this narrowly scoped to Trainers that expose the
    # cache; older/runtime-specific trainers continue to use their own path.
    if hasattr(trainer, "_train_batch_size"):
        trainer._train_batch_size = int(
            getattr(args, "train_batch_size", reduced) or reduced
        )
    trainer.train()


def _reset_trainer_after_zero_step_oom(trainer: Any) -> None:
    """Clear state that HF Trainer can leave behind after a step-zero OOM.

    This helper intentionally supports simple fakes so its failure paths can be
    unit tested without a GPU.  It is *not* used once any optimizer step has
    completed; rolling model weights back requires reconstructing the trainer
    from a known checkpoint, which callers currently cannot do safely.
    """
    _clear_neftune_hook(trainer)
    model = getattr(trainer, "model", None)
    if model is not None:
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()
        except Exception:
            pass
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
    for name, value in (("optimizer", None), ("lr_scheduler", None)):
        try:
            setattr(trainer, name, value)
        except Exception:
            pass
    if hasattr(trainer, "_created_lr_scheduler"):
        trainer._created_lr_scheduler = False
    for name in ("state", "control"):
        current = getattr(trainer, name, None)
        if current is not None:
            try:
                setattr(trainer, name, type(current)())
            except Exception:
                pass


def _clear_neftune_hook(trainer: Any) -> None:
    """Remove a NEFTune forward hook left attached when a run aborts mid-training
    (HF only detaches it on the normal return path, not on exception), so the
    retry doesn't stack a second noise hook on the embeddings.
    """
    try:
        handle = getattr(trainer, "neftune_hook_handle", None)
        if handle is not None:
            handle.remove()
            trainer.neftune_hook_handle = None
    except Exception:
        pass


def _is_oom(exc: Exception) -> bool:
    import torch

    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class ArtifactValidationError(RuntimeError):
    """A staged trainer artifact is structurally incomplete."""


class _AtomicExchangeUnavailable(OSError):
    """The filesystem/kernel cannot atomically exchange two directory names."""


_READY_MARKER = ".forge_artifact_ready"
_READY_MARKER_TMP = _READY_MARKER + ".tmp"
_READY_MAGIC = b"sn56-forge-artifact-ready-v1\n"


def save_adapter(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Stage, validate, durably flush and promote a complete model artifact.

    Linux uses ``renameat2(RENAME_EXCHANGE)`` when replacing an existing output,
    so the validator-visible path never disappears.  Other platforms use a
    rollback journal (``.old``); :func:`_recover_artifact_dirs` restores it on
    the next call if a process dies between portable renames.
    """
    final = output_dir.rstrip("/")
    tmp = final + ".tmp"
    _recover_artifact_dirs(final)
    _rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    staged_ready = False
    try:
        save_kwargs: dict[str, Any] = {"safe_serialization": True}
        peft_config = getattr(model, "peft_config", None)
        if isinstance(peft_config, Mapping) and "default" in peft_config:
            # TRL 1.5 adds a frozen `ref` adapter to PEFT DPO/GRPO models. PEFT
            # otherwise saves every adapter and G.O.D's uploader prefers the
            # first child directory containing weights, which can upload `/ref`
            # instead of the trained root adapter. Export only the policy.
            save_kwargs["selected_adapters"] = ["default"]
        model.save_pretrained(tmp, **save_kwargs)
        tokenizer.save_pretrained(tmp)

        # Carry the flight recorder into staging before promotion, so weights
        # and diagnostics become visible as one directory-generation.
        from forge import telemetry

        telemetry.write_into(tmp)
        _validate_staged_artifact(tmp)
        _fsync_tree(tmp)
        # The marker is the durable commit record for crash recovery.  It is
        # created only after the complete staged tree has validated and flushed.
        _write_ready_marker(tmp)
        staged_ready = True
        _promote_staged_dir(tmp, final)
    except BaseException:
        # Only a fully validated+flushed generation may participate in startup
        # recovery. A model write followed by tokenizer/telemetry/fsync failure
        # must never be mistaken for a committed candidate merely because its
        # weight filename already exists.
        if os.path.isdir(tmp) and not staged_ready:
            _rmtree(tmp)
        raise


def _validate_staged_artifact(path: str) -> None:
    if not _is_structurally_complete_artifact(path):
        raise ArtifactValidationError(f"staged artifact is incomplete: {path}")


def _is_structurally_complete_artifact(path: str) -> bool:
    """Non-executing pre-promotion validation for model files and indexes.

    ``save_pretrained`` can emit a full model, sharded model, or PEFT adapter, so
    verify the relevant config/index and safetensors header without loading the
    tensor payload into RAM.  A readiness marker is checked separately: prior
    live generations may predate that marker, while recovery-only ``.tmp`` must
    have it.
    """
    if not os.path.isdir(path):
        return False

    adapter = next(
        (
            os.path.join(path, name)
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if _valid_local_weight_file(os.path.join(path, name))
        ),
        None,
    )
    if adapter is not None:
        return _nonempty_local_file(os.path.join(path, "adapter_config.json"))

    for name in ("model.safetensors", "pytorch_model.bin"):
        if _valid_local_weight_file(os.path.join(path, name)):
            return True

    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(path, name)
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            root_abs = os.path.abspath(path)
            for shard in {str(value) for value in weight_map.values()}:
                resolved = os.path.abspath(os.path.join(path, shard))
                if os.path.commonpath((root_abs, resolved)) != root_abs:
                    return False
                if not _valid_local_weight_file(resolved):
                    return False
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    return False


def _nonempty_local_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _valid_local_weight_file(path: str) -> bool:
    if not _nonempty_local_file(path):
        return False
    if path.endswith(".safetensors"):
        return _valid_safetensors_header(path)
    if path.endswith(".bin"):
        return _valid_pytorch_zip(path)
    return False


def _valid_safetensors_header(path: str) -> bool:
    """Validate safetensors bounds/header JSON without reading tensor payloads."""
    try:
        size = os.path.getsize(path)
        if size < 10:
            return False
        with open(path, "rb") as fh:
            header_len = int.from_bytes(fh.read(8), "little", signed=False)
            if header_len <= 0 or header_len > min(100_000_000, size - 8):
                return False
            header = json.loads(fh.read(header_len))
        if not isinstance(header, dict):
            return False
        payload_size = size - 8 - header_len
        tensors = 0
        ranges: list[tuple[int, int]] = []
        dtype_bytes = {
            "BOOL": 1,
            "I8": 1,
            "U8": 1,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
            "F8_E8M0": 1,
            "I16": 2,
            "U16": 2,
            "F16": 2,
            "BF16": 2,
            "I32": 4,
            "U32": 4,
            "F32": 4,
            "I64": 8,
            "U64": 8,
            "F64": 8,
            "C64": 8,
            "C128": 16,
        }
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                return False
            offsets = info.get("data_offsets")
            shape = info.get("shape")
            dtype = info.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
                or not 0 <= offsets[0] <= offsets[1] <= payload_size
                or not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in shape
                )
            ):
                return False
            if dtype in dtype_bytes:
                elements = math.prod(shape)
                if offsets[1] - offsets[0] != elements * dtype_bytes[dtype]:
                    return False
            ranges.append((offsets[0], offsets[1]))
            tensors += 1
        if tensors == 0:
            return False
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                return False
            cursor = end
        return cursor == payload_size
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _valid_pytorch_zip(path: str) -> bool:
    """Validate modern torch.save structure without executing pickle.

    Legacy raw-pickle files fail closed.  A central directory plus PyTorch's
    data.pkl/tensor/version members proves structural completeness without
    deserializing attacker-controlled objects or reading tensor payloads.
    """
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path, "r") as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not infos:
                return False
            data_pickles = [
                info
                for info in infos
                if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")
            ]
            tensor_members = [
                info
                for info in infos
                if (info.filename.startswith("data/") or "/data/" in info.filename)
                and info.file_size > 0
            ]
            versions = [
                info
                for info in infos
                if info.filename == "version" or info.filename.endswith("/version")
            ]
            if len(data_pickles) != 1 or not tensor_members or not versions:
                return False
            with archive.open(data_pickles[0], "r") as fh:
                pickle_prefix = fh.read(2)
            if len(pickle_prefix) != 2 or pickle_prefix[0] != 0x80:
                return False
            archive_size = os.path.getsize(path)
            for info in data_pickles + tensor_members + versions:
                if info.header_offset < 0 or info.compress_size < 0:
                    return False
                minimum_extent = (
                    info.header_offset
                    + 30
                    + len(info.filename.encode())
                    + len(info.extra)
                    + info.compress_size
                )
                if minimum_extent > archive_size:
                    return False
            return True
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, NotImplementedError):
        return False


def _fsync_tree(path: str) -> None:
    """Flush staged regular files and directory entries before name promotion."""
    directories: list[str] = []
    for root, dirs, files in os.walk(path):
        directories.append(root)
        for filename in files:
            candidate = os.path.join(root, filename)
            if not os.path.isfile(candidate) or os.path.islink(candidate):
                continue
            fd = os.open(candidate, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        directories.extend(os.path.join(root, d) for d in dirs)
    for directory in reversed(dict.fromkeys(directories)):
        _fsync_dir(directory)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise


def _write_ready_marker(path: str) -> None:
    """Atomically create and fsync the durable staged-generation commit marker."""
    marker = os.path.join(path, _READY_MARKER)
    marker_tmp = os.path.join(path, _READY_MARKER_TMP)
    try:
        with open(marker_tmp, "wb") as fh:
            fh.write(_READY_MAGIC)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(marker_tmp, marker)
        _fsync_dir(path)
    except BaseException:
        try:
            os.unlink(marker_tmp)
        except OSError:
            pass
        raise


def _has_ready_marker(path: str) -> bool:
    try:
        marker = os.path.join(path, _READY_MARKER)
        if os.path.getsize(marker) != len(_READY_MAGIC):
            return False
        with open(marker, "rb") as fh:
            return fh.read(len(_READY_MAGIC) + 1) == _READY_MAGIC
    except OSError:
        return False


def _recover_artifact_dirs(final: str) -> None:
    """Recover a portable-promotion interruption from ``.old``/``.tmp``."""
    tmp, old = final + ".tmp", final + ".old"
    if _is_structurally_complete_artifact(final):
        _rmtree(tmp)
        _rmtree(old)
        return

    candidate = None
    if _is_structurally_complete_artifact(old):
        candidate = old  # last known-live generation wins over uncommitted tmp
    elif _has_ready_marker(tmp) and _is_structurally_complete_artifact(tmp):
        candidate = tmp
    if candidate is not None:
        _rmtree(final)
        os.replace(candidate, final)
        _fsync_dir(os.path.dirname(final) or ".")
    _rmtree(tmp)
    _rmtree(old)


def _promote_staged_dir(tmp: str, final: str) -> None:
    """Promote a validated staged directory, atomically on Linux when possible."""
    _validate_staged_artifact(tmp)
    if not _has_ready_marker(tmp):
        raise ArtifactValidationError(f"staged artifact is not durably ready: {tmp}")
    parent = os.path.dirname(final) or "."
    old = final + ".old"
    if not os.path.isdir(final):
        os.replace(tmp, final)
        _fsync_dir(parent)
        return

    try:
        _rename_exchange(tmp, final)
        _fsync_dir(parent)
        # After exchange, tmp names the prior complete generation.
        _rmtree(tmp)
        _fsync_dir(parent)
        return
    except _AtomicExchangeUnavailable:
        pass

    # Portable rollback journal.  A crash after the first replace is repaired by
    # _recover_artifact_dirs on the next process/callback invocation.
    _rmtree(old)
    os.replace(final, old)
    _fsync_dir(parent)
    try:
        os.replace(tmp, final)
        _fsync_dir(parent)
    except BaseException:
        if not os.path.exists(final) and os.path.isdir(old):
            os.replace(old, final)
            _fsync_dir(parent)
        raise
    _rmtree(old)
    _fsync_dir(parent)


def _rename_exchange(left: str, right: str) -> None:
    """Atomically exchange two paths with Linux renameat2, or signal fallback."""
    if os.name != "posix" or not hasattr(os, "uname") or os.uname().sysname != "Linux":
        raise _AtomicExchangeUnavailable(errno.ENOTSUP, "rename exchange unavailable")
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise _AtomicExchangeUnavailable(errno.ENOSYS, "renameat2 unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    rc = renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 0x2)
    if rc == 0:
        return
    err = ctypes.get_errno()
    unavailable = {errno.ENOSYS, errno.EINVAL, errno.EXDEV, errno.ENOTSUP}
    if hasattr(errno, "EOPNOTSUPP"):
        unavailable.add(errno.EOPNOTSUPP)
    if err in unavailable:
        raise _AtomicExchangeUnavailable(err, os.strerror(err))
    raise OSError(err, os.strerror(err), left, right)


def _rmtree(path: str) -> None:
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


class BestTracker:
    """Shared state for best-checkpoint selection.

    Written by the eval callback; read by the periodic latest-mirror (which
    stands down once a best exists) and by the final-save decision.
    """

    def __init__(self) -> None:
        self.observed_best: float | None = None
        self.observed_best_step: int | None = None
        self.persisted_best: float | None = None
        self.persisted_best_step: int | None = None
        self.last: float | None = None
        self.last_step: int | None = None

    # Compatibility aliases for telemetry/tests and downstream code.  "best"
    # now means best *durably exported* checkpoint, never merely observed loss.
    @property
    def best(self) -> float | None:
        return self.persisted_best

    @best.setter
    def best(self, value: float | None) -> None:
        self.persisted_best = value

    @property
    def best_step(self) -> int | None:
        return self.persisted_best_step

    @best_step.setter
    def best_step(self, value: int | None) -> None:
        self.persisted_best_step = value


def should_final_save(
    tracker: "BestTracker | None", *, final_step: int | None = None
) -> bool:
    """Whether final weights may replace the persisted measured checkpoint.

    With ``final_step`` (all production callers), replacement is allowed only if
    that exact optimizer step was evaluated and was no worse than the persisted
    best.  Thus later unevaluated training can never overwrite a measured best.
    The no-argument branch retains compatibility only for legacy trackers that
    lack step metadata; callback-populated trackers fail closed.  New code must
    pass the trainer's final global step.
    """
    if tracker is None or tracker.persisted_best is None:
        return True
    if final_step is None:
        # Legacy manually-populated trackers did not record evaluation steps.
        # A real callback-populated tracker always does, so fail closed when a
        # production caller forgets to supply the final optimizer step.
        if tracker.last_step is not None:
            return False
    elif tracker.last_step != int(final_step):
        return False
    return tracker.last is not None and tracker.last <= tracker.persisted_best + 1e-9


def _make_best_checkpoint_callback(spec: TaskSpec, tokenizer: Any, tracker: BestTracker):
    """Export the eval-minimum checkpoint the moment it becomes the minimum.

    On the production Linux filesystem, the atomic directory exchange keeps a
    complete artifact at the output path throughout replacement.  This is what
    the week-1 (reverted) version lacked: it selected only at the end, so a
    deadline cut shipped whatever was newest rather than whatever was best.
    """
    from transformers import TrainerCallback

    class BestCheckpointCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return control
            from forge import telemetry

            try:
                value = float(loss)
            except (TypeError, ValueError):
                telemetry.event(
                    "checkpoint_metric_rejected",
                    step=int(state.global_step),
                    eval_loss=repr(loss),
                )
                return control
            if not math.isfinite(value):
                telemetry.event(
                    "checkpoint_metric_rejected",
                    step=int(state.global_step),
                    eval_loss=repr(value),
                )
                return control
            step = int(state.global_step)
            tracker.last = value
            tracker.last_step = step
            if tracker.observed_best is None or value < tracker.observed_best:
                tracker.observed_best = value
                tracker.observed_best_step = step

            # Compare against what is durably available, not merely what was
            # observed: after a failed export, the next-best evaluation must get
            # another chance to become the persisted recovery checkpoint.
            if tracker.persisted_best is None or value < tracker.persisted_best:
                model = kwargs.get("model")
                if model is None:
                    return control
                try:
                    save_adapter(model, tokenizer, spec.output_dir)
                except Exception as exc:
                    telemetry.event(
                        "best_checkpoint_export_failed",
                        step=step,
                        eval_loss=round(value, 5),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    return control
                tracker.persisted_best = value
                tracker.persisted_best_step = step
                telemetry.event(
                    "best_checkpoint",
                    step=step,
                    eval_loss=round(value, 5),
                )
                telemetry.write_into(spec.output_dir)
            return control

    return BestCheckpointCallback()


def _make_periodic_save_callback(
    spec: TaskSpec, tokenizer: Any, *, every: int = 25, tracker: BestTracker | None = None
):
    """Mirror the adapter into the output path every `every` optimizer steps.

    Built as a TrainerCallback subclass at call time to keep this module usable
    (for arg/save helpers) even where transformers isn't importable. Keeps the
    latest model at the mandated output path so a wall-clock kill always uploads
    the most recent trained adapter. Once a BEST checkpoint exists (tracker),
    this mirror stands down — "latest" must never overwrite "best".
    """
    from transformers import TrainerCallback

    step = max(1, every)

    class PeriodicSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if tracker is not None and tracker.persisted_best is not None:
                return control
            if state.global_step > 0 and state.global_step % step == 0:
                model = kwargs.get("model")
                if model is not None:
                    try:
                        save_adapter(model, tokenizer, spec.output_dir)
                    except Exception as exc:
                        from forge import telemetry

                        telemetry.event(
                            "periodic_checkpoint_export_failed",
                            step=int(state.global_step),
                            error=f"{type(exc).__name__}: {exc}",
                        )
            return control

    return PeriodicSaveCallback()
