"""Make a full-weight export pass the validator's ``model_is_a_finetune`` check (release defect D1, 2026-09-29).

The validator (G.O.D a2b14db ``validator/evaluation/evaluators/instruct_text.py:446-455``) loads a non-LoRA submission
with ``AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=False)`` and compares the LOADED model's in-memory
config against ``AutoConfig.from_pretrained(original_repo)`` (``validator/evaluation/model_checks.py``):
``architectures`` (list equality) and every attribute of ``VALIDATOR_ATTRS`` the base config defines must be equal, and
``max_position_embeddings`` must not shrink. Any mismatch -> ``is_finetune=False`` -> SCORE_PENALTY.

Composite (vision-language) bases break this for every full-weight submission, not only ours. Qwen3.5 ships
``model_type qwen3_5`` / ``Qwen3_5ForConditionalGeneration`` with the LM in ``text_config``. ``AutoModelForCausalLM``
maps ``qwen3_5`` to ``Qwen3_5ForCausalLM`` and swaps in the text sub-config (transformers 5.12.1
``models/auto/auto_factory.py:396-400``), so the loaded model reports ``model_type qwen3_5_text`` and
``architectures None`` even for the untouched base (CPU-reproduced, DEFECT-FIXES.md D1). Restoring ``architectures``
alone (the pre-D1 ``_restore_base_architectures``) still fails on ``model_type``.

Fix, computed from the base config on disk (never from a family list):
* flat configs: copy each compared field whose value differs from the base onto the saved config (and raise a smaller
  ``max_position_embeddings`` to the base's);
* composite bases whose text sub-config the validator's loader would swap in: write the BASE composite config back,
  with the trained text config as ``text_config`` carrying the base's ``architectures`` and ``model_type``. A text
  config whose ``model_type`` is the composite type disables transformers' ``language_model`` prefix conversion, so the
  weight names are rewritten to the loaded module names (``model.language_model.X`` -> ``model.X``) in the safetensors
  headers (in place, header only; tensor bytes untouched) and the shard index.
Every rewrite is verified before it is kept: the emulated validator check must pass and (when names or the model type
change) the checkpoint's tensor names must be exactly the loaded model's parameter names (a tied ``lm_head`` may be
absent). A rewrite that fails verification is reverted and reported (``export_config_align_failed``); this module
never raises into a finished run.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

# G.O.D a2b14db validator/evaluation/model_checks.py attrs_to_compare (verbatim order)
VALIDATOR_ATTRS: tuple[str, ...] = (
    "architectures",
    "hidden_size",
    "n_layer",
    "intermediate_size",
    "head_dim",
    "hidden_act",
    "model_type",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
)
CONTEXT_ATTR = "max_position_embeddings"
_COMPOSITE_PREFIX = re.compile(r"^model\.language_model\.")
_DROP_FROM_TEXT = ("architectures", "transformers_version", "_name_or_path", "auto_map")


def _auto_config(path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(path, trust_remote_code=False)


def loaded_view_config(path: str) -> Any:
    """The config object ``AutoModelForCausalLM.from_pretrained(path)`` would attach to the loaded model, without
    loading weights: the same class lookup and text-sub-config swap as transformers' auto factory."""
    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

    cfg = _auto_config(path)
    try:
        model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(cfg)]
    except KeyError:
        return cfg
    if isinstance(model_class, (list, tuple)):
        model_class = model_class[0]
    sub = getattr(cfg, "sub_configs", None) or {}
    if getattr(model_class, "config_class", None) is not None and model_class.config_class == sub.get("text_config"):
        return cfg.get_text_config()
    return cfg


def validator_check(original_config: Any, finetuned_config: Any) -> tuple[bool, str | None]:
    """The comparison part of G.O.D ``model_is_a_finetune`` (a2b14db), returning the first mismatch."""
    try:
        if finetuned_config.architectures != original_config.architectures:
            return False, f"architectures base={original_config.architectures} submitted={finetuned_config.architectures}"
    except Exception as exc:  # the validator treats this as a mismatch too
        return False, f"architectures unreadable: {exc}"
    for attr in VALIDATOR_ATTRS:
        if getattr(original_config, attr, None) is not None:
            if not hasattr(finetuned_config, attr):
                return False, f"{attr} missing"
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                return False, f"{attr} base={getattr(original_config, attr)} submitted={getattr(finetuned_config, attr)}"
    om = getattr(original_config, CONTEXT_ATTR, None)
    fm = getattr(finetuned_config, CONTEXT_ATTR, None)
    if isinstance(om, int) and isinstance(fm, int) and fm < om:
        return False, f"{CONTEXT_ATTR} shrunk base={om} submitted={fm}"
    return True, None


def emulated_is_finetune(base_dir: str, export_dir: str) -> tuple[bool, str | None]:
    return validator_check(_auto_config(base_dir), loaded_view_config(export_dir))


# ---- safetensors header rewrite ----------------------------------------------------------------------------------


def _read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little", signed=False)
        return n, json.loads(fh.read(n))


def _rename_in_place(path: str, rename: dict[str, str]) -> None:
    """Rename tensors in a safetensors file by rewriting only its JSON header. Offsets are relative to the data section,
    so the payload is untouched; the (shorter) new header is space-padded to the old length, as the format allows."""
    n, header = _read_header(path)
    new = {rename.get(k, k): v for k, v in header.items()}
    raw = json.dumps(new, separators=(",", ":")).encode("utf-8")
    if len(raw) > n:
        raise ValueError(f"renamed header grew ({len(raw)} > {n} bytes): {path}")
    raw = raw + b" " * (n - len(raw))
    with open(path, "r+b") as fh:
        fh.seek(8)
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())


def _weight_files(export_dir: str) -> tuple[list[str], str | None]:
    index = os.path.join(export_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        return sorted({os.path.join(export_dir, v) for v in weight_map.values()}), index
    single = os.path.join(export_dir, "model.safetensors")
    return ([single] if os.path.isfile(single) else []), None


def checkpoint_keys(export_dir: str) -> set[str]:
    files, _ = _weight_files(export_dir)
    keys: set[str] = set()
    for f in files:
        keys |= {k for k in _read_header(f)[1] if k != "__metadata__"}
    return keys


def expected_keys(export_dir: str) -> tuple[set[str], bool]:
    """Parameter/buffer names of the model the validator's loader would build (meta device), and whether the output
    head is tied to the input embeddings."""
    import torch
    from transformers import AutoModelForCausalLM

    view = loaded_view_config(export_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(_auto_config(export_dir), trust_remote_code=False)
    tied = bool(getattr(view, "tie_word_embeddings", False) or getattr(model.config, "tie_word_embeddings", False))
    return set(model.state_dict().keys()), tied


def keys_match(export_dir: str) -> tuple[bool, str | None]:
    have = checkpoint_keys(export_dir)
    want, tied = expected_keys(export_dir)
    missing = want - have
    if tied:
        missing -= {"lm_head.weight"}
    extra = have - want
    if missing or extra:
        return False, f"missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}"
    return True, None


class ExportAlignmentError(RuntimeError):
    """A failed alignment could not be rolled back to verified original bytes. Raised (never swallowed) so the staging
    in save_adapter fails, the previously promoted artifact stays intact and the persistence fallback acts."""


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".align.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _Journal:
    """Original bytes of everything an alignment may mutate, captured BEFORE the first mutation: config.json, the shard
    index and, per safetensors file, its 8-byte length prefix + JSON header. The header rewrite keeps the header length
    (space padding) and never touches the payload, so writing the saved prefix back restores each file exactly, even
    after a partial write. ``restore`` then re-reads every file and compares bytes; any difference raises
    ExportAlignmentError."""

    def __init__(self, export_dir: str) -> None:
        self.export_dir = export_dir
        self.config_path = os.path.join(export_dir, "config.json")
        files, index = _weight_files(export_dir)
        self.index_path = index
        with open(self.config_path, "rb") as fh:
            self.config = fh.read()
        self.index = None
        if index:
            with open(index, "rb") as fh:
                self.index = fh.read()
        self.headers: dict[str, bytes] = {}
        for f in files:
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little", signed=False)
                fh.seek(0)
                self.headers[f] = fh.read(8 + n)

    def _current(self, path: str, length: int | None = None) -> bytes:
        with open(path, "rb") as fh:
            return fh.read() if length is None else fh.read(length)

    def verify(self) -> bool:
        try:
            if self._current(self.config_path) != self.config:
                return False
            if self.index_path and self._current(self.index_path) != self.index:
                return False
            return all(self._current(f, len(h)) == h for f, h in self.headers.items())
        except Exception:
            return False

    def restore(self) -> None:
        errors: list[str] = []
        for f, h in self.headers.items():
            try:
                with open(f, "r+b") as fh:
                    fh.seek(0)
                    fh.write(h)
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception as exc:
                errors.append(f"{os.path.basename(f)}: {type(exc).__name__}: {exc}")
        for path, data in ((self.index_path, self.index), (self.config_path, self.config)):
            if path and data is not None:
                try:
                    _atomic_write_bytes(path, data)
                except Exception as exc:
                    errors.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
        for name in os.listdir(self.export_dir):
            if name.endswith((".align.tmp", ".json.tmp")):
                try:
                    os.remove(os.path.join(self.export_dir, name))
                except Exception:
                    pass
        if not self.verify():
            raise ExportAlignmentError(f"alignment rollback not verified: {errors[:3]}")


def _apply_rename(export_dir: str, fn) -> dict[str, str]:
    """Rename tensors ``k -> fn(k)`` in every weight file and the shard index; returns the applied mapping. Not atomic
    on its own: callers hold a _Journal taken before the call and restore it on any exception."""
    files, index = _weight_files(export_dir)
    applied: dict[str, str] = {}
    for f in files:
        _, header = _read_header(f)
        rename = {k: fn(k) for k in header if k != "__metadata__" and fn(k) != k}
        if rename:
            _rename_in_place(f, rename)
            applied.update(rename)
    if index and applied:
        with open(index, encoding="utf-8") as fh:
            data = json.load(fh)
        data["weight_map"] = {applied.get(k, k): v for k, v in data["weight_map"].items()}
        _atomic_write_bytes(index, json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))
    return applied


def _index_consistent(export_dir: str) -> bool:
    files, index = _weight_files(export_dir)
    if not index:
        return True
    with open(index, encoding="utf-8") as fh:
        wm = json.load(fh)["weight_map"]
    for f in files:
        names = {k for k in _read_header(f)[1] if k != "__metadata__"}
        if names != {k for k, v in wm.items() if os.path.join(export_dir, v) == f}:
            return False
    return True


# ---- alignment ---------------------------------------------------------------------------------------------------


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _is_composite_swap(base_dir: str) -> bool:
    """True when the validator's causal-LM loader swaps the base's text sub-config in (Qwen3.5-style bases)."""
    return type(loaded_view_config(base_dir)) is not type(_auto_config(base_dir))


def _flat_patch(saved: dict, base_raw: dict) -> dict:
    out = dict(saved)
    for attr in VALIDATOR_ATTRS:
        if attr in base_raw and base_raw[attr] is not None and out.get(attr) != base_raw[attr]:
            out[attr] = base_raw[attr]
    bm, sm = base_raw.get(CONTEXT_ATTR), out.get(CONTEXT_ATTR)
    if isinstance(bm, int) and isinstance(sm, int) and sm < bm:
        out[CONTEXT_ATTR] = bm
    return out


def _composite_patch(saved: dict, base_raw: dict) -> dict:
    text = {k: v for k, v in saved.items() if k not in _DROP_FROM_TEXT}
    if "text_config" in saved and isinstance(saved["text_config"], dict):  # already composite: realign its text part
        text = {k: v for k, v in saved["text_config"].items() if k not in _DROP_FROM_TEXT}
    text["architectures"] = base_raw.get("architectures")
    text["model_type"] = base_raw.get("model_type")
    base_text = base_raw.get("text_config") or {}
    bm, sm = base_text.get(CONTEXT_ATTR), text.get(CONTEXT_ATTR)
    if isinstance(bm, int) and (not isinstance(sm, int) or sm < bm):
        text[CONTEXT_ATTR] = bm
    out = copy.deepcopy(base_raw)
    out["text_config"] = text
    for k in ("dtype", "torch_dtype", "transformers_version"):
        if k in saved:
            out[k] = saved[k]
    return out


def align_export_with_base(export_dir: str, base_dir: str) -> dict[str, Any]:
    """Align ``export_dir/config.json`` (and, for composite bases, the tensor names) with the base so the validator's
    check passes. Returns a report dict. On any failure every file is rolled back to its journaled original bytes and the
    rollback is verified byte for byte; only an unverifiable rollback raises (ExportAlignmentError)."""
    report: dict[str, Any] = {"changed": False}
    cfg_path = os.path.join(export_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return {**report, "skipped": "no_config_json"}
    try:
        ok, why = emulated_is_finetune(base_dir, export_dir)
    except Exception as exc:
        return {**report, "skipped": f"check_unavailable: {type(exc).__name__}: {exc}"[:300]}
    report["before"] = why
    if ok:
        return {**report, "ok": True}
    with open(cfg_path, encoding="utf-8") as fh:
        original_text = fh.read()
    saved = json.loads(original_text)
    with open(os.path.join(base_dir, "config.json"), encoding="utf-8") as fh:
        base_raw = json.load(fh)
    try:
        journal = _Journal(export_dir)  # before ANY mutation
    except Exception as exc:
        return {**report, "ok": False, "skipped": f"journal_unavailable: {type(exc).__name__}: {exc}"[:300]}
    try:
        composite = _is_composite_swap(base_dir) and "text_config" in base_raw
        report["mode"] = "composite" if composite else "flat"
        new = _composite_patch(saved, base_raw) if composite else _flat_patch(saved, base_raw)
        _write_json(cfg_path, new)
        report["changed"] = True
        type_changed = loaded_view_config(export_dir).model_type != json.loads(original_text).get("model_type")
        if composite or type_changed:
            match, detail = keys_match(export_dir)
            if not match:
                report["renamed_tensors"] = len(_apply_rename(export_dir, lambda k: _COMPOSITE_PREFIX.sub("model.", k)))
                match, detail = keys_match(export_dir)
            if not match:
                raise ValueError(f"tensor names do not match the loaded model: {detail}")
            if not _index_consistent(export_dir):
                raise ValueError("shard index does not match the shard headers")
        ok, why = emulated_is_finetune(base_dir, export_dir)
        if not ok:
            raise ValueError(f"still fails the validator check: {why}")
        return {**report, "ok": True}
    except Exception as exc:
        # roll every file back to the journaled bytes; an unverifiable rollback raises ExportAlignmentError so the
        # staged artifact is never promoted
        journal.restore()
        return {**report, "ok": False, "changed": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
