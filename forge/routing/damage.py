"""Is the delivered base a validator-damaged copy of a public checkpoint? (2026-09-22)

The validator augments about half of the text-task bases (pinned public source: weight scaling 0.40, magnitude
pruning 0.25, gaussian noise 0.20, layer re-init 0.15; norms and embeddings are never touched). On damaged bases
adapter entries trailed the winner by a median 21% against 5.7% for full-weight entries, and 20 of 21 winners shipped
full weights (research/20260922-damaged-base-audit/FINDINGS.md).

``assess`` combines named detectors into one verdict and never raises:
  "clean"    the identity check found a byte-identical public checkpoint (overrides every statistic)
  "damaged"  otherwise, at least one enabled detector says damaged
  "unknown"  anything else (no table, oversized, error, a different checkpoint of the same shape, no detector)
Callers treat only "damaged" as a reason to change routes.

Detectors (``DETECTORS``; ``register`` adds more, e.g. the weight-statistic prototype in
research/20260922-damage-detector/):
  identity       the base_identity report matched a shipped table's config, every tensor is present, some tensors
                 differ, and every norm/embedding tensor is byte-identical. A different public fine-tune of the same
                 shape changes its norms too, so it reads "unknown", not "damaged". Needs a shipped table.
  zero_fraction  reference-free magnitude-pruning signature: sampled 2-D non-norm/non-embedding matrices carry a large
                 fraction of exact zeros (validator pruning zeroes 25-50%; trained bf16 weights carry almost none).
                 Opt-in: it detects pruning only.
  fingerprint    (default ON since 2026-09-22) the research/20260922-damage-detector rule against the shipped offline
                 table data/base_fingerprints.json (73 pool models): source by norm hashes (+ model argument), first
                 16 KiB of every tensor compared; local pruning / bf16-scaling fallback when the source is not tabled.
                 Reads ~2-5 MB locally, < 0.1 s. See the section below and research/20260922-prod-patch/FINGERPRINT-TABLE.md.
"""
from __future__ import annotations

import bisect
import glob
import gzip
import hashlib
import json
import os
import re
import struct
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from forge.routing.base_identity import ALTERED, IDENTICAL, is_protected

DAMAGED, CLEAN, UNKNOWN = "damaged", "clean", "unknown"

def identity_signature(model_dir: str, identity: dict[str, Any] | None) -> dict[str, Any]:
    """Damage verdict from a forge.routing.base_identity.identify report (no extra I/O)."""
    if not identity:
        return {"verdict": UNKNOWN, "reason": "no_identity_report"}
    status = identity.get("status")
    if status == IDENTICAL:
        return {"verdict": CLEAN, "reason": "identical"}
    if status != ALTERED:
        return {"verdict": UNKNOWN, "reason": f"identity_{status}"}
    if int(identity.get("missing") or 0) or int(identity.get("extra") or 0):
        return {"verdict": UNKNOWN, "reason": "tensor_set_differs"}
    if int(identity.get("altered") or 0) <= 0:
        return {"verdict": UNKNOWN, "reason": "nothing_altered"}
    if identity.get("protected") is None or int(identity.get("protected") or 0) <= 0:
        return {"verdict": UNKNOWN, "reason": "no_protected_tensors"}
    if int(identity.get("protected_altered") or 0) > 0:
        return {"verdict": UNKNOWN, "reason": "norm_or_embedding_altered"}  # another checkpoint, not augmentation
    return {"verdict": DAMAGED, "reason": "weights_altered_norms_embeddings_intact"}


ZERO_FRACTION_MIN = 0.20      # validator pruning: 25-50% of entries per matrix
ZERO_FRACTION_SAMPLE = 32     # 2-D matrices read (<= 512 KiB each), spread over the file order
ZERO_FRACTION_MIN_HITS = 3    # sampled matrices that must clear ZERO_FRACTION_MIN (multi-layer pruning spares layers)
_ZERO_DTYPES = {"BF16": ("<u2", 0x7FFF), "F16": ("<u2", 0x7FFF), "F32": ("<u4", 0x7FFFFFFF)}  # +0.0 and -0.0


def _zero_fraction(fh: Any, base: int, meta: dict[str, Any], max_bytes: int) -> float | None:
    import numpy as np

    spec = _ZERO_DTYPES.get(meta.get("dtype"))
    if spec is None:
        return None
    width = int(spec[0][-1])
    s, e = meta["data_offsets"]
    n = min(e - s, max_bytes) // width * width
    if n <= 0:
        return None
    fh.seek(base + s)
    arr = np.frombuffer(fh.read(n), dtype=spec[0])
    return float(np.count_nonzero((arr & spec[1]) == 0)) / max(1, arr.size)


def zero_fraction(model_dir: str, identity: dict[str, Any] | None = None, *, sample: int = ZERO_FRACTION_SAMPLE,
                  max_bytes_per_tensor: int = 1 << 19) -> dict[str, Any]:
    """Pruning signature from exact-zero fractions of sampled 2-D weight matrices (reads <= sample / 2 MiB)."""
    cands: list[tuple[str, str, dict[str, Any], int]] = []
    for f in sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__" or is_protected(name) or len(meta.get("shape") or []) != 2:
                continue
            if "lm_head" in name or min(meta["shape"]) < 16:
                continue
            cands.append((f, name, meta, 8 + n))
    if len(cands) < 3:
        return {"verdict": UNKNOWN, "reason": "too_few_matrices", "n": len(cands)}
    step = max(1, len(cands) // sample)
    picked = cands[::step][:sample]
    fracs: list[float] = []
    for f, name, meta, base in picked:
        with open(f, "rb") as fh:
            z = _zero_fraction(fh, base, meta, max_bytes_per_tensor)
        if z is not None:
            fracs.append(z)
    if len(fracs) < 3:
        return {"verdict": UNKNOWN, "reason": "unsupported_dtype", "n": len(fracs)}
    hits = sum(1 for z in fracs if z >= ZERO_FRACTION_MIN)
    med = sorted(fracs)[len(fracs) // 2]
    out = {"n": len(fracs), "hits": hits, "median_zero_fraction": round(med, 4), "max_zero_fraction": round(max(fracs), 4)}
    if hits >= ZERO_FRACTION_MIN_HITS:
        return {"verdict": DAMAGED, "reason": "pruning_signature", **out}
    return {"verdict": UNKNOWN, "reason": "no_pruning_signature", **out}


# ---------------------------------------------------------------------------------------------------------------------
# fingerprint (2026-09-22): research/20260922-damage-detector recommended rule against a shipped, offline table
# ---------------------------------------------------------------------------------------------------------------------
# data/base_fingerprints.json holds, for every public model seen in the tournament pool (<= 35 B; the validator
# never augments larger ones), per-tensor {dtype, shape, sha256 of the first 16 KiB of stored bytes}, the full sha256
# of every 1-D tensor (norms / biases: the source-identification key) and sha256(model_id)[:16]. Built by
# research/20260922-prod-patch/fingerprints/build_fingerprints.py with HTTP range reads; no network at task time.
#
# Rule (validated offline on the 29 validator-damaged / 27 clean public pairs: 29/29 damaged, 0/27 false alarms):
#   1. candidate source: the model argument (literal id or its unsalted name hash) and/or every validator-protected
#      1-D tensor (name contains "norm"; the validator never touches norms) byte-identical to one table entry.
#   2. compare the first 16 KiB of every tensor with that entry: any differing tensor -> "damaged"; every tensor
#      present and identical -> "clean" (prefix-identical); dtype re-save / missing tensors -> "unknown".
#   3. source not in the table: local reference-free checks on the first 32 KiB of each per-layer matrix:
#      magnitude pruning (>= 2 matrices with >= 10% exact zeros) or bf16 weight scaling (>= 3 matrices with >= 2%
#      empty mantissa codes, unless >= 75% of the tested matrices are gapped) -> "damaged".
#   4. anything else -> "unknown" (the caller keeps its route). A name-hash miss alone is never evidence
#      (MODEL_HASH_SALT would turn every task into a miss).
# Shipped as plain, line-per-tensor JSON (readable source; no compressed blob in the submitted tree). load_table also
# reads a gzip copy (.gz suffix) for research builds.
FINGERPRINT_TABLE = Path(__file__).resolve().parent / "data" / "base_fingerprints.json"
FINGERPRINT_SCHEMA = "sn56.base_fingerprints"
FINGERPRINT_VERSION = 1
PREFIX_BYTES = 16384          # bytes hashed per tensor (first 16 KiB of the stored data range)
PREFIX_HEX = 32               # hex digits kept of each prefix / 1-D sha256
NAME_HASH_LEN = 16            # the validator's anonymous model argument: sha256(model_id)[:16] (MODEL_HASH_SALT empty)
ONE_D_MAX_BYTES = 1 << 20     # 1-D tensors up to this size are hashed in full
MIN_ID_NORMS = 2              # protected 1-D tensors that must match to name a source
R_BYTES = 32768               # reference-free fallback: bytes read per per-layer matrix
R_ZF, R_ZF_N = 0.10, 2        # pruning: exact-zero fraction, matrices
R_GAP, R_GAP_N = 0.02, 3      # scaling: empty mantissa-code fraction, matrices
_LAYER_RE = re.compile(r"\.(\d+)\.")


def name_hash(model_id: str) -> str:
    return hashlib.sha256(str(model_id).encode()).hexdigest()[:NAME_HASH_LEN]


def is_norm(name: str) -> bool:
    """Validator-protected 1-D tensors (select_target_layers skips names containing "norm")."""
    return "norm" in name.lower()


def _hx(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:PREFIX_HEX]


def _coalesce(spans: list[tuple[int, int]], gap: int = 1 << 16, cap: int = 4 << 20) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap and b - out[-1][0] <= cap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def fingerprint_files(files: Sequence[Any], read: Callable[[Any, int, int], bytes], *, threads: int = 1) -> dict[str, Any]:
    """Fingerprint safetensors files through ``read(file, start, end) -> bytes`` (local file or HTTP range reads).
    Returns {"tensors": {name: [dtype, shape, prefix_hex]}, "one_d": {name: full_hex}, "bytes_read": n}."""
    tensors: dict[str, list[Any]] = {}
    one_d: dict[str, str] = {}
    nread = 0
    for f in files:
        n = struct.unpack("<Q", read(f, 0, 8))[0]
        header = json.loads(read(f, 8, 8 + n))
        base = 8 + n
        nread += 8 + n
        want: list[tuple[str, int, int, int]] = []  # name, start, prefix_end, full_end
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            full = len(meta.get("shape") or []) <= 1 and (e - s) <= ONE_D_MAX_BYTES
            want.append((name, base + s, base + min(e, s + PREFIX_BYTES), base + e if full else -1))
        spans = _coalesce([(a, max(p, fe)) for _, a, p, fe in want if max(p, fe) > a])
        if threads > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(threads) as ex:
                blocks = dict(zip(spans, ex.map(lambda sp: read(f, sp[0], sp[1]), spans)))
        else:
            blocks = {(a, b): read(f, a, b) for a, b in spans}
        nread += sum(b - a for a, b in spans)
        starts = [a for a, _ in spans]
        for name, a, p, fe in want:
            meta = header[name]
            end = max(p, fe)
            if end <= a:
                raw = b""
            else:
                i = bisect.bisect_right(starts, a) - 1
                sa, sb = spans[i]
                raw = blocks[(sa, sb)][a - sa:end - sa]
            tensors[name] = [meta["dtype"], list(meta["shape"]), _hx(raw[:p - a] if p > a else b"")]
            if fe >= 0:
                one_d[name] = _hx(raw)
    return {"tensors": tensors, "one_d": one_d, "bytes_read": nread}


def _local_files(model_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors")))


def _read_local(path: str, a: int, b: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(a)
        return fh.read(b - a)


def local_fingerprint(model_dir: str) -> dict[str, Any]:
    return fingerprint_files(_local_files(model_dir), _read_local)


_TABLE_CACHE: dict[str, dict[str, Any]] = {}


def load_table(path: str | os.PathLike | None = None) -> dict[str, Any] | None:
    p = str(path or FINGERPRINT_TABLE)
    if p not in _TABLE_CACHE:
        try:
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as fh:
                t = json.load(fh)
            if t.get("schema") != FINGERPRINT_SCHEMA or int(t.get("version", 0)) != FINGERPRINT_VERSION:
                return None
            t["_by_hash"] = {m["name_hash"]: k for k, m in t["models"].items()}
            _TABLE_CACHE[p] = t
        except Exception:  # noqa: BLE001 - no table is "unknown", never an error
            return None
    return _TABLE_CACHE[p]


def _norm_match(local: dict[str, str], ref: dict[str, str]) -> tuple[int, int]:
    """(matching, compared) over validator-protected 1-D tensors present on both sides."""
    keys = [k for k in ref if is_norm(k) and k in local]
    return sum(local[k] == ref[k] for k in keys), len(keys)


def identify_source(fp: dict[str, Any], table: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """Candidate public source for a local fingerprint: norms first, the model argument as a tie-break / fallback."""
    models = table["models"]
    named = None
    if model_name:
        named = model_name if model_name in models else table["_by_hash"].get(str(model_name).strip().lower())
    hits = []
    for key, m in models.items():
        ok, n = _norm_match(fp["one_d"], m["one_d"])
        if n >= MIN_ID_NORMS and ok == n:
            hits.append(key)
    if named and named in hits:
        return {"source": named, "by": "name+norms", "norm_hits": hits}
    if named:
        ok, n = _norm_match(fp["one_d"], models[named]["one_d"])
        if n == 0 and not hits:
            return {"source": named, "by": "name_only", "norm_hits": hits}
        if not hits:
            return {"source": None, "by": "name_norms_differ", "named": named, "norm_hits": hits}
    if hits:
        return {"source": None, "by": "norms", "norm_hits": hits}  # resolved by compare(): best prefix agreement
    return {"source": None, "by": "none", "norm_hits": []}


def compare_prefixes(fp: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    lt, rt = fp["tensors"], ref["tensors"]
    same, differ, dtype_changed, missing = 0, [], 0, 0
    for name, (dt, shape, h) in rt.items():
        loc = lt.get(name)
        if loc is None:
            missing += 1
        elif loc[0] != dt or list(loc[1]) != list(shape):
            dtype_changed += 1
        elif loc[2] == h:
            same += 1
        else:
            differ.append(name)
    extra = sum(1 for k in lt if k not in rt)
    return {"same": same, "differ": len(differ), "dtype_or_shape_changed": dtype_changed, "missing": missing,
            "extra": extra, "n_ref": len(rt), "differ_sample": sorted(differ)[:5],
            "differ_protected": sum(1 for k in differ if is_protected(k))}


def _verdict_from_compare(c: dict[str, Any]) -> tuple[str, str]:
    if c["differ"] == 0 and c["dtype_or_shape_changed"] == 0 and c["missing"] == 0 and c["extra"] == 0:
        return CLEAN, "prefix_identical"
    if c["differ"] > 0 and c["differ_protected"] == 0 and c["same"] > 0:
        return DAMAGED, "prefix_mismatch"
    if c["differ"] > 0:
        return UNKNOWN, "protected_tensor_differs"  # an embedding changed: another checkpoint, not augmentation
    return UNKNOWN, "not_comparable"  # dtype re-save, shape change or missing tensors with nothing differing


def _mantissa_gap(u16: Any) -> float | None:
    """bf16 grid test: fraction of the 128 mantissa codes left (nearly) empty in the dominant binades."""
    import numpy as np

    b = u16.astype(np.int64) & 0x7FFF
    exp = b >> 7
    ok = (exp > 0) & (exp < 255)
    if int(ok.sum()) < 4000:
        return None
    ev, ec = np.unique(exp[ok], return_counts=True)
    top = ev[np.argsort(ec)[-3:]]
    m = (b & 0x7F)[ok & np.isin(exp, top)]
    h = np.bincount(m, minlength=128).astype(float)
    pad = np.concatenate([h[-3:], h, h[:3]])
    med = np.array([np.median(np.delete(pad[i:i + 7], 3)) for i in range(128)])
    valid = med >= 20
    if int(valid.sum()) < 64:
        return None
    return float(((h < 0.35 * med) & valid).sum() / valid.sum())


def reference_free(files: Sequence[Any], read: Callable[[Any, int, int], bytes], *,
                   max_bytes: int = R_BYTES) -> dict[str, Any]:
    """Pruning / bf16-scaling signatures on the first ``max_bytes`` of every per-layer matrix (no reference)."""
    import numpy as np

    zfs: list[float] = []
    gaps: list[float] = []
    nread = 0
    for f in files:
        n = struct.unpack("<Q", read(f, 0, 8))[0]
        header = json.loads(read(f, 8, 8 + n))
        for name, meta in header.items():
            if name == "__metadata__" or len(meta.get("shape") or []) < 2 or is_protected(name) or "lm_head" in name:
                continue
            if not _LAYER_RE.search(name) or meta.get("dtype") not in ("BF16", "F16", "F32"):
                continue
            s, e = meta["data_offsets"]
            w = 4 if meta["dtype"] == "F32" else 2
            nb = min(e - s, max_bytes) // w * w
            if nb <= 0:
                continue
            raw = read(f, 8 + n + s, 8 + n + s + nb)
            nread += nb
            if meta["dtype"] == "F32":
                u = np.frombuffer(raw, "<u4")
                zfs.append(float(np.count_nonzero((u & 0x7FFFFFFF) == 0)) / u.size)
                if float(((u & 0xFFFF) != 0).mean()) <= 0.01:  # fp32 holding bf16 values: grid test applies
                    g = _mantissa_gap((u >> 16).astype(np.uint16))
                    if g is not None:
                        gaps.append(g)
            else:
                u = np.frombuffer(raw, "<u2")
                zfs.append(float(np.count_nonzero((u & 0x7FFF) == 0)) / u.size)
                if meta["dtype"] == "BF16":
                    g = _mantissa_gap(u)
                    if g is not None:
                        gaps.append(g)
    n_pruned = sum(z >= R_ZF for z in zfs)
    n_gap = sum(g >= R_GAP for g in gaps)
    globally_gapped = bool(gaps) and sum(g == 0 for g in gaps) < 0.25 * len(gaps)
    out = {"n_matrices": len(zfs), "n_pruned": n_pruned, "max_zero_fraction": round(max(zfs), 4) if zfs else None,
           "n_gap": n_gap, "n_gap_tested": len(gaps), "globally_gapped": globally_gapped, "bytes_read": nread}
    if n_pruned >= R_ZF_N:
        return {"verdict": DAMAGED, "reason": "pruning_signature", **out}
    if n_gap >= R_GAP_N and not globally_gapped:
        return {"verdict": DAMAGED, "reason": "scaling_signature", **out}
    return {"verdict": UNKNOWN, "reason": "no_reference_free_signature", **out}


def fingerprint_assess(files: Sequence[Any], read: Callable[[Any, int, int], bytes], *, model_name: str | None = None,
                       table: dict[str, Any] | None = None, fallback: bool = True,
                       fp: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full rule over an abstract reader (local files at task start; HTTP range reads in offline validation).
    ``fp`` reuses a fingerprint_files result computed earlier from the same files."""
    table = table if table is not None else load_table()
    fp = fp if fp is not None else fingerprint_files(files, read)
    out: dict[str, Any] = {"bytes_read": fp["bytes_read"], "n_tensors": len(fp["tensors"])}
    table_reason = "no_table"
    if table is not None:
        src = identify_source(fp, table, model_name)
        out.update(identified_by=src["by"], norm_hits=src["norm_hits"][:5])
        cands = [src["source"]] if src["source"] else list(src["norm_hits"])
        best = None
        for key in cands:
            c = compare_prefixes(fp, table["models"][key])
            rank = (_verdict_from_compare(c)[0] == CLEAN, c["same"] - c["dtype_or_shape_changed"] - c["missing"])
            if best is None or rank > best[0]:
                best = (rank, key, c)
        table_reason = "source_not_in_table"
        if best is not None:
            _, key, c = best
            verdict, table_reason = _verdict_from_compare(c)
            out.update(source=key, source_revision=table["models"][key].get("revision"), compare=c)
            if verdict != UNKNOWN:
                return {"verdict": verdict, "reason": table_reason, **out}
    else:
        out["identified_by"] = "no_table"
    if not fallback:
        return {"verdict": UNKNOWN, "reason": table_reason, **out}
    r = reference_free(files, read)
    out["bytes_read"] += int(r.get("bytes_read") or 0)
    reason = r["reason"] if r["verdict"] == DAMAGED else f"{table_reason}:{r['reason']}"
    return {"verdict": r["verdict"], "reason": reason,
            "reference_free": {k: v for k, v in r.items() if k not in ("verdict", "reason")}, **out}


def fingerprint(model_dir: str, identity: dict[str, Any] | None = None, model_name: str | None = None) -> dict[str, Any]:
    """Detector entry: table rule on the local safetensors (first 16 KiB per tensor + 1-D tensors; fallback 32 KiB per
    per-layer matrix). ``identity`` is unused (kept for the detector signature)."""
    files = _local_files(model_dir)
    if not files:
        return {"verdict": UNKNOWN, "reason": "no_safetensors"}
    return fingerprint_assess(files, _read_local, model_name=model_name)


DETECTORS: dict[str, Callable[..., dict[str, Any]]] = {"identity": identity_signature, "zero_fraction": zero_fraction,
                                                       "fingerprint": fingerprint}
_NAME_AWARE = {"fingerprint"}


def register(name: str, fn: Callable[..., dict[str, Any]]) -> None:
    """Hook for further detectors: fn(model_dir, identity_report) -> {"verdict": damaged|clean|unknown, ...}."""
    DETECTORS[name] = fn


def assess(model_dir: str, *, identity: dict[str, Any] | None, detectors: Sequence[str],
           model_name: str | None = None) -> dict[str, Any]:
    """Combine the named detectors. Never raises; unknown detector names and detector errors read "unknown"."""
    t0 = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    for name in detectors:
        fn = DETECTORS.get(name)
        if fn is None:
            results[name] = {"verdict": UNKNOWN, "reason": "no_such_detector"}
            continue
        try:
            rep = (fn(str(model_dir), identity, model_name=model_name) if name in _NAME_AWARE
                   else fn(str(model_dir), identity))
            results[name] = rep if isinstance(rep, dict) else {"verdict": UNKNOWN, "reason": "bad_result"}
        except Exception as exc:  # noqa: BLE001 - detection is advisory; the caller keeps its route
            results[name] = {"verdict": UNKNOWN, "reason": f"{type(exc).__name__}: {exc}"}
    verdicts = [r.get("verdict") for r in results.values()]
    if CLEAN in verdicts:  # a byte-identical public checkpoint is proof; no statistic overrides it
        verdict = CLEAN
    elif DAMAGED in verdicts:
        verdict = DAMAGED
    else:
        verdict = UNKNOWN
    by = [n for n, r in results.items() if r.get("verdict") == verdict] if verdict != UNKNOWN else []
    return {"verdict": verdict, "by": by, "detectors": results, "seconds": round(time.monotonic() - t0, 2)}
