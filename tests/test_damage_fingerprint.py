"""forge.routing.damage fingerprint detector (2026-09-22): norm-hash source identification + first-16 KiB prefix
comparison against the shipped offline table, reference-free pruning / bf16-scaling fallback, "unknown" otherwise.

Synthetic damage follows the validator's own semantics (trainer/model_prep/augmentation.py, pinned G.O.D source in
research/20260915-codex-cpu-workstreams/augmentation/official-source): target = names containing "weight" but not
"norm" / "embed"; math in float32, cast back to the stored dtype; noise std = intensity * std; scaling x intensity;
pruning below the |w| quantile; re-init replaces a Bernoulli(intensity) mask with N(mean, std)."""
import gzip
import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest

from forge.routing import damage

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO.parents[1] / "research"
QWEN05 = RESEARCH / "20260921-next-week/execution/custody/staged/unsloth--Qwen2-0.5B"


# ---------------------------------------------------------------------------------------------------------------------
# synthetic checkpoints
# ---------------------------------------------------------------------------------------------------------------------
def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def _write_st(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """tensors: name -> (dtype tag, float32 values); BF16 values are rounded to nearest even."""
    header, blobs, off = {}, [], 0
    for name, (dt, x) in tensors.items():
        if dt == "BF16":
            raw = _to_bf16_bits(x).tobytes()
        elif dt == "F32":
            raw = x.astype("<f4").tobytes()
        else:
            raw = x.astype("<f2").tobytes()
        header[name] = {"dtype": dt, "shape": list(x.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _model(seed: int, dim: int = 128, layers: int = 4, dtype: str = "BF16") -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    t = {"model.embed_tokens.weight": (dtype, rng.normal(0, 0.02, (256, dim)))}
    for i in range(layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = (dtype, 1 + rng.normal(0, 0.05, dim))
        t[p + "self_attn.q_proj.weight"] = (dtype, rng.normal(0, 0.03, (dim, dim)))
        t[p + "self_attn.q_proj.bias"] = (dtype, rng.normal(0, 0.01, dim))
        t[p + "mlp.down_proj.weight"] = (dtype, rng.normal(0, 0.03, (dim, 2 * dim)))
        t[p + "post_attention_layernorm.weight"] = (dtype, 1 + rng.normal(0, 0.05, dim))
    t["model.norm.weight"] = (dtype, 1 + rng.normal(0, 0.05, dim))
    # round through the storage dtype once so "the public checkpoint" is exactly what is stored
    return {k: (dt, _from_bf16_bits(_to_bf16_bits(x)) if dt == "BF16" else x.astype(np.float32)) for k, (dt, x) in t.items()}


def _augment(tensors, aug_type: str, intensity: float, seed: int, scope: str = "multi_layer"):
    """Validator semantics (augment_model + apply_augmentation) on stored float32 values."""
    names = [n for n in tensors if "weight" in n and "norm" not in n.lower() and "embed" not in n.lower()]
    rng_sel = random.Random(seed)
    if scope == "all_layers":
        target = names
    else:
        target = rng_sel.sample(names, max(1, int(len(names) * rng_sel.uniform(0.25, 0.75))))
    rng = np.random.default_rng(seed)
    out = dict(tensors)
    for name in tensors:  # named_parameters order
        if name not in target:
            continue
        dt, data = tensors[name]
        data = data.astype(np.float32)
        if aug_type == "gaussian_noise":
            data = data + rng.normal(0, intensity * data.std(), size=data.shape).astype(np.float32)
        elif aug_type == "weight_scaling":
            data = data * np.float32(intensity)
        elif aug_type == "magnitude_pruning":
            thr = np.quantile(np.abs(data).ravel(), intensity)
            data = data * (np.abs(data) >= thr)
        elif aug_type == "layer_reinit":
            m, s = float(data.mean()), float(data.std())
            mask = rng.random(size=data.shape) < intensity
            data = np.where(mask, rng.normal(m, s, size=data.shape).astype(np.float32), data)
        out[name] = (dt, data)
    return out


def _table(entries: dict[str, Path], path: Path) -> Path:
    models = {}
    for name, d in entries.items():
        fp = damage.local_fingerprint(str(d))
        models[name] = {"revision": "0" * 40, "name_hash": damage.name_hash(name), "tensors": fp["tensors"],
                        "one_d": fp["one_d"], "config": {}, "files": ["model.safetensors"], "params": 1, "roles": []}
    t = {"schema": damage.FINGERPRINT_SCHEMA, "version": damage.FINGERPRINT_VERSION, "built_utc": "test",
         "models": models, "not_fingerprinted": {}}
    with gzip.open(path, "wt") as fh:
        json.dump(t, fh)
    damage._TABLE_CACHE.pop(str(path), None)
    return path


@pytest.fixture()
def world(tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    _write_st(pub / "model.safetensors", _model(1))
    other = tmp_path / "other"
    _write_st(other / "model.safetensors", _model(2))
    tpath = _table({"org/Pub-1B": pub, "org/Other-1B": other}, tmp_path / "t.json.gz")
    monkeypatch.setattr(damage, "FINGERPRINT_TABLE", tpath)
    return {"tmp": tmp_path, "pub": pub, "tensors": _model(1)}


def _save(world, name, tensors):
    d = world["tmp"] / name
    _write_st(d / "model.safetensors", tensors)
    return str(d)


# ---------------------------------------------------------------------------------------------------------------------
# synthetic tests
# ---------------------------------------------------------------------------------------------------------------------
def test_clean_copy_reads_clean_with_and_without_name(world):
    for name in (None, "org/Pub-1B", damage.name_hash("org/Pub-1B"), "0123456789abcdef"):
        rep = damage.fingerprint(str(world["pub"]), None, model_name=name)
        assert rep["verdict"] == "clean" and rep["reason"] == "prefix_identical" and rep["source"] == "org/Pub-1B", name


@pytest.mark.parametrize("aug,intensity", [("weight_scaling", 0.62), ("weight_scaling", 1.37), ("magnitude_pruning", 0.3),
                                           ("gaussian_noise", 0.02), ("layer_reinit", 0.01), ("layer_reinit", 0.08)])
@pytest.mark.parametrize("seed", [3, 11])
def test_validator_augmentation_reads_damaged(world, aug, intensity, seed):
    d = _save(world, f"aug-{aug}-{intensity}-{seed}", _augment(world["tensors"], aug, intensity, seed))
    rep = damage.fingerprint(d, None, model_name="a1b2c3d4e5f60718")  # anonymous, salted-style argument: no name hit
    assert rep["verdict"] == "damaged", rep
    assert rep["reason"] == "prefix_mismatch" and rep["source"] == "org/Pub-1B" and rep["identified_by"] == "norms"
    assert rep["compare"]["differ"] >= 1 and rep["compare"]["differ_protected"] == 0


def test_single_tensor_augmentation_is_caught(world):
    t = dict(world["tensors"])
    dt, x = t["model.layers.3.mlp.down_proj.weight"]
    t["model.layers.3.mlp.down_proj.weight"] = (dt, x * np.float32(1.01))
    rep = damage.fingerprint(_save(world, "one", t), None)
    assert rep["verdict"] == "damaged" and rep["compare"]["differ"] == 1


def test_dtype_resave_and_embedding_change_are_unknown(world):
    f32 = {k: ("F32", x) for k, (dt, x) in world["tensors"].items()}
    rep = damage.fingerprint(_save(world, "f32", f32), None)
    assert rep["verdict"] == "unknown"  # norms re-encoded: no source, no reference-free signature
    t = dict(world["tensors"])
    dt, x = t["model.embed_tokens.weight"]
    t["model.embed_tokens.weight"] = (dt, x + np.float32(0.01))
    dt, x = t["model.layers.0.self_attn.q_proj.weight"]
    t["model.layers.0.self_attn.q_proj.weight"] = (dt, x + np.float32(0.01))
    rep = damage.fingerprint(_save(world, "emb", t), None)
    assert rep["verdict"] == "unknown" and rep["reason"] == "protected_tensor_differs", rep  # final: no fallback


# ---- structural mismatch is never "damaged" (Codex review 2026-09-23) ------------------------------------------------
_CMP = dict(same=10, differ=3, differ_protected=0, dtype_or_shape_changed=0, missing=0, extra=0, n_ref=13)


@pytest.mark.parametrize("kw,verdict,reason", [
    ({}, "damaged", "prefix_mismatch"),
    (dict(missing=1), "unknown", "structure_differs"),
    (dict(extra=2), "unknown", "structure_differs"),
    (dict(dtype_or_shape_changed=1), "unknown", "structure_differs"),
    (dict(missing=1, extra=1, dtype_or_shape_changed=4), "unknown", "structure_differs"),
    (dict(missing=1, differ=0), "unknown", "structure_differs"),          # nothing differs but the set is incomplete
    (dict(differ=0), "clean", "prefix_identical"),
    (dict(differ_protected=1), "unknown", "protected_tensor_differs"),
    (dict(same=0), "unknown", "nothing_identical"),
])
def test_verdict_from_compare_structure(kw, verdict, reason):
    assert damage._verdict_from_compare(dict(_CMP, **kw)) == (verdict, reason)



@pytest.mark.parametrize("case", ["missing", "extra", "dtype", "shape", "combined", "pruned_missing"])
def test_structurally_different_damaged_copy_is_unknown(world, case):
    t = _augment(world["tensors"], "magnitude_pruning", 0.4, 5, "all_layers") if case == "pruned_missing" \
        else _augment(world["tensors"], "weight_scaling", 0.62, 3)
    if case in ("missing", "combined", "pruned_missing"):
        t.pop("model.layers.2.self_attn.q_proj.weight")
    if case in ("extra", "combined"):
        t["model.layers.3.mlp.gate_proj.weight"] = ("BF16", np.ones((128, 256), np.float32) * 0.01)
    if case in ("dtype", "combined"):
        dt, x = t["model.layers.1.mlp.down_proj.weight"]
        t["model.layers.1.mlp.down_proj.weight"] = ("F32", x)
    if case == "shape":
        dt, x = t["model.layers.1.mlp.down_proj.weight"]
        t["model.layers.1.mlp.down_proj.weight"] = (dt, x.reshape(256, 128))
    d = _save(world, f"struct-{case}", t)
    rep = damage.fingerprint(d, None, model_name="a1b2c3d4e5f60718")
    assert rep["source"] == "org/Pub-1B" and rep["identified_by"] == "norms"  # norms still name the public source
    assert rep["verdict"] == "unknown" and rep["reason"] == "structure_differs", rep
    assert "reference_free" not in rep  # an identified source is final: pruning statistics cannot override it
    assert damage.assess(d, identity=None, detectors=("identity", "fingerprint"))["verdict"] == "unknown"
    # the same damage without the structural change is "damaged"
    base = _augment(world["tensors"], "magnitude_pruning", 0.4, 5, "all_layers") if case == "pruned_missing" \
        else _augment(world["tensors"], "weight_scaling", 0.62, 3)
    assert damage.fingerprint(_save(world, f"struct-{case}-ok", base), None)["verdict"] == "damaged"


def test_name_with_different_norms_is_not_trusted(world):
    rep = damage.fingerprint(str(world["tmp"] / "other"), None, model_name="org/Pub-1B")
    # the argument names Pub but the norms are Other's: identification by norms wins
    assert rep["verdict"] == "clean" and rep["source"] == "org/Other-1B"
    d = _save(world, "stranger", _model(99))
    rep = damage.fingerprint(d, None, model_name="org/Pub-1B")
    assert rep["verdict"] == "unknown" and rep["identified_by"] == "name_norms_differ"


def test_reference_free_fallback_when_source_not_in_table(world):
    stranger = _model(7, layers=8)
    assert damage.fingerprint(_save(world, "s-clean", stranger), None)["verdict"] == "unknown"
    rep = damage.fingerprint(_save(world, "s-pruned", _augment(stranger, "magnitude_pruning", 0.4, 5, "all_layers")), None)
    assert rep["verdict"] == "damaged" and rep["reason"] == "pruning_signature"
    rep = damage.fingerprint(_save(world, "s-scaled", _augment(stranger, "weight_scaling", 1.3, 5)), None)
    assert rep["verdict"] == "damaged" and rep["reason"] == "scaling_signature", rep
    rep = damage.fingerprint(_save(world, "s-noise", _augment(stranger, "gaussian_noise", 0.1, 5, "all_layers")), None)
    assert rep["verdict"] == "unknown"  # noise leaves no reference-free trace (measured)


def test_no_table_and_no_files(world, monkeypatch, tmp_path):
    monkeypatch.setattr(damage, "FINGERPRINT_TABLE", tmp_path / "missing.json.gz")
    rep = damage.fingerprint(str(world["pub"]), None)
    assert rep["verdict"] == "unknown" and rep["identified_by"] == "no_table"
    assert damage.fingerprint(str(tmp_path / "nothing"), None)["verdict"] == "unknown"


def test_assess_passes_model_name_and_reads_prefixes_only(world):
    t = _augment(world["tensors"], "weight_scaling", 0.8, 1)
    d = _save(world, "via-assess", t)
    rep = damage.assess(d, identity=None, detectors=("identity", "fingerprint"), model_name="org/Pub-1B")
    assert rep["verdict"] == "damaged" and rep["by"] == ["fingerprint"]
    assert rep["detectors"]["fingerprint"]["identified_by"] == "name+norms"
    rep = damage.assess(str(world["pub"]), identity=None, detectors=("fingerprint",))
    assert rep["verdict"] == "clean"


def test_prefix_read_is_bounded(tmp_path):
    big = {"model.layers.0.mlp.down_proj.weight": ("BF16", np.zeros((1024, 1024), np.float32)),
           "model.layers.0.input_layernorm.weight": ("BF16", np.ones(1024, np.float32))}
    _write_st(tmp_path / "m" / "model.safetensors", big)
    fp = damage.local_fingerprint(str(tmp_path / "m"))
    assert fp["bytes_read"] < 8 + 400 + damage.PREFIX_BYTES + 2048 + (1 << 16)
    assert set(fp["one_d"]) == {"model.layers.0.input_layernorm.weight"}


# ---------------------------------------------------------------------------------------------------------------------
# shipped table
# ---------------------------------------------------------------------------------------------------------------------
def test_shipped_table_schema():
    assert damage.FINGERPRINT_TABLE.exists()
    t = damage.load_table()
    assert t is not None and t["schema"] == damage.FINGERPRINT_SCHEMA and t["version"] == damage.FINGERPRINT_VERSION
    assert t["prefix_bytes"] == damage.PREFIX_BYTES and t["built_utc"]
    assert len(t["models"]) >= 40
    for mid, m in t["models"].items():
        assert len(m["revision"]) == 40 and m["name_hash"] == damage.name_hash(mid)
        assert m["tensors"] and all(len(v) == 3 for v in m["tensors"].values())
    # the 27 public sources of the damaged-base validation set are all covered
    for mid in ("Qwen/Qwen2.5-3B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
                "microsoft/phi-1_5", "zake7749/gemma-2-2b-it-chinese-kyara-dpo", "unsloth/Qwen2-0.5B"):
        assert mid in t["models"], mid


# ---------------------------------------------------------------------------------------------------------------------
# real bytes: the staged public unsloth/Qwen2-0.5B (research custody), clean and with validator damage applied to the
# bytes the detector reads (no copy of the 1 GB file is written)
# ---------------------------------------------------------------------------------------------------------------------
_real = pytest.mark.skipif(not (QWEN05 / "model.safetensors").exists(), reason="research custody model not present")


@_real
def test_real_qwen2_0p5b_clean_against_shipped_table():
    rep = damage.fingerprint(str(QWEN05), None, model_name=damage.name_hash("unsloth/Qwen2-0.5B"))
    assert rep["verdict"] == "clean" and rep["source"] in ("unsloth/Qwen2-0.5B", "Qwen/Qwen2-0.5B"), rep
    assert rep["bytes_read"] < 8 << 20
    rep = damage.fingerprint(str(QWEN05), None)  # anonymous argument: norms alone identify it
    assert rep["verdict"] == "clean"


def _damaging_reader(path: Path, target: str, aug: str, intensity: float):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    s, e = header[target]["data_offsets"]
    a0, a1 = 8 + n + s, 8 + n + e

    def read(f, a, b):
        raw = bytearray(damage._read_local(f, a, b))
        lo, hi = max(a, a0), min(b, a1)
        if lo < hi:  # rewrite the overlapping part of the target tensor with validator math on its real values
            lo -= (lo - a0) % 2
            seg = np.frombuffer(bytes(raw[lo - a:hi - a]), "<u2")
            x = _from_bf16_bits(seg)
            if aug == "weight_scaling":
                y = x * np.float32(intensity)
            else:  # magnitude pruning with the threshold of this slice (enough for the prefix bytes)
                y = x * (np.abs(x) >= np.quantile(np.abs(x), intensity))
            raw[lo - a:hi - a] = _to_bf16_bits(y).tobytes()
        return bytes(raw)
    return read


@_real
@pytest.mark.parametrize("aug,intensity", [("weight_scaling", 0.93), ("magnitude_pruning", 0.3)])
def test_real_qwen2_0p5b_damaged_bytes(aug, intensity):
    f = str(QWEN05 / "model.safetensors")
    read = _damaging_reader(QWEN05 / "model.safetensors", "model.layers.12.mlp.up_proj.weight", aug, intensity)
    rep = damage.fingerprint_assess([f], read)
    assert rep["verdict"] == "damaged" and rep["compare"]["differ"] == 1, rep
    assert rep["compare"]["differ_sample"] == ["model.layers.12.mlp.up_proj.weight"]
