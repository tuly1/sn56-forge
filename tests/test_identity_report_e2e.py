"""End-to-end through the production detection path: identity_report.identify() -> damage.assess() with the shipped
detector set ("identity", "fingerprint"), on synthetic safetensors checkpoints with their own identity tables and
fingerprint table (Codex review 2026-09-23). Also guards the byte-pinned base_identity.py of the Gemma release."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from forge.routing import base_identity as bid
from forge.routing import damage
from forge.routing import identity_report as idr

_fp_path = Path(__file__).resolve().parent / "test_damage_fingerprint.py"
_spec = importlib.util.spec_from_file_location("_fp_helpers", _fp_path)
fph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fph)

CONFIG = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 128, "num_hidden_layers": 4,
          "intermediate_size": 256, "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
          "tie_word_embeddings": False}
DETECTORS = ("identity", "fingerprint")


def _write(d: Path, tensors) -> str:
    fph._write_st(d / "model.safetensors", tensors)
    (d / "config.json").write_text(json.dumps(CONFIG))
    return str(d)


def _derivative(tensors, seed=42):
    """Same norms and embeddings as the source, different weights (a LoRA-merge-like public derivative)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (dt, x) in tensors.items():
        if idr.is_protected(k) or x.ndim < 2:
            out[k] = (dt, x)
        else:
            out[k] = (dt, fph._from_bf16_bits(fph._to_bf16_bits(x + rng.normal(0, 0.01, x.shape).astype(np.float32))))
    return out


@pytest.fixture()
def e2e(tmp_path, monkeypatch):
    pub_t = fph._model(1)
    pub = _write(tmp_path / "pub", pub_t)
    deriv = _write(tmp_path / "deriv", _derivative(pub_t))
    tables = tmp_path / "tables"  # identity table: org/Pub-1B only (like base_hashes/)
    tables.mkdir()
    (tables / "org__Pub-1B.json").write_text(json.dumps({"tensors": bid.hash_safetensors(pub)}))
    (tables / "org__Pub-1B.config.json").write_text(json.dumps(CONFIG))
    # fingerprint table: the source AND its same-norm derivative (like Llama-3.1-8B-Instruct / Storm-8B)
    tpath = fph._table({"org/Pub-1B": Path(pub), "org/Deriv-1B": Path(deriv)}, tmp_path / "fp.json.gz")
    monkeypatch.setattr(damage, "FINGERPRINT_TABLE", tpath)
    real = idr.identify
    monkeypatch.setattr(idr, "identify", lambda d, **kw: real(d, tables_dir=tables, **kw))

    def run(name, tensors=None, path=None):
        d = path or _write(tmp_path / name, tensors)
        rep = idr.identify(d)
        return rep, damage.assess(d, identity=rep, detectors=DETECTORS, model_name="0123456789abcdef")

    return {"pub": pub, "pub_t": pub_t, "deriv": deriv, "run": run}


def _scaled(t, name, *, start=0, factor=0.5):
    t = dict(t)
    dt, x = t[name]
    flat = x.copy().ravel()
    flat[start:] *= np.float32(factor)
    t[name] = (dt, fph._from_bf16_bits(fph._to_bf16_bits(flat)).reshape(x.shape))
    return t


def test_exact_clean(e2e):
    rep, out = e2e["run"]("clean", path=e2e["pub"])
    assert rep["status"] == rep["exact_status"] == "identical" and rep["candidate"] == "org__Pub-1B"
    assert (out["verdict"], out["reason"]) == ("clean", "exact_identity")


@pytest.mark.parametrize("start", [0, 16384 // 2 + 10])  # in-prefix / tail-only (after byte 16384)
def test_unprotected_alteration_is_damaged(e2e, start):
    t = _scaled(e2e["pub_t"], "model.layers.1.mlp.down_proj.weight", start=start)
    rep, out = e2e["run"](f"unprot-{start}", t)
    assert rep["protected_altered"] == 0 and rep["altered"] == 1 and rep["structure_changed"] == 0
    assert (out["verdict"], out["reason"]) == ("damaged", "exact_identity_altered"), out


def test_protected_alteration_is_unknown_even_when_the_prefix_says_damaged(e2e):
    t = _scaled(e2e["pub_t"], "model.embed_tokens.weight", start=16384 // 2 + 10, factor=1.5)  # after the prefix
    t = _scaled(t, "model.layers.2.self_attn.q_proj.weight")                                    # inside the prefix
    rep, out = e2e["run"]("prot", t)
    assert rep["protected_altered"] == 1
    assert out["detectors"]["fingerprint"]["verdict"] == "damaged"   # the prefix alone would route it
    assert (out["verdict"], out["reason"]) == ("unknown", "exact_identity:norm_or_embedding_altered"), out
    t = _scaled(e2e["pub_t"], "model.layers.0.input_layernorm.weight", factor=1.01)             # a norm
    t = _scaled(t, "model.layers.2.self_attn.q_proj.weight")
    _, out = e2e["run"]("prot-norm", t)
    assert out["verdict"] == "unknown"


@pytest.mark.parametrize("case", ["missing", "extra", "dtype_resave", "same_bytes_other_shape",
                                  "same_bytes_other_dtype"])
def test_structural_mismatch_is_unknown(e2e, case):
    t = _scaled(e2e["pub_t"], "model.layers.2.self_attn.q_proj.weight") if case in ("missing", "extra") \
        else dict(e2e["pub_t"])
    if case == "missing":
        t.pop("model.layers.3.self_attn.q_proj.weight")
    elif case == "extra":
        t["model.layers.3.mlp.gate_proj.weight"] = ("BF16", np.full((128, 256), 0.01, np.float32))
    elif case == "dtype_resave":
        dt, x = t["model.layers.1.mlp.down_proj.weight"]
        t["model.layers.1.mlp.down_proj.weight"] = ("F32", x)
    elif case == "same_bytes_other_shape":
        dt, x = t["model.layers.1.mlp.down_proj.weight"]
        t["model.layers.1.mlp.down_proj.weight"] = (dt, x.reshape(256, 128))
    d = Path(e2e["pub"]).parent / case
    _write(d, t)
    if case == "same_bytes_other_dtype":  # relabel one BF16 tensor as F16 in the header, bytes untouched
        raw = (d / "model.safetensors").read_bytes()
        n = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8 + n])
        header["model.layers.1.mlp.down_proj.weight"]["dtype"] = "F16"
        h = json.dumps(header).encode()
        h += b" " * (n - len(h))
        assert len(h) == n
        (d / "model.safetensors").write_bytes(raw[:8] + h + raw[8 + n:])
    rep, out = e2e["run"](case, path=str(d))
    if case.startswith("same_bytes"):
        assert rep["status"] == "identical" and rep["altered"] == 0  # base_identity (c0dc300) sees equal bytes ...
        assert rep["exact_status"] == "altered" and rep["structure_changed"] == 1  # ... the exact report does not
    assert out["verdict"] == "unknown", out
    assert out["reason"] in ("exact_identity:structure_differs", "exact_identity:tensor_set_differs"), out


def test_cross_reference_conflict_is_unknown(e2e):
    # the delivered base IS the tabled same-norm derivative: identity (only org/Pub-1B tabled) says altered vs Pub,
    # the fingerprint says prefix-identical to org/Deriv-1B -> contradictory -> unknown, never "damaged"
    rep, out = e2e["run"]("deriv", path=e2e["deriv"])
    assert rep["candidate"] == "org__Pub-1B" and rep["protected_altered"] == 0 and rep["altered"] > 0
    assert out["detectors"]["identity"]["verdict"] == "damaged"
    assert out["detectors"]["fingerprint"]["verdict"] == "clean"
    assert out["detectors"]["fingerprint"]["source"] == "org/Deriv-1B"
    assert (out["verdict"], out["reason"]) == ("unknown", "contradictory_references"), out


def test_identity_report_keeps_the_c0dc300_status_and_candidate(e2e, tmp_path):
    t = _scaled(e2e["pub_t"], "model.layers.1.mlp.down_proj.weight")
    d = _write(tmp_path / "cmp", t)
    tables = tmp_path / "tables"
    legacy = bid.identify(d, tables_dir=tables)
    ext = idr.identify(d)
    assert {k: ext[k] for k in legacy if k != "seconds"} == {k: legacy[k] for k in legacy if k != "seconds"}


def test_gemma_release_binding_is_intact():
    """The Sept 21 Gemma small-batch admission pins base_identity.py and its bundled reference by sha256; any edit
    silently returns the incumbent route (reference_helper_binding_mismatch)."""
    from forge.routing import gemma_identity_gate as gate
    from forge.routing import gemma_small_batch_admission as adm

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert sha(bid.__file__) == adm.EXPECTED_FIELDS["identity_helper_sha256"]
    assert sha(gate.__file__) == adm.EXPECTED_GATE_SHA256
    ref = adm.BUNDLED_REFERENCE_DIR
    anchor = json.loads((ref / "ROOT-REFERENCE-ANCHOR.json").read_text())
    ok, reason = gate._verify_reference_receipt(ref, adm.EXPECTED_FIELDS["public_key"], ref / "BUILD-RECEIPT.json",
                                                bid, anchor)
    assert ok, reason
