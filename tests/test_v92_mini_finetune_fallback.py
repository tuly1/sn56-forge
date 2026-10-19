"""v9.2 crash fallback (owner decision 2026-10-05): a real minimal fine-tune, else an honest failure.

Every test drives the real CLI (forge.cli.main) on CPU with a tiny random Llama base and a real tokenizer, and forces a
failure at one stage. The contract asserted everywhere:

  * exit 0  => the output path holds a TRAINED artifact (optimizer steps >= 1, changed adapter weights, a model that
               differs from the base, confirmed by the vendored G.O.D model_is_a_finetune);
  * exit 1  => nothing at the output path.

No path may leave an untrained artifact, and no untrained artifact may exist at the output path at any time.
"""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import pytest
import torch

from forge import cli, telemetry
from forge.data.schema import TaskSpec
from forge.tasks import common, dispatch, fallback, instruct, sft_v2
from god_vendor.model_checks_a2b14db import model_is_a_finetune

VOCAB = 64


def _make_base(root: Path) -> Path:
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    base = root / "base"
    cfg = transformers.LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=256,
                                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg.architectures = ["LlamaForCausalLM"]
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(base)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for i in range(4, VOCAB):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>")
    fast.chat_template = ("{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
                          "{% if add_generation_prompt %}assistant {% endif %}")
    fast.save_pretrained(base)
    return base


def _words(seed: int, n: int) -> str:
    return " ".join(f"w{4 + (seed * 7 + j * 3) % (VOCAB - 4)}" for j in range(n))


INSTRUCT_TYPE = json.dumps({"field_instruction": "instruction", "field_output": "output"})
CHAT_TYPE = json.dumps({"chat_column": "conversations", "chat_role_field": "from", "chat_content_field": "value",
                        "chat_user_reference": "user", "chat_assistant_reference": "assistant"})
DPO_TYPE = json.dumps({"field_prompt": "prompt", "field_chosen": "chosen", "field_rejected": "rejected"})
GRPO_TYPE = json.dumps({"field_prompt": "prompt", "reward_functions": [
    {"reward_func": "def reward(completions, **kwargs):\n    return [0.0 for _ in completions]\n", "reward_weight": 1.0}]})


def _rows(kind: str, n: int = 24, length: int = 6) -> list[dict]:
    if kind == "instruct":
        return [{"instruction": _words(i, length), "output": _words(i + 1, length)} for i in range(n)]
    if kind == "chat":
        return [{"conversations": [{"from": "user", "value": _words(i, length)},
                                   {"from": "assistant", "value": _words(i + 1, length)}]} for i in range(n)]
    if kind == "dpo":
        return [{"prompt": _words(i, length), "chosen": _words(i + 1, length), "rejected": _words(i + 2, length)}
                for i in range(n)]
    return [{"prompt": _words(i, length)} for i in range(n)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A staged task: tiny base, TRAIN file and an (absent) output path, with TaskSpec pointed at tmp_path."""
    for name in list(os.environ):
        if name.startswith("FORGE_") or name in {"USE_KL", "KL_COEF", "BASELINE_STATS_PATH"}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = _make_base(tmp_path)
    out = tmp_path / "checkpoints" / "task" / "repo"
    data = tmp_path / "train.json"
    monkeypatch.setattr(TaskSpec, "output_dir", property(lambda self: str(out)))
    monkeypatch.setattr(TaskSpec, "cached_model_dir", property(lambda self: str(base)))
    monkeypatch.setattr(TaskSpec, "cached_dataset_path", property(lambda self: str(data)))
    telemetry._data["events"].clear()

    class Env:
        pass

    e = Env()
    e.base, e.out, e.data, e.tmp = base, out, data, tmp_path

    def run(kind: str = "instruct", task_type: str = "InstructTextTask", hours: float = 1.0, rows=None,
            dataset_type: str | None = None) -> int:
        payload = _rows(kind) if rows is None else rows
        data.write_text(json.dumps(payload))
        dtype = dataset_type or {"instruct": INSTRUCT_TYPE, "chat": CHAT_TYPE, "dpo": DPO_TYPE, "grpo": GRPO_TYPE}[kind]
        return cli.main(["--task-id", "task", "--model", "anon/base", "--dataset", str(data), "--dataset-type", dtype,
                         "--task-type", task_type, "--file-format", "s3", "--expected-repo-name", "repo",
                         "--hours-to-complete", str(hours)])

    e.run = run
    return e


def _events() -> list[str]:
    return [ev["name"] for ev in telemetry._data["events"]]


def _event(name: str) -> dict:
    return next(ev for ev in telemetry._data["events"] if ev["name"] == name)


def _fail_handler(monkeypatch, exc: BaseException | None = None):
    def handler(spec, deadline):
        raise exc or RuntimeError("forced handler failure")

    monkeypatch.setattr(dispatch, "for_task", lambda task_type: handler)


def assert_trained(e, *, objective: str | None = None) -> dict:
    """The output path holds a real, trained adapter that the validator would accept as a finetune."""
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    out = e.out
    assert fallback.output_is_trained(str(out))
    truth = json.loads((out / "forge_artifact_truth.json").read_text())
    assert truth["truth"] == "PARTIAL_TRAINED_BEST" and truth["optimizer_step"] >= 1
    assert truth["reason"] == "mini_finetune"
    done = _event("mini_finetune_done")
    assert done["steps"] == truth["optimizer_step"] >= 1
    assert all(isinstance(done[k], float) and done[k] == done[k] for k in ("first_loss", "last_loss", "mean_loss"))
    if objective:
        assert done["objective"] == objective
    # the adapter's B matrices start at zero: any non-zero B is an applied optimizer step
    tensors = load_file(str(out / "adapter_model.safetensors"))
    b_norm = sum(float(t.float().abs().sum()) for name, t in tensors.items() if "lora_B" in name)
    assert b_norm > 0.0
    assert all(bool(torch.isfinite(t.float()).all()) for t in tensors.values())
    # the validator's view: base + adapter differs from the base and is a finetune of it
    ids = torch.tensor([[1, 5, 9, 13, 17, 21]])
    base_model = AutoModelForCausalLM.from_pretrained(str(e.base), dtype=torch.float32)
    with torch.no_grad():
        base_logits = base_model(input_ids=ids).logits.clone()
    merged = PeftModel.from_pretrained(base_model, str(out)).merge_and_unload()
    with torch.no_grad():
        assert not torch.allclose(merged(input_ids=ids).logits, base_logits)
    assert model_is_a_finetune(str(e.base), merged) is True
    assert not Path(str(out) + ".tmp").exists() and not Path(str(out) + ".old").exists()
    return done


def assert_failed_clean(e, rc: int) -> None:
    assert rc == 1
    assert not e.out.exists()
    assert not Path(str(e.out) + ".tmp").exists() and not Path(str(e.out) + ".old").exists()
    assert "task_failed_no_artifact" in _events()
    assert "mini_finetune_done" not in _events()


# ---- a failure at each stage ends in a trained artifact ---------------------------------------------------------------

def test_handler_failure_runs_a_real_fine_tune(env, monkeypatch):
    _fail_handler(monkeypatch)
    assert env.run() == 0
    done = assert_trained(env, objective="instruct_sft")
    assert done["steps"] == 24 and done["lora_r"] == fallback.MINI_LORA_R  # one TRAIN row per step
    assert "handler_failed" in _events() and "run_complete_mini_finetune" in _events()


@pytest.mark.parametrize("exc", [SystemExit(3), KeyboardInterrupt(), MemoryError("host RAM")])
def test_base_exceptions_from_the_handler_also_train(env, monkeypatch, exc):
    _fail_handler(monkeypatch, exc)
    assert env.run() == 0
    assert_trained(env)


def test_missing_handler_for_a_supported_type_trains(env, monkeypatch):
    monkeypatch.setattr(dispatch, "for_task", lambda task_type: None)
    assert env.run("chat", "ChatTask") == 0
    assert_trained(env, objective="chat_sft")
    assert "handler_missing" in _events()


def test_dispatch_import_failure_trains(env, monkeypatch):
    def boom(task_type):
        raise ImportError("handler import failed")

    monkeypatch.setattr(dispatch, "for_task", boom)
    assert env.run() == 0
    assert_trained(env)
    assert "dispatch_failed" in _events()


def test_dpo_handler_failure_trains_on_the_chosen_response(env, monkeypatch):
    _fail_handler(monkeypatch)
    assert env.run("dpo", "DpoTask") == 0
    assert_trained(env, objective="dpo_chosen_sft")


def test_completion_style_task_trains(env, monkeypatch):
    _fail_handler(monkeypatch)
    rows = [{"instruction": _words(i, 40)} for i in range(8)]
    assert env.run(rows=rows, dataset_type=json.dumps({"field_instruction": "instruction"})) == 0
    assert_trained(env, objective="completion_sft")


def test_handler_that_returns_without_an_artifact_trains(env, monkeypatch):
    monkeypatch.setattr(dispatch, "for_task", lambda task_type: (lambda spec, deadline: None))
    assert env.run() == 0
    assert_trained(env)
    assert _event("no_trained_artifact")["handler_ok"] is True


def test_partial_write_from_a_crashed_stage_is_removed_first(env, monkeypatch):
    def handler(spec, deadline):
        os.makedirs(spec.output_dir + ".tmp")
        Path(spec.output_dir + ".tmp", "adapter_model.safetensors").write_bytes(b"truncated")
        os.makedirs(spec.output_dir)
        Path(spec.output_dir, "adapter_config.json").write_text("{}")
        Path(spec.output_dir, "stale.bin").write_bytes(b"x")
        raise RuntimeError("crashed mid-save")

    monkeypatch.setattr(dispatch, "for_task", lambda task_type: handler)
    assert env.run() == 0
    assert_trained(env)
    assert not (env.out / "stale.bin").exists()


def _write_adapter(path: Path, truth: str | None, step: int = 0) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    save_file({"lora_B": torch.zeros(2, 2)}, str(path / "adapter_model.safetensors"))
    if truth is not None:
        (path / "forge_artifact_truth.json").write_text(
            json.dumps({"schema": 1, "truth": truth, "optimizer_step": step, "reason": "test"}))


def test_floor_labelled_artifact_is_never_kept(env, monkeypatch):
    """An artifact labelled FLOOR (what d44a582 wrote before training) is replaced by the trained mini fine-tune."""
    def handler(spec, deadline):
        _write_adapter(Path(spec.output_dir), "FLOOR", 0)
        raise RuntimeError("crashed after an old-style floor")

    monkeypatch.setattr(dispatch, "for_task", lambda task_type: handler)
    assert env.run() == 0
    assert_trained(env)


def test_trained_checkpoint_from_a_late_failure_is_kept(env, monkeypatch):
    calls = []

    def handler(spec, deadline):
        _write_adapter(Path(spec.output_dir), "PARTIAL_TRAINED_BEST", 37)
        raise RuntimeError("late failure after a trained persist")

    monkeypatch.setattr(dispatch, "for_task", lambda task_type: handler)
    monkeypatch.setattr(fallback, "run_mini_finetune", lambda *a, **k: calls.append(1))
    assert env.run() == 0
    assert calls == [] and "handler_failed_kept_trained_artifact" in _events()
    assert json.loads((env.out / "forge_artifact_truth.json").read_text())["optimizer_step"] == 37


def test_oom_on_long_rows_retries_shorter_rows(env, monkeypatch):
    import transformers

    _fail_handler(monkeypatch)
    real = transformers.LlamaForCausalLM.forward

    def forward(self, input_ids=None, **kw):
        if input_ids is not None and input_ids.shape[1] > 70:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return real(self, input_ids=input_ids, **kw)

    monkeypatch.setattr(transformers.LlamaForCausalLM, "forward", forward)
    rows = _rows("instruct", 6, 60) + _rows("instruct", 10, 6)  # long rows first, then short ones
    assert env.run(rows=rows) == 0
    done = assert_trained(env)
    assert done["steps"] == 10 and done["seq_cap"] < 70
    assert "mini_finetune_oom" in _events()


def test_failure_inside_the_export_reserve_still_trains(env, monkeypatch):
    """The handler fails after the soft stop: the mini fine-tune uses the export reserve and keeps its save reserve."""
    _fail_handler(monkeypatch)
    assert env.run(hours=170.0 / 3600.0) == 0  # 170 s < the 180 s export reserve: remaining() == 0
    assert_trained(env)
    assert _event("mini_finetune_start")["train_window_s"] <= 170.0 - fallback.MINI_SAVE_RESERVE_S


class _Clock:
    """fallback's view of time.monotonic, with a manual offset; forge.clock keeps the real clock."""

    def __init__(self):
        import time as real_time

        self._real, self.offset = real_time.monotonic, 0.0

    def monotonic(self) -> float:
        return self._real() + self.offset


def _slow_forward(monkeypatch, clock: _Clock, plan: dict[int, tuple[float, bool]]):
    """Model call k (1-based) advances fallback's clock by plan[k][0] seconds and, if plan[k][1], returns a NaN loss."""
    import transformers

    real, calls = transformers.LlamaForCausalLM.forward, []

    def forward(self, *a, **kw):
        calls.append(1)
        out = real(self, *a, **kw)
        seconds, nan = plan.get(len(calls), (0.0, False))
        clock.offset += seconds
        if nan:
            out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(transformers.LlamaForCausalLM, "forward", forward)
    monkeypatch.setattr(fallback, "time", clock)
    return calls


def test_a_slow_failed_attempt_counts_in_the_step_admission(env, monkeypatch):
    """RESP-106 C2: the admission estimate is the slowest attempt of any kind, not only the last applied step.

    Attempt 1 is slow and non-finite (no step), attempt 2 is fast and applies a step. A third attempt would need
    2 x 1500 s before train_end and only ~1900 s are left, so the loop stops with exactly one step and saves it.
    37eece2 looked only at the last applied step (fast) and went on to all 24 rows."""
    _fail_handler(monkeypatch)
    calls = _slow_forward(monkeypatch, _Clock(), {1: (1500.0, True)})
    assert env.run() == 0  # 1 h: train_end = the soft stop, 3420 s away
    calls = len(calls)  # before assert_trained, which runs the model again
    done = assert_trained(env)
    assert done["steps"] == 1 and done["skipped_nonfinite"] == 1 and calls == 2
    assert done["first_attempt_s"] >= 1500.0 and done["slowest_attempt_s"] >= 1500.0
    assert done["train_end_margin_s"] > 0.0


def test_nothing_trained_yet_keeps_trying_until_train_end(env, monkeypatch):
    """With no applied step there is nothing for the save reserve to protect: slow failed attempts do not end the run."""
    _fail_handler(monkeypatch)
    calls = _slow_forward(monkeypatch, _Clock(), {1: (1500.0, True), 2: (1500.0, True)})
    assert env.run() == 0
    calls = len(calls)
    done = assert_trained(env)
    assert done["steps"] == 1 and done["skipped_nonfinite"] == 2 and calls == 3


def test_first_step_overrun_into_the_save_reserve_is_recorded(env, monkeypatch):
    """The reserve is best-effort: a first step cannot be pre-empted. If it runs past train_end the receipt says so
    (negative train_end_margin_s) and no further step starts; the save still runs because the hard stop is not reached."""
    _fail_handler(monkeypatch)
    clock = _Clock()
    calls = _slow_forward(monkeypatch, clock, {1: (140.0, False)})  # 170 s task: train_end is 125 s away, hard stop 170 s
    assert env.run(hours=170.0 / 3600.0) == 0
    calls = len(calls)
    done = assert_trained(env)
    assert done["steps"] == 1 and calls == 1
    assert done["train_end_margin_s"] < 0.0 and done["first_attempt_s"] >= 140.0
    assert done["save_reserve_s"] == fallback.MINI_SAVE_RESERVE_S and done["save_s"] >= 0.0


def test_receipt_records_the_clock_facts(env, monkeypatch):
    _fail_handler(monkeypatch)
    assert env.run() == 0
    done = assert_trained(env)
    for key in ("load_s", "first_attempt_s", "slowest_attempt_s", "train_end_margin_s", "save_s", "remaining_hard_s"):
        assert isinstance(done[key], float) and done[key] == done[key], key
    assert 0.0 < done["first_attempt_s"] <= done["slowest_attempt_s"]


# ---- Mamba hybrids (Falcon-H1): PEFT refuses LoRA on out_proj / conv1d (phase-bw S5, Falcon-H1-3B-Base: HONEST_FAIL) ----

# tiiuae/Falcon-H1-0.5B-Base config.json (model_type falcon_h1; the multipliers, Mamba flags and head layout are the real
# ones), with the widths, depth, vocabulary and context shrunk so that it trains on CPU in a second.
FALCON_H1_05B = dict(
    attention_bias=False, attention_dropout=0.0, attention_in_multiplier=1.0, attention_out_multiplier=0.9375,
    embedding_multiplier=5.656854249492381, hidden_act="silu", initializer_range=0.02, key_multiplier=0.39062499999999994,
    lm_head_multiplier=0.0390625, mamba_conv_bias=True, mamba_d_conv=4, mamba_expand=2, mamba_n_groups=1,
    mamba_norm_before_gate=False, mamba_proj_bias=False, mamba_rms_norm=False, mamba_use_mlp=True, mlp_bias=False,
    mlp_expansion_factor=8, projectors_bias=False, rms_norm_eps=1e-05, ssm_in_multiplier=1.25,
    ssm_out_multiplier=0.23570226039551587, tie_word_embeddings=False, time_step_max=0.1, time_step_min=0.001, use_cache=False)
FALCON_H1_SHRUNK = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, mamba_d_ssm=96, mamba_d_state=16, mamba_n_heads=6, mamba_d_head=16, mamba_chunk_size=16,
                        vocab_size=VOCAB, max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def _make_falcon_base(base: Path) -> None:
    """Replace the tiny Llama weights in `base` with a tiny random Falcon-H1 (the tokenizer files stay)."""
    from transformers import FalconH1Config, FalconH1ForCausalLM

    for f in base.glob("*.safetensors"):
        f.unlink()
    cfg = FalconH1Config(**FALCON_H1_05B, **FALCON_H1_SHRUNK)
    cfg.architectures = ["FalconH1ForCausalLM"]
    torch.manual_seed(0)
    FalconH1ForCausalLM(cfg).save_pretrained(base)
    assert json.loads((base / "config.json").read_text())["model_type"] == "falcon_h1"


def _adapter_targets(e) -> set:
    """Leaf names of the adapter's targets (PEFT stores 'all-linear' expanded to full module paths)."""
    return {t.split(".")[-1] for t in json.loads((e.out / "adapter_config.json").read_text())["target_modules"]}


def test_all_linear_lora_is_refused_on_falcon_h1_and_leaves_the_model_half_wrapped(env):
    """The cause of the S5 failure, on the pinned PEFT: 37eece2..cd7e45f attached 'all-linear' and had no way out."""
    from transformers import AutoModelForCausalLM

    from forge.model import attach_lora

    _make_falcon_base(env.base)
    model = AutoModelForCausalLM.from_pretrained(str(env.base), dtype=torch.float32)
    with pytest.raises(ValueError, match="incompatible with Mamba-based models"):
        attach_lora(model, r=8, alpha=16, dropout=0.0)
    assert any("lora_" in name for name, _ in model.named_modules())  # so the safe list must be chosen BEFORE attaching
    names = fallback._linear_leaf_names(AutoModelForCausalLM.from_pretrained(str(env.base), dtype=torch.float32))
    assert {"out_proj", "in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= set(names)


def test_falcon_h1_handler_failure_trains_an_adapter_without_the_refused_layers(env, monkeypatch):
    _make_falcon_base(env.base)
    _fail_handler(monkeypatch)
    assert env.run() == 0
    done = assert_trained(env)  # steps >= 1, non-zero lora_B, merged logits differ, vendored model_is_a_finetune True
    assert done["lora_targets"] == "safe-list:-out_proj" and done["steps"] == 24
    targets = _adapter_targets(env)
    assert targets == {"in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert not targets & {"out_proj", "conv1d", "lm_head"}
    ev = _event("mini_finetune_lora_targets")
    assert ev["how"] == "safe-list:-out_proj" and "out_proj" in ev["linear_leaf_names"]


def test_falcon_h1_still_trains_when_peft_has_no_compatibility_helper(env, monkeypatch):
    """The pinned table in fallback.py stands in when PEFT's private check is missing or renamed."""
    import peft.tuners.tuners_utils as tu

    monkeypatch.delattr(tu, "_check_lora_target_modules_mamba")
    monkeypatch.setattr("peft.tuners.tuners_utils.BaseTuner._check_target_module_compatiblity", lambda *a, **k: None, raising=False)
    _make_falcon_base(env.base)
    _fail_handler(monkeypatch)
    assert env.run() == 0
    assert assert_trained(env)["lora_targets"] == "safe-list:-out_proj" and "out_proj" not in _adapter_targets(env)


def test_non_mamba_models_attach_exactly_as_before(env, monkeypatch):
    """Every other family: forge.model.attach_lora is called once with the same arguments; no safe list, no extra event."""
    from forge import model as forge_model

    _fail_handler(monkeypatch)
    real, calls = forge_model.attach_lora, []

    def spy(model, **kw):
        calls.append(kw)
        return real(model, **kw)

    monkeypatch.setattr(forge_model, "attach_lora", spy)
    assert env.run() == 0
    done = assert_trained(env)
    assert calls == [dict(r=fallback.MINI_LORA_R, alpha=fallback.MINI_LORA_ALPHA, dropout=0.0)]
    assert done["lora_targets"] == "all-linear" and "mini_finetune_lora_targets" not in _events()
    assert _adapter_targets(env) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_an_unpredicted_incompatible_refusal_reloads_the_base_and_retries_without_the_named_layers(env, monkeypatch):
    from forge import model as forge_model

    import weakref

    _fail_handler(monkeypatch)
    loads, real_load, first = [], forge_model.load_base, []

    def counting_load(*a, **k):
        loads.append(1)
        if first:  # RESP-120 C1: the reload must start only after the first model is gone (no double residency)
            assert first[0]() is None, "the refused model is still alive when the base is loaded again"
        out = real_load(*a, **k)
        if not first:
            first.append(weakref.ref(out.model))
        return out

    def refusing(model, **kw):
        raise ValueError("[PEFT:LORA] Module 'q_proj' is incompatible with Foo-based models. Incompatible modules: {'q_proj', 'zzz'}.")

    monkeypatch.setattr(forge_model, "load_base", counting_load)
    monkeypatch.setattr(forge_model, "attach_lora", refusing)
    assert env.run() == 0
    done = assert_trained(env)
    assert done["lora_targets"] == "safe-list-reloaded:-q_proj" and len(loads) == 2
    assert first and first[0]() is None
    assert _adapter_targets(env) == {"k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


@pytest.mark.parametrize("message", ["boom", "Module 'not_a_layer' is incompatible with X"])
def test_any_other_attach_error_still_fails_honestly(env, monkeypatch, message):
    from forge import model as forge_model

    _fail_handler(monkeypatch)
    monkeypatch.setattr(forge_model, "attach_lora", lambda model, **kw: (_ for _ in ()).throw(ValueError(message)))
    assert_failed_clean(env, env.run())
    assert message in _event("mini_finetune_failed")["error"]


def test_no_lora_compatible_layer_fails_honestly(env, monkeypatch):
    _fail_handler(monkeypatch)
    monkeypatch.setattr(fallback, "_lora_incompatible_names", lambda model, names: list(names))
    assert_failed_clean(env, env.run())
    assert "no LoRA-compatible linear layer" in _event("mini_finetune_failed")["error"]


# ---- the real instruct handler: every rung of the ladder fails --------------------------------------------------------

def test_full_cheap_full_and_adapter_rungs_all_fail(env, monkeypatch):
    """full -> cheap_full -> adapter all fail inside the real instruct handler; nothing untrained is ever at the output
    path, and the run ends with the trained mini fine-tune."""
    monkeypatch.setenv("FORGE_SFT_V2_ALLOW_CPU", "1")
    monkeypatch.setattr(instruct, "gpu_topology", lambda: (1, 80.0))
    rungs = []
    seen_output = []

    def failing_v2(spec, deadline, rows, loaded, tokenizer, **kw):
        rungs.append("cheap_full" if kw.get("cheap_full") else "full")
        seen_output.append(os.path.exists(spec.output_dir))
        raise RuntimeError("CUDA out of memory (forced)")

    def failing_adapter(*a, **k):
        rungs.append("adapter")
        seen_output.append(os.path.exists(env.out))
        raise RuntimeError("adapter rung failed (forced)")

    monkeypatch.setattr(sft_v2, "run", failing_v2)
    monkeypatch.setattr(sft_v2, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(instruct, "_train_ladder", failing_adapter)
    assert env.run() == 0
    assert rungs == ["full", "cheap_full", "adapter"]
    assert seen_output == [False, False, False]  # no pre-training floor at any rung
    names = _events()
    assert "sft_v2_fallback_to_cheap_full" in names and "sft_v2_fallback_to_lora" in names and "handler_failed" in names
    assert not {"fallback_emitted", "full_ft_floor"} & set(names)
    assert_trained(env, objective="instruct_sft")


def test_real_sft_v2_crash_before_training_leaves_nothing(env, monkeypatch):
    """The real sft_v2.run crashing in data preparation (after the point where d44a582 wrote its floor) leaves no file."""
    spec = TaskSpec.build(task_id="task", task_type="InstructTextTask", model="anon/base", dataset=str(env.data),
                          dataset_type_json=INSTRUCT_TYPE, expected_repo_name="repo", baseline_stats_path=None,
                          file_format="s3", use_kl=False, kl_coef=0.0)
    from forge.clock import Deadline
    from forge.model import load_base

    def boom(*a, **k):
        raise RuntimeError("tokenization crashed (forced)")

    monkeypatch.setattr(sft_v2.prompts, "build_instruct_examples", boom)
    loaded = load_base(str(env.base))
    import time

    with pytest.raises(RuntimeError, match="tokenization crashed"):
        sft_v2.run(spec, Deadline.from_hours(1.0, started_monotonic=time.monotonic(), export_reserve_s=180),
                   _rows("instruct"), loaded, loaded.tokenizer, baseline_summary=None, params_b=0.001, per_gpu_gb=80.0)
    assert not env.out.exists()


# ---- when the mini fine-tune cannot train: non-zero exit, no artifact -------------------------------------------------

def test_missing_base_fails_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)
    monkeypatch.setattr(TaskSpec, "cached_model_dir", property(lambda self: str(env.tmp / "no-such-base")))
    assert_failed_clean(env, env.run())


def test_corrupt_base_weights_fail_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)
    (env.base / "model.safetensors").write_bytes(b"corrupt")
    assert_failed_clean(env, env.run())


@pytest.mark.parametrize("rows", [[], [{"instruction": "w5 w6", "output": ""}], [{"unrelated": 1}]])
def test_no_trainable_rows_fail_with_no_artifact(env, monkeypatch, rows):
    _fail_handler(monkeypatch)
    assert_failed_clean(env, env.run(rows=rows))


def test_rows_longer_than_the_model_context_fail_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)  # the tiny base scores at most 128 tokens; these rows are ~200
    assert_failed_clean(env, env.run(rows=_rows("instruct", 8, 100)))
    assert "no TRAIN row fits" in _event("mini_finetune_failed")["error"]


def test_missing_train_file_fails_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)
    from forge.data import loader

    monkeypatch.setattr(loader, "load_rows", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no TRAIN")))
    assert_failed_clean(env, env.run())


def test_oom_at_every_cap_fails_with_no_artifact(env, monkeypatch):
    import transformers

    _fail_handler(monkeypatch)

    def forward(self, *a, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate")

    monkeypatch.setattr(transformers.LlamaForCausalLM, "forward", forward)
    assert_failed_clean(env, env.run(rows=_rows("instruct", 8, 40)))
    assert "out of memory" in _event("mini_finetune_failed")["error"]


def test_non_finite_loss_on_every_row_fails_with_no_artifact(env, monkeypatch):
    import transformers

    _fail_handler(monkeypatch)
    real = transformers.LlamaForCausalLM.forward

    def forward(self, *a, **kw):
        out = real(self, *a, **kw)
        out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(transformers.LlamaForCausalLM, "forward", forward)
    assert_failed_clean(env, env.run())
    assert "no optimizer step" in _event("mini_finetune_failed")["error"]


def test_zero_learning_rate_is_caught_as_unchanged_weights(env, monkeypatch):
    _fail_handler(monkeypatch)
    monkeypatch.setattr(fallback, "MINI_LR", 0.0)
    assert_failed_clean(env, env.run())
    assert "did not change" in _event("mini_finetune_failed")["error"]


def test_save_failure_fails_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)

    def bad_save(model, tokenizer, output_dir, **kw):
        os.makedirs(output_dir + ".tmp", exist_ok=True)
        Path(output_dir + ".tmp", "adapter_model.safetensors").write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(common, "save_adapter", bad_save)
    assert_failed_clean(env, env.run())


def test_no_wall_clock_fails_without_loading(env, monkeypatch):
    _fail_handler(monkeypatch)
    loads = []
    from forge import model as forge_model

    monkeypatch.setattr(forge_model, "load_base", lambda *a, **k: loads.append(1))
    assert_failed_clean(env, env.run(hours=30.0 / 3600.0))
    assert loads == []


def test_no_time_left_after_the_load_fails_with_no_artifact(env, monkeypatch):
    _fail_handler(monkeypatch)
    monkeypatch.setattr(fallback, "MINI_SAVE_RESERVE_S", 10 * 3600.0)
    assert_failed_clean(env, env.run())
    assert "no time left" in _event("mini_finetune_failed")["error"]


def test_grpo_failure_fails_honestly(env, monkeypatch):
    """GRPO TRAIN has prompts and reward functions but no target text: no supervised step exists, so the task fails."""
    _fail_handler(monkeypatch)
    assert_failed_clean(env, env.run("grpo", "GrpoTask"))


@pytest.mark.parametrize("task_type", ["EnvTask", "SomeFutureTask"])
def test_unsupported_task_types_fail_honestly(env, task_type):
    assert_failed_clean(env, env.run(task_type=task_type))
    assert "handler_missing" in _events()


def test_malformed_dataset_type_fails_honestly(env):
    assert_failed_clean(env, env.run(dataset_type="{not json"))
    assert "spec_build_failed" in _events()


def test_floor_artifact_plus_failed_mini_fine_tune_leaves_nothing(env, monkeypatch):
    def handler(spec, deadline):
        _write_adapter(Path(spec.output_dir), "FLOOR", 0)
        raise RuntimeError("crashed")

    monkeypatch.setattr(dispatch, "for_task", lambda task_type: handler)
    monkeypatch.setattr(TaskSpec, "cached_model_dir", property(lambda self: str(env.tmp / "no-such-base")))
    assert_failed_clean(env, env.run())


def test_main_module_exit_code_is_the_run_result(env, monkeypatch):
    monkeypatch.setattr(cli, "_run", lambda spec, deadline: 1)
    assert env.run() == 1
    assert "raise SystemExit(main())" in inspect.getsource(cli)


# ---- no code path can write an untrained artifact ---------------------------------------------------------------------

def test_save_path_refuses_the_floor_label(tmp_path):
    with pytest.raises(common.UntrainedArtifactError):
        common.save_adapter(object(), object(), str(tmp_path / "out"), artifact_truth=common.ARTIFACT_FLOOR)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()
    _write_adapter(tmp_path / "valid", None)
    assert common.write_artifact_truth(str(tmp_path / "valid"), common.ARTIFACT_FLOOR, optimizer_step=0,
                                       reason="x") is False
    assert not (tmp_path / "valid" / "forge_artifact_truth.json").exists()


def test_output_check_reads_the_truth_label(tmp_path):
    _write_adapter(tmp_path / "a", None)
    assert fallback.output_is_trained(str(tmp_path / "a")) is True          # trained saves without a label (DPO/GRPO)
    _write_adapter(tmp_path / "b", "FLOOR", 0)
    assert fallback.output_is_trained(str(tmp_path / "b")) is False
    _write_adapter(tmp_path / "c", "COMPLETE_BEST", 12)
    assert fallback.output_is_trained(str(tmp_path / "c")) is True
    _write_adapter(tmp_path / "d", "DERIVED_FIXED_FACTOR_SPACE_MIDPOINT", 0)   # Qwen3.5 fixed-average export
    (tmp_path / "d" / "forge_artifact_truth.json").write_text(json.dumps(
        {"schema": 1, "truth": "DERIVED_FIXED_FACTOR_SPACE_MIDPOINT", "optimizer_step": None, "reason": "x"}))
    assert fallback.output_is_trained(str(tmp_path / "d")) is True
    assert fallback.output_is_trained(str(tmp_path / "missing")) is False
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "adapter_config.json").write_text("{}")                 # config without weights
    assert fallback.output_is_trained(str(tmp_path / "e")) is False


def test_no_untrained_writer_is_left_in_the_training_path():
    root = Path(inspect.getfile(cli)).parent
    text = {p: p.read_text(encoding="utf-8") for p in root.rglob("*.py")}
    banned = ("emit_untrained_copy", "_emit_perturbed_copy", "_emit_plain_copy", "_emit_lora_adapter", "sidestep",
              "nextafter", "pretraining_floor", "full_ft_floor", "fallback_emitted", "floor before training")
    hits = [(str(p.relative_to(root)), word) for p, body in text.items() for word in banned if word in body]
    assert hits == []
    # the two TRL handlers save only after training, and never when the trainer took no step
    for module in ("dpo", "grpo"):
        body = text[root / "tasks" / f"{module}.py"]
        assert body.count("save_adapter(") == 1
        assert body.index("safe_train(trainer") < body.index("save_adapter(")
        assert "returned without an optimizer step" in body
    # sft_v2 and the legacy instruct handler never write the FLOOR label
    for module in ("sft_v2", "instruct"):
        body = text[root / "tasks" / f"{module}.py"]
        assert "write_artifact_truth(spec.output_dir, ARTIFACT_FLOOR" not in body
        assert not re.search(r"(?<![a-z_])artifact_truth=ARTIFACT_FLOOR", body)
    assert "zero_progress_oom_ladder_exhausted" in text[root / "tasks" / "instruct.py"]


# ---- F5: no packed file in the submitted tree --------------------------------------------------------------------------

def test_no_compressed_or_binary_file_is_tracked():
    import subprocess

    root = Path(inspect.getfile(cli)).parent.parent
    try:
        listed = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        listed = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    packed = (".gz", ".zip", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".pyc", ".pkl", ".bin", ".npy", ".pt", ".safetensors")
    assert [name for name in listed if name.endswith(packed)] == []
