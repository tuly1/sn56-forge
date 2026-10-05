"""v9.2 crash fallback (owner decision 2026-10-05): a real minimal fine-tune, else an honest failure.

Every test drives the real CLI (forge.cli.main) on CPU with a tiny random Llama base and a real tokenizer, and forces a
failure at one stage. The contract asserted everywhere:

  * exit 0  => the output path holds a TRAINED artifact (optimizer steps >= 1, changed adapter weights, a model that
               differs from the base and passes the validator's vendored model_is_a_finetune);
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
