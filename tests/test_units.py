"""Unit tests for the ML-free logic: dataset-type parsing, prompt assembly,
dataset loading, CLI arg handling, and GRPO reward materialisation. These run
with no torch/transformers so they stay fast and cover the parsing layer where a
silent mistake would quietly tank eval scores.
"""

import json
import os

from forge import cli
from forge.data import loader, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.tasks.rewards import materialise_rewards


class _FakeTokenizer:
    """Char-level stand-in: one token per character, plus BOS/EOS."""

    bos_token_id = 1
    eos_token_id = 2
    is_fast = True

    def __call__(self, text, add_special_tokens=True, **_kw):
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


# --- schema ----------------------------------------------------------------

def test_instruct_render_with_and_without_input():
    spec = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="m", dataset=None,
        dataset_type_json=json.dumps(
            {"field_instruction": "instr", "field_input": "inp", "field_output": "out",
             "system_prompt": "SYS"}
        ),
        expected_repo_name="r", baseline_stats_path=None,
    )
    cols = spec.instruct
    with_input = cols.render_prompt({"instr": "Add", "inp": "2+2", "out": "4"})
    assert "Add 2+2" in with_input and with_input.startswith("SYS")
    without_input = cols.render_prompt({"instr": "Hi", "inp": "", "out": "x"})
    assert "Hi" in without_input and "None" not in without_input
    assert cols.render_completion({"out": "4"}) == "4"


def test_chat_defaults_do_not_crash_on_partial_payload():
    # A payload missing the chat_* keys must fall back to the validator defaults,
    # not raise (the old code treated the alias as a key and crashed).
    spec = TaskSpec.build(
        task_id="t", task_type="ChatTask", model="m", dataset=None,
        dataset_type_json="{}", expected_repo_name="r", baseline_stats_path=None,
    )
    assert spec.chat.conversation == "conversations"
    assert spec.chat.role_field == "from"
    assert spec.chat.content_field == "value"
    assert spec.chat.user_value == "user"
    assert spec.chat.assistant_value == "assistant"


def test_grpo_accepts_rewardfunction_dicts():
    dt = json.dumps({
        "field_prompt": "p",
        "reward_functions": [
            {"reward_func": "def a(completions, **k): return [0.0]", "reward_weight": 0.7},
            {"reward_func": "def b(completions, **k): return [1.0]", "reward_weight": 0.3},
        ],
    })
    spec = TaskSpec.build(
        task_id="t", task_type="GrpoTask", model="m", dataset=None,
        dataset_type_json=dt, expected_repo_name="r", baseline_stats_path=None,
    )
    assert spec.grpo.reward_weights == [0.7, 0.3]
    assert len(spec.grpo.reward_functions) == 2


def test_kl_fields_and_file_format_carry_through():
    spec = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="m", dataset=None,
        dataset_type_json=json.dumps({"field_instruction": "i", "field_output": "o"}),
        expected_repo_name="r", baseline_stats_path=None,
        file_format="s3", use_kl=True, kl_coef=0.5,
    )
    assert spec.use_kl is True and spec.kl_coef == 0.5
    assert spec.file_format == "s3"
    assert spec.cached_dataset_path == "/cache/datasets/t_train_data.json"
    assert spec.cached_model_dir == "/cache/models/m"


def test_cached_model_dir_sanitises_slashes():
    spec = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="org/Name", dataset=None,
        dataset_type_json=json.dumps({"field_instruction": "i", "field_output": "o"}),
        expected_repo_name="r", baseline_stats_path=None,
    )
    assert spec.cached_model_dir == "/cache/models/org--Name"


# --- prompts ---------------------------------------------------------------

def test_build_instruct_examples_drops_empty_completion():
    cols = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="m", dataset=None,
        dataset_type_json=json.dumps({"field_instruction": "q", "field_output": "a"}),
        expected_repo_name="r", baseline_stats_path=None,
    ).instruct
    rows = [{"q": "hi", "a": "yo"}, {"q": "x", "a": ""}, {"q": "", "a": "z"}]
    out = prompts.build_instruct_examples(rows, cols)
    assert len(out) == 1
    assert out[0]["completion_text"] == "yo"


def test_build_chat_conversations_normalises_roles_and_requires_assistant():
    cols = TaskSpec.build(
        task_id="t", task_type="ChatTask", model="m", dataset=None,
        dataset_type_json="{}", expected_repo_name="r", baseline_stats_path=None,
    ).chat
    rows = [
        {"conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi there"},
        ]},
        {"conversations": [{"from": "human", "value": "no answer"}]},  # dropped
    ]
    out = prompts.build_chat_conversations(rows, cols)
    assert len(out) == 1
    assert out[0][0]["role"] == "user"
    assert out[0][1]["role"] == "assistant"


def test_build_dpo_applies_format_templates():
    cols = TaskSpec.build(
        task_id="t", task_type="DpoTask", model="m", dataset=None,
        dataset_type_json=json.dumps({
            "field_prompt": "p", "field_chosen": "c", "field_rejected": "r",
            "prompt_format": "Q: {prompt}",
        }),
        expected_repo_name="r", baseline_stats_path=None,
    ).dpo
    out = prompts.build_dpo_examples([{"p": "why", "c": "good", "r": "bad"}], cols)
    assert out == [{"prompt": "Q: why", "chosen": "good", "rejected": "bad"}]


def test_build_grpo_keeps_prompt_and_extra_column():
    spec = TaskSpec.build(
        task_id="t", task_type="GrpoTask", model="m", dataset=None,
        dataset_type_json=json.dumps({
            "field_prompt": "p", "extra_column": "meta",
            "reward_functions": ["def r(completions, **k): return [0.0]"],
        }),
        expected_repo_name="r", baseline_stats_path=None,
    ).grpo
    out = prompts.build_grpo_examples([{"p": "solve", "meta": 42}, {"p": ""}], spec)
    assert out == [{"prompt": "solve", "meta": 42}]


# --- loader ----------------------------------------------------------------

def test_loader_reads_json_array(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    rows = loader.load_rows(str(p), dataset_arg=None, file_format="s3")
    assert rows == [{"a": 1}, {"a": 2}]


def test_loader_reads_jsonlines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"a": 1}\n{"a": 2}\n')
    rows = loader.load_rows(str(p), dataset_arg=None, file_format="s3")
    assert rows == [{"a": 1}, {"a": 2}]


def test_loader_reads_wrapped_object(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"rows": [{"a": 1}]}))
    rows = loader.load_rows(str(p), dataset_arg=None, file_format="s3")
    assert rows == [{"a": 1}]


def test_loader_missing_raises(tmp_path):
    try:
        loader.load_rows(str(tmp_path / "nope.json"), dataset_arg=None, file_format="s3")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# --- cli -------------------------------------------------------------------

def test_cli_parse_tolerates_unknown_flags_and_task_types():
    args = cli._parse([
        "--task-id", "t", "--model", "m", "--dataset", "s3://x",
        "--task-type", "SomeFutureTask", "--expected-repo-name", "r",
        "--hours-to-complete", "1.5", "--brand-new-flag", "v",
    ])
    assert args.task_type == "SomeFutureTask"  # no argparse choices rejection
    assert args.hours_to_complete == 1.5
    assert args.file_format == "s3"


def test_kl_from_env(monkeypatch):
    monkeypatch.delenv("USE_KL", raising=False)
    assert cli._kl_from_env() == (False, 0.0)
    monkeypatch.setenv("USE_KL", "1")
    monkeypatch.setenv("KL_COEF", "0.35")
    assert cli._kl_from_env() == (True, 0.35)
    monkeypatch.setenv("KL_COEF", "not-a-number")
    assert cli._kl_from_env() == (True, 0.0)


# --- strategy (full-FT vs LoRA decision) ------------------------------------

def test_decide_full_finetune_small_nonkl_single_gpu():
    from forge.model import decide_full_finetune
    # 1.7B non-KL on one 80GB card -> full fine-tune.
    assert decide_full_finetune(use_kl=False, params_b=1.7, n_gpus=1, per_gpu_gb=80.0) is True
    # KL task -> always LoRA (KL trainer needs the adapter).
    assert decide_full_finetune(use_kl=True, params_b=1.7, n_gpus=1, per_gpu_gb=80.0) is False
    # Multi-GPU (large model) -> LoRA (our validated sharding path).
    assert decide_full_finetune(use_kl=False, params_b=1.7, n_gpus=4, per_gpu_gb=80.0) is False
    # Too big to full-FT on one card -> LoRA.
    assert decide_full_finetune(use_kl=False, params_b=9.0, n_gpus=1, per_gpu_gb=80.0) is False
    # CPU / no GPU info -> LoRA.
    assert decide_full_finetune(use_kl=False, params_b=1.7, n_gpus=0, per_gpu_gb=0.0) is False


def test_full_ft_plan_shape():
    from forge.tuning.plan import make_sft_plan
    p = make_sft_plan(use_kl=False, strategy="full", params_b=1.7)
    assert p.strategy == "full"
    assert p.lora_r == 0  # no adapter
    assert p.num_epochs == 3  # use more of the budget on small data
    assert 5e-5 <= p.learning_rate <= 1.1e-4  # size-based full-FT LR
    # LoRA path unchanged and still the default.
    lora = make_sft_plan(use_kl=False)
    assert lora.strategy == "lora" and lora.lora_r == 32


def test_full_ft_lr_falls_with_size():
    from forge.tuning.plan import _full_ft_lr
    assert _full_ft_lr(0.5) >= _full_ft_lr(3.0) >= _full_ft_lr(7.0)


# --- telemetry ---------------------------------------------------------------

def test_telemetry_roundtrip_and_never_raises(tmp_path):
    from forge import telemetry

    telemetry.init(task_id="t1", task_type="InstructTextTask", use_kl=True)
    telemetry.event("model_loaded", rows=123)
    telemetry.train_point(10, 1.5, 2e-4)
    telemetry.eval_point(50, 1.23)
    telemetry.sample("kl_per_token", 0.004)

    telemetry.write_into(str(tmp_path))
    data = json.loads((tmp_path / "forge_run.json").read_text())
    assert data["meta"]["task_id"] == "t1"
    assert any(e["name"] == "model_loaded" for e in data["events"])
    assert data["eval_curve"][0][2] == 1.23
    assert "kl_per_token" in data["samples"]

    # Writing into a nonexistent dir must be a silent no-op, never a crash.
    telemetry.write_into(str(tmp_path / "does" / "not" / "exist"))


def test_save_adapter_carries_flight_recorder_atomically(tmp_path):
    # The flight recorder must ride into the swapped output dir together with the
    # adapter — present the instant the dir goes live, so a mid-swap kill can't
    # leave weights without the log.
    import os as _os

    from forge import telemetry
    from forge.tasks.common import save_adapter

    telemetry.init(task_id="carry-test")
    telemetry.event("model_loaded")

    class _FakeModel:
        def save_pretrained(self, d, **_k):
            with open(_os.path.join(d, "adapter_model.safetensors"), "w") as f:
                f.write("weights")
            with open(_os.path.join(d, "adapter_config.json"), "w") as f:
                f.write("{}")

    class _FakeTok:
        def save_pretrained(self, d, **_k):
            with open(_os.path.join(d, "tokenizer.json"), "w") as f:
                f.write("{}")

    out = str(tmp_path / "checkpoints" / "task" / "model")
    _os.makedirs(out)
    # Pre-existing dir (simulates a prior mirror-save) so the rename-aside path runs.
    with open(_os.path.join(out, "stale"), "w") as f:
        f.write("x")

    save_adapter(_FakeModel(), _FakeTok(), out)

    # Adapter and log co-present; stale contents gone; no .tmp/.old leaked.
    assert _os.path.isfile(_os.path.join(out, "adapter_model.safetensors"))
    assert _os.path.isfile(_os.path.join(out, "forge_run.json"))
    assert not _os.path.exists(_os.path.join(out, "stale"))
    assert not _os.path.exists(out + ".tmp") and not _os.path.exists(out + ".old")
    data = json.loads(open(_os.path.join(out, "forge_run.json")).read())
    assert data["meta"]["task_id"] == "carry-test"


def test_telemetry_curve_thinning():
    from forge import telemetry

    for i in range(1000):
        telemetry.train_point(i, 1.0, 1e-4)
    assert len(telemetry._data["train_curve"]) <= 600  # bounded, never unbounded


# --- rewards ---------------------------------------------------------------

def test_completion_style_instruct_supervises_whole_instruction():
    # A valid instruct task can omit field_output (completion-style); the whole
    # instruction text is then the training signal.
    spec = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="m", dataset=None,
        dataset_type_json=json.dumps({"field_instruction": "text"}),
        expected_repo_name="r", baseline_stats_path=None,
    )
    assert spec.instruct.output is None
    out = prompts.build_instruct_examples([{"text": "a story"}], spec.instruct)
    assert out == [{"prompt_text": "", "completion_text": "a story"}]


def test_instruct_system_format_applied():
    cols = TaskSpec.build(
        task_id="t", task_type="InstructTextTask", model="m", dataset=None,
        dataset_type_json=json.dumps({
            "field_instruction": "q", "field_output": "a",
            "system_prompt": "BeHelpful", "system_format": "SYSTEM: {system}",
        }),
        expected_repo_name="r", baseline_stats_path=None,
    ).instruct
    p = cols.render_prompt({"q": "hi", "a": "x"})
    assert p.startswith("SYSTEM: BeHelpful")
    assert "hi" in p


def test_main_never_raises_on_malformed_dataset_type():
    # The never-exit-nonzero guarantee: a broken --dataset-type must not raise
    # out of main(); it degrades (here the fallback no-ops because no /cache).
    rc = cli.main([
        "--task-id", "t", "--model", "no/such-model", "--dataset", "s3://x",
        "--dataset-type", "{not valid json",
        "--task-type", "InstructTextTask", "--expected-repo-name", "r",
        "--hours-to-complete", "0.01",
    ])
    assert rc == 0


# --- tokenization ----------------------------------------------------------

def test_tokenize_instruct_injects_no_separator_and_masks_prompt():
    tok = _FakeTokenizer()
    out = tokenize.tokenize_instruct(
        [{"prompt_text": "ab", "completion_text": "cd"}], tok, max_len=64
    )
    ids, labels = out[0]["input_ids"], out[0]["labels"]
    # BOS + 'a','b' + 'c','d' + EOS — nothing inserted at the boundary.
    assert ids == [1, ord("a"), ord("b"), ord("c"), ord("d"), 2]
    # Only the completion (and its EOS) carries loss.
    assert labels == [-100, -100, -100, ord("c"), ord("d"), 2]


def test_tokenize_instruct_completion_style_supervises_all_but_bos():
    tok = _FakeTokenizer()
    out = tokenize.tokenize_instruct(
        [{"prompt_text": "", "completion_text": "xy"}], tok, max_len=64
    )
    assert out[0]["input_ids"] == [1, ord("x"), ord("y"), 2]
    assert out[0]["labels"] == [-100, ord("x"), ord("y"), 2]


def test_tokenize_instruct_truncation_sacrifices_prompt_not_completion():
    tok = _FakeTokenizer()
    out = tokenize.tokenize_instruct(
        [{"prompt_text": "abcdefgh", "completion_text": "xy"}], tok, max_len=6
    )
    ids, labels = out[0]["input_ids"], out[0]["labels"]
    assert len(ids) == 6
    assert ids[0] == 1  # BOS preserved
    # The completion and its EOS survive intact; the prompt lost its left side.
    assert ids[-3:] == [ord("x"), ord("y"), 2]
    assert labels[-3:] == [ord("x"), ord("y"), 2]
    assert labels[:3] == [-100, -100, -100]


def test_tokenize_instruct_drops_example_with_no_completion_signal():
    tok = _FakeTokenizer()
    # max_len leaves no room for any completion token after a 1-token prompt.
    out = tokenize.tokenize_instruct(
        [{"prompt_text": "abcd", "completion_text": "xyz"}], tok, max_len=1
    )
    assert out == []


def test_materialise_rewards_compiles_and_skips_bad():
    good = "def scorer(completions, **kwargs):\n    return [len(c) for c in completions]"
    bad = "def broken(:"  # syntax error
    notdef = "x = 5"
    funcs, weights = materialise_rewards([good, bad, notdef], [0.5, 0.3, 0.2])
    assert len(funcs) == 1 and weights == [0.5]
    assert funcs[0](["ab", "cde"]) == [2, 3]
    assert funcs[0].__name__ == "scorer"
