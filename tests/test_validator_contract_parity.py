"""Golden contract tests against G.O.D's current text-task payload shapes.

These tests intentionally stay ML-free.  They exercise the serialization and
configuration boundary that must agree with the validator before a GPU trainer
is ever constructed.
"""

from __future__ import annotations

import json

import pytest

from forge.data import prompts, tokenize
from forge.data.schema import TaskSpec
from forge.tasks.rewards import EVAL_BETA_GRPO, materialise_rewards


def _build(task_type: str, payload: dict) -> TaskSpec:
    return TaskSpec.build(
        task_id="validator-task",
        task_type=task_type,
        model="anonymized/base",
        dataset="s3://validator-cache/train.json",
        dataset_type_json=json.dumps(payload),
        expected_repo_name="submission",
        baseline_stats_path=None,
    )


class _RecordingChatTokenizer:
    """Character tokenizer that records the literal template passed to HF."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, native_template: str | None = "NATIVE {{ messages }}") -> None:
        self.chat_template = native_template
        self.templates_seen: list[str] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        chat_template,
        **_template_kwargs,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        self.templates_seen.append(chat_template)
        # Rendering itself is intentionally simple; template selection is what
        # this boundary test verifies.
        return "".join(f"{m['role']}:{m['content']}|" for m in messages)

    def __call__(self, text, *, add_special_tokens=True, **_kwargs):
        ids = [ord(char) for char in text]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        return {"input_ids": ids}


_CHAT_ROWS = [[
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
]]


def test_chat_schema_distinguishes_omitted_default_from_explicit_null():
    assert _build("ChatTask", {}).chat.chat_template == "chatml"
    assert _build("ChatTask", {"chat_template": None}).chat.chat_template is None


def test_chatml_payload_overrides_model_native_template():
    tokenizer = _RecordingChatTokenizer(native_template="MODEL_NATIVE {{ messages }}")
    rows = tokenize.tokenize_chat(
        _CHAT_ROWS, tokenizer, 512, chat_template="chatml"
    )

    assert len(rows) == 1
    assert tokenizer.templates_seen
    assert all(
        rendered == tokenize._CHATML_TEMPLATE
        for rendered in tokenizer.templates_seen
    )


def test_chat_masking_supervises_content_not_role_header():
    tokenizer = _RecordingChatTokenizer()
    rows = tokenize.tokenize_chat(
        _CHAT_ROWS, tokenizer, 512, chat_template="chatml"
    )

    supervised = "".join(
        chr(token)
        for token, label in zip(rows[0]["input_ids"], rows[0]["labels"])
        if label != -100
    )
    assert supervised == "answer"


def test_tokenizer_default_uses_exact_model_template():
    native = "  {% for message in messages %}{{ message.content }}{% endfor %}  "
    tokenizer = _RecordingChatTokenizer(native_template=native)

    tokenize.tokenize_chat(
        _CHAT_ROWS, tokenizer, 512, chat_template="tokenizer_default"
    )

    assert tokenizer.templates_seen
    assert all(rendered == native for rendered in tokenizer.templates_seen)


def test_literal_jinja_is_rejected_like_current_scorer_contract():
    literal = "\n{% for message in messages %}{{message['content']}}{% endfor %}\n"
    tokenizer = _RecordingChatTokenizer()

    with pytest.raises(ValueError, match="literal Jinja"):
        tokenize.tokenize_chat(_CHAT_ROWS, tokenizer, 512, chat_template=literal)
    assert tokenizer.templates_seen == []


def test_named_axolotl_template_resolves_offline():
    tokenizer = _RecordingChatTokenizer()
    tokenize.tokenize_chat(_CHAT_ROWS, tokenizer, 512, chat_template="llama3")

    expected = tokenize._load_axolotl_template("llama3")
    assert tokenizer.templates_seen
    assert all(rendered == expected for rendered in tokenizer.templates_seen)


def test_explicit_null_resolves_through_tokenizer_default():
    native = "{% for message in messages %}NATIVE{% endfor %}"
    tokenizer = _RecordingChatTokenizer(native_template=native)
    tokenize.tokenize_chat(_CHAT_ROWS, tokenizer, 512, chat_template=None)

    assert all(rendered == native for rendered in tokenizer.templates_seen)


def test_complete_pinned_axolotl_named_registry_is_bundled():
    expected_names = {
        "alpaca", "aya", "chatml", "cohere", "command_a", "command_a_rag",
        "command_a_tool_use", "deepseek_v2", "deepseek_v3", "exaone", "exaone4",
        "falcon_h1", "gemma", "gemma3", "gemma3n", "gemma4", "gemma4_unified",
        "jamba", "llama3", "llama3_2_vision", "llama4", "llava", "metharme",
        "mistral_v1", "mistral_v2v3", "mistral_v3_tekken", "mistral_v7_tekken",
        "nemotron_h", "phi_3", "phi_35", "phi_4", "pixtral", "qwen2_vl",
        "qwen3", "qwen3_5", "qwen_25",
    }
    for name in expected_names:
        template = tokenize._load_axolotl_template(name)
        assert "{%" in template or "{{" in template


def test_unknown_named_chat_template_fails_before_processing_rows():
    tokenizer = _RecordingChatTokenizer()
    with pytest.raises(ValueError, match="is not bundled"):
        tokenize.tokenize_chat(
            _CHAT_ROWS, tokenizer, 512, chat_template="not_an_axolotl_template"
        )
    assert tokenizer.templates_seen == []


def test_tokenizer_default_requires_a_native_template():
    tokenizer = _RecordingChatTokenizer(native_template=None)
    with pytest.raises(ValueError, match="does not define"):
        tokenize.tokenize_chat(
            _CHAT_ROWS, tokenizer, 512, chat_template="tokenizer_default"
        )


def test_tokenizer_default_fallback_matches_axolotl_resolution():
    native = "{% for message in messages %}NATIVE{% endfor %}"
    tokenizer = _RecordingChatTokenizer(native_template=native)
    tokenize.tokenize_chat(
        _CHAT_ROWS,
        tokenizer,
        512,
        chat_template="tokenizer_default_fallback_chatml",
    )
    assert all(rendered == native for rendered in tokenizer.templates_seen)

    tokenizer = _RecordingChatTokenizer(native_template=None)
    tokenize.tokenize_chat(
        _CHAT_ROWS,
        tokenizer,
        512,
        chat_template="tokenizer_default_fallback_chatml",
    )
    assert all(
        rendered == tokenize._load_axolotl_template("chatml")
        for rendered in tokenizer.templates_seen
    )


def test_dpo_preserves_raw_scorer_fields_and_ignores_dormant_formats():
    cols = _build(
        "DpoTask",
        {
            "field_prompt": "question",
            "field_system": "policy",
            "field_chosen": "preferred",
            "field_rejected": "discarded",
            "prompt_format": "{system}|{prompt}|{chosen}|{rejected}",
            "chosen_format": "{prompt} => {chosen} (not {rejected}) [{system}]",
            "rejected_format": "{prompt} => {rejected} (not {chosen}) [{system}]",
        },
    ).dpo

    result = prompts.build_dpo_examples(
        [{
            "question": "2+2?",
            "policy": "Be exact",
            "preferred": "4",
            "discarded": "5",
        }],
        cols,
    )

    assert result == [{"prompt": "2+2?", "chosen": "4", "rejected": "5"}]


def test_eval_split_is_deterministic_bounded_and_order_preserving():
    rows = [{"prompt": str(index)} for index in range(1_000)]
    first = prompts.split_for_eval(rows, min_size=256, max_eval_rows=32)
    second = prompts.split_for_eval(rows, min_size=256, max_eval_rows=32)

    assert first == second
    train, evaluation = first
    assert len(train) == 968
    assert len(evaluation) == 32
    assert train == sorted(train, key=lambda row: int(row["prompt"]))
    assert evaluation == sorted(evaluation, key=lambda row: int(row["prompt"]))
    assert {row["prompt"] for row in train}.isdisjoint(
        {row["prompt"] for row in evaluation}
    )

    tiny = rows[:100]
    assert prompts.split_for_eval(
        tiny, min_size=256, max_eval_rows=32
    ) == (tiny, [])


def test_completion_tokenizer_supervises_and_chunks_entire_document():
    tokenizer = _RecordingChatTokenizer()
    chunks = tokenize.tokenize_completion(["abcdefgh"], tokenizer, max_len=4)

    # BOS + eight character tokens + EOS = ten supervised tokens in 4/4/2
    # contiguous chunks. The old empty-prompt path kept one chunk and masked BOS.
    assert [len(chunk["input_ids"]) for chunk in chunks] == [4, 4, 2]
    assert all(chunk["labels"] == chunk["input_ids"] for chunk in chunks)
    flattened = [token for chunk in chunks for token in chunk["input_ids"]]
    assert flattened[0] == tokenizer.bos_token_id
    assert flattened[-1] == tokenizer.eos_token_id


def test_completion_tokenizer_enforces_axolotl_64_chunk_document_cap():
    tokenizer = _RecordingChatTokenizer()
    chunks = tokenize.tokenize_completion(["x" * 1_000], tokenizer, max_len=4)

    assert len(chunks) == 64
    assert all(len(chunk["input_ids"]) == 4 for chunk in chunks)
    assert sum(len(chunk["input_ids"]) for chunk in chunks) == 256


def test_sft_retry_uses_full_cap_only_when_initial_cap_has_no_signal():
    tokenizer = _RecordingChatTokenizer()
    tokenizer.model_max_length = 12
    config = type("Config", (), {"max_position_embeddings": 12})()
    model = type("Model", (), {"config": config})()
    candidates = tokenize.sft_sequence_len_candidates(model, tokenizer, start=6)
    assert candidates == [6, 12]

    long_rows = [{"prompt_text": "abcdefgh", "completion_text": "xy"}]
    tokenized, selected = tokenize.first_nonempty_tokenization(
        candidates,
        lambda cap: tokenize.tokenize_instruct(long_rows, tokenizer, cap),
    )
    assert selected == 12
    assert len(tokenized) == 1
    assert any(label != -100 for label in tokenized[0]["labels"])


def test_sft_retry_keeps_initial_cap_when_any_mixed_row_survives():
    tokenizer = _RecordingChatTokenizer()
    rows = [
        {"prompt_text": "ab", "completion_text": "x"},
        {"prompt_text": "abcdefgh", "completion_text": "xy"},
    ]
    tokenized, selected = tokenize.first_nonempty_tokenization(
        [6, 12],
        lambda cap: tokenize.tokenize_instruct(rows, tokenizer, cap),
    )

    assert selected == 6
    assert len(tokenized) == 1


def test_instruct_overcap_row_is_dropped_not_partially_supervised():
    tokenizer = _RecordingChatTokenizer()

    rows = tokenize.tokenize_instruct(
        [{"prompt_text": "ab", "completion_text": "abcdefgh"}],
        tokenizer,
        max_len=6,
    )

    assert rows == []


def test_chat_overcap_row_is_dropped_not_truncated():
    tokenizer = _RecordingChatTokenizer()
    conversation = [[
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "a very long answer"},
    ]]

    rows = tokenize.tokenize_chat(
        conversation, tokenizer, max_len=4, chat_template="chatml"
    )

    assert rows == []


def test_chat_roles_preserve_tool_and_unknown_values_exactly():
    cols = _build(
        "ChatTask",
        {
            "chat_user_reference": "human",
            "chat_assistant_reference": "gpt",
        },
    ).chat
    conversations = prompts.build_chat_conversations(
        [{"conversations": [
            {"from": "human", "value": "question"},
            {"from": "tool", "value": "result"},
            {"from": "critic", "value": "review"},
            {"from": "gpt", "value": "answer"},
        ]}],
        cols,
    )

    assert [turn["role"] for turn in conversations[0]] == [
        "user", "tool", "critic", "assistant"
    ]


def test_grpo_normalizes_configured_extra_column_to_evaluator_key():
    spec = _build(
        "GrpoTask",
        {
            "field_prompt": "question",
            "extra_column": "grading_context",
            "reward_functions": [{
                "reward_func": (
                    "def exact(completions, extra_data, **kwargs):\n"
                    "    return [float(c == e) for c, e in zip(completions, extra_data)]"
                ),
                "reward_weight": 1.0,
                "func_hash": "validator-supplied",
                "is_generic": False,
            }],
        },
    ).grpo

    raw_json = '{"reference":"gold"}'
    result = prompts.build_grpo_examples(
        [{"question": "answer", "grading_context": raw_json}], spec
    )

    # G.O.D HEAD currently computes a decoded side-list but never writes it back
    # to the Dataset; TRL and reward functions therefore receive the raw string.
    assert result == [{"prompt": "answer", "extra_data": raw_json}]


def test_grpo_reward_wrapper_accepts_trl_kwargs_for_legacy_and_extra_data_funcs():
    sources = [
        "def legacy(completions):\n    return [len(c) for c in completions]",
        (
            "def contextual(completions, extra_data, **kwargs):\n"
            "    return [float(c == e) for c, e in zip(completions, extra_data)]"
        ),
    ]
    funcs, weights = materialise_rewards(sources, [0.25, 0.75])

    assert weights == [0.25, 0.75]
    assert funcs[0](
        completions=["one", "three"],
        prompts=["p1", "p2"],
        completion_ids=[[1], [2]],
        trainer_state=object(),
    ) == [3, 5]
    assert funcs[1](
        completions=["yes", "no"],
        prompts=["p1", "p2"],
        extra_data=["yes", "other"],
    ) == [1.0, 0.0]


def test_grpo_reward_function_and_weight_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        _build(
            "GrpoTask",
            {
                "field_prompt": "prompt",
                "reward_functions": [
                    "def first(completions, **kwargs): return [1.0]",
                    "def second(completions, **kwargs): return [0.0]",
                ],
                "reward_weights": [1.0],
            },
        )

    with pytest.raises(ValueError, match="length mismatch"):
        materialise_rewards(
            ["def only(completions, **kwargs): return [1.0]"], []
        )


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_grpo_rejects_invalid_reward_weights(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _build(
            "GrpoTask",
            {
                "field_prompt": "prompt",
                "reward_functions": [
                    "def score(completions, **kwargs): return [1.0]"
                ],
                "reward_weights": [weight],
            },
        )


def test_grpo_beta_matches_current_validator_constant():
    assert EVAL_BETA_GRPO == 0.5
