from types import SimpleNamespace

from forge.tasks.trl_compat import (
    PromptCappedPreferenceCollator,
    generation_kwargs_for_model,
    prompt_capped_grpo_trainer,
)


def test_dpo_v15_collator_caps_prompt_without_mutating_source():
    captured = []

    def delegate(features):
        captured.extend(features)
        return features

    original = {
        "prompt_ids": list(range(10)),
        "chosen_ids": [100, 101],
        "rejected_ids": [200],
    }
    result = PromptCappedPreferenceCollator(delegate, 4)([original])

    assert result[0]["prompt_ids"] == [6, 7, 8, 9]
    assert result[0]["chosen_ids"] == [100, 101]
    assert result[0]["rejected_ids"] == [200]
    assert original["prompt_ids"] == list(range(10))


def test_dpo_prompt_cap_leaves_room_for_completion_at_small_context():
    seq_len = 4
    prompt_cap = min(512, max(1, seq_len - 1))
    result = PromptCappedPreferenceCollator(lambda rows: rows, prompt_cap)(
        [{"prompt_ids": list(range(20)), "chosen_ids": [90], "rejected_ids": [91]}]
    )

    assert result[0]["prompt_ids"] == [17, 18, 19]
    assert len(result[0]["prompt_ids"]) < seq_len


def test_grpo_v15_override_left_caps_super_tokenization_result():
    class Base:
        def _tokenize_prompts(self, prompts):
            return [list(range(7)), [10, 11]], [None, None], {"field": "kept"}

    trainer = prompt_capped_grpo_trainer(Base, 3)()

    prompt_ids, images, fields = trainer._tokenize_prompts(["a", "b"])

    assert prompt_ids == [[4, 5, 6], [10, 11]]
    assert images == [None, None]
    assert fields == {"field": "kept"}


def test_quasar_grpo_disables_generation_cache_only_for_quasar():
    quasar = SimpleNamespace(config=SimpleNamespace(model_type="quasar_text"))
    native = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))

    assert generation_kwargs_for_model(quasar) == {"use_cache": False}
    assert generation_kwargs_for_model(native) is None
