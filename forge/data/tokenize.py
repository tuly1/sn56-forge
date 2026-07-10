"""Tokenize examples into input_ids/labels with correct completion masking.

Only completion tokens carry loss (prompt tokens are masked to -100), which is
what the evaluator scores.

For instruct we tokenize the prompt and the completion *separately* and
concatenate, injecting no separator of our own. That mirrors how the evaluator
assembles the same row (its prompter tokenizes the rendered prompt, then the
output with an EOS appended), so the boundary token we train on is the boundary
token we are scored on. Any separator belongs to the validator's own `format` /
`system_format` templates, not to us.

For chat we supervise every assistant turn.
"""

from __future__ import annotations

from typing import Any


def tokenize_instruct(
    examples: list[dict[str, str]], tokenizer: Any, max_len: int
) -> list[dict[str, list[int]]]:
    eos = tokenizer.eos_token_id
    out: list[dict[str, list[int]]] = []
    for ex in examples:
        # Prompt keeps the leading BOS; the completion must not add one. A
        # completion-style example has an empty prompt, so only BOS is masked and
        # every real token is supervised.
        prompt_ids = list(
            tokenizer(ex["prompt_text"], add_special_tokens=True)["input_ids"]
        )
        completion_ids = list(
            tokenizer(ex["completion_text"], add_special_tokens=False)["input_ids"]
        )
        if eos is not None:
            completion_ids.append(eos)

        prompt_ids, completion_ids = _fit(prompt_ids, completion_ids, max_len, tokenizer)
        if not completion_ids:
            continue  # nothing left to supervise

        out.append(
            {
                "input_ids": prompt_ids + completion_ids,
                "labels": [-100] * len(prompt_ids) + completion_ids,
            }
        )
    return out


def _fit(
    prompt_ids: list[int], completion_ids: list[int], max_len: int, tokenizer: Any
) -> tuple[list[int], list[int]]:
    """Trim to max_len, sacrificing prompt context before completion signal.

    Truncating from the right (the naive default) would silently delete the very
    tokens the loss is computed on, so we drop from the left of the prompt first
    and keep its BOS.
    """
    overflow = len(prompt_ids) + len(completion_ids) - max_len
    if overflow <= 0:
        return prompt_ids, completion_ids

    if len(prompt_ids) - overflow >= 1:
        bos = tokenizer.bos_token_id
        if bos is not None and prompt_ids and prompt_ids[0] == bos:
            # Drop from just after BOS so the sequence still opens correctly.
            return [bos] + prompt_ids[1 + overflow :], completion_ids
        return prompt_ids[overflow:], completion_ids

    # Prompt alone can't absorb it: keep a minimal prompt and clip the completion.
    prompt_ids = prompt_ids[:1] if prompt_ids else []
    room = max_len - len(prompt_ids)
    return prompt_ids, completion_ids[: max(0, room)]


# A standard ChatML template, used when the base tokenizer ships none — matching
# the validator's default `chat_template="chatml"` so chat tasks train instead of
# falling through to the floor.
_CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def tokenize_chat(
    conversations: list[list[dict[str, str]]], tokenizer: Any, max_len: int
) -> list[dict[str, list[int]]]:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _CHATML_TEMPLATE
    out: list[dict[str, list[int]]] = []
    for messages in conversations:
        try:
            ids, labels = _mask_assistant_turns(messages, tokenizer, max_len)
        except Exception:
            continue
        if ids and not all(l == -100 for l in labels):
            out.append({"input_ids": ids, "labels": labels})
    return out


def _mask_assistant_turns(
    messages: list[dict[str, str]], tokenizer: Any, max_len: int
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    prev_len = 0
    for i, msg in enumerate(messages):
        text_upto = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        ids_upto = tokenizer(text_upto, add_special_tokens=False)["input_ids"]
        new_ids = ids_upto[prev_len:]
        prev_len = len(ids_upto)
        input_ids = ids_upto
        if msg["role"] == "assistant":
            labels.extend(new_ids)
        else:
            labels.extend([-100] * len(new_ids))
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return input_ids, labels


class PadCollator:
    """Right-pad a batch of {input_ids, labels} to the longest member. Labels are
    padded with -100 (ignored by the loss); attention_mask marks real tokens.
    """

    def __init__(self, pad_token_id: int | None) -> None:
        # Fall back to id 0 if the tokenizer has no pad token; padding positions
        # are masked out of both attention and loss, so the exact id is inert.
        self._pad = pad_token_id if pad_token_id is not None else 0

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self._pad] * pad_n)
            labels.append(lab + [-100] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
