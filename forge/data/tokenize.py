"""Tokenize examples into input_ids/labels with correct completion masking.

Only completion tokens carry loss (prompt tokens are masked to -100), which is
what the evaluator scores. For instruct we mask by character offset so a token
straddling the prompt/completion boundary is handled precisely; for chat we
supervise every assistant turn.
"""

from __future__ import annotations

from typing import Any


def tokenize_instruct(
    examples: list[dict[str, str]], tokenizer: Any, max_len: int
) -> list[dict[str, list[int]]]:
    eos = tokenizer.eos_token_id
    fast = bool(getattr(tokenizer, "is_fast", False))
    out: list[dict[str, list[int]]] = []
    for ex in examples:
        # Completion-style examples carry an empty prompt: supervise everything
        # (only the BOS token, which has offset (0,0), gets masked).
        prefix = (ex["prompt_text"] + "\n") if ex["prompt_text"] else ""
        full = prefix + ex["completion_text"]
        if fast:
            input_ids, labels = _mask_by_offsets(tokenizer, prefix, full, max_len)
        else:
            input_ids, labels = _mask_by_length(tokenizer, prefix, full, max_len)
        if eos is not None:
            input_ids.append(eos)
            labels.append(eos)
        if all(l == -100 for l in labels):
            continue  # completion was truncated away — no signal
        out.append({"input_ids": input_ids, "labels": labels})
    return out


def _mask_by_offsets(
    tokenizer: Any, prefix: str, full: str, max_len: int
) -> tuple[list[int], list[int]]:
    completion_start = len(prefix)
    enc = tokenizer(
        full,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len - 1,  # leave room for a trailing EOS
    )
    input_ids = list(enc["input_ids"])
    labels = [
        tid if (end > completion_start) else -100
        for tid, (_start, end) in zip(input_ids, enc["offset_mapping"])
    ]
    return input_ids, labels


def _mask_by_length(
    tokenizer: Any, prefix: str, full: str, max_len: int
) -> tuple[list[int], list[int]]:
    """Slow-tokenizer path: mask by the token count of the prefix. Less precise
    at the boundary than offsets, but correct for the common case.
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"][: max_len - 1]
    boundary = min(len(prefix_ids), len(full_ids))
    labels = [-100] * boundary + list(full_ids[boundary:])
    return list(full_ids), labels


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
