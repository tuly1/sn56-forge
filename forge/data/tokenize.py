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

import re
from functools import lru_cache
from importlib.resources import files
from typing import Any, Callable


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

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        # Axolotl's evaluator uses excess_length_strategy=drop. Partial rows are
        # not scored and must not count as a surviving row in the retry ladder.
        if len(input_ids) > max_len:
            continue
        if not any(label != -100 for label in labels):
            continue  # nothing left to supervise

        out.append(
            {
                "input_ids": input_ids,
                "labels": labels,
            }
        )
    return out


def tokenize_completion(
    documents: list[str], tokenizer: Any, max_len: int
) -> list[dict[str, list[int]]]:
    """Mirror Axolotl's completion strategy without discarding long documents.

    Each document is tokenized once up to ``sequence_len * 64`` and then split
    into contiguous sequence-length examples. Every token, including a leading
    BOS, is supervised. An EOS is appended only when the tokenizer left room,
    matching Axolotl's July-1 completion strategy.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    document_cap = max_len * 64
    eos = tokenizer.eos_token_id
    out: list[dict[str, list[int]]] = []
    for text in documents:
        if not text:
            continue
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=document_cap,
            padding=False,
            return_tensors=None,
        )
        input_ids = list(encoded.get("input_ids", []))[:document_cap]
        if not input_ids:
            continue
        if eos is not None and input_ids[-1] != eos and len(input_ids) < document_cap:
            input_ids.append(eos)
        for start in range(0, len(input_ids), max_len):
            chunk = input_ids[start : start + max_len]
            if chunk:
                out.append({"input_ids": chunk, "labels": list(chunk)})
    return out


def sft_sequence_len_candidates(model: Any, tokenizer: Any, start: int) -> list[int]:
    """Mirror G.O.D's evaluator retry ladder up to model/tokenizer limits."""
    model_cap = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(model_cap, int) or model_cap <= 0:
        model_cap = 131_072
    tokenizer_cap = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_cap, int) and 0 < tokenizer_cap < 1_000_000:
        cap = min(model_cap, tokenizer_cap)
    else:
        cap = model_cap
    start = max(1, min(int(start), cap))
    candidates: list[int] = []
    current = start
    while True:
        candidates.append(current)
        if current >= cap:
            break
        current = min(current * 2, cap)
    return candidates


def first_nonempty_tokenization(
    candidates: list[int], tokenize_at: Callable[[int], list[dict[str, list[int]]]]
) -> tuple[list[dict[str, list[int]]], int]:
    """Return the first evaluator candidate that contains supervised rows."""
    if not candidates:
        raise ValueError("sequence-length candidates must not be empty")
    last: list[dict[str, list[int]]] = []
    for candidate in candidates:
        last = tokenize_at(candidate)
        if last:
            return last, candidate
    return last, candidates[-1]


# The evaluator config has no global chat-template override. Axolotl therefore
# resolves an explicit per-dataset null to its `tokenizer_default` fallback. An
# omitted payload field is different: Pydantic supplies `chatml` in schema.py.
_AXOLOTL_TEMPLATE_COMMIT = "0bda5a13e4d52ceec58104f44fabb7bd314f9c02"
_NAMED_TEMPLATE_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TOKENIZER_FALLBACK_PREFIX = "tokenizer_default_fallback_"


@lru_cache(maxsize=None)
def _load_axolotl_template(name: str) -> str:
    """Load a template vendored from the evaluator image's Axolotl revision."""
    if not _NAMED_TEMPLATE_RE.fullmatch(name):
        raise ValueError(f"invalid Axolotl chat-template name {name!r}")
    resource = files("forge.data").joinpath("chat_templates", f"{name}.jinja")
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            f"Axolotl template {name!r} is not bundled at pinned commit "
            f"{_AXOLOTL_TEMPLATE_COMMIT}"
        ) from exc


# Compatibility name retained for tests/downstream imports; unlike the old
# handwritten approximation, this is the evaluator image's exact template.
_CHATML_TEMPLATE = _load_axolotl_template("chatml")


def tokenize_chat(
    conversations: list[list[dict[str, str]]],
    tokenizer: Any,
    max_len: int,
    *,
    chat_template: str | None = "chatml",
) -> list[dict[str, list[int]]]:
    """Tokenize chat rows with the template selected by the task payload.

    Named templates resolve from the complete registry vendored from the exact
    Axolotl revision used by the current evaluator image.  This keeps offline
    training dependency-free without pretending that a model-native template is
    equivalent to an explicitly requested Axolotl template.
    """
    resolved_template = _resolve_chat_template(chat_template, tokenizer)
    out: list[dict[str, list[int]]] = []
    for messages in conversations:
        ids, labels = _mask_assistant_turns(
            messages, tokenizer, max_len, chat_template=resolved_template
        )
        if ids and not all(l == -100 for l in labels):
            out.append({"input_ids": ids, "labels": labels})
    return out


def _resolve_chat_template(requested: str | None, tokenizer: Any) -> str:
    """Resolve the validator's chat-template contract to literal Jinja.

    ``None`` means no per-dataset override and therefore resolves through
    Axolotl's ``tokenizer_default`` fallback. G.O.D's payload has no separate
    ``chat_template_jinja`` field, so literal Jinja is rejected: the evaluator
    would treat it as a template *name* and fail too.
    """
    if requested is None:
        requested = "tokenizer_default"
    if not isinstance(requested, str):
        raise TypeError("chat_template must be a string or null")

    if "{%" in requested or "{{" in requested or "{#" in requested:
        raise ValueError(
            "literal Jinja chat_template is not supported by the current G.O.D "
            "payload/scorer contract; use a bundled Axolotl template name"
        )

    named = requested
    if named == "tokenizer_default":
        native = getattr(tokenizer, "chat_template", None)
        if not isinstance(native, str) or not native.strip():
            raise ValueError(
                "chat_template='tokenizer_default' requested, but the tokenizer "
                "does not define a native chat_template"
            )
        return native
    if named.startswith(_TOKENIZER_FALLBACK_PREFIX):
        native = getattr(tokenizer, "chat_template", None)
        if isinstance(native, str) and native.strip():
            return native
        named = named[len(_TOKENIZER_FALLBACK_PREFIX) :]
    if not named:
        raise ValueError("chat_template cannot be empty")
    if named == "jinja":
        raise ValueError(
            "chat_template='jinja' requires a separate Jinja value in Axolotl; "
            "pass the literal Jinja template in the task payload instead"
        )
    return _load_axolotl_template(named)


def _mask_assistant_turns(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_len: int,
    *,
    chat_template: str,
) -> tuple[list[int], list[int]]:
    full_text = _render_chat(
        tokenizer, messages, chat_template, add_generation_prompt=False
    )
    input_ids = list(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if len(input_ids) > max_len:
        return [], []
    labels = [-100] * len(input_ids)
    real_last_index = len(messages) - 1

    for turn_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        dummy = {"role": message["role"], "content": "[[dummy_message]]"}
        render_kwargs = {
            "chat_template": chat_template,
            "add_generation_prompt": False,
            "real_last_index": real_last_index,
        }
        dummy_text = _render_chat(
            tokenizer,
            messages[:turn_index] + [dummy],
            **render_kwargs,
        )
        turn_text = _render_chat(
            tokenizer,
            messages[: turn_index + 1],
            **render_kwargs,
        )
        dummy_ids = list(tokenizer(dummy_text, add_special_tokens=False)["input_ids"])
        turn_ids = list(tokenizer(turn_text, add_special_tokens=False)["input_ids"])
        boundaries = _token_diff_boundaries(dummy_ids, turn_ids)
        if boundaries is None:
            continue
        start, end = boundaries
        end = min(end, len(input_ids))
        if start >= end:
            continue
        labels[start:end] = input_ids[start:end]

        # Axolotl defaults roles_to_train=[assistant], train_on_eos="turn".
        # Label the first nearby tokenizer EOS/EOT after the assistant content.
        eos = tokenizer.eos_token_id
        if eos is not None:
            for index in range(end, min(end + 4, len(input_ids))):
                if input_ids[index] == eos:
                    labels[index] = input_ids[index]
                    break

    return input_ids, labels


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template: str,
    *,
    add_generation_prompt: bool,
    real_last_index: int | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "chat_template": chat_template,
    }
    # Axolotl only forwards truthy real_last_index values.
    if real_last_index:
        kwargs["real_last_index"] = real_last_index
    return tokenizer.apply_chat_template(messages, **kwargs)


def _token_diff_boundaries(
    dummy_ids: list[int], full_ids: list[int]
) -> tuple[int, int] | None:
    """Find content boundaries using Axolotl's dummy-message diff algorithm."""
    if not dummy_ids or not full_ids:
        return None
    common = min(len(dummy_ids), len(full_ids))
    start = next(
        (index for index in range(common) if dummy_ids[index] != full_ids[index]),
        None,
    )
    if start is None:
        return None
    end = None
    for offset in range(common):
        dummy_pos = len(dummy_ids) - 1 - offset
        full_pos = len(full_ids) - 1 - offset
        if dummy_ids[dummy_pos] != full_ids[full_pos]:
            end = full_pos + 1
            break
    if end is None or end <= start:
        return None
    return start, end


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
