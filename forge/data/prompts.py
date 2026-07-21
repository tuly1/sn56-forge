"""Assemble raw dataset rows into per-task-type training examples.

This is pure data shaping — no tokenizer, no torch — so it is cheap to test. The
guiding principle is to track how the validator's evaluator assembles the *same*
rows at scoring time (it renders instruct rows through an axolotl user-defined
template and reads chat/DPO columns directly), so a model trained on these
examples is optimised for the distribution it will be graded on.
"""

from __future__ import annotations

import random
from typing import Any

from forge.data.schema import ChatColumns, DpoColumns, GrpoSpec, InstructColumns


def split_for_eval(
    examples: list[dict[str, Any]],
    *,
    min_size: int,
    max_eval_rows: int,
    fraction: float = 0.05,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically reserve a bounded validation slice.

    The returned lists preserve source order, which keeps training reproducible
    while avoiding a prefix/suffix split that could correlate with source-file
    ordering.  Tiny datasets stay entirely in training.
    """
    if len(examples) < max(2, int(min_size)):
        return examples, []
    desired = max(1, int(round(len(examples) * fraction)))
    n_eval = min(max(1, int(max_eval_rows)), desired, len(examples) - 1)
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_indices = set(indices[:n_eval])
    train = [row for index, row in enumerate(examples) if index not in eval_indices]
    evaluation = [row for index, row in enumerate(examples) if index in eval_indices]
    return train, evaluation


def build_instruct_examples(
    rows: list[dict[str, Any]], cols: InstructColumns
) -> list[dict[str, str]]:
    """One {prompt_text, completion_text} per row; empty completions dropped.

    When the task is completion-style (no output column) the validator supervises
    the whole instruction text, so we emit it as the completion with an empty
    prompt (every token trained).
    """
    out: list[dict[str, str]] = []
    completion_style = cols.output is None
    for row in rows:
        # Text is kept verbatim: the evaluator scores the raw completion, so any
        # trimming or substitution on our side would train on a different string
        # than we're graded on.
        if completion_style:
            text = str(row.get(cols.instruction, "") or "")
            if text.strip():
                out.append({"prompt_text": "", "completion_text": text})
            continue
        completion = cols.render_completion(row)
        if not completion.strip():
            continue
        prompt = cols.render_prompt(row)
        if not prompt.strip():
            continue
        out.append({"prompt_text": prompt, "completion_text": completion})
    return out


def build_completion_documents(
    rows: list[dict[str, Any]], cols: InstructColumns
) -> list[str]:
    """Extract non-empty documents for Axolotl-style completion chunking."""
    if cols.output is not None:
        raise ValueError("completion documents require an instruct spec without output")
    documents: list[str] = []
    for row in rows:
        text = str(row.get(cols.instruction, "") or "")
        if text.strip():
            documents.append(text)
    return documents


def build_chat_conversations(
    rows: list[dict[str, Any]], cols: ChatColumns
) -> list[list[dict[str, str]]]:
    """Normalise each row's conversation into a list of {role, content} messages
    with roles in {system, user, assistant}. Conversations without at least one
    assistant turn carry no training signal and are dropped.
    """
    out: list[list[dict[str, str]]] = []
    for row in rows:
        turns = row.get(cols.conversation)
        if not isinstance(turns, list):
            continue
        messages: list[dict[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            raw_role = str(turn.get(cols.role_field, "")).strip()
            content = turn.get(cols.content_field)
            if content is None:
                continue
            messages.append({"role": _norm_role(raw_role, cols), "content": str(content)})
        if any(m["role"] == "assistant" for m in messages):
            out.append(messages)
    return out


def _norm_role(raw: str, cols: ChatColumns) -> str:
    # Axolotl maps only the exact source references supplied in the task. Other
    # roles (including system, tool, and custom names) remain verbatim and can
    # materially affect the selected chat template.
    if raw == cols.assistant_value:
        return "assistant"
    if raw == cols.user_value:
        return "user"
    return raw


def build_dpo_examples(
    rows: list[dict[str, Any]], cols: DpoColumns
) -> list[dict[str, str]]:
    """One raw {prompt, chosen, rejected} per row, matching the live evaluator.

    G.O.D currently renames these three columns and removes every other column;
    its format helpers are dormant and have no call sites. Applying payload
    format strings here would therefore train on text the evaluator never sees.
    Rows missing any required field are dropped.
    """
    out: list[dict[str, str]] = []
    for row in rows:
        prompt_raw = row.get(cols.prompt)
        chosen_raw = row.get(cols.chosen)
        rejected_raw = row.get(cols.rejected)
        if prompt_raw is None or chosen_raw is None or rejected_raw is None:
            continue
        # Drop degenerate pairs: identical chosen/rejected carry zero preference
        # signal (and would push the DPO loss toward its ln2 floor for nothing).
        if str(chosen_raw) == str(rejected_raw):
            continue
        prompt = str(prompt_raw)
        chosen = str(chosen_raw)
        rejected = str(rejected_raw)
        if not chosen.strip() or not rejected.strip():
            continue
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


def build_grpo_examples(
    rows: list[dict[str, Any]], spec: GrpoSpec
) -> list[dict[str, Any]]:
    """One {prompt, ...extra} per row; empty prompts dropped. GRPO learns from a
    prompt-only dataset plus reward functions, so we keep just the prompt column
    (and the optional extra column, which some reward functions read via kwargs).
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get(spec.prompt)
        if prompt is None or not str(prompt).strip():
            continue
        example: dict[str, Any] = {"prompt": str(prompt)}
        if spec.extra_column and spec.extra_column in row:
            # TRL forwards non-prompt dataset columns to reward functions by
            # their dataset key.  G.O.D's evaluator contract standardizes the
            # configured source column to `extra_data`, so training must do the
            # same even when the raw dataset calls it something else.
            example["extra_data"] = row[spec.extra_column]
        out.append(example)
    return out
