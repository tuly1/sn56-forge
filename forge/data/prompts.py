"""Assemble raw dataset rows into per-task-type training examples.

This is pure data shaping — no tokenizer, no torch — so it is cheap to test. The
guiding principle is to track how the validator's evaluator assembles the *same*
rows at scoring time (it renders instruct rows through an axolotl user-defined
template and reads chat/DPO columns directly), so a model trained on these
examples is optimised for the distribution it will be graded on.
"""

from __future__ import annotations

from typing import Any

from forge.data.schema import ChatColumns, DpoColumns, GrpoSpec, InstructColumns

# Literal junk that leaks into dataset text and would otherwise be tokenised as
# real content (special-token strings written verbatim into fields).
_JUNK = ("[PAD]", "<pad>", "<PAD>")


def _clean(text: str) -> str:
    for j in _JUNK:
        if j in text:
            text = text.replace(j, "")
    return text


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
        # Emptiness is checked on a stripped copy, but the text itself is kept
        # verbatim (only junk-token literals removed): trimming would move the
        # boundary the evaluator scores.
        if completion_style:
            text = _clean(str(row.get(cols.instruction, "") or ""))
            if text.strip():
                out.append({"prompt_text": "", "completion_text": text})
            continue
        completion = _clean(cols.render_completion(row))
        if not completion.strip():
            continue
        prompt = _clean(cols.render_prompt(row))
        if not prompt.strip():
            continue
        out.append({"prompt_text": prompt, "completion_text": completion})
    return out


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
    low = raw.lower()
    if raw == cols.assistant_value or low in ("assistant", "gpt", "bot", "model"):
        return "assistant"
    if raw == cols.user_value or low in ("user", "human"):
        return "user"
    if "system" in low:
        return "system"
    # Unknown speaker: treat as user so it becomes context, never a label.
    return "user"


def build_dpo_examples(
    rows: list[dict[str, Any]], cols: DpoColumns
) -> list[dict[str, str]]:
    """One {prompt, chosen, rejected} per row, applying the validator's format
    templates. Rows missing any of the three fields are dropped.
    """
    out: list[dict[str, str]] = []
    for row in rows:
        prompt_raw = row.get(cols.prompt)
        chosen_raw = row.get(cols.chosen)
        rejected_raw = row.get(cols.rejected)
        if prompt_raw is None or chosen_raw is None or rejected_raw is None:
            continue
        # Drop degenerate pairs (identical chosen/rejected carry zero preference
        # signal) and strip junk-token literals.
        if str(chosen_raw) == str(rejected_raw):
            continue
        system = str(row.get(cols.system, "") or "") if cols.system else ""
        prompt = _clean(cols.prompt_format.format(prompt=str(prompt_raw), system=system))
        chosen = _clean(cols.chosen_format.format(chosen=str(chosen_raw)))
        rejected = _clean(cols.rejected_format.format(rejected=str(rejected_raw)))
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
            example[spec.extra_column] = row[spec.extra_column]
        out.append(example)
    return out
