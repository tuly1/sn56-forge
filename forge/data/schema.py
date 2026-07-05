"""Parse the validator's `--dataset-type` payload into a typed spec.

The validator passes a JSON blob whose shape depends on the task type. Rather
than thread a dict of stringly-typed keys through the pipeline, we resolve it
once here into a small immutable record per task family and fail loudly if a
required field is missing. Column names are taken from the payload, never
assumed, because the validator renames them per task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _require(d: dict[str, Any], *names: str) -> Any:
    """Return the first present key among `names`, else raise. Task payloads
    have drifted field names across spec versions, so we accept aliases.
    """
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    raise KeyError(f"dataset-type missing one of {names!r}; got keys {sorted(d)}")


@dataclass(frozen=True)
class InstructColumns:
    instruction: str
    output: str
    input: str | None = None
    system: str | None = None


@dataclass(frozen=True)
class ChatColumns:
    conversation: str
    role_field: str
    content_field: str
    user_value: str
    assistant_value: str
    chat_template: str | None = None


@dataclass(frozen=True)
class DpoColumns:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class GrpoSpec:
    prompt: str
    reward_functions: list[str]
    reward_weights: list[float]
    extra_column: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    model: str
    dataset: str | None
    expected_repo_name: str
    baseline_stats_path: str | None
    # Exactly one of these is populated, matching task_type.
    instruct: InstructColumns | None = None
    chat: ChatColumns | None = None
    dpo: DpoColumns | None = None
    grpo: GrpoSpec | None = None

    @property
    def output_dir(self) -> str:
        # The one path the validator mandates. Everything the uploader reads
        # lives here.
        return f"/app/checkpoints/{self.task_id}/{self.expected_repo_name}"

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        task_type: str,
        model: str,
        dataset: str | None,
        dataset_type_json: str | None,
        expected_repo_name: str,
        baseline_stats_path: str | None,
    ) -> "TaskSpec":
        payload: dict[str, Any] = {}
        if dataset_type_json:
            payload = json.loads(dataset_type_json)

        common = dict(
            task_id=task_id,
            task_type=task_type,
            model=model,
            dataset=dataset,
            expected_repo_name=expected_repo_name,
            baseline_stats_path=baseline_stats_path,
        )

        if task_type in ("InstructTextTask",):
            return cls(
                **common,
                instruct=InstructColumns(
                    instruction=_require(payload, "field_instruction", "instruction"),
                    output=_require(payload, "field_output", "output"),
                    input=payload.get("field_input") or payload.get("input"),
                    system=payload.get("field_system") or payload.get("system"),
                ),
            )
        if task_type == "ChatTask":
            return cls(
                **common,
                chat=ChatColumns(
                    conversation=_require(payload, "chat_column", "conversation"),
                    role_field=_require(payload, "chat_role_field", "role"),
                    content_field=_require(payload, "chat_content_field", "content"),
                    user_value=_require(payload, "chat_user_reference", "user"),
                    assistant_value=_require(payload, "chat_assistant_reference", "assistant"),
                    chat_template=payload.get("chat_template"),
                ),
            )
        if task_type == "DpoTask":
            return cls(
                **common,
                dpo=DpoColumns(
                    prompt=_require(payload, "field_prompt", "prompt"),
                    chosen=_require(payload, "field_chosen", "chosen"),
                    rejected=_require(payload, "field_rejected", "rejected"),
                ),
            )
        if task_type == "GrpoTask":
            fns = _require(payload, "reward_functions")
            weights = payload.get("reward_weights") or [1.0] * len(fns)
            return cls(
                **common,
                grpo=GrpoSpec(
                    prompt=_require(payload, "field_prompt", "prompt"),
                    reward_functions=list(fns),
                    reward_weights=[float(w) for w in weights],
                    extra_column=payload.get("extra_column"),
                ),
            )
        # EnvTask and anything unrecognised: carry the bare spec; the dispatcher
        # decides whether a handler exists.
        return cls(**common)
