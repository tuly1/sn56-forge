"""Parse the validator's `--dataset-type` payload into a typed spec.

The validator passes a JSON blob whose shape depends on the task type. Rather
than thread a dict of stringly-typed keys through the pipeline, we resolve it
once here into a small immutable record per task family and fail loudly if a
required field is missing. Column names are taken from the payload, never
assumed, because the validator renames them per task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _require(d: dict[str, Any], *names: str) -> Any:
    """Return the first present key among `names`, else raise. Task payloads
    have drifted field names across spec versions, so we accept aliases.
    """
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    raise KeyError(f"dataset-type missing one of {names!r}; got keys {sorted(d)}")


def _first(d: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Like `_require` but returns `default` instead of raising."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


@dataclass(frozen=True)
class InstructColumns:
    instruction: str
    # When output is None the task is completion-style: the validator supervises
    # the entire instruction text (no prompt mask). See build_instruct_examples.
    output: str | None = None
    input: str | None = None
    system: str | None = None
    # A literal system prompt applied to every row (validator field
    # `system_prompt`), distinct from `system` which names a per-row column.
    system_prompt: str = ""
    # Prompt templates as the validator's axolotl config would render them. We
    # honour whatever it sends and default to the same strings the reference
    # trainer uses, so our training prompts track the eval-time prompts.
    fmt: str | None = None
    no_input_fmt: str | None = None
    system_format: str = "{system}"

    def render_prompt(self, row: dict[str, Any]) -> str:
        """Build the prompt half of an example from a dataset row."""
        instruction = str(row.get(self.instruction, "") or "")
        has_input = bool(self.input and str(row.get(self.input, "") or "").strip())
        if has_input:
            template = self.fmt or "{instruction} {input}"
            body = template.format(
                instruction=instruction, input=str(row.get(self.input, "") or "")
            )
        else:
            template = self.no_input_fmt or "{instruction}"
            body = template.format(instruction=instruction)

        sys_text = self.system_prompt or ""
        if self.system and str(row.get(self.system, "") or "").strip():
            sys_text = str(row.get(self.system))
        if sys_text:
            # Render the system segment through the validator's system_format
            # (default "{system}") and concatenate directly. Any separator is the
            # template's business — injecting our own would shift the boundary
            # tokens away from what the evaluator scores.
            return self.system_format.format(system=sys_text) + body
        return body

    def render_completion(self, row: dict[str, Any]) -> str:
        return str(row.get(self.output, "") or "")


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
    prompt_format: str = "{prompt}"
    chosen_format: str = "{chosen}"
    rejected_format: str = "{rejected}"
    system: str | None = None


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
    file_format: str = "s3"
    # KL-regularised instruct tasks: the scorer adds kl_coef * KL(model || base)
    # over completion tokens, so we mirror the term in training. Sourced from the
    # USE_KL / KL_COEF environment variables the validator injects.
    use_kl: bool = False
    kl_coef: float = 0.0
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

    @property
    def cached_dataset_path(self) -> str:
        # Where the validator pre-stages the training data (read-only /cache).
        return f"/cache/datasets/{self.task_id}_train_data.json"

    @property
    def cached_model_dir(self) -> str:
        # The base model, keyed by a filesystem-safe form of the (possibly
        # anonymised) model id passed as --model.
        return f"/cache/models/{self.model.replace('/', '--')}"

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
        file_format: str = "s3",
        use_kl: bool = False,
        kl_coef: float = 0.0,
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
            file_format=file_format,
            use_kl=use_kl,
            kl_coef=kl_coef,
        )

        if task_type == "InstructTextTask":
            return cls(
                **common,
                instruct=InstructColumns(
                    instruction=_require(payload, "field_instruction", "instruction"),
                    # Optional: absent field_output means a completion-style task.
                    output=_first(payload, "field_output", "output"),
                    input=_first(payload, "field_input", "input"),
                    system=_first(payload, "field_system"),
                    system_prompt=_first(payload, "system_prompt", default="") or "",
                    fmt=_first(payload, "format"),
                    no_input_fmt=_first(payload, "no_input_format"),
                    system_format=_first(payload, "system_format", default="{system}")
                    or "{system}",
                ),
            )
        if task_type == "ChatTask":
            # The validator's ChatTemplateDatasetType ships real defaults; a
            # partial payload must fall back to them, not crash. These match
            # core/models/dataset_models.py in the G.O.D repo.
            return cls(
                **common,
                chat=ChatColumns(
                    conversation=_first(
                        payload, "chat_column", "conversation", default="conversations"
                    ),
                    role_field=_first(
                        payload, "chat_role_field", "role", default="from"
                    ),
                    content_field=_first(
                        payload, "chat_content_field", "content", default="value"
                    ),
                    user_value=_first(
                        payload, "chat_user_reference", "user", default="user"
                    ),
                    assistant_value=_first(
                        payload, "chat_assistant_reference", "assistant", default="assistant"
                    ),
                    chat_template=_first(payload, "chat_template"),
                ),
            )
        if task_type == "DpoTask":
            return cls(
                **common,
                dpo=DpoColumns(
                    prompt=_require(payload, "field_prompt", "prompt"),
                    chosen=_require(payload, "field_chosen", "chosen"),
                    rejected=_require(payload, "field_rejected", "rejected"),
                    prompt_format=_first(payload, "prompt_format", default="{prompt}"),
                    chosen_format=_first(payload, "chosen_format", default="{chosen}"),
                    rejected_format=_first(
                        payload, "rejected_format", default="{rejected}"
                    ),
                    system=_first(payload, "field_system"),
                ),
            )
        if task_type == "GrpoTask":
            fns_raw = _require(payload, "reward_functions")
            fns, weights = _normalise_reward_functions(fns_raw)
            weights = payload.get("reward_weights") or weights
            return cls(
                **common,
                grpo=GrpoSpec(
                    prompt=_require(payload, "field_prompt", "prompt"),
                    reward_functions=fns,
                    reward_weights=[float(w) for w in weights],
                    extra_column=_first(payload, "extra_column"),
                ),
            )
        # EnvTask and anything unrecognised: carry the bare spec; the dispatcher
        # decides whether a handler exists.
        return cls(**common)


def _normalise_reward_functions(raw: list[Any]) -> tuple[list[str], list[float]]:
    """Accept either a list of source strings or a list of RewardFunction dicts
    (`{"reward_func": "...", "reward_weight": 0.7, ...}`), which is the shape the
    live validator sends. Returns (sources, weights).
    """
    sources: list[str] = []
    weights: list[float] = []
    for item in raw:
        if isinstance(item, dict):
            sources.append(_require(item, "reward_func", "func", "source"))
            weights.append(float(_first(item, "reward_weight", "weight", default=1.0)))
        else:
            sources.append(str(item))
            weights.append(1.0)
    return sources, weights
