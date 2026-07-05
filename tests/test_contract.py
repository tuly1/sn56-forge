"""Contract-level tests: these run without any ML dependency and prove the
argument surface and dataset-type parsing match the validator spec.
"""

import json

from forge.clock import Deadline
from forge.data.schema import TaskSpec


def test_instruct_spec_and_output_path():
    dt = json.dumps({"field_instruction": "q", "field_output": "a", "field_input": "ctx"})
    spec = TaskSpec.build(
        task_id="abc123",
        task_type="InstructTextTask",
        model="unsloth/Meta-Llama-3.1-8B",
        dataset="s3://x",
        dataset_type_json=dt,
        expected_repo_name="my-model",
        baseline_stats_path=None,
    )
    assert spec.instruct.instruction == "q"
    assert spec.instruct.output == "a"
    assert spec.instruct.input == "ctx"
    assert spec.output_dir == "/app/checkpoints/abc123/my-model"


def test_dpo_alias_fields():
    dt = json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"})
    spec = TaskSpec.build(
        task_id="t",
        task_type="DpoTask",
        model="m",
        dataset=None,
        dataset_type_json=dt,
        expected_repo_name="repo",
        baseline_stats_path=None,
    )
    assert (spec.dpo.prompt, spec.dpo.chosen, spec.dpo.rejected) == ("p", "c", "r")


def test_grpo_defaults_weights_to_ones():
    dt = json.dumps({"field_prompt": "p", "reward_functions": ["def r(): pass", "def s(): pass"]})
    spec = TaskSpec.build(
        task_id="t",
        task_type="GrpoTask",
        model="m",
        dataset=None,
        dataset_type_json=dt,
        expected_repo_name="repo",
        baseline_stats_path=None,
    )
    assert spec.grpo.reward_weights == [1.0, 1.0]


def test_deadline_pacing_math():
    # Construct a deadline 1 hour out with a 3-minute export reserve.
    d = Deadline.from_hours(1.0, started_monotonic=0.0, export_reserve_s=180.0)
    assert d.hard_stop == 3600.0
    assert d.soft_stop == 3420.0
    # With no timing recorded yet, we can't estimate steps.
    assert d.per_step() is None
    assert d.affordable_steps() is None
    # Record a few 2-second steps; median should be 2.0.
    for _ in range(5):
        d.record_step(2.0)
    assert d.per_step() == 2.0
