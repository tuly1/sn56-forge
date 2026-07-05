"""Map a task type to its handler, or None if this build doesn't implement it.

Handlers are imported lazily inside `for_task` so that importing this module
(and running the CLI's contract checks) never pulls in torch/transformers for a
task type we're not running.
"""

from __future__ import annotations

from collections.abc import Callable

from forge.clock import Deadline
from forge.data.schema import TaskSpec

Handler = Callable[[TaskSpec, Deadline], None]


def for_task(task_type: str) -> Handler | None:
    if task_type in ("InstructTextTask", "ChatTask"):
        from forge.tasks.instruct import run

        return run
    if task_type == "DpoTask":
        from forge.tasks.dpo import run

        return run
    if task_type == "GrpoTask":
        from forge.tasks.grpo import run

        return run
    # EnvTask not yet implemented.
    return None
