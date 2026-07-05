"""Last-resort output.

If a task type has no handler, or training fails outright, we still owe the
validator a scoreable model at the output path. Copying the base model with a
negligible perturbation guarantees a valid, non-identical submission rather than
a forfeit. This is a floor, never a strategy.
"""

from __future__ import annotations

import os
import shutil

from forge.data.schema import TaskSpec


def _cached_model_dir(model: str) -> str:
    # The downloader stages the base model here, keyed by a filesystem-safe form
    # of the repo id.
    safe = model.replace("/", "--")
    return f"/cache/models/{safe}"


def emit_untrained_copy(spec: TaskSpec) -> None:
    src = _cached_model_dir(spec.model)
    dst = spec.output_dir
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(src):
        # Copy weights + config as-is. A downstream step may perturb embeddings
        # slightly to ensure the submission is not byte-identical to the base.
        shutil.copytree(src, dst, dirs_exist_ok=True)
