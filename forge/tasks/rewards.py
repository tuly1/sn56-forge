"""Materialise GRPO reward functions from source strings.

The validator ships reward functions as Python source inside `--dataset-type`.
We compile each into a callable in an isolated namespace with common libraries
available. Sources that fail to compile are skipped (with their weight) rather
than forfeiting the whole task. Kept free of ML imports so it is cheap to test.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Callable


def materialise_rewards(
    sources: list[str], weights: list[float]
) -> tuple[list[Callable[..., Any]], list[float]]:
    funcs: list[Callable[..., Any]] = []
    kept_weights: list[float] = []
    for src, weight in zip(sources, weights):
        fn = _compile_one(src)
        if fn is None:
            continue
        funcs.append(fn)
        kept_weights.append(float(weight))
    return funcs, kept_weights


def _compile_one(src: str) -> Callable[..., Any] | None:
    name_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", src)
    if not name_match:
        return None
    fn_name = name_match.group(1)
    namespace: dict[str, Any] = {
        "re": re,
        "json": json,
        "math": math,
        "statistics": statistics,
    }
    try:
        exec(compile(src, f"<reward:{fn_name}>", "exec"), namespace)  # noqa: S102
        fn = namespace.get(fn_name)
    except Exception:
        return None
    if not callable(fn):
        return None
    fn.__name__ = fn_name
    return fn
