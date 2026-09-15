"""Exercise the production memory-probe call and threshold without a GPU.

The original geometry must remain available outside the new field route.
SOURCE_TO_REVIEW can point at an unmodified checkout to reproduce the failure.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest


SOURCE = Path(os.environ.get("SOURCE_TO_REVIEW", str(Path(__file__).parents[1]))) / "forge/tasks/sft_v2.py"
TREE = ast.parse(SOURCE.read_text())
CALL = next(n for n in ast.walk(TREE) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "memory_probe")
FN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "memory_probe")


def routed_headroom(field, strategy, fp32):
    env = dict(memory_probe=lambda *a, **kw: kw.get("headroom", 0.07),
               model=object(), micro_c=8, geo=NS(max_len=256, optim="adamw_torch", fp32_master=fp32),
               vocab=32, autocast=False, optimizer_state_bytes=lambda *a: 8,
               field=field, strategy=strategy)
    return eval(compile(ast.Expression(CALL), str(SOURCE), "eval"), env)


def probe_with_88_percent_peak_and_state(monkeypatch, headroom):
    # 80 units of allocated memory +8 optimizer-state units, total100.
    # This deliberately sits between the old93% and new85% ceilings.
    class Tensor:
        def clone(self):
            return self

    fake_torch = NS(randint=lambda *a, **k: Tensor(), ones_like=lambda x: Tensor(),
                    cuda=NS(reset_peak_memory_stats=lambda d: None,
                            max_memory_allocated=lambda d: 80,
                            get_device_properties=lambda d: NS(total_memory=100)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = NS(parameters=lambda: iter([NS(device=NS(type="cuda"))]), train=lambda: None,
               zero_grad=lambda **k: None)

    class Model:
        parameters = staticmethod(model.parameters)
        train = staticmethod(model.train)
        zero_grad = staticmethod(model.zero_grad)

        def __call__(self, **kw):
            return NS(loss=NS(backward=lambda: None))

    ns = dict(Any=object, _event_and_print=lambda *a, **k: None,
              _free_cuda=lambda: None, _is_oom=lambda e: False)
    exec(compile(ast.Module(body=[FN], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["memory_probe"](Model(), micro=8, max_len=256, vocab=32,
                              autocast=False, reserve_bytes=8, headroom=headroom)


@pytest.mark.parametrize("field,strategy,fp32,expected", [
    (False, "full", True, True),    # e.g. long-row Qwen0.5 / small-corpus Smol
    (True, "full", True, False),    # protect the newly validated field route
    (False, "lora", False, True),
    (False, "full", False, True),
])
def test_headroom_preserves_nonfield_geometry(monkeypatch, field, strategy, fp32, expected):
    assert probe_with_88_percent_peak_and_state(
        monkeypatch, routed_headroom(field, strategy, fp32)) is expected
