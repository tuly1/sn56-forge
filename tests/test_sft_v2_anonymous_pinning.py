"""CPU regressions for the anonymous production contract at the actual v2 gate.

No model weights, datasets, GPU, or provider calls are needed. The route
expression is compiled from instruct.run so this covers the dispatch condition,
not merely the independent v2 size-eligibility helper.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from forge.data.schema import InstructColumns, TaskSpec
from forge.model import is_qwen35_model
from forge.tasks import sft_v2
from forge.tuning import granite41_epoch_cap as _g41
from forge.tuning import lfm25_epoch_cap as _lfm25


SOURCE = Path(__file__).resolve().parents[1] / "forge/tasks/instruct.py"
TREE = ast.parse(SOURCE.read_text())
ROUTE = next(
    n.value for n in ast.walk(TREE)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "pinned_route" for t in n.targets)
)
EXPR = compile(ast.Expression(ROUTE), str(SOURCE), "eval")


def spec(model="0123456789abcdef", stats="/cache/baseline_stats/task.json"):
    return TaskSpec(
        task_id="cpu-route-contract", task_type="InstructTextTask", model=model,
        dataset=None, expected_repo_name="cpu-only", baseline_stats_path=stats,
        instruct=InstructColumns("instruction", "output"),
    )


def pinned(task_spec, config):
    return eval(EXPR, globals(), {"spec": task_spec, "loaded": NS(model=NS(config=config))})


def granite(**changes):
    values = dict(model_type="granite", architectures=["GraniteForCausalLM"],
                  hidden_size=2560, intermediate_size=8192, num_hidden_layers=40,
                  num_attention_heads=40, num_key_value_heads=8)
    return NS(**(values | changes))


def lfm(**changes):
    values = dict(model_type="lfm2", architectures=["Lfm2ForCausalLM"],
                  hidden_size=2048, intermediate_size=10752, num_hidden_layers=30,
                  num_attention_heads=32, num_key_value_heads=8)
    return NS(**(values | changes))


class AnonymousPinningTests(unittest.TestCase):
    def test_unrelated_anonymous_families_reach_v2_gate(self):
        for name, model_type in [("Gemma2", "gemma2"), ("SmolLM2", "llama"),
                                 ("Qwen2.5", "qwen2"), ("Qwen3-32B", "qwen3")]:
            for stats in (None, "/cache/baseline_stats/task.json"):
                with self.subTest(model=name, stats=stats):
                    self.assertFalse(pinned(spec(stats=stats), NS(model_type=model_type)))

    def test_exact_anonymous_granite_and_lfm_stay_pinned(self):
        self.assertTrue(pinned(spec(), granite()))
        self.assertTrue(pinned(spec(), lfm()))

    def test_named_granite_and_lfm_stay_pinned(self):
        self.assertTrue(pinned(spec(_g41.EXACT_MODEL_ID), granite()))
        self.assertTrue(pinned(spec(_lfm25.EXACT_MODEL_ID), lfm()))

    def test_same_family_other_size_not_accidentally_pinned(self):
        self.assertFalse(pinned(spec(), granite(hidden_size=4096)))
        self.assertFalse(pinned(spec(), lfm(num_hidden_layers=16)))

    def test_public_name_does_not_override_wrong_architecture(self):
        self.assertFalse(pinned(spec(_g41.EXACT_MODEL_ID), NS(model_type="qwen3")))
        self.assertFalse(pinned(spec(_lfm25.EXACT_MODEL_ID), NS(model_type="gemma2")))

    def test_qwen35_all_supported_wrappers_stay_pinned(self):
        for config in (NS(model_type="qwen3_5"), NS(model_type="qwen3_5_text"),
                       NS(model_type="wrapper", text_config=NS(model_type="qwen3_5_text"))):
            self.assertTrue(pinned(spec(), config))

    def test_missing_or_incomplete_config_is_not_a_granite_lfm_match(self):
        self.assertFalse(pinned(spec(), None))
        self.assertFalse(pinned(spec(), NS(model_type="granite")))
        self.assertFalse(pinned(spec(), NS(model_type="lfm2")))


if __name__ == "__main__":
    unittest.main()
