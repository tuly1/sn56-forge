"""CPU-only numerical and evidence-contract tests; no model or data access."""
from dataclasses import replace
import itertools
import math
import random
import unittest

try:
    from forge.tuning.lr_calibration import ProbeSummary, select_stability_edge, warmup_curvature
except ModuleNotFoundError:
    from forge.tuning.lr_calibration import ProbeSummary, select_stability_edge, warmup_curvature


def arm(name, lr, loss, **overrides):
    fields = dict(probe_id=name, learning_rate=lr, loss=loss, planned_steps=25,
                  completed_steps=25, comparison_id="same-init-data-optimizer-geometry",
                  state_restored=True)
    fields.update(overrides)
    return ProbeSummary(**fields)


class WarmupTests(unittest.TestCase):
    def test_eight_step_first_and_last_two_points(self):
        result = warmup_curvature([10, 10, 500, 400, 300, 200, 9, 9])
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.window_size, 2)  # not three despite reference comment
        self.assertAlmostEqual(result.relative_drop, .1)
        self.assertAlmostEqual(result.factor, 1.24)

    def test_neutral_flat_and_clamps(self):
        self.assertAlmostEqual(warmup_curvature([10]*6 + [9.6]*2).factor, 1)
        self.assertAlmostEqual(warmup_curvature([10]*8).factor, .84)
        self.assertEqual(warmup_curvature([10]*6 + [1]*2).factor, 1.6)
        self.assertEqual(warmup_curvature([10]*6 + [20]*2).factor, .6)

    def test_rising_more_than_half_never_uses_existing_point_three_override(self):
        self.assertEqual(warmup_curvature([2]*6 + [4]*2).factor, .6)

    def test_missing_evidence_has_no_silent_recommendation(self):
        for n in range(8):
            with self.subTest(n=n):
                result = warmup_curvature([1]*n)
                self.assertEqual(result.status, "insufficient_steps")
                self.assertIsNone(result.factor)
        self.assertEqual(warmup_curvature([1]*3, require_complete=False).status, "ready")
        self.assertIsNone(warmup_curvature([1]*2, require_complete=False).factor)

    def test_nonfinite_missing_negative_and_wrong_types_fail_closed(self):
        for bad in [None, math.inf, -math.inf, math.nan, -1, "1.0", True, complex(1, 2), 10**400]:
            with self.subTest(bad=bad):
                result = warmup_curvature([1]*7 + [bad])
                self.assertEqual(result.status, "invalid_loss")
                self.assertIsNone(result.factor)

    def test_zero_and_tiny_denominator_follow_documented_floor(self):
        self.assertAlmostEqual(warmup_curvature([0]*8).factor, .84)
        result = warmup_curvature([1e-8]*6 + [0]*2)
        self.assertAlmostEqual(result.relative_drop, .01)
        self.assertAlmostEqual(result.factor, .88)

    def test_extreme_finite_values_cannot_emit_nonfinite_recommendation(self):
        result = warmup_curvature([1e308]*8)
        self.assertAlmostEqual(result.factor, .84)
        result = warmup_curvature([0]*6 + [1e308]*2)
        self.assertEqual(result.status, "numerical_overflow")
        self.assertIsNone(result.factor)

    def test_scale_invariance_away_from_denominator_floor(self):
        rng = random.Random(8841)
        for _ in range(100):
            points = [rng.uniform(.1, 5) for _ in range(8)]
            self.assertAlmostEqual(warmup_curvature(points).factor,
                                   warmup_curvature([100*x for x in points]).factor, places=13)

    def test_bounded_and_does_not_mutate(self):
        points = [1, 2, 3, 4, 5, 6, 7, 8]
        original = points[:]
        warmup_curvature(points)
        self.assertEqual(points, original)
        with self.assertRaises(ValueError):
            warmup_curvature([1]*9)


class EdgeTests(unittest.TestCase):
    def test_selects_highest_lr_inside_band_not_minimum_loss(self):
        arms = [arm("low", 1e-5, 1), arm("mid", 2e-5, 1.02),
                arm("high", 4e-5, 1.024), arm("too_hot", 8e-5, 1.03)]
        result = select_stability_edge(arms)
        self.assertEqual(result.status, "selected")
        self.assertEqual(result.selected_id, "high")
        self.assertEqual(result.learning_rate, 4e-5)
        self.assertEqual(result.best_loss, 1)
        self.assertEqual(result.eligible_ids, ("high", "low", "mid"))

    def test_inclusive_boundary_and_just_outside(self):
        threshold = 1.0 * 1.025
        self.assertEqual(select_stability_edge([arm("a", 1e-5, 1), arm("b", 2e-5, threshold)]).selected_id, "b")
        beyond = math.nextafter(threshold, math.inf)
        self.assertEqual(select_stability_edge([arm("a", 1e-5, 1), arm("b", 2e-5, beyond)]).selected_id, "a")

    def test_order_invariance_and_exact_ties(self):
        arms = [arm("a", 1e-5, 1), arm("b", 2e-5, 1), arm("c", 3e-5, 1.1)]
        expected = select_stability_edge(arms)
        for order in itertools.permutations(arms):
            self.assertEqual(select_stability_edge(order), expected)
        self.assertEqual(expected.selected_id, "b")

    def test_zero_best_only_admits_zero_loss(self):
        result = select_stability_edge([arm("a", 1e-5, 0), arm("b", 2e-5, 1e-10)])
        self.assertEqual(result.selected_id, "a")
        self.assertEqual(result.threshold, 0)

    def test_explicit_divergence_excluded_but_not_silently_rescued(self):
        arms = [arm("a", 1e-5, 1), arm("b", 2e-5, 1.01),
                arm("hot", 4e-5, math.inf, completed_steps=4, status="diverged")]
        result = select_stability_edge(arms)
        self.assertEqual(result.selected_id, "b")
        self.assertEqual(result.excluded_ids, ("hot",))
        self.assertEqual(select_stability_edge(arms[1:]).status, "insufficient_valid_probes")
        self.assertIsNone(select_stability_edge(arms[1:]).learning_rate)

    def test_empty_one_and_all_diverged(self):
        self.assertEqual(select_stability_edge([]).status, "insufficient_probes")
        self.assertEqual(select_stability_edge([arm("a", 1e-5, 1)]).status, "insufficient_valid_probes")
        arms = [arm(str(i), (i+1)*1e-5, None, status="diverged", completed_steps=3) for i in range(2)]
        self.assertEqual(select_stability_edge(arms).status, "insufficient_valid_probes")

    def test_rejects_mixed_data_horizon_or_metric(self):
        a = arm("a", 1e-5, 1)
        for b, status in [
            (arm("b", 2e-5, .9, comparison_id="different-data-or-init"), "incomparable_probes"),
            (arm("b", 2e-5, .9, planned_steps=50, completed_steps=50), "incomparable_probes"),
            (arm("b", 2e-5, .9, metric="dpo_heldout"), "unsupported_metric"),
            (arm("b", 2e-5, -.9, metric="grpo_reward"), "unsupported_metric"),
            (arm("b", 2e-5, .9, state_restored=False), "restoration_unverified"),
        ]:
            with self.subTest(status=status):
                result = select_stability_edge([a, b])
                self.assertEqual(result.status, status)
                self.assertIsNone(result.learning_rate)

    def test_incomplete_or_corrupt_arm_holds_entire_comparison(self):
        a = arm("a", 1e-5, 1)
        cases = [(dict(status="incomplete"), "incomplete_probes"),
                 (dict(completed_steps=24), "incomplete_probes"),
                 (dict(status="mystery"), "invalid_probe_status"),
                 (dict(completed_steps=26), "invalid_step_counts"),
                 (dict(planned_steps=True), "invalid_step_counts"),
                 (dict(comparison_id=" "), "missing_identity")]
        for fields, status in cases:
            with self.subTest(fields=fields):
                self.assertEqual(select_stability_edge([a, arm("b", 2e-5, .9, **fields)]).status, status)
        for loss in [None, -1, math.inf, math.nan, True, "1"]:
            with self.subTest(loss=loss):
                self.assertEqual(select_stability_edge([a, arm("b", 2e-5, loss)]).status, "invalid_completed_loss")

    def test_no_duplicate_lr_or_identity_overwrite(self):
        a = arm("a", 1e-5, 1)
        self.assertEqual(select_stability_edge([a, arm("a", 2e-5, .9)]).status, "duplicate_probe_id")
        self.assertEqual(select_stability_edge([a, arm("b", 1e-5, .9)]).status, "duplicate_learning_rate")

    def test_limits_and_numerical_failures(self):
        for lr in [0, -1, math.inf, math.nan, True]:
            self.assertEqual(select_stability_edge([arm("a", lr, 1)]).status, "invalid_learning_rate")
        for tol in [math.nan, math.inf, -1, 1.1, True]:
            with self.assertRaises(ValueError):
                select_stability_edge([], tolerance=tol)
        with self.assertRaises(ValueError):
            select_stability_edge([arm(str(i), (i+1)*1e-5, 1) for i in range(5)])
        with self.assertRaises(TypeError):
            select_stability_edge([{}])
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        result = select_stability_edge([arm("a", 1e-5, maximum), arm("b", 2e-5, maximum)])
        self.assertEqual(result.status, "numerical_overflow")

    def test_loss_scaling_and_no_input_mutation(self):
        arms = [arm("a", 1e-5, 1), arm("b", 2e-5, 1.024)]
        original = arms[:]
        result = select_stability_edge(arms)
        scaled = select_stability_edge([replace(p, loss=p.loss*100) for p in arms])
        self.assertEqual(result.selected_id, scaled.selected_id)
        self.assertEqual(arms, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
