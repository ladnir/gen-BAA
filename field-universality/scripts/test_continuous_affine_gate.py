#!/usr/bin/env python3
"""Tests for the continuous affine-gate search helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import explore_projected_kl_box_cover as projected
import search_continuous_affine_gate as continuous


class ContinuousAffineGateTests(unittest.TestCase):
    def test_distinct_points_preserves_first_representative(self) -> None:
        points = [
            np.asarray([0.0, 1.0]),
            np.asarray([5e-10, 1.0]),
            np.asarray([0.2, 0.8]),
        ]
        result = continuous.distinct_points(points)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], points[0])
        np.testing.assert_array_equal(result[1], points[2])

    def test_point_in_box_allows_roundoff_but_rejects_parent_only_points(
        self,
    ) -> None:
        lower = np.array([0.0, 0.25])
        upper = np.array([0.5, 0.75])
        self.assertTrue(
            continuous.point_in_box(
                np.array([-1e-10, 0.7500000001]), lower, upper
            )
        )
        self.assertFalse(
            continuous.point_in_box(np.array([0.51, 0.5]), lower, upper)
        )

    def test_full_seed_inherits_saved_experts(self) -> None:
        affine = {"expert_pairs": [{"first_cut": 1, "second_cut": 2}]}
        saved = {"expert_pairs": [{"first_cut": 3, "second_cut": 4}]}
        self.assertEqual(
            continuous.initial_expert_pairs(
                affine, saved, seed_points_only=False
            ),
            saved["expert_pairs"],
        )

    def test_points_only_seed_uses_local_experts(self) -> None:
        affine = {"expert_pairs": [{"first_cut": 1, "second_cut": 2}]}
        saved = {"expert_pairs": [{"first_cut": 3, "second_cut": 4}]}
        self.assertEqual(
            continuous.initial_expert_pairs(
                affine, saved, seed_points_only=True
            ),
            affine["expert_pairs"],
        )

    def test_batched_stage_values_match_scalar_evaluator(self) -> None:
        random = np.random.default_rng(17)
        increments = random.dirichlet(np.ones(5), size=7)
        states = random.dirichlet(np.ones(5), size=7)
        cuts = [
            SimpleNamespace(
                target_potential=random.normal(size=5),
                increment_support_potential=float(random.normal()),
                state_support_potential=float(random.normal()),
            )
            for _ in range(4)
        ]
        kernels = random.uniform(0.2, 1.8, size=(4, 5, 5))
        batched = continuous.stage_upper_rows(
            increments, states, cuts, kernels
        )
        scalar = np.asarray(
            [
                [
                    projected.stage_upper_with_kernel(
                        increment, state, cut, kernel
                    )
                    for cut, kernel in zip(cuts, kernels, strict=True)
                ]
                for increment, state in zip(increments, states, strict=True)
            ]
        )
        np.testing.assert_allclose(batched, scalar, rtol=1e-13, atol=1e-13)

    def test_message_directional_derivative_matches_finite_difference(self) -> None:
        random = np.random.default_rng(23)
        points = np.concatenate(
            [random.dirichlet(np.ones(5), size=3) for _ in range(3)], axis=1
        )
        cuts = [
            SimpleNamespace(
                target_potential=random.normal(size=5),
                increment_support_potential=float(random.normal()),
                state_support_potential=float(random.normal()),
            )
            for _ in range(2)
        ]
        kernels = random.uniform(0.5, 1.5, size=(2, 5, 5))

        def exponentials(values: np.ndarray) -> np.ndarray:
            return np.exp(
                continuous.entropy_rows(values[:, 0:5])[:, None] / 3.0
                + continuous.stage_upper_rows(
                    values[:, 0:5], values[:, 5:10], cuts, kernels
                )
            )

        base = exponentials(points)
        analytic = continuous.message_directional_exponential_derivatives(
            points, base, cuts, kernels, coordinate=3
        )
        epsilon = 1e-7
        plus = points.copy()
        minus = points.copy()
        plus[:, 0] += epsilon
        plus[:, 3] -= epsilon
        minus[:, 0] -= epsilon
        minus[:, 3] += epsilon
        finite = (exponentials(plus) - exponentials(minus)) / (2 * epsilon)
        np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-8)

    def test_affine_feature_direction_includes_weight_derivative(self) -> None:
        np.testing.assert_array_equal(
            continuous.affine_feature_direction([0, 1, 3, 7], 3),
            np.asarray([0.0, 1.0, 0.0, -1.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
