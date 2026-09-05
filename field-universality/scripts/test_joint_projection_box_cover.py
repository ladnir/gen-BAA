#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import explore_joint_projection_box_cover as cover
import optimize_joint_projection_bound as outer


class JointProjectionBoxCoverTests(unittest.TestCase):
    @staticmethod
    def reference_point() -> np.ndarray:
        return np.array(
            [
                0.7831573958083606, 0.008355485939891124,
                0.1000658321655323, 0.10006580515919077,
                0.008355480927024973,
                0.7753046482645539, 0.01067110536238914,
                0.10167657067429875, 0.10167656466506632,
                0.010671111033691725,
                0.74, 0.027221715349058654, 0.10277827751923847,
                0.10277831435503053, 0.027221692776672198,
                0.09725959055798944, 0.0929598642524958,
            ]
        )

    def test_scaled_cut_is_exact_at_scaled_reference(self) -> None:
        point = self.reference_point()
        cut = cover.fit_cut(point, 1)
        multiplier = 2
        scaled = point.copy()
        for start in (0, 5, 10):
            block = point[start : start + 5]
            scaled[start : start + 5] = np.array(
                [block[(multiplier * value) % 5] for value in range(5)]
            )
        scaled_cut = cover.scale_cut(cut, multiplier)
        box = cover.Box(tuple(scaled), tuple(scaled))
        lower = cover.payment_lower(box, scaled_cut)
        increment, state, crossing = cover.stage_components(scaled, 1)
        exact, diagnostic = outer.stage_payment_dual(increment, state, crossing)
        self.assertTrue(diagnostic["feasible"])
        self.assertAlmostEqual(lower, exact, places=8)

    def test_explicit_support_bounds_are_used_for_reference_membership(self) -> None:
        point = self.reference_point()
        support = tuple(
            float(np.sum(point[start + 1 : start + 5])) for start in (0, 5, 10)
        )
        box = cover.Box(tuple(point - 1e-6), tuple(point + 1e-6), support, support)
        self.assertTrue(cover.point_in_box(box, point))
        excluded = cover.Box(
            tuple(point - 1e-6),
            tuple(point + 1e-6),
            (support[0] + 1e-4, support[1], support[2]),
            (1.0, 1.0, 1.0),
        )
        self.assertFalse(cover.point_in_box(excluded, point))

    def test_point_box_reproduces_joint_objective(self) -> None:
        point = self.reference_point()
        box = cover.Box(tuple(point), tuple(point))
        first = cover.fit_cut(point, 1)
        second = cover.fit_cut(point, 2)
        bounded = cover.objective_upper(box, (first,), (second,))
        exact, diagnostic = outer.objective(point)
        self.assertTrue(diagnostic["feasible"])
        self.assertAlmostEqual(bounded, exact, places=8)

    def test_finite_box_contains_sample_values(self) -> None:
        center = self.reference_point()
        radius = 1e-5
        box = cover.Box(tuple(center - radius), tuple(center + radius))
        first = cover.fit_cut(center, 1)
        second = cover.fit_cut(center, 2)
        bounded = cover.objective_upper(box, (first,), (second,))
        random = np.random.default_rng(9)
        for _ in range(10):
            direction = random.uniform(-radius, radius, len(center))
            for start in (0, 5, 10):
                direction[start : start + 5] -= np.mean(
                    direction[start : start + 5]
                )
            sample = center + direction
            value, diagnostic = outer.objective(sample)
            if diagnostic["feasible"]:
                self.assertLessEqual(value, bounded + 1e-10)


if __name__ == "__main__":
    unittest.main()
