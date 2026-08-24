from __future__ import annotations

import unittest

import numpy as np

import diagnose_information_decomposition as decomposition
import optimize_bulk_exponent as bulk
import search_internal_information_gap as gap_search


class InformationDecompositionTests(unittest.TestCase):
    def test_support_pushforward_preserves_mass(self) -> None:
        joint = np.arange(1, 26, dtype=float).reshape((5, 5))
        joint /= joint.sum()
        pushed = decomposition.support_pushforward(joint)
        self.assertAlmostEqual(float(pushed.sum()), 1.0)
        self.assertAlmostEqual(pushed[0, 0], joint[0, 0])

    def test_gap_gradient_matches_finite_difference(self) -> None:
        layout = bulk.Layout(5)
        point = np.zeros(layout.size)
        point[layout.mu] = 1.0 / 5.0
        point[layout.beta] = 1.0 / 5.0
        point[layout.zeta] = 1.0 / 5.0
        point[layout.tau1] = 1.0 / 25.0
        point[layout.tau2] = 1.0 / 25.0
        random = np.random.default_rng(31)
        direction = random.normal(size=layout.size)
        direction /= np.linalg.norm(direction)
        value, gradient = gap_search.gap_and_gradient(point, layout, 3)
        epsilon = 1e-7
        forward = gap_search.gap_and_gradient(
            point + epsilon * direction, layout, 3
        )[0]
        backward = gap_search.gap_and_gradient(
            point - epsilon * direction, layout, 3
        )[0]
        finite_difference = (forward - backward) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(np.dot(gradient, direction)), finite_difference, places=7
        )
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
