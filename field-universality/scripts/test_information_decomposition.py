from __future__ import annotations

import unittest

import numpy as np

import diagnose_information_decomposition as decomposition
import optimize_bulk_exponent as bulk
import optimize_conditional_product_bound as product_bound
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

    def test_conditional_product_kl_bound(self) -> None:
        random = np.random.default_rng(29)
        for _ in range(20):
            transition = random.uniform(0.1, 1.0, size=(5, 5))
            transition += transition.T
            transition /= transition.sum()
            joint = decomposition.increment_source_joint(transition)
            result = decomposition.conditional_product_target_bound(joint)
            self.assertGreaterEqual(
                result["internal_information_excess"] + 1e-12,
                result["conditional_product_target_divergence"],
            )
            self.assertGreaterEqual(
                result["conditional_product_target_divergence"] + 1e-12,
                2.0
                * result["conditional_product_target_variation"] ** 2,
            )
            increment = joint.sum(axis=1)
            state = joint.sum(axis=0)
            crossing = decomposition.support_pushforward(joint)[1, 0]
            cancellation = product_bound.cancellation_divergence(
                increment, state, crossing
            )
            self.assertGreaterEqual(
                result["internal_information_excess"] + 1e-12,
                cancellation,
            )
            two_level = product_bound.two_level_divergence(
                increment, state, crossing
            )
            self.assertGreaterEqual(
                result["internal_information_excess"] + 1e-12,
                two_level,
            )
            self.assertGreaterEqual(two_level + 1e-12, cancellation)
            cell_potential, target_potential = (
                product_bound.two_level_dual_potentials(
                    increment, state, crossing
                )
            )
            reference = product_bound.conditional_product_cell_targets(
                increment, state, crossing
            )
            dual = product_bound.cell_target_dual_lower_bound(
                reference,
                reference.sum(axis=1),
                state,
                cell_potential,
                target_potential,
            )
            self.assertAlmostEqual(dual, two_level, places=11)


if __name__ == "__main__":
    unittest.main()
