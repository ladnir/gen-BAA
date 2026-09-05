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
            source_target = (
                product_bound.conditional_product_source_target(
                    increment, state, crossing
                )
            )
            np.testing.assert_allclose(
                source_target.sum(axis=1), state, atol=1e-12
            )
            stationary = product_bound.stationary_coupling_divergence(
                increment, state, crossing
            )
            source_potential, target_potential = (
                product_bound.stationary_coupling_dual_potentials(
                    increment, state, crossing
                )
            )
            stationary_dual = product_bound.stationary_coupling_dual_value(
                source_target,
                state,
                source_potential,
                target_potential,
            )
            self.assertAlmostEqual(stationary, stationary_dual, places=11)
            scaled_stationary, _ = product_bound.matrix_scaling_divergence(
                source_target, state, state
            )
            self.assertAlmostEqual(stationary, scaled_stationary, places=9)
            cell_target = product_bound.cell_target_projection_divergence(
                reference, state
            )
            hybrid = product_bound.projected_divergence(
                increment, state, crossing, "hybrid"
            )
            self.assertGreaterEqual(
                result["internal_information_excess"] + 1e-10,
                stationary,
            )
            self.assertGreaterEqual(
                result["internal_information_excess"] + 1e-10,
                hybrid,
            )
            self.assertGreaterEqual(hybrid + 1e-12, stationary)
            self.assertGreaterEqual(hybrid + 1e-12, cell_target)

    def test_support_box_constraints_cover_all_five_coordinates(self) -> None:
        layout = product_bound.Layout(5)
        lower = (0.10, 0.11, 0.12, 0.03, 0.04)
        upper = (0.20, 0.21, 0.22, 0.08, 0.09)
        rows, targets = product_bound.append_support_box_constraints(
            np.zeros((0, layout.size)),
            np.zeros(0),
            layout,
            lower,
            upper,
            free=False,
        )
        point = np.zeros(layout.size)
        for component, support in zip(
            (layout.mu, layout.beta, layout.zeta),
            (0.15, 0.16, 0.17),
            strict=True,
        ):
            point[component.start] = 1.0 - support
            point[component.start + 1 : component.stop] = support / 4.0
        point[layout.crossing1] = 0.05
        point[layout.crossing2] = 0.06
        self.assertTrue(np.all(rows @ point <= targets))
        point[layout.mu.start + 1 : layout.mu.stop] = 0.25 / 4.0
        self.assertTrue(np.any(rows @ point > targets))

    def test_sign_symmetric_free_map(self) -> None:
        layout = product_bound.Layout(7)
        mapping = product_bound.sign_symmetric_free_map(layout)
        reduced = np.arange(11, dtype=float) / 100.0
        point = product_bound.expand_free_point(mapping @ reduced, layout)
        for component in (layout.mu, layout.beta, layout.zeta):
            values = point[component]
            np.testing.assert_allclose(values[1:4], values[:3:-1])
        self.assertAlmostEqual(point[layout.crossing1], reduced[-2])
        self.assertAlmostEqual(point[layout.crossing2], reduced[-1])


if __name__ == "__main__":
    unittest.main()
