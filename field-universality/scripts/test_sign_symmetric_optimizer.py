from __future__ import annotations

import unittest

import numpy as np

import optimize_sign_symmetric_bulk as symmetric
from verify_sign_orbit_bulk import make_layout


class SignSymmetricOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = make_layout(5, sign_symmetric=True)
        random = np.random.default_rng(17)
        self.point = random.uniform(0.1, 1.0, self.layout.size)
        for component in (
            self.layout.mu,
            self.layout.beta,
            self.layout.zeta,
            self.layout.gamma1,
            self.layout.gamma2,
        ):
            self.point[component] /= self.point[component].sum()

    def test_exponent_gradient(self) -> None:
        value, gradient = symmetric.exponent_and_gradient(
            self.point, self.layout, 3
        )
        self.assertTrue(np.isfinite(value))
        step = 1e-7
        for index in range(self.layout.size):
            plus = self.point.copy()
            minus = self.point.copy()
            plus[index] += step
            minus[index] -= step
            plus_value, _ = symmetric.exponent_and_gradient(
                plus, self.layout, 3
            )
            minus_value, _ = symmetric.exponent_and_gradient(
                minus, self.layout, 3
            )
            finite_difference = (plus_value - minus_value) / (2.0 * step)
            self.assertAlmostEqual(
                finite_difference, gradient[index], places=6
            )

    def test_gap_gradient(self) -> None:
        value, gradient = symmetric.gap_and_gradient(
            self.point, self.layout, 3
        )
        self.assertTrue(np.isfinite(value))
        step = 1e-7
        for index in range(self.layout.size):
            plus = self.point.copy()
            minus = self.point.copy()
            plus[index] += step
            minus[index] -= step
            plus_value, _ = symmetric.gap_and_gradient(plus, self.layout, 3)
            minus_value, _ = symmetric.gap_and_gradient(
                minus, self.layout, 3
            )
            finite_difference = (plus_value - minus_value) / (2.0 * step)
            self.assertAlmostEqual(
                finite_difference, gradient[index], places=6
            )

    def test_information_identity(self) -> None:
        exponent, _ = symmetric.exponent_and_gradient(
            self.point, self.layout, 3
        )
        decomposition = symmetric.information_decomposition(
            self.point, self.layout, 3
        )
        self.assertAlmostEqual(
            exponent, decomposition["reconstructed_exponent"], places=12
        )

    def test_sign_branch_identity_on_a_feasible_transition(self) -> None:
        layout = self.layout
        increment = np.array([0.6, 0.3, 0.1])
        state = np.array([0.2, 0.4, 0.4])
        full_increment = np.array([0.6, 0.15, 0.05, 0.05, 0.15])
        full_state = np.full(5, 0.2)
        full_pair = np.outer(full_increment, full_state)
        orbit_values = np.zeros(len(layout.pair_representatives))
        for index, (source, target) in enumerate(
            layout.pair_representatives
        ):
            opposite = ((-source) % 5, (-target) % 5)
            orbit_values[index] = full_pair[
                (target - source) % 5, source
            ]
            if opposite != (source, target):
                orbit_values[index] += full_pair[
                    (opposite[1] - opposite[0]) % 5, opposite[0]
                ]
        stage = symmetric.orbit_transition_decomposition(
            orbit_values, increment, state, layout
        )
        full_information = (
            symmetric.full_orbit_entropy(increment)
            + symmetric.full_orbit_entropy(state)
            - symmetric.full_pair_entropy(orbit_values, layout)
        )
        self.assertAlmostEqual(
            full_information - stage["orbit_information"],
            stage["relative_sign_information_toll"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
