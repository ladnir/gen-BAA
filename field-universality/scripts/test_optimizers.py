#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import optimize_bulk_exponent as bulk
import optimize_symmetric_ternary as ternary


class OptimizerTests(unittest.TestCase):
    def test_prime_validation(self) -> None:
        self.assertTrue(bulk.is_prime(2))
        self.assertTrue(bulk.is_prime(5))
        self.assertFalse(bulk.is_prime(1))
        self.assertFalse(bulk.is_prime(9))

    def test_bulk_objective_gradient(self) -> None:
        layout = bulk.Layout(3)
        random = np.random.default_rng(7)
        point = random.uniform(0.05, 0.95, layout.size)
        analytic = bulk.objective_gradient(point, layout, 3)
        numerical = np.zeros(layout.size)
        step = 1e-7
        for coordinate in range(layout.size):
            forward = point.copy()
            backward = point.copy()
            forward[coordinate] += step
            backward[coordinate] -= step
            numerical[coordinate] = (
                -bulk.exponent(forward, layout, 3) + bulk.exponent(backward, layout, 3)
            ) / (2.0 * step)
        self.assertLess(float(np.max(np.abs(analytic - numerical))), 2e-7)

    def test_symmetric_transition_has_requested_marginals(self) -> None:
        increment_weight = 0.18
        state_weight = 0.19
        bridge = 0.04
        diagonal_zero, bridge, diagonal_nonzero, off_diagonal = ternary.transition_entries(
            increment_weight, state_weight, bridge
        )
        transition = np.array(
            [
                [diagonal_zero, bridge, bridge],
                [bridge, diagonal_nonzero, off_diagonal],
                [bridge, off_diagonal, diagonal_nonzero],
            ]
        )
        state = np.array([1.0 - state_weight, state_weight / 2.0, state_weight / 2.0])
        increment = np.array(
            [1.0 - increment_weight, increment_weight / 2.0, increment_weight / 2.0]
        )
        np.testing.assert_allclose(transition.sum(axis=0), state, atol=1e-14)
        np.testing.assert_allclose(transition.sum(axis=1), state, atol=1e-14)
        observed_increment = np.array(
            [
                sum(transition[source, (source + difference) % 3] for source in range(3))
                for difference in range(3)
            ]
        )
        np.testing.assert_allclose(observed_increment, increment, atol=1e-14)

    def test_symmetric_reduction_matches_full_exponent(self) -> None:
        repetition = 3
        message_weight = 0.18
        middle_weight = 0.19
        output_weight = 0.20
        _, first_bridge = ternary.transition_rate(message_weight, middle_weight)
        _, second_bridge = ternary.transition_rate(middle_weight, output_weight)

        def composition(weight: float) -> np.ndarray:
            return np.array([1.0 - weight, weight / 2.0, weight / 2.0])

        def transition(increment: float, state: float, bridge: float) -> np.ndarray:
            diagonal_zero, bridge, diagonal_nonzero, off_diagonal = (
                ternary.transition_entries(increment, state, bridge)
            )
            return np.array(
                [
                    [diagonal_zero, bridge, bridge],
                    [bridge, diagonal_nonzero, off_diagonal],
                    [bridge, off_diagonal, diagonal_nonzero],
                ]
            )

        layout = bulk.Layout(3)
        point = np.zeros(layout.size)
        point[layout.mu] = composition(message_weight)
        point[layout.beta] = composition(middle_weight)
        point[layout.zeta] = composition(output_weight)
        point[layout.tau1] = transition(
            message_weight, middle_weight, first_bridge
        ).reshape(-1)
        point[layout.tau2] = transition(
            middle_weight, output_weight, second_bridge
        ).reshape(-1)

        equality, target, _, _ = bulk.constraints(layout, 0.01, 0.25)
        np.testing.assert_allclose(equality @ point, target, atol=1e-13)
        self.assertAlmostEqual(
            bulk.exponent(point, layout, repetition),
            ternary.exponent(
                message_weight, middle_weight, output_weight, repetition
            ),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
