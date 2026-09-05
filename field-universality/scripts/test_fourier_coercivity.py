#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import optimize_conditional_product_bound as bound


class FourierCoercivityTests(unittest.TestCase):
    def test_bound_vanishes_on_structural_zero_set(self) -> None:
        increment = np.array([0.83, 0.01, 0.04, 0.07, 0.05])
        state = np.array([0.77, 0.0575, 0.0575, 0.0575, 0.0575])
        self.assertAlmostEqual(
            bound.fourier_coercive_divergence(increment, state, 0.17 / 5.0),
            0.0,
            places=15,
        )

    def test_bound_is_below_target_divergence(self) -> None:
        random = np.random.default_rng(91)
        for _ in range(200):
            increment_support = random.uniform(0.01, 0.35)
            state_support = random.uniform(
                increment_support / 2.0,
                min(0.7, 1.0 - increment_support / 10.0),
            )
            crossing_lower = max(0.0, increment_support - state_support)
            crossing_upper = min(increment_support / 2.0, 1.0 - state_support)
            if crossing_lower >= crossing_upper:
                continue
            crossing = random.uniform(crossing_lower, crossing_upper)
            increment_nonzero = random.dirichlet(np.ones(4))
            state_nonzero = random.dirichlet(np.ones(4))
            increment = np.concatenate(
                ([1.0 - increment_support], increment_support * increment_nonzero)
            )
            state = np.concatenate(
                ([1.0 - state_support], state_support * state_nonzero)
            )
            coercive = bound.fourier_coercive_divergence(
                increment, state, crossing
            )
            target = bound.projected_divergence(
                increment, state, crossing, "target"
            )
            self.assertLessEqual(coercive, target + 2e-12)
            target_l2 = bound.target_l2_divergence(
                increment, state, crossing
            )
            self.assertLessEqual(target_l2, target + 2e-12)
            self.assertGreaterEqual(target_l2 + 2e-12, coercive)


if __name__ == "__main__":
    unittest.main()
