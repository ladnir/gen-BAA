from __future__ import annotations

import unittest

import numpy as np

import optimize_conditional_product_bound as product
import optimize_scalar_tube_bound as scalar


class ScalarTubeBoundTests(unittest.TestCase):
    def test_message_cancellation_tradeoff_on_vertex_family(self) -> None:
        a = 0.001
        self.assertAlmostEqual(
            scalar.message_cancellation_tradeoff(a, a / 2.0),
            0.225 * a,
            places=15,
        )

    def test_coordinate_radius_relaxes_universal_cancellation(self) -> None:
        random = np.random.default_rng(8675309)
        for _ in range(100):
            increment_support = random.uniform(0.01, 0.3)
            state_support = random.uniform(increment_support / 2.0, 0.5)
            crossing = random.uniform(
                max(0.0, increment_support - state_support),
                increment_support / 2.0,
            )
            increment_nonzero = random.dirichlet(np.ones(4))
            state_nonzero = random.dirichlet(np.ones(4))
            increment = np.concatenate(
                (
                    np.array([1.0 - increment_support]),
                    increment_support * increment_nonzero,
                )
            )
            state = np.concatenate(
                (
                    np.array([1.0 - state_support]),
                    state_support * state_nonzero,
                )
            )
            radius = np.sqrt(
                0.75 * np.dot(state_nonzero - 0.25, state_nonzero - 0.25)
            )
            relaxed = scalar.scalar_cancellation_payment(
                increment_support, crossing, radius
            )
            universal = product.universal_cancellation_divergence(
                increment, state, crossing
            )
            self.assertLessEqual(relaxed, universal + 1e-14)

    def test_scalar_upper_dominates_full_composition_upper(self) -> None:
        layout = product.Layout(5)
        mu = np.array([0.98, 0.002, 0.004, 0.006, 0.008])
        beta = np.array([0.976, 0.003, 0.009, 0.005, 0.007])
        zeta = np.array([0.78, 0.04, 0.08, 0.06, 0.04])
        crossing1 = 0.006
        crossing2 = 0.008
        full_point = np.concatenate(
            (mu, beta, zeta, np.array([crossing1, crossing2]))
        )
        beta_nonzero = beta[1:] / beta[1:].sum()
        zeta_nonzero = zeta[1:] / zeta[1:].sum()
        z1 = np.sqrt(0.75 * np.dot(beta_nonzero - 0.25, beta_nonzero - 0.25))
        z2 = np.sqrt(0.75 * np.dot(zeta_nonzero - 0.25, zeta_nonzero - 0.25))
        scalar_point = np.array(
            [
                mu[1:].sum(),
                beta[1:].sum(),
                zeta[1:].sum(),
                crossing1,
                crossing2,
                z1,
                z2,
            ]
        )
        full_upper = product.upper_bound(
            full_point, layout, 3, "universal-cancellation"
        )
        scalar_upper = scalar.scalar_upper(scalar_point, 3)
        self.assertGreaterEqual(scalar_upper + 1e-14, full_upper)
        self.assertAlmostEqual(
            scalar.scalar_tube_defect(scalar_point),
            product.two_stage_zero_payment_defect(full_point, layout),
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
