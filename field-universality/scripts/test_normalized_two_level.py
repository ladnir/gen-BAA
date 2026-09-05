from __future__ import annotations

import unittest

import numpy as np

import analyze_normalized_two_level as normalized
import optimize_conditional_product_bound as product


class NormalizedTwoLevelTests(unittest.TestCase):
    def test_scale_free_payment_matches_scaled_full_distribution(self) -> None:
        support_ratio = 3.0
        crossing_ratio = 0.23
        increment = np.array([0.1, 0.2, 0.3, 0.4])
        state = np.array([0.35, 0.15, 0.1, 0.4])
        scale = 0.2
        full_increment = np.concatenate(
            (np.array([1.0 - scale]), scale * increment)
        )
        full_state = np.concatenate(
            (
                np.array([1.0 - scale * support_ratio]),
                scale * support_ratio * state,
            )
        )
        full = product.two_level_divergence(
            full_increment,
            full_state,
            scale * crossing_ratio,
        )
        reduced = normalized.scale_free_payment(
            support_ratio, crossing_ratio, increment, state
        )
        self.assertAlmostEqual(full / scale, reduced, places=12)

    def test_explicit_family_formula_matches_general_payment(self) -> None:
        support_ratio = 10.0
        state = np.array([0.1, 0.4, 0.4, 0.1])
        general = normalized.normalized_ratio(
            support_ratio,
            0.2,
            normalized.UNIFORM,
            state,
        )
        explicit = normalized.large_ratio_family(support_ratio, state)
        self.assertAlmostEqual(general, explicit, places=12)

    def test_increment_normalization_decays_on_explicit_family(self) -> None:
        state = np.array([0.1, 0.4, 0.4, 0.1])
        ratio_10 = normalized.large_ratio_family(10.0, state)
        ratio_100 = normalized.large_ratio_family(100.0, state)
        ratio_1000 = normalized.large_ratio_family(1000.0, state)
        self.assertGreater(ratio_10, ratio_100)
        self.assertGreater(ratio_100, ratio_1000)
        self.assertLess(ratio_1000, 0.01)

    def test_candidate_projection_uses_the_larger_support(self) -> None:
        increment = np.array([0.9, 0.01, 0.02, 0.03, 0.04])
        state = np.array([0.7, 0.03, 0.06, 0.09, 0.12])
        crossing = 0.02
        defect = product.zero_payment_defect(increment, state, crossing)
        candidate = product.normalized_coercive_divergence(
            increment, state, crossing
        )
        self.assertAlmostEqual(candidate, 0.8 * defect / 0.3, places=14)
        proved = product.proved_residual_coercive_divergence(
            increment, state, crossing
        )
        self.assertAlmostEqual(proved, 0.2 * defect / 0.3, places=14)

    def test_universal_cancellation_is_below_every_increment_law(self) -> None:
        random = np.random.default_rng(20260827)
        for _ in range(100):
            increment_support = random.uniform(0.01, 0.4)
            state_support = random.uniform(increment_support / 2.0, 0.6)
            crossing_lower = max(0.0, increment_support - state_support)
            crossing = random.uniform(crossing_lower, increment_support / 2.0)
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
            universal = product.universal_cancellation_divergence(
                increment, state, crossing
            )
            actual = product.cancellation_divergence(
                increment, state, crossing
            )
            self.assertLessEqual(universal, actual + 1e-14)

    def test_residual_quadratic_is_below_normalized_payment(self) -> None:
        random = np.random.default_rng(314159)
        for _ in range(100):
            support_ratio = random.uniform(0.5, 10.0)
            crossing_lower = max(0.0, 1.0 - support_ratio)
            crossing = random.uniform(crossing_lower, 0.5)
            increment = random.dirichlet(np.ones(4))
            state = random.dirichlet(np.ones(4))
            defect = normalized.scale_free_defect(crossing, state)
            if defect <= 1e-12:
                continue
            payment_ratio = (
                max(1.0, support_ratio)
                * normalized.scale_free_payment(
                    support_ratio, crossing, increment, state
                )
                / defect
            )
            residual_ratio = normalized.residual_quadratic_ratio(
                crossing, increment, state
            )
            self.assertLessEqual(residual_ratio, payment_ratio + 1e-12)


if __name__ == "__main__":
    unittest.main()
