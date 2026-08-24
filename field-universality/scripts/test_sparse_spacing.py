from __future__ import annotations

import itertools
import math
import unittest

from sparse_spacing_bound import (
    conditional_weight_histogram,
    hypergeometric_tail_count,
    laplace_polynomial,
    laplace_upper_bound,
    nonzero_interval_count,
    spacing_tail_count,
    spacing_weight_count,
)


class SparseSpacingTests(unittest.TestCase):
    def test_spacing_tail_is_hypergeometric_tail(self) -> None:
        for length in range(2, 11):
            for transitions in range(1, length + 1):
                for active in range(1, transitions + 1):
                    for bound in range(length + 1):
                        self.assertEqual(
                            spacing_tail_count(length, transitions, active, bound),
                            hypergeometric_tail_count(
                                length, transitions, active, bound
                            ),
                        )

    def test_fixed_order_histogram_matches_spacing_formula(self) -> None:
        orders = ((1,), (1, 2), (1, 1, 1), (1, 2, 1, 2), (2, 1, 1, 2, 2))
        for order in orders:
            length = len(order) + 4
            active = nonzero_interval_count(order)
            histogram = conditional_weight_histogram(length, order)
            expected = {
                weight: spacing_weight_count(length, len(order), active, weight)
                for weight in range(length + 1)
                if spacing_weight_count(length, len(order), active, weight)
            }
            self.assertEqual(dict(histogram), expected)
            self.assertEqual(sum(histogram.values()), math.comb(length, len(order)))

    def test_nonzero_intervals_are_at_least_half(self) -> None:
        for transitions in range(1, 9):
            for order in itertools.product((1, 2), repeat=transitions):
                self.assertGreaterEqual(
                    nonzero_interval_count(order), (transitions + 1) // 2
                )

    def test_laplace_polynomial_bound(self) -> None:
        for length in range(3, 13):
            for transitions in range(1, length + 1):
                for active in range(1, transitions + 1):
                    self.assertLessEqual(
                        laplace_polynomial(length, transitions, active, 1, 2),
                        laplace_upper_bound(length, transitions, active, 1, 2),
                    )

    def test_rational_sparse_constant(self) -> None:
        ratio = 30_000
        self.assertLess(
            100 * ratio**2 * 51**3,
            512 * (ratio - 3) ** 3,
        )


if __name__ == "__main__":
    unittest.main()
