from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from sparse_spacing_bound import (
    accumulate,
    binary_benchmark_sparse_certificate,
    conditional_weight_histogram,
    hamming_weight,
    hypergeometric_tail_count,
    laplace_polynomial,
    laplace_upper_bound,
    ladder_integer_check,
    least_ladder_denominator_root,
    nonzero_interval_count,
    spacing_tail_count,
    spacing_weight_count,
)


class SparseSpacingTests(unittest.TestCase):
    def test_accumulator_loses_at_most_half_the_weight(self) -> None:
        for prime in (2, 3, 5, 7):
            for length in range(1, 6):
                for word in itertools.product(range(prime), repeat=length):
                    self.assertLessEqual(
                        hamming_weight(word),
                        2 * hamming_weight(accumulate(word, prime)),
                    )

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
        self.assertLess(12, 29_997)
        ratio = 30_000
        self.assertLess(
            16 * ratio**2 * 12**3,
            (ratio - 3) ** 3,
        )

    def test_quinary_sparse_constant(self) -> None:
        self.assertLess(481, 22**2)
        self.assertLess(22, 299_997)
        ratio = 300_000
        self.assertLess(
            25 * ratio**2 * 22**3,
            (ratio - 3) ** 3,
        )

    def test_quinary_sparse_constant_at_larger_cutoff(self) -> None:
        self.assertLess(19, 5**2)
        self.assertLess(3, 29_997)
        ratio = 30_000
        self.assertLess(
            400 * ratio**2 * 3**3,
            (ratio - 3) ** 3,
        )

    def test_quinary_sparse_constant_at_one_twenty_fifth(self) -> None:
        # 49/18 < 11/4 certifies the series bound e < 11/4.
        self.assertLess(196, 198)
        self.assertLess(440**2 * 2**3, 117**3)

    def test_quinary_sparse_constant_at_one_twentieth(self) -> None:
        self.assertLess(19 * 8**2, 35**2)
        self.assertLess(196, 198)
        self.assertLess(660**2 * 7**3, 531**3)

    def test_septic_sparse_constant_at_nine_over_five_hundred(self) -> None:
        self.assertLess(4 * 4_419, 133**2)
        self.assertLess(196, 198)
        self.assertLess(264**2 * 133**3, 5_505**3)

    def test_quinary_sparse_constant_at_deterministic_cutoff(self) -> None:
        self.assertTrue(ladder_integer_check(5, 7))
        self.assertEqual(least_ladder_denominator_root(5), 7)

    def test_prime_ladder(self) -> None:
        expected = {2: 5, 3: 6, 5: 7, 7: 8, 11: 10, 13: 10, 101: 21}
        for prime, root in expected.items():
            self.assertEqual(least_ladder_denominator_root(prime), root)
            self.assertTrue(ladder_integer_check(prime, root))
            self.assertFalse(ladder_integer_check(prime, root - 1))

    def test_binary_benchmark_sparse_ladder(self) -> None:
        endpoints = {
            5: Fraction(1, 2000),
            7: Fraction(1, 5000),
            11: Fraction(1, 10_000),
            13: Fraction(1, 15_000),
            17: Fraction(1, 25_000),
        }
        for prime, eta in endpoints.items():
            certificate = binary_benchmark_sparse_certificate(prime, eta)
            self.assertTrue(certificate["sqrt_check"]["passes"])
            self.assertTrue(certificate["passes"])


if __name__ == "__main__":
    unittest.main()
