#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import analyze_zero_payment_surface as zero


class ZeroPaymentSurfaceTests(unittest.TestCase):
    def test_support_cells_have_requested_marginals(self) -> None:
        cells = zero.support_cells(0.17, 0.23).reshape((2, 2))
        np.testing.assert_allclose(cells.sum(axis=1), np.array([0.83, 0.17]))
        np.testing.assert_allclose(cells.sum(axis=0), np.array([0.77, 0.23]))

    def test_output_derivative_is_positive_on_reduced_domain(self) -> None:
        for middle in np.linspace(0.002, 0.274, 40):
            lower = 4.0 * middle / 5.0 + 1e-8
            for output in np.linspace(lower, 0.22, 20):
                if output >= lower:
                    self.assertGreater(zero.output_derivative(middle, output), 0.0)

    def test_output_derivative_matches_finite_difference(self) -> None:
        message = 0.037
        middle = 0.051
        output = 0.19
        step = 1e-6
        finite_difference = (
            zero.zero_surface_value(message, middle, output + step)
            - zero.zero_surface_value(message, middle, output - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            finite_difference,
            zero.output_derivative(middle, output),
            places=9,
        )

    def test_box_upper_dominates_random_points(self) -> None:
        random = np.random.default_rng(31)
        for _ in range(40):
            a_lower = random.uniform(0.002, 0.20)
            a_upper = random.uniform(a_lower + 1e-4, min(0.343, a_lower + 0.08))
            b_lower = random.uniform(0.0016, 0.20)
            b_upper = random.uniform(b_lower + 1e-4, min(0.275, b_lower + 0.07))
            box = zero.Box(a_lower, a_upper, b_lower, b_upper)
            evaluated = zero.box_upper(box, 0.22)
            if evaluated is None:
                continue
            upper, _ = evaluated
            for _ in range(50):
                a = random.uniform(a_lower, a_upper)
                b = random.uniform(b_lower, b_upper)
                if b < 4.0 * a / 5.0:
                    continue
                self.assertLessEqual(
                    zero.zero_surface_value(a, b, 0.22), upper + 1e-11
                )

    def test_small_cover_closes_zero(self) -> None:
        result = zero.cover(0.002, 0.22, 0.0, 20_000)
        self.assertTrue(result["closed"])
        self.assertLessEqual(result["final_upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
