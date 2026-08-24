from __future__ import annotations

import unittest
from fractions import Fraction
from types import SimpleNamespace

import verify_one_orbit_toll as toll


class OneOrbitTollTests(unittest.TestCase):
    def test_binary_benchmark_margin_is_exactly_positive(self) -> None:
        result = toll.verify(
            SimpleNamespace(
                eta=Fraction(1, 500),
                support_upper_bound=Fraction(1, 10_000),
                precision=60,
            )
        )
        self.assertEqual(
            result["status"], "verified negative one-sign-orbit exponent"
        )
        self.assertGreater(result["negative_exponent_margin"], 0.00013)


if __name__ == "__main__":
    unittest.main()
