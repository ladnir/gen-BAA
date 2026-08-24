from __future__ import annotations

import math
import unittest
from fractions import Fraction
from types import SimpleNamespace

from mpmath import iv

import verify_support_indicator_bulk as support


class SupportIndicatorVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        iv.dps = 60

    def test_cross_entropy_cut_majorizes_joint_entropy(self) -> None:
        reference_a = Fraction(1, 1000)
        reference_b = Fraction(3, 2000)
        constant, coefficient_a, coefficient_b = (
            support.joint_cross_entropy_coefficients(reference_a, reference_b)
        )
        for a, b in (
            (Fraction(1, 2000), Fraction(1, 1000)),
            (Fraction(1, 1000), Fraction(3, 2000)),
            (Fraction(1, 500), Fraction(1, 250)),
        ):
            atoms = (
                1.0 - float(b) - float(a) / 2.0,
                float(b) - float(a) / 2.0,
                float(a) / 2.0,
                float(a) / 2.0,
            )
            entropy = -sum(value * math.log(value) for value in atoms if value)
            cut = float(constant + coefficient_a * a + coefficient_b * b)
            self.assertGreaterEqual(cut, entropy)

    def test_default_quinary_slab_closes_exactly(self) -> None:
        args = SimpleNamespace(
            prime=5,
            repetition=3,
            delta=Fraction(1323, 10_000),
            eta=Fraction(1, 2000),
            cutoff=Fraction(1, 500),
            precision=60,
            max_boxes=4000,
        )
        result = support.search(args)
        self.assertEqual(
            result["status"], "verified negative support-indicator cover"
        )
        self.assertTrue(result["upper_bound_is_exactly_negative"])
        self.assertLess(result["boxes_split"], args.max_boxes)

    def test_binary_support_target_closes_exactly(self) -> None:
        args = SimpleNamespace(
            prime=5,
            repetition=3,
            delta=Fraction(1323, 10_000),
            eta=Fraction(1, 500),
            cutoff=Fraction(1323, 2500),
            precision=60,
            max_boxes=2000,
            internal_values_paid=True,
            target_upper_bound=Fraction(1, 10_000),
        )
        result = support.search(args)
        self.assertEqual(
            result["status"], "verified support-indicator target bound"
        )
        self.assertTrue(result["upper_bound_is_below_target"])
        self.assertLess(result["boxes_split"], args.max_boxes)


if __name__ == "__main__":
    unittest.main()
