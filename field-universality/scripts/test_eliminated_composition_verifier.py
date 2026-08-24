from __future__ import annotations

import unittest
from fractions import Fraction

from mpmath import iv

from verify_conditional_product_bulk import default_cuts
from verify_eliminated_composition_bulk import (
    Box,
    box_bound,
    candidate_certificate,
)


class EliminatedCompositionVerifierTests(unittest.TestCase):
    def test_candidate_point_has_exactly_negative_envelope(self) -> None:
        iv.dps = 80
        point = tuple(
            Fraction(value)
            for value in (
                "0.002",
                "0.002813303962336616",
                "0.1323",
                "0.0009924297599807304",
                "0.0013463064097057693",
            )
        )
        box = Box(point, point)
        bound = box_bound(
            box,
            default_cuts(),
            (candidate_certificate(),),
            Fraction(1, 500),
            Fraction(1323, 10_000),
        )
        self.assertIsNotNone(bound)
        assert bound is not None
        self.assertLess(bound.value, Fraction(-1, 1_000))
        self.assertGreater(bound.value, Fraction(-1, 500))


if __name__ == "__main__":
    unittest.main()
