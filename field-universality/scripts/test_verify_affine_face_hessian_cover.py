#!/usr/bin/env python3
"""Tests for exact feasible-slice cover reconstruction."""

from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_affine_face_interval import FaceBox, tighten_simplex_box
from verify_affine_face_hessian_cover import verify_feasible_guillotine_cover


ZERO = Fraction(0)
ONE = Fraction(1)


def unit_simplex_box() -> FaceBox:
    return FaceBox((ZERO,) * 15, (ONE,) * 15)


class FeasibleCoverTests(unittest.TestCase):
    variables = [index for index in range(15) if index not in (4, 9, 14)]
    dependent = [4, 9, 14]

    def test_accepts_tightened_binary_split(self):
        parent = tighten_simplex_box(
            unit_simplex_box(), self.variables, self.dependent
        )
        self.assertIsNotNone(parent)
        left_upper = list(parent.upper)
        left_upper[0] = Fraction(1, 2)
        right_lower = list(parent.lower)
        right_lower[0] = Fraction(1, 2)
        leaves = [
            tighten_simplex_box(
                FaceBox(parent.lower, tuple(left_upper)),
                self.variables,
                self.dependent,
            ),
            tighten_simplex_box(
                FaceBox(tuple(right_lower), parent.upper),
                self.variables,
                self.dependent,
            ),
        ]
        self.assertTrue(all(leaf is not None for leaf in leaves))
        verify_feasible_guillotine_cover(
            parent, list(leaves), self.variables, self.dependent
        )

    def test_rejects_feasible_gap(self):
        parent = tighten_simplex_box(
            unit_simplex_box(), self.variables, self.dependent
        )
        left_upper = list(parent.upper)
        left_upper[0] = Fraction(1, 3)
        left = tighten_simplex_box(
            FaceBox(parent.lower, tuple(left_upper)),
            self.variables,
            self.dependent,
        )
        with self.assertRaises(ValueError):
            verify_feasible_guillotine_cover(
                parent, [left], self.variables, self.dependent
            )


if __name__ == "__main__":
    unittest.main()
