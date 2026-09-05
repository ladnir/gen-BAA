#!/usr/bin/env python3
"""Tests for exact cells carried through the floating split log."""

from __future__ import annotations

import unittest
from fractions import Fraction

import verify_projected_split_log as verifier


class ProjectedSplitLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = (
            tuple(Fraction(1, 4) for _ in range(15)),
            tuple(Fraction(3, 4) for _ in range(15)),
        )

    def test_split_inside_exact_cell(self) -> None:
        threshold = Fraction(1, 2)
        lower = verifier.split_exact_cell(self.parent, 3, threshold, True)
        upper = verifier.split_exact_cell(self.parent, 3, threshold, False)
        self.assertEqual(lower[1][3], threshold)
        self.assertEqual(upper[0][3], threshold)

    def test_split_above_exact_cell_carries_cell_left(self) -> None:
        threshold = Fraction(7, 8)
        self.assertIs(
            verifier.split_exact_cell(self.parent, 3, threshold, True),
            self.parent,
        )
        self.assertIsNone(verifier.split_exact_cell(self.parent, 3, threshold, False))

    def test_split_below_exact_cell_carries_cell_right(self) -> None:
        threshold = Fraction(1, 8)
        self.assertIsNone(verifier.split_exact_cell(self.parent, 3, threshold, True))
        self.assertIs(
            verifier.split_exact_cell(self.parent, 3, threshold, False),
            self.parent,
        )

    def test_empty_exact_cell_stays_empty(self) -> None:
        self.assertIsNone(
            verifier.split_exact_cell(None, 3, Fraction(1, 2), True)
        )


if __name__ == "__main__":
    unittest.main()
