#!/usr/bin/env python3

from __future__ import annotations

import unittest
from fractions import Fraction

import verify_projected_frontier_cover_map as checker


class ProjectedFrontierCoverMapTests(unittest.TestCase):
    def test_box_containment(self) -> None:
        outer = ((Fraction(0),) * 15, (Fraction(1),) * 15)
        inner = ((Fraction(1, 4),) * 15, (Fraction(3, 4),) * 15)
        self.assertTrue(checker.contains(outer, inner))
        self.assertFalse(checker.contains(inner, outer))


if __name__ == "__main__":
    unittest.main()
