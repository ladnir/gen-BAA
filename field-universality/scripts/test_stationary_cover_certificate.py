#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest
from fractions import Fraction

import verify_stationary_cover_certificate as cover


class StationaryCoverCertificateTests(unittest.TestCase):
    def test_root_box_is_exact(self) -> None:
        lower, upper = cover.root_box(Fraction(1, 500), Fraction(11, 50))
        self.assertEqual(lower[0], Fraction(3, 25))
        self.assertEqual(upper[10], Fraction(1999, 2000))
        self.assertEqual(upper[11], Fraction(11, 50))

    def test_rejects_wrong_leaf_box_before_replay(self) -> None:
        leaf = {"box": {"lower": ["0/1"] * 15, "upper": ["1/1"] * 15}}
        root = cover.root_box(Fraction(1, 500), Fraction(11, 50))
        raw = {
            "schema": "f5-stationary-cover-v1",
            "prime": 5,
            "repetition": 3,
            "eta": "1/500",
            "delta": "11/50",
            "root_box": {
                "lower": [cover.text(x) for x in root[0]],
                "upper": [cover.text(x) for x in root[1]],
            },
            "tree": {"kind": "leaf", "certificate": leaf},
        }
        with self.assertRaisesRegex(ValueError, "unexpected box"):
            cover.verify(copy.deepcopy(raw), 100)


if __name__ == "__main__":
    unittest.main()
