#!/usr/bin/env python3
"""Tests for exact affine-gate split descriptors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_affine_gate_split import split_gate


class MakeAffineGateSplitTests(unittest.TestCase):
    def test_children_partition_at_exact_decimal(self) -> None:
        parent = {"status": "parent", "restrictions": ["6:ge:0.0275"]}
        lower, upper = split_gate(parent, 1, "0.0588359375")
        self.assertEqual(lower["restrictions"][-1], "1:le:0.0588359375")
        self.assertEqual(upper["restrictions"][-1], "1:ge:0.0588359375")
        self.assertEqual(parent["restrictions"], ["6:ge:0.0275"])

    def test_rejects_nonfinite_threshold(self) -> None:
        with self.assertRaises(ValueError):
            split_gate({}, 1, "NaN")

    def test_preserves_parent_fixed_face_metadata(self) -> None:
        parent = {
            "status": "parent",
            "restrictions": ["6:ge:0.0275"],
            "fixed": ["4:upper", "10:lower"],
        }
        lower, upper = split_gate(parent, 2, "0.0186747435836898")
        self.assertEqual(lower["fixed"], ["4:upper", "10:lower"])
        self.assertEqual(upper["fixed"], ["4:upper", "10:lower"])
        self.assertEqual(parent["restrictions"], ["6:ge:0.0275"])


if __name__ == "__main__":
    unittest.main()
