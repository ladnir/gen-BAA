import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rationalize_affine_cover import (
    balanced_round_column,
    effective_feature_coordinates,
    effective_gate_reference,
    exact_minimum_weight,
    exact_simplex_vertices,
)
from verify_rational_affine_cover import VerificationError, verify_partition_sums


class RationalizeAffineCoverTest(unittest.TestCase):
    def test_balanced_rounding_preserves_sum(self):
        rounded = balanced_round_column(np.asarray([0.2, 0.3, 0.5]), 100, 100)
        self.assertEqual(sum(rounded), 100)

    def test_exact_boxed_simplex_vertices(self):
        lower = [Fraction(0)] * 5
        upper = [Fraction(1)] * 5
        vertices = exact_simplex_vertices(lower, upper, 0)
        self.assertEqual(len(vertices), 5)
        self.assertTrue(all(sum(vertex) == 1 for vertex in vertices))

    def test_exact_minimum_uses_blockwise_vertices(self):
        unit_vertices = exact_simplex_vertices([Fraction(0)] * 5, [Fraction(1)] * 5, 0)
        minimum = exact_minimum_weight(
            [[100, 10, 20], [0, -10, -20]],
            100,
            [unit_vertices, unit_vertices, unit_vertices],
            [1, 6],
        )
        self.assertEqual(minimum, Fraction(-3, 10))

    def test_partition_sum_checker_rejects_corruption(self):
        with self.assertRaisesRegex(VerificationError, "column 0"):
            verify_partition_sums([[50, 10], [49, -10]], 100, 2)

    def test_result_override_uses_results_child_gate(self):
        leaf = {"gate": "results/parent.json"}
        result = {"gate": "results/child.json"}
        self.assertEqual(
            effective_gate_reference(leaf, result, result_override=True),
            "results/child.json",
        )

    def test_result_override_requires_gate_provenance(self):
        with self.assertRaisesRegex(ValueError, "does not identify its gate"):
            effective_gate_reference(
                {"gate": "results/parent.json"}, {}, result_override=True
            )

    def test_result_feature_basis_overrides_descriptor_basis(self):
        gate = {"affine_gating": {"feature_coordinates": [1, 2, 3]}}
        self.assertEqual(
            effective_feature_coordinates(
                {"feature_coordinates": [6, 7, 11, 14]}, gate
            ),
            [6, 7, 11, 14],
        )


if __name__ == "__main__":
    unittest.main()
