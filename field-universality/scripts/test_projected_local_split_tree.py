#!/usr/bin/env python3
"""Tests for local projected split-tree and cut-mixture helpers."""

from __future__ import annotations

import unittest

import numpy as np

import diagnose_concave_cut_mixture as mixture
import explore_projected_kl_box_cover as projected
import search_projected_local_split_tree as local_tree
import verify_projected_local_split_tree as verifier


class ProjectedLocalSplitTreeTests(unittest.TestCase):
    def test_box_encoding_round_trip(self) -> None:
        box = projected.Box((0.0, 0.2), (0.5, 0.8))
        decoded = local_tree.decode_box(local_tree.encode_box(box))
        self.assertEqual(decoded, box)
        self.assertTrue(verifier.boxes_close(decoded, box))

    def test_constant_kernel_has_zero_simplex_tangent_hessian(self) -> None:
        point = np.asarray(
            [
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.5,
                0.2,
                0.1,
                0.1,
                0.1,
                0.7,
                0.1,
                0.1,
                0.05,
                0.05,
            ]
        )
        hessian = mixture.cut_hessian(point, np.ones((5, 5)), 0)
        self.assertLess(float(np.max(np.abs(hessian))), 1e-12)

    def test_message_entropy_hessian_is_negative_semidefinite(self) -> None:
        point = np.full(15, 0.2)
        eigenvalues = np.linalg.eigvalsh(mixture.entropy_hessian(point))
        self.assertLessEqual(float(eigenvalues[-1]), 1e-12)
        self.assertEqual(int(np.sum(eigenvalues < -1e-6)), 4)


if __name__ == "__main__":
    unittest.main()
