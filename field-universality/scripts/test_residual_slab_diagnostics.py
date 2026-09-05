#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import diagnose_joint_residual_slab_cover as joint_slab
import diagnose_transition_residual_slab as transition_slab


class ResidualSlabDiagnosticTests(unittest.TestCase):
    def test_joint_zero_residual_slab_tightens_crossing(self) -> None:
        lower = [0.0] * 17
        upper = [1.0] * 17
        lower[0] = upper[0] = 0.8
        lower[1] = upper[1] = 0.2
        lower[5] = upper[5] = 0.7
        lower[6] = upper[6] = 0.3
        lower[10] = upper[10] = 0.8
        lower[11] = upper[11] = 0.2
        upper[15] = 0.1
        upper[16] = 0.15
        box = joint_slab.joint.Box(
            tuple(lower),
            tuple(upper),
            (0.2, 0.3, 0.2),
            (0.2, 0.3, 0.2),
        )
        slab = joint_slab.ResidualSlab(0, 0, 1, 0.01)
        tightened = joint_slab.tighten(box, slab)
        self.assertIsNotNone(tightened)
        assert tightened is not None
        self.assertGreaterEqual(tightened.lower[15], 0.05 - 1e-14)
        residual = joint_slab.residual_interval(tightened, slab)
        self.assertGreaterEqual(residual[1], 0.01 - 1e-14)

    def test_transition_zero_slab_row_has_the_claimed_boundary(self) -> None:
        for sign, crossing in ((1, 0.05), (-1, 0.03)):
            rows: list[np.ndarray] = []
            targets: list[float] = []
            lower = np.zeros(transition_slab.SIZE)
            upper = np.ones(transition_slab.SIZE)
            slab = transition_slab.ResidualSlab(0, 0, sign, 0.01)
            transition_slab.add_residual_slab(
                rows, targets, slab, lower, upper
            )
            point = np.zeros(transition_slab.SIZE)
            point[1] = 0.2
            for target in range(1, transition_slab.PRIME):
                point[transition_slab.LAYOUT.edge(1, 0, target)] = (
                    crossing / 4.0
                )
            self.assertAlmostEqual(float(rows[0] @ point), targets[0], places=13)

    def test_mccormick_rows_accept_an_exact_product(self) -> None:
        rows: list[np.ndarray] = []
        targets: list[float] = []
        transition_slab.add_mccormick(
            rows,
            targets,
            2,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            (0.1, 0.7),
            (0.2, 0.8),
        )
        point = np.array([0.4, 0.6, 0.24])
        self.assertTrue(
            all(float(row @ point) <= target + 1e-14 for row, target in zip(rows, targets, strict=True))
        )


if __name__ == "__main__":
    unittest.main()
