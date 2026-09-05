from __future__ import annotations

import unittest

import numpy as np

import explore_projected_kl_box_cover as projected


class ProjectedResidualSlabTests(unittest.TestCase):
    @staticmethod
    def lifted_point(compositions: np.ndarray) -> np.ndarray:
        lifted = np.zeros(projected.MOMENT_END)
        lifted[: projected.COMPOSITION_SIZE] = compositions
        for left, right in projected.MOMENT_PAIRS:
            lifted[projected.moment_index(left, right)] = (
                compositions[left] * compositions[right]
            )
        return lifted

    def test_positive_nonzero_slab_implies_lifted_inequality(self) -> None:
        point = np.zeros(15)
        point[0:5] = [0.6, 0.1, 0.1, 0.1, 0.1]
        point[5:10] = [0.7, 0.15, 0.05, 0.05, 0.05]
        point[10] = 1.0
        row, target = projected.nonzero_residual_slab_inequality(
            0, 1, 1, 0.05
        )
        # With crossing x=0.1, r_1=(A-x) beta_1/B=0.15 and
        # r_1-A/5=0.07, so this point belongs to the positive slab.
        self.assertLessEqual(
            float(row @ self.lifted_point(point)), target + 1e-14
        )

    def test_negative_nonzero_slab_implies_lifted_inequality(self) -> None:
        point = np.zeros(15)
        point[0:5] = [0.6, 0.1, 0.1, 0.1, 0.1]
        point[5:10] = [0.7, 0.01, 0.09, 0.1, 0.1]
        point[10] = 1.0
        row, target = projected.nonzero_residual_slab_inequality(
            0, 1, -1, 0.05
        )
        # Crossing x=A/2 minimizes r_1.  It gives r_1=0.00666... and
        # residual below -0.05, hence the necessary inequality must hold.
        self.assertLessEqual(
            float(row @ self.lifted_point(point)), target + 1e-14
        )


if __name__ == "__main__":
    unittest.main()
