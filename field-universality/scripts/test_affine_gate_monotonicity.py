import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_affine_gate_concavity import simplex_tangent_basis_for
from diagnose_affine_gate_monotonicity import random_product_points, tangent_directions


class AffineGateMonotonicityTest(unittest.TestCase):
    def test_random_product_points_preserve_block_sums(self):
        vertices = np.eye(5)
        points = random_product_points(
            [vertices, vertices, vertices], 20, np.random.default_rng(3)
        )
        np.testing.assert_allclose(points[:, 0:5].sum(axis=1), 1.0)
        np.testing.assert_allclose(points[:, 5:10].sum(axis=1), 1.0)
        np.testing.assert_allclose(points[:, 10:15].sum(axis=1), 1.0)

    def test_custom_dependent_coordinate_basis(self):
        basis, directions = simplex_tangent_basis_for([0, 5, 12])
        self.assertEqual(directions[-4:], [10, 11, 13, 14])
        q14_column = directions.index(14)
        expected = np.zeros(15)
        expected[14] = 1.0
        expected[12] = -1.0
        np.testing.assert_array_equal(basis[:, q14_column], expected)
        np.testing.assert_allclose(basis.sum(axis=0), 0.0)

    def test_tangent_direction_labels_follow_dependent_coordinates(self):
        self.assertEqual(
            tangent_directions([0, 5, 12]),
            [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14],
        )


if __name__ == "__main__":
    unittest.main()
