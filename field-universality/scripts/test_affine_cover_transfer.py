import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_affine_cover_transfer import coefficient_array, induced_leaves


class AffineCoverTransferTest(unittest.TestCase):
    def test_decodes_legacy_coefficients(self):
        value = coefficient_array(["1.0 2.0", "3.0 4.0"])
        np.testing.assert_array_equal(value, [[1.0, 2.0], [3.0, 4.0]])

    def test_routes_boundary_once(self):
        tree = {
            "split": {"dimension": 0, "threshold": "1/2"},
            "le": {"leaf": "le"},
            "ge": {"leaf": "ge"},
        }
        lower = np.asarray([0.5])
        upper = np.asarray([0.75])
        leaves = induced_leaves(tree, lower, upper)
        self.assertEqual([leaf[0]["leaf"] for leaf in leaves], ["ge"])

    def test_splits_straddling_interval(self):
        tree = {
            "split": {"dimension": 0, "threshold": "1/2"},
            "le": {"leaf": "le"},
            "ge": {"leaf": "ge"},
        }
        leaves = induced_leaves(tree, np.asarray([0.25]), np.asarray([0.75]))
        self.assertEqual([leaf[0]["leaf"] for leaf in leaves], ["le", "ge"])
        self.assertEqual(leaves[0][2][0], 0.5)
        self.assertEqual(leaves[1][1][0], 0.5)


if __name__ == "__main__":
    unittest.main()
