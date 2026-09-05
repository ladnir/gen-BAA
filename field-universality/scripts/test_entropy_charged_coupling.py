#!/usr/bin/env python3
"""Tests for the entropy-charged coupling diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

import diagnose_entropy_charged_coupling as charged


class EntropyChargedCouplingTests(unittest.TestCase):
    def test_charged_value_bounds_product_and_tightens(self) -> None:
        left = np.array([0.55, 0.2, 0.1, 0.1, 0.05])
        right = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
        kernel = np.arange(1.0, 26.0).reshape((5, 5)) / 10.0
        product_value = float(np.log(left @ kernel @ right))
        loose = charged.charged_partition_upper(left, right, kernel, 1.0)
        tight = charged.charged_partition_upper(left, right, kernel, 64.0)
        self.assertTrue(loose["optimizer_success"])
        self.assertTrue(tight["optimizer_success"])
        self.assertGreaterEqual(loose["upper"], product_value - 1e-10)
        self.assertGreaterEqual(tight["upper"], product_value - 1e-10)
        self.assertLessEqual(tight["upper"], loose["upper"] + 1e-10)
        self.assertLess(tight["marginal_residual"], 1e-10)


if __name__ == "__main__":
    unittest.main()
