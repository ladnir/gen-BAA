#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import diagnose_pair_replica_payment as replica


class PairReplicaPaymentTests(unittest.TestCase):
    def test_uniform_pair_state_has_zero_payment(self) -> None:
        single_increment = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        single_state = np.full(5, 0.2)
        result = replica.pair_stationary_payment(
            np.outer(single_increment, single_increment).ravel(),
            np.outer(single_state, single_state).ravel(),
        )
        self.assertTrue(result["optimizer_success"])
        self.assertLess(abs(float(result["payment"])), 1e-9)
        self.assertLess(float(result["equality_residual"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
