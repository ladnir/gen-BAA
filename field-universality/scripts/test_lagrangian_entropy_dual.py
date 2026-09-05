#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from diagnose_lagrangian_entropy_dual import (
    grouped_entropy,
    grouped_pressure_distribution,
    increment_marginal,
)


class LagrangianEntropyDualTests(unittest.TestCase):
    def test_grouped_conjugate_is_tight_at_its_induced_distribution(self) -> None:
        row = np.array([0.0, 0.2, -0.1, 0.3, -0.4])
        column = np.array([0.0, -0.3, 0.4, -0.2, 0.1])
        increment = np.array([0.0, 0.5, -0.2, 0.1, -0.4])
        multiplier = 1.5
        log_partition, transition = grouped_pressure_distribution(
            row, column, increment, multiplier
        )
        state_row = transition.sum(axis=1)
        state_column = transition.sum(axis=0)
        observed_increment = increment_marginal(transition)
        primal = grouped_entropy(transition, multiplier)
        dual = (
            log_partition
            + float(np.dot(row, state_row))
            + float(np.dot(column, state_column))
            + float(np.dot(increment, observed_increment))
        )
        self.assertAlmostEqual(float(transition.sum()), 1.0, places=12)
        self.assertAlmostEqual(primal, dual, places=12)


if __name__ == "__main__":
    unittest.main()
