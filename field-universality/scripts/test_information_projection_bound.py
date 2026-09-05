#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import diagnose_fourier_information_bound as projection
import search_internal_information_gap as information


class InformationProjectionBoundTests(unittest.TestCase):
    def test_support_cell_sum_is_below_information_excess(self) -> None:
        random = np.random.default_rng(7)
        for _ in range(20):
            transition = random.dirichlet(np.ones(25)).reshape((5, 5))
            increment = np.array(
                [
                    sum(
                        transition[source, (source + difference) % 5]
                        for source in range(5)
                    )
                    for difference in range(5)
                ]
            )
            state = transition.sum(axis=1)
            full, support, _ = information.transition_information(transition)
            lower = projection.support_cell_sum_lower(
                transition, increment, state
            )
            direct = projection.direct_support_sum_lower(
                transition, increment, state
            )
            self.assertLessEqual(lower, full - support + 1e-12)
            self.assertAlmostEqual(direct, support + lower, places=12)
            self.assertLessEqual(direct, full + 1e-12)


if __name__ == "__main__":
    unittest.main()
