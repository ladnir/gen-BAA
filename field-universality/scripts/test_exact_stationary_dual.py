#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import exact_stationary_dual as stationary


class ExactStationaryDualTests(unittest.TestCase):
    def test_fitted_cut_matches_projection_value(self) -> None:
        increment = np.array([0.71, 0.04, 0.10, 0.09, 0.06])
        state = np.array([0.68, 0.07, 0.08, 0.11, 0.06])
        cut = stationary.fit_stationary_dual(increment, state)
        self.assertTrue(cut.optimizer_success)
        self.assertLess(cut.equality_residual, 1e-8)
        self.assertAlmostEqual(
            stationary.fixed_cut_payment(increment, state, cut),
            cut.value,
            places=9,
        )

    def test_fixed_cut_is_global_lower_bound(self) -> None:
        increment = np.array([0.71, 0.04, 0.10, 0.09, 0.06])
        state = np.array([0.68, 0.07, 0.08, 0.11, 0.06])
        cut = stationary.fit_stationary_dual(increment, state)
        candidate_increment = np.array([0.73, 0.03, 0.08, 0.11, 0.05])
        candidate_state = np.array([0.69, 0.06, 0.09, 0.10, 0.06])
        candidate = stationary.fit_stationary_dual(
            candidate_increment, candidate_state
        )
        lower = stationary.fixed_cut_payment(
            candidate_increment, candidate_state, cut
        )
        self.assertLessEqual(lower, candidate.value + 1e-10)

    def test_cut_retains_replay_potentials(self) -> None:
        increment = np.array([0.71, 0.04, 0.10, 0.09, 0.06])
        state = np.array([0.68, 0.07, 0.08, 0.11, 0.06])
        cut = stationary.fit_stationary_dual(increment, state)
        self.assertIsNotNone(cut.potentials)
        self.assertIsNotNone(cut.shift)
        potentials = np.asarray(cut.potentials)
        shift = float(cut.shift)
        np.testing.assert_allclose(
            cut.moment_coefficients,
            potentials @ stationary.MOMENT_COEFFICIENTS,
        )
        np.testing.assert_allclose(
            cut.kernel,
            np.exp(
                np.maximum(potentials @ stationary.FEATURES - shift, -700.0)
            ).reshape((5, 5)),
        )

    def test_affine_pushforward_preserves_mass(self) -> None:
        values = np.array([0.31, 0.07, 0.13, 0.19, 0.30])
        transformed = stationary.affine_symbol_pushforward(values, 2, 3)
        self.assertAlmostEqual(float(transformed.sum()), 1.0)
        np.testing.assert_array_equal(
            transformed,
            np.array([values[1], values[4], values[2], values[0], values[3]]),
        )

    def test_transformed_cut_has_covariant_payment(self) -> None:
        increment = np.array([0.42, 0.08, 0.11, 0.17, 0.22])
        state = np.array([0.36, 0.14, 0.09, 0.23, 0.18])
        cut = stationary.fit_stationary_dual(increment, state)
        for multiplier in range(1, 5):
            for translation in range(5):
                for reverse in (False, True):
                    transformed = stationary.transform_stationary_cut(
                        cut, multiplier, translation, reverse
                    )
                    increment_multiplier = (
                        -multiplier if reverse else multiplier
                    ) % 5
                    expected = stationary.fixed_cut_payment(
                        stationary.affine_symbol_pushforward(
                            increment, increment_multiplier
                        ),
                        stationary.affine_symbol_pushforward(
                            state, multiplier, translation
                        ),
                        cut,
                    )
                    actual = stationary.fixed_cut_payment(
                        increment, state, transformed
                    )
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_full_cut_orbit_has_forty_elements(self) -> None:
        increment = np.array([0.42, 0.08, 0.11, 0.17, 0.22])
        state = np.array([0.36, 0.14, 0.09, 0.23, 0.18])
        cut = stationary.fit_stationary_dual(increment, state)
        self.assertEqual(len(stationary.stationary_cut_orbit(cut)), 40)
        self.assertEqual(
            len(
                stationary.stationary_cut_orbit(
                    cut, include_state_translations=False
                )
            ),
            8,
        )

    def test_uniform_state_has_zero_payment(self) -> None:
        increment = np.array([0.64, 0.06, 0.08, 0.12, 0.10])
        state = np.full(5, 0.2)
        cut = stationary.fit_stationary_dual(increment, state)
        self.assertLess(abs(cut.value), 1e-9)


if __name__ == "__main__":
    unittest.main()
