#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import diagnose_pair_correlation_hessian as hessian
import diagnose_pair_correlation_spectrum as spectrum


class PairCorrelationHessianTests(unittest.TestCase):
    def test_direction_preserves_both_marginals(self) -> None:
        marginal = np.array([0.71, 0.04, 0.11, 0.09, 0.05])
        direction = hessian.correlation_direction(
            marginal, np.random.default_rng(17)
        )
        np.testing.assert_allclose(direction.sum(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(direction.sum(axis=1), 0.0, atol=1e-14)
        product = np.outer(marginal, marginal)
        self.assertLessEqual(np.max(np.abs(direction) / product), 1.0 + 1e-14)
        self.assertGreater(np.min(product - 0.02 * direction), 0.0)
        self.assertGreater(np.min(product + 0.02 * direction), 0.0)

    def test_full_basis_preserves_both_marginals(self) -> None:
        marginal = np.array([0.71, 0.04, 0.11, 0.09, 0.05])
        basis = spectrum.tangent_basis(marginal)
        self.assertEqual(basis.shape, (16, 25))
        self.assertEqual(np.linalg.matrix_rank(basis), 16)
        for direction in basis.reshape(16, 5, 5):
            np.testing.assert_allclose(direction.sum(axis=0), 0.0, atol=1e-14)
            np.testing.assert_allclose(direction.sum(axis=1), 0.0, atol=1e-14)

    def test_pair_constraint_moments_are_linear(self) -> None:
        increment = np.outer(
            np.array([0.7, 0.04, 0.1, 0.11, 0.05]),
            np.array([0.7, 0.04, 0.1, 0.11, 0.05]),
        ).ravel()
        state = np.outer(
            np.array([0.72, 0.03, 0.1, 0.1, 0.05]),
            np.array([0.72, 0.03, 0.1, 0.1, 0.05]),
        ).ravel()
        direction = spectrum.tangent_basis(
            np.array([0.7, 0.04, 0.1, 0.11, 0.05])
        )[0]
        zero = np.zeros(25)
        derivative = spectrum.moment_derivatives(
            increment, state, direction[None, :], zero[None, :]
        )[0]
        step = 0.013
        finite_difference = (
            spectrum.feature_moments(increment + step * direction, state)
            - spectrum.feature_moments(increment, state)
        ) / step
        np.testing.assert_allclose(derivative, finite_difference, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
