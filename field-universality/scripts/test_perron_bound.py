#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import optimize_bulk_exponent as bulk
import search_internal_information_gap as internal_gap
from explore_perron_bound import (
    CURRENT_OBSTRUCTION_BOX,
    common_mixture_bound,
    composition_vertices,
    entropy,
    secant_x_log_x,
    scale_symbol_vector,
    split_composition_box,
    spectral_data,
    support_adjustment,
    tighten_composition_box,
)
from generate_perron_point_certificate import exact_type


class PerronBoundTests(unittest.TestCase):
    def test_spectral_flow_has_requested_marginals(self) -> None:
        log_increment = np.array([0.0, 0.2, -0.4, 0.1, -0.3])
        log_state = np.array([0.0, -0.1, 0.3, -0.2, 0.4])
        data = spectral_data(log_increment, log_state)
        self.assertAlmostEqual(float(data.flow.sum()), 1.0, places=12)
        np.testing.assert_allclose(
            data.flow.sum(axis=0), data.flow.sum(axis=1), atol=1e-12
        )
        exact = entropy(data.flow.reshape(-1)) - entropy(data.state_marginal)
        dual = (
            data.log_radius
            - float(np.dot(data.increment_marginal, log_increment))
            - float(np.dot(data.state_marginal, log_state))
        )
        self.assertAlmostEqual(exact, dual, places=12)

    def test_field_scaling_preserves_spectral_radius(self) -> None:
        increment = np.array([0.0, 0.2, -0.4, 0.1, -0.3])
        state = np.array([0.0, -0.1, 0.3, -0.2, 0.4])
        reference = spectral_data(increment, state).log_radius
        for multiplier in range(1, 5):
            scaled = spectral_data(
                scale_symbol_vector(increment, multiplier),
                scale_symbol_vector(state, multiplier),
            ).log_radius
            self.assertAlmostEqual(reference, scaled, places=12)

    def test_composition_vertex_enumeration(self) -> None:
        vertices = composition_vertices(
            (0.01, 0.0, 0.0, 0.0),
            (0.2, 0.2, 0.2, 0.2),
            0.1,
            0.3,
            canonical_message=True,
        )
        self.assertGreater(len(vertices), 0)
        for vertex in vertices:
            self.assertAlmostEqual(float(vertex.sum()), 1.0, places=12)
            self.assertGreaterEqual(vertex[1] + 1e-12, max(vertex[2:]))
            self.assertGreaterEqual(1.0 - vertex[0], 0.1 - 1e-12)
            self.assertLessEqual(1.0 - vertex[0], 0.3 + 1e-12)

    def test_common_mixture_respects_quantifier_order(self) -> None:
        values = np.array([[0.8, -0.2], [-0.2, 0.8]])
        upper, weights = common_mixture_bound(values)
        self.assertAlmostEqual(upper, 0.3, places=12)
        np.testing.assert_allclose(weights, np.array([0.5, 0.5]), atol=1e-12)

    def test_composition_split_covers_parent(self) -> None:
        left, right = split_composition_box(CURRENT_OBSTRUCTION_BOX, 3)
        self.assertEqual(left.lower, CURRENT_OBSTRUCTION_BOX.lower)
        self.assertEqual(right.upper, CURRENT_OBSTRUCTION_BOX.upper)
        self.assertEqual(left.upper[3], right.lower[3])
        low_support, high_support = split_composition_box(
            CURRENT_OBSTRUCTION_BOX, 8
        )
        self.assertEqual(
            low_support.support_upper[0], high_support.support_lower[0]
        )

    def test_entropy_secant_is_an_upper_bound(self) -> None:
        slope, intercept = secant_x_log_x(0.0, 0.2)
        for value in np.linspace(0.0, 0.2, 101):
            exact = 0.0 if value == 0.0 else value * np.log(value)
            self.assertLessEqual(exact, slope * value + intercept + 1e-15)

    def test_tightening_propagates_support_and_coordinate_bounds(self) -> None:
        box = CURRENT_OBSTRUCTION_BOX
        left, _ = split_composition_box(box, 0)
        tightened = tighten_composition_box(left)
        self.assertIsNotNone(tightened)
        assert tightened is not None
        self.assertLessEqual(
            sum(tightened.lower[:4]), tightened.support_upper[0] + 1e-12
        )
        self.assertGreaterEqual(
            sum(tightened.upper[:4]), tightened.support_lower[0] - 1e-12
        )
        for index in range(4):
            self.assertGreaterEqual(tightened.lower[index], left.lower[index])
            self.assertLessEqual(tightened.upper[index], left.upper[index])

    def test_lagrangian_support_identity(self) -> None:
        raw = exact_type()
        layout = bulk.Layout(5)
        candidate = np.zeros(layout.size)
        candidate[layout.mu] = [float(value) for value in raw["message"]]
        candidate[layout.beta] = [float(value) for value in raw["middle"]]
        candidate[layout.zeta] = [float(value) for value in raw["output"]]
        candidate[layout.tau1] = [
            float(value) for row in raw["first_transition"] for value in row
        ]
        candidate[layout.tau2] = [
            float(value) for row in raw["second_transition"] for value in row
        ]
        exponent = bulk.exponent(candidate, layout, 3)
        gap = internal_gap.gap_and_gradient(candidate, layout, 3)[0]
        multiplier = 1.5
        lhs = exponent - multiplier * gap
        rhs = (
            (1.0 + multiplier) * exponent
            + multiplier * support_adjustment(candidate, layout)
        )
        self.assertAlmostEqual(lhs, rhs, places=12)


if __name__ == "__main__":
    unittest.main()
