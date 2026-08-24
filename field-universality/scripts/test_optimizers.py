#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import certify_symmetric_ternary as certificate
import diagnose_entropy_dual as entropy_dual
import optimize_asymmetric_bulk as asymmetric
import optimize_bulk_exponent as bulk
import optimize_symmetric_ternary as ternary
import search_entropy_dual_cover as dual_cover
import verify_symmetric_ternary as independent


class OptimizerTests(unittest.TestCase):
    def test_circulation_symmetrization_inequality(self) -> None:
        random = np.random.default_rng(19)
        sign = np.array([0, 2, 1])

        def asymmetry(distribution: np.ndarray, matrix: bool = False) -> float:
            flipped = (
                distribution[np.ix_(sign, sign)]
                if matrix
                else distribution[sign]
            )
            return bulk.entropy((distribution + flipped) / 2.0) - bulk.entropy(
                distribution
            )

        for _ in range(1000):
            zero, b, c, e, f, h, cycle = random.exponential(size=7)
            transition = np.array(
                [
                    [zero, c + cycle, b],
                    [c, e, h + cycle],
                    [b + cycle, h, f],
                ]
            )
            transition /= transition.sum()
            state = transition.sum(axis=1)
            increment = np.array(
                [
                    sum(
                        transition[source, (source + difference) % 3]
                        for source in range(3)
                    )
                    for difference in range(3)
                ]
            )
            gap = (
                asymmetry(transition, matrix=True)
                - asymmetry(increment)
                - asymmetry(state)
            )
            self.assertGreaterEqual(gap, -2e-15)

    def test_quinary_sign_reduction_counterexample(self) -> None:
        transition = np.array(
            [
                [6, 0, 0, 5, 0],
                [5, 6, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        transition /= 28.0
        state = transition.sum(axis=1)
        increment = np.array(
            [
                sum(
                    transition[source, (source + difference) % 5]
                    for source in range(5)
                )
                for difference in range(5)
            ]
        )
        np.testing.assert_allclose(transition.sum(axis=0), state)
        np.testing.assert_allclose(state, np.array([11, 11, 0, 6, 0]) / 28.0)
        np.testing.assert_allclose(increment, np.array([13, 0, 0, 10, 5]) / 28.0)

        sign = np.array([0, 4, 3, 2, 1])

        def asymmetry(distribution: np.ndarray, matrix: bool = False) -> float:
            flipped = (
                distribution[np.ix_(sign, sign)]
                if matrix
                else distribution[sign]
            )
            return bulk.entropy((distribution + flipped) / 2.0) - bulk.entropy(
                distribution
            )

        gap = (
            asymmetry(transition, matrix=True)
            - asymmetry(increment)
            - asymmetry(state)
        )
        self.assertAlmostEqual(gap, -5.0 * np.log(2.0) / 14.0, places=14)
        full_change = gap + asymmetry(increment) / 3.0
        self.assertAlmostEqual(
            full_change, -5.0 * np.log(2.0) / 28.0, places=14
        )

    def test_entropy_dual_majorizes_transition_entropy(self) -> None:
        transition = np.array(
            [
                [6, 0, 0, 5, 0],
                [5, 6, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        transition /= transition.sum()
        state = transition.sum(axis=1)
        increment = np.array(
            [
                sum(
                    transition[source, (source + difference) % 5]
                    for source in range(5)
                )
                for difference in range(5)
            ]
        )
        u = np.array([0.0, 0.2, -0.1, 0.4, -0.3])
        v = np.array([0.0, -0.2, 0.3, 0.1, -0.4])
        w = np.array([0.0, 0.5, -0.2, -0.1, 0.2])
        potentials = entropy_dual.potential_matrix(u, v, w)
        upper = (
            np.log(np.exp(-potentials).sum())
            + np.dot(u + v, state)
            + np.dot(w, increment)
        )
        self.assertLessEqual(bulk.entropy(transition), upper)

    def test_entropy_secants_are_upper_bounds(self) -> None:
        lower, upper = 0.03, 0.81
        slope, intercept = dual_cover.secant(lower, upper)
        binary_slope, binary_intercept = dual_cover.binary_secant(lower, upper)
        for point in np.linspace(lower, upper, 1001):
            self.assertLessEqual(
                dual_cover.xlogx(point), slope * point + intercept + 1e-15
            )
            self.assertLessEqual(
                dual_cover.binary_xlogx(point),
                binary_slope * point + binary_intercept + 1e-15,
            )

    def test_decimal_certificate_closes_below_binary_threshold(self) -> None:
        result = certificate.certify(
            certificate.Decimal("0.01"),
            certificate.Decimal("0.13"),
            precision=60,
            max_boxes=1000,
        )
        self.assertLess(result.upper_bound, 0)
        self.assertLess(result.boxes_split, 1000)

    def test_independent_interval_checker_closes(self) -> None:
        result = independent.verify(
            independent.Fraction("0.01"),
            independent.Fraction("0.13"),
            precision=50,
            max_boxes=1000,
        )
        self.assertLess(independent.mpf_cmp(result.upper_bound, independent.fzero), 0)
        self.assertLess(result.boxes_split, 1000)

    def test_prime_validation(self) -> None:
        self.assertTrue(bulk.is_prime(2))
        self.assertTrue(bulk.is_prime(5))
        self.assertFalse(bulk.is_prime(1))
        self.assertFalse(bulk.is_prime(9))

    def test_sign_symmetry_constraints_accept_uniform_types(self) -> None:
        layout = bulk.Layout(5)
        equality, target, _, _ = bulk.constraints(layout, 0.01, 0.13)
        symmetric_equality, symmetric_target = bulk.add_sign_symmetry(
            layout, equality, target
        )
        point = np.zeros(layout.size)
        point[layout.mu] = 1.0 / 5.0
        point[layout.beta] = 1.0 / 5.0
        point[layout.zeta] = 1.0 / 5.0
        point[layout.tau1] = 1.0 / 25.0
        point[layout.tau2] = 1.0 / 25.0
        np.testing.assert_allclose(
            symmetric_equality @ point, symmetric_target, atol=1e-14
        )

    def test_asymmetry_permutation_is_an_involution(self) -> None:
        layout = bulk.Layout(5)
        permutation = asymmetric.sign_permutation(layout)
        np.testing.assert_array_equal(permutation[permutation], np.arange(layout.size))
        uniform = np.zeros(layout.size)
        uniform[layout.mu] = 1.0 / 5.0
        uniform[layout.beta] = 1.0 / 5.0
        uniform[layout.zeta] = 1.0 / 5.0
        uniform[layout.tau1] = 1.0 / 25.0
        uniform[layout.tau2] = 1.0 / 25.0
        mask = asymmetric.energy_mask(layout, "all")
        self.assertEqual(
            asymmetric.asymmetry_energy(uniform, permutation, mask), 0.0
        )
        self.assertAlmostEqual(
            asymmetric.entropy_asymmetry(uniform, permutation, mask), 0.0
        )

    def test_bulk_objective_gradient(self) -> None:
        layout = bulk.Layout(3)
        random = np.random.default_rng(7)
        point = random.uniform(0.05, 0.95, layout.size)
        analytic = bulk.objective_gradient(point, layout, 3)
        numerical = np.zeros(layout.size)
        step = 1e-7
        for coordinate in range(layout.size):
            forward = point.copy()
            backward = point.copy()
            forward[coordinate] += step
            backward[coordinate] -= step
            numerical[coordinate] = (
                -bulk.exponent(forward, layout, 3) + bulk.exponent(backward, layout, 3)
            ) / (2.0 * step)
        self.assertLess(float(np.max(np.abs(analytic - numerical))), 2e-7)

    def test_symmetric_transition_has_requested_marginals(self) -> None:
        increment_weight = 0.18
        state_weight = 0.19
        bridge = 0.04
        diagonal_zero, bridge, diagonal_nonzero, off_diagonal = ternary.transition_entries(
            increment_weight, state_weight, bridge
        )
        transition = np.array(
            [
                [diagonal_zero, bridge, bridge],
                [bridge, diagonal_nonzero, off_diagonal],
                [bridge, off_diagonal, diagonal_nonzero],
            ]
        )
        state = np.array([1.0 - state_weight, state_weight / 2.0, state_weight / 2.0])
        increment = np.array(
            [1.0 - increment_weight, increment_weight / 2.0, increment_weight / 2.0]
        )
        np.testing.assert_allclose(transition.sum(axis=0), state, atol=1e-14)
        np.testing.assert_allclose(transition.sum(axis=1), state, atol=1e-14)
        observed_increment = np.array(
            [
                sum(transition[source, (source + difference) % 3] for source in range(3))
                for difference in range(3)
            ]
        )
        np.testing.assert_allclose(observed_increment, increment, atol=1e-14)

    def test_symmetric_reduction_matches_full_exponent(self) -> None:
        repetition = 3
        message_weight = 0.18
        middle_weight = 0.19
        output_weight = 0.20
        _, first_bridge = ternary.transition_rate(message_weight, middle_weight)
        _, second_bridge = ternary.transition_rate(middle_weight, output_weight)

        def composition(weight: float) -> np.ndarray:
            return np.array([1.0 - weight, weight / 2.0, weight / 2.0])

        def transition(increment: float, state: float, bridge: float) -> np.ndarray:
            diagonal_zero, bridge, diagonal_nonzero, off_diagonal = (
                ternary.transition_entries(increment, state, bridge)
            )
            return np.array(
                [
                    [diagonal_zero, bridge, bridge],
                    [bridge, diagonal_nonzero, off_diagonal],
                    [bridge, off_diagonal, diagonal_nonzero],
                ]
            )

        layout = bulk.Layout(3)
        point = np.zeros(layout.size)
        point[layout.mu] = composition(message_weight)
        point[layout.beta] = composition(middle_weight)
        point[layout.zeta] = composition(output_weight)
        point[layout.tau1] = transition(
            message_weight, middle_weight, first_bridge
        ).reshape(-1)
        point[layout.tau2] = transition(
            middle_weight, output_weight, second_bridge
        ).reshape(-1)

        equality, target, _, _ = bulk.constraints(layout, 0.01, 0.25)
        np.testing.assert_allclose(equality @ point, target, atol=1e-13)
        self.assertAlmostEqual(
            bulk.exponent(point, layout, repetition),
            ternary.exponent(
                message_weight, middle_weight, output_weight, repetition
            ),
            places=12,
        )

    def test_symmetric_search_checks_the_cutoff_face(self) -> None:
        args = SimpleNamespace(
            eta=0.01,
            delta=0.19,
            repetition=3,
            max_iterations=100,
        )
        result = ternary.search_branch(
            args,
            "cutoff_and_output_boundary",
            {0: args.eta, 2: args.delta},
            seed=3,
        )
        self.assertEqual(result.point[0], args.eta)
        self.assertEqual(result.point[2], args.delta)
        self.assertAlmostEqual(-result.objective, -0.00375433102347, places=10)

    def test_transition_tangent_is_an_upper_bound(self) -> None:
        decimal = certificate.Decimal
        tangent = certificate.transition_tangent(
            decimal("0.18"), decimal("0.19"), decimal("1e-12")
        )
        for increment, state, bridge in (
            ("0.18", "0.19", "0.04"),
            ("0.10", "0.20", "0.02"),
            ("0.30", "0.25", "0.05"),
        ):
            increment = decimal(increment)
            state = decimal(state)
            bridge = decimal(bridge)
            entries = certificate.transition_entries(increment, state, bridge)
            entropy = sum(
                multiplicity * certificate.entropy_atom(entry)
                for multiplicity, entry in zip(
                    (decimal(1), decimal(4), decimal(2), decimal(2)), entries
                )
            )
            upper = (
                tangent.intercept
                + tangent.increment_coefficient * increment
                + tangent.state_coefficient * state
                + tangent.bridge_coefficient * bridge
            )
            self.assertLessEqual(entropy, upper + decimal("1e-50"))


if __name__ == "__main__":
    unittest.main()
