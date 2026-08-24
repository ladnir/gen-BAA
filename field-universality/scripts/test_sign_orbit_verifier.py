from __future__ import annotations

import math
import unittest
from fractions import Fraction

import numpy as np
from mpmath import iv
from scipy.optimize import linprog

from verify_sign_orbit_bulk import (
    Box,
    ONE,
    ZERO,
    exact_dual_upper_bound,
    log_upper,
    make_layout,
    negative_log_upper,
    orbit_automorphisms,
    phase_one_reference,
    secant_upper,
    structural_equalities,
    xlogx_upper,
)


class SignOrbitVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        iv.dps = 60

    def test_quinary_orbit_layout(self) -> None:
        layout = make_layout(5)
        self.assertEqual(layout.orbits, ((0,), (1, 4), (2, 3)))
        self.assertEqual(len(layout.triples), 13)

    def test_quinary_sign_symmetric_layout(self) -> None:
        layout = make_layout(5, sign_symmetric=True)
        self.assertTrue(layout.sign_symmetric)
        self.assertEqual(len(layout.pair_representatives), 13)
        self.assertEqual(layout.pair_representatives[0], (0, 0))
        represented = set()
        for source, target in layout.pair_representatives:
            represented.add((source, target))
            represented.add(((-source) % 5, (-target) % 5))
        self.assertEqual(len(represented), 25)

    def test_septic_orbit_automorphisms(self) -> None:
        layout = make_layout(7)
        self.assertEqual(len(layout.orbits), 4)
        automorphisms = orbit_automorphisms(layout)
        self.assertEqual(len(automorphisms), 3)
        self.assertTrue(all(permutation[0] == 0 for permutation in automorphisms))
        self.assertEqual(
            {tuple(sorted(permutation[1:])) for permutation in automorphisms},
            {(1, 2, 3)},
        )

    def test_directed_logarithm_bounds(self) -> None:
        self.assertGreater(log_upper(Fraction(2)), Fraction(69, 100))
        self.assertLess(log_upper(Fraction(2)), Fraction(70, 100))
        self.assertEqual(
            negative_log_upper(Fraction(1, 2)), log_upper(Fraction(2))
        )
        self.assertGreater(xlogx_upper(Fraction(1, 2)), Fraction(-35, 100))
        self.assertLess(xlogx_upper(Fraction(1, 2)), Fraction(-34, 100))

    def test_secant_is_an_upper_bound(self) -> None:
        lower = Fraction(1, 10)
        upper = Fraction(9, 10)
        slope, intercept = secant_upper(lower, upper)
        for numerator in range(1, 10):
            point = Fraction(numerator, 10)
            line = float(slope * point + intercept)
            actual = float(point) * math.log(float(point))
            self.assertGreaterEqual(line, actual)

    def test_exact_dual_repairs_a_floating_witness(self) -> None:
        objective = (Fraction(1),)
        upper_rows = ((Fraction(1),),)
        upper_targets = (Fraction(1, 2),)
        bounds = ((ZERO, Fraction(1)),)
        result = linprog(
            np.array([-1.0]),
            A_ub=np.array([[1.0]]),
            b_ub=np.array([0.5]),
            bounds=[(0.0, 1.0)],
            method="highs",
        )
        self.assertTrue(result.success)
        upper = exact_dual_upper_bound(
            objective,
            ZERO,
            upper_rows,
            upper_targets,
            (),
            (),
            bounds,
            result,
        )
        self.assertGreaterEqual(upper, Fraction(1, 2))
        self.assertLess(upper, Fraction(500001, 1_000_000))

    def test_phase_one_certifies_an_infeasible_box(self) -> None:
        layout = make_layout(5)
        equality, target = structural_equalities(layout)
        lower = [ZERO] * (3 * layout.orbit_count)
        upper = [ONE] * (3 * layout.orbit_count)
        lower[: layout.orbit_count] = [ONE] * layout.orbit_count
        point, certified_infeasible = phase_one_reference(
            Box(tuple(lower), tuple(upper)), layout, equality, target
        )
        self.assertIsNone(point)
        self.assertTrue(certified_infeasible)


if __name__ == "__main__":
    unittest.main()
