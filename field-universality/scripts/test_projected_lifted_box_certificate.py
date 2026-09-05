#!/usr/bin/env python3

from __future__ import annotations

import unittest
import copy
import json
from fractions import Fraction
from pathlib import Path

import verify_projected_lifted_box_certificate as checker
import verify_stationary_cover_certificate as cover


RESULTS = Path(__file__).resolve().parents[1] / "results"


class ProjectedLiftedBoxCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = json.loads(
            (RESULTS / "f5-projected-lifted-worst-local-0.22.json").read_text(
                encoding="utf-8"
            )
        )

    def test_independent_local_certificate_replays(self) -> None:
        result = checker.verify(copy.deepcopy(self.certificate), 100)
        self.assertTrue(result["box_upper_is_exactly_negative"])
        self.assertEqual(result["variables"], 138)

    def test_rejects_underestimated_projection_kernel(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["projection_tangents"][0]["kernel_upper"][0][0] = "0/1"
        with self.assertRaisesRegex(ValueError, "does not upper-bound exp"):
            checker.verify(tampered, 100)

    def test_rejects_missing_dual_multiplier(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["dual"].pop(next(iter(tampered["dual"])))
        with self.assertRaisesRegex(ValueError, "dual equality fails"):
            checker.verify(tampered, 100)

    def test_moment_index_is_symmetric_and_complete(self) -> None:
        indices = {
            checker.moment(left, right)
            for left in range(checker.COMPOSITIONS)
            for right in range(checker.COMPOSITIONS)
        }
        self.assertEqual(len(indices), 120)
        self.assertEqual(min(indices), checker.MOMENT_START)
        self.assertEqual(max(indices), checker.MOMENT_END - 1)
        self.assertEqual(checker.moment(2, 9), checker.moment(9, 2))

    def test_base_lifted_constraint_count(self) -> None:
        lower, upper = cover.root_box(Fraction(1, 500), Fraction(11, 50))
        entropy = ((Fraction(1, 5),) * 5,)
        # Constraint construction itself does not require a projection tangent.
        constraints = checker.build_constraints(lower, upper, entropy, (), (), 100)
        names = {constraint.name for constraint in constraints}
        self.assertEqual(len(names), len(constraints))
        self.assertIn("moment_simplex_2_14_lower", names)
        self.assertIn("rlt_canonical_message_4_14_upper", names)
        self.assertIn(f"variable_{checker.STAGE_TWO}_lower", names)
        diagonal = next(
            constraint
            for constraint in constraints
            if constraint.name == "moment_0_0_lower_ll"
        )
        self.assertEqual(diagonal.coefficients[0], 2 * lower[0])

    def test_covariance_cut_matches_square_identity(self) -> None:
        lower, upper = cover.root_box(Fraction(1, 500), Fraction(11, 50))
        direction = tuple(Fraction(index - 7, 11) for index in range(15))
        center = Fraction(3, 17)
        constraints = checker.build_constraints(
            lower,
            upper,
            ((Fraction(1, 5),) * 5,),
            (),
            ((direction, center),),
            100,
        )
        cut = next(value for value in constraints if value.name == "covariance_0")
        point = tuple(Fraction(index + 1, 120) for index in range(15))
        moments = {
            checker.moment(i, j): point[i] * point[j]
            for i in range(15)
            for j in range(i, 15)
        }
        values = {**{index: point[index] for index in range(15)}, **moments}
        lhs = sum(coefficient * values[index] for index, coefficient in cut.coefficients.items())
        self.assertLessEqual(lhs, cut.target)
        square = (sum(v * x for v, x in zip(direction, point, strict=True)) - center) ** 2
        self.assertEqual(cut.target - lhs, square)


if __name__ == "__main__":
    unittest.main()
