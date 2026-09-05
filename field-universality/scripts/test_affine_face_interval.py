import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpmath import iv

from explore_affine_face_interval import (
    FaceBox,
    AffineInterval,
    FastAffine,
    ThirdJet,
    exact_ldl_pivots,
    lower_float,
    third_exp,
    third_log,
    fast_affine_exp,
    fast_affine_log,
    face_affine_directional_duals,
    apply_face,
    tighten_simplex_box,
    upper_float,
)


class AffineFaceIntervalTest(unittest.TestCase):
    def assert_interval_almost_equal(self, value, expected):
        self.assertAlmostEqual(lower_float(value), expected, places=13)
        self.assertAlmostEqual(upper_float(value), expected, places=13)

    def test_tightens_dependent_simplex_coordinate(self):
        lower = [Fraction(0)] * 15
        upper = [Fraction(1)] * 15
        for start in (0, 5, 10):
            for index in range(start + 1, start + 5):
                upper[index] = Fraction(1, 4)
        box = tighten_simplex_box(
            FaceBox(tuple(lower), tuple(upper)),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14],
            [0, 5, 10],
        )
        self.assertIsNotNone(box)
        self.assertEqual(box.upper[0], 1)

    def test_rejects_infeasible_simplex(self):
        lower = [Fraction(0)] * 15
        upper = [Fraction(1)] * 15
        lower[1:5] = [Fraction(3, 10)] * 4
        box = tighten_simplex_box(
            FaceBox(tuple(lower), tuple(upper)),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14],
            [0, 5, 10],
        )
        self.assertIsNone(box)

    def test_third_jet_elementary_derivatives(self):
        zero = iv.mpf(0)
        one = iv.mpf(1)
        x = ThirdJet(iv.mpf(2), [one], [[zero]], [[[zero]]])
        logarithm = third_log(x)
        self.assert_interval_almost_equal(logarithm.gradient[0], 0.5)
        self.assert_interval_almost_equal(logarithm.hessian[0][0], -0.25)
        self.assert_interval_almost_equal(logarithm.third[0][0][0], 0.25)

        y = ThirdJet(zero, [one], [[zero]], [[[zero]]])
        exponential = third_exp(y)
        self.assert_interval_almost_equal(exponential.gradient[0], 1.0)
        self.assert_interval_almost_equal(exponential.hessian[0][0], 1.0)
        self.assert_interval_almost_equal(exponential.third[0][0][0], 1.0)

        cube = x * x * x
        self.assert_interval_almost_equal(cube.third[0][0][0], 6.0)

    def test_exact_ldl_positive_definiteness(self):
        pivots = exact_ldl_pivots(
            [
                [Fraction(2), Fraction(1)],
                [Fraction(1), Fraction(2)],
            ]
        )
        self.assertEqual(pivots, [Fraction(2), Fraction(3, 2)])
        self.assertIsNone(
            exact_ldl_pivots(
                [
                    [Fraction(1), Fraction(2)],
                    [Fraction(2), Fraction(1)],
                ]
            )
        )

    def test_affine_interval_preserves_linear_cancellation(self):
        x = AffineInterval.variable(iv.mpf(2), iv.mpf("0.1"), 0, 1)
        cancelled = x - x
        self.assert_interval_almost_equal(cancelled.interval(), 0.0)
        square = x * x
        self.assertLessEqual(lower_float(square.interval()), 1.9**2)
        self.assertGreaterEqual(upper_float(square.interval()), 2.1**2)

    def test_selected_direction_survives_endpoint_restriction(self):
        lower = [Fraction(0)] * 15
        upper = [Fraction(1)] * 15
        lower[0] = upper[0] = Fraction(1, 2)
        box = FaceBox(tuple(lower), tuple(upper))
        variables = [1, 2, 6, 7, 8, 9, 11, 13, 14]
        dependent = [3, 5, 12]
        point = face_affine_directional_duals(
            box, variables, dependent, 0, AffineInterval
        )
        self.assert_interval_almost_equal(point[0].directional.value.interval(), 1.0)
        self.assert_interval_almost_equal(point[3].directional.value.interval(), -1.0)
        for index in range(15):
            if index not in (0, 3):
                self.assert_interval_almost_equal(
                    point[index].directional.value.interval(), 0.0
                )

    def test_face_can_fix_an_exact_tightened_endpoint(self):
        lower = [Fraction(0)] * 15
        upper = [Fraction(1)] * 15
        face = {
            "fixed": [],
            "fixed_values": {"2": "1/4"},
            "dependent_coordinates": [0, 5, 10],
        }
        box, variables, dependent = apply_face(lower, upper, face)
        self.assertEqual(box.lower[2], Fraction(1, 4))
        self.assertEqual(box.upper[2], Fraction(1, 4))
        self.assertNotIn(2, variables)
        self.assertEqual(dependent, [0, 5, 10])

    def test_fast_affine_secant_forms_enclose_samples(self):
        x = FastAffine(2.0, [0.3, -0.2])
        y = FastAffine(-1.0, [0.1, 0.4])
        product = x * y
        exponential = fast_affine_exp(x)
        logarithm = fast_affine_log(x)
        reciprocal = x.reciprocal()
        for first in (-1.0, -0.4, 0.0, 0.7, 1.0):
            for second in (-1.0, -0.2, 0.0, 0.6, 1.0):
                x_value = 2.0 + 0.3 * first - 0.2 * second
                y_value = -1.0 + 0.1 * first + 0.4 * second
                for form, actual in (
                    (product, x_value * y_value),
                    (exponential, __import__("math").exp(x_value)),
                    (logarithm, __import__("math").log(x_value)),
                    (reciprocal, 1.0 / x_value),
                ):
                    self.assertLessEqual(form.lower(), actual)
                    self.assertGreaterEqual(form.upper(), actual)


if __name__ == "__main__":
    unittest.main()
