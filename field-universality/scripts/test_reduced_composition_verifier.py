from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

import verify_conditional_product_bulk as dual
import verify_reduced_composition_bulk as reduced


class ReducedCompositionVerifierTests(unittest.TestCase):
    def test_message_conjugate_dominates_a_fixed_message_law(self) -> None:
        support_mu = dual.Interval(Fraction(1, 10), Fraction(1, 10))
        support_beta = dual.Interval(Fraction(3, 20), Fraction(3, 20))
        crossing = dual.Interval(Fraction(1, 40), Fraction(1, 40))
        mu_values = (
            Fraction(1, 10),
            Fraction(2, 10),
            Fraction(3, 10),
            Fraction(4, 10),
        )
        beta_values = (
            Fraction(4, 10),
            Fraction(3, 10),
            Fraction(2, 10),
            Fraction(1, 10),
        )
        mu = tuple(dual.Interval(value, value) for value in mu_values)
        beta = tuple(dual.Interval(value, value) for value in beta_values)
        cut = dual.default_cuts()[0]
        payment_lower = reduced.cut_lower_bound(
            cut,
            mu,
            beta,
            support_mu,
            support_beta,
            crossing,
        )
        direct_upper = (
            Fraction(1, 10) * dual.entropy_bounds(mu).upper / 3
            - payment_lower
        )
        eliminated_upper = reduced.message_eliminated_upper(
            cut,
            beta,
            support_mu,
            support_beta,
            crossing,
        )
        self.assertGreaterEqual(eliminated_upper, direct_upper)

    def test_r_parameterization_gives_exact_point_defect(self) -> None:
        support = Fraction(1, 50)
        crossing = Fraction(1, 100)
        conditional = tuple(
            dual.Interval(value, value)
            for value in (
                Fraction(1, 10),
                Fraction(2, 10),
                Fraction(3, 10),
                Fraction(4, 10),
            )
        )
        r = (crossing,) + tuple(
            (support - crossing) * value
            for value in (
                Fraction(1, 10),
                Fraction(2, 10),
                Fraction(3, 10),
                Fraction(4, 10),
            )
        )
        expected = sum(
            ((value - support / 5) ** 2 for value in r), Fraction(0)
        )
        actual = reduced.zero_payment_defect_lower_bound(
            dual.Interval(support, support),
            dual.Interval(crossing, crossing),
            conditional,
        )
        self.assertEqual(actual, expected)

    def test_tube_prunes_a_point_outside_its_exact_radius(self) -> None:
        values = (
            Fraction(1, 50),
            Fraction(1, 50),
            Fraction(11, 50),
            Fraction(1, 100),
            Fraction(1, 100),
        ) + (Fraction(1, 4),) * 9
        box = reduced.Box(values, values)
        defect = reduced.two_stage_zero_payment_defect_lower_bound(box)
        self.assertGreater(defect, Fraction(0))
        self.assertIsNone(
            reduced.tighten(
                box,
                Fraction(1, 500),
                Fraction(11, 50),
                defect / 2,
            )
        )

    def test_complement_prunes_a_point_on_the_zero_set(self) -> None:
        values = (
            Fraction(1, 50),
            Fraction(1, 50),
            Fraction(11, 50),
            Fraction(1, 250),
            Fraction(1, 250),
        ) + (Fraction(1, 4),) * 9
        box = reduced.Box(values, values)
        self.assertEqual(
            reduced.two_stage_zero_payment_defect_upper_bound(box),
            Fraction(0),
        )
        self.assertIsNone(
            reduced.tighten(
                box,
                Fraction(1, 500),
                Fraction(11, 50),
                minimum_zero_defect=Fraction(1, 1_000_000),
            )
        )

    def test_signed_residual_slab_prunes_and_accepts_exact_points(self) -> None:
        threshold = Fraction(1, 3200)
        zero_values = (
            Fraction(1, 50),
            Fraction(1, 50),
            Fraction(11, 50),
            Fraction(1, 250),
            Fraction(1, 250),
        ) + (Fraction(1, 4),) * 9
        zero_box = reduced.Box(zero_values, zero_values)
        positive_crossing = reduced.ResidualSlab(0, 0, 1, threshold)
        self.assertIsNone(
            reduced.tighten(
                zero_box,
                Fraction(1, 500),
                Fraction(11, 50),
                residual_slab=positive_crossing,
            )
        )

        crossing_values = list(zero_values)
        crossing_values[3] = Fraction(1, 200)
        crossing_box = reduced.Box(
            tuple(crossing_values), tuple(crossing_values)
        )
        self.assertIsNotNone(
            reduced.tighten(
                crossing_box,
                Fraction(1, 500),
                Fraction(11, 50),
                residual_slab=positive_crossing,
            )
        )

        fourth_values = list(zero_values)
        fourth_values[8:11] = [Fraction(7, 30)] * 3
        fourth_box = reduced.Box(tuple(fourth_values), tuple(fourth_values))
        self.assertIsNotNone(
            reduced.tighten(
                fourth_box,
                Fraction(1, 500),
                Fraction(11, 50),
                residual_slab=reduced.ResidualSlab(0, 4, 1, threshold),
            )
        )

    def test_twenty_slabs_cover_the_strict_tube_complement(self) -> None:
        threshold = Fraction(1, 3200)
        self.assertLess(
            10 * threshold * threshold,
            Fraction(1, 1_000_000),
        )

    def test_numerical_maximizer_has_an_exact_negative_point_bound(self) -> None:
        mu = np.array(
            [
                0.9980000000000012,
                0.00010317395587027238,
                0.0006110650740864773,
                0.0011823776239572128,
                0.0001033833460848407,
            ]
        )
        beta = np.array(
            [
                0.9971866960376634,
                0.000024089511252901538,
                0.002364879746160249,
                0.00039798534996817093,
                0.000026349354955294586,
            ]
        )
        zeta = np.array(
            [
                0.8677000000000001,
                0.0006796742292351857,
                0.009602382386432106,
                0.1216227133119478,
                0.0003952300723848224,
            ]
        )
        mu = dual.multiply_distribution(mu, 2)
        beta = dual.multiply_distribution(beta, 2)
        zeta = dual.multiply_distribution(zeta, 2)
        values: list[Fraction] = [
            Fraction(1, 500),
            Fraction(str(beta[1:].sum())),
            Fraction(1323, 10_000),
            Fraction("0.0009924297599807304"),
            Fraction("0.0013463064097057693"),
        ]
        for distribution in (mu, beta, zeta):
            support = distribution[1:].sum()
            values.extend(
                Fraction(str(value))
                for value in distribution[1:4] / support
            )
        box = reduced.Box(tuple(values), tuple(values))
        box = reduced.tighten(
            box, Fraction(1, 500), Fraction(1323, 10_000)
        )
        self.assertIsNotNone(box)
        assert box is not None
        bound = reduced.box_bound(box, dual.default_cuts())
        self.assertLess(bound.value, Fraction(-1, 1000))
        self.assertGreater(bound.value, Fraction(-2, 1000))
        eliminated = reduced.box_bound(
            box, dual.default_cuts(), include_stationary=False
        )
        self.assertLess(eliminated.value, Fraction(-1, 1000))
        self.assertGreater(eliminated.value, Fraction(-2, 1000))


if __name__ == "__main__":
    unittest.main()
