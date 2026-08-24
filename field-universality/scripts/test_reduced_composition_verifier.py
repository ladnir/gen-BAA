from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

import verify_conditional_product_bulk as dual
import verify_reduced_composition_bulk as reduced


class ReducedCompositionVerifierTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
