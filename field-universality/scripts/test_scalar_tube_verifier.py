from __future__ import annotations

import unittest
from fractions import Fraction

import verify_scalar_tube_bound as scalar


class ScalarTubeVerifierTests(unittest.TestCase):
    def test_exact_message_cancellation_tradeoff_on_vertex_family(self) -> None:
        a = Fraction(1, 1000)
        point = scalar.Interval(a, a)
        crossing = scalar.Interval(a / 2, a / 2)
        self.assertEqual(
            scalar.message_cancellation_tradeoff_lower(point, crossing),
            Fraction(9, 40) * a,
        )
        gridded = scalar.gridded_message_tradeoff_lower(point, crossing)
        self.assertGreater(gridded, Fraction(3, 10) * a)
        self.assertLess(gridded, Fraction(1, 3) * a)

    def test_numerical_maximizer_has_negative_exact_point_bound(self) -> None:
        a = Fraction(1, 500)
        b = Fraction("0.002105500803561426")
        x = Fraction("0.0009920127155698525")
        y = Fraction("0.0007882853876143102")
        z1 = Fraction("0.4754441521368845")
        z2 = Fraction("0.1851586907599317")
        values = (
            a,
            b,
            x,
            y,
            (a - x) * z1,
            (b - y) * z2,
        )
        box = scalar.Box(values, values)
        tightened = scalar.tighten(
            box,
            Fraction(1, 500),
            Fraction(11, 50),
            Fraction(1, 1_000_000),
        )
        self.assertIsNotNone(tightened)
        assert tightened is not None
        bound = scalar.box_bound(tightened, Fraction(11, 50))
        self.assertLess(bound.value, Fraction(-4, 10_000))

    def test_tube_lower_bound_rejects_an_outside_point(self) -> None:
        values = (
            Fraction(1, 100),
            Fraction(1, 100),
            Fraction(1, 200),
            Fraction(1, 500),
            Fraction(1, 1_000),
            Fraction(1, 1_000),
        )
        box = scalar.Box(values, values)
        defect = scalar.tube_lower_bound(box)
        self.assertGreater(defect, Fraction(1, 1_000_000))
        self.assertIsNone(
            scalar.tighten(
                box,
                Fraction(1, 500),
                Fraction(11, 50),
                Fraction(1, 1_000_000),
            )
        )


if __name__ == "__main__":
    unittest.main()
