from __future__ import annotations

import unittest
from fractions import Fraction

from mpmath import iv

from verify_conditional_product_bulk import default_cuts
from verify_eliminated_composition_bulk import (
    Box,
    Certificate,
    ONE,
    ZERO,
    box_bound,
    candidate_certificate,
    reference_certificates,
    trim_certificate_bank,
)


class EliminatedCompositionVerifierTests(unittest.TestCase):
    def test_candidate_point_has_exactly_negative_envelope(self) -> None:
        iv.dps = 80
        point = tuple(
            Fraction(value)
            for value in (
                "0.002",
                "0.002813303962336616",
                "0.1323",
                "0.0009924297599807304",
                "0.0013463064097057693",
            )
        )
        box = Box(point, point)
        bound = box_bound(
            box,
            default_cuts(),
            (candidate_certificate(),),
            Fraction(1, 500),
            Fraction(1323, 10_000),
        )
        self.assertIsNotNone(bound)
        assert bound is not None
        self.assertLess(bound.value, Fraction(-1, 1_000))
        self.assertGreater(bound.value, Fraction(-1, 500))

    def test_trimmed_local_cut_indices_are_remapped_exactly(self) -> None:
        static_cuts = default_cuts()
        cuts = static_cuts + tuple(static_cuts[0] for _ in range(20))
        certificates = list(reference_certificates())
        for group in range(5):
            start = 32 + 4 * group
            certificates.append(
                Certificate(
                    tuple(((start + index, ONE),) for index in range(4)),
                    tuple((ZERO,) * 4 for _ in range(4)),
                )
            )
        trimmed_cuts, trimmed_certificates = trim_certificate_bank(
            cuts, tuple(certificates), local_history=3
        )
        self.assertEqual(len(trimmed_cuts), 44)
        self.assertEqual(len(trimmed_certificates), 7)
        referenced = {
            cut_index
            for certificate in trimmed_certificates[4:]
            for weights in certificate.first_weights
            for cut_index, _ in weights
        }
        self.assertEqual(referenced, set(range(32, 44)))


if __name__ == "__main__":
    unittest.main()
