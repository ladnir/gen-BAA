#!/usr/bin/env python3
"""Tests for exact centered-curvature algebra replay."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractions import Fraction

from verify_affine_face_centered_certificate import (
    verify_certificate,
    verify_guillotine_cover,
    verify_taylor_certificate,
)


def example_certificate():
    return {
        "variables": [6],
        "hessian_interval_lower": [["-3/1"]],
        "hessian_interval_upper": [["-2/1"]],
        "row_sum_error_bound": "1/2",
        "certified_curvature": "1/1",
        "ldl_pivots": ["1/1"],
        "center_log_value_upper": "-1/10",
        "gradient_absolute_upper": ["0/1"],
        "radii": ["1/2"],
        "maximizing_displacements": ["0/1"],
        "quadratic_contributions": ["0/1"],
        "total_log_upper": "-1/10",
        "closed": True,
    }


class VerifyCenteredCertificateTests(unittest.TestCase):
    def test_accepts_exact_certificate(self):
        verify_certificate(example_certificate())

    def test_rejects_tampered_total(self):
        certificate = copy.deepcopy(example_certificate())
        certificate["total_log_upper"] = "0/1"
        with self.assertRaises(ValueError):
            verify_certificate(certificate)

    def test_replays_directed_taylor_certificate(self):
        certificate = {
            "variables": [6],
            "center_log_value_upper": "-1/1",
            "linear_term_upper": ["1/4"],
            "quadratic_term_upper": [["1/4"]],
            "total_log_upper": "-1/2",
            "closed": True,
        }
        verify_taylor_certificate(certificate)

    def test_reconstructs_guillotine_cover(self):
        verify_guillotine_cover(
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1)),
            [
                ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))),
                ((Fraction(1, 2), Fraction(0)), (Fraction(1), Fraction(1, 2))),
                ((Fraction(1, 2), Fraction(1, 2)), (Fraction(1), Fraction(1))),
            ],
        )

    def test_rejects_cover_gap(self):
        with self.assertRaises(ValueError):
            verify_guillotine_cover(
                (Fraction(0),),
                (Fraction(1),),
                [((Fraction(0),), (Fraction(1, 3),))],
            )


if __name__ == "__main__":
    unittest.main()
