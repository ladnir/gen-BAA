#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from generate_perron_point_certificate import generate
from verify_perron_point_certificate import verify


CERTIFICATE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "f5-perron-point-certificate.json"
)


class PerronPointCertificateTests(unittest.TestCase):
    def test_stored_certificate_verifies(self) -> None:
        raw = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
        result = verify(raw, 80)
        self.assertTrue(result["perron_point_upper_is_exactly_negative"])

    def test_smaller_lambda_is_rejected(self) -> None:
        raw = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(raw)
        tampered["first_cut"]["lambda"] = "1/1000000"
        with self.assertRaisesRegex(ValueError, "exact rational row bound"):
            verify(tampered, 80)

    def test_floating_generator_proposes_a_valid_rational_certificate(self) -> None:
        result = verify(generate(60, 50), 80)
        self.assertTrue(result["perron_point_upper_is_exactly_negative"])


if __name__ == "__main__":
    unittest.main()
