#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from generate_perron_box_certificate import generate
from verify_perron_box_certificate import verify


RESULTS = Path(__file__).resolve().parents[1] / "results"
POINT_CERTIFICATE = RESULTS / "f5-perron-point-certificate.json"
BOX_CERTIFICATE = RESULTS / "f5-perron-box-certificate.json"


class PerronBoxCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(POINT_CERTIFICATE.read_text(encoding="utf-8"))

    def test_stored_box_certificate_verifies(self) -> None:
        raw = json.loads(BOX_CERTIFICATE.read_text(encoding="utf-8"))
        result = verify(raw, self.base, 80)
        self.assertTrue(result["box_upper_is_exactly_negative"])

    def test_generator_proposes_a_valid_dual(self) -> None:
        result = verify(generate(self.base, 80), self.base, 80)
        self.assertTrue(result["box_upper_is_exactly_negative"])

    def test_missing_dual_multiplier_is_rejected(self) -> None:
        raw = json.loads(BOX_CERTIFICATE.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(raw)
        tampered["dual"].pop(next(iter(tampered["dual"])))
        with self.assertRaisesRegex(ValueError, "LP dual is infeasible"):
            verify(tampered, self.base, 80)

    def test_box_must_contain_the_exact_feasible_witness(self) -> None:
        raw = json.loads(BOX_CERTIFICATE.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(raw)
        tampered["box"]["message_lower"][1] = "1001/1000000"
        with self.assertRaisesRegex(ValueError, "exact message witness"):
            verify(tampered, self.base, 80)


if __name__ == "__main__":
    unittest.main()
