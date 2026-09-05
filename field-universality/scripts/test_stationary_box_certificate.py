#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import verify_stationary_box_certificate as checker


CERTIFICATE = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "f5-stationary-box-certificate-0.22.json"
)


class StationaryBoxCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))

    def test_stored_certificate_verifies(self) -> None:
        result = checker.verify(self.certificate, 80)
        self.assertTrue(result["box_upper_is_exactly_negative"])
        self.assertLess(result["box_upper"], 0.0)

    def test_kernel_underestimate_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["first_cut"]["kernel_upper"][0][0] = "0/1"
        with self.assertRaisesRegex(ValueError, "does not upper-bound exp"):
            checker.verify(tampered, 80)

    def test_missing_lp_dual_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["dual"] = {}
        with self.assertRaisesRegex(ValueError, "dual is infeasible"):
            checker.verify(tampered, 80)

    def test_wrong_target_domain_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["delta"] = "6/25"
        with self.assertRaisesRegex(ValueError, "unexpected support cutoff"):
            checker.verify(tampered, 80)

    def test_out_of_box_witness_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.certificate)
        tampered["composition_witness"][0] = "0/1"
        with self.assertRaisesRegex(ValueError, "not normalized"):
            checker.verify(tampered, 80)


if __name__ == "__main__":
    unittest.main()
