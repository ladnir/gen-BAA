#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path

import generate_zero_payment_certificate as generator
import verify_zero_payment_certificate as verifier


class ZeroPaymentCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cls.certificate = generator.generate(
                argparse.Namespace(
                    eta="1/500",
                    delta="11/50",
                    target="-1/200",
                    maximum_boxes=100_000,
                    output=Path(directory) / "certificate.json",
                )
            )

    def test_generated_certificate_verifies(self) -> None:
        result = verifier.verify(self.certificate, 80)
        self.assertEqual(result["leaves_checked"], 22)
        self.assertTrue(result["maximum_upper_is_at_most_target"])

    def test_missing_leaf_is_rejected(self) -> None:
        corrupted = copy.deepcopy(self.certificate)
        corrupted["leaves"].pop()
        with self.assertRaisesRegex(ValueError, "do not cover"):
            verifier.verify(corrupted, 80)

    def test_stronger_unsupported_target_is_rejected(self) -> None:
        corrupted = copy.deepcopy(self.certificate)
        corrupted["target_upper"] = "-3/500"
        with self.assertRaisesRegex(ValueError, "does not meet"):
            verifier.verify(corrupted, 80)


if __name__ == "__main__":
    unittest.main()
