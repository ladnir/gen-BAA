from __future__ import annotations

import unittest

from verify_uniform_support_slab import verify


class UniformSupportSlabTests(unittest.TestCase):
    def test_all_rational_checks_pass(self) -> None:
        result = verify()
        self.assertEqual(
            result["status"], "verified all-primes support-slab constants"
        )
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
