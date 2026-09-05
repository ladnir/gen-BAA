import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_cell_affine_cover import CoverError, verify_manifest


class CellAffineCoverTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "results").mkdir()
        self.gate = {
            "frontier_index": 0,
            "restrictions": ["6:le:0.5"],
        }
        self.result = {
            "closed": True,
            "complete": True,
            "termination": "target margin reached",
            "target_margin": 0.0001,
            "gate": "results/gate.json",
            "history": [{"continuous_log_upper": -0.0002}],
        }
        self._write("results/gate.json", self.gate)
        self._write("results/result.json", self.result)
        gate_ge = copy.deepcopy(self.gate)
        gate_ge["restrictions"] = ["6:ge:1/2"]
        self._write("results/gate-ge.json", gate_ge)
        result_ge = copy.deepcopy(self.result)
        result_ge["gate"] = "results/gate-ge.json"
        self._write("results/result-ge.json", result_ge)
        self.manifest = {
            "artifact_root": "..",
            "certificate_status": "floating_only_not_a_proof",
            "frontier_index": 0,
            "target_margin": "1/10000",
            "tree": {
                "split": {"dimension": 6, "threshold": "1/2"},
                "le": {
                    "leaf": "a",
                    "gate": "results/gate.json",
                    "result": "results/result.json",
                },
                "ge": {
                    "leaf": "b",
                    "gate": "results/gate-ge.json",
                    "result": "results/result-ge.json",
                },
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, relative, value):
        path = self.root / relative
        path.write_text(json.dumps(value), encoding="utf-8")

    def _verify(self, manifest=None):
        self._write("results/manifest.json", manifest or self.manifest)
        return verify_manifest(self.root / "results/manifest.json")

    def test_accepts_exact_two_leaf_cover(self):
        summary = self._verify()
        self.assertEqual(summary.leaves, 2)
        self.assertEqual(summary.splits, 1)
        self.assertEqual(summary.weakest_leaf, "a")

    def test_rejects_corrupt_split(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["tree"]["ge"]
        with self.assertRaisesRegex(CoverError, "exactly split, le, and ge"):
            self._verify(manifest)

    def test_rejects_gate_restriction_mismatch(self):
        gate = copy.deepcopy(self.gate)
        gate["restrictions"] = ["6:le:0.49"]
        self._write("results/gate.json", gate)
        with self.assertRaisesRegex(CoverError, "restrictions do not match"):
            self._verify()

    def test_rejects_nonclosed_result(self):
        result = copy.deepcopy(self.result)
        result["closed"] = False
        self._write("results/result.json", result)
        with self.assertRaisesRegex(CoverError, "not closed and complete"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
