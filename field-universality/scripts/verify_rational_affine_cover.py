#!/usr/bin/env python3
"""Replay exact affine-partition claims in a rationalized cover artifact."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from rationalize_affine_cover import (
    chamber_bounds,
    collect_leaves,
    exact_minimum_weight,
    exact_simplex_vertices,
    rational,
)


class VerificationError(ValueError):
    pass


def verify_partition_sums(
    numerators: list[list[int]], denominator: int, feature_count: int
) -> None:
    if not numerators or any(len(row) != feature_count for row in numerators):
        raise VerificationError("malformed coefficient matrix")
    for feature, column in enumerate(zip(*numerators, strict=True)):
        expected = denominator if feature == 0 else 0
        if sum(column) != expected:
            raise VerificationError(
                f"coefficient column {feature} sums to {sum(column)}, expected {expected}"
            )


def verify(manifest_path: Path, certificate_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    certificate_path = certificate_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / manifest.get("artifact_root", "..")).resolve()
    checkpoint = json.loads(
        (root / manifest["global_checkpoint"]).read_text(encoding="utf-8")
    )
    selected = sorted(checkpoint["queue"], key=lambda entry: entry["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    base_lower = [rational(value) for value in selected["lower"]]
    base_upper = [rational(value) for value in selected["upper"]]
    manifest_leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    records = certificate.get("leaves")
    if not isinstance(records, list):
        raise VerificationError("certificate leaves must be a list")
    if {record.get("leaf") for record in records} != set(manifest_leaves):
        raise VerificationError("certificate leaf set does not equal manifest leaf set")

    global_minimum: Fraction | None = None
    for record in records:
        name = record["leaf"]
        leaf = manifest_leaves[name]
        if record.get("gate") != leaf["gate"] or record.get("result") != leaf["result"]:
            raise VerificationError(f"{name}: artifact paths do not match manifest")
        gate = json.loads(
            (root / str(leaf["gate"]).replace("\\", "/")).read_text(encoding="utf-8")
        )
        result = json.loads(
            (root / str(leaf["result"]).replace("\\", "/")).read_text(encoding="utf-8")
        )
        if record.get("expert_pairs") != result.get("coefficient_expert_pairs"):
            raise VerificationError(f"{name}: expert pairs do not match result")
        coordinates = [int(value) for value in record["feature_coordinates"]]
        if coordinates != [int(value) for value in gate["affine_gating"]["feature_coordinates"]]:
            raise VerificationError(f"{name}: feature coordinates do not match gate")
        denominator = int(record["coefficient_denominator"])
        if denominator <= 0:
            raise VerificationError(f"{name}: denominator is not positive")
        numerators = record["coefficient_numerators"]
        if len(numerators) != len(record["expert_pairs"]):
            raise VerificationError(f"{name}: coefficient/expert count mismatch")
        if any(not isinstance(value, int) for row in numerators for value in row):
            raise VerificationError(f"{name}: coefficient numerator is not an integer")
        verify_partition_sums(numerators, denominator, len(coordinates) + 1)
        lower, upper = chamber_bounds(base_lower, base_upper, gate["restrictions"])
        blocks = [exact_simplex_vertices(lower, upper, start) for start in (0, 5, 10)]
        minimum = exact_minimum_weight(numerators, denominator, blocks, coordinates)
        claimed = rational(record["minimum_exact_weight"])
        if minimum != claimed:
            raise VerificationError(
                f"{name}: claimed minimum {claimed} differs from replay {minimum}"
            )
        if minimum < 0:
            raise VerificationError(f"{name}: negative affine weight {minimum}")
        global_minimum = minimum if global_minimum is None else min(global_minimum, minimum)
    if global_minimum is None:
        raise VerificationError("certificate has no leaves")
    return {
        "status": "verified exact rational affine partitions; analytic bound pending",
        "leaves": len(records),
        "minimum_exact_weight": str(global_minimum),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    try:
        summary = verify(args.manifest, args.certificate)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
