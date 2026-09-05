#!/usr/bin/env python3
"""Verify an exact full message-coordinate derivative cover."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


class VerificationError(ValueError):
    pass


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def interval(specifications: list[str], coordinate: int):
    lower, upper = Fraction(0), Fraction(1)
    for specification in specifications:
        coordinate_text, side = specification.split(":")
        if int(coordinate_text) != coordinate:
            continue
        midpoint = (lower + upper) / 2
        if side == "lower":
            upper = midpoint
        elif side == "upper":
            lower = midpoint
        else:
            raise VerificationError("invalid dyadic side")
    return lower, upper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    root = args.manifest.resolve().parents[1]
    grid = {int(key): int(value) for key, value in manifest["audit_grid"].items()}
    if grid != {4: 16, 7: 32, 8: 2}:
        raise VerificationError("unexpected full-cover audit grid")
    occupancy = {
        index: 0
        for index in itertools.product(
            range(grid[4]), range(grid[7]), range(grid[8])
        )
    }
    records = []
    common_sources = None
    minimum_lower = None
    boxes = manifest.get("boxes", [])
    if len(boxes) != 11:
        raise VerificationError("full q4 cover must contain eleven boxes")
    for expected_index, raw in enumerate(boxes):
        if int(raw["index"]) != expected_index:
            raise VerificationError("noncanonical box ordering")
        halves = list(raw["half_restrictions"])
        if any(int(value.split(":")[0]) not in grid for value in halves):
            raise VerificationError("box refines a coordinate outside the audit grid")
        proposal_path = root / raw["proposal"]
        certificate_path = root / raw["certificate"]
        replay_path = root / raw["replay"]
        proposal = load(proposal_path)
        certificate = load(certificate_path)
        replay = load(replay_path)
        if not certificate.get("rigorous") or not certificate.get("closed"):
            raise VerificationError(f"box {expected_index} certificate is not closed")
        if not replay.get("verified"):
            raise VerificationError(f"box {expected_index} replay failed")
        if replay.get("certificate_sha256") != digest(certificate_path):
            raise VerificationError(f"box {expected_index} has a stale replay")
        if certificate["source_sha256"]["proposal"] != digest(proposal_path):
            raise VerificationError(f"box {expected_index} proposal hash changed")
        if replay["source_sha256"] != certificate["source_sha256"]:
            raise VerificationError(f"box {expected_index} source hashes disagree")
        shared = {
            key: value
            for key, value in certificate["source_sha256"].items()
            if key != "proposal"
        }
        if common_sources is None:
            common_sources = shared
        elif shared != common_sources:
            raise VerificationError("boxes do not share theorem sources")
        if (
            certificate["leaf"] != manifest["leaf"]
            or int(certificate["coordinate"]) != int(manifest["coordinate"])
            or certificate["dependent_coordinates"]
            != manifest["dependent_coordinates"]
            or certificate["fixed"] != manifest["fixed"]
            or certificate["half_restrictions"] != halves
        ):
            raise VerificationError(f"box {expected_index} metadata mismatch")
        integer_intervals = {}
        for coordinate, denominator in grid.items():
            lower, upper = interval(halves, coordinate)
            scaled_lower = lower * denominator
            scaled_upper = upper * denominator
            if scaled_lower.denominator != 1 or scaled_upper.denominator != 1:
                raise VerificationError("box endpoint is off the exact audit grid")
            left, right = int(scaled_lower), int(scaled_upper)
            if left >= right:
                raise VerificationError("empty full-cover box")
            integer_intervals[coordinate] = (left, right, lower, upper)
        for q4 in range(*integer_intervals[4][0:2]):
            for q7 in range(*integer_intervals[7][0:2]):
                for q8 in range(*integer_intervals[8][0:2]):
                    occupancy[(q4, q7, q8)] += 1
        lower_bound = Fraction(certificate["total_derivative_lower"])
        if lower_bound <= 0:
            raise VerificationError(f"box {expected_index} is not strictly positive")
        minimum_lower = (
            lower_bound if minimum_lower is None else min(minimum_lower, lower_bound)
        )
        records.append(
            {
                "index": expected_index,
                "intervals": {
                    str(coordinate): [
                        str(integer_intervals[coordinate][2]),
                        str(integer_intervals[coordinate][3]),
                    ]
                    for coordinate in sorted(grid)
                },
                "total_derivative_lower": str(lower_bound),
                "certificate_sha256": digest(certificate_path),
                "replay_sha256": digest(replay_path),
            }
        )
    counts = list(occupancy.values())
    if min(counts) != 1 or max(counts) != 1:
        raise VerificationError("full q4 boxes do not partition the audit grid")
    assert minimum_lower is not None
    result = {
        "status": "independent exact full message-derivative cover replay",
        "verified": True,
        "manifest": str(args.manifest),
        "manifest_sha256": digest(args.manifest),
        "audit_grid_cells": len(counts),
        "boxes": records,
        "minimum_total_derivative_lower": str(minimum_lower),
        "minimum_total_derivative_lower_approx": float(minimum_lower),
        "common_source_sha256": common_sources,
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
