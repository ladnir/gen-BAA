#!/usr/bin/env python3
"""Verify topology, hashes, and replay status of a message-derivative cover."""

from __future__ import annotations

import argparse
import hashlib
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


def dyadic_interval(specifications: list[str], coordinate: int):
    lower = Fraction(0)
    upper = Fraction(1)
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
            raise VerificationError("invalid half side")
    return lower, upper


def resolve(root: Path, value: str) -> Path:
    path = Path(value.replace("/", str(Path("/")).replace("/", "\\")))
    if path.is_absolute():
        return path
    return root / Path(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    root = args.manifest.resolve().parents[1]
    boxes = manifest.get("boxes", [])
    if len(boxes) != 7:
        raise VerificationError("cover must contain seven boxes")
    common = list(manifest["common_half_restrictions"])
    if common != ["4:upper"] * 4:
        raise VerificationError("cover does not target the q4 top sixteenth")
    occupancy = [[0 for _ in range(2)] for _ in range(32)]
    minimum_lower = None
    records = []
    common_sources = None
    for expected_index, raw in enumerate(boxes):
        if int(raw["index"]) != expected_index:
            raise VerificationError("cover box indices are not canonical")
        relative = list(raw["relative_half_restrictions"])
        if any(int(value.split(":")[0]) not in (7, 8) for value in relative):
            raise VerificationError("cover refines an unexpected coordinate")
        proposal_path = root / raw["proposal"]
        certificate_path = root / raw["certificate"]
        replay_path = root / raw["replay"]
        proposal = load(proposal_path)
        certificate = load(certificate_path)
        replay = load(replay_path)
        if not certificate.get("rigorous") or not certificate.get("closed"):
            raise VerificationError(f"box {expected_index} certificate is not closed")
        if not replay.get("verified"):
            raise VerificationError(f"box {expected_index} replay did not verify")
        if replay.get("certificate_sha256") != digest(certificate_path):
            raise VerificationError(f"box {expected_index} replay hash is stale")
        if certificate["source_sha256"]["proposal"] != digest(proposal_path):
            raise VerificationError(f"box {expected_index} proposal hash is stale")
        if replay["source_sha256"] != certificate["source_sha256"]:
            raise VerificationError(f"box {expected_index} source hashes disagree")
        source_without_proposal = {
            key: value
            for key, value in certificate["source_sha256"].items()
            if key != "proposal"
        }
        if common_sources is None:
            common_sources = source_without_proposal
        elif source_without_proposal != common_sources:
            raise VerificationError("cover boxes use different theorem sources")
        if (
            certificate["leaf"] != manifest["leaf"]
            or int(certificate["coordinate"]) != int(manifest["coordinate"])
            or certificate["dependent_coordinates"]
            != manifest["dependent_coordinates"]
            or certificate["fixed"] != manifest["fixed"]
        ):
            raise VerificationError(f"box {expected_index} domain metadata changed")
        expected_halves = common + relative
        if certificate["half_restrictions"] != expected_halves:
            raise VerificationError(f"box {expected_index} topology mismatch")
        q7_lower, q7_upper = dyadic_interval(relative, 7)
        q8_lower, q8_upper = dyadic_interval(relative, 8)
        scaled = (
            q7_lower * 32,
            q7_upper * 32,
            q8_lower * 2,
            q8_upper * 2,
        )
        if any(value.denominator != 1 for value in scaled):
            raise VerificationError("cover endpoint is off the audit grid")
        left, right, bottom, top = (int(value) for value in scaled)
        if left >= right or bottom >= top:
            raise VerificationError("empty cover box")
        for q7_cell in range(left, right):
            for q8_cell in range(bottom, top):
                occupancy[q7_cell][q8_cell] += 1
        lower_bound = Fraction(certificate["total_derivative_lower"])
        if lower_bound <= 0:
            raise VerificationError(f"box {expected_index} lower bound is not positive")
        minimum_lower = (
            lower_bound if minimum_lower is None else min(minimum_lower, lower_bound)
        )
        records.append(
            {
                "index": expected_index,
                "q7_interval": [str(q7_lower), str(q7_upper)],
                "q8_interval": [str(q8_lower), str(q8_upper)],
                "total_derivative_lower": str(lower_bound),
                "certificate_sha256": digest(certificate_path),
                "replay_sha256": digest(replay_path),
            }
        )
    counts = [value for row in occupancy for value in row]
    if min(counts) != 1 or max(counts) != 1:
        raise VerificationError("dyadic boxes do not form an exact partition")
    assert minimum_lower is not None
    result = {
        "status": "independent exact message-derivative cover replay",
        "verified": True,
        "manifest": str(args.manifest),
        "manifest_sha256": digest(args.manifest),
        "boxes": records,
        "audit_grid_cells": len(counts),
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
