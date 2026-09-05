#!/usr/bin/env python3
"""Directed interval tangent-derivative diagnostic for one affine-gate chamber."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from mpmath import iv

from explore_affine_face_interval import (
    FaceBox,
    exact_cut,
    lower_float,
    mixture_dual,
    tighten_simplex_box,
    upper_float,
)
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


def load_exact(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Fraction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument(
        "--zero-message-epsilon",
        default="1e-30",
        help="diagnostic lower cutoff for message coordinates whose box lower bound is zero",
    )
    parser.add_argument("--fix", action="append", default=[], metavar="DIM:SIDE")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision < 40:
        parser.error("precision must be at least 40 decimal digits")
    iv.dps = args.precision

    manifest = load_exact(args.manifest)
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[args.leaf]
    partitions = load_exact(args.partitions)
    record = next(value for value in partitions["leaves"] if value["leaf"] == args.leaf)
    denominator = int(record["coefficient_denominator"])
    coefficients = [
        [Fraction(int(value), denominator) for value in row]
        for row in record["coefficient_numerators"]
    ]
    checkpoint = load_exact(root / manifest["global_checkpoint"])
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate = load_exact(root / str(leaf["gate"]).replace("\\", "/"))
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    epsilon = Fraction(args.zero_message_epsilon)
    if epsilon <= 0:
        parser.error("--zero-message-epsilon must be positive")
    excluded_zero_message_coordinates = []
    for coordinate in range(1, 5):
        if lower[coordinate] == 0 and upper[coordinate] > 0:
            lower[coordinate] = min(epsilon, upper[coordinate])
            excluded_zero_message_coordinates.append(coordinate)
    fixed: set[int] = set()
    for specification in args.fix:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in (0, 5, 10):
            parser.error("dependent coordinates 0, 5, and 10 cannot be fixed")
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            parser.error("--fix side must be lower or upper")
    dependent = [0, 5, 10]
    variables = [
        coordinate
        for coordinate in (*range(1, 5), *range(6, 10), *range(11, 15))
        if coordinate not in fixed
    ]
    box = tighten_simplex_box(
        FaceBox(tuple(lower), tuple(upper)), variables, dependent
    )
    if box is None:
        parser.error("fixed chamber face is empty")
    bank = load_exact(root / "results/f5-projected-kl-dual-bank-0.22.json")
    first_cuts = [exact_cut(value) for value in bank["first_cuts"]]
    second_cuts = [exact_cut(value) for value in bank["second_cuts"]]
    dual = mixture_dual(
        box,
        variables,
        dependent,
        coefficients,
        [int(value) for value in record["feature_coordinates"]],
        record["expert_pairs"],
        first_cuts,
        second_cuts,
    )
    derivatives = [
        {
            "coordinate": coordinate,
            "lower": lower_float(value),
            "upper": upper_float(value),
            "certified_positive": lower_float(value) > 0.0,
            "certified_negative": upper_float(value) < 0.0,
        }
        for coordinate, value in zip(variables, dual.gradient, strict=True)
    ]
    payload = {
        "status": "directed interval tangent-derivative diagnostic",
        "leaf": args.leaf,
        "fixed": args.fix,
        "variables": variables,
        "dependent_coordinates": dependent,
        "zero_message_epsilon": str(epsilon),
        "excluded_zero_message_coordinates": excluded_zero_message_coordinates,
        "mixture_lower": lower_float(dual.value),
        "mixture_upper": upper_float(dual.value),
        "derivatives": derivatives,
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
