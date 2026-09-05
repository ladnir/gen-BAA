#!/usr/bin/env python3
"""Generate a small rational cover of the F_5 zero-payment surface."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import analyze_zero_payment_surface as zero


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def generate(args: argparse.Namespace) -> dict[str, object]:
    eta = Fraction(args.eta)
    delta = Fraction(args.delta)
    target = Fraction(args.target)
    floating = zero.cover(
        float(eta),
        float(delta),
        float(target),
        args.maximum_boxes,
    )
    if not bool(floating["closed"]):
        raise RuntimeError("floating proposal did not close the requested target")
    leaves = []
    for leaf in floating["closed_boxes"]:
        if not isinstance(leaf, dict):
            raise RuntimeError("invalid floating leaf")
        bounds = leaf.get("bounds_rational")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise RuntimeError("floating leaf has no rational bounds")
        leaves.append(
            {
                "bounds": bounds,
                "proposal_upper": format(float(leaf["upper"]), ".17g"),
            }
        )
    certificate = {
        "schema": "f5-zero-payment-surface-v1",
        "prime": 5,
        "repetition": 3,
        "eta": fraction_text(eta),
        "delta": fraction_text(delta),
        "target_upper": fraction_text(target),
        "leaves": leaves,
        "generator": {
            "floating_final_upper": format(
                float(floating["final_upper"]), ".17g"
            ),
            "boxes_split": int(floating["boxes_split"]),
            "maximum_depth": int(floating["maximum_depth"]),
        },
    }
    zero.write_json_atomic(args.output, certificate)
    return certificate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", default="1/500")
    parser.add_argument("--delta", default="11/50")
    parser.add_argument("--target", default="-1/200")
    parser.add_argument("--maximum-boxes", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-zero-payment-certificate.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(generate(arguments), indent=2, sort_keys=True))
