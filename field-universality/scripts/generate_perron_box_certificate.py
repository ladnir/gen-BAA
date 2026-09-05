#!/usr/bin/env python3
"""Generate a sparse rational LP dual for one Perron composition box."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import verify_perron_box_certificate as box_checker


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def box_data() -> dict[str, list[str]]:
    lower = (
        Fraction(99799, 100_000),
        Fraction(99, 100_000),
        Fraction(0),
        Fraction(0),
        Fraction(99, 100_000),
    )
    upper = (
        Fraction(99801, 100_000),
        Fraction(101, 100_000),
        Fraction(1, 1_000_000),
        Fraction(1, 1_000_000),
        Fraction(101, 100_000),
    )
    return {
        "message_lower": [fraction_text(value) for value in lower],
        "message_upper": [fraction_text(value) for value in upper],
        "middle_lower": [fraction_text(value) for value in lower],
        "middle_upper": [fraction_text(value) for value in upper],
    }


def generate(base: object, precision: int) -> dict[str, object]:
    raw_box = box_data()
    parsed_box = box_checker.parse_box(raw_box)
    _, objective = box_checker.affine_perron_upper(base, parsed_box, precision)
    dual = {}
    for component in ("message", "middle"):
        start = box_checker.component_start(component)
        for index in range(5):
            coefficient = objective[start + index]
            if coefficient >= 0:
                name = f"box_{component}_{index}_upper"
                multiplier = coefficient
            else:
                name = f"box_{component}_{index}_lower"
                multiplier = -coefficient
            if multiplier > 0:
                dual[name] = fraction_text(multiplier)
    result = {
        "schema": "f5-perron-box-v1",
        "base_certificate": "f5-perron-point-certificate.json",
        "eta": "1/500",
        "box": raw_box,
        "dual": dual,
        "generation": {
            "status": "untrusted deterministic dual proposal",
            "interval_precision": precision,
        },
    }
    box_checker.verify(result, base, precision)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("results/f5-perron-point-certificate.json"),
    )
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    base_certificate = json.loads(arguments.base.read_text(encoding="utf-8"))
    print(
        json.dumps(
            generate(base_certificate, arguments.precision),
            indent=2,
            sort_keys=True,
        )
    )
