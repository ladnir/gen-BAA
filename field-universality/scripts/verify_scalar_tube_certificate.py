#!/usr/bin/env python3
"""Independently replay an exact scalar F_5 tube-cover certificate."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

from mpmath import iv

import verify_scalar_tube_bound as tube


def parse_fraction(raw: object, name: str) -> Fraction:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a rational string")
    return Fraction(raw)


def replay(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("the certificate must be a JSON object")
    if raw.get("schema") != "f5-scalar-tube-cover-v1":
        raise ValueError("unsupported scalar tube certificate schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("the certificate is not for the F_5 threefold encoder")

    precision = raw.get("interval_precision")
    if not isinstance(precision, int) or precision < 32:
        raise ValueError("invalid interval precision")
    arguments = Namespace(
        delta=parse_fraction(raw.get("delta"), "delta"),
        eta=parse_fraction(raw.get("eta"), "eta"),
        maximum_defect=parse_fraction(
            raw.get("maximum_defect"), "maximum_defect"
        ),
        precision=precision,
    )
    if arguments.delta != Fraction(11, 50):
        raise ValueError("this verifier expects delta = 11/50")
    if not Fraction(0) < arguments.eta <= arguments.delta:
        raise ValueError("invalid positive message-support cutoff")
    if arguments.maximum_defect != Fraction(1, 1_000_000):
        raise ValueError("this verifier expects tube defect 10^-6")

    records = raw.get("split_records")
    if not isinstance(records, list):
        raise ValueError("missing split records")
    iv.dps = precision
    root = tube.initial_box(arguments)
    active = {0: tube.box_bound(root, arguments.delta)}
    next_identifier = 1
    for offset, record in enumerate(records):
        if (
            not isinstance(record, list)
            or len(record) != 2
            or not all(isinstance(value, int) for value in record)
        ):
            raise ValueError(f"invalid split record {offset}")
        node_identifier, dimension = record
        if dimension < 0 or dimension >= 6:
            raise ValueError(f"invalid split dimension in record {offset}")
        current = active.pop(node_identifier, None)
        if current is None:
            raise ValueError(f"split record {offset} names no active leaf")
        for child in tube.split(current.box, dimension):
            tightened = tube.tighten(
                child,
                arguments.eta,
                arguments.delta,
                arguments.maximum_defect,
            )
            if tightened is None:
                continue
            active[next_identifier] = tube.box_bound(
                tightened, arguments.delta
            )
            next_identifier += 1

    if not active:
        raise ValueError("certificate leaves no feasible boxes")
    worst = max(bound.value for bound in active.values())
    if worst >= tube.ZERO:
        raise ValueError(f"certificate has a nonnegative leaf: {float(worst)}")
    declared = Fraction(
        int(raw.get("upper_bound_numerator")),
        int(raw.get("upper_bound_denominator")),
    )
    if declared >= tube.ZERO or worst > declared:
        raise ValueError(
            "replayed leaves do not establish the declared negative upper"
        )
    if raw.get("boxes_split") != len(records):
        raise ValueError("declared split count does not match the certificate")
    return {
        "status": "verified scalar tube certificate",
        "boxes_split": len(records),
        "leaves_checked": len(active),
        "upper_bound": float(worst),
        "upper_bound_numerator": str(worst.numerator),
        "upper_bound_denominator": str(worst.denominator),
        "upper_bound_is_exactly_negative": True,
        "declared_upper_bound": float(declared),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    certificate = json.loads(arguments.certificate.read_text(encoding="utf-8"))
    print(json.dumps(replay(certificate), indent=2, sort_keys=True))
