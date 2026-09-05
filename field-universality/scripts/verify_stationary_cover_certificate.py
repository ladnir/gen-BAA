#!/usr/bin/env python3
"""Independently replay an exact split-tree cover by stationary box proofs."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import verify_stationary_box_certificate as leaf_checker


ZERO = Fraction(0)
ONE = Fraction(1)
DIMENSION = 15


def text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def root_box(eta: Fraction, delta: Fraction) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    lower = [ZERO] * DIMENSION
    upper = [ONE] * DIMENSION
    for start, support_lower, support_upper in (
        (0, eta, min(ONE, 4 * delta)),
        (5, eta / 2, min(ONE, 2 * delta)),
        (10, eta / 4, delta),
    ):
        lower[start] = ONE - support_upper
        upper[start] = ONE - support_lower
        for index in range(start + 1, start + 5):
            upper[index] = support_upper
    return tuple(lower), tuple(upper)


def same_box(raw: object, expected: tuple[tuple[Fraction, ...], tuple[Fraction, ...]]) -> None:
    if leaf_checker.parse_box(raw) != expected:
        raise ValueError("certificate node has an unexpected box")


def child_box(
    parent: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    dimension: int,
    threshold: Fraction,
    left: bool,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    lower, upper = parent
    child_lower = list(lower)
    child_upper = list(upper)
    if left:
        child_upper[dimension] = threshold
    else:
        child_lower[dimension] = threshold
    return tuple(child_lower), tuple(child_upper)


def infeasibility_reason(
    box: tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
) -> str | None:
    lower, upper = box
    for block, start in enumerate((0, 5, 10)):
        if sum(lower[start : start + 5], ZERO) > ONE:
            return f"simplex_{block}_lower_sum"
        if sum(upper[start : start + 5], ZERO) < ONE:
            return f"simplex_{block}_upper_sum"
    if sum(lower[1:5], ZERO) > 2 * sum(upper[6:10], ZERO):
        return "message_to_middle_support"
    if sum(lower[6:10], ZERO) > 2 * sum(upper[11:15], ZERO):
        return "middle_to_output_support"
    for value in range(2, 5):
        if lower[value] > upper[1]:
            return f"canonical_message_{value}"
    return None


def verify_node(
    raw: object,
    expected: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    precision: int,
    counts: dict[str, int],
) -> Fraction:
    if not isinstance(raw, dict):
        raise ValueError("cover node must be an object")
    kind = raw.get("kind")
    if kind == "infeasible":
        reason = infeasibility_reason(expected)
        if reason is None or raw.get("reason") != reason:
            raise ValueError("invalid infeasible cover leaf")
        counts["infeasible"] += 1
        return -ONE
    if kind == "leaf":
        certificate = raw.get("certificate")
        if not isinstance(certificate, dict):
            raise ValueError("cover leaf is missing its box certificate")
        same_box(certificate.get("box"), expected)
        result = leaf_checker.verify(certificate, precision)
        counts["leaves"] += 1
        counts["constraints"] += int(result["constraints_checked"])
        return Fraction(
            int(result["box_upper_numerator"]),
            int(result["box_upper_denominator"]),
        )
    if kind != "split":
        raise ValueError("unknown cover node kind")
    dimension = raw.get("dimension")
    if not isinstance(dimension, int) or dimension not in range(DIMENSION):
        raise ValueError("invalid split dimension")
    threshold = leaf_checker.parse_fraction(raw.get("threshold"))
    lower, upper = expected
    if not lower[dimension] < threshold < upper[dimension]:
        raise ValueError("split threshold is outside the parent interval")
    counts["splits"] += 1
    left_upper = verify_node(
        raw.get("left"), child_box(expected, dimension, threshold, True), precision, counts
    )
    right_upper = verify_node(
        raw.get("right"), child_box(expected, dimension, threshold, False), precision, counts
    )
    return max(left_upper, right_upper)


def verify(raw: object, precision: int) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-stationary-cover-v1":
        raise ValueError("unsupported stationary cover certificate schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("unexpected code parameters")
    eta = leaf_checker.parse_fraction(raw.get("eta"))
    delta = leaf_checker.parse_fraction(raw.get("delta"))
    if eta != Fraction(1, 500) or delta != Fraction(11, 50):
        raise ValueError("unexpected support cutoff or target distance")
    expected = root_box(eta, delta)
    same_box(raw.get("root_box"), expected)
    counts = {"splits": 0, "leaves": 0, "infeasible": 0, "constraints": 0}
    worst = verify_node(raw.get("tree"), expected, precision, counts)
    if worst >= ZERO:
        raise ValueError("cover contains a nonnegative leaf")
    return {
        "status": "verified rational stationary split-tree cover",
        "eta": text(eta),
        "delta": text(delta),
        "splits_checked": counts["splits"],
        "leaves_checked": counts["leaves"],
        "infeasible_leaves_checked": counts["infeasible"],
        "linear_constraints_checked": counts["constraints"],
        "worst_leaf_upper": float(worst),
        "worst_leaf_upper_numerator": str(worst.numerator),
        "worst_leaf_upper_denominator": str(worst.denominator),
        "coverage": "exact recursive bisection of the full bulk composition box",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    raw = json.loads(args.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(raw, args.precision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
