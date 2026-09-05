#!/usr/bin/env python3
"""Independently replay the rational F_5 zero-payment surface cover."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from mpmath import iv

import verify_perron_point_certificate as intervals


ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)
THREE = Fraction(3)
FOUR = Fraction(4)
FIVE = Fraction(5)


@dataclass(frozen=True)
class Box:
    a_lower: Fraction
    a_upper: Fraction
    b_lower: Fraction
    b_upper: Fraction


Point = tuple[Fraction, Fraction]
AffineCell = tuple[Fraction, Fraction, Fraction]


def parse_fraction(value: object) -> Fraction:
    return intervals.parse_fraction(value)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_box(value: object) -> Box:
    bounds = intervals.parse_vector(value, 4)
    box = Box(*bounds)
    if box.a_lower >= box.a_upper or box.b_lower >= box.b_upper:
        raise ValueError("certificate leaf is not a nondegenerate box")
    return box


def cross(origin: Point, left: Point, right: Point) -> Fraction:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (
        left[1] - origin[1]
    ) * (right[0] - origin[0])


def convex_hull(points: set[Point]) -> tuple[Point, ...]:
    ordered = sorted(points)
    if len(ordered) <= 1:
        return tuple(ordered)
    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= ZERO:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= ZERO:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def polygon(box: Box) -> tuple[Point, ...]:
    # Each row represents c_A A+c_B B<=target.
    inequalities = (
        (-ONE, ZERO, -box.a_lower),
        (ONE, ZERO, box.a_upper),
        (ZERO, -ONE, -box.b_lower),
        (ZERO, ONE, box.b_upper),
        (FOUR / FIVE, -ONE, ZERO),
    )
    points: set[Point] = set()
    for first, second in combinations(inequalities, 2):
        determinant = first[0] * second[1] - first[1] * second[0]
        if determinant == ZERO:
            continue
        a = (first[2] * second[1] - first[1] * second[2]) / determinant
        b = (first[0] * second[2] - first[2] * second[0]) / determinant
        if all(ca * a + cb * b <= target for ca, cb, target in inequalities):
            points.add((a, b))
    return convex_hull(points)


def polygon_area(vertices: tuple[Point, ...]) -> Fraction:
    if len(vertices) < 3:
        return ZERO
    signed = sum(
        left[0] * right[1] - left[1] * right[0]
        for left, right in zip(vertices, vertices[1:] + vertices[:1], strict=True)
    )
    return abs(signed) / TWO


def intersection_box(left: Box, right: Box) -> Box | None:
    a_lower = max(left.a_lower, right.a_lower)
    a_upper = min(left.a_upper, right.a_upper)
    b_lower = max(left.b_lower, right.b_lower)
    b_upper = min(left.b_upper, right.b_upper)
    if a_lower >= a_upper or b_lower >= b_upper:
        return None
    return Box(a_lower, a_upper, b_lower, b_upper)


def support_cells(increment: Fraction, state: Fraction) -> tuple[Fraction, ...]:
    crossing = increment / FIVE
    return (
        ONE - state - crossing,
        state - increment + crossing,
        crossing,
        increment - crossing,
    )


def entropy_lower(value: Fraction) -> Fraction:
    if not ZERO <= value <= ONE:
        raise ValueError("binary entropy input is outside [0,1]")
    return intervals.entropy((value, ONE - value)).lower


def entropy_secant_lower(
    lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction]:
    lower_value = entropy_lower(lower)
    upper_value = entropy_lower(upper)
    slope = (upper_value - lower_value) / (upper - lower)
    return slope, lower_value - slope * lower


def negative_log_upper(value: Fraction) -> Fraction:
    return intervals.log_interval(value).scale(-ONE).upper


def evaluate_affine_cell(cell: AffineCell, point: Point) -> Fraction:
    return cell[0] + cell[1] * point[0] + cell[2] * point[1]


def add_cross_entropy_tangent(
    constant: Fraction,
    a_coefficient: Fraction,
    b_coefficient: Fraction,
    cells: tuple[AffineCell, ...],
    reference: Point,
) -> tuple[Fraction, Fraction, Fraction]:
    probabilities = tuple(evaluate_affine_cell(cell, reference) for cell in cells)
    if any(value <= ZERO for value in probabilities) or sum(probabilities, ZERO) != ONE:
        raise ValueError("cross-entropy reference is not a positive distribution")
    for cell, probability in zip(cells, probabilities, strict=True):
        coefficient = negative_log_upper(probability)
        constant += coefficient * cell[0]
        a_coefficient += coefficient * cell[1]
        b_coefficient += coefficient * cell[2]
    return constant, a_coefficient, b_coefficient


def leaf_upper(box: Box, delta: Fraction) -> Fraction:
    vertices = polygon(box)
    if len(vertices) < 3 or polygon_area(vertices) == ZERO:
        raise ValueError("certificate contains an empty leaf")
    count = Fraction(len(vertices))
    reference = (
        sum((point[0] for point in vertices), ZERO) / count,
        sum((point[1] for point in vertices), ZERO) / count,
    )

    constant = -entropy_lower(delta)
    a_coefficient = intervals.log_interval(FOUR).upper / THREE
    b_coefficient = ZERO
    for coordinate, factor, lower, upper in (
        (0, TWO / THREE, box.a_lower, box.a_upper),
        (1, TWO, box.b_lower, box.b_upper),
    ):
        slope, intercept = entropy_secant_lower(lower, upper)
        constant -= factor * intercept
        if coordinate == 0:
            a_coefficient -= factor * slope
        else:
            b_coefficient -= factor * slope

    first_cells: tuple[AffineCell, ...] = (
        (ONE, -ONE / FIVE, -ONE),
        (ZERO, -FOUR / FIVE, ONE),
        (ZERO, ONE / FIVE, ZERO),
        (ZERO, FOUR / FIVE, ZERO),
    )
    second_cells: tuple[AffineCell, ...] = (
        (ONE - delta, ZERO, -ONE / FIVE),
        (delta, ZERO, -FOUR / FIVE),
        (ZERO, ZERO, ONE / FIVE),
        (ZERO, ZERO, FOUR / FIVE),
    )
    constant, a_coefficient, b_coefficient = add_cross_entropy_tangent(
        constant,
        a_coefficient,
        b_coefficient,
        first_cells,
        reference,
    )
    constant, a_coefficient, b_coefficient = add_cross_entropy_tangent(
        constant,
        a_coefficient,
        b_coefficient,
        second_cells,
        reference,
    )
    return max(
        constant + a_coefficient * a + b_coefficient * b for a, b in vertices
    )


def verify_cover(leaves: tuple[Box, ...], eta: Fraction, delta: Fraction) -> Fraction:
    b_upper = FIVE * delta / FOUR
    a_upper = FIVE * b_upper / FOUR
    root = Box(eta, a_upper, FOUR * eta / FIVE, b_upper)
    root_area = polygon_area(polygon(root))
    total_area = ZERO
    for leaf in leaves:
        if (
            leaf.a_lower < root.a_lower
            or leaf.a_upper > root.a_upper
            or leaf.b_lower < root.b_lower
            or leaf.b_upper > root.b_upper
        ):
            raise ValueError("certificate leaf leaves the root rectangle")
        area = polygon_area(polygon(leaf))
        if area == ZERO:
            raise ValueError("certificate contains a zero-area leaf")
        total_area += area
    for left, right in combinations(leaves, 2):
        overlap = intersection_box(left, right)
        if overlap is not None and polygon_area(polygon(overlap)) > ZERO:
            raise ValueError("certificate leaves overlap in positive area")
    if total_area != root_area:
        raise ValueError("certificate leaves do not cover the reduced domain")
    return root_area


def verify(raw: object, precision: int) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-zero-payment-surface-v1":
        raise ValueError("unsupported zero-payment certificate schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("unexpected code parameters")
    eta = parse_fraction(raw.get("eta"))
    delta = parse_fraction(raw.get("delta"))
    target = parse_fraction(raw.get("target_upper"))
    if eta != Fraction(1, 500) or delta != Fraction(11, 50):
        raise ValueError("unexpected support cutoff or output target")
    if target >= ZERO:
        raise ValueError("certificate target must be negative")
    raw_leaves = raw.get("leaves")
    if not isinstance(raw_leaves, list) or not raw_leaves:
        raise ValueError("certificate has no leaves")
    leaves = tuple(
        parse_box(leaf.get("bounds"))
        for leaf in raw_leaves
        if isinstance(leaf, dict)
    )
    if len(leaves) != len(raw_leaves):
        raise ValueError("invalid certificate leaf")

    iv.dps = precision
    area = verify_cover(leaves, eta, delta)
    uppers = tuple(leaf_upper(leaf, delta) for leaf in leaves)
    maximum = max(uppers)
    if maximum > target:
        raise ValueError("a leaf does not meet the requested negative target")
    return {
        "status": "verified rational zero-payment surface certificate",
        "eta": fraction_text(eta),
        "delta": fraction_text(delta),
        "target_upper": fraction_text(target),
        "maximum_upper": float(maximum),
        "maximum_upper_numerator": str(maximum.numerator),
        "maximum_upper_denominator": str(maximum.denominator),
        "maximum_upper_is_at_most_target": True,
        "leaves_checked": len(leaves),
        "covered_area": fraction_text(area),
        "pairwise_interior_disjointness_checked": True,
        "interval_precision": precision,
        "arithmetic": (
            "exact rational boxes and polygon vertices, binary-entropy secants, "
            "and directed logarithm intervals"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    certificate = json.loads(arguments.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(certificate, arguments.precision), indent=2, sort_keys=True))
