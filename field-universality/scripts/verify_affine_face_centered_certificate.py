#!/usr/bin/env python3
"""Replay exact algebra in centered-curvature face certificates."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ZERO = Fraction(0)
ONE = Fraction(1)


def fractions(values):
    return [Fraction(value) for value in values]


def ldl_pivots(matrix: list[list[Fraction]]) -> list[Fraction] | None:
    dimension = len(matrix)
    lower = [[ZERO for _ in range(dimension)] for _ in range(dimension)]
    pivots = [ZERO for _ in range(dimension)]
    for i in range(dimension):
        lower[i][i] = ONE
        pivot = matrix[i][i] - sum(
            (lower[i][k] * lower[i][k] * pivots[k] for k in range(i)),
            ZERO,
        )
        if pivot <= ZERO:
            return None
        pivots[i] = pivot
        for j in range(i + 1, dimension):
            lower[j][i] = (
                matrix[j][i]
                - sum(
                    (
                        lower[j][k] * lower[i][k] * pivots[k]
                        for k in range(i)
                    ),
                    ZERO,
                )
            ) / pivot
    return pivots


def verify_certificate(certificate: dict[str, Any]) -> None:
    lower = [fractions(row) for row in certificate["hessian_interval_lower"]]
    upper = [fractions(row) for row in certificate["hessian_interval_upper"]]
    dimension = len(certificate["variables"])
    if len(lower) != dimension or len(upper) != dimension:
        raise ValueError("Hessian certificate has the wrong dimension")
    if any(len(row) != dimension for row in [*lower, *upper]):
        raise ValueError("Hessian certificate is not square")
    for i in range(dimension):
        for j in range(dimension):
            if lower[i][j] > upper[i][j]:
                raise ValueError("Hessian interval is reversed")
            if lower[i][j] != lower[j][i] or upper[i][j] != upper[j][i]:
                raise ValueError("Hessian interval is not symmetric")
    midpoint = [
        [(lower[i][j] + upper[i][j]) / 2 for j in range(dimension)]
        for i in range(dimension)
    ]
    error = [
        [(upper[i][j] - lower[i][j]) / 2 for j in range(dimension)]
        for i in range(dimension)
    ]
    row_error = max(sum(row, ZERO) for row in error)
    if row_error != Fraction(certificate["row_sum_error_bound"]):
        raise ValueError("row-sum error bound does not replay")
    curvature = Fraction(certificate["certified_curvature"])
    if curvature <= ZERO:
        raise ValueError("certified curvature must be positive")
    negative_upper = [
        [
            -midpoint[i][j]
            - (row_error + curvature if i == j else ZERO)
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    pivots = ldl_pivots(negative_upper)
    if pivots is None or pivots != fractions(certificate["ldl_pivots"]):
        raise ValueError("exact LDL pivots do not replay")

    radii = fractions(certificate["radii"])
    gradients = fractions(certificate["gradient_absolute_upper"])
    expected_displacements = [
        min(radius, gradient / curvature)
        for radius, gradient in zip(radii, gradients, strict=True)
    ]
    if expected_displacements != fractions(certificate["maximizing_displacements"]):
        raise ValueError("quadratic maximizing displacements do not replay")
    expected_contributions = [
        gradient * displacement - curvature * displacement * displacement / 2
        for gradient, displacement in zip(
            gradients, expected_displacements, strict=True
        )
    ]
    if expected_contributions != fractions(certificate["quadratic_contributions"]):
        raise ValueError("quadratic contributions do not replay")
    total = Fraction(certificate["center_log_value_upper"]) + sum(
        expected_contributions, ZERO
    )
    if total != Fraction(certificate["total_log_upper"]):
        raise ValueError("total logarithmic upper bound does not replay")
    if certificate["closed"] != (total < ZERO):
        raise ValueError("closed flag disagrees with exact total")


def verify_taylor_certificate(certificate: dict[str, Any]) -> None:
    linear = fractions(certificate["linear_term_upper"])
    quadratic = [fractions(row) for row in certificate["quadratic_term_upper"]]
    dimension = len(certificate["variables"])
    if len(linear) != dimension or len(quadratic) != dimension:
        raise ValueError("Taylor certificate has the wrong dimension")
    if any(len(row) != dimension for row in quadratic):
        raise ValueError("Taylor quadratic table is not square")
    if any(value < ZERO for value in linear):
        raise ValueError("Taylor linear upper contribution is negative")
    if any(value < ZERO for row in quadratic for value in row):
        raise ValueError("Taylor quadratic upper contribution is negative")
    total = (
        Fraction(certificate["center_log_value_upper"])
        + sum(linear, ZERO)
        + sum((sum(row, ZERO) for row in quadratic), ZERO)
    )
    if total != Fraction(certificate["total_log_upper"]):
        raise ValueError("Taylor total logarithmic upper bound does not replay")
    if certificate["closed"] != (total < ZERO):
        raise ValueError("Taylor closed flag disagrees with exact total")


def certificate_box(certificate: dict[str, Any], variables: list[int]):
    lower = fractions(certificate["box_lower"])
    upper = fractions(certificate["box_upper"])
    return tuple(lower[index] for index in variables), tuple(
        upper[index] for index in variables
    )


def verify_guillotine_cover(
    parent_lower: tuple[Fraction, ...],
    parent_upper: tuple[Fraction, ...],
    boxes: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
) -> None:
    """Reconstruct an axis-aligned subdivision tree over the free coordinates."""

    dimension = len(parent_lower)
    if len(parent_upper) != dimension:
        raise ValueError("cover parent has inconsistent dimensions")
    for lower, upper in boxes:
        if len(lower) != dimension or len(upper) != dimension:
            raise ValueError("cover leaf has inconsistent dimensions")
        if any(
            parent_lower[i] > lower[i]
            or lower[i] > upper[i]
            or upper[i] > parent_upper[i]
            for i in range(dimension)
        ):
            raise ValueError("cover leaf lies outside the parent box")

    memo: dict[tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[int, ...]], bool] = {}

    def covered(
        lower: tuple[Fraction, ...],
        upper: tuple[Fraction, ...],
        active: tuple[int, ...],
    ) -> bool:
        key = (lower, upper, active)
        if key in memo:
            return memo[key]
        for index in active:
            box_lower, box_upper = boxes[index]
            if all(
                box_lower[i] <= lower[i] and upper[i] <= box_upper[i]
                for i in range(dimension)
            ):
                memo[key] = True
                return True
        for coordinate in range(dimension):
            thresholds = sorted(
                {
                    value
                    for index in active
                    for value in (boxes[index][0][coordinate], boxes[index][1][coordinate])
                    if lower[coordinate] < value < upper[coordinate]
                }
            )
            for threshold in thresholds:
                crossing = [
                    index
                    for index in active
                    if boxes[index][0][coordinate] < threshold
                    < boxes[index][1][coordinate]
                ]
                if crossing:
                    continue
                left_active = tuple(
                    index
                    for index in active
                    if boxes[index][0][coordinate] < threshold
                    and boxes[index][1][coordinate] > lower[coordinate]
                )
                right_active = tuple(
                    index
                    for index in active
                    if boxes[index][1][coordinate] > threshold
                    and boxes[index][0][coordinate] < upper[coordinate]
                )
                if not left_active or not right_active:
                    continue
                left_upper = list(upper)
                left_upper[coordinate] = threshold
                right_lower = list(lower)
                right_lower[coordinate] = threshold
                if covered(lower, tuple(left_upper), left_active) and covered(
                    tuple(right_lower), upper, right_active
                ):
                    memo[key] = True
                    return True
        memo[key] = False
        return False

    if not boxes or not covered(
        parent_lower, parent_upper, tuple(range(len(boxes)))
    ):
        raise ValueError("closed certificate boxes do not cover the initial box")


def verify_closed_result_cover(payload: dict[str, Any]) -> None:
    if payload.get("pending_boxes") != 0:
        raise ValueError("result is not a closed cover")
    variables = [int(value) for value in payload["variables"]]
    initial = payload.get("initial_box")
    if initial is None:
        raise ValueError("closed result does not record its initial box")
    parent_lower = tuple(Fraction(initial["lower"][index]) for index in variables)
    parent_upper = tuple(Fraction(initial["upper"][index]) for index in variables)
    certificates = [
        *payload.get("exact_centered_curvature_certificates", []),
        *payload.get("exact_directed_taylor_certificates", []),
    ]
    closed_records = payload.get("closed_box_records", [])
    certificate_by_box = {
        certificate_box(certificate, variables): certificate
        for certificate in certificates
    }
    record_boxes = [
        (
            tuple(Fraction(record["lower"][index]) for index in variables),
            tuple(Fraction(record["upper"][index]) for index in variables),
        )
        for record in closed_records
    ]
    if len(certificate_by_box) != len(certificates):
        raise ValueError("multiple exact certificates claim the same free-coordinate box")
    if set(record_boxes) != set(certificate_by_box):
        raise ValueError("closed boxes and exact certificate boxes do not match")
    verify_guillotine_cover(parent_lower, parent_upper, record_boxes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    certificates = payload.get("exact_centered_curvature_certificates", [])
    taylor_certificates = payload.get("exact_directed_taylor_certificates", [])
    if not certificates and not taylor_certificates:
        raise ValueError("result contains no exact face certificate")
    for certificate in certificates:
        verify_certificate(certificate)
    for certificate in taylor_certificates:
        verify_taylor_certificate(certificate)
    if payload.get("pending_boxes") == 0:
        verify_closed_result_cover(payload)
    summary = {
        "status": "verified exact face-bound algebra and closed-box cover",
        "curvature_certificates": len(certificates),
        "taylor_certificates": len(taylor_certificates),
        "closed_certificates": sum(
            bool(value["closed"])
            for value in [*certificates, *taylor_certificates]
        ),
        "scope": (
            "replays rational interval-matrix, LDL, Taylor, quadratic, and cover algebra; "
            "independent transcendental interval recomputation remains pending"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
