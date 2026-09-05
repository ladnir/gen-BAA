#!/usr/bin/env python3
"""Generate one rational stationary-dual certificate on a small box."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv
from scipy.optimize import linprog

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected
import verify_perron_point_certificate as intervals
import verify_stationary_box_certificate as checker


ZERO = Fraction(0)
ONE = Fraction(1)


def text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def rational_nearest(value: float, denominator: int) -> Fraction:
    return Fraction(round(value * denominator), denominator)


def rational_distribution(
    values: np.ndarray, denominator: int, canonical: bool = False
) -> tuple[Fraction, ...]:
    result = [rational_nearest(float(value), denominator) for value in values[:4]]
    result.append(ONE - sum(result, ZERO))
    if canonical:
        for index in range(2, 5):
            if result[index] > result[1]:
                excess = result[index] - result[1]
                result[index] -= excess
                result[0] += excess
    if any(value <= ZERO for value in result):
        raise ValueError("rationalized reference is not an interior distribution")
    return tuple(result)


def rational_box(
    reference: np.ndarray,
    root_lower: np.ndarray,
    root_upper: np.ndarray,
    radius: float,
    denominator: int,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    lower = []
    upper = []
    for value, root_left, root_right in zip(
        reference, root_lower, root_upper, strict=True
    ):
        left = max(float(root_left), float(value) - radius)
        right = min(float(root_right), float(value) + radius)
        lower.append(Fraction(math.floor(left * denominator), denominator))
        upper.append(Fraction(math.ceil(right * denominator), denominator))
    return tuple(lower), tuple(upper)


def rational_cut(
    increment: np.ndarray,
    state: np.ndarray,
    denominator: int,
    precision: int,
) -> dict[str, object]:
    fitted = stationary.fit_stationary_dual(increment, state)
    if (
        fitted.potentials is None
        or fitted.shift is None
        or fitted.equality_residual > 1e-6
    ):
        raise RuntimeError(
            "floating stationary fit failed at the reference: "
            f"success={fitted.optimizer_success}, "
            f"equality_residual={fitted.equality_residual}, "
            f"has_potentials={fitted.potentials is not None}, "
            f"has_shift={fitted.shift is not None}"
        )
    potentials = tuple(
        rational_nearest(float(value), denominator)
        for value in fitted.potentials
    )
    raw_shift = max(
        checker.cut_log_kernel(potentials, difference, source)
        for difference in range(5)
        for source in range(5)
    )
    shift = raw_shift
    iv.dps = precision
    kernel = []
    for difference in range(5):
        row = []
        for source in range(5):
            exponent = (
                checker.cut_log_kernel(potentials, difference, source) - shift
            )
            row.append(
                intervals.rational_interval(
                    iv.exp(intervals.iv_fraction(exponent))
                ).upper
            )
        kernel.append(tuple(row))
    return {
        "potentials": [text(value) for value in potentials],
        "shift": text(shift),
        "kernel_upper": [
            [text(value) for value in row] for row in kernel
        ],
        "fit_residual": fitted.equality_residual,
    }


def parsed_cut(raw: dict[str, object], precision: int) -> checker.RationalCut:
    return checker.parse_cut(raw, precision)


def partition(
    left: tuple[Fraction, ...],
    kernel: tuple[tuple[Fraction, ...], ...],
    right: tuple[Fraction, ...],
) -> Fraction:
    return sum(
        (
            left[i] * kernel[i][j] * right[j]
            for i in range(5)
            for j in range(5)
        ),
        ZERO,
    )


def floating_lp(
    constraints: list[checker.LinearConstraint],
    objective: tuple[Fraction, ...],
) -> object:
    matrix = np.zeros((len(constraints), checker.VARIABLES))
    targets = np.zeros(len(constraints))
    for row, constraint in enumerate(constraints):
        targets[row] = float(constraint.target)
        for index, coefficient in constraint.coefficients.items():
            matrix[row, index] = float(coefficient)
    result = linprog(
        -np.asarray([float(value) for value in objective]),
        A_ub=matrix,
        b_ub=targets,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"floating certificate LP failed: {result.message}")
    return result


def rational_dual(
    result: object,
    constraints: list[checker.LinearConstraint],
    objective: tuple[Fraction, ...],
    denominator: int,
) -> dict[str, Fraction]:
    multipliers = [
        max(ZERO, Fraction(float(-value)).limit_denominator(denominator))
        for value in result.ineqlin.marginals
    ]
    dual_left = [ZERO] * checker.VARIABLES
    for multiplier, constraint in zip(multipliers, constraints, strict=True):
        if multiplier == ZERO:
            continue
        for index, coefficient in constraint.coefficients.items():
            dual_left[index] += multiplier * coefficient
    by_name = {constraint.name: index for index, constraint in enumerate(constraints)}
    for index, coefficient in enumerate(objective):
        deficit = coefficient - dual_left[index]
        if deficit > ZERO:
            row = by_name[f"variable_{index}_upper"]
            multipliers[row] += deficit
            dual_left[index] += deficit
    return {
        constraint.name: multiplier
        for constraint, multiplier in zip(constraints, multipliers, strict=True)
        if multiplier > ZERO
    }


def generate_box(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    reference: np.ndarray,
    args: argparse.Namespace,
    generation: dict[str, object] | None = None,
    require_negative: bool = True,
) -> dict[str, object]:
    reference = projected.sanitize_composition(reference)
    if any(float(value) < float(left) - 1e-8 or float(value) > float(right) + 1e-8
           for value, left, right in zip(reference, lower, upper, strict=True)):
        raise ValueError("reference is outside the certificate box")
    first_raw = rational_cut(
        reference[0:5],
        reference[5:10],
        args.potential_denominator,
        args.precision,
    )
    second_raw = rational_cut(
        reference[5:10],
        reference[10:15],
        args.potential_denominator,
        args.precision,
    )
    first = parsed_cut(first_raw, args.precision)
    second = parsed_cut(second_raw, args.precision)
    message_reference = rational_distribution(
        reference[0:5], args.reference_denominator, canonical=True
    )
    middle_reference = rational_distribution(
        reference[5:10], args.reference_denominator
    )
    output_reference = rational_distribution(
        reference[10:15], args.reference_denominator
    )
    partition_references = (
        partition(message_reference, first.kernel, middle_reference),
        partition(middle_reference, second.kernel, output_reference),
    )
    raw = {
        "schema": "f5-stationary-box-v1",
        "prime": 5,
        "repetition": 3,
        "eta": "1/500",
        "delta": "11/50",
        "box": {
            "lower": [text(value) for value in lower],
            "upper": [text(value) for value in upper],
        },
        "entropy_reference": [text(value) for value in message_reference],
        "composition_witness": [
            text(value)
            for value in (
                *message_reference,
                *middle_reference,
                *output_reference,
            )
        ],
        "partition_references": [
            text(value) for value in partition_references
        ],
        "first_cut": first_raw,
        "second_cut": second_raw,
        "dual": {},
        "generation": {"status": "untrusted floating proposal", **(generation or {})},
    }
    constant, objective = checker.objective_upper(
        raw, first, second, args.precision
    )
    constraints = checker.build_constraints(lower, upper)
    result = floating_lp(constraints, objective)
    dual = rational_dual(
        result, constraints, objective, args.dual_denominator
    )
    raw["dual"] = {name: text(value) for name, value in dual.items()}
    raw["generation"]["floating_lp_upper"] = float(constant - result.fun)
    raw["generation"]["nonzero_dual_multipliers"] = len(dual)
    if require_negative:
        checker.verify(raw, args.precision)
    return raw


def generate(args: argparse.Namespace) -> dict[str, object]:
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    entry = sorted(frontier["queue"], key=lambda value: value["priority"])[
        args.frontier_index
    ]["evaluated"]
    root_lower = np.asarray(entry["lower"], dtype=float)
    root_upper = np.asarray(entry["upper"], dtype=float)
    if args.use_frontier_point:
        reference = projected.sanitize_composition(np.asarray(entry["point"], dtype=float))
    else:
        reference_data = json.loads(args.reference.read_text(encoding="utf-8"))
        reference = projected.sanitize_composition(np.asarray(reference_data["maximizer"], dtype=float))
    lower, upper = rational_box(reference, root_lower, root_upper, args.radius, args.box_denominator)
    return generate_box(
        lower,
        upper,
        reference,
        args,
        {
            "radius": args.radius,
            "frontier_index": args.frontier_index,
            "reference_file": None if args.use_frontier_point else args.reference.name,
            "reference_source": "frontier_lp_point" if args.use_frontier_point else "reference_file",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=166)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/f5-fixed-stationary-shared-bank.json"),
    )
    parser.add_argument("--use-frontier-point", action="store_true")
    parser.add_argument("--radius", type=float, default=5e-4)
    parser.add_argument("--box-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--reference-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--potential-denominator", type=int, default=100_000_000)
    parser.add_argument("--dual-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--precision", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-stationary-box-certificate-0.22.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    certificate = generate(arguments)
    projected.write_json_atomic(arguments.output, certificate)
    print(json.dumps(checker.verify(certificate, arguments.precision), indent=2, sort_keys=True))
