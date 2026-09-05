#!/usr/bin/env python3
"""Generate one rational lifted projected-KL frontier-leaf certificate."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected
import generate_stationary_box_certificate as rational
import projected_kl_dual as projection
import verify_perron_point_certificate as intervals
import verify_projected_lifted_box_certificate as checker
import verify_stationary_cover_certificate as cover


ZERO = Fraction(0)
ONE = Fraction(1)


def text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def nearest(value: float, denominator: int) -> Fraction:
    return Fraction(round(value * denominator), denominator)


def rational_box(
    lower: np.ndarray, upper: np.ndarray, denominator: int
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    root_lower, root_upper = cover.root_box(Fraction(1, 500), Fraction(11, 50))
    result_lower = []
    result_upper = []
    for index, (left, right) in enumerate(zip(lower, upper, strict=True)):
        result_lower.append(
            max(root_lower[index], Fraction(math.floor(float(left) * denominator), denominator))
        )
        result_upper.append(
            min(root_upper[index], Fraction(math.ceil(float(right) * denominator), denominator))
        )
    return tuple(result_lower), tuple(result_upper)


def rational_distribution(values: np.ndarray, denominator: int) -> tuple[Fraction, ...]:
    positive = np.maximum(np.asarray(values, dtype=float), 1e-12)
    positive /= positive.sum()
    return rational.rational_distribution(positive, denominator)


def rational_projection_tangent(
    stage: int,
    increment: np.ndarray,
    state: np.ndarray,
    products: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    fitted = projection.fit_projection_dual(increment, state)
    if fitted.gradient_residual > 1e-6:
        raise RuntimeError("projected dual fit did not converge")
    target = tuple(nearest(float(value), args.potential_denominator) for value in fitted.target_potential)
    increment_support = nearest(fitted.increment_support_potential, args.potential_denominator)
    state_support = nearest(fitted.state_support_potential, args.potential_denominator)
    iv.dps = args.precision
    kernel = []
    for difference in range(5):
        row = []
        for source in range(5):
            exponent = (
                target[(difference + source) % 5]
                + increment_support * int(difference != 0)
                + state_support * int(source != 0)
            )
            row.append(
                intervals.rational_interval(
                    iv.exp(intervals.iv_fraction(exponent))
                ).upper
            )
        kernel.append(tuple(row))
    partition = sum(
        (kernel[left][right] * nearest(float(products[left, right]), args.reference_denominator)
         for left in range(5) for right in range(5)),
        ZERO,
    )
    if partition <= ZERO:
        raise RuntimeError("nonpositive rational partition reference")
    return {
        "stage": stage,
        "target_potential": [text(value) for value in target],
        "increment_support_potential": text(increment_support),
        "state_support_potential": text(state_support),
        "kernel_upper": [[text(value) for value in row] for row in kernel],
        "partition_reference": text(partition),
        "fit_gradient_residual": fitted.gradient_residual,
    }


def rational_stationary_tangent(
    stage: int,
    increment: np.ndarray,
    state: np.ndarray,
    products: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    cut_raw = rational.rational_cut(
        increment, state, args.potential_denominator, args.precision
    )
    cut = checker.domain.parse_cut(cut_raw, args.precision)
    partition = sum(
        (
            cut.kernel[left][right]
            * nearest(float(products[left, right]), args.reference_denominator)
            for left in range(5)
            for right in range(5)
        ),
        ZERO,
    )
    if partition <= ZERO:
        raise RuntimeError("nonpositive stationary partition reference")
    return {
        "stage": stage,
        "cut": cut_raw,
        "partition_reference": text(partition),
    }


def parsed_data(raw: dict[str, object], precision: int) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...], tuple[checker.ProjectionTangent, ...], tuple[tuple[tuple[Fraction, ...], Fraction], ...], tuple[checker.StationaryTangent, ...]]:
    lower, upper = checker.domain.parse_box(raw["box"])
    entropy = tuple(checker.parse_entropy_reference(value) for value in raw["entropy_references"])
    tangents = tuple(checker.parse_projection_tangent(value, precision) for value in raw["projection_tangents"])
    covariance = tuple(
        (
            intervals.parse_vector(value["direction"], checker.COMPOSITIONS),
            checker.parse_fraction(value["center"]),
        )
        for value in raw["covariance_cuts"]
    )
    stationary = tuple(
        checker.parse_stationary_tangent(value, precision)
        for value in raw.get("stationary_tangents", [])
    )
    return lower, upper, entropy, tangents, covariance, stationary


def floating_solve(raw: dict[str, object], precision: int) -> tuple[object, list[checker.Constraint]]:
    lower, upper, entropy, tangents, covariance, stationary = parsed_data(raw, precision)
    constraints = checker.build_constraints(
        lower, upper, entropy, tangents, covariance, precision, stationary
    )
    matrix = np.zeros((len(constraints), checker.VARIABLES))
    targets = np.zeros(len(constraints))
    for row_index, constraint in enumerate(constraints):
        targets[row_index] = float(constraint.target)
        for index, coefficient in constraint.coefficients.items():
            matrix[row_index, index] = float(coefficient)
    objective = np.zeros(checker.VARIABLES)
    objective[[checker.ENTROPY, checker.STAGE_ONE, checker.STAGE_TWO]] = -1.0
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=targets,
        bounds=[(None, None)] * checker.VARIABLES,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"lifted certificate LP failed: {result.message}")
    return result, constraints


def products(solution: np.ndarray, stage: int) -> np.ndarray:
    return np.array(
        [
            [solution[checker.product(stage, left, right)] for right in range(5)]
            for left in range(5)
        ]
    )


def covariance_cuts(solution: np.ndarray, denominator: int) -> list[dict[str, object]]:
    point = solution[: checker.COMPOSITIONS]
    matrix = np.empty((checker.COMPOSITIONS, checker.COMPOSITIONS))
    for left in range(checker.COMPOSITIONS):
        for right in range(checker.COMPOSITIONS):
            matrix[left, right] = solution[checker.moment(left, right)]
    covariance = matrix - np.outer(point, point)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    result = []
    for eigen_index in np.flatnonzero(eigenvalues < -1e-8):
        direction = tuple(
            nearest(float(value), denominator)
            for value in eigenvectors[:, eigen_index]
        )
        center = nearest(
            sum(float(value) * coordinate for value, coordinate in zip(direction, point, strict=True)),
            denominator,
        )
        result.append(
            {
                "direction": [text(value) for value in direction],
                "center": text(center),
            }
        )
    return result


def rational_dual(
    result: object,
    constraints: list[checker.Constraint],
    denominator: int,
) -> dict[str, str]:
    multipliers = [
        max(ZERO, Fraction(float(-value)).limit_denominator(denominator))
        for value in result.ineqlin.marginals
    ]
    dual_left = [ZERO] * checker.VARIABLES
    for multiplier, constraint in zip(multipliers, constraints, strict=True):
        for index, coefficient in constraint.coefficients.items():
            dual_left[index] += multiplier * coefficient
    objective = [ZERO] * checker.VARIABLES
    objective[checker.ENTROPY] = ONE
    objective[checker.STAGE_ONE] = ONE
    objective[checker.STAGE_TWO] = ONE
    by_name = {constraint.name: index for index, constraint in enumerate(constraints)}
    for index, coefficient in enumerate(objective):
        deficit = coefficient - dual_left[index]
        if deficit > ZERO:
            row = by_name[f"variable_{index}_upper"]
            multipliers[row] += deficit
        elif deficit < ZERO:
            row = by_name[f"variable_{index}_lower"]
            multipliers[row] -= deficit
    return {
        constraint.name: text(multiplier)
        for constraint, multiplier in zip(constraints, multipliers, strict=True)
        if multiplier > ZERO
    }


def generate(args: argparse.Namespace) -> dict[str, object]:
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    entry = sorted(frontier["queue"], key=lambda value: value["priority"])[args.frontier_index]["evaluated"]
    lower, upper = rational_box(
        np.asarray(entry["lower"], dtype=float),
        np.asarray(entry["upper"], dtype=float),
        args.box_denominator,
    )
    point = projected.sanitize_composition(np.asarray(entry["point"], dtype=float))
    if args.radius is not None:
        lower, upper = rational.rational_box(
            point,
            np.asarray(entry["lower"], dtype=float),
            np.asarray(entry["upper"], dtype=float),
            args.radius,
            args.box_denominator,
        )
    raw: dict[str, object] = {
        "schema": "f5-projected-lifted-box-v1",
        "prime": 5,
        "repetition": 3,
        "eta": "1/500",
        "delta": "11/50",
        "box": {
            "lower": [text(value) for value in lower],
            "upper": [text(value) for value in upper],
        },
        "entropy_references": [
            [text(value) for value in rational_distribution(point[:5], args.reference_denominator)]
        ],
        "projection_tangents": [],
        "stationary_tangents": [],
        "covariance_cuts": [],
        "dual": {},
        "generation": {
            "status": "untrusted floating proposal",
            "frontier_index": args.frontier_index,
            "radius": args.radius,
        },
    }
    initial_products = (
        np.outer(point[:5], point[5:10]),
        np.outer(point[5:10], point[10:15]),
    )
    raw["projection_tangents"] = [
        rational_projection_tangent(0, point[:5], point[5:10], initial_products[0], args),
        rational_projection_tangent(1, point[5:10], point[10:15], initial_products[1], args),
    ]
    raw["stationary_tangents"] = [
        rational_stationary_tangent(0, point[:5], point[5:10], initial_products[0], args),
        rational_stationary_tangent(1, point[5:10], point[10:15], initial_products[1], args),
    ]
    seen_entropy = set()
    seen_tangents = set()
    seen_covariance = set()
    seen_stationary = set()
    for _ in range(args.outer_iterations):
        result, _ = floating_solve(raw, args.precision)
        lp_point = projected.sanitize_composition(result.x[: checker.COMPOSITIONS])
        entropy_reference = [
            text(value)
            for value in rational_distribution(lp_point[:5], args.reference_denominator)
        ]
        entropy_key = tuple(entropy_reference)
        if entropy_key not in seen_entropy:
            raw["entropy_references"].append(entropy_reference)
            seen_entropy.add(entropy_key)
        for stage, increment_start, state_start in ((0, 0, 5), (1, 5, 10)):
            tangent = rational_projection_tangent(
                stage,
                lp_point[increment_start : increment_start + 5],
                lp_point[state_start : state_start + 5],
                products(result.x, stage),
                args,
            )
            tangent_key = json.dumps(tangent, sort_keys=True)
            if tangent_key not in seen_tangents:
                raw["projection_tangents"].append(tangent)
                seen_tangents.add(tangent_key)
            try:
                stationary = rational_stationary_tangent(
                    stage,
                    lp_point[increment_start : increment_start + 5],
                    lp_point[state_start : state_start + 5],
                    products(result.x, stage),
                    args,
                )
            except RuntimeError:
                stationary = None
            if stationary is not None:
                stationary_key = json.dumps(stationary, sort_keys=True)
                if stationary_key not in seen_stationary:
                    raw["stationary_tangents"].append(stationary)
                    seen_stationary.add(stationary_key)
        for cut in covariance_cuts(result.x, args.direction_denominator):
            cut_key = json.dumps(cut, sort_keys=True)
            if cut_key not in seen_covariance:
                raw["covariance_cuts"].append(cut)
                seen_covariance.add(cut_key)
    result, constraints = floating_solve(raw, args.precision)
    raw["dual"] = rational_dual(result, constraints, args.dual_denominator)
    raw["generation"].update(
        {
            "floating_lp_upper": float(-result.fun),
            "entropy_tangents": len(raw["entropy_references"]),
            "projection_tangents": len(raw["projection_tangents"]),
            "stationary_tangents": len(raw["stationary_tangents"]),
            "covariance_cuts": len(raw["covariance_cuts"]),
            "nonzero_dual_multipliers": len(raw["dual"]),
        }
    )
    if not args.allow_positive:
        checker.verify(raw, args.precision)
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", type=Path, default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"))
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--precision", type=int, default=100)
    parser.add_argument("--box-denominator", type=int, default=1_000_000_000_000)
    parser.add_argument("--reference-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--potential-denominator", type=int, default=100_000_000)
    parser.add_argument("--direction-denominator", type=int, default=100_000_000)
    parser.add_argument("--dual-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--output", type=Path, default=Path("results/f5-projected-lifted-worst-leaf-0.22.json"))
    parser.add_argument("--allow-positive", action="store_true")
    args = parser.parse_args()
    if args.precision < 50 or args.outer_iterations < 1:
        parser.error("invalid precision or iteration count")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    certificate = generate(arguments)
    projected.write_json_atomic(arguments.output, certificate)
    if arguments.allow_positive:
        print(json.dumps(certificate["generation"], indent=2, sort_keys=True))
    else:
        print(json.dumps(checker.verify(certificate, arguments.precision), indent=2, sort_keys=True))
