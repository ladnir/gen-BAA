#!/usr/bin/env python3
"""Optimize the seven-variable scalar cancellation relaxation of the F_5 tube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, minimize

import analyze_zero_payment_surface as zero
import optimize_conditional_product_bound as product


def scalar_cancellation_payment(
    increment_support: float,
    crossing: float,
    coordinate_radius: float,
) -> float:
    both_nonzero = increment_support - crossing
    if both_nonzero <= 0.0:
        return 0.0
    observed = crossing / both_nonzero
    lower = max(0.0, 0.25 - coordinate_radius)
    upper = min(1.0, 0.25 + coordinate_radius)
    reference = float(np.clip(observed, lower, upper))
    return both_nonzero * product.relative_entropy(
        np.array([observed, 1.0 - observed]),
        np.array([reference, 1.0 - reference]),
    )


def message_cancellation_tradeoff(
    increment_support: float, crossing: float
) -> float:
    """Lower-bound message entropy deficit plus first-stage cancellation KL."""
    both_nonzero = increment_support - crossing
    defect = max(0.0, 5.0 * crossing - increment_support)
    denominator = 8.0 * both_nonzero * (
        4.0 * increment_support - 3.0 * crossing
    )
    if defect <= 0.0 or denominator <= 0.0:
        return 0.0
    return increment_support * defect * defect / denominator


def scalar_tube_defect(point: np.ndarray) -> float:
    a, b, _, x, y, first_radius, second_radius = point
    return float(
        1.25 * (x - a / 5.0) ** 2
        + (4.0 / 3.0) * (a - x) ** 2 * first_radius**2
        + 1.25 * (y - b / 5.0) ** 2
        + (4.0 / 3.0) * (b - y) ** 2 * second_radius**2
    )


def scalar_upper(point: np.ndarray, repetition: int) -> float:
    a, b, c, x, y, first_radius, second_radius = point
    message_support_entropy = product.entropy(np.array([1.0 - a, a]))
    value = message_support_entropy / repetition
    value += a * np.log(4.0) / repetition
    value -= product.support_information(a, b, x)
    value -= product.support_information(b, c, y)
    value -= max(
        scalar_cancellation_payment(a, x, first_radius),
        message_cancellation_tradeoff(a, x),
    )
    value -= scalar_cancellation_payment(b, y, second_radius)
    return float(value)


def feasibility(point: np.ndarray, eta: float, delta: float) -> np.ndarray:
    a, b, c, x, y, _, _ = point
    return np.array(
        [
            a - eta,
            delta - c,
            x,
            a / 2.0 - x,
            x - a + b,
            1.0 - b - x,
            y,
            b / 2.0 - y,
            y - b + c,
            1.0 - c - y,
        ]
    )


def zero_start(a: float, delta: float) -> np.ndarray:
    b = min(0.9 * a, 1.2 * delta)
    b = max(b, 0.8 * a)
    return np.array([a, b, delta, a / 5.0, b / 5.0, 0.0, 0.0])


def optimize(args: argparse.Namespace) -> dict[str, object]:
    random = np.random.default_rng(args.seed)
    maximum_a = min(4.0 * args.delta, 25.0 * args.delta / 16.0)
    starts = [zero_start(args.eta, args.delta)]
    for index in range(max(1, args.restarts - 1)):
        fraction = (index + 0.5) / max(1, args.restarts - 1)
        a = args.eta * (maximum_a / args.eta) ** fraction
        start = zero_start(a, args.delta)
        start[1] *= random.uniform(0.9, 1.1)
        start[1] = np.clip(start[1], 0.8 * a, min(1.25 * args.delta, 1.2 * a))
        start[4] = start[1] / 5.0
        starts.append(start)

    bounds = Bounds(
        np.array([args.eta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array(
            [
                4.0 * args.delta,
                2.0 * args.delta,
                args.delta,
                2.0 * args.delta,
                args.delta,
                0.75,
                0.75,
            ]
        ),
    )
    constraints = (
        {
            "type": "ineq",
            "fun": lambda point: feasibility(point, args.eta, args.delta),
        },
        {
            "type": "ineq",
            "fun": lambda point: args.maximum_defect
            - scalar_tube_defect(point),
        },
    )
    best = None
    rejected = 0
    for start in starts:
        result = minimize(
            lambda point: -scalar_upper(point, args.repetition),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": args.tolerance,
                "maxiter": args.maximum_iterations,
                "disp": False,
            },
        )
        feasibility_error = max(0.0, -float(np.min(feasibility(
            result.x, args.eta, args.delta
        ))))
        defect_error = max(
            0.0, scalar_tube_defect(result.x) - args.maximum_defect
        )
        if max(feasibility_error, defect_error) > args.feasibility_tolerance:
            rejected += 1
            continue
        value = scalar_upper(result.x, args.repetition)
        if np.isfinite(value) and (best is None or value > best[0]):
            best = (value, result)
    if best is None:
        raise RuntimeError("no scalar-tube restart returned a feasible point")
    value, result = best
    point = result.x
    report = {
        "prime": 5,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "maximum_defect": args.maximum_defect,
        "maximum_upper_bound_found": value,
        "point": {
            "A": float(point[0]),
            "B": float(point[1]),
            "C": float(point[2]),
            "x1": float(point[3]),
            "x2": float(point[4]),
            "z1": float(point[5]),
            "z2": float(point[6]),
        },
        "defect": scalar_tube_defect(point),
        "first_payment": scalar_cancellation_payment(
            point[0], point[3], point[5]
        ),
        "second_payment": scalar_cancellation_payment(
            point[1], point[4], point[6]
        ),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "restarts": len(starts),
        "rejected_infeasible_restarts": rejected,
        "status": (
            "floating seven-variable scalar relaxation; not a global certificate"
        ),
    }
    zero.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--maximum-defect", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8081)
    parser.add_argument("--maximum-iterations", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-scalar-tube.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(optimize(arguments), indent=2, sort_keys=True))
