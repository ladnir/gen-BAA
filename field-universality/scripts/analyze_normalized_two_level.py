#!/usr/bin/env python3
"""Numerically audit support-normalized coercivity of the F_5 two-level toll."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import analyze_zero_payment_surface as zero


UNIFORM = np.full(4, 0.25)


def relative_entropy(actual: np.ndarray, reference: np.ndarray) -> float:
    if np.any(actual < 0.0) or np.any(reference <= 0.0):
        return float("inf")
    positive = actual > 0.0
    return float(
        np.sum(actual[positive] * np.log(actual[positive] / reference[positive]))
    )


def softmax_with_fixed_gauge(logits: np.ndarray) -> np.ndarray:
    full = np.concatenate((logits, np.zeros(1)))
    full -= np.max(full)
    weights = np.exp(full)
    return weights / weights.sum()


def convolution_nonzero(
    increment: np.ndarray, state: np.ndarray
) -> tuple[float, np.ndarray]:
    convolution = np.zeros(5)
    for difference in range(1, 5):
        for source in range(1, 5):
            convolution[(source + difference) % 5] += (
                increment[difference - 1] * state[source - 1]
            )
    cancellation = float(convolution[0])
    return cancellation, convolution[1:]


def scale_free_payment(
    support_ratio: float,
    crossing_ratio: float,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    """Return L/A for B/A=s and x/A=t."""
    s = support_ratio
    t = crossing_ratio
    cancellation, nonzero_convolution = convolution_nonzero(increment, state)
    both_nonzero = 1.0 - t
    observed = t / both_nonzero
    if not 0.0 < cancellation < 1.0:
        return float("inf")
    cancellation_cost = both_nonzero * relative_entropy(
        np.array([observed, 1.0 - observed]),
        np.array([cancellation, 1.0 - cancellation]),
    )
    reference = (s - 1.0 + t) * state + t * increment
    convolution_mass = 1.0 - 2.0 * t
    if convolution_mass > 0.0:
        reference += convolution_mass * nonzero_convolution / (1.0 - cancellation)
    reference /= s
    return cancellation_cost + s * relative_entropy(state, reference)


def scale_free_defect(crossing_ratio: float, state: np.ndarray) -> float:
    r = np.concatenate(
        (np.array([crossing_ratio]), (1.0 - crossing_ratio) * state)
    )
    return float(np.dot(r - 0.2, r - 0.2))


def residual_quadratic_ratio(
    crossing_ratio: float,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    """Return (2 e_0^2+||e_+||_2^2)/R for the two-level residuals."""
    cancellation, nonzero_convolution = convolution_nonzero(increment, state)
    t = crossing_ratio
    cancellation_residual = t - (1.0 - t) * cancellation
    conditional_residual = (1.0 - t) * state - t * increment
    if 1.0 - 2.0 * t > 0.0:
        conditional_residual -= (
            (1.0 - 2.0 * t)
            * nonzero_convolution
            / (1.0 - cancellation)
        )
    defect = scale_free_defect(t, state)
    if defect <= 1e-16:
        return float("inf")
    return float(
        (
            2.0 * cancellation_residual * cancellation_residual
            + np.dot(conditional_residual, conditional_residual)
        )
        / defect
    )


def normalized_ratio(
    support_ratio: float,
    crossing_ratio: float,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    defect = scale_free_defect(crossing_ratio, state)
    if defect <= 1e-16:
        return float("inf")
    return scale_free_payment(
        support_ratio, crossing_ratio, increment, state
    ) / defect


def large_ratio_family(support_ratio: float, state: np.ndarray) -> float:
    """Exact t=1/5, uniform-increment family that disproves A-normalization."""
    if support_ratio < 0.8:
        return float("nan")
    reference = (1.0 - 1.0 / support_ratio) * state
    reference += UNIFORM / support_ratio
    payment = support_ratio * relative_entropy(state, reference)
    defect = 0.64 * float(np.dot(state - UNIFORM, state - UNIFORM))
    return payment / defect


def decode(point: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return (
        float(point[0]),
        softmax_with_fixed_gauge(point[1:4]),
        softmax_with_fixed_gauge(point[4:7]),
    )


def decode_opposite_pair(
    point: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    crossing = float(point[0])
    first = 1.0 / (1.0 + np.exp(-float(point[1])))
    increment = np.array([0.0, first, 1.0 - first, 0.0])
    state = softmax_with_fixed_gauge(point[2:5])
    return crossing, increment, state


def decode_fixed_support(
    point: np.ndarray, support: tuple[int, ...]
) -> tuple[float, np.ndarray, np.ndarray]:
    if not support:
        raise ValueError("the increment support must be nonempty")
    crossing = float(point[0])
    increment = np.zeros(4)
    if len(support) == 1:
        increment[support[0]] = 1.0
        state_start = 1
    else:
        weights = softmax_with_fixed_gauge(point[1 : len(support)])
        for index, weight in zip(support, weights, strict=True):
            increment[index] = weight
        state_start = len(support)
    state = softmax_with_fixed_gauge(point[state_start : state_start + 3])
    return crossing, increment, state


def optimize_ratio(
    support_ratio: float,
    restarts: int,
    seed: int,
    maximum_iterations: int,
) -> dict[str, object]:
    lower_crossing = max(0.0, 1.0 - support_ratio)
    upper_crossing = 0.5
    random = np.random.default_rng(seed)

    def objective(point: np.ndarray) -> float:
        crossing, increment, state = decode(point)
        value = normalized_ratio(
            support_ratio, crossing, increment, state
        )
        return value if np.isfinite(value) else 1e12

    starts = [
        np.array([0.2, 0.0, 0.0, 0.0, -1.386294, 1.386294, 1.386294]),
        np.array([0.3, 1.0, -1.0, 0.5, -0.5, 0.8, -0.8]),
    ]
    for _ in range(restarts):
        crossing = random.uniform(lower_crossing, upper_crossing)
        starts.append(
            np.concatenate(
                (
                    np.array([crossing]),
                    random.normal(0.0, 2.0, size=6),
                )
            )
        )
    bounds = [(lower_crossing, upper_crossing)] + [(-16.0, 16.0)] * 6
    best = None
    for start in starts:
        start = start.copy()
        start[0] = np.clip(start[0], lower_crossing, upper_crossing)
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maximum_iterations, "ftol": 1e-13},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("normalized-ratio optimization produced no result")
    crossing, increment, state = decode(best.x)
    increment_ratio = normalized_ratio(
        support_ratio, crossing, increment, state
    )
    return {
        "support_ratio": support_ratio,
        "increment_normalized_ratio": increment_ratio,
        "max_support_normalized_ratio": max(1.0, support_ratio)
        * increment_ratio,
        "crossing_ratio": crossing,
        "increment_nonzero_law": increment.tolist(),
        "state_nonzero_law": state.tolist(),
        "scale_free_payment": scale_free_payment(
            support_ratio, crossing, increment, state
        ),
        "scale_free_defect": scale_free_defect(crossing, state),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def optimize_opposite_pair_ratio(
    support_ratio: float,
    restarts: int,
    seed: int,
    maximum_iterations: int,
) -> dict[str, object]:
    """Minimize on the boundary where the increment uses one sign pair."""
    lower_crossing = max(0.0, 1.0 - support_ratio)
    upper_crossing = 0.5
    random = np.random.default_rng(seed)

    def objective(point: np.ndarray) -> float:
        crossing, increment, state = decode_opposite_pair(point)
        value = normalized_ratio(
            support_ratio, crossing, increment, state
        )
        return value if np.isfinite(value) else 1e12

    starts = [
        np.array([0.2, 0.0, -0.5, 0.5, 0.5]),
        np.array([0.3, 0.0, 0.5, -0.5, -0.5]),
    ]
    for _ in range(restarts):
        starts.append(
            np.concatenate(
                (
                    np.array(
                        [
                            random.uniform(lower_crossing, upper_crossing),
                            random.normal(0.0, 2.0),
                        ]
                    ),
                    random.normal(0.0, 2.0, size=3),
                )
            )
        )
    bounds = [
        (lower_crossing, upper_crossing),
        (-20.0, 20.0),
    ] + [(-16.0, 16.0)] * 3
    best = None
    for start in starts:
        start = start.copy()
        start[0] = np.clip(start[0], lower_crossing, upper_crossing)
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maximum_iterations, "ftol": 1e-14},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("opposite-pair optimization produced no result")
    crossing, increment, state = decode_opposite_pair(best.x)
    increment_ratio = normalized_ratio(
        support_ratio, crossing, increment, state
    )
    return {
        "increment_normalized_ratio": increment_ratio,
        "max_support_normalized_ratio": max(1.0, support_ratio)
        * increment_ratio,
        "crossing_ratio": crossing,
        "increment_nonzero_law": increment.tolist(),
        "state_nonzero_law": state.tolist(),
        "scale_free_payment": scale_free_payment(
            support_ratio, crossing, increment, state
        ),
        "scale_free_defect": scale_free_defect(crossing, state),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def optimize_fixed_support_ratio(
    support_ratio: float,
    support: tuple[int, ...],
    restarts: int,
    seed: int,
    maximum_iterations: int,
) -> dict[str, object]:
    lower_crossing = max(0.0, 1.0 - support_ratio)
    upper_crossing = 0.5
    random = np.random.default_rng(seed)
    dimension = len(support) + 3

    def objective(point: np.ndarray) -> float:
        crossing, increment, state = decode_fixed_support(point, support)
        value = normalized_ratio(
            support_ratio, crossing, increment, state
        )
        return value if np.isfinite(value) else 1e12

    starts = []
    for _ in range(restarts + 2):
        point = random.normal(0.0, 2.0, size=dimension)
        point[0] = random.uniform(lower_crossing, upper_crossing)
        starts.append(point)
    bounds = [(lower_crossing, upper_crossing)] + [(-20.0, 20.0)] * (
        dimension - 1
    )
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maximum_iterations, "ftol": 1e-14},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("fixed-support optimization produced no result")
    crossing, increment, state = decode_fixed_support(best.x, support)
    increment_ratio = normalized_ratio(
        support_ratio, crossing, increment, state
    )
    return {
        "support": list(support),
        "increment_normalized_ratio": increment_ratio,
        "max_support_normalized_ratio": max(1.0, support_ratio)
        * increment_ratio,
        "crossing_ratio": crossing,
        "increment_nonzero_law": increment.tolist(),
        "state_nonzero_law": state.tolist(),
        "scale_free_payment": scale_free_payment(
            support_ratio, crossing, increment, state
        ),
        "scale_free_defect": scale_free_defect(crossing, state),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    family_state = np.array([0.1, 0.4, 0.4, 0.1])
    for index, support_ratio in enumerate(args.support_ratios):
        row = optimize_ratio(
            support_ratio,
            args.restarts,
            args.seed + index,
            args.maximum_iterations,
        )
        opposite_pair = optimize_opposite_pair_ratio(
            support_ratio,
            args.boundary_restarts,
            args.seed + 10_000 + index,
            args.maximum_iterations,
        )
        row["opposite_pair"] = opposite_pair
        if float(opposite_pair["increment_normalized_ratio"]) < float(
            row["increment_normalized_ratio"]
        ):
            row["best_family"] = "opposite-pair boundary"
            for key in (
                "increment_normalized_ratio",
                "max_support_normalized_ratio",
                "crossing_ratio",
                "increment_nonzero_law",
                "state_nonzero_law",
                "scale_free_payment",
                "scale_free_defect",
                "optimizer_success",
                "optimizer_message",
            ):
                row[key] = opposite_pair[key]
        else:
            row["best_family"] = "unrestricted softmax interior"
        family_ratio = large_ratio_family(support_ratio, family_state)
        row["explicit_family_increment_normalized_ratio"] = (
            family_ratio if np.isfinite(family_ratio) else None
        )
        row["explicit_family_max_support_normalized_ratio"] = (
            max(1.0, support_ratio) * family_ratio
            if np.isfinite(family_ratio)
            else None
        )
        rows.append(row)
    report = {
        "prime": 5,
        "normalization": {
            "increment": "A*L/R",
            "max_support": "max(A,B)*L/R",
            "support_ratio": "s=B/A",
        },
        "rows": rows,
        "minimum_increment_normalized_ratio": min(
            float(row["increment_normalized_ratio"]) for row in rows
        ),
        "minimum_max_support_normalized_ratio": min(
            float(row["max_support_normalized_ratio"]) for row in rows
        ),
        "status": "floating multistart minimization; not a global certificate",
        "analytic_obstruction": (
            "For t=1/5 and uniform nonzero increments, L/A equals "
            "s*D(v||(1-1/s)v+u4/s), while R/A^2 equals "
            "0.64*||v-u4||_2^2. Hence A*L/R tends to zero as s tends "
            "to infinity for every fixed positive nonuniform v."
        ),
    }
    zero.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support-ratios",
        type=float,
        nargs="+",
        default=(
            0.5001,
            0.55,
            0.6,
            0.7,
            0.8,
            1.0,
            1.25,
            1.5,
            2.0,
            3.0,
            5.0,
            10.0,
            20.0,
            50.0,
            100.0,
            220.0,
        ),
    )
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--boundary-restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--maximum-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-normalized-two-level.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))
