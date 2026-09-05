#!/usr/bin/env python3
"""Minimize the algebraic residual ratio behind normalized coercivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import analyze_normalized_boundaries as boundaries
import analyze_normalized_two_level as normalized
import analyze_zero_payment_surface as zero


def optimize_stratum(
    support: tuple[int, ...],
    restarts: int,
    seed: int,
    maximum_iterations: int,
) -> dict[str, object]:
    random = np.random.default_rng(seed)
    dimension = len(support) + 3

    def objective(point: np.ndarray) -> float:
        crossing, increment, state = normalized.decode_fixed_support(
            point, support
        )
        value = normalized.residual_quadratic_ratio(
            crossing, increment, state
        )
        return value if np.isfinite(value) else 1e12

    best = None
    bounds = [(0.0, 0.5)] + [(-20.0, 20.0)] * (dimension - 1)
    for _ in range(restarts + 2):
        start = random.normal(0.0, 2.0, size=dimension)
        start[0] = random.uniform(0.0, 0.5)
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
        raise RuntimeError("residual optimization produced no result")
    crossing, increment, state = normalized.decode_fixed_support(best.x, support)
    return {
        "support": list(support),
        "residual_ratio": normalized.residual_quadratic_ratio(
            crossing, increment, state
        ),
        "crossing_ratio": crossing,
        "increment_nonzero_law": increment.tolist(),
        "state_nonzero_law": state.tolist(),
        "scale_free_defect": normalized.scale_free_defect(crossing, state),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    strata = {}
    for index, (name, support) in enumerate(boundaries.SUPPORT_STRATA.items()):
        strata[name] = optimize_stratum(
            support,
            args.restarts,
            args.seed + index,
            args.maximum_iterations,
        )
    best_name = min(
        strata, key=lambda name: float(strata[name]["residual_ratio"])
    )
    report = {
        "prime": 5,
        "ratio": "(2*e0^2+||e_nonzero||_2^2)/R",
        "best_stratum": best_name,
        "minimum_found": strata[best_name]["residual_ratio"],
        "strata": strata,
        "implication": (
            "Pinsker gives L >= (2*e0^2+||e_nonzero||_2^2)/max(A,B)."
        ),
        "status": "floating multistart audit; not a global certificate",
    }
    zero.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts", type=int, default=96)
    parser.add_argument("--seed", type=int, default=3907)
    parser.add_argument("--maximum-iterations", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-residual-coercivity.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))
