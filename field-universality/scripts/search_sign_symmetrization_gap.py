#!/usr/bin/env python3
"""Search for a negative sign-symmetrization gap in the bulk domain."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import optimize_bulk_exponent as bulk


def sign_permutation(prime: int, matrix: bool = False) -> np.ndarray:
    sign = (-np.arange(prime)) % prime
    if not matrix:
        return sign
    return np.array(
        [source * prime + target for source in sign for target in sign], dtype=int
    )


def asymmetry(distribution: np.ndarray, permutation: np.ndarray) -> float:
    flipped = distribution[permutation]
    return bulk.entropy((distribution + flipped) / 2.0) - bulk.entropy(
        distribution
    )


def asymmetry_gradient(
    distribution: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    symmetric = (distribution + distribution[permutation]) / 2.0
    safe_distribution = np.maximum(distribution, 1e-300)
    safe_symmetric = np.maximum(symmetric, 1e-300)
    return np.log(safe_distribution) - np.log(safe_symmetric)


def gap(point: np.ndarray, layout: bulk.Layout, repetition: int) -> float:
    vector_sign = sign_permutation(layout.prime)
    matrix_sign = sign_permutation(layout.prime, matrix=True)
    return (
        asymmetry(point[layout.tau1], matrix_sign)
        + asymmetry(point[layout.tau2], matrix_sign)
        - (1.0 - 1.0 / repetition) * asymmetry(point[layout.mu], vector_sign)
        - 2.0 * asymmetry(point[layout.beta], vector_sign)
        - asymmetry(point[layout.zeta], vector_sign)
    )


def gap_gradient(
    point: np.ndarray, layout: bulk.Layout, repetition: int
) -> np.ndarray:
    vector_sign = sign_permutation(layout.prime)
    matrix_sign = sign_permutation(layout.prime, matrix=True)
    gradient = np.zeros(layout.size)
    gradient[layout.tau1] = asymmetry_gradient(
        point[layout.tau1], matrix_sign
    )
    gradient[layout.tau2] = asymmetry_gradient(
        point[layout.tau2], matrix_sign
    )
    gradient[layout.mu] = -(
        1.0 - 1.0 / repetition
    ) * asymmetry_gradient(point[layout.mu], vector_sign)
    gradient[layout.beta] = -2.0 * asymmetry_gradient(
        point[layout.beta], vector_sign
    )
    gradient[layout.zeta] = -asymmetry_gradient(
        point[layout.zeta], vector_sign
    )
    return gradient


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(args.prime)
    equality, equality_target, upper, upper_target = bulk.constraints(
        layout, args.eta, args.delta
    )
    random = np.random.default_rng(args.seed)
    vertices = bulk.feasible_vertices(
        layout,
        equality,
        equality_target,
        upper,
        upper_target,
        max(32, args.restarts),
        random,
    )
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(len(vertices), size=min(16, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )
    constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    best = None
    for start in starts:
        result = minimize(
            gap,
            start,
            args=(layout, args.repetition),
            jac=gap_gradient,
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 3000, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        if equality_error <= 1e-7 and upper_error <= 1e-7:
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible point")
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "minimum_gap_found": float(best.fun),
        "message_nonzero_fraction": 1.0 - float(best.x[layout.mu.start]),
        "output_nonzero_fraction": 1.0 - float(best.x[layout.zeta.start]),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical counterexample search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.196)
    parser.add_argument("--eta", type=float, default=0.0001)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
