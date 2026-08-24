#!/usr/bin/env python3
"""Maximize the bulk exponent away from the sign-symmetric subspace."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

import optimize_bulk_exponent as bulk


def sign_permutation(layout: bulk.Layout) -> np.ndarray:
    p = layout.prime
    permutation = np.arange(layout.size)
    sign = (-np.arange(p)) % p
    for component in (layout.mu, layout.beta, layout.zeta):
        for value in range(p):
            permutation[component.start + value] = component.start + sign[value]
    for layer in (1, 2):
        for source in range(p):
            for target in range(p):
                permutation[layout.edge(layer, source, target)] = layout.edge(
                    layer, sign[source], sign[target]
                )
    return permutation


def energy_mask(layout: bulk.Layout, scope: str) -> np.ndarray:
    mask = np.zeros(layout.size)
    if scope in ("all", "compositions"):
        mask[layout.mu] = 1.0
        mask[layout.beta] = 1.0
        mask[layout.zeta] = 1.0
    if scope in ("all", "transitions"):
        mask[layout.tau1] = 1.0
        mask[layout.tau2] = 1.0
    if scope == "message":
        mask[layout.mu] = 1.0
    if scope == "states":
        mask[layout.beta] = 1.0
        mask[layout.zeta] = 1.0
    return mask


def asymmetry_energy(
    point: np.ndarray, permutation: np.ndarray, mask: np.ndarray
) -> float:
    antisymmetric = (point - point[permutation]) / 2.0
    return float(np.dot(mask * antisymmetric, antisymmetric))


def asymmetry_gradient(
    point: np.ndarray, permutation: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    return mask * (point - point[permutation])


def entropy_asymmetry(
    point: np.ndarray, permutation: np.ndarray, mask: np.ndarray
) -> float:
    symmetric = (point + point[permutation]) / 2.0
    selected = mask > 0.0
    return bulk.entropy(symmetric[selected]) - bulk.entropy(point[selected])


def entropy_asymmetry_gradient(
    point: np.ndarray, permutation: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    symmetric = (point + point[permutation]) / 2.0
    safe_point = np.maximum(point, 1e-300)
    safe_symmetric = np.maximum(symmetric, 1e-300)
    return mask * (np.log(safe_point) - np.log(safe_symmetric))


def asymmetry_value(
    point: np.ndarray,
    permutation: np.ndarray,
    mask: np.ndarray,
    metric: str,
) -> float:
    if metric == "entropy":
        return entropy_asymmetry(point, permutation, mask)
    return asymmetry_energy(point, permutation, mask)


def asymmetry_value_gradient(
    point: np.ndarray,
    permutation: np.ndarray,
    mask: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "entropy":
        return entropy_asymmetry_gradient(point, permutation, mask)
    return asymmetry_gradient(point, permutation, mask)


def energy_breakdown(
    point: np.ndarray, layout: bulk.Layout, permutation: np.ndarray
) -> dict[str, float]:
    return {
        scope: asymmetry_energy(point, permutation, energy_mask(layout, scope))
        for scope in ("all", "compositions", "message", "states", "transitions")
    }


def entropy_breakdown(
    point: np.ndarray, layout: bulk.Layout, permutation: np.ndarray
) -> dict[str, float]:
    return {
        scope: entropy_asymmetry(
            point, permutation, energy_mask(layout, scope)
        )
        for scope in ("all", "compositions", "message", "states", "transitions")
    }


def optimize(args: argparse.Namespace) -> dict[str, object]:
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
        max(64, args.restarts * 2),
        random,
    )
    permutation = sign_permutation(layout)
    mask = energy_mask(layout, args.asymmetry_scope)
    starts = sorted(
        vertices,
        key=lambda point: asymmetry_value(
            point, permutation, mask, args.asymmetry_metric
        ),
        reverse=True,
    )[: args.restarts]
    constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
        NonlinearConstraint(
            lambda point: asymmetry_value(
                point, permutation, mask, args.asymmetry_metric
            ),
            args.minimum_asymmetry,
            np.inf,
            jac=lambda point: asymmetry_value_gradient(
                point, permutation, mask, args.asymmetry_metric
            ),
        ),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda point: -bulk.exponent(point, layout, args.repetition),
            start,
            jac=lambda point: bulk.objective_gradient(
                point, layout, args.repetition
            ),
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 4000, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        energy = asymmetry_value(
            result.x, permutation, mask, args.asymmetry_metric
        )
        feasible = (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and energy + 1e-9 >= args.minimum_asymmetry
        )
        if feasible and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible asymmetric point")
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "asymmetry_metric": args.asymmetry_metric,
        "asymmetry_scope": args.asymmetry_scope,
        "minimum_asymmetry": args.minimum_asymmetry,
        "asymmetry_value": asymmetry_value(
            best.x, permutation, mask, args.asymmetry_metric
        ),
        "asymmetry_energy_breakdown": energy_breakdown(
            best.x, layout, permutation
        ),
        "entropy_asymmetry_breakdown": entropy_breakdown(
            best.x, layout, permutation
        ),
        "maximum_exponent_found": bulk.exponent(best.x, layout, args.repetition),
        "message_nonzero_fraction": 1.0 - float(best.x[layout.mu.start]),
        "output_nonzero_fraction": 1.0 - float(best.x[layout.zeta.start]),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical asymmetric-region search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument(
        "--minimum-asymmetry",
        "--minimum-energy",
        dest="minimum_asymmetry",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--asymmetry-scope",
        "--energy-scope",
        dest="asymmetry_scope",
        choices=("all", "compositions", "message", "states", "transitions"),
        default="all",
    )
    parser.add_argument(
        "--asymmetry-metric",
        choices=("euclidean", "entropy"),
        default="euclidean",
    )
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
