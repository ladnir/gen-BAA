#!/usr/bin/env python3
"""Numerically maximize a support-plus-projection exponent surrogate.

For each accumulator, data processing gives the support mutual information.
Data processing of an exact KL identity gives several additional lower bounds
on the full mutual information.  The strongest retained statistic is
``(U,V,T)``, where ``T=D+S``.  The resulting exponent is a rigorous pointwise
upper bound when evaluated exactly; this script only searches it numerically.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import diagnose_fourier_information_bound as character
import optimize_bulk_exponent as bulk
import search_internal_information_gap as information


def stage_lower(
    transition: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
) -> tuple[float, float, float]:
    _, support, _ = information.transition_information(transition)
    cell = max(
        character.nonzero_cell_convolution_lower(transition),
        character.coarse_product_sum_lower(transition, increment, state),
        character.support_cell_sum_lower(transition, increment, state),
    )
    return support + cell, support, cell


def surrogate(point: np.ndarray, layout: bulk.Layout, repetition: int) -> float:
    message_entropy = bulk.entropy(point[layout.mu]) / repetition
    first, _, _ = stage_lower(
        point[layout.tau1].reshape((layout.prime, layout.prime)),
        point[layout.mu],
        point[layout.beta],
    )
    second, _, _ = stage_lower(
        point[layout.tau2].reshape((layout.prime, layout.prime)),
        point[layout.beta],
        point[layout.zeta],
    )
    return message_entropy - first - second


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
        max(32, 2 * args.restarts),
        random,
    )
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(len(vertices), size=min(16, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(
                weight * vertices[index]
                for weight, index in zip(weights, chosen, strict=True)
            )
        )
    constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda point: -surrogate(point, layout, args.repetition),
            start,
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": args.max_iterations, "disp": False},
        )
        equality_error = float(
            np.max(np.abs(equality @ result.x - equality_target))
        )
        upper_error = float(np.max(upper @ result.x - upper_target))
        if (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and (best is None or result.fun < best.fun)
        ):
            best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible point")
    tau1 = best.x[layout.tau1].reshape((args.prime, args.prime))
    tau2 = best.x[layout.tau2].reshape((args.prime, args.prime))
    first, first_support, first_cell = stage_lower(
        tau1, best.x[layout.mu], best.x[layout.beta]
    )
    second, second_support, second_cell = stage_lower(
        tau2, best.x[layout.beta], best.x[layout.zeta]
    )
    return {
        "surrogate_maximum_found": -float(best.fun),
        "message_entropy_over_repetition": bulk.entropy(best.x[layout.mu])
        / args.repetition,
        "first_stage_lower": first,
        "first_support_information": first_support,
        "first_projection_lower": first_cell,
        "second_stage_lower": second,
        "second_support_information": second_support,
        "second_projection_lower": second_cell,
        "message": best.x[layout.mu].tolist(),
        "middle": best.x[layout.beta].tolist(),
        "output": best.x[layout.zeta].tolist(),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical surrogate search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))
