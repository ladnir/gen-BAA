#!/usr/bin/env python3
"""Maximize the F_5 bulk exponent where sign averaging loses value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

import explore_projected_kl_box_cover as projected
import optimize_bulk_exponent as bulk
import search_sign_symmetrization_gap as sign_gap


def feasible_error(
    point: np.ndarray,
    equality: np.ndarray,
    equality_target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
) -> float:
    return max(
        float(np.max(np.abs(equality @ point - equality_target))),
        float(np.max(upper @ point - upper_target)),
        -float(np.min(point)),
        float(np.max(point - 1.0)),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(5)
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
        max(48, 2 * args.restarts),
        random,
    )
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(
            len(vertices), size=min(16, len(vertices)), replace=True
        )
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(
                weight * vertices[index]
                for weight, index in zip(weights, chosen, strict=True)
            )
        )

    linear_constraints = (
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    )
    gap_seeds = []
    for start in starts:
        result = minimize(
            sign_gap.gap,
            start,
            args=(layout, 3),
            jac=sign_gap.gap_gradient,
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=linear_constraints,
            options={"ftol": 1e-11, "maxiter": 2000, "disp": False},
        )
        if feasible_error(
            result.x,
            equality,
            equality_target,
            upper,
            upper_target,
        ) <= 1e-7:
            gap_seeds.append(result.x)
    if not gap_seeds:
        raise RuntimeError("the gap search returned no feasible seed")
    deepest = min(
        gap_seeds, key=lambda point: sign_gap.gap(point, layout, 3)
    )

    vector_sign = sign_gap.sign_permutation(5)
    matrix_sign = sign_gap.sign_permutation(5, matrix=True)

    def symmetrize(point: np.ndarray) -> np.ndarray:
        result = point.copy()
        for component in (layout.mu, layout.beta, layout.zeta):
            result[component] = (
                point[component] + point[component][vector_sign]
            ) / 2.0
        for component in (layout.tau1, layout.tau2):
            result[component] = (
                point[component] + point[component][matrix_sign]
            ) / 2.0
        return result

    candidate_starts = [deepest]
    candidate_starts.extend(symmetrize(point) for point in starts)
    candidate_starts.extend(gap_seeds[: args.restarts])
    records = []
    for gap_upper in sorted(args.gap_uppers):
        nonlinear = NonlinearConstraint(
            lambda point: sign_gap.gap(point, layout, 3),
            -np.inf,
            gap_upper,
            jac=lambda point: sign_gap.gap_gradient(point, layout, 3),
        )
        best = None
        successes = 0
        for start in candidate_starts:
            result = minimize(
                lambda point: -bulk.exponent(point, layout, 3),
                start,
                jac=lambda point: bulk.objective_gradient(point, layout, 3),
                method="SLSQP",
                bounds=Bounds(0.0, 1.0),
                constraints=(*linear_constraints, nonlinear),
                options={"ftol": 1e-11, "maxiter": 3000, "disp": False},
            )
            error = feasible_error(
                result.x,
                equality,
                equality_target,
                upper,
                upper_target,
            )
            candidate_gap = sign_gap.gap(result.x, layout, 3)
            accepted = error <= 1e-7 and candidate_gap <= gap_upper + 1e-7
            successes += int(accepted)
            if accepted and (best is None or result.fun < best.fun):
                best = result
        if best is None:
            records.append(
                {
                    "gap_upper": gap_upper,
                    "status": "no feasible optimizer result",
                    "successful_restarts": successes,
                }
            )
            continue
        point = best.x
        candidate_starts.append(point.copy())
        records.append(
            {
                "gap_upper": gap_upper,
                "maximum_exponent_found": bulk.exponent(point, layout, 3),
                "sign_gap": sign_gap.gap(point, layout, 3),
                "message_nonzero_fraction": 1.0 - float(point[layout.mu.start]),
                "output_nonzero_fraction": 1.0 - float(point[layout.zeta.start]),
                "successful_restarts": successes,
                "optimizer_success": bool(best.success),
                "point": point.tolist(),
            }
        )
    for record in records:
        if "maximum_exponent_found" not in record:
            continue
        gap_upper = float(record["gap_upper"])
        feasible_records = [
            candidate
            for candidate in records
            if "maximum_exponent_found" in candidate
            and float(candidate["sign_gap"]) <= gap_upper + 1e-7
        ]
        best_feasible = max(
            feasible_records,
            key=lambda candidate: float(candidate["maximum_exponent_found"]),
        )
        record["best_exponent_across_all_feasible_runs"] = float(
            best_feasible["maximum_exponent_found"]
        )
        record["best_feasible_run_gap"] = float(best_feasible["sign_gap"])
    records.sort(key=lambda record: float(record["gap_upper"]), reverse=True)
    report = {
        "prime": 5,
        "repetition": 3,
        "eta": args.eta,
        "delta": args.delta,
        "minimum_gap_seed": sign_gap.gap(deepest, layout, 3),
        "minimum_gap_seed_exponent": bulk.exponent(deepest, layout, 3),
        "records": records,
        "status": "floating constrained multistart diagnostics; not a global bound",
    }
    projected.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument(
        "--gap-uppers",
        type=float,
        nargs="+",
        default=(0.0, -1e-5, -1e-4, -1e-3, -1e-2),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-symmetry-split-audit.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
