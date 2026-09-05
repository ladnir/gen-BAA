#!/usr/bin/env python3
"""Search for a locally concave convex mixture of projected dual cuts.

For each stage, the projected envelope is at most any convex combination of
its cuts.  The Hessian of such a combination is affine in the cut weights.
This diagnostic uses eigenvector separation and linear programming to minimize
the mixed upper value subject to a negative-semidefinite Hessian at one
frontier composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected


def simplex_tangent_basis() -> np.ndarray:
    basis = np.zeros((15, 12))
    column = 0
    for start in (0, 5, 10):
        for value in range(1, 5):
            basis[start, column] = -1.0
            basis[start + value, column] = 1.0
            column += 1
    return basis


def cut_hessian(
    point: np.ndarray, kernel: np.ndarray, stage: int
) -> np.ndarray:
    left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
    left = point[left_start : left_start + 5]
    right = point[right_start : right_start + 5]
    partition = float(left @ kernel @ right)
    left_gradient = kernel @ right / partition
    right_gradient = kernel.T @ left / partition
    local = np.zeros((10, 10))
    local[0:5, 0:5] = -np.outer(left_gradient, left_gradient)
    local[5:10, 5:10] = -np.outer(right_gradient, right_gradient)
    cross = kernel / partition - np.outer(left_gradient, right_gradient)
    local[0:5, 5:10] = cross
    local[5:10, 0:5] = cross.T
    ambient = np.zeros((15, 15))
    indices = list(range(left_start, left_start + 5)) + list(
        range(right_start, right_start + 5)
    )
    ambient[np.ix_(indices, indices)] = local
    basis = simplex_tangent_basis()
    return basis.T @ ambient @ basis


def entropy_hessian(point: np.ndarray) -> np.ndarray:
    ambient = np.zeros((15, 15))
    ambient[0:5, 0:5] = np.diag(-1.0 / (3.0 * point[0:5]))
    basis = simplex_tangent_basis()
    return basis.T @ ambient @ basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--eigenvectors-per-round", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--candidate-cuts",
        type=int,
        default=0,
        help="retain this many lowest-value cuts per stage; zero retains all",
    )
    parser.add_argument(
        "--value-target",
        type=float,
        help="if set, minimize the Hessian cap subject to this mixed upper",
    )
    args = parser.parse_args()

    bank_args = SimpleNamespace(
        eta=0.002,
        delta=0.22,
        bank_iterations=20,
        bank_restarts=12,
        seed=11,
        bank_cache=str(args.bank),
        rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    kernels = [
        tuple(projected.envelope.kernel(cut) for cut in cuts)
        for cuts in (first_cuts, second_cuts)
    ]
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    point = projected.sanitize_composition(np.asarray(selected["point"]))

    values = []
    hessians = []
    for stage, cuts, stage_kernels in (
        (0, first_cuts, kernels[0]),
        (1, second_cuts, kernels[1]),
    ):
        left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
        stage_values = np.asarray(
            [
                projected.stage_upper_with_kernel(
                    point[left_start : left_start + 5],
                    point[right_start : right_start + 5],
                    cut,
                    kernel,
                )
                for cut, kernel in zip(cuts, stage_kernels, strict=True)
            ]
        )
        values.append(stage_values)
        hessians.append(
            np.stack(
                [cut_hessian(point, kernel, stage) for kernel in stage_kernels]
            )
        )

    original_indices = [
        np.arange(len(first_cuts), dtype=int),
        np.arange(len(second_cuts), dtype=int),
    ]
    if args.candidate_cuts > 0:
        for stage in range(2):
            selected_indices = np.argsort(values[stage])[: args.candidate_cuts]
            values[stage] = values[stage][selected_indices]
            hessians[stage] = hessians[stage][selected_indices]
            original_indices[stage] = original_indices[stage][selected_indices]
    first_count = len(values[0])
    second_count = len(values[1])
    variable_count = first_count + second_count
    value_objective = np.concatenate(values)
    entropy_value = projected.bulk_entropy(point[0:5]) / 3.0
    use_curvature_epigraph = args.value_target is not None
    optimization_size = variable_count + int(use_curvature_epigraph)
    objective = (
        np.concatenate((np.zeros(variable_count), [1.0]))
        if use_curvature_epigraph
        else value_objective
    )
    equality = np.zeros((2, optimization_size))
    equality[0, :first_count] = 1.0
    equality[1, first_count:variable_count] = 1.0
    entropy = entropy_hessian(point)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    if use_curvature_epigraph:
        rows.append(np.concatenate((value_objective, [0.0])))
        targets.append(float(args.value_target - entropy_value))
        for coordinate in range(entropy.shape[0]):
            vector = np.zeros(entropy.shape[0])
            vector[coordinate] = 1.0
            curvature_row = np.concatenate(
                (
                    np.einsum("i,kij,j->k", vector, hessians[0], vector),
                    np.einsum("i,kij,j->k", vector, hessians[1], vector),
                    [-1.0],
                )
            )
            curvature_target = -float(vector @ entropy @ vector)
            scale = (
                1.0
                if args.candidate_cuts > 0
                else max(
                    1.0,
                    float(np.max(np.abs(curvature_row))),
                    abs(curvature_target),
                )
            )
            rows.append(curvature_row / scale)
            targets.append(curvature_target / scale)
    result = None
    last_successful_result = None
    constructed_feasible_violation = None
    constructed_feasible_cap = None
    constructed_equality_residual = None
    constructed_value = None
    maximum_eigenvalue = float("inf")

    for iteration in range(args.iterations):
        result = linprog(
            objective,
            A_ub=np.stack(rows) if rows else None,
            b_ub=np.asarray(targets) if targets else None,
            A_eq=equality,
            b_eq=np.ones(2),
            bounds=(
                [(0.0, 1.0)] * variable_count + [(-1000.0, 1000.0)]
                if use_curvature_epigraph
                else (0.0, 1.0)
            ),
            method="highs-ds" if use_curvature_epigraph else "highs",
            options={"presolve": False} if use_curvature_epigraph else None,
        )
        if not result.success:
            if use_curvature_epigraph and last_successful_result is not None:
                previous_weights = last_successful_result.x[:variable_count]
                required_caps = [
                    (float(row[:variable_count] @ previous_weights) - target)
                    / (-float(row[-1]))
                    for row, target in zip(rows, targets, strict=True)
                    if row[-1] < 0.0
                ]
                test_point = np.concatenate(
                    (previous_weights, [max(required_caps, default=0.0) + 1.0])
                )
                constructed_feasible_cap = float(test_point[-1])
                constructed_equality_residual = float(
                    np.max(np.abs(equality @ test_point - np.ones(2)))
                )
                constructed_value = float(entropy_value + value_objective @ previous_weights)
                constructed_feasible_violation = float(
                    np.max(np.stack(rows) @ test_point - np.asarray(targets))
                )
            break
        last_successful_result = result
        weights = result.x[:variable_count]
        mixed_hessian = entropy.copy()
        mixed_hessian += np.tensordot(
            weights[:first_count], hessians[0], axes=(0, 0)
        )
        mixed_hessian += np.tensordot(
            weights[first_count:], hessians[1], axes=(0, 0)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(mixed_hessian)
        maximum_eigenvalue = float(eigenvalues[-1])
        curvature_cap = (
            float(result.x[-1]) if use_curvature_epigraph else 0.0
        )
        if maximum_eigenvalue <= curvature_cap + args.tolerance:
            break
        violated = np.flatnonzero(
            eigenvalues > curvature_cap + args.tolerance
        )
        for eigen_index in violated[-args.eigenvectors_per_round :]:
            vector = eigenvectors[:, eigen_index]
            row = np.concatenate(
                (
                    np.einsum("i,kij,j->k", vector, hessians[0], vector),
                    np.einsum("i,kij,j->k", vector, hessians[1], vector),
                )
            )
            if use_curvature_epigraph:
                row = np.concatenate((row, [-1.0]))
            curvature_target = -float(vector @ entropy @ vector)
            scale = (
                1.0
                if args.candidate_cuts > 0
                else max(
                    1.0, float(np.max(np.abs(row))), abs(curvature_target)
                )
            )
            rows.append(row / scale)
            targets.append(curvature_target / scale)

    payload: dict[str, object] = {
        "status": (
            "found requested cut-mixture tradeoff"
            if result is not None
            and result.success
            and maximum_eigenvalue
            <= (float(result.x[-1]) if use_curvature_epigraph else 0.0)
            + args.tolerance
            else "no locally concave cut mixture found"
        ),
        "frontier_index": args.frontier_index,
        "point": point.tolist(),
        "curvature_constraints": len(rows),
        "maximum_tangent_eigenvalue": maximum_eigenvalue,
        "unconstrained_mixed_minimum": float(
            entropy_value + min(values[0]) + min(values[1])
        ),
        "hessian_abs_max": float(
            max(np.max(np.abs(hessians[0])), np.max(np.abs(hessians[1])))
        ),
        "optimizer_message": (str(result.message) if result is not None else None),
        "constructed_feasible_violation": constructed_feasible_violation,
        "constructed_feasible_cap": constructed_feasible_cap,
        "constructed_equality_residual": constructed_equality_residual,
        "constructed_value": constructed_value,
    }
    if result is not None and result.success:
        weights = result.x[:variable_count]
        payload.update(
            {
                "mixed_upper_at_point": float(
                    entropy_value + value_objective @ weights
                ),
                "optimized_curvature_cap": (
                    float(result.x[-1]) if use_curvature_epigraph else 0.0
                ),
                "value_target": args.value_target,
                "first_support": [
                    {
                        "cut": int(original_indices[0][index]),
                        "weight": float(weight),
                    }
                    for index, weight in enumerate(weights[:first_count])
                    if weight > 1e-8
                ],
                "second_support": [
                    {
                        "cut": int(original_indices[1][index]),
                        "weight": float(weight),
                    }
                    for index, weight in enumerate(weights[first_count:])
                    if weight > 1e-8
                ],
                "iterations_used": iteration + 1,
            }
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
