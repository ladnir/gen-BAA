#!/usr/bin/env python3
"""Numerically maximize one fixed exact-dual pair on a saved box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def constraints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equality = np.zeros((3, 15))
    for block, start in enumerate((0, 5, 10)):
        equality[block, start : start + 5] = 1.0
    rows = []
    targets = []
    # A <= 2B and B <= 2C.
    row = np.zeros(15)
    row[1:5] = 1.0
    row[6:10] = -2.0
    rows.append(row)
    targets.append(0.0)
    row = np.zeros(15)
    row[6:10] = 1.0
    row[11:15] = -2.0
    rows.append(row)
    targets.append(0.0)
    # Multiplicative canonical sector: mu_1 is a largest nonzero mass.
    for value in range(2, 5):
        row = np.zeros(15)
        row[value] = 1.0
        row[1] = -1.0
        rows.append(row)
        targets.append(0.0)
    return equality, np.ones(3), np.stack(rows), np.asarray(targets)


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    with open(args.checkpoint, "r", encoding="utf-8") as source:
        saved = json.load(source)
    entry = sorted(saved["queue"], key=lambda value: value["priority"])[
        args.box_index
    ]["evaluated"]
    lower = np.asarray(entry["lower"], dtype=float)
    upper = np.asarray(entry["upper"], dtype=float)
    reference = projected.sanitize_composition(np.asarray(entry["point"]))
    first_cuts = [
        stationary.fit_stationary_dual(reference[0:5], reference[5:10])
    ]
    second_cuts = [
        stationary.fit_stationary_dual(reference[5:10], reference[10:15])
    ]
    if getattr(args, "seed_bank", None):
        with open(args.seed_bank, "r", encoding="utf-8") as source:
            bank = json.load(source)
        first_cuts = [stationary.decode_cut(value) for value in bank["first_cuts"]]
        second_cuts = [stationary.decode_cut(value) for value in bank["second_cuts"]]
    equality, equality_target, inequalities, inequality_target = constraints()

    def usable(cut: stationary.StationaryDualCut) -> bool:
        return bool(
            cut.optimizer_success
            and cut.equality_residual <= args.maximum_cut_residual
            and cut.gradient_residual <= args.maximum_cut_residual
            and np.isfinite(cut.moment_constant)
            and np.all(np.isfinite(cut.moment_coefficients))
            and np.all(np.isfinite(cut.kernel))
        )

    def fit_usable(
        increment: np.ndarray, state: np.ndarray
    ) -> tuple[stationary.StationaryDualCut | None, float | None]:
        candidate_masses = {0.03, 0.1, 0.2, 0.3, 0.5}
        interior_mass = args.interior_mass
        while interior_mass <= args.maximum_interior_mass * (1.0 + 1e-12):
            candidate_masses.add(interior_mass)
            interior_mass *= 10.0
        for interior_mass in sorted(candidate_masses):
            if interior_mass > args.maximum_interior_mass * (1.0 + 1e-12):
                continue
            interior_increment = (
                (1.0 - interior_mass) * increment + interior_mass / 5.0
            )
            interior_state = (
                (1.0 - interior_mass) * state + interior_mass / 5.0
            )
            cut = stationary.fit_stationary_dual(
                interior_increment, interior_state
            )
            if usable(cut):
                return cut, interior_mass
        return None, None

    if not usable(first_cuts[0]):
        replacement, _ = fit_usable(reference[0:5], reference[5:10])
        if replacement is None:
            raise RuntimeError("could not fit a usable first-stage seed cut")
        first_cuts = [replacement]
    if not usable(second_cuts[0]):
        replacement, _ = fit_usable(reference[5:10], reference[10:15])
        if replacement is None:
            raise RuntimeError("could not fit a usable second-stage seed cut")
        second_cuts = [replacement]

    def objective(point: np.ndarray) -> float:
        return (
            projected.bulk_entropy(point[0:5]) / 3.0
            - max(
                stationary.fixed_cut_payment(
                    point[0:5], point[5:10], cut
                )
                for cut in first_cuts
            )
            - max(
                stationary.fixed_cut_payment(
                    point[5:10], point[10:15], cut
                )
                for cut in second_cuts
            )
        )

    bounds = list(zip(lower, upper, strict=True))
    initial_reference_value = objective(reference)
    random = np.random.default_rng(args.seed)
    starts = [reference]
    for _ in range(args.restarts - 1):
        feasible = linprog(
            random.standard_normal(15),
            A_ub=inequalities,
            b_ub=inequality_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=bounds,
            method="highs",
        )
        if feasible.success:
            starts.append(feasible.x)
    history = []
    total_successes = 0
    best_point = reference
    best_value = objective(reference)
    for cut_iteration in range(args.cut_iterations):
        best_value = -np.inf
        iteration_point = best_point
        iteration_successes = 0
        iteration_starts = [best_point]
        iteration_starts.extend(
            start
            for start in starts
            if np.max(np.abs(start - best_point)) > 1e-10
        )
        augmented_equality = np.pad(equality, ((0, 0), (0, 2)))
        augmented_inequalities = np.pad(inequalities, ((0, 0), (0, 2)))

        def cut_constraints(augmented: np.ndarray) -> np.ndarray:
            point = augmented[:15]
            return np.concatenate(
                (
                    [
                        augmented[15]
                        - stationary.fixed_cut_payment(point[0:5], point[5:10], cut)
                        for cut in first_cuts
                    ],
                    [
                        augmented[16]
                        - stationary.fixed_cut_payment(point[5:10], point[10:15], cut)
                        for cut in second_cuts
                    ],
                )
            )

        for start in iteration_starts:
            augmented_start = np.concatenate(
                (
                    start,
                    [
                        max(
                            stationary.fixed_cut_payment(start[0:5], start[5:10], cut)
                            for cut in first_cuts
                        )
                        + 1e-9,
                        max(
                            stationary.fixed_cut_payment(start[5:10], start[10:15], cut)
                            for cut in second_cuts
                        )
                        + 1e-9,
                    ],
                )
            )
            result = minimize(
                lambda augmented: (
                    -projected.bulk_entropy(augmented[0:5]) / 3.0
                    + augmented[15]
                    + augmented[16]
                ),
                augmented_start,
                method="SLSQP",
                bounds=Bounds(
                    np.concatenate((lower, [-10.0, -10.0])),
                    np.concatenate((upper, [10.0, 10.0])),
                ),
                constraints=[
                    LinearConstraint(
                        augmented_equality, equality_target, equality_target
                    ),
                    LinearConstraint(
                        augmented_inequalities, -np.inf, inequality_target
                    ),
                    NonlinearConstraint(cut_constraints, 0.0, np.inf),
                ],
                options={
                    "ftol": 1e-12,
                    "maxiter": getattr(args, "optimizer_maxiter", 2000),
                },
            )
            candidate = result.x[:15]
            cut_slack = cut_constraints(result.x)
            feasible_result = bool(
                result.success and np.min(cut_slack) >= -1e-7
            )
            iteration_successes += int(feasible_result)
            value = objective(candidate)
            if value > best_value:
                best_value = value
                iteration_point = candidate.copy()
        best_point = projected.sanitize_composition(iteration_point)
        total_successes += iteration_successes
        history.append(
            {
                "iteration": cut_iteration,
                "maximum_found": best_value,
                "cuts_per_stage": len(first_cuts),
                "successful_restarts": iteration_successes,
            }
        )
        if best_value <= args.target:
            break
        first, first_mass = fit_usable(best_point[0:5], best_point[5:10])
        second, second_mass = fit_usable(best_point[5:10], best_point[10:15])
        if first is None or second is None:
            history[-1]["cut_admission"] = "rejected"
            break
        history[-1]["cut_interior_mass"] = [first_mass, second_mass]
        first_cuts.append(first)
        second_cuts.append(second)
    report = {
        "checkpoint": args.checkpoint,
        "box_index": args.box_index,
        "stored_box_upper": float(entry["bound"]),
        "fixed_cut_reference_value": initial_reference_value,
        "maximum_found": best_value,
        "maximizer": best_point.tolist(),
        "successful_restarts": total_successes,
        "restarts_per_iteration": len(starts) + 1,
        "exact_cuts_per_stage": len(first_cuts),
        "history": history,
        "maximum_first_cut_residual": max(
            cut.equality_residual for cut in first_cuts
        ),
        "maximum_second_cut_residual": max(
            cut.equality_residual for cut in second_cuts
        ),
        "status": "floating local optimization of one fixed valid cut pair",
    }
    if getattr(args, "include_cuts", False):
        report["first_cuts"] = [stationary.encode_cut(cut) for cut in first_cuts]
        report["second_cuts"] = [stationary.encode_cut(cut) for cut in second_cuts]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/f5-projected-kl-box-cover-0.22.checkpoint.json",
    )
    parser.add_argument("--box-index", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--cut-iterations", type=int, default=12)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--interior-mass", type=float, default=1e-6)
    parser.add_argument("--maximum-interior-mass", type=float, default=0.1)
    parser.add_argument("--maximum-cut-residual", type=float, default=1e-6)
    parser.add_argument("--optimizer-maxiter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--include-cuts", action="store_true")
    parser.add_argument("--seed-bank")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = diagnose(arguments)
    if arguments.output:
        projected.write_json_atomic(Path(arguments.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
