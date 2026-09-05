#!/usr/bin/env python3
"""Diagnose the joint support-cell information projection over F_5."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import logsumexp

import optimize_bulk_exponent as bulk
import optimize_conditional_product_bound as product_bound


def support_table(increment_support: float, state_support: float, crossing: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - state_support - crossing, state_support - increment_support + crossing],
            [crossing, increment_support - crossing],
        ]
    )


def reference_law(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> tuple[np.ndarray, np.ndarray]:
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    cells = support_table(increment_support, state_support, crossing)
    reference = np.zeros((len(increment), len(state)))
    increment_masses = np.array([1.0 - increment_support, increment_support])
    state_masses = np.array([1.0 - state_support, state_support])
    for difference in range(len(increment)):
        for source in range(len(state)):
            u, v = int(difference != 0), int(source != 0)
            reference[difference, source] = (
                cells[u, v]
                * increment[difference]
                / increment_masses[u]
                * state[source]
                / state_masses[v]
            )
    return reference, cells


def constraints(
    increment: np.ndarray, state: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prime = len(increment)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for difference in range(prime):
        row = np.zeros((prime, prime))
        row[difference, :] = 1.0
        rows.append(row.ravel())
        targets.append(float(increment[difference]))
    for source in range(prime):
        row = np.zeros((prime, prime))
        row[:, source] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[source]))
    for target in range(prime):
        row = np.zeros((prime, prime))
        for difference in range(prime):
            row[difference, (target - difference) % prime] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[target]))
    for u in range(2):
        for v in range(2):
            row = np.zeros((prime, prime))
            for difference in range(prime):
                for source in range(prime):
                    if int(difference != 0) == u and int(source != 0) == v:
                        row[difference, source] = 1.0
            rows.append(row.ravel())
            targets.append(float(cells[u, v]))
    return bulk.independent_equalities(np.stack(rows), np.asarray(targets))


def relative_entropy(actual: np.ndarray, reference: np.ndarray) -> float:
    positive = actual > 0.0
    return float(np.dot(actual[positive], np.log(actual[positive] / reference[positive])))


def constraint_feature_basis(
    equality: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove equality dependencies and the normalization gauge."""
    ones = np.ones(equality.shape[1])
    selected_rows = [ones]
    selected_targets = [1.0]
    rank = 1
    for row, value in zip(equality, target, strict=True):
        candidate = np.stack((*selected_rows, row))
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=1e-10))
        if candidate_rank > rank:
            selected_rows.append(row)
            selected_targets.append(float(value))
            rank = candidate_rank
    return np.stack(selected_rows[1:]), np.asarray(selected_targets[1:])


def joint_projection_dual(
    reference: np.ndarray,
    equality: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    """Solve the normalized KL dual for one joint projection."""
    flat_reference = reference.ravel()
    active = flat_reference > 0.0
    features, moments = constraint_feature_basis(equality, target)
    active_features = features[:, active]
    log_reference = np.log(flat_reference[active])

    def value_gradient(potentials: np.ndarray) -> tuple[float, np.ndarray]:
        log_weights = log_reference + potentials @ active_features
        log_partition = float(logsumexp(log_weights))
        tilted = np.exp(log_weights - log_partition)
        value = log_partition - float(np.dot(potentials, moments))
        gradient = active_features @ tilted - moments
        return value, gradient

    result = minimize(
        value_gradient,
        np.zeros(len(moments)),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 20_000},
    )
    negative_value, gradient = value_gradient(result.x)
    log_weights = log_reference + result.x @ active_features
    tilted_active = np.exp(log_weights - logsumexp(log_weights))
    tilted = np.zeros_like(flat_reference)
    tilted[active] = tilted_active
    equality_residual = float(np.max(np.abs(equality @ tilted - target)))
    return {
        "value": -float(negative_value),
        "potentials": result.x.tolist(),
        "feature_rows": features.tolist(),
        "feature_targets": moments.tolist(),
        "tilted_law": tilted.reshape(reference.shape).tolist(),
        "gradient_residual": float(np.max(np.abs(gradient))),
        "equality_residual": equality_residual,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def joint_projection(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> dict[str, object]:
    reference, cells = reference_law(increment, state, crossing)
    equality, target = constraints(increment, state, cells)
    flat_reference = reference.ravel()
    positive_reference = flat_reference > 0.0
    variable_bounds = [
        (0.0, 1.0) if positive else (0.0, 0.0)
        for positive in positive_reference
    ]
    feasible = linprog(
        np.zeros(reference.size),
        A_eq=equality,
        b_eq=target,
        bounds=variable_bounds,
        method="highs",
        # HiGHS presolve can incorrectly discard feasible cells when the
        # sparse maximizer gives the reference law masses around 1e-14.
        options={"presolve": False},
    )
    if not feasible.success:
        return {"feasible": False, "value": float("inf")}

    dual = joint_projection_dual(reference, equality, target)

    def value_gradient(point: np.ndarray) -> tuple[float, np.ndarray]:
        safe = np.maximum(point[positive_reference], 1e-300)
        value = float(
            np.dot(
                point[positive_reference],
                np.log(safe / flat_reference[positive_reference]),
            )
        )
        gradient = np.zeros_like(point)
        gradient[positive_reference] = (
            np.log(safe / flat_reference[positive_reference]) + 1.0
        )
        return value, gradient

    result = minimize(
        value_gradient,
        feasible.x,
        jac=True,
        method="SLSQP",
        bounds=Bounds(
            np.zeros(reference.size), positive_reference.astype(float)
        ),
        constraints=[LinearConstraint(equality, target, target)],
        options={"ftol": 1e-13, "maxiter": 5000, "disp": False},
    )
    residual = float(np.max(np.abs(equality @ result.x - target)))
    return {
        "feasible": residual <= 1e-7,
        "value": relative_entropy(result.x, flat_reference),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "equality_residual": residual,
        "law": result.x.reshape(reference.shape).tolist(),
        "dual": dual,
        "duality_gap": relative_entropy(result.x, flat_reference)
        - float(dual["value"]),
    }


def support_information(cells: np.ndarray) -> float:
    product = np.outer(cells.sum(axis=1), cells.sum(axis=0))
    return relative_entropy(cells.ravel(), product.ravel())


def stage_payment(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> dict[str, object]:
    projection = joint_projection(increment, state, crossing)
    _, cells = reference_law(increment, state, crossing)
    support = support_information(cells)
    return {
        "support_information": support,
        "internal_projection": projection["value"],
        "total_payment": support + float(projection["value"]),
        "projection": projection,
    }


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    if args.source == "full":
        optimized = bulk.optimize(
            SimpleNamespace(
                prime=5,
                repetition=3,
                delta=args.delta,
                eta=args.eta,
                restarts=args.restarts,
                seed=args.seed,
                max_iterations=3000,
                tolerance=1e-11,
                sign_symmetric=False,
            )
        )
        message = np.asarray(optimized["mu"])
        middle = np.asarray(optimized["beta"])
        output = np.asarray(optimized["zeta"])
        first_transition = np.asarray(optimized["tau1"])
        second_transition = np.asarray(optimized["tau2"])
        first_crossing = float(first_transition[0, 1:].sum())
        second_crossing = float(second_transition[0, 1:].sum())
        first = stage_payment(message, middle, first_crossing)
        second = stage_payment(middle, output, second_crossing)
        joint_upper = (
            bulk.entropy(message) / 3.0
            - first["total_payment"]
            - second["total_payment"]
        )
        return {
            "source": "full-composition maximizer",
            "delta": args.delta,
            "full_exponent": optimized["exponent"],
            "joint_projection_value": joint_upper,
            "reconstruction_gap": joint_upper - optimized["exponent"],
            "message": message.tolist(),
            "middle": middle.tolist(),
            "output": output.tolist(),
            "first_crossing": first_crossing,
            "second_crossing": second_crossing,
            "first_stage": first,
            "second_stage": second,
            "status": "numerical joint information-projection diagnostic only",
        }

    optimized = product_bound.optimize(
        SimpleNamespace(
            prime=5,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-11,
            support_lower=None,
            support_upper=None,
            projection="hybrid",
            sign_symmetric=False,
        )
    )
    message = np.asarray(optimized["mu"])
    middle = np.asarray(optimized["beta"])
    output = np.asarray(optimized["zeta"])
    first = stage_payment(message, middle, optimized["first_crossing_fraction"])
    second = stage_payment(middle, output, optimized["second_crossing_fraction"])
    joint_upper = bulk.entropy(message) / 3.0 - first["total_payment"] - second["total_payment"]
    return {
        "source": "hybrid false maximizer",
        "delta": args.delta,
        "hybrid_upper_at_false_maximizer": optimized["maximum_upper_bound_found"],
        "joint_projection_upper_at_same_compositions": joint_upper,
        "message": message.tolist(),
        "middle": middle.tolist(),
        "output": output.tolist(),
        "first_crossing": optimized["first_crossing_fraction"],
        "second_crossing": optimized["second_crossing_fraction"],
        "first_stage": first,
        "second_stage": second,
        "status": "numerical joint information-projection diagnostic only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.24)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--source", choices=("hybrid", "full"), default="hybrid")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
