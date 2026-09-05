#!/usr/bin/env python3
"""Optimize the exact F_5 joint-projection composition objective."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import logsumexp

import diagnose_joint_information_projection as joint
import optimize_bulk_exponent as bulk


PRIME = 5
VARIABLES = 17
MU = slice(0, 5)
BETA = slice(5, 10)
ZETA = slice(10, 15)
X1 = 15
X2 = 16


def stage_payment_dual(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> tuple[float, dict[str, object]]:
    try:
        reference, cells = joint.reference_law(increment, state, crossing)
    except (FloatingPointError, ZeroDivisionError):
        return float("inf"), {"feasible": False}
    if np.any(reference < -1e-12) or np.any(cells < -1e-12):
        return float("inf"), {"feasible": False}
    reference = np.maximum(reference, 0.0)
    cells = np.maximum(cells, 0.0)
    equality, target = joint.constraints(increment, state, cells)
    active = reference.ravel() > 0.0
    feasible = linprog(
        np.zeros(reference.size),
        A_eq=equality,
        b_eq=target,
        bounds=[(0.0, 1.0) if value else (0.0, 0.0) for value in active],
        method="highs",
        options={"presolve": False},
    )
    if not feasible.success:
        return float("inf"), {"feasible": False}
    dual = joint.joint_projection_dual(reference, equality, target)
    if dual["equality_residual"] > 1e-6:
        return float("inf"), {"feasible": False, "dual": dual}
    support = joint.support_information(cells)
    return support + float(dual["value"]), {
        "feasible": True,
        "support_information": support,
        "internal_projection": dual["value"],
        "dual_residual": dual["equality_residual"],
        "dual_gradient_residual": dual["gradient_residual"],
        "law": dual["tilted_law"],
        "dual": dual,
    }


def fixed_dual_stage_payment(
    increment: np.ndarray,
    state: np.ndarray,
    crossing: float,
    potentials: np.ndarray,
    expected_features: np.ndarray,
) -> float:
    reference, cells = joint.reference_law(increment, state, crossing)
    equality, target = joint.constraints(increment, state, cells)
    features, moments = joint.constraint_feature_basis(equality, target)
    if not np.array_equal(features, expected_features):
        raise RuntimeError("joint-projection feature basis changed")
    flat_reference = reference.ravel()
    active = flat_reference > 0.0
    log_partition = float(
        logsumexp(
            np.log(flat_reference[active])
            + potentials @ features[:, active]
        )
    )
    internal = float(np.dot(potentials, moments)) - log_partition
    return joint.support_information(cells) + internal


def stage_payment_value_gradient(
    increment: np.ndarray,
    state: np.ndarray,
    crossing: float,
    step: float = 2e-6,
) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, object]]:
    value, data = stage_payment_dual(increment, state, crossing)
    if not np.isfinite(value):
        return value, np.zeros(5), np.zeros(5), 0.0, data
    dual = data["dual"]
    potentials = np.asarray(dual["potentials"])
    features = np.asarray(dual["feature_rows"])

    def directional(
        base_increment: np.ndarray,
        base_state: np.ndarray,
        base_crossing: float,
        component: str,
        index: int,
    ) -> float:
        local_step = step
        if component == "increment":
            local_step = min(local_step, base_increment[0] / 4.0, base_increment[index] / 4.0)
        elif component == "state":
            local_step = min(local_step, base_state[0] / 4.0, base_state[index] / 4.0)
        else:
            local_step = min(local_step, base_crossing / 4.0)
        local_step = max(local_step, 1e-9)
        plus_increment, minus_increment = base_increment.copy(), base_increment.copy()
        plus_state, minus_state = base_state.copy(), base_state.copy()
        plus_crossing = minus_crossing = base_crossing
        if component == "increment":
            plus_increment[index] += local_step
            plus_increment[0] -= local_step
            minus_increment[index] -= local_step
            minus_increment[0] += local_step
        elif component == "state":
            plus_state[index] += local_step
            plus_state[0] -= local_step
            minus_state[index] -= local_step
            minus_state[0] += local_step
        else:
            plus_crossing += local_step
            minus_crossing -= local_step
        plus = fixed_dual_stage_payment(
            plus_increment,
            plus_state,
            plus_crossing,
            potentials,
            features,
        )
        minus = fixed_dual_stage_payment(
            minus_increment,
            minus_state,
            minus_crossing,
            potentials,
            features,
        )
        return (plus - minus) / (2.0 * local_step)

    increment_gradient = np.zeros(5)
    state_gradient = np.zeros(5)
    for index in range(1, 5):
        increment_gradient[index] = directional(
            increment, state, crossing, "increment", index
        )
        state_gradient[index] = directional(
            increment, state, crossing, "state", index
        )
    crossing_gradient = directional(
        increment, state, crossing, "crossing", 0
    )
    return value, increment_gradient, state_gradient, crossing_gradient, data


def objective(point: np.ndarray) -> tuple[float, dict[str, object]]:
    message = point[MU]
    middle = point[BETA]
    output = point[ZETA]
    first, first_data = stage_payment_dual(message, middle, point[X1])
    second, second_data = stage_payment_dual(middle, output, point[X2])
    if not np.isfinite(first) or not np.isfinite(second):
        return -1e6, {"feasible": False}
    return bulk.entropy(message) / 3.0 - first - second, {
        "feasible": True,
        "first": first_data,
        "second": second_data,
    }


def objective_value_gradient(
    point: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, object]]:
    message, middle, output = point[MU], point[BETA], point[ZETA]
    first, first_increment, first_state, first_crossing, first_data = (
        stage_payment_value_gradient(message, middle, point[X1])
    )
    second, second_increment, second_state, second_crossing, second_data = (
        stage_payment_value_gradient(middle, output, point[X2])
    )
    if not np.isfinite(first) or not np.isfinite(second):
        return -1e6, np.zeros(VARIABLES), {"feasible": False}
    gradient = np.zeros(VARIABLES)
    for index in range(1, 5):
        gradient[index] = (
            np.log(max(message[0], 1e-300) / max(message[index], 1e-300))
            / 3.0
        )
    gradient[MU] -= first_increment
    gradient[BETA] -= first_state + second_increment
    gradient[ZETA] -= second_state
    gradient[X1] = -first_crossing
    gradient[X2] = -second_crossing
    value = bulk.entropy(message) / 3.0 - first - second
    return value, gradient, {
        "feasible": True,
        "first": first_data,
        "second": second_data,
    }


def crossing(transition: np.ndarray) -> float:
    return float(transition[0, 1:].sum())


def full_start(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=PRIME,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.full_restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-11,
            sign_symmetric=False,
        )
    )
    point = np.zeros(VARIABLES)
    point[MU] = optimized["mu"]
    point[BETA] = optimized["beta"]
    point[ZETA] = optimized["zeta"]
    point[X1] = crossing(np.asarray(optimized["tau1"]))
    point[X2] = crossing(np.asarray(optimized["tau2"]))
    return point, optimized


def outer_constraints(eta: float, delta: float) -> tuple[np.ndarray, ...]:
    equalities = np.zeros((3, VARIABLES))
    equalities[0, MU] = 1.0
    equalities[1, BETA] = 1.0
    equalities[2, ZETA] = 1.0
    equality_target = np.ones(3)
    rows: list[np.ndarray] = []
    targets: list[float] = []

    row = np.zeros(VARIABLES)
    row[0] = 1.0
    rows.append(row)
    targets.append(1.0 - eta)
    row = np.zeros(VARIABLES)
    row[10] = -1.0
    rows.append(row)
    targets.append(-(1.0 - delta))
    for increment_zero, state_zero, crossing_index in (
        (0, 5, X1),
        (5, 10, X2),
    ):
        # x <= increment support, x <= zero-state mass, x >= a-b.
        row = np.zeros(VARIABLES)
        row[crossing_index] = 1.0
        row[increment_zero] = 1.0
        rows.append(row)
        targets.append(1.0)
        row = np.zeros(VARIABLES)
        row[crossing_index] = 1.0
        row[state_zero] = -1.0
        rows.append(row)
        targets.append(0.0)
        row = np.zeros(VARIABLES)
        row[state_zero] = 1.0
        row[increment_zero] = -1.0
        row[crossing_index] = -1.0
        rows.append(row)
        targets.append(0.0)
    return equalities, equality_target, np.stack(rows), np.asarray(targets)


def optimize(args: argparse.Namespace) -> dict[str, object]:
    start, full = full_start(args)
    equality, equality_target, upper, upper_target = outer_constraints(
        args.eta, args.delta
    )
    evaluations = 0
    start_value, start_diagnostic = objective(start)
    if not start_diagnostic["feasible"]:
        raise RuntimeError("the full-composition start has no joint projection")
    best_value = start_value
    best_point = start.copy()
    best_diagnostic = start_diagnostic

    def minimized(point: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations, best_value, best_point, best_diagnostic
        evaluations += 1
        value, gradient, diagnostic = objective_value_gradient(point)
        if diagnostic["feasible"] and value > best_value:
            best_value = value
            best_point = point.copy()
            best_diagnostic = diagnostic
        return -value, -gradient

    result = minimize(
        minimized,
        start,
        jac=True,
        method="SLSQP",
        bounds=Bounds(0.0, 1.0),
        constraints=[
            LinearConstraint(equality, equality_target, equality_target),
            LinearConstraint(upper, -np.inf, upper_target),
        ],
        options={"ftol": args.tolerance, "maxiter": args.max_iterations, "disp": False},
    )
    final_value, final_diagnostic = objective(result.x)
    if final_diagnostic["feasible"] and final_value > best_value:
        best_value = final_value
        best_point = result.x.copy()
        best_diagnostic = final_diagnostic
    point = best_point
    value, diagnostic = objective(point)
    return {
        "delta": args.delta,
        "eta": args.eta,
        "joint_projection_maximum_found": value,
        "full_composition_maximum_found": full["exponent"],
        "representation_gap": value - full["exponent"],
        "start_value": start_value,
        "message": point[MU].tolist(),
        "middle": point[BETA].tolist(),
        "output": point[ZETA].tolist(),
        "first_crossing": float(point[X1]),
        "second_crossing": float(point[X2]),
        "evaluations": evaluations,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "first_dual_residual": diagnostic["first"]["dual_residual"],
        "second_dual_residual": diagnostic["second"]["dual_residual"],
        "status": "numerical joint-projection optimizer only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--full-restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))
