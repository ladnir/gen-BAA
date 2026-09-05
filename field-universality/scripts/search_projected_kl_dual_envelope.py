#!/usr/bin/env python3
"""Constraint generation for the projected-KL composition dual envelope."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import optimize_bulk_exponent as bulk
import projected_kl_dual as dual


def kernel(cut: dual.ProjectionDual) -> np.ndarray:
    prime = len(cut.target_potential)
    result = np.empty((prime, prime))
    for difference in range(prime):
        for source in range(prime):
            result[difference, source] = np.exp(
                cut.target_potential[(difference + source) % prime]
                + cut.increment_support_potential * int(difference != 0)
                + cut.state_support_potential * int(source != 0)
            )
    return result


def stage_upper_gradient(
    increment: np.ndarray,
    state: np.ndarray,
    cut: dual.ProjectionDual,
) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = kernel(cut)
    partition = float(increment @ matrix @ state)
    value = (
        np.log(partition)
        - float(np.dot(cut.target_potential, state))
        - cut.increment_support_potential * float(increment[1:].sum())
        - cut.state_support_potential * float(state[1:].sum())
    )
    increment_gradient = matrix @ state / partition
    increment_gradient[1:] -= cut.increment_support_potential
    state_gradient = matrix.T @ increment / partition - cut.target_potential
    state_gradient[1:] -= cut.state_support_potential
    return value, increment_gradient, state_gradient


def composition_constraints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equality = np.zeros((3, 17))
    for component in range(3):
        equality[component, 5 * component : 5 * component + 5] = 1.0
    equality_target = np.ones(3)
    upper = []
    target = []
    # A <= 2B and B <= 2C.
    first = np.zeros(17)
    first[1:5] = 1.0
    first[6:10] = -2.0
    upper.append(first)
    target.append(0.0)
    second = np.zeros(17)
    second[6:10] = 1.0
    second[11:15] = -2.0
    upper.append(second)
    target.append(0.0)
    # Multiplication by F_5^* lets mu_1 be chosen as a largest nonzero mass.
    for value in range(2, 5):
        canonical = np.zeros(17)
        canonical[value] = 1.0
        canonical[1] = -1.0
        upper.append(canonical)
        target.append(0.0)
    return equality, equality_target, np.stack(upper), np.array(target)


def envelope_constraints(
    extended: np.ndarray,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> tuple[np.ndarray, np.ndarray]:
    mu = extended[0:5]
    beta = extended[5:10]
    zeta = extended[10:15]
    values = []
    jacobian = []
    for cut in first_cuts:
        value, mu_gradient, beta_gradient = stage_upper_gradient(mu, beta, cut)
        row = np.zeros(17)
        row[0:5] = mu_gradient
        row[5:10] = beta_gradient
        row[15] = -1.0
        values.append(value - extended[15])
        jacobian.append(row)
    for cut in second_cuts:
        value, beta_gradient, zeta_gradient = stage_upper_gradient(beta, zeta, cut)
        row = np.zeros(17)
        row[5:10] = beta_gradient
        row[10:15] = zeta_gradient
        row[16] = -1.0
        values.append(value - extended[16])
        jacobian.append(row)
    return np.array(values), np.stack(jacobian)


def objective_gradient(extended: np.ndarray) -> tuple[float, np.ndarray]:
    mu = extended[0:5]
    positive = np.maximum(mu, 1e-300)
    value = bulk.entropy(mu) / 3.0 + extended[15] + extended[16]
    gradient = np.zeros(17)
    gradient[0:5] = -(np.log(positive) + 1.0) / 3.0
    gradient[15:17] = 1.0
    return -value, -gradient


def random_composition(support: float, random: np.random.Generator) -> np.ndarray:
    labels = random.dirichlet(np.ones(4))
    return np.array([1.0 - support, *(support * labels)])


def starts(args: argparse.Namespace) -> list[np.ndarray]:
    random = np.random.default_rng(args.seed)
    result = []
    known = np.array(
        [
            0.998,
            7.904336419562844e-05,
            0.0007543782304007307,
            0.0010870935577962585,
            7.94848476073528e-05,
            0.9973311535518833,
            2.480963947518967e-05,
            0.0020909952766732303,
            0.0005232893734466101,
            2.975215852181995e-05,
            0.8676999999999999,
            0.0009884859794976459,
            0.009804622288825098,
            0.12106836509944331,
            0.00043852663223407196,
        ]
    )
    known_multiplier = pow(int(np.argmax(known[1:5])) + 1, -1, 5)
    for start in (0, 5, 10):
        known[start : start + 5] = dual.scale_symbol_vector(
            known[start : start + 5], known_multiplier
        )
    result.append(np.concatenate((known, [-0.01, -0.01])))
    while len(result) < args.restarts:
        output_support = random.uniform(args.eta / 4.0, args.delta)
        middle_support = random.uniform(args.eta / 2.0, 2.0 * output_support)
        message_support = random.uniform(args.eta, min(1.0, 2.0 * middle_support))
        composition = np.concatenate(
            (
                random_composition(message_support, random),
                random_composition(middle_support, random),
                random_composition(output_support, random),
            )
        )
        largest = int(np.argmax(composition[1:5])) + 1
        multiplier = pow(largest, -1, 5)
        for start in (0, 5, 10):
            composition[start : start + 5] = dual.scale_symbol_vector(
                composition[start : start + 5], multiplier
            )
        result.append(np.concatenate((composition, [-0.01, -0.01])))
    return result


def maximize_envelope(
    args: argparse.Namespace,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> object:
    equality, equality_target, upper, upper_target = composition_constraints()
    lower = np.zeros(17)
    upper_bounds = np.ones(17)
    upper_bounds[0] = 1.0 - args.eta
    lower[10] = 1.0 - args.delta
    lower[15:17] = -20.0
    upper_bounds[15:17] = 20.0
    constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
        {
            "type": "ineq",
            "fun": lambda point: envelope_constraints(
                point, first_cuts, second_cuts
            )[0],
            "jac": lambda point: envelope_constraints(
                point, first_cuts, second_cuts
            )[1],
        },
    ]
    best = None
    for start in starts(args):
        first_values = [
            stage_upper_gradient(start[0:5], start[5:10], cut)[0]
            for cut in first_cuts
        ]
        second_values = [
            stage_upper_gradient(start[5:10], start[10:15], cut)[0]
            for cut in second_cuts
        ]
        start[15] = min(first_values) - 1e-9
        start[16] = min(second_values) - 1e-9
        result = minimize(
            lambda point: objective_gradient(point)[0],
            start,
            jac=lambda point: objective_gradient(point)[1],
            method="SLSQP",
            bounds=Bounds(lower, upper_bounds),
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": 3000, "disp": False},
        )
        equality_error = float(
            np.max(np.abs(equality @ result.x - equality_target))
        )
        upper_error = float(np.max(upper @ result.x - upper_target))
        envelope_error = float(
            np.min(
                envelope_constraints(result.x, first_cuts, second_cuts)[0]
            )
        )
        bounds_error = max(
            float(np.max(lower - result.x)),
            float(np.max(result.x - upper_bounds)),
        )
        if (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and envelope_error >= -1e-7
            and bounds_error <= 1e-7
            and np.isfinite(result.fun)
            and (best is None or result.fun < best.fun)
        ):
            best = result
    if best is None:
        raise RuntimeError("no envelope restart returned a feasible point")
    return best


def cut_orbit(cut: dual.ProjectionDual) -> list[dual.ProjectionDual]:
    return [dual.scale_projection_dual(cut, multiplier) for multiplier in range(1, 5)]


def generate_bank(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dual.ProjectionDual], list[dual.ProjectionDual]]:
    reference = starts(args)[0][0:15]
    first_cuts = cut_orbit(
        dual.fit_projection_dual(reference[0:5], reference[5:10])
    )
    second_cuts = cut_orbit(
        dual.fit_projection_dual(reference[5:10], reference[10:15])
    )
    history = []
    for iteration in range(args.iterations):
        result = maximize_envelope(args, first_cuts, second_cuts)
        point = result.x
        fitted_first = dual.fit_projection_dual(point[0:5], point[5:10])
        fitted_second = dual.fit_projection_dual(point[5:10], point[10:15])
        true_upper = (
            bulk.entropy(point[0:5]) / 3.0
            - fitted_first.value
            - fitted_second.value
        )
        envelope = -float(result.fun)
        history.append(
            {
                "iteration": iteration + 1,
                "first_cuts": len(first_cuts),
                "second_cuts": len(second_cuts),
                "envelope_upper": envelope,
                "locally_fitted_upper": true_upper,
                "cut_gap": envelope - true_upper,
                "message": point[0:5].tolist(),
                "middle": point[5:10].tolist(),
                "output": point[10:15].tolist(),
                "optimizer_success": bool(result.success),
            }
        )
        if envelope - true_upper <= args.tolerance:
            break
        first_cuts.extend(cut_orbit(fitted_first))
        second_cuts.extend(cut_orbit(fitted_second))
    result = {
        "iterations": len(history),
        "final_envelope_upper": history[-1]["envelope_upper"],
        "numerically_negative": history[-1]["envelope_upper"] < 0.0,
        "history": history,
        "status": "numerical projected-KL dual envelope only",
    }
    return result, first_cuts, second_cuts


def search(args: argparse.Namespace) -> dict[str, object]:
    result, _, _ = generate_bank(args)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))
