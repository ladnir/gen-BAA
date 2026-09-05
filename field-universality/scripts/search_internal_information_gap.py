#!/usr/bin/env python3
"""Search for violations of a two-stage internal-information inequality."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

import optimize_bulk_exponent as bulk


def entropy_gradient(distribution: np.ndarray) -> np.ndarray:
    return -(np.log(np.maximum(distribution, 1e-300)) + 1.0)


def transition_information(
    transition: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    prime = transition.shape[0]
    increment = np.zeros(prime)
    state = transition.sum(axis=1)
    support_joint = np.zeros((2, 2))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            value = transition[source, target]
            increment[difference] += value
            support_joint[int(difference != 0), int(source != 0)] += value

    increment_support = support_joint.sum(axis=1)
    state_support = support_joint.sum(axis=0)
    full = (
        bulk.entropy(increment)
        + bulk.entropy(state)
        - bulk.entropy(transition.ravel())
    )
    support = (
        bulk.entropy(increment_support)
        + bulk.entropy(state_support)
        - bulk.entropy(support_joint.ravel())
    )

    increment_gradient = entropy_gradient(increment)
    state_gradient = entropy_gradient(state)
    transition_gradient = entropy_gradient(transition.ravel()).reshape(
        (prime, prime)
    )
    increment_support_gradient = entropy_gradient(increment_support)
    state_support_gradient = entropy_gradient(state_support)
    support_joint_gradient = entropy_gradient(support_joint.ravel()).reshape((2, 2))
    gradient = np.zeros_like(transition)
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            increment_nonzero = int(difference != 0)
            source_nonzero = int(source != 0)
            full_derivative = (
                increment_gradient[difference]
                + state_gradient[source]
                - transition_gradient[source, target]
            )
            support_derivative = (
                increment_support_gradient[increment_nonzero]
                + state_support_gradient[source_nonzero]
                - support_joint_gradient[increment_nonzero, source_nonzero]
            )
            gradient[source, target] = full_derivative - support_derivative
    return full, support, gradient


def gap_and_gradient(
    point: np.ndarray, layout: bulk.Layout, repetition: int
) -> tuple[float, np.ndarray]:
    mu = point[layout.mu]
    nonzero = float(mu[1:].sum())
    safe_nonzero = min(max(nonzero, 1e-15), 1.0 - 1e-15)
    message_internal = bulk.entropy(mu)
    message_internal -= -safe_nonzero * np.log(safe_nonzero) - (
        1.0 - safe_nonzero
    ) * np.log(1.0 - safe_nonzero)

    gradient = np.zeros(layout.size)
    message_gradient = entropy_gradient(mu)
    binary_derivative = np.log((1.0 - safe_nonzero) / safe_nonzero)
    gradient[layout.mu] -= message_gradient / repetition
    gradient[layout.mu.start + 1 : layout.mu.stop] += (
        binary_derivative / repetition
    )

    extra = 0.0
    for transition_slice in (layout.tau1, layout.tau2):
        transition = point[transition_slice].reshape((layout.prime, layout.prime))
        full, support, transition_gradient = transition_information(transition)
        extra += full - support
        gradient[transition_slice] += transition_gradient.ravel()

    return extra - message_internal / repetition, gradient


def exponent_from_point(
    point: np.ndarray, layout: bulk.Layout, repetition: int
) -> float:
    return bulk.exponent(point, layout, repetition)


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(args.prime)
    equality, equality_target, upper, upper_target = bulk.constraints(
        layout, args.eta, args.delta
    )
    fixed_message_weight = getattr(args, "fixed_message_weight", None)
    if fixed_message_weight is not None:
        fixed_row = np.zeros(layout.size)
        fixed_row[layout.mu.start + 1 : layout.mu.stop] = 1.0
        equality = np.vstack((equality, fixed_row))
        equality_target = np.concatenate(
            (equality_target, np.array([fixed_message_weight]))
        )
    if args.max_message_weight is not None:
        cap_row = np.zeros(layout.size)
        cap_row[layout.mu.start] = -1.0
        upper = np.vstack((upper, cap_row))
        upper_target = np.concatenate(
            (upper_target, np.array([-(1.0 - args.max_message_weight)]))
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
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(len(vertices), size=min(16, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )
    constraints: list[object] = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    if args.mode == "max-exponent":
        starts = sorted(
            starts,
            key=lambda point: gap_and_gradient(
                point, layout, args.repetition
            )[0],
        )
        constraints.append(
            NonlinearConstraint(
                lambda point: gap_and_gradient(
                    point, layout, args.repetition
                )[0],
                -np.inf,
                args.maximum_gap,
                jac=lambda point: gap_and_gradient(
                    point, layout, args.repetition
                )[1],
            )
        )
    best = None
    for start in starts:
        if args.mode == "max-exponent":
            objective = lambda point: -bulk.exponent(
                point, layout, args.repetition
            )
            objective_jacobian = lambda point: bulk.objective_gradient(
                point, layout, args.repetition
            )
        elif args.mode == "lagrangian":
            objective = lambda point: (
                -bulk.exponent(point, layout, args.repetition)
                + args.gap_multiplier
                * gap_and_gradient(point, layout, args.repetition)[0]
            )
            objective_jacobian = lambda point: (
                bulk.objective_gradient(point, layout, args.repetition)
                + args.gap_multiplier
                * gap_and_gradient(point, layout, args.repetition)[1]
            )
        else:
            objective = (
                lambda point: gap_and_gradient(
                    point, layout, args.repetition
                )[0]
            )
            objective_jacobian = (
                lambda point: gap_and_gradient(
                    point, layout, args.repetition
                )[1]
            )
        result = minimize(
            objective,
            start,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 4000, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        gap_value = gap_and_gradient(result.x, layout, args.repetition)[0]
        nonlinear_feasible = (
            args.mode != "max-exponent"
            or gap_value <= args.maximum_gap + 1e-7
        )
        if (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and nonlinear_feasible
        ):
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible point")
    gap, _ = gap_and_gradient(best.x, layout, args.repetition)
    exponent = exponent_from_point(best.x, layout, args.repetition)
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "max_message_weight": args.max_message_weight,
        "fixed_message_weight": fixed_message_weight,
        "mode": args.mode,
        "maximum_gap_constraint": (
            args.maximum_gap if args.mode == "max-exponent" else None
        ),
        "gap_multiplier": (
            args.gap_multiplier if args.mode == "lagrangian" else None
        ),
        "minimum_gap_found": gap,
        "exponent_at_result": exponent,
        "lagrangian_at_result": (
            exponent - args.gap_multiplier * gap
            if args.mode == "lagrangian"
            else None
        ),
        "message_nonzero_fraction": 1.0 - float(best.x[layout.mu.start]),
        "middle_nonzero_fraction": 1.0 - float(best.x[layout.beta.start]),
        "output_nonzero_fraction": 1.0 - float(best.x[layout.zeta.start]),
        "mu": best.x[layout.mu].tolist(),
        "beta": best.x[layout.beta].tolist(),
        "zeta": best.x[layout.zeta].tolist(),
        "tau1": best.x[layout.tau1].reshape(
            (layout.prime, layout.prime)
        ).tolist(),
        "tau2": best.x[layout.tau2].reshape(
            (layout.prime, layout.prime)
        ).tolist(),
        "first_stage": transition_information(
            best.x[layout.tau1].reshape((layout.prime, layout.prime))
        )[:2],
        "second_stage": transition_information(
            best.x[layout.tau2].reshape((layout.prime, layout.prime))
        )[:2],
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical constrained search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("min-gap", "max-exponent", "lagrangian"),
        default="min-gap",
    )
    parser.add_argument("--maximum-gap", type=float, default=0.0)
    parser.add_argument("--gap-multiplier", type=float, default=1.5)
    parser.add_argument("--max-message-weight", type=float)
    parser.add_argument("--fixed-message-weight", type=float)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
