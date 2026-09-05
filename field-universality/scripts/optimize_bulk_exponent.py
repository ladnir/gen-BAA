#!/usr/bin/env python3
"""Heuristic multistart optimizer for the addition-only bulk exponent.

The program searches the finite-dimensional Markov-type optimization derived
in CLEAN_THEORY.md. A positive value supplies a candidate bad type. A negative
value is not a proof of global negativity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1 if divisor == 2 else 2
    return True


@dataclass(frozen=True)
class Layout:
    prime: int

    @property
    def mu(self) -> slice:
        return slice(0, self.prime)

    @property
    def beta(self) -> slice:
        return slice(self.prime, 2 * self.prime)

    @property
    def zeta(self) -> slice:
        return slice(2 * self.prime, 3 * self.prime)

    @property
    def tau1(self) -> slice:
        start = 3 * self.prime
        return slice(start, start + self.prime * self.prime)

    @property
    def tau2(self) -> slice:
        start = 3 * self.prime + self.prime * self.prime
        return slice(start, start + self.prime * self.prime)

    @property
    def size(self) -> int:
        return 3 * self.prime + 2 * self.prime * self.prime

    def edge(self, layer: int, source: int, target: int) -> int:
        start = self.tau1.start if layer == 1 else self.tau2.start
        return start + source * self.prime + target


def independent_equalities(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, triangular, pivots = qr(matrix.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = max(matrix.shape) * np.finfo(float).eps * diagonal.max(initial=0.0)
    rank = int(np.count_nonzero(diagonal > tolerance))
    selected = np.sort(pivots[:rank])
    return matrix[selected], target[selected]


def constraints(layout: Layout, eta: float, delta: float) -> tuple[np.ndarray, ...]:
    p = layout.prime
    rows: list[np.ndarray] = []
    targets: list[float] = []

    for component in (layout.mu, layout.beta, layout.zeta):
        row = np.zeros(layout.size)
        row[component] = 1.0
        rows.append(row)
        targets.append(1.0)

    for layer, state_component, increment_component in (
        (1, layout.beta, layout.mu),
        (2, layout.zeta, layout.beta),
    ):
        for source in range(p):
            row = np.zeros(layout.size)
            for target in range(p):
                row[layout.edge(layer, source, target)] = 1.0
            row[state_component.start + source] -= 1.0
            rows.append(row)
            targets.append(0.0)
        for target in range(p):
            row = np.zeros(layout.size)
            for source in range(p):
                row[layout.edge(layer, source, target)] = 1.0
            row[state_component.start + target] -= 1.0
            rows.append(row)
            targets.append(0.0)
        for increment in range(p):
            row = np.zeros(layout.size)
            for source in range(p):
                target = (source + increment) % p
                row[layout.edge(layer, source, target)] = 1.0
            row[increment_component.start + increment] -= 1.0
            rows.append(row)
            targets.append(0.0)

    equality, equality_target = independent_equalities(np.stack(rows), np.array(targets))
    upper = np.zeros((2, layout.size))
    upper[0, layout.mu.start] = 1.0
    upper[1, layout.zeta.start] = -1.0
    upper_target = np.array([1.0 - eta, -(1.0 - delta)])
    return equality, equality_target, upper, upper_target


def append_support_box_constraints(
    layout: Layout,
    rows: np.ndarray,
    targets: np.ndarray,
    lower: tuple[float, float, float, float, float],
    upper: tuple[float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = []
    for component in (layout.mu, layout.beta, layout.zeta):
        row = np.zeros(layout.size)
        row[component.start + 1 : component.stop] = 1.0
        coordinates.append(row)
    for layer in (1, 2):
        row = np.zeros(layout.size)
        for target in range(1, layout.prime):
            row[layout.edge(layer, 0, target)] = 1.0
        coordinates.append(row)
    additional_rows = []
    additional_targets = []
    for row, lower_value, upper_value in zip(
        coordinates, lower, upper, strict=True
    ):
        additional_rows.extend((row, -row))
        additional_targets.extend((upper_value, -lower_value))
    return (
        np.concatenate((rows, np.stack(additional_rows))),
        np.concatenate((targets, np.array(additional_targets))),
    )


def add_sign_symmetry(
    layout: Layout, equality: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row.copy() for row in equality]
    targets = target.tolist()
    p = layout.prime
    for component in (layout.mu, layout.beta, layout.zeta):
        for value in range(1, p):
            opposite = (-value) % p
            if value >= opposite:
                continue
            row = np.zeros(layout.size)
            row[component.start + value] = 1.0
            row[component.start + opposite] = -1.0
            rows.append(row)
            targets.append(0.0)
    for layer in (1, 2):
        for source in range(p):
            for destination in range(p):
                opposite_source = (-source) % p
                opposite_destination = (-destination) % p
                if (source, destination) >= (
                    opposite_source,
                    opposite_destination,
                ):
                    continue
                row = np.zeros(layout.size)
                row[layout.edge(layer, source, destination)] = 1.0
                row[
                    layout.edge(layer, opposite_source, opposite_destination)
                ] = -1.0
                rows.append(row)
                targets.append(0.0)
    return independent_equalities(np.stack(rows), np.array(targets))


def entropy(vector: np.ndarray) -> float:
    positive = vector[vector > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def exponent(point: np.ndarray, layout: Layout, repetition: int) -> float:
    return (
        entropy(point[layout.tau1])
        + entropy(point[layout.tau2])
        - (1.0 - 1.0 / repetition) * entropy(point[layout.mu])
        - 2.0 * entropy(point[layout.beta])
        - entropy(point[layout.zeta])
    )


def objective_gradient(point: np.ndarray, layout: Layout, repetition: int) -> np.ndarray:
    safe = np.maximum(point, 1e-300)
    entropy_gradient = -(np.log(safe) + 1.0)
    gradient = np.zeros_like(point)
    gradient[layout.tau1] = entropy_gradient[layout.tau1]
    gradient[layout.tau2] = entropy_gradient[layout.tau2]
    gradient[layout.mu] -= (1.0 - 1.0 / repetition) * entropy_gradient[layout.mu]
    gradient[layout.beta] -= 2.0 * entropy_gradient[layout.beta]
    gradient[layout.zeta] -= entropy_gradient[layout.zeta]
    return -gradient


def feasible_vertices(
    layout: Layout,
    equality: np.ndarray,
    equality_target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
    count: int,
    random: np.random.Generator,
) -> list[np.ndarray]:
    vertices: list[np.ndarray] = []
    for _ in range(count):
        objective = random.standard_normal(layout.size)
        result = linprog(
            objective,
            A_ub=upper,
            b_ub=upper_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=(0.0, 1.0),
            method="highs",
        )
        if result.success:
            vertices.append(result.x)
    if not vertices:
        raise RuntimeError("the requested bulk type region is infeasible")
    return vertices


def optimize(args: argparse.Namespace) -> dict[str, object]:
    layout = Layout(args.prime)
    equality, equality_target, upper, upper_target = constraints(layout, args.eta, args.delta)
    support_lower = getattr(args, "support_lower", None)
    support_upper = getattr(args, "support_upper", None)
    if support_lower is not None:
        upper, upper_target = append_support_box_constraints(
            layout,
            upper,
            upper_target,
            support_lower,
            support_upper,
        )
    if args.sign_symmetric:
        equality, equality_target = add_sign_symmetry(
            layout, equality, equality_target
        )
    random = np.random.default_rng(args.seed)
    vertices = feasible_vertices(
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
        chosen = random.choice(len(vertices), size=min(12, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(sum(weight * vertices[index] for weight, index in zip(weights, chosen)))

    linear_constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda point: -exponent(point, layout, args.repetition),
            start,
            jac=lambda point: objective_gradient(point, layout, args.repetition),
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=linear_constraints,
            options={"ftol": args.tolerance, "maxiter": args.max_iterations, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        feasible = equality_error <= 1e-7 and upper_error <= 1e-7
        if feasible and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("no restart returned a numerically feasible point")

    point = best.x
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "restarts": args.restarts,
        "sign_symmetric": args.sign_symmetric,
        "exponent": exponent(point, layout, args.repetition),
        "message_nonzero_fraction": 1.0 - point[layout.mu.start],
        "output_nonzero_fraction": 1.0 - point[layout.zeta.start],
        "middle_nonzero_fraction": 1.0 - point[layout.beta.start],
        "first_crossing_fraction": float(
            sum(point[layout.edge(1, 0, target)] for target in range(1, layout.prime))
        ),
        "second_crossing_fraction": float(
            sum(point[layout.edge(2, 0, target)] for target in range(1, layout.prime))
        ),
        "support_lower": support_lower,
        "support_upper": support_upper,
        "mu": point[layout.mu].tolist(),
        "beta": point[layout.beta].tolist(),
        "zeta": point[layout.zeta].tolist(),
        "tau1": point[layout.tau1].reshape((layout.prime, layout.prime)).tolist(),
        "tau2": point[layout.tau2].reshape((layout.prime, layout.prime)).tolist(),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "iterations": int(best.nit),
        "equality_residual": float(np.max(np.abs(equality @ point - equality_target))),
        "inequality_residual": float(max(0.0, np.max(upper @ point - upper_target))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--sign-symmetric", action="store_true")
    parser.add_argument(
        "--support-lower",
        type=lambda value: tuple(float(entry) for entry in value.split(",")),
    )
    parser.add_argument(
        "--support-upper",
        type=lambda value: tuple(float(entry) for entry in value.split(",")),
    )
    args = parser.parse_args()
    if not is_prime(args.prime):
        parser.error("prime must be a prime integer")
    if args.repetition < 1:
        parser.error("repetition must be positive")
    if not 0.0 < args.delta < 1.0:
        parser.error("delta must lie in (0,1)")
    if not 0.0 < args.eta < 1.0:
        parser.error("eta must lie in (0,1)")
    if args.restarts < 1:
        parser.error("restarts must be positive")
    if (args.support_lower is None) != (args.support_upper is None):
        parser.error("support-lower and support-upper must be supplied together")
    if args.support_lower is not None:
        if len(args.support_lower) != 5 or len(args.support_upper) != 5:
            parser.error("support bounds must contain A,B,C,x1,x2")
        if any(
            lower > upper
            for lower, upper in zip(
                args.support_lower, args.support_upper, strict=True
            )
        ):
            parser.error("every support lower bound must not exceed its upper bound")
    return args


if __name__ == "__main__":
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))
