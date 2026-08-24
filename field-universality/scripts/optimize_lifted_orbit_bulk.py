#!/usr/bin/env python3
"""Optimize the sign-orbit exponent subject to a full-symbol lift."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import optimize_bulk_exponent as full


def sign_orbit_map(prime: int) -> tuple[int, ...]:
    representatives: dict[tuple[int, ...], int] = {}
    result = []
    for value in range(prime):
        orbit = tuple(sorted({value, (-value) % prime}))
        if orbit not in representatives:
            representatives[orbit] = len(representatives)
        result.append(representatives[orbit])
    return tuple(result)


@dataclass(frozen=True)
class Layout:
    prime: int

    @property
    def orbit_count(self) -> int:
        return (self.prime + 1) // 2

    @property
    def orbit_mu(self) -> slice:
        return slice(0, self.orbit_count)

    @property
    def orbit_beta(self) -> slice:
        return slice(self.orbit_count, 2 * self.orbit_count)

    @property
    def orbit_zeta(self) -> slice:
        return slice(2 * self.orbit_count, 3 * self.orbit_count)

    @property
    def full_start(self) -> int:
        return 3 * self.orbit_count

    @property
    def full(self) -> full.Layout:
        return full.Layout(self.prime)

    @property
    def size(self) -> int:
        return self.full_start + self.full.size

    def shifted(self, component: slice) -> slice:
        return slice(self.full_start + component.start, self.full_start + component.stop)


def independent_equalities(
    matrix: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, triangular, pivots = qr(matrix.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = max(matrix.shape) * np.finfo(float).eps * diagonal.max(initial=0.0)
    rank = int(np.count_nonzero(diagonal > tolerance))
    selected = np.sort(pivots[:rank])
    return matrix[selected], target[selected]


def constraints(
    layout: Layout, eta: float, delta: float
) -> tuple[np.ndarray, ...]:
    full_equality, full_target, _, _ = full.constraints(layout.full, eta, delta)
    rows = []
    targets = []
    for row, value in zip(full_equality, full_target, strict=True):
        lifted = np.zeros(layout.size)
        lifted[layout.full_start :] = row
        rows.append(lifted)
        targets.append(value)
    orbit_map = sign_orbit_map(layout.prime)
    for orbit_component, full_component in (
        (layout.orbit_mu, layout.full.mu),
        (layout.orbit_beta, layout.full.beta),
        (layout.orbit_zeta, layout.full.zeta),
    ):
        for orbit in range(layout.orbit_count):
            row = np.zeros(layout.size)
            row[orbit_component.start + orbit] = -1.0
            for symbol, symbol_orbit in enumerate(orbit_map):
                if symbol_orbit == orbit:
                    row[layout.full_start + full_component.start + symbol] = 1.0
            rows.append(row)
            targets.append(0.0)
    equality, equality_target = independent_equalities(
        np.stack(rows), np.array(targets)
    )
    upper = np.zeros((2, layout.size))
    upper[0, layout.orbit_mu.start] = 1.0
    upper[1, layout.orbit_zeta.start] = -1.0
    upper_target = np.array([1.0 - eta, -(1.0 - delta)])
    return equality, equality_target, upper, upper_target


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def orbit_pair(
    point: np.ndarray, layout: Layout, transition: slice
) -> np.ndarray:
    orbit_map = sign_orbit_map(layout.prime)
    pair = np.zeros((layout.orbit_count, layout.orbit_count))
    values = point[layout.shifted(transition)].reshape((layout.prime, layout.prime))
    for source in range(layout.prime):
        for target in range(layout.prime):
            increment = (target - source) % layout.prime
            pair[orbit_map[increment], orbit_map[source]] += values[source, target]
    return pair


def exponent(point: np.ndarray, layout: Layout, repetition: int) -> float:
    message = point[layout.orbit_mu]
    middle = point[layout.orbit_beta]
    output = point[layout.orbit_zeta]
    first_pair = orbit_pair(point, layout, layout.full.tau1)
    second_pair = orbit_pair(point, layout, layout.full.tau2)
    return (
        entropy(first_pair)
        + entropy(second_pair)
        - (1.0 - 1.0 / repetition) * entropy(message)
        - 2.0 * entropy(middle)
        - entropy(output)
        + (1.0 - message[0]) * np.log(2.0) / repetition
    )


def gradient(point: np.ndarray, layout: Layout, repetition: int) -> np.ndarray:
    result = np.zeros(layout.size)
    safe = np.maximum(point, 1e-300)
    entropy_gradient = -(np.log(safe) + 1.0)
    result[layout.orbit_mu] -= (
        1.0 - 1.0 / repetition
    ) * entropy_gradient[layout.orbit_mu]
    result[layout.orbit_beta] -= 2.0 * entropy_gradient[layout.orbit_beta]
    result[layout.orbit_zeta] -= entropy_gradient[layout.orbit_zeta]
    result[layout.orbit_mu.start + 1 : layout.orbit_mu.stop] += (
        np.log(2.0) / repetition
    )
    orbit_map = sign_orbit_map(layout.prime)
    for transition in (layout.full.tau1, layout.full.tau2):
        pair = orbit_pair(point, layout, transition)
        pair_gradient = -(np.log(np.maximum(pair, 1e-300)) + 1.0)
        shifted = layout.shifted(transition)
        for source in range(layout.prime):
            for target in range(layout.prime):
                increment = (target - source) % layout.prime
                index = shifted.start + source * layout.prime + target
                result[index] = pair_gradient[
                    orbit_map[increment], orbit_map[source]
                ]
    return -result


def feasible_vertices(
    layout: Layout,
    equality: np.ndarray,
    equality_target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
    count: int,
    random: np.random.Generator,
) -> list[np.ndarray]:
    vertices = []
    for _ in range(count):
        result = linprog(
            random.standard_normal(layout.size),
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
        raise RuntimeError("the lifted orbit domain is infeasible")
    return vertices


def optimize(args: argparse.Namespace) -> dict[str, object]:
    layout = Layout(args.prime)
    equality, equality_target, upper, upper_target = constraints(
        layout, args.eta, args.delta
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
        starts.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )
    constraints_list = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda point: -exponent(point, layout, args.repetition),
            start,
            jac=lambda point: gradient(point, layout, args.repetition),
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=constraints_list,
            options={"ftol": 1e-12, "maxiter": 4000, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - equality_target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        if equality_error <= 1e-7 and upper_error <= 1e-7:
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible point")
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "lifted_orbit_exponent": exponent(best.x, layout, args.repetition),
        "message_orbit_composition": best.x[layout.orbit_mu].tolist(),
        "middle_orbit_composition": best.x[layout.orbit_beta].tolist(),
        "output_orbit_composition": best.x[layout.orbit_zeta].tolist(),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical lifted-relaxation search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
