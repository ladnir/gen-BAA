#!/usr/bin/env python3
"""Optimize the rigorous sign-orbit relaxation of the bulk exponent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import optimize_bulk_exponent as full


def sign_orbits(prime: int) -> tuple[tuple[int, ...], ...]:
    seen = set()
    orbits = []
    for value in range(prime):
        if value in seen:
            continue
        orbit = tuple(sorted({value, (-value) % prime}))
        seen.update(orbit)
        orbits.append(orbit)
    return tuple(orbits)


def compatible_triples(prime: int) -> tuple[tuple[int, int, int], ...]:
    orbits = sign_orbits(prime)
    triples = []
    for increment, increment_orbit in enumerate(orbits):
        for source, source_orbit in enumerate(orbits):
            for target, target_orbit in enumerate(orbits):
                if any(
                    (state + difference) % prime in target_orbit
                    for difference in increment_orbit
                    for state in source_orbit
                ):
                    triples.append((increment, source, target))
    return tuple(triples)


@dataclass(frozen=True)
class Layout:
    prime: int

    @property
    def orbit_count(self) -> int:
        return (self.prime + 1) // 2

    @property
    def triples(self) -> tuple[tuple[int, int, int], ...]:
        return compatible_triples(self.prime)

    @property
    def mu(self) -> slice:
        return slice(0, self.orbit_count)

    @property
    def beta(self) -> slice:
        return slice(self.orbit_count, 2 * self.orbit_count)

    @property
    def zeta(self) -> slice:
        return slice(2 * self.orbit_count, 3 * self.orbit_count)

    @property
    def gamma1(self) -> slice:
        start = 3 * self.orbit_count
        return slice(start, start + len(self.triples))

    @property
    def gamma2(self) -> slice:
        start = self.gamma1.stop
        return slice(start, start + len(self.triples))

    @property
    def size(self) -> int:
        return self.gamma2.stop


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
    rows = []
    targets = []
    for component in (layout.mu, layout.beta, layout.zeta):
        row = np.zeros(layout.size)
        row[component] = 1.0
        rows.append(row)
        targets.append(1.0)

    for gamma, increment_component, state_component in (
        (layout.gamma1, layout.mu, layout.beta),
        (layout.gamma2, layout.beta, layout.zeta),
    ):
        for orbit in range(layout.orbit_count):
            row = np.zeros(layout.size)
            for index, (increment, _, _) in enumerate(layout.triples):
                if increment == orbit:
                    row[gamma.start + index] = 1.0
            row[increment_component.start + orbit] -= 1.0
            rows.append(row)
            targets.append(0.0)

            row = np.zeros(layout.size)
            for index, (_, source, _) in enumerate(layout.triples):
                if source == orbit:
                    row[gamma.start + index] = 1.0
            row[state_component.start + orbit] -= 1.0
            rows.append(row)
            targets.append(0.0)

            row = np.zeros(layout.size)
            for index, (_, _, target) in enumerate(layout.triples):
                if target == orbit:
                    row[gamma.start + index] = 1.0
            row[state_component.start + orbit] -= 1.0
            rows.append(row)
            targets.append(0.0)

    equality, equality_target = independent_equalities(
        np.stack(rows), np.array(targets)
    )
    upper = np.zeros((2, layout.size))
    upper[0, layout.mu.start] = 1.0
    upper[1, layout.zeta.start] = -1.0
    upper_target = np.array([1.0 - eta, -(1.0 - delta)])
    return equality, equality_target, upper, upper_target


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def pair_distribution(point: np.ndarray, layout: Layout, gamma: slice) -> np.ndarray:
    pair = np.zeros((layout.orbit_count, layout.orbit_count))
    for value, (increment, source, _) in zip(point[gamma], layout.triples):
        pair[increment, source] += value
    return pair


def exponent(point: np.ndarray, layout: Layout, repetition: int) -> float:
    message = point[layout.mu]
    middle = point[layout.beta]
    output = point[layout.zeta]
    first_pair = pair_distribution(point, layout, layout.gamma1)
    second_pair = pair_distribution(point, layout, layout.gamma2)
    internal_value_entropy = (1.0 - message[0]) * np.log(2.0) / repetition
    return (
        entropy(first_pair)
        + entropy(second_pair)
        - (1.0 - 1.0 / repetition) * entropy(message)
        - 2.0 * entropy(middle)
        - entropy(output)
        + internal_value_entropy
    )


def objective_gradient(
    point: np.ndarray, layout: Layout, repetition: int
) -> np.ndarray:
    safe = np.maximum(point, 1e-300)
    entropy_gradient = -(np.log(safe) + 1.0)
    gradient = np.zeros(layout.size)
    gradient[layout.mu] -= (
        1.0 - 1.0 / repetition
    ) * entropy_gradient[layout.mu]
    gradient[layout.beta] -= 2.0 * entropy_gradient[layout.beta]
    gradient[layout.zeta] -= entropy_gradient[layout.zeta]
    gradient[layout.mu.start + 1 : layout.mu.stop] += np.log(2.0) / repetition
    for gamma in (layout.gamma1, layout.gamma2):
        pair = pair_distribution(point, layout, gamma)
        pair_gradient = -(np.log(np.maximum(pair, 1e-300)) + 1.0)
        for index, (increment, source, _) in enumerate(layout.triples):
            gradient[gamma.start + index] = pair_gradient[increment, source]
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
        raise RuntimeError("the orbit relaxation is infeasible")
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
            options={"ftol": 1e-12, "maxiter": 3000, "disp": False},
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
        "orbit_count": layout.orbit_count,
        "compatible_triple_count": len(layout.triples),
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "exponent_upper_relaxation": exponent(best.x, layout, args.repetition),
        "message_orbit_composition": best.x[layout.mu].tolist(),
        "middle_orbit_composition": best.x[layout.beta].tolist(),
        "output_orbit_composition": best.x[layout.zeta].tolist(),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "status": "numerical upper-relaxation search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.196)
    parser.add_argument("--eta", type=float, default=0.0001)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not full.is_prime(args.prime) or args.prime == 2:
        parser.error("prime must be an odd prime")
    return args


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
