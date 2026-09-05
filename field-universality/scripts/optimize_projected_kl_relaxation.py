#!/usr/bin/env python3
"""Optimize the 20-cell support-sum KL relaxation for two accumulators.

This discards liftability of the projected tables to full transition laws.
Its objective is therefore a pointwise upper bound on the exact ensemble
exponent.  A negative global maximum would reduce the F_5 proof to this
55-variable finite KL problem.  The search itself is numerical only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import diagnose_fourier_information_bound as projection
import optimize_bulk_exponent as bulk
import projected_kl_dual as projection_dual


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
    def w1(self) -> slice:
        return slice(3 * self.prime, 7 * self.prime)

    @property
    def w2(self) -> slice:
        return slice(7 * self.prime, 11 * self.prime)

    @property
    def size(self) -> int:
        return 11 * self.prime

    def projected(self, point: np.ndarray, stage: int) -> np.ndarray:
        selected = self.w1 if stage == 1 else self.w2
        return point[selected].reshape((2, 2, self.prime))


def constraints(
    layout: Layout, eta: float, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Bounds]:
    equalities = []
    targets = []
    for component in (layout.mu, layout.beta, layout.zeta):
        row = np.zeros(layout.size)
        row[component] = 1.0
        equalities.append(row)
        targets.append(1.0)
    for stage, increment, state, projected_slice in (
        (1, layout.mu, layout.beta, layout.w1),
        (2, layout.beta, layout.zeta, layout.w2),
    ):
        base = projected_slice.start
        for target_value in range(layout.prime):
            row = np.zeros(layout.size)
            for u in range(2):
                for v in range(2):
                    row[base + (2 * u + v) * layout.prime + target_value] = 1.0
            row[state.start + target_value] -= 1.0
            equalities.append(row)
            targets.append(0.0)
        row_u = np.zeros(layout.size)
        for v in range(2):
            start = base + (2 + v) * layout.prime
            row_u[start : start + layout.prime] = 1.0
        row_u[increment.start + 1 : increment.stop] -= 1.0
        equalities.append(row_u)
        targets.append(0.0)
        row_v = np.zeros(layout.size)
        for u in range(2):
            start = base + (2 * u + 1) * layout.prime
            row_v[start : start + layout.prime] = 1.0
        row_v[state.start + 1 : state.stop] -= 1.0
        equalities.append(row_v)
        targets.append(0.0)
    inequalities = []
    inequality_targets = []
    message_row = np.zeros(layout.size)
    message_row[layout.mu.start] = 1.0
    inequalities.append(message_row)
    inequality_targets.append(1.0 - eta)
    output_row = np.zeros(layout.size)
    output_row[layout.zeta.start] = -1.0
    inequalities.append(output_row)
    inequality_targets.append(-(1.0 - delta))
    lower = np.zeros(layout.size)
    upper = np.ones(layout.size)
    for projected_slice in (layout.w1, layout.w2):
        base = projected_slice.start
        for target_value in range(1, layout.prime):
            upper[base + target_value] = 0.0
        upper[base + layout.prime] = 0.0
        upper[base + 2 * layout.prime] = 0.0
    return (
        np.stack(equalities),
        np.array(targets),
        np.stack(inequalities),
        np.array(inequality_targets),
        Bounds(lower, upper),
    )


def kl(actual: np.ndarray, comparison: np.ndarray) -> float:
    positive = actual > 1e-300
    if np.any(comparison[positive] <= 0.0):
        return float("inf")
    return float(
        np.dot(
            actual[positive],
            np.log(actual[positive] / comparison[positive]),
        )
    )


def objective(point: np.ndarray, layout: Layout, repetition: int) -> float:
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    first_comparison = projection.independent_support_sum_projection(mu, beta)
    second_comparison = projection.independent_support_sum_projection(beta, zeta)
    return (
        bulk.entropy(mu) / repetition
        - kl(layout.projected(point, 1), first_comparison)
        - kl(layout.projected(point, 2), second_comparison)
    )


def feasible_starts(
    layout: Layout,
    equality: np.ndarray,
    target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
    bounds: Bounds,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    random = np.random.default_rng(seed)
    vertices = []
    for _ in range(max(64, 4 * count)):
        result = linprog(
            random.standard_normal(layout.size),
            A_ub=upper,
            b_ub=upper_target,
            A_eq=equality,
            b_eq=target,
            bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
            method="highs",
        )
        if result.success:
            vertices.append(result.x)
    if not vertices:
        raise RuntimeError("reduced projection polytope is empty")
    starts = [np.mean(vertices, axis=0)]
    for _ in range(count - 1):
        chosen = random.choice(len(vertices), size=min(24, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(
                weight * vertices[index]
                for weight, index in zip(weights, chosen, strict=True)
            )
        )
    return starts


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = Layout(args.prime)
    equality, target, upper, upper_target, bounds = constraints(
        layout, args.eta, args.delta
    )
    starts = feasible_starts(
        layout,
        equality,
        target,
        upper,
        upper_target,
        bounds,
        args.restarts,
        args.seed,
    )
    linear_constraints = [
        LinearConstraint(equality, target, target),
        LinearConstraint(upper, -np.inf, upper_target),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda point: -objective(point, layout, args.repetition),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=linear_constraints,
            options={"ftol": 1e-11, "maxiter": args.max_iterations, "disp": False},
        )
        equality_error = float(np.max(np.abs(equality @ result.x - target)))
        upper_error = float(np.max(upper @ result.x - upper_target))
        if (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and np.isfinite(result.fun)
            and (best is None or result.fun < best.fun)
        ):
            best = result
    if best is None:
        raise RuntimeError("no reduced restart returned a feasible point")
    first_actual = kl(
        layout.projected(best.x, 1),
        projection.independent_support_sum_projection(
            best.x[layout.mu], best.x[layout.beta]
        ),
    )
    second_actual = kl(
        layout.projected(best.x, 2),
        projection.independent_support_sum_projection(
            best.x[layout.beta], best.x[layout.zeta]
        ),
    )
    first_dual = projection_dual.fit_projection_dual(
        best.x[layout.mu], best.x[layout.beta]
    )
    second_dual = projection_dual.fit_projection_dual(
        best.x[layout.beta], best.x[layout.zeta]
    )
    return {
        "maximum_found": -float(best.fun),
        "message": best.x[layout.mu].tolist(),
        "middle": best.x[layout.beta].tolist(),
        "output": best.x[layout.zeta].tolist(),
        "message_support": float(best.x[layout.mu][1:].sum()),
        "output_support": float(best.x[layout.zeta][1:].sum()),
        "first_projected_kl": first_actual,
        "first_dual_value": first_dual.value,
        "first_duality_gap": first_actual - first_dual.value,
        "first_dual_gradient_residual": first_dual.gradient_residual,
        "second_projected_kl": second_actual,
        "second_dual_value": second_dual.value,
        "second_duality_gap": second_actual - second_dual.value,
        "second_dual_gradient_residual": second_dual.gradient_residual,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "variables": layout.size,
        "status": "numerical projected-KL relaxation search only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))
