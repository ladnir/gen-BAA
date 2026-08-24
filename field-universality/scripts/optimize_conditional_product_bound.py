#!/usr/bin/env python3
"""Optimize the composition-only conditional-product upper bound.

The bound is rigorous as an inequality, but this program is only a numerical
search for its maximum.  A negative value from this program is not a global
certificate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def relative_entropy(first: np.ndarray, second: np.ndarray) -> float:
    positive = first > 0.0
    if np.any(second[positive] <= 0.0):
        return float("inf")
    return float(
        np.dot(first[positive], np.log(first[positive] / second[positive]))
    )


def support_joint(increment: float, state: float, crossing: float) -> np.ndarray:
    return np.array(
        [
            1.0 - state - crossing,
            state - increment + crossing,
            crossing,
            increment - crossing,
        ]
    )


def support_information(
    increment: float, state: float, crossing: float
) -> float:
    return (
        entropy(np.array([1.0 - increment, increment]))
        + entropy(np.array([1.0 - state, state]))
        - entropy(support_joint(increment, state, crossing))
    )


def conditional_product_target(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> np.ndarray:
    prime = increment.size
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    joint = support_joint(increment_support, state_support, crossing)
    target = np.zeros(prime)
    target[0] = joint[0]
    if increment_support > 0.0:
        target[1:] += joint[2] * increment[1:] / increment_support
    if state_support > 0.0:
        target[1:] += joint[1] * state[1:] / state_support
    if increment_support > 0.0 and state_support > 0.0:
        scale = joint[3] / (increment_support * state_support)
        for difference in range(1, prime):
            for source in range(1, prime):
                target[(source + difference) % prime] += (
                    scale * increment[difference] * state[source]
                )
    return target


def conditional_product_cell_targets(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> np.ndarray:
    """Return the reference law of (support cell, target symbol)."""
    prime = increment.size
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    joint = support_joint(increment_support, state_support, crossing)
    cells = np.zeros((4, prime))
    cells[0, 0] = joint[0]
    if state_support > 0.0:
        cells[1, 1:] = joint[1] * state[1:] / state_support
    if increment_support > 0.0:
        cells[2, 1:] = joint[2] * increment[1:] / increment_support
    if increment_support > 0.0 and state_support > 0.0:
        scale = joint[3] / (increment_support * state_support)
        for difference in range(1, prime):
            for source in range(1, prime):
                cells[3, (source + difference) % prime] += (
                    scale * increment[difference] * state[source]
                )
    return cells


def matrix_scaling_divergence(
    reference: np.ndarray,
    row_marginal: np.ndarray,
    column_marginal: np.ndarray,
    tolerance: float = 1e-13,
    max_iterations: int = 10_000,
) -> tuple[float, np.ndarray]:
    """Return the KL projection onto prescribed row and column marginals."""
    projection = reference.copy()
    for _ in range(max_iterations):
        current_rows = projection.sum(axis=1)
        if np.any((row_marginal > 0.0) & (current_rows <= 0.0)):
            return float("inf"), np.zeros_like(reference)
        row_scale = np.divide(
            row_marginal,
            current_rows,
            out=np.zeros_like(row_marginal),
            where=current_rows > 0.0,
        )
        projection *= row_scale[:, np.newaxis]
        current_columns = projection.sum(axis=0)
        if np.any((column_marginal > 0.0) & (current_columns <= 0.0)):
            return float("inf"), np.zeros_like(reference)
        column_scale = np.divide(
            column_marginal,
            current_columns,
            out=np.zeros_like(column_marginal),
            where=current_columns > 0.0,
        )
        projection *= column_scale[np.newaxis, :]
        error = max(
            float(np.max(np.abs(projection.sum(axis=1) - row_marginal))),
            float(np.max(np.abs(projection.sum(axis=0) - column_marginal))),
        )
        if error <= tolerance:
            positive = projection > 0.0
            divergence = float(
                np.sum(
                    projection[positive]
                    * np.log(projection[positive] / reference[positive])
                )
            )
            return divergence, projection
    return float("nan"), projection


def cell_target_projection_divergence(
    reference: np.ndarray, target: np.ndarray
) -> float:
    """KL projection for the structural four-cell target table."""
    rows = reference.sum(axis=1)
    zero_from_both_nonzero = float(target[0] - rows[0])
    if (
        zero_from_both_nonzero < -1e-12
        or zero_from_both_nonzero > rows[3] + 1e-12
    ):
        return float("inf")
    zero_from_both_nonzero = min(
        max(zero_from_both_nonzero, 0.0), float(rows[3])
    )
    projection = np.zeros_like(reference)
    projection[0, 0] = rows[0]
    projection[3, 0] = zero_from_both_nonzero
    reduced_rows = np.array(
        [rows[1], rows[2], rows[3] - zero_from_both_nonzero]
    )
    active = reduced_rows > 1e-15
    if np.any(target[1:] > 0.0) and not np.any(active):
        return float("inf")
    if np.any(active):
        reduced_divergence, reduced_projection = matrix_scaling_divergence(
            reference[1:4, 1:][active],
            reduced_rows[active],
            target[1:],
        )
        if not np.isfinite(reduced_divergence):
            return float("inf")
        for index, row in zip(
            np.flatnonzero(active), reduced_projection, strict=True
        ):
            projection[1 + index, 1:] = row
    positive = projection > 0.0
    if np.any(reference[positive] <= 0.0):
        return float("inf")
    return float(
        np.sum(
            projection[positive]
            * np.log(projection[positive] / reference[positive])
        )
    )


def projected_divergence(
    increment: np.ndarray,
    state: np.ndarray,
    crossing: float,
    projection: str,
) -> float:
    if projection == "cancellation":
        return cancellation_divergence(increment, state, crossing)
    if projection == "two-level":
        return two_level_divergence(increment, state, crossing)
    reference = conditional_product_cell_targets(increment, state, crossing)
    if projection == "target":
        return relative_entropy(state, reference.sum(axis=0))
    return cell_target_projection_divergence(reference, state)


def cancellation_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    both_nonzero = increment_support - crossing
    if both_nonzero <= 0.0:
        return 0.0
    cancellation = 0.0
    if increment_support > 0.0 and state_support > 0.0:
        for difference in range(1, increment.size):
            cancellation += (
                increment[difference]
                * state[(-difference) % increment.size]
            )
        cancellation /= increment_support * state_support
    observed = crossing / both_nonzero
    return both_nonzero * relative_entropy(
        np.array([observed, 1.0 - observed]),
        np.array([cancellation, 1.0 - cancellation]),
    )


def two_level_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    both_nonzero = increment_support - crossing
    if increment_support <= 0.0 or state_support <= 0.0:
        return 0.0
    increment_nonzero = increment[1:] / increment_support
    state_nonzero = state[1:] / state_support
    convolution = np.zeros(increment.size)
    for difference in range(1, increment.size):
        for source in range(1, increment.size):
            convolution[(source + difference) % increment.size] += (
                increment[difference]
                / increment_support
                * state[source]
                / state_support
            )
    cancellation = float(convolution[0])
    observed = crossing / both_nonzero if both_nonzero > 0.0 else 0.0
    cancellation_cost = both_nonzero * relative_entropy(
        np.array([observed, 1.0 - observed]),
        np.array([cancellation, 1.0 - cancellation]),
    )
    row_state = state_support - increment_support + crossing
    row_increment = crossing
    row_convolution = increment_support - 2.0 * crossing
    if row_convolution < -1e-12:
        return float("inf")
    row_convolution = max(row_convolution, 0.0)
    reference_nonzero = (
        row_state * state_nonzero + row_increment * increment_nonzero
    )
    if row_convolution > 0.0:
        if cancellation >= 1.0:
            return float("inf")
        reference_nonzero += (
            row_convolution * convolution[1:] / (1.0 - cancellation)
        )
    reference_nonzero /= state_support
    return cancellation_cost + state_support * relative_entropy(
        state_nonzero, reference_nonzero
    )


def two_level_dual_potentials(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive cell/target potentials attaining the two-level cut."""
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    both_nonzero = increment_support - crossing
    increment_nonzero = increment[1:] / increment_support
    state_nonzero = state[1:] / state_support
    convolution = np.zeros(increment.size)
    for difference in range(1, increment.size):
        for source in range(1, increment.size):
            convolution[(source + difference) % increment.size] += (
                increment[difference]
                / increment_support
                * state[source]
                / state_support
            )
    cancellation = float(convolution[0])
    observed_cancel = crossing
    observed_nonzero = increment_support - 2.0 * crossing
    reference_cancel = both_nonzero * cancellation
    reference_nonzero_mass = both_nonzero * (1.0 - cancellation)
    if min(
        observed_cancel,
        observed_nonzero,
        reference_cancel,
        reference_nonzero_mass,
    ) <= 0.0:
        raise ValueError("two-level potentials require an interior reference")
    cancel_potential = np.log(observed_cancel / reference_cancel)
    nonzero_potential = np.log(
        observed_nonzero / reference_nonzero_mass
    )
    row_state = state_support - increment_support + crossing
    reference_target = (
        row_state * state_nonzero + crossing * increment_nonzero
    )
    reference_target += (
        observed_nonzero * convolution[1:] / (1.0 - cancellation)
    )
    if np.any(state[1:] <= 0.0) or np.any(reference_target <= 0.0):
        raise ValueError("target potentials require positive nonzero masses")
    target_nonzero_potential = np.log(state[1:] / reference_target)
    target_potential = np.empty(increment.size)
    target_potential[0] = cancel_potential - nonzero_potential
    target_potential[1:] = target_nonzero_potential
    cell_potential = np.array(
        [
            -target_potential[0],
            0.0,
            0.0,
            nonzero_potential,
        ]
    )
    return cell_potential, target_potential


def cell_target_dual_lower_bound(
    reference: np.ndarray,
    row_marginal: np.ndarray,
    target: np.ndarray,
    cell_potential: np.ndarray,
    target_potential: np.ndarray,
) -> float:
    exponent = np.exp(
        cell_potential[:, np.newaxis]
        + target_potential[np.newaxis, :]
    )
    return float(
        np.dot(cell_potential, row_marginal)
        + np.dot(target_potential, target)
        - np.sum(reference * (exponent - 1.0))
    )


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
    def crossing1(self) -> int:
        return 3 * self.prime

    @property
    def crossing2(self) -> int:
        return 3 * self.prime + 1

    @property
    def size(self) -> int:
        return 3 * self.prime + 2


def expand_free_point(free: np.ndarray, layout: Layout) -> np.ndarray:
    width = layout.prime - 1
    point = np.zeros(layout.size)
    for component_index, component in enumerate(
        (layout.mu, layout.beta, layout.zeta)
    ):
        values = free[component_index * width : (component_index + 1) * width]
        point[component.start] = 1.0 - float(values.sum())
        point[component.start + 1 : component.stop] = values
    point[layout.crossing1 :] = free[3 * width :]
    return point


def compress_point(point: np.ndarray, layout: Layout) -> np.ndarray:
    return np.concatenate(
        (
            point[layout.mu.start + 1 : layout.mu.stop],
            point[layout.beta.start + 1 : layout.beta.stop],
            point[layout.zeta.start + 1 : layout.zeta.stop],
            point[layout.crossing1 :],
        )
    )


def free_constraints(
    layout: Layout, eta: float, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    width = layout.prime - 1
    size = 3 * width + 2
    mu = slice(0, width)
    beta = slice(width, 2 * width)
    zeta = slice(2 * width, 3 * width)
    crossing1 = 3 * width
    crossing2 = crossing1 + 1
    rows = []
    targets = []

    for component, target in ((mu, 1.0), (beta, 1.0), (zeta, delta)):
        row = np.zeros(size)
        row[component] = 1.0
        rows.append(row)
        targets.append(target)
    row = np.zeros(size)
    row[mu] = -1.0
    rows.append(row)
    targets.append(-eta)

    for increment, state, crossing in (
        (mu, beta, crossing1),
        (beta, zeta, crossing2),
    ):
        row = np.zeros(size)
        row[increment] = 1.0
        row[state] = -1.0
        row[crossing] = -1.0
        rows.append(row)
        targets.append(0.0)

        row = np.zeros(size)
        row[increment] = -0.5
        row[crossing] = 1.0
        rows.append(row)
        targets.append(0.0)

        row = np.zeros(size)
        row[state] = 1.0
        row[crossing] = 1.0
        rows.append(row)
        targets.append(1.0)

    return np.stack(rows), np.array(targets)


def linear_constraints(
    layout: Layout, eta: float, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equality = np.zeros((3, layout.size))
    for row, component in enumerate((layout.mu, layout.beta, layout.zeta)):
        equality[row, component] = 1.0
    equality_target = np.ones(3)
    rows = []
    targets = []

    message_cutoff = np.zeros(layout.size)
    message_cutoff[layout.mu.start] = 1.0
    rows.append(message_cutoff)
    targets.append(1.0 - eta)

    output_cutoff = np.zeros(layout.size)
    output_cutoff[layout.zeta.start] = -1.0
    rows.append(output_cutoff)
    targets.append(-(1.0 - delta))

    for increment, state, crossing in (
        (layout.mu, layout.beta, layout.crossing1),
        (layout.beta, layout.zeta, layout.crossing2),
    ):
        # crossing >= increment_support-state_support
        row = np.zeros(layout.size)
        row[state.start] = 1.0
        row[increment.start] = -1.0
        row[crossing] = -1.0
        rows.append(row)
        targets.append(0.0)

        # crossing <= increment_support/2
        row = np.zeros(layout.size)
        row[increment.start] = 0.5
        row[crossing] = 1.0
        rows.append(row)
        targets.append(0.5)

        # crossing <= zero-state mass
        row = np.zeros(layout.size)
        row[state.start] = -1.0
        row[crossing] = 1.0
        rows.append(row)
        targets.append(0.0)

    return equality, equality_target, np.stack(rows), np.array(targets)


def upper_bound(
    point: np.ndarray,
    layout: Layout,
    repetition: int,
    projection: str = "cell-target",
) -> float:
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    first_crossing = float(point[layout.crossing1])
    second_crossing = float(point[layout.crossing2])
    message_support = float(mu[1:].sum())
    middle_support = float(beta[1:].sum())
    output_support = float(zeta[1:].sum())
    message_support_entropy = entropy(
        np.array([1.0 - message_support, message_support])
    )
    phi = message_support_entropy / repetition
    phi -= support_information(
        message_support, middle_support, first_crossing
    )
    phi -= support_information(
        middle_support, output_support, second_crossing
    )
    internal_message = entropy(mu) - message_support_entropy
    first_divergence = projected_divergence(
        mu, beta, first_crossing, projection
    )
    second_divergence = projected_divergence(
        beta, zeta, second_crossing, projection
    )
    return (
        phi
        + internal_message / repetition
        - first_divergence
        - second_divergence
    )


def feasible_vertices(
    equality: np.ndarray,
    equality_target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
    bounds: list[tuple[float, float]],
    count: int,
    random: np.random.Generator,
) -> list[np.ndarray]:
    vertices = []
    for _ in range(count):
        result = linprog(
            random.normal(size=len(bounds)),
            A_ub=upper,
            b_ub=upper_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=bounds,
            method="highs",
        )
        if result.success:
            vertices.append(result.x)
    if not vertices:
        raise RuntimeError("could not construct a feasible reduced point")
    return vertices


def optimize(args: argparse.Namespace) -> dict[str, object]:
    layout = Layout(args.prime)
    equality, equality_target, upper, upper_target = linear_constraints(
        layout, args.eta, args.delta
    )
    bounds = [(0.0, 1.0)] * (3 * args.prime) + [(0.0, 0.5)] * 2
    random = np.random.default_rng(args.seed)
    vertices = feasible_vertices(
        equality,
        equality_target,
        upper,
        upper_target,
        bounds,
        max(64, args.restarts * 4),
        random,
    )
    starts_full = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(len(vertices), size=min(24, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts_full.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )
    small_support = min(args.eta * 1.01, 0.99)
    middle_support = 0.998 * small_support
    output_support = 0.999 * args.delta
    tiny = min(1e-8, small_support / (1000 * (args.prime - 1)))
    orbit_representatives = range(1, (args.prime + 1) // 2)
    for representative in orbit_representatives:
        opposite = (-representative) % args.prime
        for concentration in (0.5, 0.9, 0.99):
            mu = np.full(args.prime, tiny)
            beta = np.full(args.prime, tiny)
            zeta = np.full(args.prime, tiny)
            mu[0] = 0.0
            beta[0] = 0.0
            zeta[0] = 0.0
            mu[representative] += (
                small_support - tiny * (args.prime - 1)
            ) / 2
            mu[opposite] += (
                small_support - tiny * (args.prime - 1)
            ) / 2
            beta[representative] += (
                middle_support - tiny * (args.prime - 1)
            ) / 2
            beta[opposite] += (
                middle_support - tiny * (args.prime - 1)
            ) / 2
            concentrated_mass = concentration * output_support
            zeta[representative] += concentrated_mass / 2
            zeta[opposite] += concentrated_mass / 2
            remaining = (
                output_support
                - concentrated_mass
                - tiny * (args.prime - 1)
            )
            for value in range(1, args.prime):
                zeta[value] += remaining / (args.prime - 1)
            mu[0] = 1.0 - mu[1:].sum()
            beta[0] = 1.0 - beta[1:].sum()
            zeta[0] = 1.0 - zeta[1:].sum()
            starts_full.append(
                np.concatenate(
                    (
                        mu,
                        beta,
                        zeta,
                        np.array(
                            [0.49 * small_support, 0.45 * middle_support]
                        ),
                    )
                )
            )
    free_upper, free_upper_target = free_constraints(
        layout, args.eta, args.delta
    )
    free_size = 3 * (args.prime - 1) + 2
    free_bounds = Bounds(
        np.zeros(free_size),
        np.concatenate((np.ones(free_size - 2), np.array([0.5, 0.5]))),
    )
    constraints = (
        LinearConstraint(free_upper, -np.inf, free_upper_target),
    )
    best = None
    for start_full in starts_full:
        start = compress_point(start_full, layout)
        result = minimize(
            lambda point: -upper_bound(
                expand_free_point(point, layout),
                layout,
                args.repetition,
                args.projection,
            ),
            start,
            method="SLSQP",
            bounds=free_bounds,
            constraints=constraints,
            options={
                "ftol": args.tolerance,
                "maxiter": args.max_iterations,
                "disp": False,
            },
        )
        upper_error = float(
            np.max(free_upper @ result.x - free_upper_target)
        )
        if upper_error > 1e-7:
            continue
        full_point = expand_free_point(result.x, layout)
        value = upper_bound(
            full_point, layout, args.repetition, args.projection
        )
        if np.isfinite(value) and (best is None or value > best[0]):
            best = (value, result, full_point)
    if best is None:
        raise RuntimeError("no restart returned a finite feasible point")
    value, result, point = best
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    first_target = conditional_product_target(
        mu, beta, point[layout.crossing1]
    )
    second_target = conditional_product_target(
        beta, zeta, point[layout.crossing2]
    )
    first_projected = projected_divergence(
        mu, beta, point[layout.crossing1], args.projection
    )
    second_projected = projected_divergence(
        beta, zeta, point[layout.crossing2], args.projection
    )
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "projection": args.projection,
        "maximum_upper_bound_found": value,
        "message_nonzero_fraction": float(mu[1:].sum()),
        "middle_nonzero_fraction": float(beta[1:].sum()),
        "output_nonzero_fraction": float(zeta[1:].sum()),
        "first_crossing_fraction": float(point[layout.crossing1]),
        "second_crossing_fraction": float(point[layout.crossing2]),
        "first_target_divergence": relative_entropy(beta, first_target),
        "second_target_divergence": relative_entropy(zeta, second_target),
        "first_projected_divergence": first_projected,
        "second_projected_divergence": second_projected,
        "mu": mu.tolist(),
        "beta": beta.tolist(),
        "zeta": zeta.tolist(),
        "first_conditional_product_target": first_target.tolist(),
        "second_conditional_product_target": second_target.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "status": "numerical composition-only upper-bound search",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--projection",
        choices=("target", "cancellation", "two-level", "cell-target"),
        default="two-level",
    )
    args = parser.parse_args()
    if args.prime < 3:
        parser.error("prime must be an odd prime")
    if args.repetition != 3:
        parser.error("the current ladder study is specialized to repetition three")
    return args


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
