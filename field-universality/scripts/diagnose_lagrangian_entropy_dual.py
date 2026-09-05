#!/usr/bin/env python3
"""Diagnose a grouped entropy dual for the internal-gap Lagrangian.

For a transition law ``tau``, let ``J`` be the joint law of the indicators
``[increment != 0]`` and ``[source != 0]``.  The grouped entropy term is

    (1 + lambda) H(tau) - lambda H(J)
      = H(J) + (1 + lambda) H(tau | J).

Its entropy conjugate keeps the cancellation between full symbols and support
indicators.  This script tests pointwise dual fidelity.  It is numerical and
does not certify a global maximum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import optimize_bulk_exponent as bulk
import search_internal_information_gap as gap_search


@dataclass(frozen=True)
class GroupedCut:
    log_partition: float
    row_potential: np.ndarray
    column_potential: np.ndarray
    increment_potential: np.ndarray
    value_at_reference: float
    gradient_residual: float
    optimizer_success: bool


def unpack(parameters: np.ndarray, prime: int) -> tuple[np.ndarray, ...]:
    width = prime - 1
    row = np.zeros(prime)
    column = np.zeros(prime)
    increment = np.zeros(prime)
    row[1:] = parameters[:width]
    column[1:] = parameters[width : 2 * width]
    increment[1:] = parameters[2 * width :]
    return row, column, increment


def cell_index(source: int, target: int, prime: int) -> int:
    difference = (target - source) % prime
    return 2 * int(difference != 0) + int(source != 0)


def grouped_pressure_distribution(
    row: np.ndarray,
    column: np.ndarray,
    increment: np.ndarray,
    multiplier: float,
) -> tuple[float, np.ndarray]:
    prime = len(row)
    edge_log_weight = np.empty((prime, prime))
    cell_edges: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            potential = row[source] + column[target] + increment[difference]
            edge_log_weight[source, target] = -potential / (1.0 + multiplier)
            cell_edges[cell_index(source, target, prime)].append((source, target))

    cell_log_sums = np.array(
        [
            logsumexp(
                [edge_log_weight[source, target] for source, target in edges]
            )
            for edges in cell_edges
        ]
    )
    log_partition = float(logsumexp((1.0 + multiplier) * cell_log_sums))
    transition = np.empty((prime, prime))
    for cell, edges in enumerate(cell_edges):
        for source, target in edges:
            transition[source, target] = np.exp(
                multiplier * cell_log_sums[cell]
                + edge_log_weight[source, target]
                - log_partition
            )
    return log_partition, transition


def increment_marginal(transition: np.ndarray) -> np.ndarray:
    prime = transition.shape[0]
    return np.array(
        [
            sum(
                transition[source, (source + difference) % prime]
                for source in range(prime)
            )
            for difference in range(prime)
        ]
    )


def fit_grouped_cut(
    prescribed_increment: np.ndarray,
    prescribed_state: np.ndarray,
    multiplier: float,
) -> GroupedCut:
    prime = len(prescribed_increment)

    def value_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        row, column, increment = unpack(parameters, prime)
        log_partition, transition = grouped_pressure_distribution(
            row, column, increment, multiplier
        )
        observed_row = transition.sum(axis=1)
        observed_column = transition.sum(axis=0)
        observed_increment = increment_marginal(transition)
        value = (
            log_partition
            + float(np.dot(row + column, prescribed_state))
            + float(np.dot(increment, prescribed_increment))
        )
        gradient = np.concatenate(
            (
                prescribed_state[1:] - observed_row[1:],
                prescribed_state[1:] - observed_column[1:],
                prescribed_increment[1:] - observed_increment[1:],
            )
        )
        return value, gradient

    result = minimize(
        value_gradient,
        np.zeros(3 * (prime - 1)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-100.0, 100.0)] * (3 * (prime - 1)),
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 10_000},
    )
    value, gradient = value_gradient(result.x)
    row, column, increment = unpack(result.x, prime)
    log_partition, _ = grouped_pressure_distribution(
        row, column, increment, multiplier
    )
    return GroupedCut(
        log_partition,
        row,
        column,
        increment,
        value,
        float(np.max(np.abs(gradient))),
        bool(result.success),
    )


def support_joint(transition: np.ndarray) -> np.ndarray:
    prime = transition.shape[0]
    result = np.zeros(4)
    for source in range(prime):
        for target in range(prime):
            result[cell_index(source, target, prime)] += transition[source, target]
    return result


def grouped_entropy(transition: np.ndarray, multiplier: float) -> float:
    return (
        (1.0 + multiplier) * bulk.entropy(transition.reshape(-1))
        - multiplier * bulk.entropy(support_joint(transition))
    )


def binary_entropy(value: float) -> float:
    return bulk.entropy(np.array([1.0 - value, value]))


def lagrangian_cut_value(
    point: np.ndarray,
    layout: bulk.Layout,
    repetition: int,
    multiplier: float,
    first: GroupedCut,
    second: GroupedCut,
) -> float:
    message = point[layout.mu]
    middle = point[layout.beta]
    output = point[layout.zeta]
    message_support = float(message[1:].sum())
    middle_support = float(middle[1:].sum())
    output_support = float(output[1:].sum())
    return (
        first.log_partition
        + second.log_partition
        + float(np.dot(first.increment_potential, message))
        + float(
            np.dot(
                first.row_potential
                + first.column_potential
                + second.increment_potential,
                middle,
            )
        )
        + float(
            np.dot(second.row_potential + second.column_potential, output)
        )
        - (1.0 + multiplier)
        * (1.0 - 1.0 / repetition)
        * bulk.entropy(message)
        - 2.0 * (1.0 + multiplier) * bulk.entropy(middle)
        - (1.0 + multiplier) * bulk.entropy(output)
        + multiplier
        * (1.0 - 1.0 / repetition)
        * binary_entropy(message_support)
        + 2.0 * multiplier * binary_entropy(middle_support)
        + multiplier * binary_entropy(output_support)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--gap-multiplier", type=float, default=1.5)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer_args = SimpleNamespace(
        prime=args.prime,
        repetition=args.repetition,
        delta=args.delta,
        eta=args.eta,
        restarts=args.restarts,
        seed=args.seed,
        max_message_weight=None,
        mode="lagrangian",
        maximum_gap=0.0,
        gap_multiplier=args.gap_multiplier,
    )
    optimized = gap_search.search(optimizer_args)
    layout = bulk.Layout(args.prime)
    point = np.zeros(layout.size)
    point[layout.mu] = optimized["mu"]
    point[layout.beta] = optimized["beta"]
    point[layout.zeta] = optimized["zeta"]
    point[layout.tau1] = np.array(optimized["tau1"]).reshape(-1)
    point[layout.tau2] = np.array(optimized["tau2"]).reshape(-1)
    first = fit_grouped_cut(
        point[layout.mu], point[layout.beta], args.gap_multiplier
    )
    second = fit_grouped_cut(
        point[layout.beta], point[layout.zeta], args.gap_multiplier
    )
    first_exact = grouped_entropy(
        point[layout.tau1].reshape((args.prime, args.prime)),
        args.gap_multiplier,
    )
    second_exact = grouped_entropy(
        point[layout.tau2].reshape((args.prime, args.prime)),
        args.gap_multiplier,
    )
    cut_value = lagrangian_cut_value(
        point,
        layout,
        args.repetition,
        args.gap_multiplier,
        first,
        second,
    )
    true_value = float(optimized["lagrangian_at_result"])
    print(
        json.dumps(
            {
                "prime": args.prime,
                "repetition": args.repetition,
                "delta": args.delta,
                "eta": args.eta,
                "gap_multiplier": args.gap_multiplier,
                "true_lagrangian_at_optimizer": true_value,
                "grouped_dual_upper_at_optimizer": cut_value,
                "total_fidelity_gap": cut_value - true_value,
                "first_grouped_entropy": first_exact,
                "first_grouped_cut": first.value_at_reference,
                "first_fidelity_gap": first.value_at_reference - first_exact,
                "second_grouped_entropy": second_exact,
                "second_grouped_cut": second.value_at_reference,
                "second_fidelity_gap": second.value_at_reference - second_exact,
                "first_gradient_residual": first.gradient_residual,
                "second_gradient_residual": second.gradient_residual,
                "first_optimizer_success": first.optimizer_success,
                "second_optimizer_success": second.optimizer_success,
                "status": "numerical grouped-entropy dual diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
