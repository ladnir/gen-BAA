#!/usr/bin/env python3
"""Diagnose affine entropy-dual bounds for the general bulk exponent.

For arbitrary potentials ``u``, ``v``, and ``w``, the entropy conjugate gives

    H(tau) <= log Z + <u+v, beta> + <w, alpha>,

where ``Z=sum_{s,t} exp(-u_s-v_t-w_(t-s))``.  This script optimizes the
potentials at a numerical bulk maximizer and reports the resulting global
outer bound.  It is a design tool, not a rigorous interval certificate.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import optimize_bulk_exponent as bulk


def unpack_potentials(point: np.ndarray, prime: int) -> tuple[np.ndarray, ...]:
    width = prime - 1
    u = np.zeros(prime)
    v = np.zeros(prime)
    w = np.zeros(prime)
    u[1:] = point[:width]
    v[1:] = point[width : 2 * width]
    w[1:] = point[2 * width :]
    return u, v, w


def potential_matrix(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> np.ndarray:
    prime = len(u)
    return np.array(
        [
            [u[source] + v[target] + w[(target - source) % prime] for target in range(prime)]
            for source in range(prime)
        ]
    )


def optimize_dual(
    increment: np.ndarray, state: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    prime = len(increment)

    def value_gradient(point: np.ndarray) -> tuple[float, np.ndarray]:
        u, v, w = unpack_potentials(point, prime)
        potentials = potential_matrix(u, v, w)
        log_partition = float(logsumexp(-potentials))
        transition = np.exp(-potentials - log_partition)
        row = transition.sum(axis=1)
        column = transition.sum(axis=0)
        observed_increment = np.array(
            [
                sum(
                    transition[source, (source + difference) % prime]
                    for source in range(prime)
                )
                for difference in range(prime)
            ]
        )
        value = (
            log_partition
            + float(np.dot(u + v, state))
            + float(np.dot(w, increment))
        )
        gradient = np.concatenate(
            (
                state[1:] - row[1:],
                state[1:] - column[1:],
                increment[1:] - observed_increment[1:],
            )
        )
        return value, gradient

    result = minimize(
        lambda point: value_gradient(point),
        np.zeros(3 * (prime - 1)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-100.0, 100.0)] * (3 * (prime - 1)),
        options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10_000},
    )
    value, gradient = value_gradient(result.x)
    u, v, w = unpack_potentials(result.x, prime)
    return value, u, v, w, float(np.max(np.abs(gradient)))


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def objective_at_vertex(
    coefficients: np.ndarray, entropy_coefficient: float, vertex: np.ndarray
) -> float:
    return float(np.dot(coefficients, vertex)) - entropy_coefficient * entropy(
        vertex
    )


def message_vertices(prime: int, eta: float) -> list[np.ndarray]:
    vertices = []
    for symbol in range(1, prime):
        unit = np.zeros(prime)
        unit[symbol] = 1.0
        vertices.append(unit)
        cutoff = np.zeros(prime)
        cutoff[0] = 1.0 - eta
        cutoff[symbol] = eta
        vertices.append(cutoff)
    return vertices


def simplex_vertices(prime: int) -> list[np.ndarray]:
    return [np.eye(prime)[symbol] for symbol in range(prime)]


def output_vertices(prime: int, delta: float) -> list[np.ndarray]:
    zero = np.zeros(prime)
    zero[0] = 1.0
    vertices = [zero]
    for symbol in range(1, prime):
        boundary = np.zeros(prime)
        boundary[0] = 1.0 - delta
        boundary[symbol] = delta
        vertices.append(boundary)
    return vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.196)
    parser.add_argument("--eta", type=float, default=0.0001)
    parser.add_argument("--restarts", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer_args = SimpleNamespace(
        prime=args.prime,
        repetition=args.repetition,
        delta=args.delta,
        eta=args.eta,
        restarts=args.restarts,
        seed=1,
        max_iterations=3000,
        tolerance=1e-11,
    )
    point = bulk.optimize(optimizer_args)
    mu = np.array(point["mu"])
    beta = np.array(point["beta"])
    zeta = np.array(point["zeta"])
    first_value, u1, v1, w1, first_residual = optimize_dual(mu, beta)
    second_value, u2, v2, w2, second_residual = optimize_dual(beta, zeta)

    message_coefficients = w1
    middle_coefficients = u1 + v1 + w2
    output_coefficients = u2 + v2
    message_maximum = max(
        objective_at_vertex(
            message_coefficients, 1.0 - 1.0 / args.repetition, vertex
        )
        for vertex in message_vertices(args.prime, args.eta)
    )
    middle_maximum = max(
        objective_at_vertex(middle_coefficients, 2.0, vertex)
        for vertex in simplex_vertices(args.prime)
    )
    output_maximum = max(
        objective_at_vertex(output_coefficients, 1.0, vertex)
        for vertex in output_vertices(args.prime, args.delta)
    )
    upper_bound = (
        first_value
        + second_value
        - float(np.dot(w1, mu))
        - float(np.dot(u1 + v1, beta))
        - float(np.dot(w2, beta))
        - float(np.dot(u2 + v2, zeta))
    )
    # optimize_dual returns the value at its reference marginals.  Recover the
    # two log-partition constants before adding the independent outer maxima.
    upper_bound += message_maximum + middle_maximum + output_maximum

    print(
        json.dumps(
            {
                "prime": args.prime,
                "repetition": args.repetition,
                "delta": args.delta,
                "eta": args.eta,
                "numerical_bulk_exponent": point["exponent"],
                "first_dual_residual": first_residual,
                "second_dual_residual": second_residual,
                "global_single_tangent_upper_bound": upper_bound,
                "status": "diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
