#!/usr/bin/env python3
"""Compare character-covariance information bounds at the F_5 maximizer."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

import optimize_bulk_exponent as bulk
import search_internal_information_gap as information


def fourier(distribution: np.ndarray) -> np.ndarray:
    prime = len(distribution)
    root = np.exp(2j * np.pi / prime)
    return np.array(
        [
            sum(
                distribution[value] * root ** (frequency * value)
                for value in range(prime)
            )
            for frequency in range(prime)
        ]
    )


def covariance_lower(
    increment: np.ndarray, state: np.ndarray
) -> tuple[float, list[float]]:
    increment_hat = fourier(increment)
    state_hat = fourier(state)
    values = [
        0.5
        * abs(state_hat[frequency]) ** 2
        * abs(1.0 - increment_hat[frequency]) ** 2
        for frequency in range(1, len(increment))
    ]
    return max(values), values


def conditional_mutual_information_parts(
    transition: np.ndarray,
) -> dict[str, float]:
    prime = transition.shape[0]
    joint_ds = np.zeros((prime, prime))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            joint_ds[difference, source] += transition[source, target]
    joint_uv = np.zeros((2, 2))
    joint_dv = np.zeros((prime, 2))
    joint_us = np.zeros((2, prime))
    for difference in range(prime):
        for source in range(prime):
            value = joint_ds[difference, source]
            u = int(difference != 0)
            v = int(source != 0)
            joint_uv[u, v] += value
            joint_dv[difference, v] += value
            joint_us[u, source] += value

    def conditional_entropy(joint: np.ndarray, conditioning_axis: int) -> float:
        marginal = joint.sum(axis=1 - conditioning_axis)
        return bulk.entropy(joint.ravel()) - bulk.entropy(marginal)

    # H(V|U)-H(V|D), with rows U/D and columns V.
    d_v_given_u = conditional_entropy(joint_uv, 0) - conditional_entropy(
        joint_dv, 0
    )
    # H(U|V)-H(U|S), with rows U and columns V/S.
    u_given_s = conditional_entropy(joint_uv, 1) - conditional_entropy(
        joint_us, 1
    )
    full, coarse, _ = information.transition_information(transition)
    within_cells = full - coarse - d_v_given_u - u_given_s
    return {
        "label_to_support": d_v_given_u,
        "support_to_label": u_given_s,
        "within_support_cells": within_cells,
        "sum": d_v_given_u + u_given_s + within_cells,
    }


def conditional_character_lower(
    transition: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
) -> tuple[float, list[float]]:
    prime = len(increment)
    joint = np.zeros((2, 2))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            joint[int(difference != 0), int(source != 0)] += transition[
                source, target
            ]
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    root = np.exp(2j * np.pi / prime)
    values = []
    for frequency in range(1, prime):
        increment_label = sum(
            increment[value] * root ** (frequency * value)
            for value in range(1, prime)
        ) / increment_support
        state_label = sum(
            state[value] * root ** (frequency * value)
            for value in range(1, prime)
        ) / state_support
        comparison = (
            joint[0, 0]
            + joint[0, 1] * state_label
            + joint[1, 0] * increment_label
            + joint[1, 1] * increment_label * state_label
        )
        stationary = fourier(state)[frequency]
        values.append(0.5 * abs(stationary - comparison) ** 2)
    return max(values), values


def nonzero_cell_character_lower(
    transition: np.ndarray,
) -> tuple[float, list[float], float]:
    prime = transition.shape[0]
    cell = np.zeros((prime - 1, prime - 1))
    for source in range(1, prime):
        for target in range(prime):
            difference = (target - source) % prime
            if difference != 0:
                cell[difference - 1, source - 1] += transition[source, target]
    mass = float(cell.sum())
    conditional = cell / mass
    increment = conditional.sum(axis=1)
    state = conditional.sum(axis=0)
    root = np.exp(2j * np.pi / prime)
    values = []
    for frequency in range(1, prime):
        joint_moment = sum(
            conditional[difference - 1, source - 1]
            * root ** (frequency * (difference + source))
            for difference in range(1, prime)
            for source in range(1, prime)
        )
        increment_moment = sum(
            increment[difference - 1] * root ** (frequency * difference)
            for difference in range(1, prime)
        )
        state_moment = sum(
            state[source - 1] * root ** (frequency * source)
            for source in range(1, prime)
        )
        values.append(
            0.5 * mass * abs(joint_moment - increment_moment * state_moment) ** 2
        )
    return max(values), values, mass


def nonzero_cell_convolution_lower(transition: np.ndarray) -> float:
    prime = transition.shape[0]
    joint = np.zeros((prime, prime))
    for source in range(1, prime):
        for target in range(prime):
            difference = (target - source) % prime
            if difference != 0:
                joint[difference, source] += transition[source, target]
    mass = float(joint.sum())
    conditional = joint / mass
    increment = conditional.sum(axis=1)
    state = conditional.sum(axis=0)
    actual_sum = np.zeros(prime)
    independent_sum = np.zeros(prime)
    for difference in range(prime):
        for source in range(prime):
            actual_sum[(difference + source) % prime] += conditional[
                difference, source
            ]
            independent_sum[(difference + source) % prime] += (
                increment[difference] * state[source]
            )
    positive = actual_sum > 0.0
    if np.any(independent_sum[positive] <= 0.0):
        return float("inf")
    divergence = float(
        np.dot(
            actual_sum[positive],
            np.log(actual_sum[positive] / independent_sum[positive]),
        )
    )
    return mass * divergence


def coarse_product_sum_lower(
    transition: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    prime = len(increment)
    joint_support = np.zeros((2, 2))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            joint_support[int(difference != 0), int(source != 0)] += transition[
                source, target
            ]
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    increment_conditionals = (
        np.array([1.0, *(0.0 for _ in range(prime - 1))]),
        np.array([0.0, *(increment[1:] / increment_support)]),
    )
    state_conditionals = (
        np.array([1.0, *(0.0 for _ in range(prime - 1))]),
        np.array([0.0, *(state[1:] / state_support)]),
    )
    comparison_sum = np.zeros(prime)
    for u in range(2):
        for v in range(2):
            for difference in range(prime):
                for source in range(prime):
                    comparison_sum[(difference + source) % prime] += (
                        joint_support[u, v]
                        * increment_conditionals[u][difference]
                        * state_conditionals[v][source]
                    )
    positive = state > 0.0
    if np.any(comparison_sum[positive] <= 0.0):
        return float("inf")
    return float(
        np.dot(
            state[positive],
            np.log(state[positive] / comparison_sum[positive]),
        )
    )


def support_cell_sum_lower(
    transition: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    prime = len(increment)
    actual = np.zeros((2, 2, prime))
    joint_support = np.zeros((2, 2))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            u = int(difference != 0)
            v = int(source != 0)
            actual[u, v, target] += transition[source, target]
            joint_support[u, v] += transition[source, target]
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    increment_conditionals = (
        np.array([1.0, *(0.0 for _ in range(prime - 1))]),
        np.array([0.0, *(increment[1:] / increment_support)]),
    )
    state_conditionals = (
        np.array([1.0, *(0.0 for _ in range(prime - 1))]),
        np.array([0.0, *(state[1:] / state_support)]),
    )
    comparison = np.zeros_like(actual)
    for u in range(2):
        for v in range(2):
            for difference in range(prime):
                for source in range(prime):
                    comparison[u, v, (difference + source) % prime] += (
                        joint_support[u, v]
                        * increment_conditionals[u][difference]
                        * state_conditionals[v][source]
                    )
    positive = actual > 0.0
    if np.any(comparison[positive] <= 0.0):
        return float("inf")
    return float(
        np.dot(
            actual[positive],
            np.log(actual[positive] / comparison[positive]),
        )
    )


def independent_support_sum_projection(
    increment: np.ndarray, state: np.ndarray
) -> np.ndarray:
    prime = len(increment)
    result = np.zeros((2, 2, prime))
    for difference in range(prime):
        for source in range(prime):
            result[
                int(difference != 0),
                int(source != 0),
                (difference + source) % prime,
            ] += increment[difference] * state[source]
    return result


def direct_support_sum_lower(
    transition: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
) -> float:
    prime = len(increment)
    actual = np.zeros((2, 2, prime))
    for source in range(prime):
        for target in range(prime):
            difference = (target - source) % prime
            actual[
                int(difference != 0),
                int(source != 0),
                target,
            ] += transition[source, target]
    comparison = independent_support_sum_projection(increment, state)
    positive = actual > 0.0
    if np.any(comparison[positive] <= 0.0):
        return float("inf")
    return float(
        np.dot(
            actual[positive],
            np.log(actual[positive] / comparison[positive]),
        )
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=5,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-11,
            sign_symmetric=False,
            support_lower=None,
            support_upper=None,
        )
    )
    mu = np.asarray(optimized["mu"])
    beta = np.asarray(optimized["beta"])
    zeta = np.asarray(optimized["zeta"])
    tau1 = np.asarray(optimized["tau1"])
    tau2 = np.asarray(optimized["tau2"])
    first_full, first_support, _ = information.transition_information(tau1)
    second_full, second_support, _ = information.transition_information(tau2)
    first_fourier, first_frequencies = covariance_lower(mu, beta)
    second_fourier, second_frequencies = covariance_lower(beta, zeta)
    first_conditional, first_conditional_frequencies = conditional_character_lower(
        tau1, mu, beta
    )
    second_conditional, second_conditional_frequencies = conditional_character_lower(
        tau2, beta, zeta
    )
    first_cell, first_cell_frequencies, first_cell_mass = (
        nonzero_cell_character_lower(tau1)
    )
    second_cell, second_cell_frequencies, second_cell_mass = (
        nonzero_cell_character_lower(tau2)
    )
    first_convolution = nonzero_cell_convolution_lower(tau1)
    second_convolution = nonzero_cell_convolution_lower(tau2)
    first_coarse_product = coarse_product_sum_lower(tau1, mu, beta)
    second_coarse_product = coarse_product_sum_lower(tau2, beta, zeta)
    first_cell_sum = support_cell_sum_lower(tau1, mu, beta)
    second_cell_sum = support_cell_sum_lower(tau2, beta, zeta)
    first_direct = direct_support_sum_lower(tau1, mu, beta)
    second_direct = direct_support_sum_lower(tau2, beta, zeta)
    message_term = bulk.entropy(mu) / 3.0
    hybrid_upper = message_term - max(first_support, first_fourier) - max(
        second_support, second_fourier
    )
    return {
        "exact_exponent": optimized["exponent"],
        "message_entropy_over_three": message_term,
        "first_exact_information": first_full,
        "first_support_information": first_support,
        "first_fourier_pinsker_lower": first_fourier,
        "first_frequency_bounds": first_frequencies,
        "first_conditional_character_lower": first_conditional,
        "first_conditional_character_frequencies": first_conditional_frequencies,
        "first_nonzero_cell_character_lower": first_cell,
        "first_nonzero_cell_character_frequencies": first_cell_frequencies,
        "first_nonzero_cell_mass": first_cell_mass,
        "first_nonzero_cell_convolution_lower": first_convolution,
        "first_coarse_product_sum_lower": first_coarse_product,
        "first_support_cell_sum_lower": first_cell_sum,
        "first_direct_support_sum_lower": first_direct,
        "first_conditional_parts": conditional_mutual_information_parts(tau1),
        "second_exact_information": second_full,
        "second_support_information": second_support,
        "second_fourier_pinsker_lower": second_fourier,
        "second_frequency_bounds": second_frequencies,
        "second_conditional_character_lower": second_conditional,
        "second_conditional_character_frequencies": second_conditional_frequencies,
        "second_nonzero_cell_character_lower": second_cell,
        "second_nonzero_cell_character_frequencies": second_cell_frequencies,
        "second_nonzero_cell_mass": second_cell_mass,
        "second_nonzero_cell_convolution_lower": second_convolution,
        "second_coarse_product_sum_lower": second_coarse_product,
        "second_support_cell_sum_lower": second_cell_sum,
        "second_direct_support_sum_lower": second_direct,
        "second_conditional_parts": conditional_mutual_information_parts(tau2),
        "support_or_fourier_exponent_upper": hybrid_upper,
        "remaining_gap_to_exact_exponent": hybrid_upper
        - float(optimized["exponent"]),
        "status": "numerical point diagnostic only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
