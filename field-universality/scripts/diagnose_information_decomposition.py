#!/usr/bin/env python3
"""Decompose full bulk mutual information into support and sign-orbit layers."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

import optimize_bulk_exponent as bulk


def mutual_information(joint: np.ndarray) -> float:
    joint = np.maximum(joint, 0.0)
    total = joint.sum()
    if total <= 0.0:
        return 0.0
    joint = joint / total
    rows = joint.sum(axis=1)
    columns = joint.sum(axis=0)
    value = 0.0
    for row in range(joint.shape[0]):
        for column in range(joint.shape[1]):
            probability = joint[row, column]
            if probability > 0.0:
                value += probability * np.log(
                    probability / (rows[row] * columns[column])
                )
    return float(value)


def increment_source_joint(transition: np.ndarray) -> np.ndarray:
    prime = transition.shape[0]
    joint = np.zeros_like(transition)
    for source in range(prime):
        for target in range(prime):
            increment = (target - source) % prime
            joint[increment, source] += transition[source, target]
    return joint


def support_pushforward(joint: np.ndarray) -> np.ndarray:
    pushed = np.zeros((2, 2))
    for increment in range(joint.shape[0]):
        for source in range(joint.shape[1]):
            pushed[int(increment != 0), int(source != 0)] += joint[
                increment, source
            ]
    return pushed


def sign_orbit_map(prime: int) -> np.ndarray:
    labels = np.full(prime, -1, dtype=int)
    orbit = 0
    for value in range(prime):
        if labels[value] >= 0:
            continue
        opposite = (-value) % prime
        labels[value] = orbit
        labels[opposite] = orbit
        orbit += 1
    return labels


def orbit_pushforward(joint: np.ndarray) -> np.ndarray:
    labels = sign_orbit_map(joint.shape[0])
    orbit_count = int(labels.max()) + 1
    pushed = np.zeros((orbit_count, orbit_count))
    for increment in range(joint.shape[0]):
        for source in range(joint.shape[1]):
            pushed[labels[increment], labels[source]] += joint[increment, source]
    return pushed


def stage_decomposition(transition: np.ndarray) -> dict[str, float]:
    joint = increment_source_joint(transition)
    full = mutual_information(joint)
    support_joint = support_pushforward(joint)
    support = mutual_information(support_joint)
    orbit = mutual_information(orbit_pushforward(joint))
    increment_source_support = np.zeros((joint.shape[0], 2))
    increment_support_source = np.zeros((2, joint.shape[1]))
    for increment in range(joint.shape[0]):
        for source in range(joint.shape[1]):
            increment_source_support[increment, int(source != 0)] += joint[
                increment, source
            ]
            increment_support_source[int(increment != 0), source] += joint[
                increment, source
            ]
    one_sided = (
        mutual_information(increment_source_support)
        + mutual_information(increment_support_source)
        - support
    )
    product_target_bound = conditional_product_target_bound(joint)
    return {
        "support_mutual_information": support,
        "one_sided_symbol_lower_bound": one_sided,
        "within_block_residual": full - one_sided,
        "magnitude_increment": orbit - support,
        "sign_increment": full - orbit,
        "sign_orbit_mutual_information": orbit,
        "full_mutual_information": full,
        **product_target_bound,
    }


def relative_entropy(first: np.ndarray, second: np.ndarray) -> float:
    value = 0.0
    for probability, reference in zip(first, second, strict=True):
        if probability <= 0.0:
            continue
        if reference <= 0.0:
            return float("inf")
        value += probability * np.log(probability / reference)
    return float(value)


def conditional_product_target_bound(
    joint: np.ndarray,
) -> dict[str, float]:
    """Push the support-preserving conditional product through addition."""
    prime = joint.shape[0]
    increment = joint.sum(axis=1)
    state = joint.sum(axis=0)
    support_joint = support_pushforward(joint)
    increment_support = support_joint.sum(axis=1)
    state_support = support_joint.sum(axis=0)
    reference_joint = np.zeros_like(joint)
    for difference in range(prime):
        for source in range(prime):
            difference_support = int(difference != 0)
            source_support = int(source != 0)
            if (
                increment_support[difference_support] > 0.0
                and state_support[source_support] > 0.0
            ):
                reference_joint[difference, source] = (
                    support_joint[difference_support, source_support]
                    * increment[difference]
                    / increment_support[difference_support]
                    * state[source]
                    / state_support[source_support]
                )
    target = np.zeros(prime)
    for difference in range(prime):
        for source in range(prime):
            target[(source + difference) % prime] += reference_joint[
                difference, source
            ]
    full_information = mutual_information(joint)
    support_information = mutual_information(support_joint)
    excess = full_information - support_information
    target_divergence = relative_entropy(state, target)
    target_variation = 0.5 * float(np.abs(state - target).sum())
    return {
        "internal_information_excess": excess,
        "conditional_product_target_divergence": target_divergence,
        "conditional_product_target_variation": target_variation,
        "conditional_product_data_processing_slack": (
            excess - target_divergence
        ),
    }


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    result = bulk.optimize(
        SimpleNamespace(
            prime=args.prime,
            repetition=args.repetition,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-12,
            sign_symmetric=args.sign_symmetric,
        )
    )
    mu = np.array(result["mu"])
    tau1 = np.array(result["tau1"])
    tau2 = np.array(result["tau2"])
    first = stage_decomposition(tau1)
    second = stage_decomposition(tau2)
    total_support = first["support_mutual_information"] + second[
        "support_mutual_information"
    ]
    total_magnitude = first["magnitude_increment"] + second["magnitude_increment"]
    total_sign = first["sign_increment"] + second["sign_increment"]
    message_entropy = bulk.entropy(mu)
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "sign_symmetric": args.sign_symmetric,
        "maximum_exponent_found": result["exponent"],
        "message_nonzero_fraction": result["message_nonzero_fraction"],
        "output_nonzero_fraction": result["output_nonzero_fraction"],
        "message_entropy_over_repetition": message_entropy / args.repetition,
        "first_stage": first,
        "second_stage": second,
        "total_support_mutual_information": total_support,
        "total_magnitude_increment": total_magnitude,
        "total_sign_increment": total_sign,
        "reconstructed_exponent": (
            message_entropy / args.repetition
            - total_support
            - total_magnitude
            - total_sign
        ),
        "status": "numerical optimizer decomposition only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sign-symmetric", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
