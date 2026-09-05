#!/usr/bin/env python3
"""Test replica symmetry for the F_5 stationary information payment."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

import diagnose_joint_information_projection as joint
import optimize_bulk_exponent as bulk


PRIME = 5
PAIR_SIZE = PRIME * PRIME


def pair_index(left: int, right: int) -> int:
    return PRIME * left + right


def pair_add(left: int, right: int) -> int:
    return pair_index(
        ((left // PRIME) + (right // PRIME)) % PRIME,
        ((left % PRIME) + (right % PRIME)) % PRIME,
    )


def pair_constraints(
    increment: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for difference in range(PAIR_SIZE):
        row = np.zeros((PAIR_SIZE, PAIR_SIZE))
        row[difference, :] = 1.0
        rows.append(row.ravel())
        targets.append(float(increment[difference]))
    for source in range(PAIR_SIZE):
        row = np.zeros((PAIR_SIZE, PAIR_SIZE))
        row[:, source] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[source]))
    for target in range(PAIR_SIZE):
        row = np.zeros((PAIR_SIZE, PAIR_SIZE))
        for difference in range(PAIR_SIZE):
            for source in range(PAIR_SIZE):
                if pair_add(difference, source) == target:
                    row[difference, source] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[target]))
    return bulk.independent_equalities(np.stack(rows), np.asarray(targets))


def pair_stationary_payment(
    increment: np.ndarray, state: np.ndarray
) -> dict[str, object]:
    equality, target = pair_constraints(increment, state)
    reference = np.outer(increment, state)
    dual = joint.joint_projection_dual(reference, equality, target)
    return {
        "payment": dual["value"],
        "equality_residual": dual["equality_residual"],
        "gradient_residual": dual["gradient_residual"],
        "potentials": len(dual["potentials"]),
        "optimizer_success": dual["optimizer_success"],
        "optimizer_message": dual["optimizer_message"],
    }


def transition_information(
    transition: np.ndarray, increment: np.ndarray, state: np.ndarray
) -> float:
    law = np.zeros_like(transition)
    for source in range(PRIME):
        for target in range(PRIME):
            law[(target - source) % PRIME, source] = transition[source, target]
    return joint.relative_entropy(law.ravel(), np.outer(increment, state).ravel())


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=PRIME,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-11,
            sign_symmetric=False,
        )
    )
    message = np.asarray(optimized["mu"])
    middle = np.asarray(optimized["beta"])
    output = np.asarray(optimized["zeta"])
    first_transition = np.asarray(optimized["tau1"])
    second_transition = np.asarray(optimized["tau2"])
    first_single = transition_information(first_transition, message, middle)
    second_single = transition_information(second_transition, middle, output)

    pair_message = np.outer(message, message).ravel()
    pair_middle = np.outer(middle, middle).ravel()
    pair_output = np.outer(output, output).ravel()
    first_pair = pair_stationary_payment(pair_message, pair_middle)
    second_pair = pair_stationary_payment(pair_middle, pair_output)
    pair_exponent = (
        bulk.entropy(pair_message) / 3.0
        - float(first_pair["payment"])
        - float(second_pair["payment"])
    )
    twice_single = 2.0 * optimized["exponent"]
    return {
        "delta": args.delta,
        "eta": args.eta,
        "single_exponent": optimized["exponent"],
        "twice_single_exponent": twice_single,
        "single_stage_payments": [first_single, second_single],
        "pair_product_stage_payments": [first_pair, second_pair],
        "pair_product_exponent_after_joint_optimization": pair_exponent,
        "replica_correlation_advantage": pair_exponent - twice_single,
        "status": "numerical product-composition replica diagnostic only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.261)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
