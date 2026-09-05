#!/usr/bin/env python3
"""Estimate the full 48-dimensional correlation Hessian at the F_5 product.

The coordinates are 16 zero-marginal directions for each of the message,
middle, and output pair compositions.  The Hessian is obtained by central
differences of envelope-theorem gradients.  This remains a floating-point
diagnostic, not a proof.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.linalg import null_space

import diagnose_joint_information_projection as joint
import diagnose_pair_replica_payment as pair
import optimize_bulk_exponent as bulk


def tangent_basis(marginal: np.ndarray) -> np.ndarray:
    """Return 16 independently scaled zero-marginal 5-by-5 directions."""
    root = np.sqrt(marginal)
    functions = null_space(root.reshape(1, -1)) / root[:, None]
    weighted = marginal[:, None] * functions
    directions = []
    product = np.outer(marginal, marginal)
    for left in range(pair.PRIME - 1):
        for right in range(pair.PRIME - 1):
            direction = np.outer(weighted[:, left], weighted[:, right])
            direction /= np.max(np.abs(direction) / product)
            directions.append(direction.ravel())
    return np.stack(directions)


def raw_stage_dual(increment: np.ndarray, state: np.ndarray) -> dict[str, object]:
    equality, target = pair.pair_constraints(increment, state)
    return joint.joint_projection_dual(
        np.outer(increment, state), equality, target
    )


def feature_moments(increment: np.ndarray, state: np.ndarray) -> np.ndarray:
    equality, target = pair.pair_constraints(increment, state)
    _, moments = joint.constraint_feature_basis(equality, target)
    return moments


def moment_derivatives(
    increment: np.ndarray,
    state: np.ndarray,
    increment_directions: np.ndarray,
    state_directions: np.ndarray,
) -> np.ndarray:
    baseline = feature_moments(increment, state)
    derivatives = []
    for increment_direction, state_direction in zip(
        increment_directions, state_directions, strict=True
    ):
        shifted = feature_moments(
            increment + increment_direction, state + state_direction
        )
        derivatives.append(shifted - baseline)
    return np.stack(derivatives)


def exponent_gradient(
    compositions: list[np.ndarray],
    coordinate_directions: list[np.ndarray],
    stage_moment_derivatives: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    first = raw_stage_dual(compositions[0], compositions[1])
    second = raw_stage_dual(compositions[1], compositions[2])
    potentials = [np.asarray(first["potentials"]), np.asarray(second["potentials"])]
    gradient = np.zeros(len(coordinate_directions))
    log_message = np.log(compositions[0]) + 1.0
    for coordinate, direction in enumerate(coordinate_directions):
        gradient[coordinate] = (
            -float(np.dot(direction[0], log_message)) / 3.0
            - float(np.dot(potentials[0], stage_moment_derivatives[0][coordinate]))
            - float(np.dot(potentials[1], stage_moment_derivatives[1][coordinate]))
        )
    return gradient, [first, second]


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=pair.PRIME,
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
    marginals = [
        np.asarray(optimized["mu"]),
        np.asarray(optimized["beta"]),
        np.asarray(optimized["zeta"]),
    ]
    compositions = [np.outer(value, value).ravel() for value in marginals]
    bases = [tangent_basis(value) for value in marginals]

    coordinate_directions: list[np.ndarray] = []
    for group in range(3):
        for basis_direction in bases[group]:
            direction = np.zeros((3, pair.PAIR_SIZE))
            direction[group] = basis_direction
            coordinate_directions.append(direction)

    zeros = np.zeros((len(coordinate_directions), pair.PAIR_SIZE))
    first_increment = np.stack([value[0] for value in coordinate_directions])
    first_state = np.stack([value[1] for value in coordinate_directions])
    second_increment = first_state
    second_state = np.stack([value[2] for value in coordinate_directions])
    stage_moment_derivatives = [
        moment_derivatives(
            compositions[0], compositions[1], first_increment, first_state
        ),
        moment_derivatives(
            compositions[1], compositions[2], second_increment, second_state
        ),
    ]

    baseline_gradient, baseline_duals = exponent_gradient(
        compositions, coordinate_directions, stage_moment_derivatives
    )
    dimension = len(coordinate_directions)
    hessian = np.zeros((dimension, dimension))
    maximum_residual = max(
        float(value["equality_residual"]) for value in baseline_duals
    )
    for column, direction in enumerate(coordinate_directions):
        shifted_gradients = []
        for sign in (-1.0, 1.0):
            shifted = [
                composition + sign * args.step * component_direction
                for composition, component_direction in zip(
                    compositions, direction, strict=True
                )
            ]
            if min(float(np.min(value)) for value in shifted) <= 0.0:
                raise RuntimeError("step left the positive simplex")
            gradient, duals = exponent_gradient(
                shifted, coordinate_directions, stage_moment_derivatives
            )
            shifted_gradients.append(gradient)
            maximum_residual = max(
                maximum_residual,
                *(float(value["equality_residual"]) for value in duals),
            )
        hessian[:, column] = (
            shifted_gradients[1] - shifted_gradients[0]
        ) / (2.0 * args.step)

    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    hessian = (hessian + hessian.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    leading = eigenvectors[:, -1]
    group_norms = [
        float(np.linalg.norm(leading[16 * group : 16 * (group + 1)]))
        for group in range(3)
    ]
    return {
        "delta": args.delta,
        "step": args.step,
        "dimension": dimension,
        "maximum_baseline_gradient_magnitude": float(
            np.max(np.abs(baseline_gradient))
        ),
        "maximum_dual_equality_residual": maximum_residual,
        "maximum_hessian_asymmetry_before_symmetrizing": asymmetry,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "largest_eigenvalues": eigenvalues[-10:].tolist(),
        "positive_eigenvalues_above_1e-6": int(np.sum(eigenvalues > 1e-6)),
        "leading_eigenvector_group_norms": {
            "message": group_norms[0],
            "middle": group_norms[1],
            "output": group_norms[2],
        },
        "status": "full local floating-point Hessian; not a global certificate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.261)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
