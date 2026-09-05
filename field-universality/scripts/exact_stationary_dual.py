#!/usr/bin/env python3
"""Fixed entropy-dual cuts for the exact stationary accumulator payment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import diagnose_joint_information_projection as joint
import optimize_bulk_exponent as bulk


PRIME = 5


@dataclass(frozen=True)
class StationaryDualCut:
    moment_constant: float
    moment_coefficients: np.ndarray
    kernel: np.ndarray
    value: float
    equality_residual: float
    gradient_residual: float
    optimizer_success: bool
    potentials: np.ndarray | None = None
    shift: float | None = None


def encode_cut(cut: StationaryDualCut) -> dict[str, object]:
    result = {
        "moment_constant": cut.moment_constant,
        "moment_coefficients": cut.moment_coefficients.tolist(),
        "kernel": cut.kernel.tolist(),
        "value": cut.value,
        "equality_residual": cut.equality_residual,
        "gradient_residual": cut.gradient_residual,
        "optimizer_success": cut.optimizer_success,
    }
    if cut.potentials is not None:
        result["potentials"] = cut.potentials.tolist()
    if cut.shift is not None:
        result["shift"] = cut.shift
    return result


def decode_cut(value: dict[str, object]) -> StationaryDualCut:
    return StationaryDualCut(
        moment_constant=float(value["moment_constant"]),
        moment_coefficients=np.asarray(value["moment_coefficients"], dtype=float),
        kernel=np.asarray(value["kernel"], dtype=float),
        value=float(value["value"]),
        equality_residual=float(value["equality_residual"]),
        gradient_residual=float(value["gradient_residual"]),
        optimizer_success=bool(value["optimizer_success"]),
        potentials=(
            np.asarray(value["potentials"], dtype=float)
            if "potentials" in value
            else None
        ),
        shift=(float(value["shift"]) if "shift" in value else None),
    )


def stationary_constraints(
    increment: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for difference in range(PRIME):
        row = np.zeros((PRIME, PRIME))
        row[difference, :] = 1.0
        rows.append(row.ravel())
        targets.append(float(increment[difference]))
    for source in range(PRIME):
        row = np.zeros((PRIME, PRIME))
        row[:, source] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[source]))
    for target in range(PRIME):
        row = np.zeros((PRIME, PRIME))
        for difference in range(PRIME):
            row[difference, (target - difference) % PRIME] = 1.0
        rows.append(row.ravel())
        targets.append(float(state[target]))
    return bulk.independent_equalities(np.stack(rows), np.asarray(targets))


def feature_moments(
    increment: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    equality, target = stationary_constraints(increment, state)
    return joint.constraint_feature_basis(equality, target)


def moment_affine_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros(PRIME)
    features, constant = feature_moments(zero, zero)
    coefficients = np.empty((len(constant), 2 * PRIME))
    for index in range(2 * PRIME):
        increment = np.zeros(PRIME)
        state = np.zeros(PRIME)
        if index < PRIME:
            increment[index] = 1.0
        else:
            state[index - PRIME] = 1.0
        candidate_features, moments = feature_moments(increment, state)
        if not np.array_equal(candidate_features, features):
            raise RuntimeError("stationary feature basis changed")
        coefficients[:, index] = moments - constant
    return features, constant, coefficients


FEATURES, MOMENT_CONSTANT, MOMENT_COEFFICIENTS = moment_affine_map()


def fit_stationary_dual(
    increment: np.ndarray, state: np.ndarray
) -> StationaryDualCut:
    increment = np.asarray(increment, dtype=float)
    state = np.asarray(state, dtype=float)
    equality, target = stationary_constraints(increment, state)
    dual = joint.joint_projection_dual(
        np.outer(increment, state), equality, target
    )
    potentials = np.asarray(dual["potentials"], dtype=float)
    if not np.array_equal(np.asarray(dual["feature_rows"]), FEATURES):
        raise RuntimeError("stationary dual used an unexpected feature basis")
    log_kernel = (potentials @ FEATURES).reshape((PRIME, PRIME))
    shift = float(np.max(log_kernel))
    return StationaryDualCut(
        moment_constant=float(np.dot(potentials, MOMENT_CONSTANT) - shift),
        moment_coefficients=potentials @ MOMENT_COEFFICIENTS,
        # Raising underflowed kernel entries weakens the dual lower bound and
        # keeps fixed-cut evaluations finite on boundary compositions.
        kernel=np.exp(np.maximum(log_kernel - shift, -700.0)),
        value=float(dual["value"]),
        equality_residual=float(dual["equality_residual"]),
        gradient_residual=float(dual["gradient_residual"]),
        optimizer_success=bool(dual["optimizer_success"]),
        potentials=potentials.copy(),
        shift=shift,
    )


def fixed_cut_payment(
    increment: np.ndarray, state: np.ndarray, cut: StationaryDualCut
) -> float:
    compositions = np.concatenate((increment, state))
    partition = float(increment @ cut.kernel @ state)
    return float(
        cut.moment_constant
        + np.dot(cut.moment_coefficients, compositions)
        - np.log(partition)
    )


def affine_symbol_pushforward(
    values: np.ndarray, multiplier: int, translation: int = 0
) -> np.ndarray:
    """Push a symbol-indexed vector through x |-> multiplier*x+translation."""
    values = np.asarray(values)
    multiplier %= PRIME
    translation %= PRIME
    if multiplier == 0:
        raise ValueError("the affine multiplier must be nonzero")
    result = np.empty_like(values)
    for source in range(PRIME):
        result[(multiplier * source + translation) % PRIME] = values[source]
    return result


def transform_stationary_cut(
    cut: StationaryDualCut,
    multiplier: int,
    state_translation: int = 0,
    reverse: bool = False,
) -> StationaryDualCut:
    """Pull a valid cut back through an affine stationary-edge symmetry.

    The underlying edge map sends the state s to a*s+b.  It sends the
    increment d to a*d, or to -a*d after source and target are exchanged.
    """
    multiplier %= PRIME
    state_translation %= PRIME
    if multiplier == 0:
        raise ValueError("the affine multiplier must be nonzero")
    increment_multiplier = (-multiplier if reverse else multiplier) % PRIME
    increment_indices = (
        increment_multiplier * np.arange(PRIME, dtype=int)
    ) % PRIME
    state_indices = (
        multiplier * np.arange(PRIME, dtype=int) + state_translation
    ) % PRIME
    coefficients = np.concatenate(
        (
            cut.moment_coefficients[:PRIME][increment_indices],
            cut.moment_coefficients[PRIME:][state_indices],
        )
    )
    return StationaryDualCut(
        moment_constant=cut.moment_constant,
        moment_coefficients=coefficients,
        kernel=cut.kernel[np.ix_(increment_indices, state_indices)],
        value=cut.value,
        equality_residual=cut.equality_residual,
        gradient_residual=cut.gradient_residual,
        optimizer_success=cut.optimizer_success,
        # The transformed inequality is replayed from the base cut and the
        # group action.  Its coordinates need not use the stored gauge basis.
        potentials=None,
        shift=None,
    )


def stationary_cut_orbit(
    cut: StationaryDualCut, include_state_translations: bool = True
) -> tuple[StationaryDualCut, ...]:
    """Return the 8- or 40-element universal symmetry orbit of one cut."""
    translations = range(PRIME) if include_state_translations else (0,)
    return tuple(
        transform_stationary_cut(cut, multiplier, translation, reverse)
        for multiplier in range(1, PRIME)
        for translation in translations
        for reverse in (False, True)
    )
