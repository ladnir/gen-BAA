#!/usr/bin/env python3
"""Entropy dual for the support-sum information projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class ProjectionDual:
    target_potential: np.ndarray
    increment_support_potential: float
    state_support_potential: float
    value: float
    gradient_residual: float
    optimizer_success: bool


def scale_symbol_vector(values: np.ndarray, multiplier: int) -> np.ndarray:
    prime = len(values)
    inverse = pow(multiplier, -1, prime)
    return np.array(
        [values[(inverse * value) % prime] for value in range(prime)]
    )


def scale_projection_dual(
    cut: ProjectionDual, multiplier: int
) -> ProjectionDual:
    return ProjectionDual(
        scale_symbol_vector(cut.target_potential, multiplier),
        cut.increment_support_potential,
        cut.state_support_potential,
        cut.value,
        cut.gradient_residual,
        cut.optimizer_success,
    )


def unpack(parameters: np.ndarray, prime: int) -> tuple[np.ndarray, float, float]:
    target = np.zeros(prime)
    target[1:] = parameters[: prime - 1]
    return target, float(parameters[-2]), float(parameters[-1])


def pressure(
    increment: np.ndarray,
    state: np.ndarray,
    target_potential: np.ndarray,
    increment_support_potential: float,
    state_support_potential: float,
) -> tuple[float, np.ndarray]:
    prime = len(increment)
    log_weights = np.full((prime, prime), -np.inf)
    for difference in range(prime):
        for source in range(prime):
            if increment[difference] <= 0.0 or state[source] <= 0.0:
                continue
            log_weights[difference, source] = (
                np.log(increment[difference])
                + np.log(state[source])
                + target_potential[(difference + source) % prime]
                + increment_support_potential * int(difference != 0)
                + state_support_potential * int(source != 0)
            )
    log_partition = float(logsumexp(log_weights))
    tilted = np.exp(log_weights - log_partition)
    return log_partition, tilted


def dual_value(
    increment: np.ndarray,
    state: np.ndarray,
    target_state: np.ndarray,
    target_potential: np.ndarray,
    increment_support_potential: float,
    state_support_potential: float,
) -> float:
    log_partition, _ = pressure(
        increment,
        state,
        target_potential,
        increment_support_potential,
        state_support_potential,
    )
    return (
        float(np.dot(target_potential, target_state))
        + increment_support_potential * float(increment[1:].sum())
        + state_support_potential * float(state[1:].sum())
        - log_partition
    )


def fit_projection_dual(
    increment: np.ndarray,
    state: np.ndarray,
    target_state: np.ndarray | None = None,
    bound: float = 100.0,
) -> ProjectionDual:
    prime = len(increment)
    target_state = state if target_state is None else target_state

    def value_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        target, increment_support, state_support = unpack(parameters, prime)
        log_partition, tilted = pressure(
            increment,
            state,
            target,
            increment_support,
            state_support,
        )
        observed_target = np.zeros(prime)
        for difference in range(prime):
            for source in range(prime):
                observed_target[(difference + source) % prime] += tilted[
                    difference, source
                ]
        observed_increment_support = float(tilted[1:, :].sum())
        observed_state_support = float(tilted[:, 1:].sum())
        dual = (
            float(np.dot(target, target_state))
            + increment_support * float(increment[1:].sum())
            + state_support * float(state[1:].sum())
            - log_partition
        )
        gradient = np.concatenate(
            (
                target_state[1:] - observed_target[1:],
                [
                    float(increment[1:].sum()) - observed_increment_support,
                    float(state[1:].sum()) - observed_state_support,
                ],
            )
        )
        return -dual, -gradient

    result = minimize(
        value_gradient,
        np.zeros(prime + 1),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-bound, bound)] * (prime + 1),
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    negative_value, negative_gradient = value_gradient(result.x)
    target, increment_support, state_support = unpack(result.x, prime)
    return ProjectionDual(
        target,
        increment_support,
        state_support,
        -float(negative_value),
        float(np.max(np.abs(negative_gradient))),
        bool(result.success),
    )
