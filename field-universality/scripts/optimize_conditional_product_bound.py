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
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize
from scipy.special import logsumexp


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


def conditional_product_source_target(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> np.ndarray:
    """Push the support-preserving conditional product to (source, target)."""
    prime = increment.size
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    joint = support_joint(increment_support, state_support, crossing)
    increment_conditional = np.zeros(prime)
    state_conditional = np.zeros(prime)
    if increment_support < 1.0:
        increment_conditional[0] = increment[0] / (1.0 - increment_support)
    if increment_support > 0.0:
        increment_conditional[1:] = increment[1:] / increment_support
    if state_support < 1.0:
        state_conditional[0] = state[0] / (1.0 - state_support)
    if state_support > 0.0:
        state_conditional[1:] = state[1:] / state_support
    reference = np.zeros((prime, prime))
    for difference in range(prime):
        difference_support = int(difference != 0)
        difference_mass = increment_conditional[difference]
        if difference_mass == 0.0:
            continue
        for source in range(prime):
            source_support = int(source != 0)
            source_mass = state_conditional[source]
            if source_mass == 0.0:
                continue
            reference[source, (source + difference) % prime] += (
                joint[2 * difference_support + source_support]
                * difference_mass
                * source_mass
            )
    return reference


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


def stationary_coupling_dual_potentials(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return source and target potentials for the stationary KL dual.

    The reference already has source marginal ``state``.  Eliminating the
    source potentials leaves a convex dual in one target potential per active
    symbol.  This is much faster near sparse compositions than alternating
    row and column scaling.
    """
    reference = conditional_product_source_target(increment, state, crossing)
    active = state > 0.0
    marginal = state[active]
    restricted = reference[np.ix_(active, active)]
    if marginal.size == 1:
        if restricted[0, 0] <= 0.0:
            raise ValueError("stationary coupling has no feasible support")
        source = np.zeros_like(state)
        target = np.zeros_like(state)
        source[active] = np.log(marginal[0] / restricted[0, 0])
        return source, target
    conditional = restricted / marginal[:, np.newaxis]
    if np.any(conditional.sum(axis=1) <= 0.0):
        raise ValueError("stationary coupling has no feasible support")
    log_conditional = np.full_like(conditional, -np.inf)
    positive = conditional > 0.0
    log_conditional[positive] = np.log(conditional[positive])

    def value_gradient(free_potential: np.ndarray) -> tuple[float, np.ndarray]:
        potential = np.concatenate((free_potential, np.zeros(1)))
        normalizer = logsumexp(
            log_conditional + potential[np.newaxis, :], axis=1
        )
        probabilities = np.exp(
            log_conditional
            + potential[np.newaxis, :]
            - normalizer[:, np.newaxis]
        )
        value = float(np.dot(marginal, normalizer) - np.dot(marginal, potential))
        gradient = marginal @ probabilities - marginal
        return value, gradient[:-1]

    result = minimize(
        value_gradient,
        np.zeros(marginal.size - 1),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 200},
    )
    if not np.isfinite(result.fun):
        raise ValueError("stationary dual optimization failed")
    active_target = np.concatenate((result.x, np.zeros(1)))
    active_source = -logsumexp(
        log_conditional + active_target[np.newaxis, :], axis=1
    )
    source = np.zeros_like(state)
    target = np.zeros_like(state)
    source[active] = active_source
    target[active] = active_target
    return source, target


def stationary_coupling_dual_value(
    reference: np.ndarray,
    state: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
) -> float:
    exponent = source_potential[:, np.newaxis] + target_potential[np.newaxis, :]
    return float(
        np.dot(state, source_potential + target_potential)
        - np.sum(reference * np.expm1(exponent))
    )


def stationary_coupling_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """KL projection retaining both stationary state marginals."""
    try:
        source, target = stationary_coupling_dual_potentials(
            increment, state, crossing
        )
    except ValueError:
        return float("inf")
    reference = conditional_product_source_target(increment, state, crossing)
    return max(
        0.0,
        stationary_coupling_dual_value(reference, state, source, target),
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
    if projection == "fourier-coercive":
        return fourier_coercive_divergence(increment, state, crossing)
    if projection == "fourier-cancellation":
        return max(
            fourier_coercive_divergence(increment, state, crossing),
            universal_cancellation_divergence(increment, state, crossing),
        )
    if projection == "normalized-coercive":
        return normalized_coercive_divergence(increment, state, crossing)
    if projection == "universal-cancellation":
        return universal_cancellation_divergence(increment, state, crossing)
    if projection == "normalized-cancellation":
        return max(
            normalized_coercive_divergence(increment, state, crossing),
            universal_cancellation_divergence(increment, state, crossing),
        )
    if projection == "residual-cancellation":
        return max(
            residual_coercive_divergence(increment, state, crossing),
            universal_cancellation_divergence(increment, state, crossing),
        )
    if projection == "proved-residual-cancellation":
        return max(
            proved_residual_coercive_divergence(increment, state, crossing),
            universal_cancellation_divergence(increment, state, crossing),
        )
    if projection == "target-l2":
        return target_l2_divergence(increment, state, crossing)
    if projection == "stationary":
        return stationary_coupling_divergence(increment, state, crossing)
    reference = conditional_product_cell_targets(increment, state, crossing)
    if projection == "target":
        return relative_entropy(state, reference.sum(axis=0))
    cell_target = cell_target_projection_divergence(reference, state)
    if projection == "hybrid":
        return max(
            cell_target,
            stationary_coupling_divergence(increment, state, crossing),
        )
    return cell_target


def fourier_coercive_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Pinsker/Fourier lower bound for the target-mismatch divergence.

    This specialization uses the exact F_5 spectral gap between 1 and the
    convex hull of the four nontrivial fifth roots of unity.
    """
    if len(increment) != 5 or len(state) != 5:
        raise ValueError("the Fourier coercivity bound is specialized to F_5")
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    if increment_support <= 0.0 or state_support <= 0.0:
        return 0.0
    if crossing < 0.0 or crossing > increment_support:
        return float("inf")
    defect = zero_payment_defect(increment, state, crossing)
    spectral_gap = 1.0 - np.cos(2.0 * np.pi / 5.0)
    # The mismatch has total mass zero, so ||v||_1^2 >= 2||v||_2^2.
    # Pinsker therefore gives D >= ||v||_2^2, rather than only half of it.
    return spectral_gap**2 * defect


def zero_payment_defect(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Squared distance from the target-mismatch zero set for one stage."""
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    if increment_support <= 0.0 or state_support <= 0.0:
        return 0.0
    state_nonzero = state[1:] / state_support
    comparison = np.empty(5)
    comparison[0] = crossing
    comparison[1:] = (increment_support - crossing) * state_nonzero
    uniform = np.full(5, increment_support / 5.0)
    difference = comparison - uniform
    return float(np.dot(difference, difference))


def normalized_coercive_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Candidate 4/5 support-normalized coercivity payment for F_5.

    Numerical minimization supports this inequality, but it is not yet a
    theorem.  Keep this projection separate from the proved Fourier bound.
    """
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    scale = max(increment_support, state_support)
    if scale <= 0.0:
        return 0.0
    return 0.8 * zero_payment_defect(increment, state, crossing) / scale


def universal_cancellation_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Minimize the cancellation KL over every nonzero increment law."""
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    both_nonzero = increment_support - crossing
    if both_nonzero <= 0.0 or state_support <= 0.0:
        return 0.0
    state_nonzero = state[1:] / state_support
    observed = crossing / both_nonzero
    reference = float(
        np.clip(observed, np.min(state_nonzero), np.max(state_nonzero))
    )
    return both_nonzero * relative_entropy(
        np.array([observed, 1.0 - observed]),
        np.array([reference, 1.0 - reference]),
    )


def residual_coercive_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Candidate 2/5 algebraic-residual coercivity payment for F_5."""
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    scale = max(increment_support, state_support)
    if scale <= 0.0:
        return 0.0
    return 0.4 * zero_payment_defect(increment, state, crossing) / scale


def proved_residual_coercive_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Proved 1/5 residual payment obtained from Fourier coercivity.

    If w is the target mismatch and (e_0,e_+) are the cancellation and
    conditional residuals, then ||w||_2^2 <= 2(2e_0^2+||e_+||_2^2).
    The F_5 Fourier gap gives ||w||_2^2 >= g^2 R with g^2>2/5, hence the
    residual quadratic is at least R/5.  Pinsker supplies the support scale.
    """
    increment_support = float(increment[1:].sum())
    state_support = float(state[1:].sum())
    scale = max(increment_support, state_support)
    if scale <= 0.0:
        return 0.0
    return 0.2 * zero_payment_defect(increment, state, crossing) / scale


def two_stage_zero_payment_defect(point: np.ndarray, layout: "Layout") -> float:
    return zero_payment_defect(
        point[layout.mu], point[layout.beta], float(point[layout.crossing1])
    ) + zero_payment_defect(
        point[layout.beta], point[layout.zeta], float(point[layout.crossing2])
    )


def target_l2_divergence(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> float:
    """Squared-l2 lower bound for the exact target mismatch."""
    target = conditional_product_target(increment, state, crossing)
    mismatch = state - target
    return float(np.dot(mismatch, mismatch))


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


def sign_symmetric_free_map(layout: Layout) -> np.ndarray:
    """Map one variable per nonzero sign orbit to the full free layout."""
    width = layout.prime - 1
    orbit_count = width // 2
    mapping = np.zeros((3 * width + 2, 3 * orbit_count + 2))
    for component in range(3):
        for value in range(1, layout.prime):
            representative = min(value, layout.prime - value)
            mapping[
                component * width + value - 1,
                component * orbit_count + representative - 1,
            ] = 1.0
    mapping[-2, -2] = 1.0
    mapping[-1, -1] = 1.0
    return mapping


def compress_sign_symmetric_point(
    point: np.ndarray, layout: Layout
) -> np.ndarray:
    """Project a full point to sign orbits while preserving support masses."""
    full = compress_point(point, layout)
    width = layout.prime - 1
    orbit_count = width // 2
    reduced = np.zeros(3 * orbit_count + 2)
    for component in range(3):
        for representative in range(1, orbit_count + 1):
            opposite = layout.prime - representative
            reduced[component * orbit_count + representative - 1] = 0.5 * (
                full[component * width + representative - 1]
                + full[component * width + opposite - 1]
            )
    reduced[-2:] = full[-2:]
    return reduced


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


def append_support_box_constraints(
    rows: np.ndarray,
    targets: np.ndarray,
    layout: Layout,
    lower: tuple[float, float, float, float, float],
    upper: tuple[float, float, float, float, float],
    *,
    free: bool,
) -> tuple[np.ndarray, np.ndarray]:
    width = layout.prime - 1
    size = 3 * width + 2 if free else layout.size
    if free:
        coordinates: tuple[slice | int, ...] = (
            slice(0, width),
            slice(width, 2 * width),
            slice(2 * width, 3 * width),
            3 * width,
            3 * width + 1,
        )
    else:
        coordinates = (
            slice(layout.mu.start + 1, layout.mu.stop),
            slice(layout.beta.start + 1, layout.beta.stop),
            slice(layout.zeta.start + 1, layout.zeta.stop),
            layout.crossing1,
            layout.crossing2,
        )
    additional_rows = []
    additional_targets = []
    for coordinate, lower_value, upper_value in zip(
        coordinates, lower, upper, strict=True
    ):
        upper_row = np.zeros(size)
        upper_row[coordinate] = 1.0
        additional_rows.append(upper_row)
        additional_targets.append(upper_value)
        lower_row = -upper_row
        additional_rows.append(lower_row)
        additional_targets.append(-lower_value)
    return (
        np.concatenate((rows, np.stack(additional_rows))),
        np.concatenate((targets, np.array(additional_targets))),
    )


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
    support_box = None
    if args.support_lower is not None:
        support_box = (args.support_lower, args.support_upper)
        upper, upper_target = append_support_box_constraints(
            upper,
            upper_target,
            layout,
            args.support_lower,
            args.support_upper,
            free=False,
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
    special_start_index = len(starts_full)
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
    special_starts = starts_full[special_start_index:]
    ordinary_starts = starts_full[:special_start_index]
    if len(special_starts) >= args.restarts:
        starts_full = ordinary_starts[:1] + special_starts[: args.restarts - 1]
    else:
        starts_full = (
            ordinary_starts[: args.restarts - len(special_starts)]
            + special_starts
        )
    free_upper, free_upper_target = free_constraints(
        layout, args.eta, args.delta
    )
    if support_box is not None:
        free_upper, free_upper_target = append_support_box_constraints(
            free_upper,
            free_upper_target,
            layout,
            support_box[0],
            support_box[1],
            free=True,
        )
    free_size = 3 * (args.prime - 1) + 2
    free_map = np.eye(free_size)
    compress_start = lambda point: compress_point(point, layout)
    if args.sign_symmetric:
        free_map = sign_symmetric_free_map(layout)
        free_upper = free_upper @ free_map
        free_size = free_map.shape[1]
        compress_start = lambda point: compress_sign_symmetric_point(
            point, layout
        )
    free_bounds = Bounds(
        np.zeros(free_size),
        np.concatenate((np.ones(free_size - 2), np.array([0.5, 0.5]))),
    )
    constraints = (
        LinearConstraint(free_upper, -np.inf, free_upper_target),
    )
    maximum_zero_defect = getattr(args, "maximum_zero_defect", None)
    if maximum_zero_defect is not None:
        zero_start = np.concatenate(
            (
                np.array([1.0 - args.eta, *([args.eta / 4.0] * 4)]),
                np.array(
                    [
                        1.0 - 1.5 * args.eta,
                        *([1.5 * args.eta / 4.0] * 4),
                    ]
                ),
                np.array([1.0 - args.delta, *([args.delta / 4.0] * 4)]),
                np.array([args.eta / 5.0, 1.5 * args.eta / 5.0]),
            )
        )
        starts_full[0] = zero_start
        constraints += (
            NonlinearConstraint(
                lambda free: two_stage_zero_payment_defect(
                    expand_free_point(free_map @ free, layout), layout
                ),
                0.0,
                maximum_zero_defect,
            ),
        )
    best = None
    for start_full in starts_full:
        start = compress_start(start_full)
        result = minimize(
            lambda point: -upper_bound(
                expand_free_point(free_map @ point, layout),
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
        full_point = expand_free_point(free_map @ result.x, layout)
        if maximum_zero_defect is not None:
            defect_error = (
                two_stage_zero_payment_defect(full_point, layout)
                - maximum_zero_defect
            )
            nonlinear_tolerance = max(1e-12, 10.0 * args.tolerance)
            if defect_error > nonlinear_tolerance:
                continue
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
    first_cell_target = projected_divergence(
        mu, beta, point[layout.crossing1], "cell-target"
    )
    second_cell_target = projected_divergence(
        beta, zeta, point[layout.crossing2], "cell-target"
    )
    first_stationary = projected_divergence(
        mu, beta, point[layout.crossing1], "stationary"
    )
    second_stationary = projected_divergence(
        beta, zeta, point[layout.crossing2], "stationary"
    )
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "projection": args.projection,
        "sign_symmetric": args.sign_symmetric,
        "support_lower": args.support_lower,
        "support_upper": args.support_upper,
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
        "zero_payment_defect": two_stage_zero_payment_defect(point, layout),
        "maximum_zero_payment_defect": maximum_zero_defect,
        "first_cell_target_divergence": first_cell_target,
        "second_cell_target_divergence": second_cell_target,
        "first_stationary_divergence": first_stationary,
        "second_stationary_divergence": second_stationary,
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
        "--support-lower",
        type=lambda value: tuple(float(entry) for entry in value.split(",")),
    )
    parser.add_argument(
        "--support-upper",
        type=lambda value: tuple(float(entry) for entry in value.split(",")),
    )
    parser.add_argument(
        "--projection",
        choices=(
            "target",
            "cancellation",
            "two-level",
            "fourier-coercive",
            "fourier-cancellation",
            "normalized-coercive",
            "universal-cancellation",
            "normalized-cancellation",
            "residual-cancellation",
            "proved-residual-cancellation",
            "target-l2",
            "cell-target",
            "stationary",
            "hybrid",
        ),
        default="two-level",
    )
    parser.add_argument("--sign-symmetric", action="store_true")
    parser.add_argument("--maximum-zero-defect", type=float)
    args = parser.parse_args()
    if args.prime < 3:
        parser.error("prime must be an odd prime")
    if args.repetition != 3:
        parser.error("the current ladder study is specialized to repetition three")
    if (args.support_lower is None) != (args.support_upper is None):
        parser.error("support-lower and support-upper must be supplied together")
    if args.support_lower is not None:
        if len(args.support_lower) != 5 or len(args.support_upper) != 5:
            parser.error("support bounds must contain A,B,C,x1,x2")
        if any(
            lower > upper
            for lower, upper in zip(
                args.support_lower, args.support_upper, strict=True
            )
        ):
            parser.error("every support lower bound must not exceed its upper bound")
    return args


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
