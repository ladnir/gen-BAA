#!/usr/bin/env python3
"""Numerical prototype for Perron bounds on the exact quinary exponent.

The script answers two diagnostic questions.

1. Does an optimized spectral cut reproduce the exact exponent at a numerical
   full-composition optimizer?
2. Can a small common mixture of such cuts control the broad box that defeats
   the current conditional-product checker?

All results are numerical.  The proposed proof checker would instead store
positive rational fugacities and a rational Collatz--Wielandt vector.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eig
from scipy.optimize import Bounds, linprog, minimize

import optimize_bulk_exponent as bulk
import search_internal_information_gap as internal_gap


@dataclass(frozen=True)
class SpectralData:
    log_radius: float
    increment_marginal: np.ndarray
    state_marginal: np.ndarray
    flow: np.ndarray


@dataclass(frozen=True)
class FirstCut:
    log_increment: np.ndarray
    log_state: np.ndarray
    log_radius: float
    value_at_reference: float
    gradient_residual: float
    optimizer_success: bool


@dataclass(frozen=True)
class ContractedCut:
    log_increment: np.ndarray
    log_q: float
    log_radius: float
    value_at_reference: float
    gradient_residual: float
    optimizer_success: bool


SecondCut = ContractedCut | FirstCut


@dataclass(frozen=True)
class CompositionBox:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    support_lower: tuple[float, ...]
    support_upper: tuple[float, ...]


def scale_symbol_vector(values: np.ndarray, multiplier: int) -> np.ndarray:
    """Pull back a symbol-indexed vector by multiplication in F_5."""

    inverse = pow(multiplier, -1, 5)
    return np.array([values[(inverse * value) % 5] for value in range(5)])


def scale_first_cut(cut: FirstCut, multiplier: int) -> FirstCut:
    return FirstCut(
        scale_symbol_vector(cut.log_increment, multiplier),
        scale_symbol_vector(cut.log_state, multiplier),
        cut.log_radius,
        cut.value_at_reference,
        cut.gradient_residual,
        cut.optimizer_success,
    )


def scale_second_cut(cut: SecondCut, multiplier: int) -> SecondCut:
    if isinstance(cut, FirstCut):
        return scale_first_cut(cut, multiplier)
    return ContractedCut(
        scale_symbol_vector(cut.log_increment, multiplier),
        cut.log_q,
        cut.log_radius,
        cut.value_at_reference,
        cut.gradient_residual,
        cut.optimizer_success,
    )


def uniform_nonzero_composition(composition: np.ndarray) -> np.ndarray:
    support = float(composition[1:].sum())
    return np.array([1.0 - support, *(support / 4.0 for _ in range(4))])


# This box is emitted after 200 splits by verify_conditional_product_bulk.py
# at delta=1323/10000 and eta=1/500 in the current tree.  The exact checker
# gives it upper bound 1.4638637550 because both relaxed KL payments vanish.
CURRENT_OBSTRUCTION_BOX = CompositionBox(
    lower=(
        1 / 2000,
        0.0,
        0.0,
        0.0,
        1323 / 10000,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    upper=(
        5297 / 20000,
        5297 / 40000,
        5297 / 40000,
        5297 / 20000,
        1323 / 5000,
        1323 / 10000,
        1323 / 10000,
        1323 / 10000,
        1323 / 20000,
        1323 / 20000,
        1323 / 20000,
        1323 / 20000,
        83 / 625,
        1323 / 10000,
    ),
    support_lower=(1 / 500, 1323 / 10000, 1323 / 20000),
    support_upper=(166 / 625, 1323 / 5000, 1323 / 10000),
)


def initial_composition_box(eta: float, delta: float) -> CompositionBox:
    """Return a simple box covering the complete bulk domain."""

    return CompositionBox(
        lower=(0.0,) * 14,
        upper=(
            *((min(1.0, 4.0 * delta),) * 4),
            *((min(1.0, 2.0 * delta),) * 4),
            *((delta,) * 4),
            min(1.0, 2.0 * delta),
            delta,
        ),
        support_lower=(eta, eta / 2.0, eta / 4.0),
        support_upper=(min(1.0, 4.0 * delta), min(1.0, 2.0 * delta), delta),
    )


def tighten_composition_box(box: CompositionBox) -> CompositionBox | None:
    """Propagate coordinate-sum bounds through the three compositions.

    The box stores four nonzero coordinates and their total support
    separately.  Bisection can make either description much sharper than the
    other.  Keeping both descriptions synchronized is cheap and substantially
    reduces entropy-secant loss in descendant boxes.
    """

    lower = list(box.lower)
    upper = list(box.upper)
    support_lower = list(box.support_lower)
    support_upper = list(box.support_upper)
    tolerance = 1e-12
    for component in range(3):
        start = 4 * component
        indices = range(start, start + 4)
        for _ in range(8):
            old = (
                tuple(lower[index] for index in indices),
                tuple(upper[index] for index in indices),
                support_lower[component],
                support_upper[component],
            )
            support_lower[component] = max(
                support_lower[component],
                sum(lower[index] for index in indices),
            )
            support_upper[component] = min(
                support_upper[component],
                sum(upper[index] for index in indices),
            )
            if support_lower[component] > support_upper[component] + tolerance:
                return None
            for index in indices:
                other_upper = sum(
                    upper[other] for other in indices if other != index
                )
                other_lower = sum(
                    lower[other] for other in indices if other != index
                )
                lower[index] = max(
                    lower[index], support_lower[component] - other_upper
                )
                upper[index] = min(
                    upper[index], support_upper[component] - other_lower
                )
                if lower[index] > upper[index] + tolerance:
                    return None
            new = (
                tuple(lower[index] for index in indices),
                tuple(upper[index] for index in indices),
                support_lower[component],
                support_upper[component],
            )
            if new == old:
                break
    return CompositionBox(
        tuple(max(0.0, value) for value in lower),
        tuple(min(1.0, value) for value in upper),
        tuple(max(0.0, value) for value in support_lower),
        tuple(min(1.0, value) for value in support_upper),
    )


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def transfer_matrix(
    log_increment: np.ndarray, log_state: np.ndarray
) -> np.ndarray:
    prime = len(log_increment)
    return np.array(
        [
            [
                np.exp(log_increment[(target - source) % prime] + log_state[target])
                for target in range(prime)
            ]
            for source in range(prime)
        ]
    )


def spectral_data(
    log_increment: np.ndarray, log_state: np.ndarray
) -> SpectralData:
    matrix = transfer_matrix(log_increment, log_state)
    values, left_vectors, right_vectors = eig(
        matrix, left=True, right=True, check_finite=False
    )
    index = int(np.argmax(values.real))
    radius = float(values[index].real)
    if radius <= 0.0 or abs(values[index].imag) > 1e-8 * radius:
        raise RuntimeError("failed to isolate the positive Perron root")

    left = np.asarray(left_vectors[:, index].real)
    right = np.asarray(right_vectors[:, index].real)
    if left.sum() < 0.0:
        left = -left
    if right.sum() < 0.0:
        right = -right
    # Positive matrices have positive Perron eigenvectors.  Tiny negative
    # components can appear when a fitted fugacity is near the log bound.
    left = np.abs(left)
    right = np.abs(right)
    normalization = float(np.dot(left, right))
    if normalization <= 0.0:
        raise RuntimeError("Perron eigenvectors have inconsistent signs")

    flow = (
        left[:, np.newaxis]
        * matrix
        * right[np.newaxis, :]
        / (radius * normalization)
    )
    prime = len(log_increment)
    increment = np.array(
        [
            sum(flow[source, (source + difference) % prime] for source in range(prime))
            for difference in range(prime)
        ]
    )
    state = flow.sum(axis=0)
    return SpectralData(float(np.log(radius)), increment, state, flow)


def unpack_first(parameters: np.ndarray, prime: int) -> tuple[np.ndarray, np.ndarray]:
    width = prime - 1
    log_increment = np.zeros(prime)
    log_state = np.zeros(prime)
    log_increment[1:] = parameters[:width]
    log_state[1:] = parameters[width:]
    return log_increment, log_state


def fit_first_cut(
    increment: np.ndarray, state: np.ndarray, log_bound: float
) -> FirstCut:
    prime = len(increment)

    def value_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_increment, log_state = unpack_first(parameters, prime)
        data = spectral_data(log_increment, log_state)
        value = (
            data.log_radius
            - float(np.dot(increment, log_increment))
            - float(np.dot(state, log_state))
        )
        gradient = np.concatenate(
            (
                data.increment_marginal[1:] - increment[1:],
                data.state_marginal[1:] - state[1:],
            )
        )
        return value, gradient

    result = minimize(
        value_gradient,
        np.zeros(2 * (prime - 1)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-log_bound, log_bound)] * (2 * (prime - 1)),
        options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 5000},
    )
    value, gradient = value_gradient(result.x)
    log_increment, log_state = unpack_first(result.x, prime)
    data = spectral_data(log_increment, log_state)
    return FirstCut(
        log_increment,
        log_state,
        data.log_radius,
        value,
        float(np.max(np.abs(gradient))),
        bool(result.success),
    )


def unpack_contracted(
    parameters: np.ndarray, prime: int
) -> tuple[np.ndarray, float, np.ndarray]:
    log_increment = np.zeros(prime)
    log_increment[1:] = parameters[: prime - 1]
    log_q = float(parameters[-1])
    log_state = np.zeros(prime)
    log_state[1:] = log_q
    return log_increment, log_q, log_state


def fit_contracted_cut(
    increment: np.ndarray, delta: float, log_bound: float
) -> ContractedCut:
    prime = len(increment)

    def value_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_increment, log_q, log_state = unpack_contracted(parameters, prime)
        data = spectral_data(log_increment, log_state)
        value = (
            data.log_radius
            - float(np.dot(increment, log_increment))
            - delta * log_q
        )
        gradient = np.concatenate(
            (
                data.increment_marginal[1:] - increment[1:],
                np.array([data.state_marginal[1:].sum() - delta]),
            )
        )
        return value, gradient

    result = minimize(
        value_gradient,
        np.zeros(prime),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-log_bound, log_bound)] * (prime - 1) + [(-log_bound, 0.0)],
        options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 5000},
    )
    value, gradient = value_gradient(result.x)
    log_increment, log_q, log_state = unpack_contracted(result.x, prime)
    data = spectral_data(log_increment, log_state)
    return ContractedCut(
        log_increment,
        log_q,
        data.log_radius,
        value,
        float(np.max(np.abs(gradient))),
        bool(result.success),
    )


def perron_upper(
    message: np.ndarray,
    middle: np.ndarray,
    delta: float,
    first: FirstCut,
    second: ContractedCut,
    repetition: int,
) -> float:
    return (
        first.log_radius
        + second.log_radius
        - delta * second.log_q
        - (1.0 - 1.0 / repetition) * entropy(message)
        - entropy(middle)
        - float(np.dot(message, first.log_increment))
        - float(np.dot(middle, first.log_state + second.log_increment))
    )


def full_perron_upper(
    message: np.ndarray,
    middle: np.ndarray,
    output: np.ndarray,
    first: FirstCut,
    second: FirstCut,
    repetition: int,
) -> float:
    return (
        first.log_radius
        + second.log_radius
        - (1.0 - 1.0 / repetition) * entropy(message)
        - entropy(middle)
        - float(np.dot(message, first.log_increment))
        - float(np.dot(middle, first.log_state + second.log_increment))
        - float(np.dot(output, second.log_state))
    )


def composition_vertices(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    support_lower: float,
    support_upper: float,
    canonical_message: bool,
) -> list[np.ndarray]:
    """Enumerate vertices of a four-symbol composition slice."""

    dimension = len(lower)
    inequalities: list[tuple[np.ndarray, float]] = []
    for index in range(dimension):
        unit = np.zeros(dimension)
        unit[index] = 1.0
        inequalities.append((unit, upper[index]))
        inequalities.append((-unit, -lower[index]))
    inequalities.append((np.ones(dimension), support_upper))
    inequalities.append((-np.ones(dimension), -support_lower))
    if canonical_message:
        for index in range(1, dimension):
            row = np.zeros(dimension)
            row[index] = 1.0
            row[0] = -1.0
            inequalities.append((row, 0.0))

    matrix = np.stack([row for row, _ in inequalities])
    target = np.array([value for _, value in inequalities])
    vertices: dict[tuple[float, ...], np.ndarray] = {}
    for active in itertools.combinations(range(len(inequalities)), dimension):
        active_matrix = matrix[list(active)]
        if np.linalg.matrix_rank(active_matrix) < dimension:
            continue
        point = np.linalg.solve(active_matrix, target[list(active)])
        if np.max(matrix @ point - target) > 1e-9:
            continue
        if point.sum() > 1.0 + 1e-9:
            continue
        composition = np.concatenate(([1.0 - point.sum()], point))
        key = tuple(np.round(composition, 12))
        vertices[key] = composition
    if not vertices:
        raise RuntimeError("composition box has no vertices")
    return list(vertices.values())


def full_box_constraints(
    layout: bulk.Layout,
    box: CompositionBox,
    eta: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Bounds]:
    equality, equality_target, upper, upper_target = bulk.constraints(
        layout, eta, delta
    )
    support_lower = (*box.support_lower, box.lower[12], box.lower[13])
    support_upper = (*box.support_upper, box.upper[12], box.upper[13])
    upper, upper_target = bulk.append_support_box_constraints(
        layout, upper, upper_target, support_lower, support_upper
    )
    canonical_rows = []
    for value in range(2, 5):
        row = np.zeros(layout.size)
        row[layout.mu.start + value] = 1.0
        row[layout.mu.start + 1] = -1.0
        canonical_rows.append(row)
    upper = np.concatenate((upper, np.stack(canonical_rows)))
    upper_target = np.concatenate((upper_target, np.zeros(len(canonical_rows))))
    lower_bounds = np.zeros(layout.size)
    upper_bounds = np.ones(layout.size)
    for component, start in (
        (layout.mu, 0),
        (layout.beta, 4),
        (layout.zeta, 8),
    ):
        lower_bounds[component.start + 1 : component.stop] = box.lower[start : start + 4]
        upper_bounds[component.start + 1 : component.stop] = box.upper[start : start + 4]
    return equality, equality_target, upper, upper_target, Bounds(lower_bounds, upper_bounds)


def feasible_box_points(
    box: CompositionBox,
    eta: float,
    delta: float,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    layout = bulk.Layout(5)
    equality, equality_target, upper, upper_target, bounds = full_box_constraints(
        layout, box, eta, delta
    )
    random = np.random.default_rng(seed)
    points = []
    for _ in range(max(count * 3, 24)):
        result = linprog(
            random.standard_normal(layout.size),
            A_ub=upper,
            b_ub=upper_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
            method="highs",
        )
        if result.success:
            points.append(result.x)
        if len(points) >= count:
            break
    return points


def reference_points(
    vertices: list[np.ndarray], count: int, seed: int
) -> list[np.ndarray]:
    if not vertices:
        return []
    random = np.random.default_rng(seed)
    references = [np.mean(vertices, axis=0)]
    while len(references) < count:
        chosen = random.choice(len(vertices), size=min(8, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        references.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )
    return references


def common_mixture_bound(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimize the worst vertex value over one cut mixture."""

    vertex_count, cut_count = values.shape
    objective = np.zeros(cut_count + 1)
    objective[-1] = 1.0
    inequalities = np.concatenate((values, -np.ones((vertex_count, 1))), axis=1)
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(vertex_count),
        A_eq=np.array([[1.0] * cut_count + [0.0]]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * cut_count + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"cut-mixture LP failed: {result.message}")
    return float(result.x[-1]), result.x[:-1]


def x_log_x(value: float) -> float:
    return 0.0 if value <= 0.0 else value * float(np.log(value))


def secant_x_log_x(lower: float, upper: float) -> tuple[float, float]:
    """Return slope and intercept of a secant upper bound for x log x."""

    if upper < lower:
        raise ValueError("invalid entropy interval")
    if upper - lower <= 1e-15:
        return 0.0, x_log_x((lower + upper) / 2.0)
    slope = (x_log_x(upper) - x_log_x(lower)) / (upper - lower)
    return slope, x_log_x(lower) - slope * lower


def linearized_cut(
    box: CompositionBox,
    first: FirstCut,
    second: ContractedCut,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Upper-bound one fixed spectral cut by an affine full-type objective."""

    layout = bulk.Layout(5)
    coefficients = np.zeros(layout.size)
    coefficients[layout.mu] = -first.log_increment
    coefficients[layout.beta] = -(first.log_state + second.log_increment)
    constant = first.log_radius + second.log_radius - delta * second.log_q
    component_intervals = (
        (
            layout.mu,
            (1.0 - box.support_upper[0], *box.lower[0:4]),
            (1.0 - box.support_lower[0], *box.upper[0:4]),
            2.0 / 3.0,
        ),
        (
            layout.beta,
            (1.0 - box.support_upper[1], *box.lower[4:8]),
            (1.0 - box.support_lower[1], *box.upper[4:8]),
            1.0,
        ),
    )
    for component, lowers, uppers, entropy_coefficient in component_intervals:
        for offset, (lower, upper) in enumerate(zip(lowers, uppers, strict=True)):
            slope, intercept = secant_x_log_x(lower, upper)
            coefficients[component.start + offset] += entropy_coefficient * slope
            constant += entropy_coefficient * intercept
    return constant, coefficients


def linearized_full_cut(
    box: CompositionBox,
    first: FirstCut,
    second: FirstCut,
) -> tuple[float, np.ndarray]:
    layout = bulk.Layout(5)
    coefficients = np.zeros(layout.size)
    coefficients[layout.mu] = -first.log_increment
    coefficients[layout.beta] = -(first.log_state + second.log_increment)
    coefficients[layout.zeta] = -second.log_state
    constant = first.log_radius + second.log_radius
    component_intervals = (
        (
            layout.mu,
            (1.0 - box.support_upper[0], *box.lower[0:4]),
            (1.0 - box.support_lower[0], *box.upper[0:4]),
            2.0 / 3.0,
        ),
        (
            layout.beta,
            (1.0 - box.support_upper[1], *box.lower[4:8]),
            (1.0 - box.support_lower[1], *box.upper[4:8]),
            1.0,
        ),
    )
    for component, lowers, uppers, entropy_coefficient in component_intervals:
        for offset, (lower, upper) in enumerate(zip(lowers, uppers, strict=True)):
            slope, intercept = secant_x_log_x(lower, upper)
            coefficients[component.start + offset] += entropy_coefficient * slope
            constant += entropy_coefficient * intercept
    return constant, coefficients


def binary_entropy(value: float) -> float:
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return -value * np.log(value) - (1.0 - value) * np.log(1.0 - value)


def support_joint_cells(
    increment_support: float, state_support: float, crossing: float
) -> np.ndarray:
    return np.array(
        [
            1.0 - state_support - crossing,
            state_support - increment_support + crossing,
            crossing,
            increment_support - crossing,
        ]
    )


def support_adjustment(point: np.ndarray, layout: bulk.Layout) -> float:
    message_support = float(point[layout.mu][1:].sum())
    middle_support = float(point[layout.beta][1:].sum())
    output_support = float(point[layout.zeta][1:].sum())
    first_crossing = float(
        sum(point[layout.edge(1, 0, target)] for target in range(1, 5))
    )
    second_crossing = float(
        sum(point[layout.edge(2, 0, target)] for target in range(1, 5))
    )
    first_joint = support_joint_cells(
        message_support, middle_support, first_crossing
    )
    second_joint = support_joint_cells(
        middle_support, output_support, second_crossing
    )
    return (
        (2.0 / 3.0) * binary_entropy(message_support)
        + 2.0 * binary_entropy(middle_support)
        + binary_entropy(output_support)
        - entropy(first_joint)
        - entropy(second_joint)
    )


def add_scaled_linear_form(
    coefficients: np.ndarray,
    constant: float,
    scale: float,
    form_constant: float,
    terms: dict[int, float],
) -> float:
    for index, value in terms.items():
        coefficients[index] += scale * value
    return constant + scale * form_constant


def joint_cell_forms(
    layout: bulk.Layout, stage: int
) -> tuple[tuple[float, dict[int, float]], ...]:
    if stage == 1:
        increment = list(range(layout.mu.start + 1, layout.mu.stop))
        state = list(range(layout.beta.start + 1, layout.beta.stop))
    else:
        increment = list(range(layout.beta.start + 1, layout.beta.stop))
        state = list(range(layout.zeta.start + 1, layout.zeta.stop))
    crossing = [layout.edge(stage, 0, target) for target in range(1, 5)]

    def form(
        constant: float,
        increment_coefficient: float,
        state_coefficient: float,
        crossing_coefficient: float,
    ) -> tuple[float, dict[int, float]]:
        terms: dict[int, float] = {}
        for index in increment:
            terms[index] = terms.get(index, 0.0) + increment_coefficient
        for index in state:
            terms[index] = terms.get(index, 0.0) + state_coefficient
        for index in crossing:
            terms[index] = terms.get(index, 0.0) + crossing_coefficient
        return constant, terms

    return (
        form(1.0, 0.0, -1.0, -1.0),
        form(0.0, -1.0, 1.0, 1.0),
        form(0.0, 0.0, 0.0, 1.0),
        form(0.0, 1.0, 0.0, -1.0),
    )


def joint_cell_intervals(
    increment_lower: float,
    increment_upper: float,
    state_lower: float,
    state_upper: float,
    crossing_lower: float,
    crossing_upper: float,
) -> tuple[tuple[float, float], ...]:
    return (
        (
            max(0.0, 1.0 - state_upper - crossing_upper),
            min(1.0, 1.0 - state_lower - crossing_lower),
        ),
        (
            max(0.0, state_lower - increment_upper + crossing_lower),
            min(1.0, state_upper - increment_lower + crossing_upper),
        ),
        (max(0.0, crossing_lower), min(1.0, crossing_upper)),
        (
            max(0.0, increment_lower - crossing_upper),
            min(1.0, increment_upper - crossing_lower),
        ),
    )


def linearized_lagrangian_cut(
    box: CompositionBox,
    first: FirstCut,
    second: SecondCut,
    delta: float,
    multiplier: float,
) -> tuple[float, np.ndarray]:
    if isinstance(second, ContractedCut):
        constant, coefficients = linearized_cut(box, first, second, delta)
    else:
        constant, coefficients = linearized_full_cut(box, first, second)
    constant *= 1.0 + multiplier
    coefficients *= 1.0 + multiplier
    layout = bulk.Layout(5)

    support_data = (
        (
            box.support_lower[0],
            box.support_upper[0],
            2.0 / 3.0,
            range(layout.mu.start + 1, layout.mu.stop),
        ),
        (
            box.support_lower[1],
            box.support_upper[1],
            2.0,
            range(layout.beta.start + 1, layout.beta.stop),
        ),
        (
            box.support_lower[2],
            box.support_upper[2],
            1.0,
            range(layout.zeta.start + 1, layout.zeta.stop),
        ),
    )
    for lower, upper, entropy_coefficient, indices in support_data:
        midpoint = min(max((lower + upper) / 2.0, 1e-12), 1.0 - 1e-12)
        slope = np.log((1.0 - midpoint) / midpoint)
        intercept = binary_entropy(midpoint) - slope * midpoint
        constant += multiplier * entropy_coefficient * intercept
        for index in indices:
            coefficients[index] += multiplier * entropy_coefficient * slope

    for stage, increment_index, state_index, crossing_index in (
        (1, 0, 1, 12),
        (2, 1, 2, 13),
    ):
        intervals = joint_cell_intervals(
            box.support_lower[increment_index],
            box.support_upper[increment_index],
            box.support_lower[state_index],
            box.support_upper[state_index],
            box.lower[crossing_index],
            box.upper[crossing_index],
        )
        forms = joint_cell_forms(layout, stage)
        for (lower, upper), (form_constant, terms) in zip(
            intervals, forms, strict=True
        ):
            slope, intercept = secant_x_log_x(lower, upper)
            constant += multiplier * intercept
            constant = add_scaled_linear_form(
                coefficients,
                constant,
                multiplier * slope,
                form_constant,
                terms,
            )
    return constant, coefficients


def linearized_objective(
    box: CompositionBox,
    first: FirstCut,
    second: SecondCut,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    if args.gap_multiplier == 0.0:
        if isinstance(second, ContractedCut):
            return linearized_cut(box, first, second, args.delta)
        return linearized_full_cut(box, first, second)
    return linearized_lagrangian_cut(
        box,
        first,
        second,
        args.delta,
        args.gap_multiplier,
    )


def maximize_affine_on_box(
    box: CompositionBox,
    eta: float,
    delta: float,
    constant: float,
    coefficients: np.ndarray,
) -> tuple[float, np.ndarray]:
    layout = bulk.Layout(5)
    equality, equality_target, upper, upper_target, bounds = full_box_constraints(
        layout, box, eta, delta
    )
    result = linprog(
        -coefficients,
        A_ub=upper,
        b_ub=upper_target,
        A_eq=equality,
        b_eq=equality_target,
        bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"spectral-cut LP failed: {result.message}")
    return constant - float(result.fun), result.x


def robust_transition_mixture(
    box: CompositionBox,
    args: argparse.Namespace,
    cut_pairs: list[tuple[FirstCut, SecondCut]],
    seed_points: list[np.ndarray],
    max_iterations: int = 100,
) -> tuple[float, float, np.ndarray, int, np.ndarray]:
    """Solve the common-cut minimax problem by constraint generation."""

    affine = [
        linearized_objective(box, first, second, args)
        for first, second in cut_pairs
    ]

    def values_at(point: np.ndarray) -> np.ndarray:
        return np.array(
            [constant + float(np.dot(coefficients, point)) for constant, coefficients in affine]
        )

    rows = [values_at(point) for point in seed_points]
    if not rows:
        raise RuntimeError("mixture optimization needs one feasible seed point")
    lower = float("-inf")
    weights = np.full(len(cut_pairs), 1.0 / len(cut_pairs))
    upper = float("inf")
    for iteration in range(1, max_iterations + 1):
        lower, weights = common_mixture_bound(np.stack(rows))
        combined_constant = sum(
            weight * constant for weight, (constant, _) in zip(weights, affine, strict=True)
        )
        combined_coefficients = sum(
            (weight * coefficients for weight, (_, coefficients) in zip(weights, affine, strict=True)),
            np.zeros(bulk.Layout(5).size),
        )
        upper, maximizer = maximize_affine_on_box(
            box,
            args.eta,
            args.delta,
            combined_constant,
            combined_coefficients,
        )
        if upper <= lower + 1e-9:
            return upper, lower, weights, iteration, maximizer
        rows.append(values_at(maximizer))
    return upper, lower, weights, max_iterations, maximizer


def entropy_split_dimension(box: CompositionBox, point: np.ndarray) -> int:
    layout = bulk.Layout(5)
    scores = []
    for dimension in range(8):
        component = layout.mu if dimension < 4 else layout.beta
        offset = dimension if dimension < 4 else dimension - 4
        coefficient = 2.0 / 3.0 if dimension < 4 else 1.0
        lower = box.lower[dimension]
        upper = box.upper[dimension]
        slope, intercept = secant_x_log_x(lower, upper)
        value = point[component.start + 1 + offset]
        gap = slope * value + intercept - x_log_x(value)
        scores.append(coefficient * max(0.0, gap))
    for support_dimension, component in enumerate((layout.mu, layout.beta)):
        coefficient = 2.0 / 3.0 if support_dimension == 0 else 1.0
        lower = 1.0 - box.support_upper[support_dimension]
        upper = 1.0 - box.support_lower[support_dimension]
        slope, intercept = secant_x_log_x(lower, upper)
        value = point[component.start]
        gap = slope * value + intercept - x_log_x(value)
        scores.append(coefficient * max(0.0, gap))
    return int(np.argmax(scores))


def point_diagnostic(args: argparse.Namespace) -> tuple[dict[str, object], FirstCut, ContractedCut]:
    optimizer_args = SimpleNamespace(
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
    point = bulk.optimize(optimizer_args)
    layout = bulk.Layout(5)
    message = np.array(point["mu"])
    middle = np.array(point["beta"])
    output = np.array(point["zeta"])
    first = fit_first_cut(message, middle, args.log_bound)
    second = fit_contracted_cut(middle, args.delta, args.log_bound)
    upper = perron_upper(message, middle, args.delta, first, second, 3)
    tau1 = np.array(point["tau1"])
    tau2 = np.array(point["tau2"])
    exact_first = entropy(tau1.reshape(-1)) - entropy(middle)
    exact_second = entropy(tau2.reshape(-1)) - entropy(output)
    return (
        {
            "full_composition_exponent": point["exponent"],
            "perron_upper_at_optimizer": upper,
            "total_fidelity_gap": upper - float(point["exponent"]),
            "first_stage_exact_value": exact_first,
            "first_stage_cut_value": first.value_at_reference,
            "first_stage_fidelity_gap": first.value_at_reference - exact_first,
            "contracted_second_cut_value": second.value_at_reference,
            "second_stage_value_at_optimizer": exact_second,
            "contracted_second_gap": second.value_at_reference - exact_second,
            "first_gradient_residual": first.gradient_residual,
            "second_gradient_residual": second.gradient_residual,
            "first_optimizer_success": first.optimizer_success,
            "second_optimizer_success": second.optimizer_success,
            "message": message.tolist(),
            "middle": middle.tolist(),
            "output": output.tolist(),
        },
        first,
        second,
    )


def obstruction_box_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    diagnostic = evaluate_composition_box(
        CURRENT_OBSTRUCTION_BOX,
        args,
        args.box_cuts,
        args.seed,
    )
    if args.lookahead and diagnostic["full_transition_box_feasible"]:
        candidates = []
        for dimension in range(10):
            children = split_composition_box(CURRENT_OBSTRUCTION_BOX, dimension)
            child_diagnostics = [
                evaluate_composition_box(
                    child,
                    args,
                    args.lookahead_cuts,
                    args.seed + 100 * dimension + child_index,
                )
                for child_index, child in enumerate(children)
            ]
            feasible_bounds = [
                float(child["common_mixture_transition_lp_upper"])
                for child in child_diagnostics
                if child["full_transition_box_feasible"]
            ]
            candidates.append(
                {
                    "dimension": dimension,
                    "children": child_diagnostics,
                    "post_split_upper": max(feasible_bounds, default=float("-inf")),
                }
            )
        best = min(candidates, key=lambda candidate: candidate["post_split_upper"])
        diagnostic["one_step_lookahead"] = {
            "best_dimension": best["dimension"],
            "best_post_split_upper": best["post_split_upper"],
            "candidate_post_split_uppers": [
                candidate["post_split_upper"] for candidate in candidates
            ],
            "cuts_per_child": args.lookahead_cuts,
        }
    return diagnostic


def evaluate_composition_box(
    box: CompositionBox,
    args: argparse.Namespace,
    cut_count: int,
    seed: int,
    include_projection: bool = True,
    inherited_first_cuts: list[FirstCut] | None = None,
    inherited_second_cuts: list[SecondCut] | None = None,
    cut_bank_out: dict[str, object] | None = None,
) -> dict[str, object]:
    tightened = tighten_composition_box(box)
    if tightened is None:
        return {
            "full_transition_box_feasible": False,
            "conclusion": "coordinate and support bounds are inconsistent",
        }
    box = tightened
    feasible = feasible_box_points(box, args.eta, args.delta, cut_count * 4, seed)
    if not feasible:
        return {
            "full_transition_box_feasible": False,
            "conclusion": "the relaxed obstruction box is empty for the exact transition polytope",
        }

    references = reference_points(feasible, cut_count, seed + 1)
    layout = bulk.Layout(5)
    first_cuts = list(inherited_first_cuts or ()) + [
        fit_first_cut(reference[layout.mu], reference[layout.beta], args.log_bound)
        for reference in references
    ]
    if args.symmetric_reference_cuts:
        first_cuts.extend(
            fit_first_cut(
                uniform_nonzero_composition(reference[layout.mu]),
                uniform_nonzero_composition(reference[layout.beta]),
                args.log_bound,
            )
            for reference in references
        )
    if args.retain_output_composition:
        second_cuts: list[SecondCut] = list(inherited_second_cuts or ()) + [
            fit_first_cut(
                reference[layout.beta], reference[layout.zeta], args.log_bound
            )
            for reference in references
        ]
    else:
        second_cuts = list(inherited_second_cuts or ()) + [
            fit_contracted_cut(
                reference[layout.beta], args.delta, args.log_bound
            )
            for reference in references
        ]
    if args.symmetric_reference_cuts:
        if args.retain_output_composition:
            second_cuts.extend(
                fit_first_cut(
                    uniform_nonzero_composition(reference[layout.beta]),
                    uniform_nonzero_composition(reference[layout.zeta]),
                    args.log_bound,
                )
                for reference in references
            )
        else:
            second_cuts.extend(
                fit_contracted_cut(
                    uniform_nonzero_composition(reference[layout.beta]),
                    args.delta,
                    args.log_bound,
                )
                for reference in references
            )

    def paired_cuts() -> list[tuple[FirstCut, SecondCut]]:
        expanded_first = (
            [
                scale_first_cut(cut, multiplier)
                for cut in first_cuts
                for multiplier in range(1, 5)
            ]
            if args.cut_orbits
            else first_cuts
        )
        expanded_second = (
            [
                scale_second_cut(cut, multiplier)
                for cut in second_cuts
                for multiplier in range(1, 5)
            ]
            if args.cut_orbits
            else second_cuts
        )
        if args.independent_stage_cuts:
            return [
                (first, second)
                for first in expanded_first
                for second in expanded_second
            ]
        return list(zip(expanded_first, expanded_second, strict=True))

    transition_upper = float("inf")
    transition_lower = float("-inf")
    transition_weights = np.array([])
    mixture_iterations = 0
    transition_maximizer = feasible[0]
    for adaptive_round in range(args.adaptive_cuts + 1):
        cut_pairs = paired_cuts()
        (
            transition_upper,
            transition_lower,
            transition_weights,
            round_iterations,
            transition_maximizer,
        ) = robust_transition_mixture(box, args, cut_pairs, feasible)
        mixture_iterations += round_iterations
        if adaptive_round == args.adaptive_cuts:
            break
        first_cuts.append(
            fit_first_cut(
                transition_maximizer[layout.mu],
                transition_maximizer[layout.beta],
                args.log_bound,
            )
        )
        if args.retain_output_composition:
            second_cuts.append(
                fit_first_cut(
                    transition_maximizer[layout.beta],
                    transition_maximizer[layout.zeta],
                    args.log_bound,
                )
            )
        else:
            second_cuts.append(
                fit_contracted_cut(
                    transition_maximizer[layout.beta],
                    args.delta,
                    args.log_bound,
                )
            )

    cut_pairs = paired_cuts()
    projection: dict[str, object] = {}
    if include_projection and not args.retain_output_composition:
        try:
            message_vertices = composition_vertices(
                box.lower[0:4],
                box.upper[0:4],
                box.support_lower[0],
                box.support_upper[0],
                canonical_message=True,
            )
            middle_vertices = composition_vertices(
                box.lower[4:8],
                box.upper[4:8],
                box.support_lower[1],
                box.support_upper[1],
                canonical_message=False,
            )
            rows = []
            for message in message_vertices:
                for middle in middle_vertices:
                    rows.append(
                        [
                            perron_upper(
                                message, middle, args.delta, first, second, 3
                            )
                            for first, second in cut_pairs
                        ]
                    )
            values = np.array(rows)
            mixture_bound, weights = common_mixture_bound(values)
            individual_bounds = values.max(axis=0)
            projection = {
                "message_vertices": len(message_vertices),
                "middle_vertices": len(middle_vertices),
                "best_single_cut_box_upper": float(individual_bounds.min()),
                "common_mixture_box_upper": mixture_bound,
                "active_mixture_cuts": int(np.count_nonzero(weights > 1e-9)),
                "mixture_weights": weights.tolist(),
            }
        except RuntimeError as error:
            projection = {"composition_projection_status": str(error)}
    elif include_projection:
        projection = {
            "composition_projection_status": (
                "omitted because the retained-output cut depends on zeta"
            )
        }
    transition_single: dict[str, object] = {}
    if include_projection:
        affine = [
            linearized_objective(box, first, second, args)
            for first, second in cut_pairs
        ]
        transition_single_bounds = [
            maximize_affine_on_box(
                box,
                args.eta,
                args.delta,
                constant,
                coefficients,
            )[0]
            for constant, coefficients in affine
        ]
        transition_single = {
            "best_single_cut_transition_lp_upper": float(
                min(transition_single_bounds)
            )
        }
    maximizing_message = transition_maximizer[layout.mu]
    maximizing_middle = transition_maximizer[layout.beta]
    maximizing_output = transition_maximizer[layout.zeta]
    adjustment = support_adjustment(transition_maximizer, layout)

    def exact_entropy_objective(first: FirstCut, second: SecondCut) -> float:
        if isinstance(second, ContractedCut):
            perron = perron_upper(
                maximizing_message,
                maximizing_middle,
                args.delta,
                first,
                second,
                3,
            )
        else:
            perron = full_perron_upper(
                maximizing_message,
                maximizing_middle,
                maximizing_output,
                first,
                second,
                3,
            )
        return (
            (1.0 + args.gap_multiplier) * perron
            + args.gap_multiplier * adjustment
        )

    mixture_exact_entropy = sum(
        weight * exact_entropy_objective(first, second)
        for weight, (first, second) in zip(
            transition_weights, cut_pairs, strict=True
        )
    )
    fitted_first = fit_first_cut(
        maximizing_message, maximizing_middle, args.log_bound
    )
    if args.retain_output_composition:
        fitted_second: SecondCut = fit_first_cut(
            maximizing_middle, maximizing_output, args.log_bound
        )
    else:
        fitted_second = fit_contracted_cut(
            maximizing_middle, args.delta, args.log_bound
        )
    fitted_at_maximizer = exact_entropy_objective(fitted_first, fitted_second)
    feasible_exponent = bulk.exponent(transition_maximizer, layout, 3)
    information_gap = internal_gap.gap_and_gradient(
        transition_maximizer, layout, 3
    )[0]
    feasible_objective = (
        feasible_exponent - args.gap_multiplier * information_gap
    )
    if cut_bank_out is not None:
        cut_bank_out["first"] = first_cuts
        cut_bank_out["second"] = second_cuts
    return {
        "full_transition_box_feasible": True,
        "feasible_points_sampled": len(feasible),
        "first_stage_cuts": len(first_cuts),
        "second_stage_cuts": len(second_cuts),
        "cut_pairs": len(cut_pairs),
        **projection,
        **transition_single,
        "common_mixture_transition_lp_upper": transition_upper,
        "common_mixture_transition_lp_lower": transition_lower,
        "transition_mixture_iterations": mixture_iterations,
        "active_transition_mixture_cuts": int(
            np.count_nonzero(transition_weights > 1e-9)
        ),
        "recommended_split_dimension": entropy_split_dimension(
            box, transition_maximizer
        ),
        "worst_type_message": maximizing_message.tolist(),
        "worst_type_middle": maximizing_middle.tolist(),
        "worst_type_output": maximizing_output.tolist(),
        "objective_kind": (
            "exponent"
            if args.gap_multiplier == 0.0
            else "exponent-minus-gap-multiplier-times-internal-gap"
        ),
        "worst_type_full_exponent": feasible_exponent,
        "worst_type_internal_gap": information_gap,
        "worst_type_true_objective": feasible_objective,
        "worst_type_locally_fitted_perron_upper": fitted_at_maximizer,
        "worst_type_mixture_exact_entropy_upper": mixture_exact_entropy,
        "worst_type_contraction_gap": fitted_at_maximizer - feasible_objective,
        "worst_type_cut_bank_gap": mixture_exact_entropy - fitted_at_maximizer,
        "worst_type_entropy_secant_gap": transition_upper - mixture_exact_entropy,
        "scope": (
            "all displayed uppers are sound for their numerical cuts; the transition-LP "
            "bounds also retain exact full-type feasibility, but neither result is an "
            "exact-arithmetic certificate"
        ),
    }


def search_obstruction_cover(args: argparse.Namespace) -> dict[str, object]:
    domain_box = (
        initial_composition_box(args.eta, args.delta)
        if args.search_full_domain
        else CURRENT_OBSTRUCTION_BOX
    )
    identifier = 0
    initial_bank: dict[str, object] = {}
    initial = evaluate_composition_box(
        domain_box,
        args,
        args.search_cuts,
        args.seed + 10_000,
        include_projection=False,
        cut_bank_out=initial_bank,
    )
    if not initial["full_transition_box_feasible"]:
        return {"status": "initial box infeasible"}
    queue: list[
        tuple[
            float,
            int,
            CompositionBox,
            dict[str, object],
            dict[str, object],
        ]
    ] = [
        (
            -float(initial["common_mixture_transition_lp_upper"]),
            identifier,
            domain_box,
            initial,
            initial_bank,
        )
    ]
    identifier += 1
    history = [float(initial["common_mixture_transition_lp_upper"])]
    for split_index in range(args.search_splits):
        negated_parent_value, _, box, diagnostic, parent_bank = heapq.heappop(queue)
        parent_value = -negated_parent_value
        dimension = int(diagnostic["recommended_split_dimension"])
        for child_index, child in enumerate(split_composition_box(box, dimension)):
            tightened_child = tighten_composition_box(child)
            if tightened_child is None:
                continue
            child_bank: dict[str, object] = {}
            child_diagnostic = evaluate_composition_box(
                tightened_child,
                args,
                args.search_cuts,
                args.seed + 20_000 + 2 * split_index + child_index,
                include_projection=False,
                inherited_first_cuts=(
                    parent_bank["first"] if args.inherit_cut_banks else None
                ),
                inherited_second_cuts=(
                    parent_bank["second"] if args.inherit_cut_banks else None
                ),
                cut_bank_out=child_bank,
            )
            if not child_diagnostic["full_transition_box_feasible"]:
                continue
            local_value = float(child_diagnostic["common_mixture_transition_lp_upper"])
            # The parent's common mixture remains valid on every child.  Keep
            # it when a randomly proposed local bank happens to be weaker.
            value = min(parent_value, local_value)
            child_diagnostic["common_mixture_transition_lp_upper"] = value
            child_diagnostic["inherited_parent_upper"] = parent_value
            child_diagnostic["local_transition_lp_upper"] = local_value
            heapq.heappush(
                queue,
                (
                    -value,
                    identifier,
                    tightened_child,
                    child_diagnostic,
                    child_bank,
                ),
            )
            identifier += 1
        if not queue:
            return {
                "status": "all children infeasible",
                "splits": split_index + 1,
            }
        history.append(-queue[0][0])
        if -queue[0][0] < 0.0:
            return {
                "status": "numerically negative cover",
                "domain": "full bulk domain" if args.search_full_domain else "recorded obstruction box",
                "splits": split_index + 1,
                "boxes_remaining": len(queue),
                "current_upper": -queue[0][0],
                "upper_history": history,
            }
    return {
        "status": "split budget exhausted",
        "domain": "full bulk domain" if args.search_full_domain else "recorded obstruction box",
        "splits": args.search_splits,
        "boxes_remaining": len(queue),
        "current_upper": -queue[0][0],
        "upper_history": history,
        "worst_recommended_split_dimension": int(
            queue[0][3]["recommended_split_dimension"]
        ),
        "worst_box": {
            "lower": list(queue[0][2].lower),
            "upper": list(queue[0][2].upper),
            "support_lower": list(queue[0][2].support_lower),
            "support_upper": list(queue[0][2].support_upper),
            "diagnostic": queue[0][3],
        },
    }


def split_composition_box(
    box: CompositionBox, dimension: int
) -> tuple[CompositionBox, CompositionBox]:
    """Bisect a message/middle coordinate or its total support."""

    if not 0 <= dimension < 10:
        raise ValueError("composition split dimension must lie in [0,10)")
    if dimension >= 8:
        support_dimension = dimension - 8
        midpoint = (
            box.support_lower[support_dimension]
            + box.support_upper[support_dimension]
        ) / 2.0
        left_support_upper = list(box.support_upper)
        left_support_upper[support_dimension] = midpoint
        right_support_lower = list(box.support_lower)
        right_support_lower[support_dimension] = midpoint
        return (
            CompositionBox(
                box.lower,
                box.upper,
                box.support_lower,
                tuple(left_support_upper),
            ),
            CompositionBox(
                box.lower,
                box.upper,
                tuple(right_support_lower),
                box.support_upper,
            ),
        )
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return (
        CompositionBox(box.lower, tuple(left_upper), box.support_lower, box.support_upper),
        CompositionBox(tuple(right_lower), box.upper, box.support_lower, box.support_upper),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-bound", type=float, default=40.0)
    parser.add_argument("--box-cuts", type=int, default=8)
    parser.add_argument("--lookahead", action="store_true")
    parser.add_argument("--lookahead-cuts", type=int, default=6)
    parser.add_argument("--search-splits", type=int, default=0)
    parser.add_argument("--search-cuts", type=int, default=8)
    parser.add_argument("--independent-stage-cuts", action="store_true")
    parser.add_argument("--inherit-cut-banks", action="store_true")
    parser.add_argument("--cut-orbits", action="store_true")
    parser.add_argument("--symmetric-reference-cuts", action="store_true")
    parser.add_argument("--adaptive-cuts", type=int, default=0)
    parser.add_argument("--gap-multiplier", type=float, default=0.0)
    parser.add_argument("--retain-output-composition", action="store_true")
    parser.add_argument("--search-full-domain", action="store_true")
    parser.add_argument("--skip-box", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.eta < 1.0:
        parser.error("eta must lie in (0,1)")
    if not 0.0 < args.delta < 1.0:
        parser.error("delta must lie in (0,1)")
    if (
        args.restarts < 1
        or args.box_cuts < 1
        or args.lookahead_cuts < 1
        or args.search_cuts < 1
        or args.search_splits < 0
        or args.adaptive_cuts < 0
    ):
        parser.error("restarts and cut counts must be positive")
    if args.log_bound <= 0.0:
        parser.error("log-bound must be positive")
    if args.gap_multiplier < 0.0:
        parser.error("gap-multiplier must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    point, _, _ = point_diagnostic(args)
    result: dict[str, object] = {
        "prime": 5,
        "repetition": 3,
        "delta": args.delta,
        "eta": args.eta,
        "gap_multiplier": args.gap_multiplier,
        "point_fidelity": point,
        "status": "numerical Perron prototype only",
    }
    if not args.skip_box:
        result["obstruction_box"] = obstruction_box_diagnostic(args)
    if args.search_splits:
        result["obstruction_box_search"] = search_obstruction_cover(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
