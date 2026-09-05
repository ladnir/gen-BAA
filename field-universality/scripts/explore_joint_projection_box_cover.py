#!/usr/bin/env python3
"""Floating interval cover for the exact F_5 joint-projection objective.

Every fitted KL dual is a pointwise lower bound on one stationary information
payment.  The interval routines below lower-bound that fixed dual throughout a
box.  Consequently, each reported box value is a mathematically valid upper
bound, apart from floating-point roundoff.  This script is a prototype for a
directed-rounding certificate; it is not itself an exact certificate.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp

import diagnose_joint_information_projection as joint
import optimize_bulk_exponent as bulk
import optimize_joint_projection_bound as outer


PRIME = 5
VARIABLES = 17


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    support_lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    support_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class StageCut:
    stage: int
    moment_constant: float
    moment_coefficients: np.ndarray
    log_kernel: np.ndarray
    residual: float


def affine_interval(
    constant: float, coefficients: np.ndarray, box: Box
) -> tuple[float, float]:
    lower = constant
    upper = constant
    for coefficient, left, right in zip(
        coefficients, box.lower, box.upper, strict=True
    ):
        if coefficient >= 0.0:
            lower += coefficient * left
            upper += coefficient * right
        else:
            lower += coefficient * right
            upper += coefficient * left
    return lower, upper


def stage_components(point: np.ndarray, stage: int) -> tuple[np.ndarray, np.ndarray, float]:
    if stage == 1:
        return point[0:5], point[5:10], float(point[15])
    if stage == 2:
        return point[5:10], point[10:15], float(point[16])
    raise ValueError("stage must be 1 or 2")


def stage_features_moments(
    point: np.ndarray, stage: int
) -> tuple[np.ndarray, np.ndarray]:
    increment, state, crossing = stage_components(point, stage)
    cells = joint.support_table(
        float(increment[1:].sum()), float(state[1:].sum()), crossing
    )
    equality, target = joint.constraints(increment, state, cells)
    return joint.constraint_feature_basis(equality, target)


def moment_affine_map(stage: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros(VARIABLES)
    features, constant = stage_features_moments(zero, stage)
    coefficients = np.empty((len(constant), VARIABLES))
    for index in range(VARIABLES):
        basis = np.zeros(VARIABLES)
        basis[index] = 1.0
        candidate_features, value = stage_features_moments(basis, stage)
        if not np.array_equal(candidate_features, features):
            raise RuntimeError("the joint-projection feature basis changed")
        coefficients[:, index] = value - constant
    return features, constant, coefficients


FEATURE_DATA = {stage: moment_affine_map(stage) for stage in (1, 2)}


def fit_cut(point: np.ndarray, stage: int) -> StageCut:
    increment, state, crossing = stage_components(point, stage)
    value, diagnostic = outer.stage_payment_dual(increment, state, crossing)
    if not np.isfinite(value):
        raise RuntimeError("reference point has no feasible stage projection")
    dual = diagnostic["dual"]
    potentials = np.asarray(dual["potentials"], dtype=float)
    features, constant, coefficients = FEATURE_DATA[stage]
    if not np.array_equal(np.asarray(dual["feature_rows"]), features):
        raise RuntimeError("fitted dual uses an unexpected feature basis")
    return StageCut(
        stage,
        float(np.dot(potentials, constant)),
        potentials @ coefficients,
        (potentials @ features).reshape((PRIME, PRIME)),
        float(dual["equality_residual"]),
    )


def scale_cut(cut: StageCut, multiplier: int) -> StageCut:
    coefficients = cut.moment_coefficients.copy()
    for start in (0, 5, 10):
        original = cut.moment_coefficients[start : start + PRIME]
        coefficients[start : start + PRIME] = np.array(
            [original[(multiplier * value) % PRIME] for value in range(PRIME)]
        )
    kernel = np.empty_like(cut.log_kernel)
    for difference in range(PRIME):
        for source in range(PRIME):
            kernel[difference, source] = cut.log_kernel[
                (multiplier * difference) % PRIME,
                (multiplier * source) % PRIME,
            ]
    return StageCut(
        cut.stage,
        cut.moment_constant,
        coefficients,
        kernel,
        cut.residual,
    )


def cut_orbit(cut: StageCut) -> tuple[StageCut, ...]:
    return tuple(scale_cut(cut, multiplier) for multiplier in range(1, PRIME))


def support_interval(box: Box, start: int) -> tuple[float, float]:
    support_index = start // PRIME
    lower_coordinates, upper_coordinates = simplex_intervals(box, start)
    lower = max(
        0.0,
        box.support_lower[support_index],
        1.0 - upper_coordinates[0],
        float(np.sum(lower_coordinates[1:])),
    )
    upper = min(
        1.0,
        box.support_upper[support_index],
        1.0 - lower_coordinates[0],
        float(np.sum(upper_coordinates[1:])),
    )
    return lower, upper


def simplex_intervals(box: Box, start: int) -> tuple[np.ndarray, np.ndarray]:
    support = start // PRIME
    raw_lower = np.asarray(
        box.lower[start : start + PRIME], dtype=float
    ).copy()
    raw_upper = np.asarray(
        box.upper[start : start + PRIME], dtype=float
    ).copy()
    raw_lower[0] = max(raw_lower[0], 1.0 - box.support_upper[support])
    raw_upper[0] = min(raw_upper[0], 1.0 - box.support_lower[support])
    if float(np.max(raw_upper - raw_lower)) <= 1e-14:
        return raw_lower.copy(), raw_upper.copy()
    lower = np.maximum(
        raw_lower,
        1.0 - float(np.sum(raw_upper)) + raw_upper,
    )
    upper = np.minimum(
        raw_upper,
        1.0 - float(np.sum(raw_lower)) + raw_lower,
    )
    return np.maximum(0.0, lower - 1e-15), np.minimum(1.0, upper + 1e-15)


def x_log_x(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def binary_entropy(value: float) -> float:
    return -x_log_x(value) - x_log_x(1.0 - value)


def minimum_x_log_x(lower: float, upper: float) -> float:
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    if lower > upper:
        return float("inf")
    minimizer = min(max(np.exp(-1.0), lower), upper)
    return x_log_x(minimizer)


def support_information_lower(box: Box, stage: int) -> float:
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if stage == 1 else (5, 10, 16)
    )
    increment_support = support_interval(box, increment_start)
    state_support = support_interval(box, state_start)
    cell_forms = (
        # J00 = state[0] - x.
        (0.0, {state_start: 1.0, crossing_index: -1.0}),
        # J01 = increment[0] - state[0] + x.
        (0.0, {increment_start: 1.0, state_start: -1.0, crossing_index: 1.0}),
        # J10 = x.
        (0.0, {crossing_index: 1.0}),
        # J11 = 1 - increment[0] - x.
        (1.0, {increment_start: -1.0, crossing_index: -1.0}),
    )
    cell_lower = 0.0
    for constant, sparse in cell_forms:
        coefficients = np.zeros(VARIABLES)
        for index, value in sparse.items():
            coefficients[index] = value
        left, right = affine_interval(constant, coefficients, box)
        cell_lower += minimum_x_log_x(left, right)
    marginal_lower = min(
        binary_entropy(increment_support[0]),
        binary_entropy(increment_support[1]),
    ) + min(
        binary_entropy(state_support[0]),
        binary_entropy(state_support[1]),
    )
    return max(0.0, cell_lower + marginal_lower)


def log_weighted_average_upper(
    masses: tuple[float, ...], log_weights: np.ndarray, support_lower: float
) -> float:
    possible = np.asarray(masses) > 0.0
    if not np.any(possible):
        return -np.inf
    maximum = float(np.max(log_weights[possible]))
    if support_lower <= 0.0:
        return maximum
    relaxed = float(
        logsumexp(np.log(np.asarray(masses)[possible]) + log_weights[possible])
        - np.log(support_lower)
    )
    return min(maximum, relaxed)


def log_partition_upper(box: Box, cut: StageCut) -> float:
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if cut.stage == 1 else (5, 10, 16)
    )
    increment_support = support_interval(box, increment_start)
    state_support = support_interval(box, state_start)
    _, increment_coordinates_upper = simplex_intervals(box, increment_start)
    _, state_coordinates_upper = simplex_intervals(box, state_start)
    increment_upper = tuple(increment_coordinates_upper[1:])
    state_upper = tuple(state_coordinates_upper[1:])

    cell_coefficients = []
    for constant, sparse in (
        (0.0, {state_start: 1.0, crossing_index: -1.0}),
        (0.0, {increment_start: 1.0, state_start: -1.0, crossing_index: 1.0}),
        (0.0, {crossing_index: 1.0}),
        (1.0, {increment_start: -1.0, crossing_index: -1.0}),
    ):
        coefficients = np.zeros(VARIABLES)
        for index, value in sparse.items():
            coefficients[index] = value
        _, upper = affine_interval(constant, coefficients, box)
        cell_coefficients.append(max(0.0, min(1.0, upper)))

    log_averages = [float(cut.log_kernel[0, 0])]
    log_averages.append(
        log_weighted_average_upper(
            state_upper, cut.log_kernel[0, 1:], state_support[0]
        )
    )
    log_averages.append(
        log_weighted_average_upper(
            increment_upper, cut.log_kernel[1:, 0], increment_support[0]
        )
    )
    pair_masses = tuple(
        increment_upper[difference - 1] * state_upper[source - 1]
        for difference in range(1, PRIME)
        for source in range(1, PRIME)
    )
    pair_support_lower = increment_support[0] * state_support[0]
    log_averages.append(
        log_weighted_average_upper(
            pair_masses, cut.log_kernel[1:, 1:].ravel(), pair_support_lower
        )
    )
    terms = [
        np.log(cell) + average
        for cell, average in zip(cell_coefficients, log_averages, strict=True)
        if cell > 0.0 and np.isfinite(average)
    ]
    return float(logsumexp(terms))


def payment_lower(box: Box, cut: StageCut) -> float:
    moment_lower, _ = affine_interval(
        cut.moment_constant, cut.moment_coefficients, box
    )
    lower = (
        support_information_lower(box, cut.stage)
        + moment_lower
        - log_partition_upper(box, cut)
    )
    return max(0.0, lower)


def cell_affine_forms(stage: int) -> tuple[tuple[float, np.ndarray], ...]:
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if stage == 1 else (5, 10, 16)
    )
    forms = []
    for constant, sparse in (
        (0.0, {state_start: 1.0, crossing_index: -1.0}),
        (0.0, {increment_start: 1.0, state_start: -1.0, crossing_index: 1.0}),
        (0.0, {crossing_index: 1.0}),
        (1.0, {increment_start: -1.0, crossing_index: -1.0}),
    ):
        coefficients = np.zeros(VARIABLES)
        for index, value in sparse.items():
            coefficients[index] = value
        forms.append((constant, coefficients))
    return tuple(forms)


def binary_entropy_chord(lower: float, upper: float) -> tuple[float, float]:
    lower = max(0.0, min(1.0, lower))
    upper = max(lower, min(1.0, upper))
    if upper - lower <= 1e-14:
        if 0.0 < lower < 1.0:
            slope = np.log((1.0 - lower) / lower)
        else:
            slope = 0.0
        return binary_entropy(lower) - slope * lower, slope
    left = binary_entropy(lower)
    right = binary_entropy(upper)
    slope = (right - left) / (upper - lower)
    return left - slope * lower, slope


def support_information_affine_lower(
    box: Box, stage: int
) -> tuple[float, np.ndarray]:
    increment_start, state_start = (0, 5) if stage == 1 else (5, 10)
    constant = 0.0
    coefficients = np.zeros(VARIABLES)
    for cell_constant, cell_coefficients in cell_affine_forms(stage):
        lower, upper = affine_interval(cell_constant, cell_coefficients, box)
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        if upper <= 0.0:
            continue
        reference = max((lower + upper) / 2.0, 1e-15)
        slope = np.log(reference) + 1.0
        constant += -reference + slope * cell_constant
        coefficients += slope * cell_coefficients
    for start in (increment_start, state_start):
        support_lower, support_upper = support_interval(box, start)
        intercept, slope = binary_entropy_chord(support_lower, support_upper)
        # support = 1 - composition[0].
        constant += intercept + slope
        coefficients[start] -= slope
    return constant, coefficients


def payment_affine_lower(box: Box, cut: StageCut) -> tuple[float, np.ndarray]:
    support_constant, support_coefficients = support_information_affine_lower(
        box, cut.stage
    )
    return (
        support_constant + cut.moment_constant - log_partition_upper(box, cut),
        support_coefficients + cut.moment_coefficients,
    )


def entropy_affine_upper(box: Box) -> tuple[float, np.ndarray]:
    lower, upper = simplex_intervals(box, 0)
    constant = 0.0
    coefficients = np.zeros(VARIABLES)
    for index in range(PRIME):
        if upper[index] <= 0.0:
            continue
        reference = max((lower[index] + upper[index]) / 2.0, 1e-15)
        slope = -(np.log(reference) + 1.0) / 3.0
        constant += reference / 3.0
        coefficients[index] = slope
    return constant, coefficients


def entropy_tangent_points(lower: float, upper: float) -> tuple[float, ...]:
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    if upper - lower <= 1e-14:
        return (max(lower, 1e-15),)
    points = {
        max(lower, 1e-15),
        max(upper, 1e-15),
        min(max(np.exp(-1.0), lower), upper),
    }
    for fraction in (0.2, 0.4, 0.6, 0.8):
        points.add(max(lower + fraction * (upper - lower), 1e-15))
    return tuple(sorted(points))


def joint_lp_constraints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equality, target, upper, upper_target = outer.outer_constraints(0.0, 1.0)
    # eta and delta are imposed by the box bounds supplied to the LP.  Retain
    # only the crossing constraints from the generic outer system.
    upper = upper[2:]
    upper_target = upper_target[2:]
    triangle_rows = np.zeros((2, VARIABLES))
    # A <= 2B and B <= 2C for D = S' - S.
    triangle_rows[0, 0] = -1.0
    triangle_rows[0, 5] = 2.0
    triangle_rows[1, 5] = -1.0
    triangle_rows[1, 10] = 2.0
    return equality, target, np.vstack((upper, triangle_rows)), np.concatenate(
        (upper_target, np.ones(2))
    )


JOINT_EQUALITY, JOINT_TARGET, JOINT_UPPER, JOINT_UPPER_TARGET = joint_lp_constraints()


def lp_objective_upper(
    box: Box, first_cuts: tuple[StageCut, ...], second_cuts: tuple[StageCut, ...]
) -> float:
    entropy_start = VARIABLES + 2
    variables = entropy_start + PRIME
    rows = []
    targets = []
    outer_rows = np.zeros((len(JOINT_UPPER), variables))
    outer_rows[:, :VARIABLES] = JOINT_UPPER
    rows.extend(outer_rows)
    targets.extend(JOINT_UPPER_TARGET)
    for component, start in enumerate((0, 5, 10)):
        row = np.zeros(variables)
        row[start + 1 : start + PRIME] = 1.0
        rows.append(row)
        targets.append(box.support_upper[component])
        rows.append(-row)
        targets.append(-box.support_lower[component])
    for cuts, epigraph in ((first_cuts, VARIABLES), (second_cuts, VARIABLES + 1)):
        for cut in cuts:
            constant, coefficients = payment_affine_lower(box, cut)
            row = np.zeros(variables)
            row[:VARIABLES] = coefficients
            row[epigraph] = -1.0
            rows.append(row)
            targets.append(-constant)
    message_lower, message_upper = simplex_intervals(box, 0)
    for index in range(PRIME):
        for reference in entropy_tangent_points(
            message_lower[index], message_upper[index]
        ):
            slope = -(np.log(reference) + 1.0)
            row = np.zeros(variables)
            row[index] = -slope
            row[entropy_start + index] = 1.0
            rows.append(row)
            targets.append(reference)
    equality = np.zeros((len(JOINT_EQUALITY), variables))
    equality[:, :VARIABLES] = JOINT_EQUALITY
    objective = np.zeros(variables)
    objective[VARIABLES:] = 1.0
    objective[entropy_start:] = -1.0 / 3.0
    result = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(targets),
        A_eq=equality,
        b_eq=JOINT_TARGET,
        bounds=[
            *(zip(box.lower, box.upper, strict=True)),
            (0.0, None),
            (0.0, None),
            *((0.0, None) for _ in range(PRIME)),
        ],
        method="highs",
    )
    if not result.success:
        return -np.inf
    return -float(result.fun)


def entropy_coordinate_upper(lower: float, upper: float) -> float:
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    maximizer = min(max(np.exp(-1.0), lower), upper)
    return -x_log_x(maximizer)


def message_entropy_upper(box: Box) -> float:
    return composition_entropy_upper(box, 0)


def composition_entropy_upper(box: Box, start: int) -> float:
    lower, upper = simplex_intervals(box, start)
    if float(np.sum(lower)) > 1.0 + 1e-10 or float(np.sum(upper)) < 1.0 - 1e-10:
        return -np.inf
    left = float(np.min(lower))
    right = float(np.max(upper))
    for _ in range(80):
        level = (left + right) / 2.0
        mass = float(np.sum(np.clip(level, lower, upper)))
        if mass < 1.0:
            left = level
        else:
            right = level
    distribution = np.clip((left + right) / 2.0, lower, upper)
    distribution /= float(np.sum(distribution))
    return float(sum(-x_log_x(value) for value in distribution))


def composition_entropy_lower(box: Box, start: int) -> float:
    lower, upper = simplex_intervals(box, start)
    best = float("inf")
    for free in range(PRIME):
        fixed = [index for index in range(PRIME) if index != free]
        for choices in itertools.product((0, 1), repeat=len(fixed)):
            point = np.zeros(PRIME)
            for index, choice in zip(fixed, choices, strict=True):
                point[index] = lower[index] if choice == 0 else upper[index]
            point[free] = 1.0 - float(np.sum(point))
            if point[free] < lower[free] - 1e-12 or point[free] > upper[free] + 1e-12:
                continue
            point[free] = min(max(point[free], lower[free]), upper[free])
            best = min(best, float(sum(-x_log_x(value) for value in point)))
    return best


def telescoping_entropy_upper(box: Box) -> float:
    return composition_entropy_upper(box, 10) - (
        2.0 / 3.0
    ) * composition_entropy_lower(box, 0)


def separated_entropy_upper(box: Box) -> float:
    """Optimize both entropy-difference payments over sound entropy ranges."""
    entropy_bounds = [
        (
            composition_entropy_lower(box, start),
            composition_entropy_upper(box, start),
        )
        for start in (0, 5, 10)
    ]
    if any(not np.isfinite(value) for bounds in entropy_bounds for value in bounds):
        return -np.inf
    # Variables are H(mu), H(beta), H(zeta), payment_1, payment_2.
    # payment_1 >= H(mu)-H(beta), payment_2 >= H(beta)-H(zeta).
    objective = np.array([-1.0 / 3.0, 0.0, 0.0, 1.0, 1.0])
    upper = np.array(
        [
            [1.0, -1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, -1.0],
        ]
    )
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(2),
        bounds=[*entropy_bounds, (0.0, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        return -np.inf
    return -float(result.fun)


def objective_upper(
    box: Box, first_cuts: tuple[StageCut, ...], second_cuts: tuple[StageCut, ...]
) -> float:
    return min(
        message_entropy_upper(box) / 3.0,
        telescoping_entropy_upper(box),
        separated_entropy_upper(box),
        lp_objective_upper(box, first_cuts, second_cuts),
    )


def bound_diagnostic(
    box: Box, first_cuts: tuple[StageCut, ...], second_cuts: tuple[StageCut, ...]
) -> dict[str, float]:
    entropy = message_entropy_upper(box) / 3.0
    first = max(payment_lower(box, cut) for cut in first_cuts)
    second = max(payment_lower(box, cut) for cut in second_cuts)
    joint_lp = lp_objective_upper(box, first_cuts, second_cuts)
    telescoping = telescoping_entropy_upper(box)
    separated = separated_entropy_upper(box)
    return {
        "entropy_upper": entropy,
        "first_payment_lower": first,
        "second_payment_lower": second,
        "separable_interval_upper": entropy - first - second,
        "joint_lp_upper": joint_lp,
        "telescoping_entropy_upper": telescoping,
        "separated_entropy_upper": separated,
        "upper": min(entropy, telescoping, separated, joint_lp),
    }


def extended_linear_system(eta: float, delta: float) -> tuple[np.ndarray, ...]:
    layout = bulk.Layout(PRIME)
    equality, target, upper, upper_target = bulk.constraints(layout, eta, delta)
    extended_equality = np.zeros((len(equality) + 2, layout.size + 2))
    extended_equality[: len(equality), : layout.size] = equality
    extended_target = np.concatenate((target, np.zeros(2)))
    for layer, crossing_index in ((1, layout.size), (2, layout.size + 1)):
        row = extended_equality[len(equality) + layer - 1]
        for destination in range(1, PRIME):
            row[layout.edge(layer, 0, destination)] = 1.0
        row[crossing_index] = -1.0
    extended_upper = np.zeros((len(upper), layout.size + 2))
    extended_upper[:, : layout.size] = upper
    return extended_equality, extended_target, extended_upper, upper_target


def extended_bounds(box: Box) -> list[tuple[float, float]]:
    layout = bulk.Layout(PRIME)
    bounds = [(0.0, 1.0)] * (layout.size + 2)
    for index in range(15):
        bounds[index] = (box.lower[index], box.upper[index])
    bounds[layout.size] = (box.lower[15], box.upper[15])
    bounds[layout.size + 1] = (box.lower[16], box.upper[16])
    return bounds


def feasible_reference(
    box: Box, eta: float, delta: float, seed: int, vertices: int = 4
) -> np.ndarray | None:
    layout = bulk.Layout(PRIME)
    equality, target, upper, upper_target = extended_linear_system(eta, delta)
    support_rows = np.zeros((6, layout.size + 2))
    support_targets = np.zeros(6)
    for component, start in enumerate((0, 5, 10)):
        support_rows[2 * component, start + 1 : start + PRIME] = 1.0
        support_targets[2 * component] = box.support_upper[component]
        support_rows[2 * component + 1, start + 1 : start + PRIME] = -1.0
        support_targets[2 * component + 1] = -box.support_lower[component]
    upper = np.vstack((upper, support_rows))
    upper_target = np.concatenate((upper_target, support_targets))
    random = np.random.default_rng(seed)
    points = []
    for _ in range(vertices):
        result = linprog(
            random.standard_normal(layout.size + 2),
            A_ub=upper,
            b_ub=upper_target,
            A_eq=equality,
            b_eq=target,
            bounds=extended_bounds(box),
            method="highs",
        )
        if result.success:
            points.append(result.x)
    if not points:
        return None
    average = np.mean(points, axis=0)
    return np.concatenate((average[:15], average[layout.size : layout.size + 2]))


def point_in_box(box: Box, point: np.ndarray, tolerance: float = 1e-10) -> bool:
    coordinates = np.asarray(point[:VARIABLES])
    if np.any(coordinates < np.asarray(box.lower) - tolerance):
        return False
    if np.any(coordinates > np.asarray(box.upper) + tolerance):
        return False
    supports = np.array(
        [float(np.sum(coordinates[start + 1 : start + PRIME])) for start in (0, 5, 10)]
    )
    return bool(
        np.all(supports >= np.asarray(box.support_lower) - tolerance)
        and np.all(supports <= np.asarray(box.support_upper) + tolerance)
    )


def saddle_reference(args: argparse.Namespace) -> np.ndarray | None:
    if args.reference_restarts <= 0:
        return None
    result = outer.optimize(
        SimpleNamespace(
            delta=args.delta,
            eta=args.eta,
            full_restarts=args.reference_restarts,
            seed=args.seed,
            max_iterations=args.reference_iterations,
            tolerance=1e-10,
        )
    )
    return np.concatenate(
        (
            np.asarray(result["message"]),
            np.asarray(result["middle"]),
            np.asarray(result["output"]),
            np.array([result["first_crossing"], result["second_crossing"]]),
        )
    )


def initial_box(eta: float, delta: float) -> Box:
    lower = np.zeros(VARIABLES)
    upper = np.ones(VARIABLES)
    upper[0] = 1.0 - eta
    lower[10] = 1.0 - delta
    upper[11:15] = delta
    return Box(
        tuple(lower),
        tuple(upper),
        (eta, eta / 2.0, eta / 4.0),
        (1.0, 1.0, delta),
    )


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    if dimension >= VARIABLES:
        support = dimension - VARIABLES
        midpoint = (
            box.support_lower[support] + box.support_upper[support]
        ) / 2.0
        left_upper = list(box.support_upper)
        left_upper[support] = midpoint
        right_lower = list(box.support_lower)
        right_lower[support] = midpoint
        return (
            Box(box.lower, box.upper, box.support_lower, tuple(left_upper)),
            Box(box.lower, box.upper, tuple(right_lower), box.support_upper),
        )
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return (
        Box(box.lower, tuple(left_upper), box.support_lower, box.support_upper),
        Box(tuple(right_lower), box.upper, box.support_lower, box.support_upper),
    )


def split_dimension_by_width(box: Box, delta: float) -> int:
    scales = np.ones(VARIABLES)
    scales[11:15] = max(delta, 1e-12)
    coordinate_widths = (np.asarray(box.upper) - np.asarray(box.lower)) / scales
    support_scales = np.array([1.0, 1.0, max(delta, 1e-12)])
    support_widths = (
        np.asarray(box.support_upper) - np.asarray(box.support_lower)
    ) / support_scales
    widths = np.concatenate((coordinate_widths, support_widths))
    return int(np.argmax(widths))


def split_dimension(
    box: Box,
    delta: float,
    first_cuts: tuple[StageCut, ...],
    second_cuts: tuple[StageCut, ...],
) -> int:
    widths = np.concatenate(
        (
            np.asarray(box.upper) - np.asarray(box.lower),
            np.asarray(box.support_upper) - np.asarray(box.support_lower),
        )
    )
    candidates = np.flatnonzero(widths > 1e-12)
    if len(candidates) == 0:
        return 0
    current_upper = objective_upper(box, first_cuts, second_cuts)
    best_dimension = split_dimension_by_width(box, delta)
    best_upper = float("inf")
    for dimension in candidates:
        child_upper = max(
            objective_upper(child, first_cuts, second_cuts)
            for child in split(box, int(dimension))
        )
        if child_upper < best_upper:
            best_upper = child_upper
            best_dimension = int(dimension)
    if best_upper >= current_upper - 1e-12:
        return split_dimension_by_width(box, delta)
    return best_dimension


def evaluate(
    box: Box,
    args: argparse.Namespace,
    seed: int,
    inherited_first: tuple[StageCut, ...] = (),
    inherited_second: tuple[StageCut, ...] = (),
    preferred_reference: np.ndarray | None = None,
) -> tuple[dict[str, object], tuple[StageCut, ...], tuple[StageCut, ...]] | None:
    reference = None
    if preferred_reference is not None and point_in_box(box, preferred_reference):
        value, diagnostic = outer.objective(preferred_reference)
        if diagnostic["feasible"] and np.isfinite(value):
            reference = preferred_reference
    if reference is None:
        reference = feasible_reference(box, args.eta, args.delta, seed, args.vertices)
    if reference is None:
        return None
    fitted_first = fit_cut(reference, 1)
    fitted_second = fit_cut(reference, 2)
    new_first = cut_orbit(fitted_first) if args.cut_orbits else (fitted_first,)
    new_second = cut_orbit(fitted_second) if args.cut_orbits else (fitted_second,)
    first = (*inherited_first, *new_first)[-args.cut_depth :]
    second = (*inherited_second, *new_second)[-args.cut_depth :]
    bounded = bound_diagnostic(box, first, second)
    value, _ = outer.objective(reference)
    diagnostic = {
        **bounded,
        "reference": reference.tolist(),
        "reference_value": value,
        "interval_gap": bounded["upper"] - value,
        "first_residual": first[-1].residual,
        "second_residual": second[-1].residual,
        "first_cuts": len(first),
        "second_cuts": len(second),
        "split_dimension": split_dimension(box, args.delta, first, second),
    }
    return diagnostic, first, second


def cover(args: argparse.Namespace) -> dict[str, object]:
    root = initial_box(args.eta, args.delta)
    preferred_reference = saddle_reference(args)
    evaluated = evaluate(
        root, args, args.seed, preferred_reference=preferred_reference
    )
    if evaluated is None:
        raise RuntimeError("initial box is infeasible")
    diagnostic, first_cuts, second_cuts = evaluated
    queue = [
        (-float(diagnostic["upper"]), 0, root, diagnostic, first_cuts, second_cuts)
    ]
    history = [float(diagnostic["upper"])]
    identifier = 1
    for index in range(args.max_splits):
        _, _, box, parent, first_cuts, second_cuts = heapq.heappop(queue)
        dimension = int(parent["split_dimension"])
        for child_index, child in enumerate(split(box, dimension)):
            child_evaluated = evaluate(
                child,
                args,
                args.seed + 2 * index + child_index + 1,
                first_cuts,
                second_cuts,
                preferred_reference,
            )
            if child_evaluated is None:
                continue
            child_diagnostic, child_first, child_second = child_evaluated
            child_diagnostic["upper"] = min(
                float(parent["upper"]), float(child_diagnostic["upper"])
            )
            heapq.heappush(
                queue,
                (
                    -float(child_diagnostic["upper"]),
                    identifier,
                    child,
                    child_diagnostic,
                    child_first,
                    child_second,
                ),
            )
            identifier += 1
        if not queue:
            raise RuntimeError("all child boxes were rejected")
        history.append(-queue[0][0])
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {"splits": index + 1, "upper": -queue[0][0], "boxes": len(queue)}
                ),
                flush=True,
            )
        if -queue[0][0] < args.target:
            return {
                "status": "floating interval cover reached target",
                "delta": args.delta,
                "upper": -queue[0][0],
                "splits": index + 1,
                "boxes": len(queue),
                "history": history,
            }
    worst = queue[0]
    return {
        "status": "floating interval cover incomplete",
        "delta": args.delta,
        "upper": -worst[0],
        "splits": args.max_splits,
        "boxes": len(queue),
        "history": history,
        "worst_box": {
            "lower": list(worst[2].lower),
            "upper": list(worst[2].upper),
            "support_lower": list(worst[2].support_lower),
            "support_upper": list(worst[2].support_upper),
            "diagnostic": worst[3],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--max-splits", type=int, default=20)
    parser.add_argument("--vertices", type=int, default=4)
    parser.add_argument("--cut-depth", type=int, default=8)
    parser.add_argument("--reference-restarts", type=int, default=0)
    parser.add_argument("--reference-iterations", type=int, default=300)
    parser.add_argument(
        "--cut-orbits", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if (
        args.max_splits < 0
        or args.vertices < 1
        or args.cut_depth < 1
        or args.reference_restarts < 0
    ):
        parser.error("split and vertex counts must be nonnegative")
    return args


if __name__ == "__main__":
    print(json.dumps(cover(parse_args()), indent=2, sort_keys=True))
