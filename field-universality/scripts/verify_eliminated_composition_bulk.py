#!/usr/bin/env python3
"""Exact p=5 bulk checker after eliminating conditional compositions."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from mpmath import iv
from scipy.optimize import linprog

from verify_conditional_product_bulk import (
    Cut,
    Interval,
    ONE,
    TWO,
    ZERO,
    default_cuts,
    entropy_bounds,
    iv_fraction,
    negative_log_upper,
    rational_interval,
    make_cut,
)
from verify_reduced_composition_bulk import support_vertices


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, Fraction, Fraction, Fraction, Fraction]
    upper: tuple[Fraction, Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class Certificate:
    first_weights: tuple[tuple[tuple[int, Fraction], ...], ...]
    order_multipliers: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True)
class Bound:
    value: Fraction
    box: Box
    split_dimension: int


def candidate_certificate() -> Certificate:
    scale = 1_000_000_000_000
    return Certificate(
        (
            ((4, ONE),),
            ((0, ONE),),
            ((6, ONE),),
            ((2, ONE),),
        ),
        (
            (ZERO, ZERO, ZERO, Fraction(501_256_116, scale)),
            (ZERO, ZERO, Fraction(974_842_536, scale), ZERO),
            (ZERO, Fraction(975_915_118, scale), ZERO, ZERO),
            (ZERO, ZERO, ZERO, ZERO),
        ),
    )


def reference_certificates() -> tuple[Certificate, ...]:
    scale = 1_000_000_000_000

    def multiplier_rows(first: int, second: int, third: int) -> tuple[tuple[Fraction, ...], ...]:
        return (
            (ZERO, ZERO, ZERO, Fraction(first, scale)),
            (ZERO, ZERO, Fraction(second, scale), ZERO),
            (ZERO, Fraction(third, scale), ZERO, ZERO),
            (ZERO, ZERO, ZERO, ZERO),
        )

    def certificate(
        first: tuple[int, int, int, int],
        multipliers: tuple[int, int, int],
    ) -> Certificate:
        return Certificate(
            tuple(((cut, ONE),) for cut in first),
            multiplier_rows(*multipliers),
        )

    return (
        candidate_certificate(),
        certificate(
            (4, 8, 14, 2),
            (2_560_561_870, 5_221_715_199, 5_249_865_380),
        ),
        certificate(
            (12, 8, 14, 2),
            (14_603_488_787, 26_429_801_486, 26_573_940_142),
        ),
        certificate(
            (12, 8, 14, 10),
            (79_207_684_151, 139_454_701_176, 140_225_522_976),
        ),
    )


def anchored_adversary(logits: np.ndarray) -> tuple[float, np.ndarray, set[int]]:
    pool = {0}
    while True:
        mean_logit = float(np.mean(logits[list(pool)]))
        violating = [
            index
            for index in range(len(logits))
            if index not in pool and logits[index] > mean_logit
        ]
        if not violating:
            break
        pool.add(max(violating, key=lambda index: logits[index]))
    mean_logit = float(np.mean(logits[list(pool)]))
    effective = np.array(logits, copy=True)
    for index in pool:
        effective[index] = mean_logit
    maximum = float(np.max(effective))
    weights = np.exp(effective - maximum)
    normalizer = float(np.sum(weights))
    probability = weights / normalizer
    pooled = sum(probability[index] for index in pool) / len(pool)
    for index in pool:
        probability[index] = pooled
    return maximum + math.log(normalizer), probability, pool


def local_certificate(
    box: Box,
    cuts: tuple[Cut, ...],
) -> tuple[tuple[Cut, ...], Certificate] | None:
    vertices = joint_support_vertices(box)
    reference = tuple(
        sum((vertex[index] for vertex in vertices), ZERO) / len(vertices)
        for index in range(5)
    )
    message_support = float(reference[0])
    middle_support = float(reference[1])
    crossing = float(reference[3])
    new_cuts: list[Cut] = []
    first_weights = []
    multiplier_rows = []
    scale = 1 << 40
    for middle_symbol in range(1, 5):
        best: tuple[float, np.ndarray] | None = None
        for cut in cuts:
            costs = np.array(
                [
                    float(
                        evaluate_affine(
                            pure_cut_coefficients(
                                cut, difference, middle_symbol
                            ),
                            (reference[0], reference[1], reference[3]),
                        )
                    )
                    for difference in range(1, 5)
                ]
            )
            normalizer, probability, _ = anchored_adversary(
                -3.0 * costs / message_support
            )
            value = message_support * normalizer / 3.0
            if best is None or value < best[0]:
                best = (value, probability)
        if best is None:
            return None
        state_conditional = np.full(4, 2.5e-9)
        state_conditional[middle_symbol - 1] += 1.0 - 1e-8
        increment = np.concatenate(
            ([1.0 - message_support], message_support * best[1])
        )
        state = np.concatenate(
            ([1.0 - middle_support], middle_support * state_conditional)
        )
        try:
            proposed = make_cut(increment, state, crossing)
        except ValueError:
            return None
        cut_index = len(cuts) + len(new_cuts)
        new_cuts.append(proposed)
        costs = np.array(
            [
                float(
                    evaluate_affine(
                        pure_cut_coefficients(
                            proposed, difference, middle_symbol
                        ),
                        (reference[0], reference[1], reference[3]),
                    )
                )
                for difference in range(1, 5)
            ]
        )
        _, _, pool = anchored_adversary(-3.0 * costs / message_support)
        mean_cost = float(np.mean(costs[list(pool)]))
        multipliers = [ZERO] * 4
        for index in pool:
            if index != 0:
                value = max(0.0, mean_cost - costs[index])
                multipliers[index] = Fraction(round(value * scale), scale)
        first_weights.append(((cut_index, ONE),))
        multiplier_rows.append(tuple(multipliers))
    return cuts + tuple(new_cuts), Certificate(
        tuple(first_weights), tuple(multiplier_rows)
    )


def tighten(raw: Box, eta: Fraction, delta: Fraction) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    for _ in range(12):
        before = (tuple(lower), tuple(upper))
        lower[0] = max(lower[0], eta, TWO * lower[3])
        upper[0] = min(upper[0], 4 * delta, TWO * upper[1])
        lower[1] = max(lower[1], lower[0] / TWO, TWO * lower[4])
        upper[1] = min(upper[1], 2 * delta, TWO * upper[2])
        lower[2] = max(lower[2], lower[1] / TWO)
        upper[2] = min(upper[2], delta)

        lower[3] = max(lower[3], ZERO, lower[0] - upper[1])
        upper[3] = min(upper[3], upper[0] / TWO, ONE - lower[1])
        upper[0] = min(upper[0], upper[1] + upper[3])
        lower[1] = max(lower[1], lower[0] - upper[3])

        lower[4] = max(lower[4], ZERO, lower[1] - upper[2])
        upper[4] = min(upper[4], upper[1] / TWO, ONE - lower[2])
        upper[1] = min(upper[1], upper[2] + upper[4])
        lower[2] = max(lower[2], lower[1] - upper[4])
        if any(lower[index] > upper[index] for index in range(5)):
            return None
        if before == (tuple(lower), tuple(upper)):
            break
    return Box(tuple(lower), tuple(upper))


def affine_extrema(
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction],
    increment: Interval,
    state: Interval,
    crossing: Interval,
) -> Interval:
    constant, increment_coefficient, state_coefficient, crossing_coefficient = (
        coefficients
    )
    values = tuple(
        constant
        + increment_coefficient * vertex[0]
        + state_coefficient * vertex[1]
        + crossing_coefficient * vertex[2]
        for vertex in support_vertices(increment, state, crossing)
    )
    return Interval(min(values), max(values))


@lru_cache(maxsize=None)
def pure_cut_coefficients(
    cut: Cut, difference: int, source: int
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    target = (difference + source) % 5
    row_zero = cut.exponential_minus_one[0][0].upper
    row_state = cut.exponential_minus_one[1][source].upper
    row_increment = cut.exponential_minus_one[2][difference].upper
    row_both = cut.exponential_minus_one[3][target].upper
    return (
        cut.cell[0] + cut.target[0] - row_zero,
        -cut.cell[1] + cut.cell[3] + row_state - row_both,
        -cut.cell[0]
        + cut.cell[1]
        + cut.target[source]
        - cut.target[0]
        + row_zero
        - row_state,
        -cut.cell[0]
        + cut.cell[1]
        + cut.cell[2]
        - cut.cell[3]
        + row_zero
        - row_state
        - row_increment
        + row_both,
    )


@lru_cache(maxsize=None)
def mixed_coefficients(
    cuts: tuple[Cut, ...],
    weights: tuple[tuple[int, Fraction], ...],
    difference: int,
    source: int,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    coefficients = [ZERO] * 4
    for cut_index, weight in weights:
        pure = pure_cut_coefficients(cuts[cut_index], difference, source)
        for index in range(4):
            coefficients[index] += weight * pure[index]
    return tuple(coefficients)


@lru_cache(maxsize=262_144)
def exp_bounds(value: Interval) -> Interval:
    lower = rational_interval(iv.exp(iv_fraction(value.lower))).lower
    upper = rational_interval(iv.exp(iv_fraction(value.upper))).upper
    return Interval(lower, upper)


@lru_cache(maxsize=262_144)
def log_bounds(value: Interval) -> Interval:
    if value.lower <= ZERO:
        raise ValueError("logarithm interval must be positive")
    lower = rational_interval(iv.log(iv_fraction(value.lower))).lower
    upper = rational_interval(iv.log(iv_fraction(value.upper))).upper
    return Interval(lower, upper)


def conditional_envelope_upper(
    box: Box,
    cuts: tuple[Cut, ...],
    certificates: tuple[Certificate, ...],
) -> Fraction:
    message = Interval(box.lower[0], box.upper[0])
    middle = Interval(box.lower[1], box.upper[1])
    output = Interval(box.lower[2], box.upper[2])
    crossing1 = Interval(box.lower[3], box.upper[3])
    crossing2 = Interval(box.lower[4], box.upper[4])
    second_by_symbols: dict[tuple[int, int], Fraction] = {}
    best: Fraction | None = None
    for middle_symbol in range(1, 5):
        message_conjugate = min(
            first_conjugate_upper(
                cuts,
                certificate.first_weights[middle_symbol - 1],
                certificate.order_multipliers[middle_symbol - 1],
                middle_symbol,
                message,
                middle,
                crossing1,
            )
            for certificate in certificates
        )
        for output_symbol in range(1, 5):
            symbols = (middle_symbol, output_symbol)
            if symbols not in second_by_symbols:
                second_by_symbols[symbols] = second_stage_lower(
                    cuts,
                    middle_symbol,
                    output_symbol,
                    middle,
                    output,
                    crossing2,
                )
            second_cost = second_by_symbols[symbols]
            upper = message_conjugate - second_cost
            best = upper if best is None else max(best, upper)
    if best is None:
        raise RuntimeError("empty conditional envelope")
    return best


def rational_mixture(weights: np.ndarray) -> tuple[Fraction, ...]:
    scale = 1 << 30
    proposed = tuple(
        Fraction(max(0, round(float(weight) * scale)), scale)
        for weight in weights
    )
    total = sum(proposed, ZERO)
    if total == ZERO:
        raise RuntimeError("numerical LP proposed an empty cut mixture")
    return tuple(weight / total for weight in proposed)


def second_stage_mixture(
    cuts: tuple[Cut, ...],
    difference: int,
    source: int,
    increment: Interval,
    state: Interval,
    crossing: Interval,
) -> tuple[Fraction, ...]:
    vertices = support_vertices(increment, state, crossing)
    values = tuple(
        tuple(
            evaluate_affine(
                pure_cut_coefficients(cut, difference, source), vertex
            )
            for vertex in vertices
        )
        for cut in cuts
    )
    cut_count = len(cuts)
    objective = np.zeros(cut_count + 1)
    objective[-1] = -1.0
    upper = np.zeros((len(vertices), cut_count + 1))
    for vertex_index in range(len(vertices)):
        upper[vertex_index, :cut_count] = [
            -float(values[cut_index][vertex_index])
            for cut_index in range(cut_count)
        ]
        upper[vertex_index, -1] = 1.0
    equality = np.zeros((1, cut_count + 1))
    equality[0, :cut_count] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(len(vertices)),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * cut_count + [(None, None)],
        method="highs",
    )
    if result.success:
        return rational_mixture(result.x[:cut_count])
    best_cut = max(range(cut_count), key=lambda index: min(values[index]))
    return tuple(ONE if index == best_cut else ZERO for index in range(cut_count))


def second_stage_lower(
    cuts: tuple[Cut, ...],
    difference: int,
    source: int,
    increment: Interval,
    state: Interval,
    crossing: Interval,
) -> Fraction:
    weights = second_stage_mixture(
        cuts, difference, source, increment, state, crossing
    )
    return min(
        sum(
            (
                weight
                * evaluate_affine(
                    pure_cut_coefficients(cut, difference, source), vertex
                )
                for cut, weight in zip(cuts, weights, strict=True)
            ),
            ZERO,
        )
        for vertex in support_vertices(increment, state, crossing)
    )


def evaluate_affine(
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction],
    vertex: tuple[Fraction, Fraction, Fraction],
) -> Fraction:
    return coefficients[0] + sum(
        (
            coefficient * coordinate
            for coefficient, coordinate in zip(
                coefficients[1:], vertex, strict=True
            )
        ),
        ZERO,
    )


JOINT_CONSTRAINT_ROWS = (
    (-ONE, ZERO, ZERO, ZERO, ZERO),
    (ONE, ZERO, ZERO, ZERO, ZERO),
    (ZERO, -ONE, ZERO, ZERO, ZERO),
    (ZERO, ONE, ZERO, ZERO, ZERO),
    (ZERO, ZERO, -ONE, ZERO, ZERO),
    (ZERO, ZERO, ONE, ZERO, ZERO),
    (ZERO, ZERO, ZERO, -ONE, ZERO),
    (ZERO, ZERO, ZERO, ONE, ZERO),
    (ZERO, ZERO, ZERO, ZERO, -ONE),
    (ZERO, ZERO, ZERO, ZERO, ONE),
    (ONE, -ONE, ZERO, -ONE, ZERO),
    (-ONE, ZERO, ZERO, TWO, ZERO),
    (ZERO, ONE, ZERO, ONE, ZERO),
    (ZERO, ONE, -ONE, ZERO, -ONE),
    (ZERO, -ONE, ZERO, ZERO, TWO),
    (ZERO, ZERO, ONE, ZERO, ONE),
)


def solve_square(
    matrix: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
) -> tuple[Fraction, ...] | None:
    dimension = len(target)
    augmented = [
        list(row) + [value]
        for row, value in zip(matrix, target, strict=True)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] != ZERO
            ),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != ZERO:
                augmented[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return tuple(augmented[row][dimension] for row in range(dimension))


def invert_square(
    matrix: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[Fraction, ...], ...] | None:
    dimension = len(matrix)
    augmented = [
        list(row)
        + [ONE if row_index == column else ZERO for column in range(dimension)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] != ZERO
            ),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != ZERO:
                augmented[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return tuple(
        tuple(augmented[row][dimension:]) for row in range(dimension)
    )


@lru_cache(maxsize=1)
def joint_vertex_bases() -> tuple[
    tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...
]:
    bases = []
    for active in itertools.combinations(range(16), 5):
        matrix = tuple(JOINT_CONSTRAINT_ROWS[index] for index in active)
        inverse = invert_square(matrix)
        if inverse is None:
            continue
        if any(
            (4 * entry).denominator != 1
            for row in inverse
            for entry in row
        ):
            raise RuntimeError("joint vertex inverse exceeds quarter scaling")
        bases.append(
            (
                active,
                tuple(
                    tuple(int(4 * entry) for entry in row)
                    for row in inverse
                ),
            )
        )
    return tuple(bases)


@lru_cache(maxsize=1)
def joint_vertex_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bases = joint_vertex_bases()
    active = np.array([basis[0] for basis in bases], dtype=np.int64)
    inverse = np.array([basis[1] for basis in bases], dtype=np.int64)
    rows = np.array(JOINT_CONSTRAINT_ROWS, dtype=np.int64)
    return active, inverse, rows


@lru_cache(maxsize=16_384)
def joint_support_vertices(box: Box) -> tuple[tuple[Fraction, ...], ...]:
    bounds = (
        -box.lower[0],
        box.upper[0],
        -box.lower[1],
        box.upper[1],
        -box.lower[2],
        box.upper[2],
        -box.lower[3],
        box.upper[3],
        -box.lower[4],
        box.upper[4],
        ZERO,
        ZERO,
        ONE,
        ZERO,
        ZERO,
        ONE,
    )
    common_denominator = math.lcm(
        *(bound.denominator for bound in bounds)
    )
    integer_bounds = tuple(
        bound.numerator * (common_denominator // bound.denominator)
        for bound in bounds
    )
    if max(abs(bound) for bound in integer_bounds) < 1 << 55:
        active, inverse, rows = joint_vertex_arrays()
        integer_array = np.array(integer_bounds, dtype=np.int64)
        targets = integer_array[active]
        solutions = np.einsum("bij,bj->bi", inverse, targets)
        feasible = np.all(
            solutions @ rows.T <= 4 * integer_array,
            axis=1,
        )
        integer_vertices = np.unique(solutions[feasible], axis=0).tolist()
    else:
        integer_rows = tuple(
            tuple(int(coefficient) for coefficient in row)
            for row in JOINT_CONSTRAINT_ROWS
        )
        proposed: set[tuple[int, ...]] = set()
        for active, inverse in joint_vertex_bases():
            target = tuple(integer_bounds[index] for index in active)
            solution = tuple(
                sum(
                    inverse[row][column] * target[column]
                    for column in range(5)
                )
                for row in range(5)
            )
            if all(
                sum(
                    coefficient * coordinate
                    for coefficient, coordinate in zip(
                        row, solution, strict=True
                    )
                )
                <= 4 * bound
                for row, bound in zip(
                    integer_rows, integer_bounds, strict=True
                )
            ):
                proposed.add(solution)
        integer_vertices = sorted(proposed)
    if len(integer_vertices) == 0:
        raise RuntimeError("joint support polytope has no vertices")
    denominator = 4 * common_denominator
    return tuple(
        tuple(Fraction(coordinate, denominator) for coordinate in vertex)
        for vertex in integer_vertices
    )


def first_conjugate_upper(
    cuts: tuple[Cut, ...],
    weights: tuple[tuple[int, Fraction], ...],
    multipliers: tuple[Fraction, ...],
    middle_symbol: int,
    message: Interval,
    middle: Interval,
    crossing: Interval,
) -> Fraction:
    best: Fraction | None = None
    for vertex in support_vertices(message, middle, crossing):
        message_support = vertex[0]
        exponential_sum = Interval(ZERO, ZERO)
        for difference in range(1, 5):
            cost = evaluate_affine(
                mixed_coefficients(
                    cuts, weights, difference, middle_symbol
                ),
                vertex,
            )
            cost += (
                -sum(multipliers, ZERO)
                if difference == 1
                else multipliers[difference - 1]
            )
            exponent = Fraction(-3) * cost / message_support
            exponential_sum += exp_bounds(Interval(exponent, exponent))
        conjugate = (
            message_support * log_bounds(exponential_sum).upper / 3
        )
        best = conjugate if best is None else max(best, conjugate)
    if best is None:
        raise RuntimeError("empty first-stage support polytope")
    return best


def first_conjugate_point_upper(
    cuts: tuple[Cut, ...],
    weights: tuple[tuple[int, Fraction], ...],
    multipliers: tuple[Fraction, ...],
    middle_symbol: int,
    vertex: tuple[Fraction, Fraction, Fraction],
) -> Fraction:
    message_support = vertex[0]
    exponential_sum = Interval(ZERO, ZERO)
    for difference in range(1, 5):
        cost = evaluate_affine(
            mixed_coefficients(cuts, weights, difference, middle_symbol),
            vertex,
        )
        cost += (
            -sum(multipliers, ZERO)
            if difference == 1
            else multipliers[difference - 1]
        )
        exponent = Fraction(-3) * cost / message_support
        exponential_sum += exp_bounds(Interval(exponent, exponent))
    return message_support * log_bounds(exponential_sum).upper / 3


def joint_entropy_cross_coefficients(
    reference: tuple[Fraction, Fraction, Fraction]
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    atoms = (
        ONE - reference[1] - reference[2],
        reference[1] - reference[0] + reference[2],
        reference[2],
        reference[0] - reference[2],
    )
    mixing = Fraction(1, 1 << 40)
    positive = tuple((ONE - mixing) * atom + mixing / 4 for atom in atoms)
    coefficients = tuple(negative_log_upper(atom) for atom in positive)
    return (
        coefficients[0],
        -coefficients[1] + coefficients[3],
        -coefficients[0] + coefficients[1],
        -coefficients[0] + coefficients[1] + coefficients[2] - coefficients[3],
    )


def joint_entropy_cross_upper(
    increment: Interval, state: Interval, crossing: Interval
) -> Fraction:
    vertices = support_vertices(increment, state, crossing)
    reference = tuple(
        sum((vertex[index] for vertex in vertices), ZERO) / len(vertices)
        for index in range(3)
    )
    affine = joint_entropy_cross_coefficients(reference)
    return affine_extrema(affine, increment, state, crossing).upper


def binary_entropy_point_lower(value: Fraction) -> Fraction:
    return entropy_bounds(
        (
            Interval(ONE - value, ONE - value),
            Interval(value, value),
        )
    ).lower


def support_surrogate_data(
    vertices: tuple[tuple[Fraction, ...], ...]
) -> tuple[
    tuple[Fraction, Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction, Fraction],
]:
    reference = tuple(
        sum((vertex[index] for vertex in vertices), ZERO) / len(vertices)
        for index in range(5)
    )
    return (
        joint_entropy_cross_coefficients(
            (reference[0], reference[1], reference[3])
        ),
        joint_entropy_cross_coefficients(
            (reference[1], reference[2], reference[4])
        ),
    )


def support_surrogate_at(
    vertex: tuple[Fraction, ...],
    coefficients: tuple[
        tuple[Fraction, Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction, Fraction],
    ],
) -> Fraction:
    message, middle, output, crossing1, crossing2 = vertex
    value = evaluate_affine(
        coefficients[0], (message, middle, crossing1)
    )
    value += evaluate_affine(
        coefficients[1], (middle, output, crossing2)
    )
    value -= Fraction(2, 3) * binary_entropy_point_lower(message)
    value -= 2 * binary_entropy_point_lower(middle)
    value -= binary_entropy_point_lower(output)
    return value


def binary_entropy_lower(value: Interval) -> Fraction:
    distribution = (
        Interval(ONE - value.upper, ONE - value.lower),
        value,
    )
    return entropy_bounds(distribution).lower


def support_upper(box: Box) -> Fraction:
    message = Interval(box.lower[0], box.upper[0])
    middle = Interval(box.lower[1], box.upper[1])
    output = Interval(box.lower[2], box.upper[2])
    crossing1 = Interval(box.lower[3], box.upper[3])
    crossing2 = Interval(box.lower[4], box.upper[4])
    upper = joint_entropy_cross_upper(message, middle, crossing1)
    upper += joint_entropy_cross_upper(middle, output, crossing2)
    upper -= Fraction(2, 3) * binary_entropy_lower(message)
    upper -= 2 * binary_entropy_lower(middle)
    upper -= binary_entropy_lower(output)
    return upper


def joint_envelope_upper(
    box: Box,
    cuts: tuple[Cut, ...],
    certificates: tuple[Certificate, ...],
) -> Fraction:
    vertices = joint_support_vertices(box)
    support_coefficients = support_surrogate_data(vertices)
    support_values = tuple(
        support_surrogate_at(vertex, support_coefficients)
        for vertex in vertices
    )
    middle = Interval(box.lower[1], box.upper[1])
    output = Interval(box.lower[2], box.upper[2])
    crossing2 = Interval(box.lower[4], box.upper[4])
    second_mixtures = {
        (middle_symbol, output_symbol): second_stage_mixture(
            cuts,
            middle_symbol,
            output_symbol,
            middle,
            output,
            crossing2,
        )
        for middle_symbol in range(1, 5)
        for output_symbol in range(1, 5)
    }
    first_cache: dict[tuple[int, int, tuple[Fraction, ...]], Fraction] = {}
    second_cache: dict[tuple[int, int, tuple[Fraction, ...]], Fraction] = {}
    best: Fraction | None = None
    for middle_symbol in range(1, 5):
        for output_symbol in range(1, 5):
            branch: Fraction | None = None
            weights = second_mixtures[(middle_symbol, output_symbol)]
            for certificate_index, certificate in enumerate(certificates):
                certificate_upper: Fraction | None = None
                for vertex_index, vertex in enumerate(vertices):
                    first_vertex = (vertex[0], vertex[1], vertex[3])
                    first_key = (
                        certificate_index,
                        middle_symbol,
                        first_vertex,
                    )
                    if first_key not in first_cache:
                        first_cache[first_key] = first_conjugate_point_upper(
                            cuts,
                            certificate.first_weights[middle_symbol - 1],
                            certificate.order_multipliers[middle_symbol - 1],
                            middle_symbol,
                            first_vertex,
                        )
                    second_vertex = (vertex[1], vertex[2], vertex[4])
                    second_key = (
                        middle_symbol,
                        output_symbol,
                        second_vertex,
                    )
                    if second_key not in second_cache:
                        second_cache[second_key] = sum(
                            (
                                weight
                                * evaluate_affine(
                                    pure_cut_coefficients(
                                        cut, middle_symbol, output_symbol
                                    ),
                                    second_vertex,
                                )
                                for cut, weight in zip(
                                    cuts, weights, strict=True
                                )
                            ),
                            ZERO,
                        )
                    value = (
                        support_values[vertex_index]
                        + first_cache[first_key]
                        - second_cache[second_key]
                    )
                    certificate_upper = (
                        value
                        if certificate_upper is None
                        else max(certificate_upper, value)
                    )
                if certificate_upper is None:
                    raise RuntimeError("empty joint support polytope")
                branch = (
                    certificate_upper
                    if branch is None
                    else min(branch, certificate_upper)
                )
            if branch is None:
                raise RuntimeError("empty first-stage certificate bank")
            best = branch if best is None else max(best, branch)
    if best is None:
        raise RuntimeError("empty symbol branch set")
    return best


def box_bound(
    raw: Box,
    cuts: tuple[Cut, ...],
    certificates: tuple[Certificate, ...],
    eta: Fraction,
    delta: Fraction,
) -> Bound | None:
    box = tighten(raw, eta, delta)
    if box is None:
        return None
    proposed = local_certificate(box, cuts)
    if proposed is None:
        active_cuts = cuts
        active_certificates = certificates
    else:
        active_cuts, certificate = proposed
        active_certificates = certificates + (certificate,)
    value = joint_envelope_upper(
        box, active_cuts, active_certificates
    )
    scores = []
    for index, (lower, upper) in enumerate(zip(box.lower, box.upper, strict=True)):
        scale_index = index if index < 3 else index - 3
        scale = max(box.upper[scale_index], Fraction(1, 10_000))
        scores.append(float((upper - lower) / scale))
    return Bound(
        value,
        box,
        max(range(5), key=scores.__getitem__),
    )


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def search(args: argparse.Namespace) -> dict[str, object]:
    iv.dps = args.precision
    cuts = default_cuts()
    certificates = reference_certificates()
    initial_box = Box(
        (args.eta, ZERO, ZERO, ZERO, ZERO),
        (4 * args.delta, 2 * args.delta, args.delta, 2 * args.delta, args.delta),
    )
    initial = box_bound(initial_box, cuts, certificates, args.eta, args.delta)
    if initial is None:
        raise RuntimeError("initial eliminated domain is infeasible")
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < ZERO:
            return {
                "status": "verified negative eliminated-composition cover",
                "upper_bound": float(current.value),
                "upper_bound_numerator": str(current.value.numerator),
                "upper_bound_denominator": str(current.value.denominator),
                "upper_bound_is_exactly_negative": True,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
            }
        for child in split(current.box, current.split_dimension):
            bounded = box_bound(child, cuts, certificates, args.eta, args.delta)
            if bounded is None:
                continue
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    worst = queue[0][2] if queue else None
    return {
        "status": "exact eliminated-composition cover incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
        "worst_box": (
            {
                "lower": [str(value) for value in worst.box.lower],
                "upper": [str(value) for value in worst.box.upper],
                "joint_surrogate_upper": float(worst.value),
                "split_dimension": worst.split_dimension,
            }
            if worst is not None
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--delta", type=Fraction, default=Fraction(1323, 10_000))
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 500))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    args = parser.parse_args()
    if args.prime != 5:
        parser.error("this first eliminated checker is specialized to p=5")
    return args


def main() -> None:
    args = parse_args()
    result = search(args)
    result.update(
        {
            "prime": args.prime,
            "repetition": 3,
            "delta": str(args.delta),
            "eta": str(args.eta),
            "interval_precision": args.precision,
            "arithmetic": "exact rationals and directed transcendental intervals",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
