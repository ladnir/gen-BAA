#!/usr/bin/env python3
"""Exact quinary cover in support and conditional-composition coordinates."""

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

from verify_conditional_product_bulk import (
    Cut,
    DualCut,
    Interval,
    ONE,
    TWO,
    ZERO,
    cross_entropy_upper,
    default_cuts,
    default_hybrid_cuts,
    dot_interval,
    entropy_bounds,
    iv_fraction,
    make_cut,
    make_stationary_cut,
    rational_interval,
    support_joint_intervals,
    StationaryCut,
    stationary_pure_coefficients,
)
from verify_support_indicator_bulk import (
    Box as SupportBox,
    box_bound as support_indicator_box_bound,
    search as support_indicator_search,
)


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]


@dataclass(frozen=True)
class Bound:
    value: Fraction
    box: Box
    split_dimension: int
    base_upper: Fraction
    first_payment: Fraction
    second_payment: Fraction
    conditional_upper: Fraction
    support_upper: Fraction


@dataclass(frozen=True)
class ResidualSlab:
    stage: int
    coordinate: int
    sign: int
    threshold: Fraction


def conditional_distribution(box: Box, start: int) -> tuple[Interval, ...]:
    first_three = tuple(
        Interval(box.lower[start + index], box.upper[start + index])
        for index in range(3)
    )
    lower_sum = sum((value.lower for value in first_three), ZERO)
    upper_sum = sum((value.upper for value in first_three), ZERO)
    fourth = Interval(max(ZERO, ONE - upper_sum), min(ONE, ONE - lower_sum))
    return first_three + (fourth,)


def square_distance_lower(interval: Interval, center: Fraction) -> Fraction:
    """Lower-bound (x-center)^2 for x in a rational interval."""
    if interval.lower <= center <= interval.upper:
        return ZERO
    distance = min(abs(interval.lower - center), abs(interval.upper - center))
    return distance * distance


def zero_payment_defect_lower_bound(
    support: Interval,
    crossing: Interval,
    state: tuple[Interval, ...],
) -> Fraction:
    """Lower-bound the squared r-distance from the one-stage zero set.

    Here r_0=x and r_a=(A-x)b_a for nonzero a.  The identity

      ||r-(A/5)1||_2^2
        = (5/4)(x-A/5)^2+(A-x)^2 sum_a (b_a-1/4)^2

    makes the support normalization explicit and avoids division by A.
    """
    crossing_center = Interval(
        crossing.lower - support.upper / 5,
        crossing.upper - support.lower / 5,
    )
    if crossing_center.lower <= ZERO <= crossing_center.upper:
        crossing_square = ZERO
    else:
        crossing_square = min(
            crossing_center.lower * crossing_center.lower,
            crossing_center.upper * crossing_center.upper,
        )
    conditional_square = sum(
        (square_distance_lower(value, Fraction(1, 4)) for value in state),
        ZERO,
    )
    both_nonzero_lower = max(ZERO, support.lower - crossing.upper)
    return Fraction(5, 4) * crossing_square + (
        both_nonzero_lower * both_nonzero_lower * conditional_square
    )


def two_stage_zero_payment_defect_lower_bound(box: Box) -> Fraction:
    mu_support = Interval(box.lower[0], box.upper[0])
    beta_support = Interval(box.lower[1], box.upper[1])
    crossing1 = Interval(box.lower[3], box.upper[3])
    crossing2 = Interval(box.lower[4], box.upper[4])
    beta = conditional_distribution(box, 8)
    zeta = conditional_distribution(box, 11)
    return zero_payment_defect_lower_bound(
        mu_support, crossing1, beta
    ) + zero_payment_defect_lower_bound(beta_support, crossing2, zeta)


def square_distance_upper(interval: Interval, center: Fraction) -> Fraction:
    return max(
        (interval.lower - center) ** 2,
        (interval.upper - center) ** 2,
    )


def zero_payment_defect_upper_bound(
    support: Interval,
    crossing: Interval,
    state: tuple[Interval, ...],
) -> Fraction:
    crossing_center = Interval(
        crossing.lower - support.upper / 5,
        crossing.upper - support.lower / 5,
    )
    crossing_square = square_distance_upper(crossing_center, ZERO)
    conditional_square = sum(
        (
            square_distance_upper(value, Fraction(1, 4))
            for value in state
        ),
        ZERO,
    )
    both_nonzero_upper = support.upper - crossing.lower
    return Fraction(5, 4) * crossing_square + (
        both_nonzero_upper * both_nonzero_upper * conditional_square
    )


def two_stage_zero_payment_defect_upper_bound(box: Box) -> Fraction:
    mu_support = Interval(box.lower[0], box.upper[0])
    beta_support = Interval(box.lower[1], box.upper[1])
    crossing1 = Interval(box.lower[3], box.upper[3])
    crossing2 = Interval(box.lower[4], box.upper[4])
    beta = conditional_distribution(box, 8)
    zeta = conditional_distribution(box, 11)
    return zero_payment_defect_upper_bound(
        mu_support, crossing1, beta
    ) + zero_payment_defect_upper_bound(beta_support, crossing2, zeta)


def rational_sqrt_upper(value: Fraction, bits: int = 48) -> Fraction:
    """Return a dyadic upper bound on the square root of a nonnegative rational."""
    if value < ZERO:
        raise ValueError("square-root input must be nonnegative")
    if value == ZERO:
        return ZERO
    scale = 1 << bits
    quotient = (value.numerator * scale * scale) // value.denominator
    numerator = math.isqrt(quotient)
    candidate = Fraction(numerator, scale)
    if candidate * candidate < value:
        candidate += Fraction(1, scale)
    return candidate


def tighten(
    raw: Box,
    eta: Fraction,
    delta: Fraction,
    maximum_zero_defect: Fraction | None = None,
    minimum_zero_defect: Fraction | None = None,
    residual_slab: ResidualSlab | None = None,
) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    crossing_radius = (
        rational_sqrt_upper(Fraction(4, 5) * maximum_zero_defect)
        if maximum_zero_defect is not None
        else None
    )
    conditional_radius_numerator = (
        rational_sqrt_upper(maximum_zero_defect)
        if maximum_zero_defect is not None
        else None
    )
    for _ in range(12):
        before = (tuple(lower), tuple(upper))
        slab_aggregate_bounds: dict[int, tuple[Fraction, Fraction]] = {}
        # A, B, C and the two stationary crossing masses.
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

        if residual_slab is not None:
            support_index, crossing_index, state_start = (
                (0, 3, 8) if residual_slab.stage == 0 else (1, 4, 11)
            )
            threshold = residual_slab.threshold
            if residual_slab.coordinate == 0:
                if residual_slab.sign > 0:
                    lower[crossing_index] = max(
                        lower[crossing_index],
                        lower[support_index] / 5 + threshold,
                    )
                    upper[support_index] = min(
                        upper[support_index],
                        5 * (upper[crossing_index] - threshold),
                    )
                else:
                    upper[crossing_index] = min(
                        upper[crossing_index],
                        upper[support_index] / 5 - threshold,
                    )
                    lower[support_index] = max(
                        lower[support_index],
                        5 * (lower[crossing_index] + threshold),
                    )
            else:
                both_nonzero_upper = (
                    upper[support_index] - lower[crossing_index]
                )
                both_nonzero_lower = max(
                    lower[support_index] / 2,
                    lower[support_index] - upper[crossing_index],
                )
                if residual_slab.sign > 0:
                    conditional_lower = (
                        lower[support_index] / 5 + threshold
                    ) / both_nonzero_upper
                    conditional_upper = ONE
                else:
                    numerator_upper = (
                        upper[support_index] / 5 - threshold
                    )
                    if numerator_upper < ZERO:
                        return None
                    conditional_lower = ZERO
                    conditional_upper = numerator_upper / both_nonzero_lower
                conditional_lower = max(ZERO, conditional_lower)
                conditional_upper = min(ONE, conditional_upper)
                if conditional_lower > conditional_upper:
                    return None
                if residual_slab.coordinate < 4:
                    state_index = state_start + residual_slab.coordinate - 1
                    lower[state_index] = max(
                        lower[state_index], conditional_lower
                    )
                    upper[state_index] = min(
                        upper[state_index], conditional_upper
                    )
                else:
                    # The fourth conditional coordinate is one minus the
                    # three stored coordinates.
                    slab_aggregate_bounds[state_start] = (
                        ONE - conditional_upper,
                        ONE - conditional_lower,
                    )
            if any(lower[index] > upper[index] for index in range(14)):
                return None

        if crossing_radius is not None:
            for support_index, crossing_index in ((0, 3), (1, 4)):
                lower[crossing_index] = max(
                    lower[crossing_index],
                    lower[support_index] / 5 - crossing_radius,
                )
                upper[crossing_index] = min(
                    upper[crossing_index],
                    upper[support_index] / 5 + crossing_radius,
                )
                lower[support_index] = max(
                    lower[support_index],
                    5 * (lower[crossing_index] - crossing_radius),
                )
                upper[support_index] = min(
                    upper[support_index],
                    5 * (upper[crossing_index] + crossing_radius),
                )
                if lower[support_index] > upper[support_index]:
                    return None
                if lower[crossing_index] > upper[crossing_index]:
                    return None

            assert conditional_radius_numerator is not None
            for support_index, crossing_index, state_start in (
                (0, 3, 8),
                (1, 4, 11),
            ):
                both_nonzero_lower = max(
                    ZERO, lower[support_index] - upper[crossing_index]
                )
                if both_nonzero_lower > ZERO:
                    radius = conditional_radius_numerator / both_nonzero_lower
                    for index in range(state_start, state_start + 3):
                        lower[index] = max(lower[index], Fraction(1, 4) - radius)
                        upper[index] = min(upper[index], Fraction(1, 4) + radius)
                        if lower[index] > upper[index]:
                            return None

        for start in (5, 8, 11):
            if sum(lower[start : start + 3], ZERO) > ONE:
                return None
            aggregate_lower = ZERO
            aggregate_upper = ONE
            if start in slab_aggregate_bounds:
                slab_lower, slab_upper = slab_aggregate_bounds[start]
                aggregate_lower = max(aggregate_lower, slab_lower)
                aggregate_upper = min(aggregate_upper, slab_upper)
            if maximum_zero_defect is not None and start in (8, 11):
                support_index, crossing_index = (
                    (0, 3) if start == 8 else (1, 4)
                )
                both_nonzero_lower = max(
                    ZERO, lower[support_index] - upper[crossing_index]
                )
                if both_nonzero_lower > ZERO:
                    assert conditional_radius_numerator is not None
                    radius = conditional_radius_numerator / both_nonzero_lower
                    aggregate_lower = max(ZERO, Fraction(3, 4) - radius)
                    aggregate_upper = min(ONE, Fraction(3, 4) + radius)
            if sum(upper[start : start + 3], ZERO) < aggregate_lower:
                return None
            if sum(lower[start : start + 3], ZERO) > aggregate_upper:
                return None
            for offset in range(3):
                index = start + offset
                other_lower = sum(lower[start : start + 3], ZERO) - lower[index]
                other_upper = sum(upper[start : start + 3], ZERO) - upper[index]
                lower[index] = max(lower[index], aggregate_lower - other_upper)
                upper[index] = min(upper[index], aggregate_upper - other_lower)
                if lower[index] > upper[index]:
                    return None

        # Canonicalize the largest nonzero message symbol to value one.
        lower[5] = max(
            lower[5],
            Fraction(1, 4),
            lower[6],
            lower[7],
            (ONE - upper[6] - upper[7]) / TWO,
        )
        upper[6] = min(upper[6], upper[5])
        upper[7] = min(upper[7], upper[5])
        if any(lower[index] > upper[index] for index in range(5, 8)):
            return None
        if before == (tuple(lower), tuple(upper)):
            break
    result = Box(tuple(lower), tuple(upper))
    if (
        maximum_zero_defect is not None
        and two_stage_zero_payment_defect_lower_bound(result)
        > maximum_zero_defect
    ):
        return None
    if (
        minimum_zero_defect is not None
        and two_stage_zero_payment_defect_upper_bound(result)
        <= minimum_zero_defect
    ):
        return None
    if residual_slab is not None:
        support_index, crossing_index, state_start = (
            (0, 3, 8) if residual_slab.stage == 0 else (1, 4, 11)
        )
        support = Interval(lower[support_index], upper[support_index])
        crossing = Interval(lower[crossing_index], upper[crossing_index])
        if residual_slab.coordinate == 0:
            residual = crossing - support.scale(Fraction(1, 5))
        else:
            state = conditional_distribution(result, state_start)
            residual = (
                (support - crossing)
                * state[residual_slab.coordinate - 1]
                - support.scale(Fraction(1, 5))
            )
        if residual_slab.sign > 0:
            if residual.upper < residual_slab.threshold:
                return None
        elif residual.lower > -residual_slab.threshold:
            return None
    return result


def conditional_reference_cells(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    joint: tuple[Interval, ...],
) -> tuple[tuple[Interval, ...], ...]:
    zero = Interval(ZERO, ZERO)
    cells = [[zero for _ in range(5)] for _ in range(4)]
    cells[0][0] = joint[0]
    for target in range(1, 5):
        cells[1][target] = joint[1] * state[target - 1]
        cells[2][target] = joint[2] * increment[target - 1]
    for target in range(5):
        convolution = Interval(ZERO, ZERO)
        for difference in range(1, 5):
            source = (target - difference) % 5
            if source != 0:
                convolution += (
                    increment[difference - 1] * state[source - 1]
                )
        cells[3][target] = joint[3] * convolution
    return tuple(tuple(row) for row in cells)


def full_state_distribution(
    support: Interval, conditional: tuple[Interval, ...]
) -> tuple[Interval, ...]:
    return (Interval(ONE - support.upper, ONE - support.lower),) + tuple(
        support * value for value in conditional
    )


def cut_lower_bound(
    cut: DualCut,
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Fraction:
    best: Fraction | None = None
    for increment_vertex in simplex_vertices(increment):
        for state_vertex in simplex_vertices(state):
            if isinstance(cut, StationaryCut):
                coefficients = [ZERO] * 4
                for difference in range(1, 5):
                    for source in range(1, 5):
                        weight = (
                            increment_vertex[difference - 1]
                            * state_vertex[source - 1]
                        )
                        pure = stationary_pure_coefficients(
                            cut, difference, source
                        )
                        for index in range(4):
                            coefficients[index] += weight * pure[index]
                for support_vertex in support_vertices(
                    increment_support, state_support, crossing
                ):
                    lower = coefficients[0] + sum(
                        (
                            coefficient * coordinate
                            for coefficient, coordinate in zip(
                                coefficients[1:], support_vertex, strict=True
                            )
                        ),
                        ZERO,
                    )
                    best = lower if best is None else min(best, lower)
                continue
            target_nonzero = sum(
                cut.target[target] * state_vertex[target - 1]
                for target in range(1, 5)
            )
            row_zero = cut.exponential_minus_one[0][0].upper
            row_state = sum(
                cut.exponential_minus_one[1][target].upper
                * state_vertex[target - 1]
                for target in range(1, 5)
            )
            row_increment = sum(
                cut.exponential_minus_one[2][target].upper
                * increment_vertex[target - 1]
                for target in range(1, 5)
            )
            row_both = ZERO
            for difference in range(1, 5):
                for source in range(1, 5):
                    target = (source + difference) % 5
                    row_both += (
                        cut.exponential_minus_one[3][target].upper
                        * increment_vertex[difference - 1]
                        * state_vertex[source - 1]
                    )
            constant = cut.cell[0] + cut.target[0] - row_zero
            increment_coefficient = (
                -cut.cell[1]
                + cut.cell[3]
                + row_state
                - row_both
            )
            state_coefficient = (
                -cut.cell[0]
                + cut.cell[1]
                + target_nonzero
                - cut.target[0]
                + row_zero
                - row_state
            )
            crossing_coefficient = (
                -cut.cell[0]
                + cut.cell[1]
                + cut.cell[2]
                - cut.cell[3]
                + row_zero
                - row_state
                - row_increment
                + row_both
            )
            for support_vertex in support_vertices(
                increment_support, state_support, crossing
            ):
                lower = constant + sum(
                    (
                        coefficient * coordinate
                        for coefficient, coordinate in zip(
                            (
                                increment_coefficient,
                                state_coefficient,
                                crossing_coefficient,
                            ),
                            support_vertex,
                            strict=True,
                        )
                    ),
                    ZERO,
                )
                best = lower if best is None else min(best, lower)
    if best is None:
        raise RuntimeError("conditional simplex has no vertices")
    return best


def message_eliminated_upper(
    cut: Cut,
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Fraction:
    """Upper-bound A H(mu+)/3 minus a first-stage two-level cut.

    For fixed support data and state law, the cut is affine in the unknown
    nonzero message law u.  Writing its u-dependent correction as
    ``sum_d u_d q_d``, the entropy conjugate eliminates u exactly:

      max_u { A H(u)/3 + sum_d u_d q_d }
        = A/3 log sum_d exp(3 q_d/A).

    All interval operations below are outward directed.  No message box is
    used, so one common cut remains valid for the entire message simplex.
    """
    joint = support_joint_intervals(
        increment_support, state_support, crossing
    )
    full_state = full_state_distribution(state_support, state)
    linear = dot_interval(cut.cell, joint) + dot_interval(cut.target, full_state)

    constant_correction = (
        joint[0] * cut.exponential_minus_one[0][0]
    )
    row_state = Interval(ZERO, ZERO)
    for target in range(1, 5):
        row_state += state[target - 1] * cut.exponential_minus_one[1][target]
    constant_correction += joint[1] * row_state
    constant_lower = linear.lower - constant_correction.upper

    both_nonzero = increment_support - crossing
    scaled_exponentials = Interval(ZERO, ZERO)
    for difference in range(1, 5):
        q = crossing * cut.exponential_minus_one[2][difference]
        convolution_row = Interval(ZERO, ZERO)
        for source in range(1, 5):
            target = (source + difference) % 5
            convolution_row += (
                state[source - 1]
                * cut.exponential_minus_one[3][target]
            )
        q += both_nonzero * convolution_row
        normalized_q = q.divide_positive(increment_support).scale(Fraction(3))
        exponential = rational_interval(iv.exp(iv_fraction(normalized_q.lower)))
        exponential_upper = rational_interval(
            iv.exp(iv_fraction(normalized_q.upper))
        )
        scaled_exponentials += Interval(
            exponential.lower, exponential_upper.upper
        )
    log_sum = Interval(
        rational_interval(
            iv.log(iv_fraction(scaled_exponentials.lower))
        ).lower,
        rational_interval(
            iv.log(iv_fraction(scaled_exponentials.upper))
        ).upper,
    )
    conjugate = (increment_support * log_sum).upper / 3
    return -constant_lower + conjugate


@lru_cache(maxsize=16_384)
def simplex_vertices(
    values: tuple[Interval, ...]
) -> tuple[tuple[Fraction, ...], ...]:
    vertices: set[tuple[Fraction, ...]] = set()
    for free_index in range(4):
        fixed = [index for index in range(4) if index != free_index]
        for choices in itertools.product((0, 1), repeat=3):
            vertex = [ZERO] * 4
            for index, choice in zip(fixed, choices, strict=True):
                vertex[index] = (
                    values[index].upper if choice else values[index].lower
                )
            vertex[free_index] = ONE - sum(
                (vertex[index] for index in fixed), ZERO
            )
            if (
                values[free_index].lower
                <= vertex[free_index]
                <= values[free_index].upper
            ):
                vertices.add(tuple(vertex))
    return tuple(sorted(vertices))


@lru_cache(maxsize=16_384)
def support_vertices(
    increment: Interval, state: Interval, crossing: Interval
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    # Every constraint has the form a*A+b*B+c*x <= d.
    constraints = (
        ((-ONE, ZERO, ZERO), -increment.lower),
        ((ONE, ZERO, ZERO), increment.upper),
        ((ZERO, -ONE, ZERO), -state.lower),
        ((ZERO, ONE, ZERO), state.upper),
        ((ZERO, ZERO, -ONE), -crossing.lower),
        ((ZERO, ZERO, ONE), crossing.upper),
        ((ONE, -ONE, -ONE), ZERO),
        ((-ONE, ZERO, TWO), ZERO),
        ((ZERO, ONE, ONE), ONE),
    )
    vertices: set[tuple[Fraction, Fraction, Fraction]] = set()
    for active in itertools.combinations(constraints, 3):
        matrix = tuple(row for row, _ in active)
        target = tuple(bound for _, bound in active)
        solution = solve_three_by_three(matrix, target)
        if solution is None:
            continue
        if all(
            sum(
                (coefficient * coordinate for coefficient, coordinate in zip(row, solution, strict=True)),
                ZERO,
            )
            <= bound
            for row, bound in constraints
        ):
            vertices.add(solution)
    if not vertices:
        raise RuntimeError("support polytope has no vertices")
    return tuple(sorted(vertices))


def solve_three_by_three(
    matrix: tuple[tuple[Fraction, Fraction, Fraction], ...],
    target: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction] | None:
    augmented = [list(row) + [value] for row, value in zip(matrix, target, strict=True)]
    for column in range(3):
        pivot = next(
            (row for row in range(column, 3) if augmented[row][column] != ZERO),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(3):
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
    return tuple(augmented[row][3] for row in range(3))


def binary_divergence_at(probability: Fraction, reference: Fraction) -> Fraction:
    if probability == reference:
        return ZERO
    value = iv.mpf(0)
    if probability > ZERO:
        if reference <= ZERO:
            return Fraction(10**9)
        value += iv_fraction(probability) * iv.log(
            iv_fraction(probability) / iv_fraction(reference)
        )
    if probability < ONE:
        if reference >= ONE:
            return Fraction(10**9)
        value += iv_fraction(ONE - probability) * iv.log(
            iv_fraction(ONE - probability) / iv_fraction(ONE - reference)
        )
    return rational_interval(value).lower


def cancellation_lower_bound(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    crossing: Interval,
) -> Fraction:
    both_nonzero = increment_support - crossing
    if both_nonzero.lower <= ZERO:
        return ZERO
    raw_observed = crossing.divide_positive(both_nonzero)
    observed = Interval(
        max(ZERO, raw_observed.lower), min(ONE, raw_observed.upper)
    )
    cancellation = Interval(ZERO, ZERO)
    for difference in range(1, 5):
        cancellation += (
            increment[difference - 1] * state[(-difference) % 5 - 1]
        )
    cancellation = Interval(
        max(ZERO, cancellation.lower), min(ONE, cancellation.upper)
    )
    if observed.upper < cancellation.lower:
        divergence = binary_divergence_at(
            observed.upper, cancellation.lower
        )
    elif cancellation.upper < observed.lower:
        divergence = binary_divergence_at(
            observed.lower, cancellation.upper
        )
    else:
        divergence = ZERO
    return both_nonzero.lower * divergence


def target_pinsker_lower_bound(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Fraction:
    joint = support_joint_intervals(
        increment_support, state_support, crossing
    )
    reference = conditional_reference_cells(increment, state, joint)
    reference_target = []
    for target in range(5):
        total = Interval(ZERO, ZERO)
        for row in range(4):
            total += reference[row][target]
        reference_target.append(total)
    actual_target = full_state_distribution(state_support, state)
    variation_lower = ZERO
    for actual, proposed in zip(actual_target, reference_target, strict=True):
        if actual.upper < proposed.lower:
            variation_lower += proposed.lower - actual.upper
        elif proposed.upper < actual.lower:
            variation_lower += actual.lower - proposed.upper
    variation_lower /= TWO
    return TWO * variation_lower * variation_lower


def nearby_cuts(
    cuts: tuple[DualCut, ...], support: Interval
) -> tuple[DualCut, ...]:
    midpoint = (support.lower + support.upper) / TWO
    return tuple(
        sorted(
            cuts,
            key=lambda cut: abs(cut.reference_support - midpoint),
        )[:8]
    )


def reference_conditional(values: tuple[Interval, ...]) -> np.ndarray:
    midpoint = np.array(
        [float((value.lower + value.upper) / TWO) for value in values]
    )
    midpoint = np.maximum(midpoint, 0.0)
    midpoint /= midpoint.sum()
    mixing = 1e-7
    return (1.0 - mixing) * midpoint + mixing / len(midpoint)


def local_cut(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Cut | None:
    support_increment = float(
        (increment_support.lower + increment_support.upper) / TWO
    )
    support_state = float((state_support.lower + state_support.upper) / TWO)
    lower_crossing = max(
        float(crossing.lower), 0.0, support_increment - support_state
    )
    upper_crossing = min(
        float(crossing.upper), support_increment / 2.0, 1.0 - support_state
    )
    if lower_crossing >= upper_crossing:
        return None
    crossing_reference = 0.5 * (lower_crossing + upper_crossing)
    increment_reference = np.concatenate(
        (
            np.array([1.0 - support_increment]),
            support_increment * reference_conditional(increment),
        )
    )
    state_reference = np.concatenate(
        (
            np.array([1.0 - support_state]),
            support_state * reference_conditional(state),
        )
    )
    try:
        return make_cut(
            increment_reference, state_reference, crossing_reference
        )
    except ValueError:
        return None


def local_stationary_cut(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> StationaryCut | None:
    support_increment = float(
        (increment_support.lower + increment_support.upper) / TWO
    )
    support_state = float((state_support.lower + state_support.upper) / TWO)
    lower_crossing = max(
        float(crossing.lower), 0.0, support_increment - support_state
    )
    upper_crossing = min(
        float(crossing.upper), support_increment / 2.0, 1.0 - support_state
    )
    if lower_crossing >= upper_crossing:
        return None
    crossing_reference = 0.5 * (lower_crossing + upper_crossing)
    increment_reference = np.concatenate(
        (
            np.array([1.0 - support_increment]),
            support_increment * reference_conditional(increment),
        )
    )
    state_reference = np.concatenate(
        (
            np.array([1.0 - support_state]),
            support_state * reference_conditional(state),
        )
    )
    try:
        return make_stationary_cut(
            increment_reference, state_reference, crossing_reference
        )
    except ValueError:
        return None


def binary_entropy_lower(support: Interval) -> Fraction:
    return entropy_bounds(
        (Interval(ONE - support.upper, ONE - support.lower), support)
    ).lower


def box_bound(
    box: Box,
    cuts: tuple[DualCut, ...],
    include_stationary: bool = True,
    maximum_zero_defect: Fraction | None = None,
    residual_slab: ResidualSlab | None = None,
    support_upper_cap: Fraction | None = None,
) -> Bound:
    support_mu = Interval(box.lower[0], box.upper[0])
    support_beta = Interval(box.lower[1], box.upper[1])
    support_zeta = Interval(box.lower[2], box.upper[2])
    crossing1 = Interval(box.lower[3], box.upper[3])
    crossing2 = Interval(box.lower[4], box.upper[4])
    mu = conditional_distribution(box, 5)
    beta = conditional_distribution(box, 8)
    zeta = conditional_distribution(box, 11)
    first_joint = support_joint_intervals(
        support_mu, support_beta, crossing1
    )
    second_joint = support_joint_intervals(
        support_beta, support_zeta, crossing2
    )

    internal_message_upper = (
        support_mu * Interval(ZERO, cross_entropy_upper(mu))
    ).upper
    base_upper = internal_message_upper / 3
    base_upper += cross_entropy_upper(first_joint)
    base_upper += cross_entropy_upper(second_joint)
    base_upper -= Fraction(2, 3) * binary_entropy_lower(support_mu)
    base_upper -= 2 * binary_entropy_lower(support_beta)
    base_upper -= binary_entropy_lower(support_zeta)

    first_local = local_cut(
        mu, beta, support_mu, support_beta, crossing1
    )
    second_local = local_cut(
        beta, zeta, support_beta, support_zeta, crossing2
    )
    first_stationary = (
        local_stationary_cut(mu, beta, support_mu, support_beta, crossing1)
        if include_stationary
        else None
    )
    second_stationary = (
        local_stationary_cut(beta, zeta, support_beta, support_zeta, crossing2)
        if include_stationary
        else None
    )
    first_cuts = nearby_cuts(cuts, support_mu) + (
        (first_local,) if first_local is not None else ()
    ) + (
        (first_stationary,) if first_stationary is not None else ()
    )
    second_cuts = nearby_cuts(cuts, support_beta) + (
        (second_local,) if second_local is not None else ()
    ) + (
        (second_stationary,) if second_stationary is not None else ()
    )

    first_payment = (
        max(
            ZERO,
            cancellation_lower_bound(
                mu, beta, support_mu, crossing1
            ),
            target_pinsker_lower_bound(
                mu,
                beta,
                support_mu,
                support_beta,
                crossing1,
            ),
            *(
                cut_lower_bound(
                    cut,
                    mu,
                    beta,
                    support_mu,
                    support_beta,
                    crossing1,
                )
                for cut in first_cuts
            ),
        )
        if include_stationary
        else ZERO
    )
    second_payment = max(
        ZERO,
        cancellation_lower_bound(
            beta, zeta, support_beta, crossing2
        ),
        target_pinsker_lower_bound(
            beta,
            zeta,
            support_beta,
            support_zeta,
            crossing2,
        ),
        *(
            cut_lower_bound(
                cut,
                beta,
                zeta,
                support_beta,
                support_zeta,
                crossing2,
            )
            for cut in second_cuts
        ),
    )
    if include_stationary:
        conditional_upper = (
            internal_message_upper / 3 - first_payment - second_payment
        )
        direct_upper = base_upper - first_payment - second_payment
    else:
        first_eliminated_upper = min(
            message_eliminated_upper(
                cut,
                beta,
                support_mu,
                support_beta,
                crossing1,
            )
            for cut in first_cuts
            if isinstance(cut, Cut)
        )
        conditional_upper = first_eliminated_upper - second_payment
        support_base_upper = base_upper - internal_message_upper / 3
        direct_upper = support_base_upper + conditional_upper
    support_result = support_indicator_box_bound(
        SupportBox(box.lower[:3], box.upper[:3]),
        5,
        3,
        True,
    )
    if support_result is None:
        raise RuntimeError("conditional box has infeasible support projection")
    support_upper = support_result[0]
    if support_upper_cap is not None:
        support_upper = min(support_upper, support_upper_cap)
    value = min(
        direct_upper,
        support_upper + conditional_upper,
    )
    scores = []
    for index, (lower, upper) in enumerate(
        zip(box.lower, box.upper, strict=True)
    ):
        width = upper - lower
        if index <= 2:
            scale = max(upper, Fraction(1, 10_000))
            priority = 4.0
        elif index <= 4:
            scale = max(box.upper[index - 3], Fraction(1, 10_000))
            priority = 2.0
        elif not include_stationary and 5 <= index < 8:
            # The first-stage entropy conjugate eliminated the message law.
            scores.append(0.0)
            continue
        else:
            scale = ONE
            priority = 1.0
        scores.append(priority * float(width / scale))
    if residual_slab is not None:
        support_index, crossing_index, state_start = (
            (0, 3, 8) if residual_slab.stage == 0 else (1, 4, 11)
        )
        scores[support_index] *= 1.5
        scores[crossing_index] *= 2.5
        if residual_slab.coordinate == 4:
            for index in range(state_start, state_start + 3):
                scores[index] *= 5.0
        elif residual_slab.coordinate > 0:
            scores[state_start + residual_slab.coordinate - 1] *= 5.0
    return Bound(
        value,
        box,
        int(np.argmax(scores)),
        base_upper,
        first_payment,
        second_payment,
        conditional_upper,
        support_upper,
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
    include_stationary = args.projection == "hybrid"
    cuts = default_hybrid_cuts() if include_stationary else default_cuts()
    lower = [args.eta, ZERO, ZERO, ZERO, ZERO] + [ZERO] * 9
    maximum_message_support = (
        args.message_support_upper
        if args.message_support_upper is not None
        else 4 * args.delta
    )
    support_cap_certificate = None
    if args.support_upper_cap is not None:
        if args.message_support_upper is None:
            raise ValueError(
                "a support upper cap requires a finite message-support upper bound"
            )
        support_cap_certificate = support_indicator_search(
            argparse.Namespace(
                eta=args.eta,
                cutoff=args.message_support_upper,
                delta=args.delta,
                prime=5,
                repetition=3,
                precision=args.precision,
                max_boxes=args.support_cap_max_boxes,
                internal_values_paid=False,
                target_upper_bound=args.support_upper_cap,
            )
        )
        if not support_cap_certificate.get("upper_bound_is_below_target"):
            raise RuntimeError("the requested support upper cap was not certified")
    upper = [min(4 * args.delta, maximum_message_support), 2 * args.delta, args.delta]
    upper += [2 * args.delta, args.delta] + [ONE] * 9
    initial_box = tighten(
        Box(tuple(lower), tuple(upper)),
        args.eta,
        args.delta,
        args.maximum_zero_defect,
        args.minimum_zero_defect,
        args.residual_slab,
    )
    if initial_box is None:
        raise RuntimeError("initial conditional-composition domain is infeasible")
    initial = box_bound(
        initial_box,
        cuts,
        include_stationary,
        args.maximum_zero_defect,
        args.residual_slab,
        args.support_upper_cap,
    )
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < ZERO:
            result = {
                "status": "verified negative reduced-composition cover",
                "upper_bound": float(current.value),
                "upper_bound_numerator": str(current.value.numerator),
                "upper_bound_denominator": str(current.value.denominator),
                "upper_bound_is_exactly_negative": True,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
                "dual_cuts": len(cuts),
            }
            if support_cap_certificate is not None:
                result["support_cap_certificate"] = support_cap_certificate
            return result
        for child in split(current.box, current.split_dimension):
            tightened = tighten(
                child,
                args.eta,
                args.delta,
                args.maximum_zero_defect,
                args.minimum_zero_defect,
                args.residual_slab,
            )
            if tightened is None:
                continue
            bounded = box_bound(
                tightened,
                cuts,
                include_stationary,
                args.maximum_zero_defect,
                args.residual_slab,
                args.support_upper_cap,
            )
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    worst = queue[0][2] if queue else None
    return {
        "status": "exact reduced-composition cover incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
        "dual_cuts": len(cuts),
        "worst_box": (
            {
                "lower": [str(value) for value in worst.box.lower],
                "upper": [str(value) for value in worst.box.upper],
                "base_upper": float(worst.base_upper),
                "first_payment": float(worst.first_payment),
                "second_payment": float(worst.second_payment),
                "conditional_upper": float(worst.conditional_upper),
                "support_upper": float(worst.support_upper),
                "support_plus_conditional_upper": float(
                    worst.support_upper + worst.conditional_upper
                ),
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
    parser.add_argument(
        "--message-support-upper",
        type=Fraction,
        default=None,
        help="optionally restrict the cover to A at most this rational value",
    )
    parser.add_argument(
        "--support-upper-cap",
        type=Fraction,
        default=None,
        help=(
            "use this global support-only upper after certifying it on the "
            "message-support band"
        ),
    )
    parser.add_argument("--support-cap-max-boxes", type=int, default=100_000)
    parser.add_argument(
        "--projection",
        choices=("two-level", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--maximum-zero-defect",
        type=Fraction,
        default=None,
        help=(
            "restrict the exact cover to the normalized two-stage tube "
            "R_1+R_2 at most this rational value"
        ),
    )
    parser.add_argument(
        "--minimum-zero-defect",
        type=Fraction,
        default=None,
        help=(
            "restrict the exact cover to the complement R_1+R_2 greater "
            "than this rational value"
        ),
    )
    parser.add_argument(
        "--residual-slab",
        help=(
            "restrict to one signed raw-residual slab STAGE:COORDINATE:SIGN; "
            "stages are 1 or 2, coordinates are 0 through 4, and signs are + or -"
        ),
    )
    parser.add_argument(
        "--residual-threshold",
        type=Fraction,
        default=Fraction(1, 3200),
    )
    args = parser.parse_args()
    if args.prime != 5:
        parser.error("the first reduced-composition verifier is specialized to p=5")
    if (
        args.maximum_zero_defect is not None
        and args.minimum_zero_defect is not None
    ):
        parser.error("choose either a maximum or a minimum zero defect")
    if (
        args.message_support_upper is not None
        and args.message_support_upper < args.eta
    ):
        parser.error("message support upper bound must be at least eta")
    if args.residual_slab is not None:
        try:
            stage_text, coordinate_text, sign_text = args.residual_slab.split(":")
            stage = int(stage_text) - 1
            coordinate = int(coordinate_text)
            sign = 1 if sign_text == "+" else -1 if sign_text == "-" else 0
        except (TypeError, ValueError) as error:
            parser.error(f"invalid residual slab: {error}")
        if stage not in (0, 1) or coordinate not in range(5) or sign == 0:
            parser.error("invalid residual slab")
        args.residual_slab = ResidualSlab(
            stage, coordinate, sign, args.residual_threshold
        )
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
            "projection": args.projection,
            "message_support_upper": (
                str(args.message_support_upper)
                if args.message_support_upper is not None
                else None
            ),
            "support_upper_cap": (
                str(args.support_upper_cap)
                if args.support_upper_cap is not None
                else None
            ),
            "maximum_zero_defect": (
                str(args.maximum_zero_defect)
                if args.maximum_zero_defect is not None
                else None
            ),
            "minimum_zero_defect": (
                str(args.minimum_zero_defect)
                if args.minimum_zero_defect is not None
                else None
            ),
            "residual_slab": (
                {
                    "stage": args.residual_slab.stage + 1,
                    "coordinate": args.residual_slab.coordinate,
                    "sign": "+" if args.residual_slab.sign > 0 else "-",
                    "threshold": str(args.residual_slab.threshold),
                }
                if args.residual_slab is not None
                else None
            ),
            "arithmetic": "exact rational boxes and directed transcendental intervals",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
