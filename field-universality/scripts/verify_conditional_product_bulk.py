#!/usr/bin/env python3
"""Exact interval cover for the quinary conditional-product bulk bound.

Floating-point data is used only to choose rational KL-dual potentials.
Every retained box bound uses exact rational arithmetic and directed
intervals for logarithms and exponentials.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import cast

import numpy as np
from mpmath import iv

import optimize_conditional_product_bound as product


ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)
COEFFICIENT_BITS = 48
POTENTIAL_BITS = 30


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    def divide_positive(self, other: "Interval") -> "Interval":
        if other.lower <= ZERO:
            raise ValueError("positive interval division requires a positive lower bound")
        return Interval(self.lower / other.upper, self.upper / other.lower)

    def scale(self, coefficient: Fraction) -> "Interval":
        if coefficient >= ZERO:
            return Interval(
                coefficient * self.lower, coefficient * self.upper
            )
        return Interval(coefficient * self.upper, coefficient * self.lower)


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]
    support_lower: tuple[Fraction, ...] = ()
    support_upper: tuple[Fraction, ...] = ()


@dataclass(frozen=True)
class Cut:
    cell: tuple[Fraction, ...]
    target: tuple[Fraction, ...]
    exponential_minus_one: tuple[tuple[Interval, ...], ...]
    reference_support: Fraction


@dataclass(frozen=True)
class StationaryCut:
    source: tuple[Fraction, ...]
    target: tuple[Fraction, ...]
    exponential_minus_one: tuple[tuple[Interval, ...], ...]
    reference_support: Fraction


DualCut = Cut | StationaryCut


@dataclass(frozen=True)
class Bound:
    value: Fraction
    box: Box
    split_dimension: int
    base_upper: Fraction
    first_payment: Fraction
    second_payment: Fraction


def mpf_fraction(raw: tuple[int, int, int, int]) -> Fraction:
    sign, mantissa, exponent, _ = raw
    value = Fraction(mantissa)
    value = value * 2**exponent if exponent >= 0 else value / 2 ** (-exponent)
    return -value if sign else value


def rational_interval(value: object) -> Interval:
    scale = 1 << COEFFICIENT_BITS

    def round_down(exact: Fraction) -> Fraction:
        return Fraction(exact.numerator // exact.denominator, 1) / scale

    def round_up(exact: Fraction) -> Fraction:
        return Fraction(-((-exact.numerator) // exact.denominator), 1) / scale

    lower = mpf_fraction(value._mpi_[0]) * scale
    upper = mpf_fraction(value._mpi_[1]) * scale
    return Interval(round_down(lower), round_up(upper))


def iv_fraction(value: Fraction) -> object:
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


@lru_cache(maxsize=None)
def entropy_atom_at(value: Fraction) -> Interval:
    if value == ZERO:
        return Interval(ZERO, ZERO)
    point = iv_fraction(value)
    return rational_interval(-point * iv.log(point))


@lru_cache(maxsize=None)
def negative_log_upper(value: Fraction) -> Fraction:
    if value <= ZERO:
        raise ValueError("logarithm reference must be positive")
    return rational_interval(-iv.log(iv_fraction(value))).upper


@lru_cache(maxsize=None)
def inverse_e_interval() -> Interval:
    return rational_interval(iv.mpf(1) / iv.e)


def entropy_atom_bounds(interval: Interval) -> Interval:
    left = entropy_atom_at(interval.lower)
    right = entropy_atom_at(interval.upper)
    lower = min(left.lower, right.lower)
    upper = max(left.upper, right.upper)
    inverse_e = inverse_e_interval()
    if interval.lower <= inverse_e.upper and inverse_e.lower <= interval.upper:
        upper = max(upper, inverse_e.upper)
    return Interval(lower, upper)


def entropy_bounds(values: tuple[Interval, ...]) -> Interval:
    total = Interval(ZERO, ZERO)
    for value in values:
        total += entropy_atom_bounds(value)
    return total


def linear_simplex_maximum(
    coefficients: tuple[Fraction, ...],
    values: tuple[Interval, ...],
) -> Fraction:
    masses = [value.lower for value in values]
    remaining = ONE - sum(masses, ZERO)
    if remaining < ZERO:
        raise ValueError("inconsistent simplex lower bounds")
    for index in sorted(
        range(len(values)), key=lambda offset: coefficients[offset], reverse=True
    ):
        addition = min(remaining, values[index].upper - masses[index])
        masses[index] += addition
        remaining -= addition
    if remaining > ZERO:
        raise ValueError("simplex upper bounds do not reach total mass one")
    return sum(
        coefficient * mass
        for coefficient, mass in zip(coefficients, masses, strict=True)
    )


def cross_entropy_upper(values: tuple[Interval, ...]) -> Fraction:
    midpoint = [
        (value.lower + value.upper) / TWO for value in values
    ]
    total = sum(midpoint, ZERO)
    if total <= ZERO:
        midpoint = [ONE / len(values)] * len(values)
    else:
        midpoint = [value / total for value in midpoint]
    mixing = Fraction(1, 1 << 32)
    reference = tuple(
        (ONE - mixing) * value + mixing / len(values)
        for value in midpoint
    )
    coefficients = tuple(negative_log_upper(value) for value in reference)
    return linear_simplex_maximum(coefficients, values)


def dot_interval(
    coefficients: tuple[Fraction, ...], values: tuple[Interval, ...]
) -> Interval:
    total = Interval(ZERO, ZERO)
    for coefficient, value in zip(coefficients, values, strict=True):
        total += value.scale(coefficient)
    return total


def sum_interval(values: tuple[Interval, ...]) -> Interval:
    total = Interval(ZERO, ZERO)
    for value in values:
        total += value
    return total


def composition_intervals(
    box: Box, start: int, support: Interval
) -> tuple[Interval, ...]:
    nonzero = tuple(
        Interval(box.lower[start + index], box.upper[start + index])
        for index in range(4)
    )
    return (Interval(ONE - support.upper, ONE - support.lower),) + nonzero


def support_joint_intervals(
    increment: Interval, state: Interval, crossing: Interval
) -> tuple[Interval, ...]:
    raw = (
        Interval(
            ONE - state.upper - crossing.upper,
            ONE - state.lower - crossing.lower,
        ),
        Interval(
            state.lower - increment.upper + crossing.lower,
            state.upper - increment.lower + crossing.upper,
        ),
        crossing,
        Interval(
            increment.lower - crossing.upper,
            increment.upper - crossing.lower,
        ),
    )
    return tuple(
        Interval(max(ZERO, value.lower), min(ONE, value.upper))
        for value in raw
    )


def tighten(raw: Box, eta: Fraction, delta: Fraction) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    total_lower = (
        list(raw.support_lower)
        if raw.support_lower
        else [ZERO, ZERO, ZERO]
    )
    total_upper = (
        list(raw.support_upper)
        if raw.support_upper
        else [ONE, ONE, delta]
    )
    for _ in range(12):
        before = (tuple(lower), tuple(upper), tuple(total_lower), tuple(total_upper))
        for component in range(3):
            start = 4 * component
            total_lower[component] = max(
                total_lower[component], sum(lower[start : start + 4], ZERO)
            )
            total_upper[component] = min(
                total_upper[component], sum(upper[start : start + 4], ZERO)
            )
        total_lower[0] = max(total_lower[0], eta)
        # Multiplication by a nonzero field element is transitive on the four
        # nonzero symbols.  Work on the canonical sector where mu_1 is largest.
        lower[0] = max(lower[0], *(lower[index] for index in range(1, 4)))
        for index in range(1, 4):
            upper[index] = min(upper[index], upper[0])
        lower[0] = max(lower[0], total_lower[0] / 4)
        total_upper[0] = min(total_upper[0], 4 * upper[0])
        total_upper[0] = min(total_upper[0], TWO * total_upper[1])
        total_lower[1] = max(total_lower[1], total_lower[0] / TWO)
        total_upper[1] = min(total_upper[1], TWO * total_upper[2])
        total_lower[2] = max(total_lower[2], total_lower[1] / TWO)
        if any(total_lower[index] > total_upper[index] for index in range(3)):
            return None
        for component in range(3):
            start = 4 * component
            for offset in range(4):
                index = start + offset
                other_lower = sum(lower[start : start + 4], ZERO) - lower[index]
                other_upper = sum(upper[start : start + 4], ZERO) - upper[index]
                lower[index] = max(
                    lower[index], total_lower[component] - other_upper
                )
                upper[index] = min(
                    upper[index], total_upper[component] - other_lower
                )
                if lower[index] > upper[index]:
                    return None
        for stage, (increment_index, state_index, crossing_index) in enumerate(
            ((0, 1, 12), (1, 2, 13))
        ):
            del stage
            increment_lower = total_lower[increment_index]
            increment_upper = total_upper[increment_index]
            state_lower = total_lower[state_index]
            state_upper = total_upper[state_index]
            lower[crossing_index] = max(
                lower[crossing_index], ZERO, increment_lower - state_upper
            )
            upper[crossing_index] = min(
                upper[crossing_index],
                increment_upper / TWO,
                ONE - state_lower,
            )
            total_lower[increment_index] = max(
                total_lower[increment_index], TWO * lower[crossing_index]
            )
            total_upper[increment_index] = min(
                total_upper[increment_index],
                state_upper + upper[crossing_index],
            )
            total_lower[state_index] = max(
                total_lower[state_index],
                increment_lower - upper[crossing_index],
            )
            total_upper[state_index] = min(
                total_upper[state_index], ONE - lower[crossing_index]
            )
            if lower[crossing_index] > upper[crossing_index]:
                return None
        after = (tuple(lower), tuple(upper), tuple(total_lower), tuple(total_upper))
        if after == before:
            break
    return Box(
        tuple(lower),
        tuple(upper),
        tuple(total_lower),
        tuple(total_upper),
    )


def reference_cell_targets(
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> tuple[tuple[Interval, ...], ...]:
    joint = support_joint_intervals(
        increment_support, state_support, crossing
    )
    zero = Interval(ZERO, ZERO)
    cells = [[zero for _ in range(5)] for _ in range(4)]
    cells[0][0] = joint[0]
    for target in range(1, 5):
        cells[1][target] = (
            joint[1] * state[target]
        ).divide_positive(state_support)
        cells[2][target] = (
            joint[2] * increment[target]
        ).divide_positive(increment_support)
    denominator = increment_support * state_support
    for target in range(5):
        convolution = Interval(ZERO, ZERO)
        for difference in range(1, 5):
            source = (target - difference) % 5
            if source != 0:
                convolution += increment[difference] * state[source]
        cells[3][target] = (
            joint[3] * convolution
        ).divide_positive(denominator)
    return tuple(tuple(row) for row in cells)


def cut_lower_bound(
    cut: Cut,
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Fraction:
    joint = support_joint_intervals(
        increment_support, state_support, crossing
    )
    value = dot_interval(cut.cell, joint) + dot_interval(cut.target, state)
    reference = reference_cell_targets(
        increment,
        state,
        increment_support,
        state_support,
        crossing,
    )
    correction = Interval(ZERO, ZERO)
    for row in range(4):
        for target in range(5):
            correction += reference[row][target] * cut.exponential_minus_one[
                row
            ][target]
    return (value - correction).lower


def box_bound(box: Box, cuts: tuple[Cut, ...]) -> Bound:
    support_mu = Interval(box.support_lower[0], box.support_upper[0])
    support_beta = Interval(box.support_lower[1], box.support_upper[1])
    support_zeta = Interval(box.support_lower[2], box.support_upper[2])
    mu = composition_intervals(box, 0, support_mu)
    beta = composition_intervals(box, 4, support_beta)
    zeta = composition_intervals(box, 8, support_zeta)
    first_crossing = Interval(box.lower[12], box.upper[12])
    second_crossing = Interval(box.lower[13], box.upper[13])
    first_joint = support_joint_intervals(
        support_mu, support_beta, first_crossing
    )
    second_joint = support_joint_intervals(
        support_beta, support_zeta, second_crossing
    )
    base_upper = cross_entropy_upper(mu) / 3
    base_upper += cross_entropy_upper(first_joint)
    base_upper += cross_entropy_upper(second_joint)
    base_upper -= entropy_bounds(
        (Interval(ONE - support_mu.upper, ONE - support_mu.lower), support_mu)
    ).lower
    base_upper -= TWO * entropy_bounds(
        (Interval(ONE - support_beta.upper, ONE - support_beta.lower), support_beta)
    ).lower
    base_upper -= entropy_bounds(
        (Interval(ONE - support_zeta.upper, ONE - support_zeta.lower), support_zeta)
    ).lower
    def nearby_cuts(support: Interval) -> tuple[Cut, ...]:
        midpoint = (support.lower + support.upper) / TWO
        return tuple(
            sorted(
                cuts,
                key=lambda cut: abs(cut.reference_support - midpoint),
            )[:8]
        )

    first_payment = max(
        ZERO,
        *(
            cut_lower_bound(
                cut,
                mu,
                beta,
                support_mu,
                support_beta,
                first_crossing,
            )
            for cut in nearby_cuts(support_mu)
        ),
    )
    second_payment = max(
        ZERO,
        *(
            cut_lower_bound(
                cut,
                beta,
                zeta,
                support_beta,
                support_zeta,
                second_crossing,
            )
            for cut in nearby_cuts(support_beta)
        ),
    )
    upper = base_upper - first_payment - second_payment
    scores = []
    for index, (lower, upper_endpoint) in enumerate(
        zip(box.lower, box.upper, strict=True)
    ):
        width = upper_endpoint - lower
        if index < 12:
            component = index // 4
            support_upper = (support_mu, support_beta, support_zeta)[component].upper
            scale = max(support_upper, Fraction(1, 10_000))
            scores.append(float(width / scale))
        else:
            scores.append(float(width / max(support_mu.upper, Fraction(1, 10_000))))
    for support in (support_mu, support_beta, support_zeta):
        scores.append(
            float(
                (support.upper - support.lower)
                / max(support.upper, Fraction(1, 10_000))
            )
        )
    return Bound(
        upper,
        box,
        int(np.argmax(scores)),
        base_upper,
        first_payment,
        second_payment,
    )


def rational_potential(value: float) -> Fraction:
    scale = 1 << POTENTIAL_BITS
    return Fraction(round(value * scale), scale)


def make_cut(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> Cut:
    cell_float, target_float = product.two_level_dual_potentials(
        increment, state, crossing
    )
    cell = tuple(rational_potential(value) for value in cell_float)
    target = tuple(rational_potential(value) for value in target_float)
    exponentials = []
    for cell_value in cell:
        row = []
        for target_value in target:
            exponent = iv.exp(iv_fraction(cell_value + target_value)) - 1
            row.append(rational_interval(exponent))
        exponentials.append(tuple(row))
    return Cut(
        cell,
        target,
        tuple(exponentials),
        rational_potential(float(increment[1:].sum())),
    )


def make_stationary_cut(
    increment: np.ndarray, state: np.ndarray, crossing: float
) -> StationaryCut:
    source_float, target_float = product.stationary_coupling_dual_potentials(
        increment, state, crossing
    )
    source = tuple(rational_potential(value) for value in source_float)
    target = tuple(rational_potential(value) for value in target_float)
    exponentials = []
    for source_value in source:
        row = []
        for target_value in target:
            exponent = iv.exp(iv_fraction(source_value + target_value)) - 1
            row.append(rational_interval(exponent))
        exponentials.append(tuple(row))
    return StationaryCut(
        source,
        target,
        tuple(exponentials),
        rational_potential(float(increment[1:].sum())),
    )


def multiply_stationary_cut(
    cut: StationaryCut, multiplier: int
) -> StationaryCut:
    prime = len(cut.source)
    source = [ZERO] * prime
    target = [ZERO] * prime
    exponentials: list[list[Interval | None]] = [
        [None] * prime for _ in range(prime)
    ]
    for old_source in range(prime):
        new_source = (multiplier * old_source) % prime
        source[new_source] = cut.source[old_source]
        for old_target in range(prime):
            new_target = (multiplier * old_target) % prime
            target[new_target] = cut.target[old_target]
            exponentials[new_source][new_target] = (
                cut.exponential_minus_one[old_source][old_target]
            )
    if any(value is None for row in exponentials for value in row):
        raise RuntimeError("field multiplier did not permute every symbol")
    return StationaryCut(
        tuple(source),
        tuple(target),
        tuple(
            tuple(cast(Interval, value) for value in row)
            for row in exponentials
        ),
        cut.reference_support,
    )


def stationary_pure_coefficients(
    cut: StationaryCut, difference: int, source: int
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    target = (difference + source) % len(cut.source)
    row_zero = cut.exponential_minus_one[0][0].upper
    row_state = cut.exponential_minus_one[source][source].upper
    row_increment = cut.exponential_minus_one[0][difference].upper
    row_both = cut.exponential_minus_one[source][target].upper
    linear_zero = cut.source[0] + cut.target[0]
    linear_state = cut.source[source] + cut.target[source]
    return (
        linear_zero - row_zero,
        row_state - row_both,
        -linear_zero + linear_state + row_zero - row_state,
        row_zero - row_state - row_increment + row_both,
    )


def multiply_distribution(distribution: np.ndarray, multiplier: int) -> np.ndarray:
    image = np.zeros_like(distribution)
    for value, mass in enumerate(distribution):
        image[(multiplier * value) % distribution.size] = mass
    return image


def default_cut_points() -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, float, float], ...
]:
    candidates = (
        (
            np.array(
                [
                    0.9980000000000012,
                    0.00010317395587027238,
                    0.0006110650740864773,
                    0.0011823776239572128,
                    0.0001033833460848407,
                ]
            ),
            np.array(
                [
                    0.9971866960376634,
                    0.000024089511252901538,
                    0.002364879746160249,
                    0.00039798534996817093,
                    0.000026349354955294586,
                ]
            ),
            np.array(
                [
                    0.8677000000000001,
                    0.0006796742292351857,
                    0.009602382386432106,
                    0.1216227133119478,
                    0.0003952300723848224,
                ]
            ),
            0.0009924297599807304,
            0.0013463064097057693,
        ),
        (
            np.array(
                [
                    0.9900000000000007,
                    0.0004817295777350599,
                    0.0033292524007485567,
                    0.0057069617537894345,
                    0.0004820562677262373,
                ]
            ),
            np.array(
                [
                    0.98770087652055,
                    0.00011757440020742018,
                    0.009749117766694178,
                    0.0022957750122034838,
                    0.00013665630034491075,
                ]
            ),
            np.array(
                [
                    0.8676999999999999,
                    0.0015434054752647701,
                    0.017543986800500577,
                    0.11211553810675778,
                    0.0010970696174770215,
                ]
            ),
            0.004920958852997632,
            0.005794215928141168,
        ),
        (
            np.array(
                [
                    0.9499999999999996,
                    0.0021068813283377672,
                    0.01979104135587067,
                    0.025999739084091908,
                    0.0021023382317000598,
                ]
            ),
            np.array(
                [
                    0.9504417825900682,
                    0.0005839598214430296,
                    0.03295643275091272,
                    0.015330641524374864,
                    0.0006871833132012314,
                ]
            ),
            np.array(
                [
                    0.8676999999999995,
                    0.0034208297549106867,
                    0.03213543044378888,
                    0.09420647385329259,
                    0.0025372659480083714,
                ]
            ),
            0.02420947491599,
            0.022886529490094412,
        ),
        (
            np.array(
                [
                    0.7400000000000739,
                    0.01034407694418354,
                    0.11965592025997594,
                    0.11965591787835088,
                    0.010344084917415713,
                ]
            ),
            np.array(
                [
                    0.8164663528590175,
                    0.003158822238045735,
                    0.08860799604986291,
                    0.08860800355152322,
                    0.0031588253015506188,
                ]
            ),
            np.array(
                [
                    0.8676999999999551,
                    0.0034247123547526625,
                    0.0627252994915048,
                    0.0627252793532872,
                    0.0034247088005002944,
                ]
            ),
            0.12340366447329709,
            0.08713621468109453,
        ),
    )
    return candidates


def default_cuts() -> tuple[Cut, ...]:
    cuts = []
    for mu, beta, zeta, crossing1, crossing2 in default_cut_points():
        for multiplier in range(1, 5):
            mu_image = multiply_distribution(mu, multiplier)
            beta_image = multiply_distribution(beta, multiplier)
            zeta_image = multiply_distribution(zeta, multiplier)
            cuts.append(make_cut(mu_image, beta_image, crossing1))
            cuts.append(make_cut(beta_image, zeta_image, crossing2))
    return tuple(cuts)


def default_stationary_cuts() -> tuple[StationaryCut, ...]:
    cuts = []
    for mu, beta, zeta, crossing1, crossing2 in default_cut_points():
        first = make_stationary_cut(mu, beta, crossing1)
        second = make_stationary_cut(beta, zeta, crossing2)
        for multiplier in range(1, 5):
            cuts.append(multiply_stationary_cut(first, multiplier))
            cuts.append(multiply_stationary_cut(second, multiplier))
    return tuple(cuts)


def default_hybrid_cuts() -> tuple[DualCut, ...]:
    return default_cuts() + default_stationary_cuts()


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    if dimension >= 14:
        support_dimension = dimension - 14
        midpoint = (
            box.support_lower[support_dimension]
            + box.support_upper[support_dimension]
        ) / TWO
        left_support_upper = list(box.support_upper)
        left_support_upper[support_dimension] = midpoint
        right_support_lower = list(box.support_lower)
        right_support_lower[support_dimension] = midpoint
        return (
            Box(
                box.lower,
                box.upper,
                box.support_lower,
                tuple(left_support_upper),
            ),
            Box(
                box.lower,
                box.upper,
                tuple(right_support_lower),
                box.support_upper,
            ),
        )
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return (
        Box(
            box.lower,
            tuple(left_upper),
            box.support_lower,
            box.support_upper,
        ),
        Box(
            tuple(right_lower),
            box.upper,
            box.support_lower,
            box.support_upper,
        ),
    )


def search(args: argparse.Namespace) -> dict[str, object]:
    iv.dps = args.precision
    cuts = default_cuts()
    delta = args.delta
    lower = [ZERO] * 14
    upper = [4 * delta] * 4 + [2 * delta] * 4 + [delta] * 4
    upper.extend([2 * delta, delta])
    initial_box = tighten(Box(tuple(lower), tuple(upper)), args.eta, delta)
    if initial_box is None:
        raise RuntimeError("initial reduced domain is infeasible")
    initial = box_bound(initial_box, cuts)
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < ZERO:
            return {
                "status": "verified negative conditional-product cover",
                "upper_bound": float(current.value),
                "upper_bound_numerator": str(current.value.numerator),
                "upper_bound_denominator": str(current.value.denominator),
                "upper_bound_is_exactly_negative": True,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
                "dual_cuts": len(cuts),
            }
        for child in split(current.box, current.split_dimension):
            tightened = tighten(child, args.eta, delta)
            if tightened is None:
                continue
            bounded = box_bound(tightened, cuts)
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    return {
        "status": "exact conditional-product cover incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
        "dual_cuts": len(cuts),
        "worst_box": (
            {
                "support_lower": [str(value) for value in queue[0][2].box.support_lower],
                "support_upper": [str(value) for value in queue[0][2].box.support_upper],
                "lower": [str(value) for value in queue[0][2].box.lower],
                "upper": [str(value) for value in queue[0][2].box.upper],
                "base_upper": float(queue[0][2].base_upper),
                "first_payment": float(queue[0][2].first_payment),
                "second_payment": float(queue[0][2].second_payment),
                "split_dimension": queue[0][2].split_dimension,
            }
            if queue
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
        parser.error("the first exact reduced verifier is specialized to p=5")
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
            "arithmetic": "exact rational boxes and directed transcendental intervals",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
