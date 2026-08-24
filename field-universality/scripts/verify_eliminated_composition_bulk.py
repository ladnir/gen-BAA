#!/usr/bin/env python3
"""Exact p=5 bulk checker after eliminating conditional compositions."""

from __future__ import annotations

import argparse
import heapq
import json
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
    support_upper: Fraction
    conditional_upper: Fraction


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


def exp_bounds(value: Interval) -> Interval:
    lower = rational_interval(iv.exp(iv_fraction(value.lower))).lower
    upper = rational_interval(iv.exp(iv_fraction(value.upper))).upper
    return Interval(lower, upper)


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


def second_stage_lower(
    cuts: tuple[Cut, ...],
    difference: int,
    source: int,
    increment: Interval,
    state: Interval,
    crossing: Interval,
) -> Fraction:
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
        weights = rational_mixture(result.x[:cut_count])
        return min(
            sum(
                (
                    weights[cut_index] * values[cut_index][vertex_index]
                    for cut_index in range(cut_count)
                ),
                ZERO,
            )
            for vertex_index in range(len(vertices))
        )
    return max(min(row) for row in values)


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


def joint_entropy_cross_upper(
    increment: Interval, state: Interval, crossing: Interval
) -> Fraction:
    vertices = support_vertices(increment, state, crossing)
    reference = tuple(
        sum((vertex[index] for vertex in vertices), ZERO) / len(vertices)
        for index in range(3)
    )
    atoms = (
        ONE - reference[1] - reference[2],
        reference[1] - reference[0] + reference[2],
        reference[2],
        reference[0] - reference[2],
    )
    mixing = Fraction(1, 1 << 40)
    positive = tuple((ONE - mixing) * atom + mixing / 4 for atom in atoms)
    coefficients = tuple(negative_log_upper(atom) for atom in positive)
    affine = (
        coefficients[0],
        -coefficients[1] + coefficients[3],
        -coefficients[0] + coefficients[1],
        -coefficients[0] + coefficients[1] + coefficients[2] - coefficients[3],
    )
    return affine_extrema(affine, increment, state, crossing).upper


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
    support = support_upper(box)
    conditional = conditional_envelope_upper(box, cuts, certificates)
    scores = []
    for index, (lower, upper) in enumerate(zip(box.lower, box.upper, strict=True)):
        scale_index = index if index < 3 else index - 3
        scale = max(box.upper[scale_index], Fraction(1, 10_000))
        scores.append(float((upper - lower) / scale))
    return Bound(
        support + conditional,
        box,
        max(range(5), key=scores.__getitem__),
        support,
        conditional,
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
                "support_upper": float(worst.support_upper),
                "conditional_upper": float(worst.conditional_upper),
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
