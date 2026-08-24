#!/usr/bin/env python3
"""Exact quinary cover in support and conditional-composition coordinates."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from mpmath import iv

from verify_conditional_product_bulk import (
    Cut,
    Interval,
    ONE,
    TWO,
    ZERO,
    cross_entropy_upper,
    default_cuts,
    dot_interval,
    entropy_bounds,
    iv_fraction,
    make_cut,
    rational_interval,
    support_joint_intervals,
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


def conditional_distribution(box: Box, start: int) -> tuple[Interval, ...]:
    first_three = tuple(
        Interval(box.lower[start + index], box.upper[start + index])
        for index in range(3)
    )
    lower_sum = sum((value.lower for value in first_three), ZERO)
    upper_sum = sum((value.upper for value in first_three), ZERO)
    fourth = Interval(max(ZERO, ONE - upper_sum), min(ONE, ONE - lower_sum))
    return first_three + (fourth,)


def tighten(raw: Box, eta: Fraction, delta: Fraction) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    for _ in range(12):
        before = (tuple(lower), tuple(upper))
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

        for start in (5, 8, 11):
            if sum(lower[start : start + 3], ZERO) > ONE:
                return None
            for offset in range(3):
                index = start + offset
                other_lower = sum(lower[start : start + 3], ZERO) - lower[index]
                upper[index] = min(upper[index], ONE - other_lower)
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
    return Box(tuple(lower), tuple(upper))


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
    cut: Cut,
    increment: tuple[Interval, ...],
    state: tuple[Interval, ...],
    increment_support: Interval,
    state_support: Interval,
    crossing: Interval,
) -> Fraction:
    best: Fraction | None = None
    for increment_vertex in simplex_vertices(increment):
        for state_vertex in simplex_vertices(state):
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


def nearby_cuts(cuts: tuple[Cut, ...], support: Interval) -> tuple[Cut, ...]:
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


def binary_entropy_lower(support: Interval) -> Fraction:
    return entropy_bounds(
        (Interval(ONE - support.upper, ONE - support.lower), support)
    ).lower


def box_bound(box: Box, cuts: tuple[Cut, ...]) -> Bound:
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
    first_cuts = nearby_cuts(cuts, support_mu) + (
        (first_local,) if first_local is not None else ()
    )
    second_cuts = nearby_cuts(cuts, support_beta) + (
        (second_local,) if second_local is not None else ()
    )

    first_payment = max(
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
    value = base_upper - first_payment - second_payment
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
        else:
            scale = ONE
            priority = 1.0
        scores.append(priority * float(width / scale))
    return Bound(
        value,
        box,
        int(np.argmax(scores)),
        base_upper,
        first_payment,
        second_payment,
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
    lower = [args.eta, ZERO, ZERO, ZERO, ZERO] + [ZERO] * 9
    upper = [4 * args.delta, 2 * args.delta, args.delta]
    upper += [2 * args.delta, args.delta] + [ONE] * 9
    initial_box = tighten(Box(tuple(lower), tuple(upper)), args.eta, args.delta)
    if initial_box is None:
        raise RuntimeError("initial conditional-composition domain is infeasible")
    initial = box_bound(initial_box, cuts)
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < ZERO:
            return {
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
        for child in split(current.box, current.split_dimension):
            tightened = tighten(child, args.eta, args.delta)
            if tightened is None:
                continue
            bounded = box_bound(tightened, cuts)
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
        parser.error("the first reduced-composition verifier is specialized to p=5")
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
