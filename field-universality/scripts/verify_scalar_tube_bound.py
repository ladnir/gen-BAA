#!/usr/bin/env python3
"""Exact rational box cover for the scalar F_5 normalized tube.

The last two coordinates are the mass-scaled radii
``w1 = (A-x) z1`` and ``w2 = (B-y) z2``.  This avoids the singular
conditional radius near small supports and makes the tube constraint a
diagonal quadratic.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from mpmath import iv

from verify_conditional_product_bulk import (
    Interval,
    ONE,
    ZERO,
    cross_entropy_upper,
    entropy_bounds,
    negative_log_upper,
)
from verify_reduced_composition_bulk import (
    binary_divergence_at,
    rational_sqrt_upper,
)
from verify_support_indicator_bulk import (
    Box as SupportBox,
    box_bound as support_indicator_box_bound,
)


TWO = Fraction(2)
THREE = Fraction(3)
FOUR = Fraction(4)
FIVE = Fraction(5)


@dataclass(frozen=True)
class Box:
    # Coordinates are A,B,x,y,w1,w2.  The output support C is fixed at delta.
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
    tube_lower: Fraction


def square_distance_lower(interval: Interval, center: Fraction) -> Fraction:
    if interval.lower <= center <= interval.upper:
        return ZERO
    distance = min(abs(interval.lower - center), abs(interval.upper - center))
    return distance * distance


def difference_interval(left: Interval, right: Interval) -> Interval:
    return Interval(left.lower - right.upper, left.upper - right.lower)


def tube_lower_bound(box: Box) -> Fraction:
    a = Interval(box.lower[0], box.upper[0])
    b = Interval(box.lower[1], box.upper[1])
    x = Interval(box.lower[2], box.upper[2])
    y = Interval(box.lower[3], box.upper[3])
    w1 = Interval(box.lower[4], box.upper[4])
    w2 = Interval(box.lower[5], box.upper[5])
    first_defect = difference_interval(x.scale(FIVE), a)
    second_defect = difference_interval(y.scale(FIVE), b)
    first_crossing = square_distance_lower(first_defect, ZERO) / 20
    second_crossing = square_distance_lower(second_defect, ZERO) / 20
    first_composition = Fraction(4, 3) * w1.lower**2
    second_composition = Fraction(4, 3) * w2.lower**2
    return first_crossing + second_crossing + first_composition + second_composition


def tighten(
    raw: Box,
    eta: Fraction,
    delta: Fraction,
    maximum_defect: Fraction,
) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    crossing_radius = rational_sqrt_upper(Fraction(4, 5) * maximum_defect)
    maximum_w = rational_sqrt_upper(Fraction(3, 4) * maximum_defect)
    for _ in range(16):
        before = (tuple(lower), tuple(upper))
        lower[0] = max(lower[0], eta, TWO * lower[2])
        upper[0] = min(upper[0], TWO * upper[1], upper[1] + upper[2])
        lower[1] = max(lower[1], lower[0] - upper[2], TWO * lower[3])
        upper[1] = min(upper[1], TWO * delta, delta + upper[3])

        lower[2] = max(lower[2], ZERO, lower[0] - upper[1])
        upper[2] = min(upper[2], upper[0] / TWO, ONE - lower[1])
        lower[3] = max(lower[3], ZERO, lower[1] - delta)
        upper[3] = min(upper[3], upper[1] / TWO, ONE - delta)

        lower[2] = max(lower[2], lower[0] / FIVE - crossing_radius)
        upper[2] = min(upper[2], upper[0] / FIVE + crossing_radius)
        lower[0] = max(lower[0], FIVE * (lower[2] - crossing_radius))
        upper[0] = min(upper[0], FIVE * (upper[2] + crossing_radius))
        lower[3] = max(lower[3], lower[1] / FIVE - crossing_radius)
        upper[3] = min(upper[3], upper[1] / FIVE + crossing_radius)
        lower[1] = max(lower[1], FIVE * (lower[3] - crossing_radius))
        upper[1] = min(upper[1], FIVE * (upper[3] + crossing_radius))

        # Since each conditional radius is at most 3/4, w is at most
        # 3/4 times the corresponding both-nonzero mass.
        upper[4] = min(
            upper[4], maximum_w, Fraction(3, 4) * (upper[0] - lower[2])
        )
        upper[5] = min(
            upper[5], maximum_w, Fraction(3, 4) * (upper[1] - lower[3])
        )

        if any(lower[index] > upper[index] for index in range(6)):
            return None
        candidate = Box(tuple(lower), tuple(upper))
        tube_lower = tube_lower_bound(candidate)
        if tube_lower > maximum_defect:
            return None

        # Charge the other three nonnegative tube terms before bounding each
        # w-coordinate.  This is valid because the tube is
        # d1^2/20 + 4w1^2/3 + d2^2/20 + 4w2^2/3.
        for coordinate in (4, 5):
            own_lower = Fraction(4, 3) * lower[coordinate] ** 2
            remaining = maximum_defect - (tube_lower - own_lower)
            if remaining < ZERO:
                return None
            upper[coordinate] = min(
                upper[coordinate],
                rational_sqrt_upper(Fraction(3, 4) * remaining),
            )
        if before == (tuple(lower), tuple(upper)):
            return candidate
    return Box(tuple(lower), tuple(upper))


def binary_entropy_lower(value: Interval) -> Fraction:
    return entropy_bounds(
        (Interval(ONE - value.upper, ONE - value.lower), value)
    ).lower


def binary_entropy_at(value: Fraction) -> Interval:
    return entropy_bounds(
        (
            Interval(ONE - value, ONE - value),
            Interval(value, value),
        )
    )


def entropy_secant_lower(interval: Interval) -> tuple[Fraction, Fraction]:
    if interval.lower == interval.upper:
        return ZERO, binary_entropy_at(interval.lower).lower
    left = binary_entropy_at(interval.lower).lower
    right = binary_entropy_at(interval.upper).lower
    slope = (right - left) / (interval.upper - interval.lower)
    return slope, left - slope * interval.lower


def affine_half_support_upper(
    support_coefficient: Fraction,
    crossing_coefficient: Fraction,
    support: Interval,
    crossing: Interval,
) -> Fraction:
    """Maximize c_A A+c_x x on a box intersected with 2x<=A."""
    candidates: set[tuple[Fraction, Fraction]] = set()
    for support_value in (support.lower, support.upper):
        for crossing_value in (crossing.lower, crossing.upper):
            if TWO * crossing_value <= support_value:
                candidates.add((support_value, crossing_value))
    for crossing_value in (crossing.lower, crossing.upper):
        support_value = TWO * crossing_value
        if support.lower <= support_value <= support.upper:
            candidates.add((support_value, crossing_value))
    for support_value in (support.lower, support.upper):
        crossing_value = support_value / TWO
        if crossing.lower <= crossing_value <= crossing.upper:
            candidates.add((support_value, crossing_value))
    if not candidates:
        raise RuntimeError("support-crossing box misses 2x<=A")
    return max(
        support_coefficient * support_value
        + crossing_coefficient * crossing_value
        for support_value, crossing_value in candidates
    )


Affine = tuple[Fraction, Fraction, Fraction, Fraction, Fraction]


def add_affine(
    accumulator: list[Fraction], cell: Affine, coefficient: Fraction
) -> None:
    for index, value in enumerate(cell):
        accumulator[index] += coefficient * value


def add_cross_entropy_tangent(
    accumulator: list[Fraction],
    cells: tuple[Affine, ...],
    reference: tuple[Fraction, Fraction, Fraction, Fraction],
) -> None:
    mixing = Fraction(1, 1 << 32)
    mixed = tuple(
        (ONE - mixing) * value + mixing / 4 for value in reference
    )
    for cell, probability in zip(cells, mixed, strict=True):
        add_affine(accumulator, cell, negative_log_upper(probability))


def simplex_reference(
    values: tuple[Fraction, Fraction, Fraction, Fraction]
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    nonnegative = tuple(max(ZERO, value) for value in values)
    total = sum(nonnegative, ZERO)
    if total == ZERO:
        return (Fraction(1, 4),) * 4
    return tuple(value / total for value in nonnegative)


def coupled_base_upper(
    a: Interval,
    b: Interval,
    x: Interval,
    y: Interval,
    delta: Fraction,
) -> Fraction:
    # Affine coordinates are (constant,A,B,x,y).
    coefficients = [ZERO] * 5
    first_cells: tuple[Affine, ...] = (
        (ONE, ZERO, -ONE, -ONE, ZERO),
        (ZERO, -ONE, ONE, ONE, ZERO),
        (ZERO, ZERO, ZERO, ONE, ZERO),
        (ZERO, ONE, ZERO, -ONE, ZERO),
    )
    second_cells: tuple[Affine, ...] = (
        (ONE - delta, ZERO, ZERO, ZERO, -ONE),
        (delta, ZERO, -ONE, ZERO, ONE),
        (ZERO, ZERO, ZERO, ZERO, ONE),
        (ZERO, ZERO, ONE, ZERO, -ONE),
    )
    a_midpoint = (a.lower + a.upper) / TWO
    b_midpoint = (b.lower + b.upper) / TWO
    x_midpoint = (x.lower + x.upper) / TWO
    y_midpoint = (y.lower + y.upper) / TWO
    first_reference = simplex_reference(
        (
            ONE - b_midpoint - x_midpoint,
            b_midpoint - a_midpoint + x_midpoint,
            x_midpoint,
            a_midpoint - x_midpoint,
        )
    )
    second_reference = simplex_reference(
        (
            ONE - delta - y_midpoint,
            delta - b_midpoint + y_midpoint,
            y_midpoint,
            b_midpoint - y_midpoint,
        )
    )
    add_cross_entropy_tangent(coefficients, first_cells, first_reference)
    add_cross_entropy_tangent(coefficients, second_cells, second_reference)

    for interval, factor, coordinate in (
        (a, Fraction(2, 3), 1),
        (b, TWO, 2),
    ):
        slope, intercept = entropy_secant_lower(interval)
        coefficients[0] -= factor * intercept
        coefficients[coordinate] -= factor * slope
    coefficients[0] -= binary_entropy_at(delta).lower
    coefficients[1] += negative_log_upper(Fraction(1, 4)) / 3

    return (
        coefficients[0]
        + affine_half_support_upper(
            coefficients[1], coefficients[3], a, x
        )
        + affine_half_support_upper(
            coefficients[2], coefficients[4], b, y
        )
    )


def support_joint(
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


def scalar_payment_lower(
    support: Interval, crossing: Interval, scaled_radius: Interval
) -> Fraction:
    both_nonzero = difference_interval(support, crossing)
    if both_nonzero.lower <= ZERO:
        return ZERO
    defect = difference_interval(crossing.scale(FIVE), support)
    if defect.lower <= ZERO <= defect.upper:
        absolute_defect_lower = ZERO
    else:
        absolute_defect_lower = min(abs(defect.lower), abs(defect.upper))
    mass_gap = max(
        ZERO, absolute_defect_lower / FOUR - scaled_radius.upper
    )
    pinsker = TWO * mass_gap**2 / both_nonzero.upper

    radius = scaled_radius.divide_positive(both_nonzero)
    observed = crossing.divide_positive(both_nonzero)
    observed = Interval(max(ZERO, observed.lower), min(ONE, observed.upper))
    reference = Interval(
        max(ZERO, Fraction(1, 4) - radius.upper),
        min(ONE, Fraction(1, 4) + radius.upper),
    )
    if observed.upper < reference.lower:
        divergence = binary_divergence_at(observed.upper, reference.lower)
    elif reference.upper < observed.lower:
        divergence = binary_divergence_at(observed.lower, reference.upper)
    else:
        divergence = ZERO
    interval_kl = both_nonzero.lower * divergence
    return max(pinsker, interval_kl)


def message_cancellation_tradeoff_lower(
    support: Interval, crossing: Interval
) -> Fraction:
    defect = crossing.scale(FIVE) - support
    positive_defect_lower = max(ZERO, defect.lower)
    if positive_defect_lower == ZERO:
        return ZERO
    both_nonzero_upper = support.upper - crossing.lower
    second_factor_upper = FOUR * support.upper - THREE * crossing.lower
    denominator_upper = (
        8 * both_nonzero_upper * second_factor_upper
    )
    if denominator_upper <= ZERO:
        return ZERO
    return (
        support.lower * positive_defect_lower**2 / denominator_upper
    )


@lru_cache(maxsize=None)
def message_tradeoff_at_grid_point(
    grid_index: int,
    t_steps: int = 128,
    m_steps: int = 256,
) -> Fraction:
    """Directed lower bound for the entropy--cancellation envelope at t."""
    if grid_index <= 0:
        return ZERO
    t = Fraction(1, 5) + Fraction(3 * grid_index, 10 * t_steps)
    q = t / (ONE - t)
    width = Fraction(3, 4 * m_steps)
    best: Fraction | None = None
    for index in range(m_steps):
        left = Fraction(1, 4) + index * width
        right = left + width
        entropy_deficit = binary_divergence_at(
            left, Fraction(1, 4)
        ) / 3
        cancellation = ZERO
        if right < q:
            cancellation = (ONE - t) * binary_divergence_at(q, right)
        lower = entropy_deficit + cancellation
        best = lower if best is None else min(best, lower)
    if best is None:
        raise RuntimeError("empty message-tradeoff grid")
    return best


def gridded_message_tradeoff_lower(
    support: Interval,
    crossing: Interval,
    t_steps: int = 128,
) -> Fraction:
    if support.upper <= ZERO:
        return ZERO
    t_lower = crossing.lower / support.upper
    if t_lower <= Fraction(1, 5):
        return ZERO
    scaled = (t_lower - Fraction(1, 5)) * Fraction(10 * t_steps, 3)
    grid_index = min(t_steps, scaled.numerator // scaled.denominator)
    return support.lower * message_tradeoff_at_grid_point(
        grid_index, t_steps
    )


def box_bound(box: Box, delta: Fraction) -> Bound:
    a = Interval(box.lower[0], box.upper[0])
    b = Interval(box.lower[1], box.upper[1])
    x = Interval(box.lower[2], box.upper[2])
    y = Interval(box.lower[3], box.upper[3])
    w1 = Interval(box.lower[4], box.upper[4])
    w2 = Interval(box.lower[5], box.upper[5])
    c = Interval(delta, delta)
    first_joint = support_joint(a, b, x)
    second_joint = support_joint(b, c, y)
    base_upper = cross_entropy_upper(first_joint)
    base_upper += cross_entropy_upper(second_joint)
    base_upper -= Fraction(2, 3) * binary_entropy_lower(a)
    base_upper -= TWO * binary_entropy_lower(b)
    base_upper -= binary_entropy_lower(c)
    base_upper += a.upper * entropy_bounds(
        (Interval(Fraction(1, 4), Fraction(1, 4)),) * 4
    ).upper / 3
    base_upper = min(base_upper, coupled_base_upper(a, b, x, y, delta))
    support_result = support_indicator_box_bound(
        SupportBox(
            (a.lower, b.lower, delta),
            (a.upper, b.upper, delta),
        ),
        5,
        3,
        False,
    )
    if support_result is None:
        raise RuntimeError("scalar box has infeasible support projection")
    base_upper = min(base_upper, support_result[0])
    first_payment = max(
        scalar_payment_lower(a, x, w1),
        message_cancellation_tradeoff_lower(a, x),
        gridded_message_tradeoff_lower(a, x),
    )
    second_payment = scalar_payment_lower(b, y, w2)
    value = base_upper - first_payment - second_payment

    scores = []
    for index, (lower, upper) in enumerate(
        zip(box.lower, box.upper, strict=True)
    ):
        width = upper - lower
        if index in (0, 1):
            scale = max(upper, Fraction(1, 10_000))
            priority = 4.0
        elif index in (2, 3):
            scale = max(box.upper[index - 2], Fraction(1, 10_000))
            priority = 3.0
        else:
            # The mass-scaled radii are O(sqrt(epsilon)), not O(1).
            # Normalizing them against 10^-2 starves the small-support cover
            # of the splits that activate cancellation and tube coupling.
            scale = max(upper, Fraction(1, 1_000_000))
            priority = 0.2
        scores.append(priority * float(width / scale))
    return Bound(
        value,
        box,
        int(np.argmax(scores)),
        base_upper,
        first_payment,
        second_payment,
        tube_lower_bound(box),
    )


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def initial_box(args: argparse.Namespace) -> Box:
    iv.dps = args.precision
    maximum_a = Fraction(25, 16) * args.delta + FIVE * rational_sqrt_upper(
        Fraction(4, 5) * args.maximum_defect
    )
    maximum_b = Fraction(5, 4) * (
        args.delta
        + rational_sqrt_upper(Fraction(4, 5) * args.maximum_defect)
    )
    initial = Box(
        (args.eta, ZERO, ZERO, ZERO, ZERO, ZERO),
        (
            maximum_a,
            maximum_b,
            maximum_a / TWO,
            maximum_b / TWO,
            rational_sqrt_upper(Fraction(3, 4) * args.maximum_defect),
            rational_sqrt_upper(Fraction(3, 4) * args.maximum_defect),
        ),
    )
    tightened = tighten(
        initial, args.eta, args.delta, args.maximum_defect
    )
    if tightened is None:
        raise RuntimeError("the initial scalar tube is infeasible")
    return tightened


def search(
    args: argparse.Namespace, *, record_splits: bool = False
) -> dict[str, object]:
    iv.dps = args.precision
    initial = initial_box(args)
    first = box_bound(initial, args.delta)
    queue = [(-first.value, 0, first)]
    identifier = 1
    splits = 0
    split_records: list[list[int]] = []
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, node_identifier, current = heapq.heappop(queue)
        if current.value < ZERO:
            result: dict[str, object] = {
                "status": "verified negative scalar tube cover",
                "upper_bound": float(current.value),
                "upper_bound_numerator": str(current.value.numerator),
                "upper_bound_denominator": str(current.value.denominator),
                "upper_bound_is_exactly_negative": True,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
            }
            if record_splits:
                result["split_records"] = split_records
            return result
        if record_splits:
            split_records.append(
                [node_identifier, current.split_dimension]
            )
        for child in split(current.box, current.split_dimension):
            tightened = tighten(
                child, args.eta, args.delta, args.maximum_defect
            )
            if tightened is None:
                continue
            bounded = box_bound(tightened, args.delta)
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
        if args.progress_every and splits % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "boxes_split": splits,
                        "boxes_remaining": len(queue),
                        "current_upper_bound": (
                            float(-queue[0][0]) if queue else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    worst = queue[0][2] if queue else None
    return {
        "status": "exact scalar tube cover incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
        "worst_box": (
            {
                "lower": [str(value) for value in worst.box.lower],
                "upper": [str(value) for value in worst.box.upper],
                "base_upper": float(worst.base_upper),
                "first_payment": float(worst.first_payment),
                "second_payment": float(worst.second_payment),
                "tube_lower": float(worst.tube_lower),
                "split_dimension": worst.split_dimension,
            }
            if worst is not None
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=Fraction, default=Fraction(11, 50))
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 500))
    parser.add_argument(
        "--maximum-defect", type=Fraction, default=Fraction(1, 1_000_000)
    )
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--certificate",
        type=Path,
        help="write the successful split tree as a replayable JSON certificate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = search(arguments, record_splits=arguments.certificate is not None)
    result.update(
        {
            "prime": 5,
            "repetition": 3,
            "delta": str(arguments.delta),
            "eta": str(arguments.eta),
            "maximum_defect": str(arguments.maximum_defect),
            "interval_precision": arguments.precision,
            "arithmetic": (
                "exact rational boxes and directed transcendental intervals"
            ),
        }
    )
    if arguments.certificate is not None:
        if result["status"] != "verified negative scalar tube cover":
            raise RuntimeError("refusing to write an incomplete tube certificate")
        certificate = dict(result)
        certificate["schema"] = "f5-scalar-tube-cover-v1"
        arguments.certificate.parent.mkdir(parents=True, exist_ok=True)
        arguments.certificate.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = dict(result)
        result.pop("split_records", None)
        result["certificate"] = str(arguments.certificate)
    print(json.dumps(result, indent=2, sort_keys=True))
