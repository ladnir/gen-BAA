#!/usr/bin/env python3
"""Independently verify a negative sign-symmetric ternary bulk exponent.

This checker deliberately shares no implementation code with
``certify_symmetric_ternary.py``.  Boxes and stationary bridges use exact
``Fraction`` arithmetic.  Entropies use mpmath's arbitrary-precision interval
arithmetic.  The checker constructs its own branch-and-bound cover.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction

from mpmath import iv, mp
from mpmath.libmp import fzero, mpf_cmp


ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)
THREE = Fraction(3)
FOUR = Fraction(4)
UNIFORM = Fraction(2, 3)


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, Fraction, Fraction]
    upper: tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class Tangent:
    intercept: object
    increment_coefficient: object
    state_coefficient: object
    bridge_coefficient: object


@dataclass(frozen=True)
class Verification:
    eta: Fraction
    delta: Fraction
    precision: int
    upper_bound: tuple
    boxes_split: int
    boxes_remaining: int
    maximum_queue_size: int


def interval(value: Fraction | int) -> object:
    """Return an interval containing an exact rational value."""

    if isinstance(value, int):
        value = Fraction(value)
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def upper_endpoint(value: object) -> tuple:
    return value._mpi_[1]


def upper_less(left: object, right: object) -> bool:
    return mpf_cmp(upper_endpoint(left), upper_endpoint(right)) < 0


def upper_is_negative(value: object) -> bool:
    return mpf_cmp(upper_endpoint(value), fzero) < 0


def upper_as_float(value: object) -> float:
    return float(value.b)


def entropy_atom(value: Fraction) -> object:
    if value == ZERO:
        return interval(0)
    if value < ZERO:
        raise ValueError("entropy received a negative argument")
    point = interval(value)
    return -point * iv.log(point)


def composition_entropy(nonzero_fraction: Fraction) -> object:
    return entropy_atom(ONE - nonzero_fraction) + interval(2) * entropy_atom(
        nonzero_fraction / TWO
    )


def bridge_interval(
    increment_weight: Fraction, state_weight: Fraction
) -> tuple[Fraction, Fraction]:
    lower = max(ZERO, (increment_weight - state_weight) / TWO)
    upper = min((ONE - state_weight) / TWO, increment_weight / FOUR)
    return lower, upper


def transition_entries(
    increment_weight: Fraction, state_weight: Fraction, bridge: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        ONE - state_weight - TWO * bridge,
        bridge,
        (state_weight - increment_weight + TWO * bridge) / TWO,
        increment_weight / TWO - TWO * bridge,
    )


def stationary_bridge(
    increment_weight: Fraction, state_weight: Fraction
) -> Fraction:
    """Enclose the entropy maximizer by exact rational bisection."""

    lower, upper = bridge_interval(increment_weight, state_weight)
    if not lower < upper:
        raise ValueError("the reference transition is not in the relative interior")

    def balance(bridge: Fraction) -> Fraction:
        diagonal_zero, bridge, diagonal_nonzero, off_diagonal = transition_entries(
            increment_weight, state_weight, bridge
        )
        return diagonal_zero * off_diagonal**2 - bridge**2 * diagonal_nonzero

    for _ in range(160):
        midpoint = (lower + upper) / TWO
        if balance(midpoint) > ZERO:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / TWO


def transition_tangent(
    increment_weight: Fraction,
    state_weight: Fraction,
    mixing: Fraction,
) -> Tangent:
    """Construct an interval-enclosed global affine entropy majorant."""

    increment_weight = (ONE - mixing) * increment_weight + mixing * UNIFORM
    state_weight = (ONE - mixing) * state_weight + mixing * UNIFORM
    bridge = stationary_bridge(increment_weight, state_weight)
    entries = transition_entries(increment_weight, state_weight, bridge)
    if min(entries) <= ZERO:
        raise ArithmeticError("mixed reference transition is not strictly positive")

    multiplicities = (ONE, FOUR, TWO, TWO)
    affine = (
        (ONE, ZERO, -ONE, -TWO),
        (ZERO, ZERO, ZERO, ONE),
        (ZERO, -ONE / TWO, ONE / TWO, ONE),
        (ZERO, ONE / TWO, ZERO, -TWO),
    )
    coefficients = [interval(0) for _ in range(4)]
    entropy = interval(0)
    tangent_at_reference = interval(0)
    for multiplicity, entry, entry_affine in zip(
        multiplicities, entries, affine, strict=True
    ):
        multiplier = interval(multiplicity)
        point = interval(entry)
        entropy += multiplier * entropy_atom(entry)
        derivative = -(iv.log(point) + interval(1))
        tangent_at_reference += multiplier * derivative * point
        for coordinate in range(4):
            coefficients[coordinate] += (
                multiplier * derivative * interval(entry_affine[coordinate])
            )

    intercept = entropy - tangent_at_reference + coefficients[0]
    return Tangent(intercept, coefficients[1], coefficients[2], coefficients[3])


def tighten(box: Box) -> Box | None:
    """Intersect a rational box with A <= 2B and B <= 2C."""

    lower = list(box.lower)
    upper = list(box.upper)
    for _ in range(3):
        lower[2] = max(lower[2], lower[1] / TWO, lower[0] / FOUR)
        lower[1] = max(lower[1], lower[0] / TWO)
        upper[1] = min(upper[1], TWO * upper[2])
        upper[0] = min(upper[0], TWO * upper[1], FOUR * upper[2])
        if any(lower[index] > upper[index] for index in range(3)):
            return None
    return Box(tuple(lower), tuple(upper))


def reference_point(box: Box) -> tuple[Fraction, Fraction, Fraction]:
    output = (box.lower[2] + box.upper[2]) / TWO
    state_lower = max(box.lower[1], box.lower[0] / TWO)
    state_upper = min(box.upper[1], TWO * output)
    state = (state_lower + state_upper) / TWO
    increment_upper = min(box.upper[0], TWO * state)
    increment = (box.lower[0] + increment_upper) / TWO
    return increment, state, output


def endpoint_maximum(
    coefficient: object,
    entropy_coefficient: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> object:
    def value(point: Fraction) -> object:
        return coefficient * interval(point) - interval(
            entropy_coefficient
        ) * composition_entropy(point)

    lower_value = value(lower)
    upper_value = value(upper)
    return upper_value if upper_less(lower_value, upper_value) else lower_value


def box_bound(box: Box, mixing: Fraction) -> tuple[object, Box] | None:
    box = tighten(box)
    if box is None:
        return None
    increment, state, output = reference_point(box)
    first = transition_tangent(increment, state, mixing)
    second = transition_tangent(state, output, mixing)

    # The product intervals bound each bridge term for every b in [0, 1/4].
    bridge_range = iv.mpf([0, 0.25])
    result = (
        first.intercept
        + second.intercept
        + first.bridge_coefficient * bridge_range
        + second.bridge_coefficient * bridge_range
    )
    coefficients = (
        first.increment_coefficient,
        first.state_coefficient + second.increment_coefficient,
        second.state_coefficient,
    )
    entropy_coefficients = (Fraction(2, 3), TWO, ONE)
    for index in range(3):
        result += endpoint_maximum(
            coefficients[index],
            entropy_coefficients[index],
            box.lower[index],
            box.upper[index],
        )
    return result, box


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def maximum_by_upper(items: list[tuple[object, Box]]) -> tuple[object, Box]:
    maximum = items[0]
    for item in items[1:]:
        if upper_less(maximum[0], item[0]):
            maximum = item
    return maximum


def verify(
    eta: Fraction,
    delta: Fraction,
    precision: int = 80,
    max_boxes: int = 100_000,
) -> Verification:
    if not ZERO < eta < ONE or not ZERO < delta < ONE:
        raise ValueError("eta and delta must lie in (0,1)")
    if precision < 50:
        raise ValueError("precision must be at least 50 decimal digits")

    iv.dps = precision
    mp.dps = precision
    mixing = Fraction(1, 10**12)
    initial = Box((eta, ZERO, ZERO), (ONE, ONE, delta))
    bounded = box_bound(initial, mixing)
    if bounded is None:
        raise RuntimeError("the requested domain is empty")

    initial_bound, initial = bounded
    queue: list[tuple[float, int, object, Box]] = [
        (-upper_as_float(initial_bound), 0, initial_bound, initial)
    ]
    identifier = 1
    boxes_split = 0
    maximum_queue_size = 1
    closed_upper: object | None = None
    closed_boxes = 0

    while queue:
        _, _, bound, box = heapq.heappop(queue)
        if upper_is_negative(bound):
            remaining = [(bound, box)] + [(item[2], item[3]) for item in queue]
            if all(upper_is_negative(item[0]) for item in remaining):
                maximum_bound = maximum_by_upper(remaining)[0]
                if closed_upper is not None and upper_less(
                    maximum_bound, closed_upper
                ):
                    maximum_bound = closed_upper
                return Verification(
                    eta,
                    delta,
                    precision,
                    upper_endpoint(maximum_bound),
                    boxes_split,
                    closed_boxes + len(remaining),
                    maximum_queue_size,
                )
            if closed_upper is None or upper_less(closed_upper, bound):
                closed_upper = bound
            closed_boxes += 1
            continue
        if boxes_split >= max_boxes:
            raise RuntimeError(
                f"verification did not close after {max_boxes} box splits; "
                f"current upper bound is {upper_as_float(bound)}"
            )

        best_children: list[tuple[object, Box]] | None = None
        best_child_maximum: object | None = None
        for dimension in range(3):
            candidates = []
            for child in split(box, dimension):
                child_bound = box_bound(child, mixing)
                if child_bound is not None:
                    candidates.append(child_bound)
            if not candidates:
                continue
            candidate_maximum = maximum_by_upper(candidates)[0]
            if best_child_maximum is None or upper_less(
                candidate_maximum, best_child_maximum
            ):
                best_child_maximum = candidate_maximum
                best_children = candidates
        if best_children is None:
            raise ArithmeticError("a feasible parent had no feasible child")

        for child_bound, child in best_children:
            heapq.heappush(
                queue,
                (-upper_as_float(child_bound), identifier, child_bound, child),
            )
            identifier += 1
        boxes_split += 1
        maximum_queue_size = max(maximum_queue_size, len(queue))

    raise AssertionError("unreachable")


def parse_fraction(value: str) -> Fraction:
    return Fraction(value)


def endpoint_text(endpoint: tuple, digits: int) -> str:
    return mp.nstr(mp.make_mpf(endpoint), digits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=parse_fraction, default=Fraction(1, 10_000))
    parser.add_argument("--delta", type=parse_fraction, default=Fraction(49, 250))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.eta, args.delta, args.precision, args.max_boxes)
    print(
        json.dumps(
            {
                "eta": str(result.eta),
                "delta": str(result.delta),
                "precision": result.precision,
                "certified_upper_bound": endpoint_text(
                    result.upper_bound, result.precision
                ),
                "boxes_split": result.boxes_split,
                "boxes_remaining": result.boxes_remaining,
                "maximum_queue_size": result.maximum_queue_size,
                "arithmetic": "exact rational boxes and mpmath intervals",
                "implementation": "independent checker",
                "status": "negative bulk exponent independently verified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
