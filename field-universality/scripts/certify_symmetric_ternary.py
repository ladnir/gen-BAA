#!/usr/bin/env python3
"""Deterministically certify a negative sign-symmetric ternary bulk exponent.

The certificate uses supporting hyperplanes for the two transition entropies
and a branch-and-bound cover of the three outer composition variables. A small
interval layer rounds every Decimal operation outwards.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext


ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
THREE = Decimal(3)
FOUR = Decimal(4)


@dataclass(frozen=True)
class Box:
    lower: tuple[Decimal, Decimal, Decimal]
    upper: tuple[Decimal, Decimal, Decimal]


@dataclass(frozen=True)
class Tangent:
    intercept: Decimal
    increment_coefficient: Decimal
    state_coefficient: Decimal
    bridge_coefficient: Decimal


@dataclass(frozen=True)
class Interval:
    lower: Decimal
    upper: Decimal

    @staticmethod
    def exact(value: Decimal | int) -> Interval:
        value = value if isinstance(value, Decimal) else Decimal(value)
        return Interval(value, value)

    def __add__(self, other: Interval) -> Interval:
        return Interval(
            (self.lower + other.lower).next_minus(),
            (self.upper + other.upper).next_plus(),
        )

    def __neg__(self) -> Interval:
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: Interval) -> Interval:
        return self + (-other)

    def __mul__(self, other: Interval) -> Interval:
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products).next_minus(), max(products).next_plus())

    def ln(self) -> Interval:
        if self.lower <= ZERO:
            raise ValueError("interval logarithm requires a positive lower endpoint")
        return Interval(self.lower.ln().next_minus(), self.upper.ln().next_plus())


def rational_interval(numerator: Decimal, denominator: Decimal) -> Interval:
    quotient = numerator / denominator
    return Interval(quotient.next_minus(), quotient.next_plus())


@dataclass(frozen=True)
class IntervalTangent:
    intercept: Interval
    increment_coefficient: Interval
    state_coefficient: Interval
    bridge_coefficient: Interval


@dataclass(frozen=True)
class Certificate:
    eta: Decimal
    delta: Decimal
    precision: int
    upper_bound: Decimal
    boxes_split: int
    boxes_remaining: int
    maximum_queue_size: int


def entropy_atom(value: Decimal) -> Decimal:
    if value == ZERO:
        return ZERO
    if value < ZERO:
        raise ValueError("entropy received a negative argument")
    return -value * value.ln()


def ternary_composition_entropy(nonzero_fraction: Decimal) -> Decimal:
    return entropy_atom(ONE - nonzero_fraction) + TWO * entropy_atom(
        nonzero_fraction / TWO
    )


def bridge_interval(
    increment_weight: Decimal, state_weight: Decimal
) -> tuple[Decimal, Decimal]:
    lower = max(ZERO, (increment_weight - state_weight) / TWO)
    upper = min((ONE - state_weight) / TWO, increment_weight / FOUR)
    return lower, upper


def transition_entries(
    increment_weight: Decimal, state_weight: Decimal, bridge: Decimal
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    return (
        ONE - state_weight - TWO * bridge,
        bridge,
        (state_weight - increment_weight + TWO * bridge) / TWO,
        increment_weight / TWO - TWO * bridge,
    )


def stationary_bridge(increment_weight: Decimal, state_weight: Decimal) -> Decimal:
    """Find the unique interior entropy-maximizing bridge by bisection."""

    lower, upper = bridge_interval(increment_weight, state_weight)
    if not lower < upper:
        raise ValueError("the reference transition is not in the relative interior")

    def balance(bridge: Decimal) -> Decimal:
        diagonal_zero, bridge, diagonal_nonzero, off_diagonal = transition_entries(
            increment_weight, state_weight, bridge
        )
        return diagonal_zero * off_diagonal * off_diagonal - (
            bridge * bridge * diagonal_nonzero
        )

    for _ in range(320):
        midpoint = (lower + upper) / TWO
        if balance(midpoint) > ZERO:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / TWO


def transition_tangent(
    increment_weight: Decimal, state_weight: Decimal, mixing: Decimal
) -> Tangent:
    """Return a global affine upper bound on maximum transition entropy."""

    uniform = TWO / THREE
    increment_weight = (ONE - mixing) * increment_weight + mixing * uniform
    state_weight = (ONE - mixing) * state_weight + mixing * uniform
    bridge = stationary_bridge(increment_weight, state_weight)
    entries = transition_entries(increment_weight, state_weight, bridge)
    if min(entries) <= ZERO:
        raise ArithmeticError("mixed reference transition is not strictly positive")

    multiplicities = (ONE, FOUR, TWO, TWO)
    # Each tuple gives the constant, increment, state, and bridge coefficient
    # of the corresponding distinct transition entry.
    affine = (
        (ONE, ZERO, -ONE, -TWO),
        (ZERO, ZERO, ZERO, ONE),
        (ZERO, -ONE / TWO, ONE / TWO, ONE),
        (ZERO, ONE / TWO, ZERO, -TWO),
    )
    coefficients = [ZERO, ZERO, ZERO, ZERO]
    entropy = ZERO
    tangent_at_reference = ZERO
    for multiplicity, entry, entry_affine in zip(multiplicities, entries, affine):
        entropy += multiplicity * entropy_atom(entry)
        derivative = -(entry.ln() + ONE)
        tangent_at_reference += multiplicity * derivative * entry
        for coordinate in range(4):
            coefficients[coordinate] += (
                multiplicity * derivative * entry_affine[coordinate]
            )

    intercept = entropy - tangent_at_reference + coefficients[0]
    return Tangent(intercept, coefficients[1], coefficients[2], coefficients[3])


def interval_entropy_atom_value(value: Decimal) -> Interval:
    if value == ZERO:
        return Interval.exact(ZERO)
    point = Interval.exact(value)
    return -(point * point.ln())


def interval_entropy_atom(value: Decimal) -> Interval:
    return interval_entropy_atom_value(value)


def interval_entropy_range(lower: Decimal, upper: Decimal) -> Interval:
    lower = max(ZERO, lower)
    upper = min(ONE, upper)
    candidates = [
        interval_entropy_atom_value(lower),
        interval_entropy_atom_value(upper),
    ]
    reciprocal_e = (-ONE).exp()
    reciprocal_e_interval = Interval(
        reciprocal_e.next_minus(), reciprocal_e.next_plus()
    )
    if (
        lower <= reciprocal_e_interval.upper
        and reciprocal_e_interval.lower <= upper
    ):
        candidates.append(reciprocal_e_interval)
    return Interval(
        min(candidate.lower for candidate in candidates),
        max(candidate.upper for candidate in candidates),
    )


def interval_composition_entropy(nonzero_fraction: Decimal) -> Interval:
    zero = ONE - nonzero_fraction
    half = nonzero_fraction / TWO
    zero_entropy = interval_entropy_range(zero.next_minus(), zero.next_plus())
    half_entropy = interval_entropy_range(half.next_minus(), half.next_plus())
    return zero_entropy + Interval.exact(TWO) * half_entropy


def interval_transition_tangent(
    increment_weight: Decimal, state_weight: Decimal, mixing: Decimal
) -> IntervalTangent:
    """Return an outward-rounded global affine entropy upper bound."""

    uniform = TWO / THREE
    increment_weight = (ONE - mixing) * increment_weight + mixing * uniform
    state_weight = (ONE - mixing) * state_weight + mixing * uniform
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
    coefficients = [Interval.exact(ZERO) for _ in range(4)]
    entropy = Interval.exact(ZERO)
    tangent_at_reference = Interval.exact(ZERO)
    for multiplicity, entry, entry_affine in zip(multiplicities, entries, affine):
        multiplier = Interval.exact(multiplicity)
        entropy = entropy + multiplier * interval_entropy_atom(entry)
        derivative = -(Interval.exact(entry).ln() + Interval.exact(ONE))
        tangent_at_reference = tangent_at_reference + (
            multiplier * derivative * Interval.exact(entry)
        )
        for coordinate in range(4):
            coefficients[coordinate] = coefficients[coordinate] + (
                multiplier * derivative * Interval.exact(entry_affine[coordinate])
            )

    intercept = entropy - tangent_at_reference + coefficients[0]
    return IntervalTangent(
        intercept, coefficients[1], coefficients[2], coefficients[3]
    )


def tighten(box: Box) -> Box | None:
    """Intersect a box with A <= 2B and B <= 2C."""

    lower = list(box.lower)
    upper = list(box.upper)
    for _ in range(3):
        lower[2] = max(
            lower[2],
            (lower[1] / TWO).next_minus(),
            (lower[0] / FOUR).next_minus(),
        )
        lower[1] = max(lower[1], (lower[0] / TWO).next_minus())
        upper[1] = min(upper[1], (TWO * upper[2]).next_plus())
        upper[0] = min(
            upper[0],
            (TWO * upper[1]).next_plus(),
            (FOUR * upper[2]).next_plus(),
        )
        if any(lower[index] > upper[index] for index in range(3)):
            return None
    return Box(tuple(lower), tuple(upper))


def reference_point(box: Box) -> tuple[Decimal, Decimal, Decimal]:
    """Choose a feasible point away from the slanted faces when possible."""

    lower, upper = box.lower, box.upper
    output = (lower[2] + upper[2]) / TWO
    state_lower = max(lower[1], lower[0] / TWO)
    state_upper = min(upper[1], TWO * output)
    state = (state_lower + state_upper) / TWO
    increment_upper = min(upper[0], TWO * state)
    increment = (lower[0] + increment_upper) / TWO
    return increment, state, output


def interval_endpoint_maximum(
    coefficient: Interval,
    entropy_coefficient: Interval,
    lower: Decimal,
    upper: Decimal,
) -> Decimal:
    """Maximize coefficient*x-k*h_3(x), a convex function, on an interval."""

    def value(point: Decimal) -> Decimal:
        result = coefficient * Interval.exact(
            point
        ) - entropy_coefficient * interval_composition_entropy(point)
        return result.upper

    return max(value(lower), value(upper))


def box_bound(box: Box, mixing: Decimal) -> tuple[Decimal, Box] | None:
    box = tighten(box)
    if box is None:
        return None
    increment, state, output = reference_point(box)
    first = interval_transition_tangent(increment, state, mixing)
    second = interval_transition_tangent(state, output, mixing)

    # A bridge always lies in [0,1/4]. The stationary reference makes these
    # coefficients essentially zero, but retaining them keeps the bound valid
    # without trusting the root solver to return the exact cubic root.
    first_allowance = (first.bridge_coefficient.upper / FOUR).next_plus()
    second_allowance = (second.bridge_coefficient.upper / FOUR).next_plus()
    bridge_allowance = Interval.exact(
        max(ZERO, first_allowance) + max(ZERO, second_allowance)
    )
    coefficients = (
        first.increment_coefficient,
        first.state_coefficient + second.increment_coefficient,
        second.state_coefficient,
    )
    entropy_coefficients = (
        rational_interval(TWO, THREE),
        Interval.exact(TWO),
        Interval.exact(ONE),
    )
    upper_bound = first.intercept + second.intercept + bridge_allowance
    for index in range(3):
        upper_bound = upper_bound + Interval.exact(
            interval_endpoint_maximum(
                coefficients[index],
                entropy_coefficients[index],
                box.lower[index],
                box.upper[index],
            )
        )
    return upper_bound.upper, box


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def certify(
    eta: Decimal,
    delta: Decimal,
    precision: int = 80,
    max_boxes: int = 100_000,
) -> Certificate:
    if not ZERO < eta < ONE or not ZERO < delta < ONE:
        raise ValueError("eta and delta must lie in (0,1)")
    if precision < 50:
        raise ValueError("precision must be at least 50 decimal digits")

    with localcontext() as context:
        context.prec = precision
        mixing = Decimal("1e-12")
        initial = Box((eta, ZERO, ZERO), (ONE, ONE, delta))
        bounded = box_bound(initial, mixing)
        if bounded is None:
            raise RuntimeError("the requested domain is empty")

        initial_bound, initial = bounded
        queue: list[tuple[Decimal, int, Box]] = [(-initial_bound, 0, initial)]
        identifier = 1
        boxes_split = 0
        maximum_queue_size = 1

        while queue:
            negative_bound, _, box = heapq.heappop(queue)
            upper_bound = -negative_bound
            if upper_bound < ZERO:
                return Certificate(
                    eta,
                    delta,
                    precision,
                    upper_bound,
                    boxes_split,
                    len(queue) + 1,
                    maximum_queue_size,
                )
            if boxes_split >= max_boxes:
                raise RuntimeError(
                    f"certificate did not close after {max_boxes} box splits; "
                    f"current upper bound is {upper_bound}"
                )

            best_children: list[tuple[Decimal, Box]] | None = None
            best_child_bound: Decimal | None = None
            for dimension in range(3):
                candidates: list[tuple[Decimal, Box]] = []
                for child in split(box, dimension):
                    child_bound = box_bound(child, mixing)
                    if child_bound is not None:
                        candidates.append(child_bound)
                if not candidates:
                    continue
                candidate_maximum = max(bound for bound, _ in candidates)
                if best_child_bound is None or candidate_maximum < best_child_bound:
                    best_child_bound = candidate_maximum
                    best_children = candidates
            if best_children is None:
                raise ArithmeticError("a feasible parent had no feasible child")

            for child_bound, child in best_children:
                heapq.heappush(queue, (-child_bound, identifier, child))
                identifier += 1
            boxes_split += 1
            maximum_queue_size = max(maximum_queue_size, len(queue))

    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=Decimal, default=Decimal("0.0001"))
    parser.add_argument("--delta", type=Decimal, default=Decimal("0.1323"))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    certificate = certify(args.eta, args.delta, args.precision, args.max_boxes)
    print(
        json.dumps(
            {
                "eta": str(certificate.eta),
                "delta": str(certificate.delta),
                "precision": certificate.precision,
                "certified_upper_bound": str(certificate.upper_bound),
                "boxes_split": certificate.boxes_split,
                "boxes_remaining": certificate.boxes_remaining,
                "maximum_queue_size": certificate.maximum_queue_size,
                "arithmetic": "outward-rounded Decimal intervals",
                "status": "negative bulk exponent certified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
