#!/usr/bin/env python3
"""Certify a support-indicator upper bound for the small bulk regime.

The checker uses directed logarithm intervals and exact rational box
arithmetic.  It proves negativity of the field-uniform data-processing bound

  (h(A) + A log(p-1))/r - J(A,B) - J(B,C),

where J(a,b) is the mutual information forced by stationarity after retaining
only the zero/nonzero indicators.  On the domain used here, the minimizing
crossing mass is a/2.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from mpmath import iv


ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, Fraction, Fraction]
    upper: tuple[Fraction, Fraction, Fraction]


def mpf_fraction(raw: tuple[int, int, int, int]) -> Fraction:
    sign, mantissa, exponent, _ = raw
    value = Fraction(mantissa)
    value = value * 2**exponent if exponent >= 0 else value / 2 ** (-exponent)
    return -value if sign else value


def interval(value: Fraction) -> object:
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def interval_lower(value: object) -> Fraction:
    return mpf_fraction(value._mpi_[0])


def interval_upper(value: object) -> Fraction:
    return mpf_fraction(value._mpi_[1])


@lru_cache(maxsize=None)
def negative_log_upper(value: Fraction) -> Fraction:
    if value <= ZERO:
        raise ValueError("logarithm argument must be positive")
    return interval_upper(-iv.log(interval(value)))


@lru_cache(maxsize=None)
def entropy_atom_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    if value == ZERO:
        return ZERO, ZERO
    point = interval(value)
    image = -point * iv.log(point)
    return interval_lower(image), interval_upper(image)


def binary_entropy_lower(lower: Fraction, upper: Fraction) -> Fraction:
    left = entropy_atom_bounds(lower)[0] + entropy_atom_bounds(ONE - lower)[0]
    right = entropy_atom_bounds(upper)[0] + entropy_atom_bounds(ONE - upper)[0]
    return min(left, right)


def joint_cross_entropy_coefficients(
    increment: Fraction, state: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """Return constant, increment, and state coefficients of an entropy cut."""
    atoms = (
        ONE - state - increment / TWO,
        state - increment / TWO,
        increment / TWO,
        increment / TWO,
    )
    # Mixing with the uniform distribution keeps every reference atom
    # positive, including on a slanted feasibility face.  Cross entropy at any
    # positive reference distribution is a global entropy upper bound.
    mixing = Fraction(1, 1 << 40)
    reference = tuple(
        (ONE - mixing) * atom + mixing / 4 for atom in atoms
    )
    coefficients = tuple(negative_log_upper(atom) for atom in reference)
    constant = coefficients[0]
    increment_coefficient = (
        -coefficients[0]
        - coefficients[1]
        + coefficients[2]
        + coefficients[3]
    ) / TWO
    state_coefficient = -coefficients[0] + coefficients[1]
    return constant, increment_coefficient, state_coefficient


def endpoint_convex_maximum(
    linear: Fraction,
    entropy_coefficient: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> Fraction:
    def value(point: Fraction) -> Fraction:
        entropy_lower = binary_entropy_lower(point, point)
        return linear * point - entropy_coefficient * entropy_lower

    return max(value(lower), value(upper))


def tighten(box: Box) -> Box | None:
    lower = list(box.lower)
    upper = list(box.upper)
    for _ in range(4):
        lower[1] = max(lower[1], lower[0] / TWO)
        lower[2] = max(lower[2], lower[1] / TWO)
        upper[0] = min(upper[0], TWO * upper[1])
        upper[1] = min(upper[1], TWO * upper[2])
        if any(lower[index] > upper[index] for index in range(3)):
            return None
    return Box(tuple(lower), tuple(upper))


def box_bound(
    raw_box: Box, prime: int, repetition: int
) -> tuple[Fraction, int, Box] | None:
    box = tighten(raw_box)
    if box is None:
        return None
    reference = tuple(
        (box.lower[index] + box.upper[index]) / TWO for index in range(3)
    )
    first = joint_cross_entropy_coefficients(reference[0], reference[1])
    second = joint_cross_entropy_coefficients(reference[1], reference[2])
    log_field = interval_upper(iv.log(iv.mpf(prime - 1)))
    linear = (
        first[1] + log_field / repetition,
        first[2] + second[1],
        second[2],
    )
    entropy_coefficients = (
        Fraction(repetition - 1, repetition),
        TWO,
        ONE,
    )
    upper = first[0] + second[0]
    for index in range(3):
        upper += endpoint_convex_maximum(
            linear[index],
            entropy_coefficients[index],
            box.lower[index],
            box.upper[index],
        )

    # Split the coordinate with the largest affine range.  Cross-entropy cuts
    # are refreshed after every split, so this also tracks tangent slack.
    widths = [box.upper[index] - box.lower[index] for index in range(3)]
    scores = [
        widths[index] * (ONE + abs(linear[index]))
        for index in range(3)
    ]
    return upper, max(range(3), key=scores.__getitem__), box


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / TWO
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1 if divisor == 2 else 2
    return True


def search(args: argparse.Namespace) -> dict[str, object]:
    iv.dps = args.precision
    initial_box = Box(
        (args.eta, args.eta / TWO, args.eta / 4),
        (args.cutoff, TWO * args.delta, args.delta),
    )
    initial = box_bound(initial_box, args.prime, args.repetition)
    if initial is None:
        raise RuntimeError("initial support-indicator domain is infeasible")
    queue = [(-initial[0], 0, initial[1], initial[2])]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        negative_bound, _, dimension, box = heapq.heappop(queue)
        bound = -negative_bound
        if bound < ZERO:
            return {
                "status": "verified negative support-indicator cover",
                "upper_bound": float(bound),
                "upper_bound_numerator": str(bound.numerator),
                "upper_bound_denominator": str(bound.denominator),
                "upper_bound_is_exactly_negative": True,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
            }
        for child in split(box, dimension):
            result = box_bound(child, args.prime, args.repetition)
            if result is None:
                continue
            child_bound, child_dimension, child_box = result
            heapq.heappush(
                queue,
                (-child_bound, identifier, child_dimension, child_box),
            )
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    return {
        "status": "exact support-indicator cover incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=Fraction, default=Fraction(1323, 10_000))
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 2000))
    parser.add_argument("--cutoff", type=Fraction, default=Fraction(1, 500))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    args = parser.parse_args()
    if args.prime < 3 or not is_prime(args.prime):
        parser.error("prime must be an odd prime")
    if args.repetition < 1:
        parser.error("repetition must be positive")
    if not ZERO < args.eta <= args.cutoff:
        parser.error("require 0 < eta <= cutoff")
    if not ZERO < args.delta < Fraction(1, 4):
        parser.error("support reduction currently requires delta < 1/4")
    return args


def main() -> None:
    args = parse_args()
    result = search(args)
    result.update(
        {
            "prime": args.prime,
            "repetition": args.repetition,
            "delta": str(args.delta),
            "eta": str(args.eta),
            "cutoff": str(args.cutoff),
            "interval_precision": args.precision,
            "arithmetic": "directed logarithm intervals and exact rational boxes",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
