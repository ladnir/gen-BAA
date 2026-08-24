#!/usr/bin/env python3
"""Exact spacing identities used by the ternary sparse-distance proof.

The functions in this file use integer arithmetic for the finite counting
identities.  Floating-point arithmetic appears only in the human-readable
summary printed by the command-line interface.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from fractions import Fraction


def spacing_weight_count(length: int, transitions: int, active: int, weight: int) -> int:
    """Count transition supports whose selected post-transition gaps sum to weight."""
    if not 1 <= active <= transitions <= length:
        return 0
    if not active <= weight <= length - (transitions - active):
        return 0
    return math.comb(weight - 1, active - 1) * math.comb(
        length - weight, transitions - active
    )


def spacing_tail_count(length: int, transitions: int, active: int, bound: int) -> int:
    """Count transition supports whose selected gap sum is at most bound."""
    return sum(
        spacing_weight_count(length, transitions, active, weight)
        for weight in range(active, min(bound, length) + 1)
    )


def hypergeometric_tail_count(
    length: int, transitions: int, active: int, bound: int
) -> int:
    """Count supports having at least active positions in a fixed bound-set."""
    if bound < 0 or bound > length:
        return 0
    return sum(
        math.comb(bound, selected) * math.comb(length - bound, transitions - selected)
        for selected in range(active, min(transitions, bound) + 1)
        if transitions - selected <= length - bound
    )


def nonzero_interval_count(value_order: tuple[int, ...], prime: int = 3) -> int:
    """Return the number of nonzero states following the listed transitions."""
    state = 0
    active = 0
    for value in value_order:
        if value % prime == 0:
            raise ValueError("value_order must contain only nonzero increments")
        state = (state + value) % prime
        active += state != 0
    return active


def conditional_weight_histogram(
    length: int, value_order: tuple[int, ...], prime: int = 3
) -> Counter[int]:
    """Enumerate accumulator weights over all supports for a fixed value order."""
    transitions = len(value_order)
    histogram: Counter[int] = Counter()
    for positions in itertools.combinations(range(length), transitions):
        increments = [0] * length
        for position, value in zip(positions, value_order, strict=True):
            increments[position] = value
        state = 0
        weight = 0
        for value in increments:
            state = (state + value) % prime
            weight += state != 0
        histogram[weight] += 1
    return histogram


def laplace_polynomial(
    length: int, transitions: int, active: int, numerator: int, denominator: int
) -> Fraction:
    """Evaluate E[x^W] exactly at x=numerator/denominator."""
    x = Fraction(numerator, denominator)
    total = math.comb(length, transitions)
    return sum(
        Fraction(spacing_weight_count(length, transitions, active, weight), total)
        * x**weight
        for weight in range(active, length - (transitions - active) + 1)
    )


def laplace_upper_bound(
    length: int, transitions: int, active: int, numerator: int, denominator: int
) -> Fraction:
    """Return the negative-binomial upper bound for E[x^W]."""
    x = Fraction(numerator, denominator)
    falling_ratio = Fraction(1, 1)
    for offset in range(active):
        falling_ratio *= Fraction(transitions - offset, length - offset)
    return falling_ratio * (x / (1 - x)) ** active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.13)
    parser.add_argument("--eta", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta = args.delta
    eta = args.eta
    min_ratio = 3.0 / eta
    divergence = -0.5 * math.log(4.0 * delta * (1.0 - delta))
    laplace_denominator = math.exp(divergence) - 1.0
    exact_base = (
        2.0
        * math.e
        * min_ratio
        / 3.0
        * (3.0 / ((min_ratio - 3.0) * laplace_denominator)) ** 1.5
    )

    exact_proof_parameters = delta == 0.13 and eta == 0.0001
    rational_certificate = None
    if exact_proof_parameters:
        # exp(D(1/2 || 13/100))-1 > 8/17 and e<3.  Squaring the
        # resulting base bound proves that it is less than 1/5.
        ratio_integer = 30_000
        rational_left = 100 * ratio_integer**2 * 51**3
        rational_right = 512 * (ratio_integer - 3) ** 3
        rational_certificate = {
            "base_upper_bound": 0.2,
            "left": rational_left,
            "right": rational_right,
            "passes": rational_left < rational_right,
        }
    print(
        json.dumps(
            {
                "delta": delta,
                "eta": eta,
                "minimum_N_over_s": min_ratio,
                "D_half_parallel_delta": divergence,
                "exp_D_minus_one": laplace_denominator,
                "geometric_base": exact_base,
                "rational_certificate": rational_certificate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
