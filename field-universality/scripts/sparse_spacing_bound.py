#!/usr/bin/env python3
"""Exact spacing identities used by the prime-field sparse-distance proofs.

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


def hamming_weight(word: tuple[int, ...]) -> int:
    return sum(value != 0 for value in word)


def accumulate(word: tuple[int, ...], prime: int) -> tuple[int, ...]:
    state = 0
    output = []
    for value in word:
        state = (state + value) % prime
        output.append(state)
    return tuple(output)


def ladder_integer_check(prime: int, denominator_root: int) -> bool:
    """Check the repetition-three distance criterion at delta=1/m^2."""
    m = denominator_root
    if prime < 2 or m < 4:
        return False
    return 288 * (prime - 1) ** 2 * m**4 < (m - 2) ** 6 * (m + 2) ** 3


def least_ladder_denominator_root(prime: int) -> int:
    """Return the least integer m certified by ladder_integer_check."""
    m = 4
    while not ladder_integer_check(prime, m):
        m += 1
    return m


def binary_benchmark_sparse_certificate(
    prime: int, eta: Fraction
) -> dict[str, object]:
    """Return the exact sparse check at delta=1323/10000."""
    benchmark = Fraction(1323, 10_000)
    radicand = 4 * benchmark * (1 - benchmark)
    sqrt_check = radicand * 59**2 < 40**2
    spacing_lower = Fraction(19, 40)
    q_squared_upper = (
        Fraction((prime - 1) ** 2 * 121, 16)
        * eta
        / ((1 - eta) ** 3 * spacing_lower**3)
    )
    return {
        "domain": (
            f"p = {prime}, r = 3, 0 < delta <= 1323/10000, "
            f"and 0 < eta <= {eta}"
        ),
        "base_upper_bound": 1.0,
        "sqrt_check": {
            "left": (radicand * 59**2).numerator,
            "left_denominator": (radicand * 59**2).denominator,
            "right": 40**2,
            "passes": sqrt_check,
        },
        "spacing_denominator_lower": {
            "numerator": spacing_lower.numerator,
            "denominator": spacing_lower.denominator,
        },
        "q_squared_upper": {
            "numerator": q_squared_upper.numerator,
            "denominator": q_squared_upper.denominator,
        },
        "passes": sqrt_check and q_squared_upper < 1,
    }


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
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=Fraction, default=Fraction(49, 250))
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 10_000))
    return parser.parse_args()


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1 if divisor == 2 else 2
    return True


def main() -> None:
    args = parse_args()
    prime = args.prime
    repetition = args.repetition
    delta_fraction = args.delta
    eta_fraction = args.eta
    delta = float(delta_fraction)
    eta = float(eta_fraction)
    if not is_prime(prime):
        raise SystemExit("prime must be a prime integer")
    if repetition < 3:
        raise SystemExit("repetition must be at least three")
    if not 0.0 < delta < 0.5:
        raise SystemExit("delta must lie strictly between 0 and 0.5")
    if not 0.0 < eta < 1.0:
        raise SystemExit("eta must lie strictly between 0 and 1")
    min_ratio = repetition / eta
    divergence = -0.5 * math.log(4.0 * delta * (1.0 - delta))
    laplace_denominator = math.exp(divergence) - 1.0
    exact_base = (
        (prime - 1.0)
        * math.e
        * min_ratio
        / repetition
        * (
            repetition
            / ((min_ratio - repetition) * laplace_denominator)
        )
        ** (repetition / 2.0)
    )

    rational_certificate = None
    ladder_root = least_ladder_denominator_root(prime)
    ladder_delta = 1.0 / ladder_root**2
    ladder_eta = 4.0 / ladder_root**2
    endpoint_tolerance = 1e-14
    if (
        repetition == 3
        and delta <= ladder_delta * (1.0 + endpoint_tolerance)
        and eta <= ladder_eta * (1.0 + endpoint_tolerance)
    ):
        rational_left = 288 * (prime - 1) ** 2 * ladder_root**4
        rational_right = (ladder_root - 2) ** 6 * (ladder_root + 2) ** 3
        rational_certificate = {
            "domain": (
                f"p = {prime}, r = 3, 0 < delta <= 1/{ladder_root**2}, "
                f"and 0 < eta <= 4/{ladder_root**2}"
            ),
            "base_upper_bound": 1.0,
            "denominator_root": ladder_root,
            "left": rational_left,
            "right": rational_right,
            "passes": rational_left < rational_right,
        }
    elif prime == 3 and repetition == 3 and delta <= 0.2 and eta <= 0.0001:
        # For delta <= 1/5, exp(D(1/2 || delta))-1 >= 1/4.  Together
        # with e<3, squaring the worst-case base bound proves that it is
        # less than 1/2.
        ratio_integer = 30_000
        rational_left = 16 * ratio_integer**2 * 12**3
        rational_right = (ratio_integer - 3) ** 3
        rational_certificate = {
            "domain": "0 < delta <= 1/5 and 0 < eta <= 1/10000",
            "base_upper_bound": 0.5,
            "ratio_check": {"left": 12, "right": 29_997, "passes": 12 < 29_997},
            "left": rational_left,
            "right": rational_right,
            "passes": 12 < 29_997 and rational_left < rational_right,
        }
    elif prime == 5 and repetition == 3 and delta <= 0.04 and eta <= 0.025:
        # For delta <= 1/25, sqrt(4*delta*(1-delta)) < 2/5, so
        # exp(D)-1 > 3/2.  The elementary series bound e<11/4 then
        # reduces Q<1 to the exact integer comparison below.
        rational_left = 440**2 * 2**3
        rational_right = 117**3
        rational_certificate = {
            "domain": (
                "p = 5, r = 3, 0 < delta <= 1/25, "
                "and 0 < eta <= 1/40"
            ),
            "base_upper_bound": 1.0,
            "sqrt_check": {"left": 4**2 * 6, "right": 10**2, "passes": 96 < 100},
            "e_check": {"left": 196, "right": 198, "passes": 196 < 198},
            "spacing_ratio": {"numerator": 2, "denominator": 117},
            "left": rational_left,
            "right": rational_right,
            "passes": 96 < 100 and 196 < 198 and rational_left < rational_right,
        }
    elif (
        prime == 5
        and repetition == 3
        and delta <= 0.05
        and eta <= float(Fraction(1, 60)) * (1.0 + endpoint_tolerance)
    ):
        # For delta <= 1/20, sqrt(4*delta*(1-delta)) < 7/16, so
        # exp(D)-1 > 9/7.  With e<11/4 and eta<=1/60, Q<1 is the
        # exact integer comparison below.
        rational_left = 660**2 * 7**3
        rational_right = 531**3
        rational_certificate = {
            "domain": (
                "p = 5, r = 3, 0 < delta <= 1/20, "
                "and 0 < eta <= 1/60"
            ),
            "base_upper_bound": 1.0,
            "sqrt_check": {"left": 19 * 8**2, "right": 35**2, "passes": 1216 < 1225},
            "e_check": {"left": 196, "right": 198, "passes": 196 < 198},
            "spacing_ratio": {"numerator": 7, "denominator": 531},
            "left": rational_left,
            "right": rational_right,
            "passes": (
                1216 < 1225
                and 196 < 198
                and rational_left < rational_right
            ),
        }
    elif (
        prime == 7
        and repetition == 3
        and delta <= 0.018
        and eta <= 0.0625 * (1.0 + endpoint_tolerance)
    ):
        # At delta=9/500, sqrt(4*delta*(1-delta)) < 133/500,
        # so exp(D)-1 > 367/133.  With eta<=1/16 and e<11/4,
        # Q<1 reduces to the exact integer comparison below.
        rational_left = 264**2 * 133**3
        rational_right = 5_505**3
        rational_certificate = {
            "domain": (
                "p = 7, r = 3, 0 < delta <= 9/500, "
                "and 0 < eta <= 1/16"
            ),
            "base_upper_bound": 1.0,
            "sqrt_check": {"left": 4 * 4_419, "right": 133**2, "passes": 17_676 < 17_689},
            "e_check": {"left": 196, "right": 198, "passes": 196 < 198},
            "spacing_ratio": {"numerator": 133, "denominator": 5_505},
            "left": rational_left,
            "right": rational_right,
            "passes": (
                17_676 < 17_689
                and 196 < 198
                and rational_left < rational_right
            ),
        }
    elif (
        repetition == 3
        and delta_fraction <= Fraction(1323, 10_000)
    ):
        # At delta=1323/10000, monotonicity on (0,1/2) and the integer
        # comparison below give
        #
        #   sqrt(4 delta (1-delta)) < 40/59,
        #   exp(D(1/2 || delta))-1 > 19/40.
        #
        # Together with e<11/4, squaring the geometric base reduces Q<1
        # to an exact rational comparison for any supplied eta.
        rational_certificate = binary_benchmark_sparse_certificate(
            prime, eta_fraction
        )
    elif prime == 5 and repetition == 3 and delta <= 0.05 and eta <= 0.0001:
        # For delta <= 1/20, sqrt(4*delta*(1-delta)) <= sqrt(19)/10
        # and sqrt(19) < 5.  Hence exp(D)-1 > 1.  Together with
        # e<3, the integer inequality below makes the base less than 1/5.
        ratio_integer = 30_000
        rational_left = 400 * ratio_integer**2 * 3**3
        rational_right = (ratio_integer - 3) ** 3
        rational_certificate = {
            "domain": (
                "p = 5, r = 3, 0 < delta <= 1/20, "
                "and 0 < eta <= 1/10000"
            ),
            "base_upper_bound": 0.2,
            "sqrt_check": {"left": 19, "right": 5**2, "passes": 19 < 5**2},
            "ratio_check": {"left": 3, "right": 29_997, "passes": 3 < 29_997},
            "left": rational_left,
            "right": rational_right,
            "passes": (
                19 < 5**2
                and 3 < 29_997
                and rational_left < rational_right
            ),
        }
    elif prime == 5 and repetition == 3 and delta <= 0.26 and eta <= 0.00001:
        # For delta <= 13/50, sqrt(4*delta*(1-delta)) <= sqrt(481)/25
        # and sqrt(481) < 22.  Hence exp(D)-1 > 3/22.  Together with
        # e<3, the integer inequality below makes the base less than 4/5.
        ratio_integer = 300_000
        rational_left = 25 * ratio_integer**2 * 22**3
        rational_right = (ratio_integer - 3) ** 3
        rational_certificate = {
            "domain": (
                "p = 5, r = 3, 0 < delta <= 13/50, "
                "and 0 < eta <= 1/100000"
            ),
            "base_upper_bound": 0.8,
            "sqrt_check": {"left": 481, "right": 22**2, "passes": 481 < 22**2},
            "ratio_check": {"left": 22, "right": 299_997, "passes": 22 < 299_997},
            "left": rational_left,
            "right": rational_right,
            "passes": (
                481 < 22**2
                and 22 < 299_997
                and rational_left < rational_right
            ),
        }
    print(
        json.dumps(
            {
                "prime": prime,
                "repetition": repetition,
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
