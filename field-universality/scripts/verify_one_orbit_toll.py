#!/usr/bin/env python3
"""Check the exact margin supplied by the one-sign-orbit information toll."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

from mpmath import iv


def mpf_fraction(raw: tuple[int, int, int, int]) -> Fraction:
    sign, mantissa, exponent, _ = raw
    value = Fraction(mantissa)
    value = value * 2**exponent if exponent >= 0 else value / 2 ** (-exponent)
    return -value if sign else value


def interval(value: Fraction) -> object:
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def interval_lower(value: object) -> Fraction:
    return mpf_fraction(value._mpi_[0])


def verify(args: argparse.Namespace) -> dict[str, object]:
    iv.dps = args.precision
    log_two_lower = interval_lower(iv.log(interval(Fraction(2))))
    toll = args.eta * log_two_lower / 6
    if toll <= args.support_upper_bound:
        raise RuntimeError("the one-orbit toll does not exceed the support bound")
    margin = toll - args.support_upper_bound
    return {
        "status": "verified negative one-sign-orbit exponent",
        "eta": str(args.eta),
        "support_upper_bound": str(args.support_upper_bound),
        "log_two_lower_numerator": str(log_two_lower.numerator),
        "log_two_lower_denominator": str(log_two_lower.denominator),
        "information_toll_lower_bound": float(toll),
        "negative_exponent_margin": float(margin),
        "margin_numerator": str(margin.numerator),
        "margin_denominator": str(margin.denominator),
        "arithmetic": "directed logarithm interval and exact rational comparison",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 500))
    parser.add_argument(
        "--support-upper-bound", type=Fraction, default=Fraction(1, 10_000)
    )
    parser.add_argument("--precision", type=int, default=80)
    args = parser.parse_args()
    if args.eta <= 0:
        parser.error("eta must be positive")
    if args.support_upper_bound < 0:
        parser.error("support-upper-bound must be nonnegative")
    return args


def main() -> None:
    print(json.dumps(verify(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
