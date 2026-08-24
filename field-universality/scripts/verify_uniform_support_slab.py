#!/usr/bin/env python3
"""Check the rational constants in the all-primes small-message theorem."""

from __future__ import annotations

import json
from fractions import Fraction


def verify() -> dict[str, object]:
    delta = Fraction(1323, 10_000)
    radicand = 4 * delta * (1 - delta)

    # The elementary exponential-series bounds used in CLEAN_THEORY.md are
    # e < 11/4 and e > 1+1+1/2+1/6+1/24 = 65/24 > 8/3.
    e_upper = Fraction(11, 4)
    e_lower_series = Fraction(65, 24)
    e_lower_target = Fraction(8, 3)

    # e < 11/4 and (11/4)^3 < radicand^-4 imply
    # log(1/radicand) > 3/4.
    log_lower_left = e_upper**3 * radicand**4

    # This square comparison gives exp(D)-1 > 19/40 for the sparse bound.
    spacing_denominator_check = radicand * 59**2 < 40**2

    # At A <= 1/(60q^2), the support exponent is negative once
    # e * 60 * (3/16)^3 > 1.  Replace e by the rational lower bound 8/3.
    support_margin_lower = e_lower_target * 60 * Fraction(3, 16) ** 3

    # For eta=1/(100q^2) and q>=2, the squared sparse geometric base is at
    # most this rational number.  Here 19/40 lower-bounds exp(D)-1,
    # 11/4 upper-bounds e, and 1-eta >= 399/400.
    sparse_base_squared_upper = (
        Fraction(121, 1600)
        / (Fraction(399, 400) ** 3 * Fraction(19, 40) ** 3)
    )

    checks = {
        "e_lower_series": e_lower_series > e_lower_target,
        "log_one_over_radicand_gt_three_quarters": log_lower_left < 1,
        "spacing_denominator_gt_nineteen_fortieth": spacing_denominator_check,
        "support_margin": support_margin_lower > 1,
        "sparse_base_squared": sparse_base_squared_upper < 1,
        "slabs_overlap": Fraction(1, 100) < Fraction(1, 60),
    }
    return {
        "status": (
            "verified all-primes support-slab constants"
            if all(checks.values())
            else "constant check failed"
        ),
        "checks": checks,
        "delta": str(delta),
        "sparse_endpoint": "1/(100*(p-1)^2)",
        "support_endpoint": "1/(60*(p-1)^2)",
        "radicand": str(radicand),
        "e_upper": str(e_upper),
        "e_lower_series": str(e_lower_series),
        "log_lower_check": {
            "numerator": log_lower_left.numerator,
            "denominator": log_lower_left.denominator,
        },
        "spacing_denominator_check": {
            "left_numerator": (radicand * 59**2).numerator,
            "left_denominator": (radicand * 59**2).denominator,
            "right": 40**2,
        },
        "support_margin_lower": {
            "numerator": support_margin_lower.numerator,
            "denominator": support_margin_lower.denominator,
        },
        "sparse_base_squared_upper": {
            "numerator": sparse_base_squared_upper.numerator,
            "denominator": sparse_base_squared_upper.denominator,
        },
        "arithmetic": "integer and rational comparisons only",
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
