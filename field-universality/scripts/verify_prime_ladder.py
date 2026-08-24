#!/usr/bin/env python3
"""Independently check the integer certificates in the prime-ladder theorem.

This verifier deliberately imports no code from ``sparse_spacing_bound.py``.
It checks only exact integers.  The analytic reduction from the displayed
integer inequality to the distance theorem is proved in ``CLEAN_THEORY.md``.
"""

from __future__ import annotations

import argparse
import json


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def certificate_sides(prime: int, root: int) -> tuple[int, int]:
    left = 288 * (prime - 1) * (prime - 1) * root**4
    right = (root - 2) ** 6 * (root + 2) ** 3
    return left, right


def least_root(prime: int) -> int:
    root = 4
    while True:
        left, right = certificate_sides(prime, root)
        if left < right:
            return root
        root += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--root", type=int)
    args = parser.parse_args()
    if not is_prime(args.prime):
        parser.error("prime must be a prime integer")
    if args.root is not None and args.root < 4:
        parser.error("root must be at least four")
    return args


def main() -> None:
    args = parse_args()
    minimum = least_root(args.prime)
    root = minimum if args.root is None else args.root
    left, right = certificate_sides(args.prime, root)
    previous_left, previous_right = certificate_sides(args.prime, minimum - 1)
    result = {
        "prime": args.prime,
        "root": root,
        "relative_distance": f"1/{root**2}",
        "left": left,
        "right": right,
        "passes": left < right,
        "least_certified_root": minimum,
        "minimality_check": {
            "root": minimum - 1,
            "left": previous_left,
            "right": previous_right,
            "passes": previous_left >= previous_right,
        },
        "arithmetic": "exact integers",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
