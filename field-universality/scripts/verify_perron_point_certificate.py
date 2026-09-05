#!/usr/bin/env python3
"""Independently replay one rational Perron point certificate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from mpmath import iv


ZERO = Fraction(0)
ONE = Fraction(1)
LOG_BITS = 120


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def scale(self, coefficient: Fraction) -> "Interval":
        if coefficient >= ZERO:
            return Interval(coefficient * self.lower, coefficient * self.upper)
        return Interval(coefficient * self.upper, coefficient * self.lower)


def parse_fraction(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("certificate rationals must be strings")
    result = Fraction(value)
    if result.denominator <= 0:
        raise ValueError("invalid rational denominator")
    return result


def parse_vector(value: object, length: int) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected a rational vector of length {length}")
    return tuple(parse_fraction(entry) for entry in value)


def parse_matrix(value: object, length: int) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected a rational matrix with {length} rows")
    return tuple(parse_vector(row, length) for row in value)


def mpf_fraction(raw: tuple[int, int, int, int]) -> Fraction:
    sign, mantissa, exponent, _ = raw
    value = Fraction(mantissa)
    value = value * 2**exponent if exponent >= 0 else value / 2 ** (-exponent)
    return -value if sign else value


def rational_interval(value: object) -> Interval:
    scale = 1 << LOG_BITS

    def round_down(exact: Fraction) -> Fraction:
        return Fraction(exact.numerator // exact.denominator, scale)

    def round_up(exact: Fraction) -> Fraction:
        return Fraction(-((-exact.numerator) // exact.denominator), scale)

    lower = mpf_fraction(value._mpi_[0]) * scale
    upper = mpf_fraction(value._mpi_[1]) * scale
    return Interval(round_down(lower), round_up(upper))


def iv_fraction(value: Fraction) -> object:
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def log_interval(value: Fraction) -> Interval:
    if value <= ZERO:
        raise ValueError("logarithm input must be positive")
    return rational_interval(iv.log(iv_fraction(value)))


def entropy(values: tuple[Fraction, ...]) -> Interval:
    result = Interval(ZERO, ZERO)
    for value in values:
        if value < ZERO:
            raise ValueError("distribution contains a negative mass")
        if value > ZERO:
            result += log_interval(value).scale(-value)
    return result


def flatten(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[Fraction, ...]:
    return tuple(value for row in matrix for value in row)


def check_distribution(values: tuple[Fraction, ...], name: str) -> None:
    if any(value < ZERO for value in values) or sum(values, ZERO) != ONE:
        raise ValueError(f"{name} is not a probability distribution")


def transition_marginals(
    transition: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    prime = len(transition)
    row = tuple(sum(transition[source], ZERO) for source in range(prime))
    column = tuple(
        sum(transition[source][target] for source in range(prime))
        for target in range(prime)
    )
    increment = tuple(
        sum(
            transition[source][(source + difference) % prime]
            for source in range(prime)
        )
        for difference in range(prime)
    )
    return row, column, increment


def check_transition(
    transition: tuple[tuple[Fraction, ...], ...],
    state: tuple[Fraction, ...],
    increment: tuple[Fraction, ...],
    name: str,
) -> None:
    check_distribution(flatten(transition), name)
    row, column, observed_increment = transition_marginals(transition)
    if row != state or column != state or observed_increment != increment:
        raise ValueError(f"{name} has incorrect exact marginals")


def transfer_matrix(
    increment: tuple[Fraction, ...], state: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], ...]:
    prime = len(increment)
    return tuple(
        tuple(
            increment[(target - source) % prime] * state[target]
            for target in range(prime)
        )
        for source in range(prime)
    )


def check_collatz(
    increment: tuple[Fraction, ...],
    state: tuple[Fraction, ...],
    vector: tuple[Fraction, ...],
    claimed_lambda: Fraction,
    name: str,
) -> Fraction:
    if any(value <= ZERO for value in (*increment, *state, *vector)):
        raise ValueError(f"{name} certificate data must be positive")
    matrix = transfer_matrix(increment, state)
    exact_lambda = max(
        sum(
            matrix[source][target] * vector[target]
            for target in range(len(vector))
        )
        / vector[source]
        for source in range(len(vector))
    )
    if claimed_lambda != exact_lambda:
        raise ValueError(f"{name} does not store the exact rational row bound")
    for source in range(len(vector)):
        left = sum(
            matrix[source][target] * vector[target]
            for target in range(len(vector))
        )
        if left > claimed_lambda * vector[source]:
            raise ValueError(f"{name} violates its Collatz inequality")
    return vector[0] / min(vector)


def exact_type_exponent(
    message: tuple[Fraction, ...],
    middle: tuple[Fraction, ...],
    output: tuple[Fraction, ...],
    first_transition: tuple[tuple[Fraction, ...], ...],
    second_transition: tuple[tuple[Fraction, ...], ...],
) -> Interval:
    result = entropy(flatten(first_transition))
    result += entropy(flatten(second_transition))
    result += entropy(message).scale(Fraction(-2, 3))
    result += entropy(middle).scale(Fraction(-2))
    result += entropy(output).scale(Fraction(-1))
    return result


def perron_point_upper(
    message: tuple[Fraction, ...],
    middle: tuple[Fraction, ...],
    delta: Fraction,
    first_increment: tuple[Fraction, ...],
    first_state: tuple[Fraction, ...],
    first_lambda: Fraction,
    second_increment: tuple[Fraction, ...],
    q: Fraction,
    second_lambda: Fraction,
) -> Interval:
    result = log_interval(first_lambda) + log_interval(second_lambda)
    result += log_interval(q).scale(-delta)
    result += entropy(message).scale(Fraction(-2, 3))
    result += entropy(middle).scale(Fraction(-1))
    for mass, fugacity in zip(message, first_increment, strict=True):
        result += log_interval(fugacity).scale(-mass)
    for mass, first_fugacity, second_fugacity in zip(
        middle, first_state, second_increment, strict=True
    ):
        result += log_interval(first_fugacity).scale(-mass)
        result += log_interval(second_fugacity).scale(-mass)
    return result


def verify(raw: object, precision: int) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-perron-point-v1":
        raise ValueError("unsupported Perron certificate schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("certificate has unexpected code parameters")
    iv.dps = precision
    delta = parse_fraction(raw["delta"])
    point = raw["exact_type"]
    if not isinstance(point, dict):
        raise ValueError("missing exact type")
    message = parse_vector(point["message"], 5)
    middle = parse_vector(point["middle"], 5)
    output = parse_vector(point["output"], 5)
    first_transition = parse_matrix(point["first_transition"], 5)
    second_transition = parse_matrix(point["second_transition"], 5)
    for values, name in (
        (message, "message"),
        (middle, "middle"),
        (output, "output"),
    ):
        check_distribution(values, name)
    check_transition(first_transition, middle, message, "first transition")
    check_transition(second_transition, output, middle, "second transition")
    if sum(output[1:], ZERO) > delta:
        raise ValueError("exact type exceeds the output-weight threshold")

    first = raw["first_cut"]
    second = raw["contracted_second_cut"]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("missing Perron cut")
    first_increment = parse_vector(first["increment_fugacity"], 5)
    first_state = parse_vector(first["state_fugacity"], 5)
    first_vector = parse_vector(first["collatz_vector"], 5)
    first_lambda = parse_fraction(first["lambda"])
    second_increment = parse_vector(second["increment_fugacity"], 5)
    q = parse_fraction(second["q"])
    if q > ONE:
        raise ValueError("contracted output fugacity must not exceed one")
    second_vector = parse_vector(second["collatz_vector"], 5)
    second_lambda = parse_fraction(second["lambda"])
    first_prefactor = check_collatz(
        first_increment,
        first_state,
        first_vector,
        first_lambda,
        "first cut",
    )
    second_prefactor = check_collatz(
        second_increment,
        (ONE,) + (q,) * 4,
        second_vector,
        second_lambda,
        "contracted second cut",
    )

    exact_exponent = exact_type_exponent(
        message,
        middle,
        output,
        first_transition,
        second_transition,
    )
    point_upper = perron_point_upper(
        message,
        middle,
        delta,
        first_increment,
        first_state,
        first_lambda,
        second_increment,
        q,
        second_lambda,
    )
    if exact_exponent.upper >= ZERO:
        raise ValueError("exact transition type was not verified negative")
    if point_upper.upper >= ZERO:
        raise ValueError("rational Perron point upper was not verified negative")
    total_prefactor = first_prefactor * second_prefactor
    return {
        "status": "verified rational Perron point certificate",
        "exact_type_exponent_lower": float(exact_exponent.lower),
        "exact_type_exponent_upper": float(exact_exponent.upper),
        "perron_point_upper_lower": float(point_upper.lower),
        "perron_point_upper_upper": float(point_upper.upper),
        "perron_point_upper_is_exactly_negative": True,
        "finite_length_prefactor": f"{total_prefactor.numerator}/{total_prefactor.denominator}",
        "interval_precision": precision,
        "arithmetic": "exact rational Collatz inequalities and directed logarithm intervals",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        type=Path,
        default=Path("results/f5-perron-point-certificate.json"),
    )
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    return args


def main() -> None:
    args = parse_args()
    raw = json.loads(args.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(raw, args.precision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
