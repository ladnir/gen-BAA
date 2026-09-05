#!/usr/bin/env python3
"""Generate one rational Perron certificate from floating spectral fits.

The generator is untrusted.  It uses SciPy only to propose fugacities and
positive Collatz--Wielandt vectors.  ``verify_perron_point_certificate.py``
replays the resulting certificate with rational arithmetic and directed
logarithm intervals.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

import numpy as np
from scipy.linalg import eig

from explore_perron_bound import fit_contracted_cut, fit_first_cut


def exact_type() -> dict[str, object]:
    zero = Fraction(0)
    message = (
        Fraction(499, 500),
        Fraction(1, 1000),
        zero,
        zero,
        Fraction(1, 1000),
    )
    output = (
        Fraction(8677, 10_000),
        Fraction(3, 50),
        Fraction(123, 20_000),
        Fraction(123, 20_000),
        Fraction(3, 50),
    )

    first = [[zero for _ in range(5)] for _ in range(5)]
    first[0][0] = Fraction(997, 1000)
    for source, target in ((0, 1), (1, 0), (0, 4), (4, 0)):
        first[source][target] = Fraction(1, 2000)
    first[1][1] = Fraction(1, 2000)
    first[4][4] = Fraction(1, 2000)

    second = [[zero for _ in range(5)] for _ in range(5)]
    for state in range(5):
        second[state][state] = output[state] - Fraction(2, 5000)
        second[state][(state + 1) % 5] = Fraction(1, 5000)
        second[(state + 1) % 5][state] = Fraction(1, 5000)

    return {
        "message": message,
        "middle": message,
        "output": output,
        "first_transition": tuple(tuple(row) for row in first),
        "second_transition": tuple(tuple(row) for row in second),
    }


def rationalize_positive(value: float, denominator_bound: int) -> Fraction:
    numerator = max(1, round(float(value) * denominator_bound))
    result = Fraction(numerator, denominator_bound)
    if result <= 0:
        raise RuntimeError("positive floating certificate data rounded to zero")
    return result


def rational_matrix(
    increment: tuple[Fraction, ...], state: tuple[Fraction, ...]
) -> list[list[Fraction]]:
    prime = len(increment)
    return [
        [increment[(target - source) % prime] * state[target] for target in range(prime)]
        for source in range(prime)
    ]


def propose_collatz_vector(
    increment: tuple[Fraction, ...], state: tuple[Fraction, ...], denominator_bound: int
) -> tuple[Fraction, ...]:
    matrix = np.array(
        [[float(value) for value in row] for row in rational_matrix(increment, state)]
    )
    values, vectors = eig(matrix)
    right = vectors[:, int(np.argmax(values.real))].real
    if right.sum() < 0.0:
        right = -right
    right /= right[0]
    return tuple(
        rationalize_positive(float(value), denominator_bound) for value in right
    )


def collatz_bound(
    matrix: list[list[Fraction]], vector: tuple[Fraction, ...]
) -> Fraction:
    return max(
        sum(
            matrix[source][target] * vector[target]
            for target in range(len(vector))
        )
        / vector[source]
        for source in range(len(vector))
    )


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def vector_text(values: tuple[Fraction, ...]) -> list[str]:
    return [fraction_text(value) for value in values]


def matrix_text(values: tuple[tuple[Fraction, ...], ...]) -> list[list[str]]:
    return [vector_text(row) for row in values]


def generate(bits: int, vector_bits: int) -> dict[str, object]:
    point = exact_type()
    message = np.array([float(value) for value in point["message"]])
    middle = np.array([float(value) for value in point["middle"]])
    delta = Fraction(1323, 10_000)
    first_fit = fit_first_cut(message, middle, 40.0)
    second_fit = fit_contracted_cut(middle, float(delta), 40.0)

    denominator_bound = 1 << bits
    vector_denominator_bound = 1 << vector_bits
    first_increment = tuple(
        rationalize_positive(float(np.exp(value)), denominator_bound)
        for value in first_fit.log_increment
    )
    first_state = tuple(
        rationalize_positive(float(np.exp(value)), denominator_bound)
        for value in first_fit.log_state
    )
    second_increment = tuple(
        rationalize_positive(float(np.exp(value)), denominator_bound)
        for value in second_fit.log_increment
    )
    q = rationalize_positive(float(np.exp(second_fit.log_q)), denominator_bound)
    second_state = (Fraction(1),) + (q,) * 4

    first_vector = propose_collatz_vector(
        first_increment, first_state, vector_denominator_bound
    )
    second_vector = propose_collatz_vector(
        second_increment, second_state, vector_denominator_bound
    )
    first_matrix = rational_matrix(first_increment, first_state)
    second_matrix = rational_matrix(second_increment, second_state)
    first_lambda = collatz_bound(first_matrix, first_vector)
    second_lambda = collatz_bound(second_matrix, second_vector)

    return {
        "schema": "f5-perron-point-v1",
        "prime": 5,
        "repetition": 3,
        "delta": fraction_text(delta),
        "exact_type": {
            "message": vector_text(point["message"]),
            "middle": vector_text(point["middle"]),
            "output": vector_text(point["output"]),
            "first_transition": matrix_text(point["first_transition"]),
            "second_transition": matrix_text(point["second_transition"]),
        },
        "first_cut": {
            "increment_fugacity": vector_text(first_increment),
            "state_fugacity": vector_text(first_state),
            "collatz_vector": vector_text(first_vector),
            "lambda": fraction_text(first_lambda),
        },
        "contracted_second_cut": {
            "increment_fugacity": vector_text(second_increment),
            "q": fraction_text(q),
            "collatz_vector": vector_text(second_vector),
            "lambda": fraction_text(second_lambda),
        },
        "generation": {
            "status": "untrusted floating proposal",
            "fugacity_denominator_bits": bits,
            "vector_denominator_bits": vector_bits,
            "first_fit_gradient_residual": first_fit.gradient_residual,
            "second_fit_gradient_residual": second_fit.gradient_residual,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=60)
    parser.add_argument("--vector-bits", type=int, default=50)
    args = parser.parse_args()
    if args.bits < 20 or args.vector_bits < 20:
        parser.error("certificate denominators need at least 20 bits")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(generate(arguments.bits, arguments.vector_bits), indent=2))
