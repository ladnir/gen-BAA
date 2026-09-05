#!/usr/bin/env python3
"""Generate Bernstein bounds for one affine-gate face exponential polynomial.

The default floating-point mode is only a subdivision proposal.  ``--arb``
uses exact rational input data and Arb balls, and refuses to generate a
certificate unless the affine partition and all positivity/multi-affinity
preconditions used by the Taylor remainder bound have been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import arb, ctx, fmpq

from explore_affine_face_interval import (
    FaceBox,
    apply_face,
    serialized_box,
    tighten_simplex_box,
)
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


Exponent = tuple[int, ...]
Polynomial = dict[Exponent, float]
ArbPolynomial = dict[Exponent, arb]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, exact_decimals: bool = False):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=Fraction if exact_decimals else float,
    )


def verify_partition_record(record: dict) -> dict:
    if record.get("partition_identity_exact") is not True:
        raise ValueError("affine partition identity has not been certified")
    if record.get("nonnegative_on_product_box_exact") is not True:
        raise ValueError("affine partition nonnegativity has not been certified")
    denominator = int(record["coefficient_denominator"])
    numerators = record["coefficient_numerators"]
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]
    if denominator <= 0 or not numerators:
        raise ValueError("malformed affine partition")
    if any(len(row) != len(feature_coordinates) + 1 for row in numerators):
        raise ValueError("affine partition coefficient rows have inconsistent widths")
    if any(not isinstance(value, int) for row in numerators for value in row):
        raise ValueError("affine partition coefficients are not integers")
    for column_index, column in enumerate(zip(*numerators, strict=True)):
        expected = denominator if column_index == 0 else 0
        if sum(column) != expected:
            raise ValueError(
                f"affine partition column {column_index} does not sum to {expected}"
            )
    return {
        "partition_identity_replayed": True,
        "source_nonnegative_certificate_present": True,
    }


def exact_face_points(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    vertex: tuple[int, ...],
) -> list[Fraction]:
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    point: list[Fraction | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        value = box.lower[coordinate]
        if coordinate in positions and vertex[positions[coordinate]]:
            value = box.upper[coordinate]
        point[coordinate] = value
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        point[coordinate] = 1 - sum(
            (point[other] for other in range(start, start + 5) if other != coordinate),
            Fraction(0),
        )
    return [value for value in point if value is not None]


def verify_face_product_box(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    record: dict,
    entropy_tangent: bool,
) -> dict:
    vertices = list(itertools.product((0, 1), repeat=len(variables)))
    points = [exact_face_points(box, variables, dependent, vertex) for vertex in vertices]
    minimum_coordinate = min(value for point in points for value in point)
    if minimum_coordinate < 0:
        raise ValueError(f"face product box contains probability {minimum_coordinate}")
    for point in points:
        for start in (0, 5, 10):
            if sum(point[start : start + 5], Fraction(0)) != 1:
                raise ValueError("face product box does not preserve a simplex identity")
    messages = {tuple(point[0:5]) for point in points}
    if len(messages) != 1 and not entropy_tangent:
        raise ValueError("message distribution varies on the claimed fixed-message face")
    if any(value <= 0 for message in messages for value in message):
        raise ValueError("message distribution is not strictly positive on the face box")
    denominator = int(record["coefficient_denominator"])
    coordinates = [int(value) for value in record["feature_coordinates"]]
    minimum_weight: Fraction | None = None
    for point in points:
        weights = [
            (
                row[0] + sum(
                    (coefficient * point[coordinate]
                     for coefficient, coordinate in zip(row[1:], coordinates, strict=True)),
                    Fraction(0),
                )
            ) / denominator
            for row in record["coefficient_numerators"]
        ]
        if sum(weights, Fraction(0)) != 1:
            raise ValueError("affine weights do not sum to one on a face vertex")
        local_minimum = min(weights)
        minimum_weight = (
            local_minimum if minimum_weight is None else min(minimum_weight, local_minimum)
        )
    if minimum_weight is None:
        raise ValueError("face product box has no affine weights")
    return {
        "product_box_probability_minimum": str(minimum_coordinate),
        "product_box_affine_weight_minimum": str(minimum_weight),
        "actual_face_affine_nonnegativity_inherited_from_chamber": True,
        "fixed_message": (
            [str(value) for value in next(iter(messages))]
            if len(messages) == 1
            else None
        ),
        "message_entropy_mode": (
            "concave-tangent-upper" if entropy_tangent else "fixed-exact"
        ),
        "message_entropy_concavity_used": entropy_tangent,
        "vertices_checked": len(vertices),
    }


def constant(value: float, dimension: int) -> Polynomial:
    return {(0,) * dimension: value} if value else {}


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, 0.0) + value
        if result[exponent] == 0.0:
            del result[exponent]
    return result


def scale(polynomial: Polynomial, scalar: float) -> Polynomial:
    return {
        exponent: scalar * value
        for exponent, value in polynomial.items()
        if scalar * value
    }


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_value in left.items():
        for right_exponent, right_value in right.items():
            exponent = tuple(
                a + b for a, b in zip(left_exponent, right_exponent, strict=True)
            )
            result[exponent] = result.get(exponent, 0.0) + left_value * right_value
    return {exponent: value for exponent, value in result.items() if value}


def evaluate(polynomial: Polynomial, point: tuple[float, ...]) -> float:
    return sum(
        coefficient
        * math.prod(value**power for value, power in zip(point, exponent, strict=True))
        for exponent, coefficient in polynomial.items()
    )


def maximum_on_vertices(polynomial: Polynomial, dimension: int) -> float:
    return max(
        evaluate(polynomial, tuple(float(value) for value in vertex))
        for vertex in itertools.product((0, 1), repeat=dimension)
    )


def point_polynomials(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list[Polynomial]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    points: list[Polynomial | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        lower = float(box.lower[coordinate])
        width = float(box.upper[coordinate] - box.lower[coordinate])
        value = constant(lower, dimension)
        if coordinate in positions and width:
            exponent = [0] * dimension
            exponent[positions[coordinate]] = 1
            value = add(value, {tuple(exponent): width})
        points[coordinate] = value
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        value = constant(1.0, dimension)
        for other in range(start, start + 5):
            if other != coordinate:
                value = add(value, scale(points[other], -1.0))
        points[coordinate] = value
    return [value for value in points if value is not None]


def entropy_tangent_polynomial(
    message: list[Polynomial], dimension: int
) -> Polynomial:
    center_point = (0.5,) * dimension
    center = [evaluate(value, center_point) for value in message]
    if any(value <= 0.0 for value in center):
        raise ValueError("message entropy tangent center is not strictly positive")
    entropy = -sum(value * math.log(value) for value in center) / 3.0
    gradient = [-(math.log(value) + 1.0) / 3.0 for value in center]
    tangent = constant(
        entropy - sum(left * right for left, right in zip(gradient, center, strict=True)),
        dimension,
    )
    for slope, coordinate in zip(gradient, message, strict=True):
        tangent = add(tangent, scale(coordinate, slope))
    return tangent


def cut_data(raw: dict) -> tuple[list[float], float, float, list[list[float]]]:
    target = [float(value) for value in raw["target_potential"]]
    increment = float(raw["increment_support_potential"])
    state = float(raw["state_support_potential"])
    kernel = []
    for difference in range(5):
        row = []
        for source in range(5):
            exponent = target[(difference + source) % 5]
            if difference:
                exponent += increment
            if source:
                exponent += state
            row.append(math.exp(exponent))
        kernel.append(row)
    return target, increment, state, kernel


def stage_polynomials(
    left: list[Polynomial],
    right: list[Polynomial],
    cut,
    dimension: int,
) -> tuple[Polynomial, Polynomial]:
    target, increment, state, kernel = cut
    partition: Polynomial = {}
    for difference in range(5):
        for source in range(5):
            partition = add(
                partition,
                scale(
                    multiply(left[difference], right[source]),
                    kernel[difference][source],
                ),
            )
    linear: Polynomial = {}
    for index in range(5):
        linear = add(linear, scale(right[index], target[index]))
    for value in left[1:]:
        linear = add(linear, scale(value, increment))
    for value in right[1:]:
        linear = add(linear, scale(value, state))
    if any(sum(exponent) > 1 for exponent in linear):
        raise ValueError("stage dual exponent is not affine")
    return partition, linear


def affine_center_and_coefficients(polynomial: Polynomial, dimension: int):
    zero = (0,) * dimension
    center = polynomial.get(zero, 0.0)
    coefficients = []
    for position in range(dimension):
        exponent = [0] * dimension
        exponent[position] = 1
        coefficient = polynomial.get(tuple(exponent), 0.0)
        coefficients.append(coefficient)
        center += coefficient / 2.0
    return center, coefficients


def power_to_bernstein(polynomial: Polynomial, degree: int, dimension: int):
    maximum_degrees = [
        max((exponent[position] for exponent in polynomial), default=0)
        for position in range(dimension)
    ]
    if max(maximum_degrees, default=0) > degree:
        raise ValueError("requested Bernstein degree is below the power degree")
    shape = (degree + 1,) * dimension
    power = np.zeros(shape, dtype=np.float64)
    for exponent, coefficient in polynomial.items():
        power[exponent] += coefficient
    transform = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for index in range(degree + 1):
        for power_degree in range(index + 1):
            transform[index, power_degree] = (
                math.comb(index, power_degree) / math.comb(degree, power_degree)
            )
    bernstein = power
    for axis in range(dimension):
        bernstein = np.tensordot(transform, bernstein, axes=(1, axis))
        bernstein = np.moveaxis(bernstein, 0, axis)
    return bernstein, maximum_degrees


def bernstein_bound(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[float]],
    feature_coordinates: list[int],
    expert_pairs: list[dict],
    first_cuts,
    second_cuts,
    taylor_degree: int,
    bernstein_degree: int,
    entropy_tangent: bool,
) -> dict:
    dimension = len(variables)
    point = point_polynomials(box, variables, dependent)
    message = [evaluate(value, (0.5,) * dimension) for value in point[0:5]]
    if entropy_tangent:
        entropy_exponent = entropy_tangent_polynomial(point[0:5], dimension)
        entropy_factor = 1.0
    else:
        entropy_exponent = constant(0.0, dimension)
        entropy_factor = math.exp(
            -sum(value * math.log(value) for value in message) / 3.0
        )
    approximation: Polynomial = {}
    remainder = 0.0
    maximum_rho = 0.0
    for row, pair in zip(coefficients, expert_pairs, strict=True):
        weight = constant(row[0], dimension)
        for coefficient, coordinate in zip(
            row[1:], feature_coordinates, strict=True
        ):
            weight = add(weight, scale(point[coordinate], coefficient))
        first_partition, first_linear = stage_polynomials(
            point[0:5],
            point[5:10],
            first_cuts[int(pair["first_cut"])],
            dimension,
        )
        second_partition, second_linear = stage_polynomials(
            point[5:10],
            point[10:15],
            second_cuts[int(pair["second_cut"])],
            dimension,
        )
        polynomial_factor = multiply(
            multiply(weight, first_partition), second_partition
        )
        exponent = add(
            scale(add(first_linear, second_linear), -1.0), entropy_exponent
        )
        exponent_center, exponent_coefficients = affine_center_and_coefficients(
            exponent, dimension
        )
        centered_exponent = constant(
            -sum(exponent_coefficients) / 2.0, dimension
        )
        for position, coefficient in enumerate(exponent_coefficients):
            power_exponent = [0] * dimension
            power_exponent[position] = 1
            centered_exponent = add(
                centered_exponent, {tuple(power_exponent): coefficient}
            )
        exponential_polynomial = constant(1.0, dimension)
        power = constant(1.0, dimension)
        for order in range(1, taylor_degree + 1):
            power = multiply(power, centered_exponent)
            exponential_polynomial = add(
                exponential_polynomial,
                scale(power, 1.0 / math.factorial(order)),
            )
        exponential_center = math.exp(exponent_center)
        approximation = add(
            approximation,
            scale(
                multiply(polynomial_factor, exponential_polynomial),
                entropy_factor * exponential_center,
            ),
        )
        rho = sum(abs(value) for value in exponent_coefficients) / 2.0
        maximum_rho = max(maximum_rho, rho)
        weight_upper = maximum_on_vertices(weight, dimension)
        first_upper = maximum_on_vertices(first_partition, dimension)
        second_upper = maximum_on_vertices(second_partition, dimension)
        remainder += (
            entropy_factor
            * weight_upper
            * first_upper
            * second_upper
            * math.exp(exponent_center + rho)
            * rho ** (taylor_degree + 1)
            / math.factorial(taylor_degree + 1)
        )
    approximation = add(approximation, constant(-1.0, dimension))
    bernstein, power_degrees = power_to_bernstein(
        approximation, bernstein_degree, dimension
    )
    maximum_coefficient = float(np.max(bernstein))
    maximum_index = tuple(
        int(value)
        for value in np.unravel_index(np.argmax(bernstein), bernstein.shape)
    )
    return {
        "power_degrees": power_degrees,
        "power_terms": len(approximation),
        "bernstein_coefficients": int(bernstein.size),
        "maximum_bernstein_coefficient": maximum_coefficient,
        "maximum_bernstein_index": maximum_index,
        "exponential_taylor_remainder_upper": remainder,
        "maximum_centered_exponent_radius": maximum_rho,
        "mixture_minus_one_upper_proposal": maximum_coefficient + remainder,
        "closed_proposal": maximum_coefficient + remainder < 0.0,
    }


def exact_arb(value) -> arb:
    rational_value = value if isinstance(value, Fraction) else Fraction(str(value))
    return arb(fmpq(rational_value.numerator, rational_value.denominator))


def arb_constant(value, dimension: int) -> ArbPolynomial:
    return {(0,) * dimension: value if isinstance(value, arb) else exact_arb(value)}


def arb_add(left: ArbPolynomial, right: ArbPolynomial) -> ArbPolynomial:
    result = dict(left)
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, arb(0)) + value
    return result


def arb_scale(polynomial: ArbPolynomial, scalar) -> ArbPolynomial:
    scalar = scalar if isinstance(scalar, arb) else exact_arb(scalar)
    return {exponent: scalar * value for exponent, value in polynomial.items()}


def arb_multiply(left: ArbPolynomial, right: ArbPolynomial) -> ArbPolynomial:
    result: ArbPolynomial = {}
    for left_exponent, left_value in left.items():
        for right_exponent, right_value in right.items():
            exponent = tuple(
                a + b for a, b in zip(left_exponent, right_exponent, strict=True)
            )
            result[exponent] = result.get(exponent, arb(0)) + left_value * right_value
    return result


def arb_evaluate(polynomial: ArbPolynomial, point: tuple[int, ...]) -> arb:
    result = arb(0)
    for exponent, coefficient in polynomial.items():
        if all(not power or point[index] for index, power in enumerate(exponent)):
            result += coefficient
    return result


def require_multiaffine(polynomial: ArbPolynomial, dimension: int, label: str) -> None:
    if any(
        len(exponent) != dimension or any(power < 0 or power > 1 for power in exponent)
        for exponent in polynomial
    ):
        raise ValueError(f"{label} is not multi-affine")


def arb_bounds_on_vertices(
    polynomial: ArbPolynomial, dimension: int, label: str
) -> tuple[fmpq, fmpq]:
    require_multiaffine(polynomial, dimension, label)
    values = [
        arb_evaluate(polynomial, vertex)
        for vertex in itertools.product((0, 1), repeat=dimension)
    ]
    minimum_lower = min(value.lower().fmpq() for value in values)
    maximum_upper = max(value.upper().fmpq() for value in values)
    return minimum_lower, maximum_upper


def arb_point_polynomials(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list[ArbPolynomial]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    points: list[ArbPolynomial | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        value = arb_constant(box.lower[coordinate], dimension)
        width = box.upper[coordinate] - box.lower[coordinate]
        if coordinate in positions and width:
            exponent = [0] * dimension
            exponent[positions[coordinate]] = 1
            value = arb_add(
                value, {tuple(exponent): exact_arb(width)}
            )
        points[coordinate] = value
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        value = arb_constant(1, dimension)
        for other in range(start, start + 5):
            if other != coordinate:
                value = arb_add(value, arb_scale(points[other], -1))
        points[coordinate] = value
    return [value for value in points if value is not None]


def arb_entropy_tangent_polynomial(
    message: list[ArbPolynomial], dimension: int
) -> ArbPolynomial:
    center_point = (arb(fmpq(1, 2)),) * dimension
    center = [arb_evaluate(value, center_point) for value in message]
    if any(value.lower().fmpq() <= 0 for value in center):
        raise ValueError("message entropy tangent center is not strictly positive")
    entropy = -sum((value * value.log() for value in center), arb(0)) / 3
    gradient = [-(value.log() + 1) / 3 for value in center]
    tangent = arb_constant(
        entropy
        - sum(
            (left * right for left, right in zip(gradient, center, strict=True)),
            arb(0),
        ),
        dimension,
    )
    for slope, coordinate in zip(gradient, message, strict=True):
        tangent = arb_add(tangent, arb_scale(coordinate, slope))
    return tangent


def arb_cut_data(raw: dict):
    target = [exact_arb(value) for value in raw["target_potential"]]
    increment = exact_arb(raw["increment_support_potential"])
    state = exact_arb(raw["state_support_potential"])
    kernel = []
    for difference in range(5):
        row = []
        for source in range(5):
            exponent = target[(difference + source) % 5]
            if difference:
                exponent += increment
            if source:
                exponent += state
            row.append(exponent.exp())
        kernel.append(row)
    return target, increment, state, kernel


def arb_stage_polynomials(
    left: list[ArbPolynomial],
    right: list[ArbPolynomial],
    cut,
    dimension: int,
) -> tuple[ArbPolynomial, ArbPolynomial]:
    target, increment, state, kernel = cut
    partition: ArbPolynomial = {}
    for difference in range(5):
        for source in range(5):
            partition = arb_add(
                partition,
                arb_scale(
                    arb_multiply(left[difference], right[source]),
                    kernel[difference][source],
                ),
            )
    linear: ArbPolynomial = {}
    for index in range(5):
        linear = arb_add(linear, arb_scale(right[index], target[index]))
    for value in left[1:]:
        linear = arb_add(linear, arb_scale(value, increment))
    for value in right[1:]:
        linear = arb_add(linear, arb_scale(value, state))
    return partition, linear


def arb_affine_center_and_coefficients(
    polynomial: ArbPolynomial, dimension: int
):
    center = polynomial.get((0,) * dimension, arb(0))
    coefficients = []
    for position in range(dimension):
        exponent = [0] * dimension
        exponent[position] = 1
        coefficient = polynomial.get(tuple(exponent), arb(0))
        coefficients.append(coefficient)
        center += coefficient / 2
    return center, coefficients


def arb_power_to_bernstein(
    polynomial: ArbPolynomial, degree: int, dimension: int
):
    maximum_degrees = [
        max((exponent[position] for exponent in polynomial), default=0)
        for position in range(dimension)
    ]
    if max(maximum_degrees, default=0) > degree:
        raise ValueError("requested Bernstein degree is below the power degree")
    shape = (degree + 1,) * dimension
    power = np.empty(shape, dtype=object)
    power.fill(arb(0))
    for exponent, coefficient in polynomial.items():
        power[exponent] = power[exponent] + coefficient
    transform = np.empty((degree + 1, degree + 1), dtype=object)
    transform.fill(arb(0))
    for index in range(degree + 1):
        for power_degree in range(index + 1):
            transform[index, power_degree] = arb(
                fmpq(math.comb(index, power_degree), math.comb(degree, power_degree))
            )
    bernstein = power
    for axis in range(dimension):
        bernstein = np.tensordot(transform, bernstein, axes=(1, axis))
        bernstein = np.moveaxis(bernstein, 0, axis)
    return bernstein, maximum_degrees


def arb_bernstein_bound(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficient_numerators: list[list[int]],
    coefficient_denominator: int,
    feature_coordinates: list[int],
    expert_pairs: list[dict],
    first_cuts,
    second_cuts,
    taylor_degree: int,
    bernstein_degree: int,
    entropy_tangent: bool,
) -> dict:
    dimension = len(variables)
    point = arb_point_polynomials(box, variables, dependent)
    message = [arb_evaluate(value, (0,) * dimension) for value in point[0:5]]
    if entropy_tangent:
        entropy_exponent = arb_entropy_tangent_polynomial(point[0:5], dimension)
        entropy_factor = arb(1)
    else:
        entropy_exponent = arb_constant(0, dimension)
        entropy = -sum((value * value.log() for value in message), arb(0)) / 3
        entropy_factor = entropy.exp()
    approximation: ArbPolynomial = {}
    remainder = arb(0)
    maximum_rho_upper = fmpq(0)
    minimum_weight_lower: fmpq | None = None
    minimum_partition_lower: fmpq | None = None
    for numerator_row, pair in zip(
        coefficient_numerators, expert_pairs, strict=True
    ):
        row = [
            arb(fmpq(int(value), coefficient_denominator))
            for value in numerator_row
        ]
        weight = arb_constant(row[0], dimension)
        for coefficient, coordinate in zip(
            row[1:], feature_coordinates, strict=True
        ):
            weight = arb_add(weight, arb_scale(point[coordinate], coefficient))
        first_partition, first_linear = arb_stage_polynomials(
            point[0:5],
            point[5:10],
            first_cuts[int(pair["first_cut"])],
            dimension,
        )
        second_partition, second_linear = arb_stage_polynomials(
            point[5:10],
            point[10:15],
            second_cuts[int(pair["second_cut"])],
            dimension,
        )
        polynomial_factor = arb_multiply(
            arb_multiply(weight, first_partition), second_partition
        )
        weight_lower, weight_upper = arb_bounds_on_vertices(
            weight, dimension, "affine expert weight"
        )
        first_lower, first_upper = arb_bounds_on_vertices(
            first_partition, dimension, "first-stage partition factor"
        )
        second_lower, second_upper = arb_bounds_on_vertices(
            second_partition, dimension, "second-stage partition factor"
        )
        if first_lower <= 0 or second_lower <= 0:
            raise ValueError(
                "stage partition factor is not certified strictly positive: "
                f"{first_lower}, {second_lower}"
            )
        minimum_weight_lower = (
            weight_lower
            if minimum_weight_lower is None
            else min(minimum_weight_lower, weight_lower)
        )
        local_partition_lower = min(first_lower, second_lower)
        minimum_partition_lower = (
            local_partition_lower
            if minimum_partition_lower is None
            else min(minimum_partition_lower, local_partition_lower)
        )
        exponent = arb_add(
            arb_scale(arb_add(first_linear, second_linear), -1),
            entropy_exponent,
        )
        exponent_center, exponent_coefficients = arb_affine_center_and_coefficients(
            exponent, dimension
        )
        centered_exponent = arb_constant(
            -sum(exponent_coefficients, arb(0)) / 2, dimension
        )
        for position, coefficient in enumerate(exponent_coefficients):
            power_exponent = [0] * dimension
            power_exponent[position] = 1
            centered_exponent = arb_add(
                centered_exponent, {tuple(power_exponent): coefficient}
            )
        exponential_polynomial = arb_constant(1, dimension)
        power = arb_constant(1, dimension)
        for order in range(1, taylor_degree + 1):
            power = arb_multiply(power, centered_exponent)
            exponential_polynomial = arb_add(
                exponential_polynomial,
                arb_scale(power, fmpq(1, math.factorial(order))),
            )
        exponential_center = exponent_center.exp()
        approximation = arb_add(
            approximation,
            arb_scale(
                arb_multiply(polynomial_factor, exponential_polynomial),
                entropy_factor * exponential_center,
            ),
        )
        rho = sum((abs(value) for value in exponent_coefficients), arb(0)) / 2
        maximum_rho_upper = max(maximum_rho_upper, rho.upper().fmpq())
        remainder += (
            entropy_factor
            * arb(max(-weight_lower, weight_upper))
            * arb(first_upper)
            * arb(second_upper)
            * (exponent_center + rho).exp()
            * rho ** (taylor_degree + 1)
            / math.factorial(taylor_degree + 1)
        )
    approximation = arb_add(approximation, arb_constant(-1, dimension))
    bernstein, power_degrees = arb_power_to_bernstein(
        approximation, bernstein_degree, dimension
    )
    maximum_upper = None
    maximum_index = None
    for flat_index, value in enumerate(bernstein.flat):
        upper = value.upper().fmpq()
        if maximum_upper is None or upper > maximum_upper:
            maximum_upper = upper
            maximum_index = tuple(
                int(index)
                for index in np.unravel_index(flat_index, bernstein.shape)
            )
    remainder_upper = remainder.upper().fmpq()
    total_upper = maximum_upper + remainder_upper
    return {
        "arithmetic": "python-flint Arb ball arithmetic",
        "precision_bits": ctx.prec,
        "power_degrees": power_degrees,
        "power_terms": len(approximation),
        "bernstein_coefficients": int(bernstein.size),
        "maximum_bernstein_coefficient_upper": str(maximum_upper),
        "maximum_bernstein_index": maximum_index,
        "exponential_taylor_remainder_upper": str(remainder_upper),
        "maximum_centered_exponent_radius_upper": str(maximum_rho_upper),
        "minimum_affine_weight_lower": str(minimum_weight_lower),
        "affine_weight_absolute_value_used_for_remainder": True,
        "minimum_stage_partition_lower": str(minimum_partition_lower),
        "mixture_minus_one_upper": str(total_upper),
        "closed": total_upper < 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("--face", required=True)
    parser.add_argument("--taylor-degree", type=int, default=8)
    parser.add_argument("--bernstein-degree", type=int, default=12)
    parser.add_argument("--dyadic-level", type=int, default=0)
    parser.add_argument(
        "--dyadic-coordinate",
        action="append",
        type=int,
        default=[],
        help="split only this free coordinate at each dyadic level (repeatable)",
    )
    parser.add_argument(
        "--dyadic-step",
        action="append",
        type=int,
        default=[],
        help="append one coordinate to an explicit ordered dyadic split schedule",
    )
    parser.add_argument("--arb", action="store_true")
    parser.add_argument("--precision-bits", type=int, default=192)
    parser.add_argument("--box-index", type=int)
    parser.add_argument(
        "--entropy-tangent",
        action="store_true",
        help="upper-bound varying message entropy by its affine tangent on each box",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only a compact completion summary (the output artifact is unchanged)",
    )
    args = parser.parse_args()
    if (
        args.taylor_degree < 0
        or args.bernstein_degree < 1
        or args.dyadic_level < 0
        or args.precision_bits < 80
    ):
        parser.error("degrees must be nonnegative")
    if args.box_index is not None and args.box_index < 1:
        parser.error("box index is one-based")
    if args.arb:
        ctx.prec = args.precision_bits

    manifest = load_json(args.manifest, args.arb)
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    face_file = load_json(args.faces, args.arb)
    leaf_name = face_file["leaf"]
    leaf = leaves[leaf_name]
    face = next(value for value in face_file["faces"] if value["name"] == args.face)
    partitions = load_json(args.partitions, args.arb)
    record = next(value for value in partitions["leaves"] if value["leaf"] == leaf_name)
    partition_checks = verify_partition_record(record)
    denominator = int(record["coefficient_denominator"])
    coefficients = [
        [float(Fraction(int(value), denominator)) for value in row]
        for row in record["coefficient_numerators"]
    ]
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]

    frontier = load_json(root / manifest["global_checkpoint"], args.arb)
    selected = sorted(frontier["queue"], key=lambda value: value["priority"])[0]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate_path = root / str(record.get("gate", leaf["gate"])).replace("\\", "/")
    gate = load_json(gate_path, args.arb)
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    box, variables, dependent = apply_face(lower, upper, face)
    split_coordinates = args.dyadic_coordinate or variables
    if args.dyadic_step and (args.dyadic_level != 0 or args.dyadic_coordinate):
        parser.error("--dyadic-step cannot be combined with levels or coordinates")
    if len(set(split_coordinates)) != len(split_coordinates):
        parser.error("--dyadic-coordinate values must be distinct")
    if any(value not in variables for value in split_coordinates):
        parser.error("--dyadic-coordinate must name a free face coordinate")
    if any(value not in variables for value in args.dyadic_step):
        parser.error("--dyadic-step must name a free face coordinate")
    base_face_box = box
    face_checks = verify_face_product_box(
        box, variables, dependent, record, args.entropy_tangent
    )

    bank = load_json(root / "results/f5-projected-kl-dual-bank-0.22.json", args.arb)
    cut_loader = arb_cut_data if args.arb else cut_data
    first_cuts = [cut_loader(value) for value in bank["first_cuts"]]
    second_cuts = [cut_loader(value) for value in bank["second_cuts"]]
    boxes = [box]
    dyadic_schedule = (
        list(args.dyadic_step)
        if args.dyadic_step
        else split_coordinates * args.dyadic_level
    )
    for coordinate in dyadic_schedule:
        children = []
        for parent in boxes:
            threshold = (
                parent.lower[coordinate] + parent.upper[coordinate]
            ) / 2
            for left in (True, False):
                child_lower = list(parent.lower)
                child_upper = list(parent.upper)
                if left:
                    child_upper[coordinate] = threshold
                else:
                    child_lower[coordinate] = threshold
                child = tighten_simplex_box(
                    FaceBox(tuple(child_lower), tuple(child_upper)),
                    variables,
                    dependent,
                )
                if child is not None:
                    children.append(child)
        boxes = children
    total_boxes = len(boxes)
    if args.box_index is not None:
        if args.box_index > total_boxes:
            parser.error("box index exceeds the generated cover")
        boxes = [boxes[args.box_index - 1]]
    box_records = []
    for index, subbox in enumerate(boxes, start=1):
        if args.arb:
            result = arb_bernstein_bound(
                subbox,
                variables,
                dependent,
                record["coefficient_numerators"],
                denominator,
                feature_coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
                args.taylor_degree,
                args.bernstein_degree,
                args.entropy_tangent,
            )
            result["upper_sort_value"] = float(
                Fraction(result["mixture_minus_one_upper"])
            )
            result["closed_result"] = bool(result["closed"])
        else:
            result = bernstein_bound(
                subbox,
                variables,
                dependent,
                coefficients,
                feature_coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
                args.taylor_degree,
                args.bernstein_degree,
                args.entropy_tangent,
            )
            result["upper_sort_value"] = result[
                "mixture_minus_one_upper_proposal"
            ]
            result["closed_result"] = bool(result["closed_proposal"])
        box_records.append({"box": serialized_box(subbox), **result})
        if len(boxes) > 1 and not args.quiet:
            print(
                json.dumps(
                    {
                        "box": index,
                        "boxes": len(boxes),
                        "upper": result["upper_sort_value"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    worst = max(
        box_records,
        key=lambda value: value["upper_sort_value"],
    )
    payload = {
        "status": (
            "Arb Bernstein exponential-polynomial certificate"
            if args.arb
            else "floating Bernstein exponential-polynomial proposal; not a proof"
        ),
        "rigorous": args.arb,
        "leaf": leaf_name,
        "face": args.face,
        "variables": variables,
        "dependent_coordinates": dependent,
        "base_face_box": serialized_box(base_face_box),
        "proof_preconditions": {
            **partition_checks,
            **face_checks,
            "stage_factors_checked_multiaffine": args.arb,
            "stage_factors_checked_strictly_positive": args.arb,
        },
        "source_sha256": {
            "manifest": sha256_file(args.manifest),
            "partitions": sha256_file(args.partitions),
            "faces": sha256_file(args.faces),
            "dual_bank": sha256_file(root / "results/f5-projected-kl-dual-bank-0.22.json"),
            "gate": sha256_file(gate_path),
        },
        "taylor_degree": args.taylor_degree,
        "bernstein_degree": args.bernstein_degree,
        "entropy_mode": (
            "concave-tangent-upper" if args.entropy_tangent else "fixed-exact"
        ),
        "dyadic_level": args.dyadic_level,
        "dyadic_coordinates": split_coordinates,
        "dyadic_schedule": dyadic_schedule,
        "total_cover_boxes": total_boxes,
        "selected_box_index": args.box_index,
        "boxes": len(boxes),
        "closed_boxes": sum(value["closed_result"] for value in box_records),
        "mixture_minus_one_upper": (
            worst.get("mixture_minus_one_upper")
            if args.arb
            else worst["mixture_minus_one_upper_proposal"]
        ),
        "closed": all(value["closed_result"] for value in box_records),
        "worst_box": worst,
        "box_records": box_records,
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    if args.quiet:
        print(
            json.dumps(
                {
                    "boxes": payload["boxes"],
                    "closed": payload["closed"],
                    "closed_boxes": payload["closed_boxes"],
                    "mixture_minus_one_upper": payload["mixture_minus_one_upper"],
                    "output": str(args.output) if args.output is not None else None,
                    "rigorous": payload["rigorous"],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
