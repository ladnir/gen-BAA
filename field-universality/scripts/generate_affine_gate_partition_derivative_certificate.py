#!/usr/bin/env python3
"""Generate an exact partition-aware output-derivative certificate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

from flint import arb, ctx, fmpq

from explore_affine_face_interval import FaceBox, serialized_box, tighten_simplex_box
from rationalize_affine_cover import (
    chamber_bounds,
    collect_leaves,
    exact_simplex_vertices,
    rational,
)


Exponent = tuple[int, ...]
Polynomial = dict[Exponent, arb]


def load_exact(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Fraction)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def as_ball(value) -> arb:
    value = q(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value: fmpq) -> Fraction:
    return Fraction(str(value))


def const(value, dimension: int) -> Polynomial:
    return {(0,) * dimension: value if isinstance(value, arb) else as_ball(value)}


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, arb(0)) + coefficient
    return result


def scale(polynomial: Polynomial, scalar) -> Polynomial:
    scalar = scalar if isinstance(scalar, arb) else as_ball(scalar)
    return {exponent: scalar * coefficient for exponent, coefficient in polynomial.items()}


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(
                a + b for a, b in zip(left_exponent, right_exponent, strict=True)
            )
            result[exponent] = (
                result.get(exponent, arb(0)) + left_coefficient * right_coefficient
            )
    return result


def derivative(polynomial: Polynomial, position: int) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        power = exponent[position]
        if not power:
            continue
        reduced = list(exponent)
        reduced[position] -= 1
        key = tuple(reduced)
        result[key] = result.get(key, arb(0)) + power * coefficient
    return result


def point_polynomials(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list[Polynomial]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    points: list[Polynomial | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        value = const(box.lower[coordinate], dimension)
        if coordinate in positions and box.lower[coordinate] != box.upper[coordinate]:
            exponent = [0] * dimension
            exponent[positions[coordinate]] = 1
            value = add(
                value,
                {tuple(exponent): as_ball(box.upper[coordinate] - box.lower[coordinate])},
            )
        points[coordinate] = value
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        value = const(1, dimension)
        for other in range(start, start + 5):
            if other != coordinate:
                value = add(value, scale(points[other], -1))
        points[coordinate] = value
    return [value for value in points if value is not None]


def load_cut(raw: dict):
    target = [as_ball(value) for value in raw["target_potential"]]
    increment = as_ball(raw["increment_support_potential"])
    state = as_ball(raw["state_support_potential"])
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


def stage(left: list[Polynomial], right: list[Polynomial], cut, dimension: int):
    target, increment, state, kernel = cut
    partition: Polynomial = {}
    for difference in range(5):
        for source in range(5):
            partition = add(
                partition,
                scale(multiply(left[difference], right[source]), kernel[difference][source]),
            )
    linear: Polynomial = {}
    for index in range(5):
        linear = add(linear, scale(right[index], target[index]))
    for value in left[1:]:
        linear = add(linear, scale(value, increment))
    for value in right[1:]:
        linear = add(linear, scale(value, state))
    return partition, linear


def vertex_value(polynomial: Polynomial, vertex: tuple[int, ...]) -> arb:
    total = arb(0)
    for exponent, coefficient in polynomial.items():
        if all(power == 0 or vertex[index] for index, power in enumerate(exponent)):
            total += coefficient
    return total


def vertex_bounds(
    polynomial: Polynomial, vertices: list[tuple[int, ...]], *, multiaffine: bool
) -> tuple[Fraction, Fraction]:
    if multiaffine and any(
        any(power < 0 or power > 1 for power in exponent) for exponent in polynomial
    ):
        raise ValueError("claimed multi-affine polynomial has higher powers")
    values = [vertex_value(polynomial, vertex) for vertex in vertices]
    return (
        min(endpoint(value.lower().fmpq()) for value in values),
        max(endpoint(value.upper().fmpq()) for value in values),
    )


def exact_weight_bounds(
    record: dict, box: FaceBox
) -> tuple[list[Fraction], list[Fraction], int]:
    blocks = [
        exact_simplex_vertices(list(box.lower), list(box.upper), start)
        for start in (0, 5, 10)
    ]
    points = [
        (*product[0], *product[1], *product[2])
        for product in itertools.product(*blocks)
    ]
    denominator = int(record["coefficient_denominator"])
    coordinates = [int(value) for value in record["feature_coordinates"]]
    lowers = []
    uppers = []
    for row in record["coefficient_numerators"]:
        values = [
            (
                row[0]
                + sum(
                    (coefficient * point[coordinate]
                     for coefficient, coordinate in zip(row[1:], coordinates, strict=True)),
                    Fraction(0),
                )
            )
            / denominator
            for point in points
        ]
        lowers.append(min(values))
        uppers.append(max(values))
    if min(lowers) < 0 or sum(lowers, Fraction(0)) > 1 or sum(uppers, Fraction(0)) < 1:
        raise ValueError("invalid exact affine-weight ranges")
    return lowers, uppers, len(points)


def weight_derivative(
    row: list[int],
    denominator: int,
    feature_coordinates: list[int],
    coordinate: int,
    dependent: list[int],
    box: FaceBox,
) -> Fraction:
    coefficients = {
        feature_coordinate: Fraction(value, denominator)
        for value, feature_coordinate in zip(
            row[1:], feature_coordinates, strict=True
        )
    }
    block_start = (coordinate // 5) * 5
    block_dependent = next(
        value for value in dependent if block_start <= value < block_start + 5
    )
    width = box.upper[coordinate] - box.lower[coordinate]
    return width * (
        coefficients.get(coordinate, Fraction(0))
        - coefficients.get(block_dependent, Fraction(0))
    )


def bounded_simplex_minimum(
    lowers: list[Fraction], uppers: list[Fraction], costs: list[Fraction]
) -> tuple[Fraction, list[Fraction]]:
    weights = list(lowers)
    remaining = 1 - sum(weights, Fraction(0))
    for index in sorted(range(len(costs)), key=costs.__getitem__):
        take = min(remaining, uppers[index] - weights[index])
        weights[index] += take
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError("affine-weight intervals do not contain a simplex point")
    return sum((weight * cost for weight, cost in zip(weights, costs, strict=True)), Fraction(0)), weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--coordinate", type=int, required=True, choices=range(11, 15))
    parser.add_argument("--fix", action="append", default=[])
    parser.add_argument("--dependent-coordinates", default="0,5,10")
    parser.add_argument("--precision-bits", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision_bits < 128:
        parser.error("precision must be at least 128 bits")
    ctx.prec = args.precision_bits
    dependent = [int(value) for value in args.dependent_coordinates.split(",")]
    if len(dependent) != 3 or args.coordinate in dependent:
        parser.error("invalid dependent-coordinate choice")

    manifest = load_exact(args.manifest)
    partitions = load_exact(args.partitions)
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[args.leaf]
    record = next(value for value in partitions["leaves"] if value["leaf"] == args.leaf)
    if not record.get("partition_identity_exact") or not record.get(
        "nonnegative_on_product_box_exact"
    ):
        raise ValueError("affine partition lacks exact source certification")
    gate_path = root / str(record.get("gate", leaf["gate"])).replace("\\", "/")
    gate = load_exact(gate_path)
    checkpoint = load_exact(root / manifest["global_checkpoint"])
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    fixed: set[int] = set()
    for specification in args.fix:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in dependent or coordinate == args.coordinate:
            parser.error("fixed coordinate conflicts with derivative coordinates")
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            parser.error("fixed side must be lower or upper")
    variables = [
        coordinate
        for coordinate in range(15)
        if coordinate not in fixed and coordinate not in dependent
    ]
    box = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if box is None:
        raise ValueError("derivative domain is empty")
    position = variables.index(args.coordinate)
    dimension = len(variables)
    vertices = list(itertools.product((0, 1), repeat=dimension))
    points = point_polynomials(box, variables, dependent)
    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    bank = load_exact(bank_path)
    first_cuts = [load_cut(value) for value in bank["first_cuts"]]
    second_cuts = [load_cut(value) for value in bank["second_cuts"]]
    weight_lowers, weight_uppers, weight_vertices = exact_weight_bounds(record, box)
    denominator = int(record["coefficient_denominator"])
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]
    expert_records = []
    gate_lower = Fraction(0)
    response_costs = []
    for expert, (row, pair) in enumerate(
        zip(record["coefficient_numerators"], record["expert_pairs"], strict=True)
    ):
        first_partition, first_linear = stage(
            points[0:5], points[5:10], first_cuts[int(pair["first_cut"])], dimension
        )
        second_partition, second_linear = stage(
            points[5:10], points[10:15], second_cuts[int(pair["second_cut"])], dimension
        )
        response = add(
            derivative(second_partition, position),
            scale(
                second_partition,
                -derivative(second_linear, position).get((0,) * dimension, arb(0)),
            ),
        )
        active_lower, active_upper = vertex_bounds(
            second_partition, vertices, multiaffine=True
        )
        inactive_lower, inactive_upper = vertex_bounds(
            first_partition, vertices, multiaffine=True
        )
        response_lower, response_upper = vertex_bounds(
            response, vertices, multiaffine=True
        )
        total_linear = add(first_linear, second_linear)
        exponent = scale(total_linear, -1)
        exponent_lower, exponent_upper = vertex_bounds(
            exponent, vertices, multiaffine=True
        )
        exponential_lower = endpoint(as_ball(exponent_lower).exp().lower().fmpq())
        exponential_upper = endpoint(as_ball(exponent_upper).exp().upper().fmpq())
        scale_lower = inactive_lower * exponential_lower
        scale_upper = inactive_upper * exponential_upper
        value_lower = scale_lower * active_lower
        value_upper = scale_upper * active_upper
        derivative_coefficient = weight_derivative(
            row,
            denominator,
            feature_coordinates,
            args.coordinate,
            dependent,
            box,
        )
        gate_contribution = derivative_coefficient * (
            value_lower if derivative_coefficient >= 0 else value_upper
        )
        response_cost = response_lower * (
            scale_lower if response_lower >= 0 else scale_upper
        )
        gate_lower += gate_contribution
        response_costs.append(response_cost)
        expert_records.append(
            {
                "expert": expert,
                "weight_lower": str(weight_lowers[expert]),
                "weight_upper": str(weight_uppers[expert]),
                "weight_derivative": str(derivative_coefficient),
                "active_partition_lower": str(active_lower),
                "active_partition_upper": str(active_upper),
                "inactive_partition_lower": str(inactive_lower),
                "inactive_partition_upper": str(inactive_upper),
                "exponent_lower": str(exponent_lower),
                "exponent_upper": str(exponent_upper),
                "exponential_lower": str(exponential_lower),
                "exponential_upper": str(exponential_upper),
                "positive_value_lower": str(value_lower),
                "positive_value_upper": str(value_upper),
                "stage_response_lower": str(response_lower),
                "stage_response_upper": str(response_upper),
                "gate_derivative_contribution_lower": str(gate_contribution),
                "weighted_response_cost_lower": str(response_cost),
            }
        )
    response_lower, lp_weights = bounded_simplex_minimum(
        weight_lowers, weight_uppers, response_costs
    )
    total_lower = gate_lower + response_lower
    payload = {
        "status": "exact Arb partition-aware output-derivative certificate",
        "rigorous": True,
        "precision_bits": ctx.prec,
        "leaf": args.leaf,
        "coordinate": args.coordinate,
        "fixed": args.fix,
        "dependent_coordinates": dependent,
        "variables": variables,
        "box": serialized_box(box),
        "weight_vertices": weight_vertices,
        "gate_derivative_lower": str(gate_lower),
        "bounded_simplex_response_lower": str(response_lower),
        "total_derivative_lower": str(total_lower),
        "lp_weights": [str(value) for value in lp_weights],
        "experts": expert_records,
        "closed": total_lower > 0,
        "source_sha256": {
            "manifest": sha256(args.manifest),
            "partitions": sha256(args.partitions),
            "gate": sha256(gate_path),
            "dual_bank": sha256(bank_path),
        },
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "closed": payload["closed"],
                "coordinate": args.coordinate,
                "fixed": args.fix,
                "gate_derivative_lower": payload["gate_derivative_lower"],
                "bounded_simplex_response_lower": payload[
                    "bounded_simplex_response_lower"
                ],
                "total_derivative_lower": payload["total_derivative_lower"],
                "output": str(args.output) if args.output is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
