#!/usr/bin/env python3
"""Generate an exact sign-aware message-derivative certificate from a proposal."""

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

from explore_affine_face_interval import FaceBox, serialized_box, tighten_simplex_box
from generate_affine_gate_partition_derivative_certificate import (
    add,
    as_ball,
    derivative,
    endpoint,
    exact_weight_bounds,
    load_cut,
    load_exact,
    multiply,
    point_polynomials,
    q,
    scale,
    sha256,
    stage,
    vertex_bounds,
    weight_derivative,
)
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


Polynomial = dict[tuple[int, ...], arb]


def constant(value, dimension: int) -> Polynomial:
    return {(0,) * dimension: value if isinstance(value, arb) else as_ball(value)}


def exact_half_box(
    lower: list[Fraction],
    upper: list[Fraction],
    specifications: list[str],
    fixed: set[int],
    dependent: list[int],
) -> None:
    for specification in specifications:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in fixed or coordinate in dependent:
            raise ValueError("half restriction conflicts with fixed/dependent coordinate")
        midpoint = (lower[coordinate] + upper[coordinate]) / 2
        if side == "lower":
            upper[coordinate] = midpoint
        elif side == "upper":
            lower[coordinate] = midpoint
        else:
            raise ValueError("half side must be lower or upper")


def anisotropic_bernstein(
    polynomial: Polynomial, dimension: int, minimum_degrees: list[int] | None = None
):
    degrees = [
        max((exponent[position] for exponent in polynomial), default=0)
        for position in range(dimension)
    ]
    if minimum_degrees is not None:
        degrees = [
            max(degree, minimum)
            for degree, minimum in zip(degrees, minimum_degrees, strict=True)
        ]
    shape = tuple(degree + 1 for degree in degrees)
    power = np.empty(shape, dtype=object)
    power.fill(arb(0))
    for exponent, coefficient in polynomial.items():
        power[exponent] = power[exponent] + coefficient
    bernstein = power
    for axis, degree in enumerate(degrees):
        transform = np.empty((degree + 1, degree + 1), dtype=object)
        transform.fill(arb(0))
        for index in range(degree + 1):
            for power_degree in range(index + 1):
                transform[index, power_degree] = arb(
                    fmpq(
                        math.comb(index, power_degree),
                        math.comb(degree, power_degree),
                    )
                )
        bernstein = np.tensordot(transform, bernstein, axes=(1, axis))
        bernstein = np.moveaxis(bernstein, 0, axis)
    return bernstein, degrees


def minimum_bernstein_lower(
    polynomial: Polynomial, dimension: int, minimum_degrees: list[int] | None = None
):
    bernstein, degrees = anisotropic_bernstein(
        polynomial, dimension, minimum_degrees
    )
    minimum = None
    minimum_index = None
    for flat_index, value in enumerate(bernstein.flat):
        lower = endpoint(value.lower().fmpq())
        if minimum is None or lower < minimum:
            minimum = lower
            minimum_index = tuple(int(x) for x in np.unravel_index(flat_index, bernstein.shape))
    assert minimum is not None and minimum_index is not None
    return minimum, minimum_index, degrees, int(bernstein.size)


def named_slacks(
    points: list[Polynomial], variables: list[int], dependent: list[int], box: FaceBox
) -> dict[str, Polynomial]:
    dimension = len(variables)
    result: dict[str, Polynomial] = {}
    for position, coordinate in enumerate(variables):
        exponent = tuple(1 if index == position else 0 for index in range(dimension))
        variable = {exponent: as_ball(1)}
        result[f"box-{coordinate}-lower"] = variable
        result[f"box-{coordinate}-upper"] = add(
            constant(1, dimension), scale(variable, -1)
        )
    for coordinate in dependent:
        result[f"coordinate-{coordinate}-lower"] = add(
            points[coordinate], constant(-box.lower[coordinate], dimension)
        )
        result[f"coordinate-{coordinate}-upper"] = add(
            constant(box.upper[coordinate], dimension), scale(points[coordinate], -1)
        )
    first_support = constant(0, dimension)
    second_support = constant(0, dimension)
    for coordinate in range(1, 5):
        first_support = add(first_support, scale(points[coordinate], -1))
    for coordinate in range(6, 10):
        first_support = add(first_support, scale(points[coordinate], 2))
        second_support = add(second_support, scale(points[coordinate], -1))
    for coordinate in range(11, 15):
        second_support = add(second_support, scale(points[coordinate], 2))
    result["first-support"] = first_support
    result["second-support"] = second_support
    for coordinate in range(2, 5):
        result[f"message-order-1-ge-{coordinate}"] = add(
            points[1], scale(points[coordinate], -1)
        )
    return result


def proposal_multipliers(raw: dict) -> list[tuple[str, Fraction]]:
    result = []
    for value in raw.get("multipliers", []):
        multiplier = q(value["value"])
        if multiplier < 0:
            raise ValueError("facet multiplier is negative")
        result.append((str(value["slack"]), multiplier))
    return result


def facet_certificate(
    polynomial: Polynomial,
    multiplier_data: list[tuple[str, Fraction]],
    slacks: dict[str, Polynomial],
    dimension: int,
    minimum_degrees: list[int] | None = None,
) -> dict:
    adjusted = dict(polynomial)
    for name, multiplier in multiplier_data:
        if name not in slacks:
            raise ValueError(f"unknown facet slack {name}")
        adjusted = add(adjusted, scale(slacks[name], -multiplier))
    lower, index, degrees, coefficients = minimum_bernstein_lower(
        adjusted, dimension, minimum_degrees
    )
    return {
        "multipliers": [
            {"slack": name, "value": str(value)} for name, value in multiplier_data
        ],
        "minimum_adjusted_bernstein_lower": str(lower),
        "minimum_index": list(index),
        "degrees": degrees,
        "coefficients": coefficients,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("proposal", type=Path)
    parser.add_argument("--precision-bits", type=int, default=256)
    parser.add_argument(
        "--elevate-coordinate",
        action="append",
        type=int,
        default=[],
        help="elevate this free coordinate's Bernstein degree to at least three",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision_bits < 192:
        parser.error("precision must be at least 192 bits")
    ctx.prec = args.precision_bits

    manifest = load_exact(args.manifest)
    partitions = load_exact(args.partitions)
    proposal = load_exact(args.proposal)
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaf_name = str(proposal["leaf"])
    coordinate = int(proposal["coordinate"])
    if coordinate not in range(0, 5):
        raise ValueError("proposal is not a message-coordinate derivative")
    dependent = [int(value) for value in proposal["dependent_coordinates"]]
    if coordinate in dependent or len(dependent) != 3:
        raise ValueError("invalid dependent-coordinate choice")
    dependent_message_coordinate = next(
        value for value in dependent if 0 <= value < 5
    )
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[leaf_name]
    record = next(value for value in partitions["leaves"] if value["leaf"] == leaf_name)
    if not record.get("partition_identity_exact") or not record.get(
        "nonnegative_on_product_box_exact"
    ):
        raise ValueError("affine partition lacks exact source certification")
    gate_reference = str(proposal.get("gate", record.get("gate", leaf["gate"])))
    gate_path = root / gate_reference.replace("\\", "/")
    gate = load_exact(gate_path)
    checkpoint_path = root / manifest["global_checkpoint"]
    checkpoint = load_exact(checkpoint_path)
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    fixed: set[int] = set()
    for specification in proposal["fixed"]:
        coordinate_text, side = specification.split(":")
        fixed_coordinate = int(coordinate_text)
        if fixed_coordinate in dependent or fixed_coordinate == coordinate:
            raise ValueError("fixed coordinate conflicts with derivative")
        fixed.add(fixed_coordinate)
        if side == "lower":
            upper[fixed_coordinate] = lower[fixed_coordinate]
        elif side == "upper":
            lower[fixed_coordinate] = upper[fixed_coordinate]
        else:
            raise ValueError("invalid fixed side")
    exact_half_box(
        lower,
        upper,
        list(proposal["half_restrictions"]),
        fixed,
        dependent,
    )
    variables = [
        value for value in range(15) if value not in fixed and value not in dependent
    ]
    box = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if box is None:
        raise ValueError("proposal box is empty")
    dimension = len(variables)
    position = variables.index(coordinate)
    unknown_elevations = set(args.elevate_coordinate) - set(variables)
    if unknown_elevations:
        raise ValueError(
            f"cannot elevate non-free coordinates: {sorted(unknown_elevations)}"
        )
    minimum_degrees = [
        3 if value in set(args.elevate_coordinate) else 0 for value in variables
    ]
    vertices = list(itertools.product((0, 1), repeat=dimension))
    points = point_polynomials(box, variables, dependent)
    slacks = named_slacks(points, variables, dependent, box)

    q0_lower = box.lower[dependent_message_coordinate]
    q0_upper = box.upper[dependent_message_coordinate]
    qj_lower = box.lower[coordinate]
    qj_upper = box.upper[coordinate]
    if q0_lower <= 0 or qj_upper <= 0:
        raise ValueError("entropy envelope requires positive q0 and positive qj upper")
    if proposal.get("entropy_affine_lower_proposal") is None:
        entropy_slope_lower = constant(
            (as_ball(q0_lower).log() - as_ball(qj_upper).log())
            * as_ball((box.upper[coordinate] - box.lower[coordinate]) / 3),
            dimension,
        )
        tangent = None
        entropy_kind = "constant log(q0_lower/qj_upper)"
    else:
        if qj_lower <= 0:
            raise ValueError("affine entropy envelope requires positive qj lower")
        if q0_lower == q0_upper:
            log_q0_lower = constant(as_ball(q0_lower).log(), dimension)
        else:
            slope = (as_ball(q0_upper).log() - as_ball(q0_lower).log()) / as_ball(
                q0_upper - q0_lower
            )
            intercept = as_ball(q0_lower).log() - slope * as_ball(q0_lower)
            log_q0_lower = add(
                constant(intercept, dimension),
                scale(points[dependent_message_coordinate], slope),
            )
        tangent = (qj_lower + qj_upper) / 2
        negative_log_qj_lower = add(
            constant(as_ball(1) - as_ball(tangent).log(), dimension),
            scale(points[coordinate], -Fraction(1, 1) / tangent),
        )
        entropy_slope_lower = scale(
            add(log_q0_lower, negative_log_qj_lower),
            (box.upper[coordinate] - box.lower[coordinate]) / 3,
        )
        entropy_kind = "secant(log q0) plus tangent(-log qj)"

    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    bank = load_exact(bank_path)
    first_cuts = [load_cut(value) for value in bank["first_cuts"]]
    second_cuts = [load_cut(value) for value in bank["second_cuts"]]
    first_cache = {}
    second_cache = {}
    response_polynomials = {}
    expert_data = []
    denominator = int(record["coefficient_denominator"])
    features = [int(value) for value in record["feature_coordinates"]]
    weight_lowers, weight_uppers, weight_vertices = exact_weight_bounds(record, box)
    gate_derivative_lower = Fraction(0)
    for expert, (row, pair) in enumerate(
        zip(record["coefficient_numerators"], record["expert_pairs"], strict=True)
    ):
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_cache:
            first_cache[first_index] = stage(
                points[0:5], points[5:10], first_cuts[first_index], dimension
            )
        if second_index not in second_cache:
            second_cache[second_index] = stage(
                points[5:10], points[10:15], second_cuts[second_index], dimension
            )
        first_partition, first_linear = first_cache[first_index]
        second_partition, second_linear = second_cache[second_index]
        if first_index not in response_polynomials:
            response = add(
                derivative(first_partition, position),
                scale(
                    first_partition,
                    -derivative(first_linear, position).get(
                        (0,) * dimension, arb(0)
                    ),
                ),
            )
            response_polynomials[first_index] = add(
                response, multiply(first_partition, entropy_slope_lower)
            )
        weight = constant(Fraction(int(row[0]), denominator), dimension)
        for numerator, feature in zip(row[1:], features, strict=True):
            weight = add(
                weight,
                scale(points[feature], Fraction(int(numerator), denominator)),
            )
        inactive_lower, inactive_upper = vertex_bounds(
            second_partition, vertices, multiaffine=True
        )
        active_lower, active_upper = vertex_bounds(
            first_partition, vertices, multiaffine=True
        )
        exponent = scale(add(first_linear, second_linear), -1)
        exponent_lower, exponent_upper = vertex_bounds(
            exponent, vertices, multiaffine=True
        )
        exponential_lower = endpoint(as_ball(exponent_lower).exp().lower().fmpq())
        exponential_upper = endpoint(as_ball(exponent_upper).exp().upper().fmpq())
        prefactor_lower = inactive_lower * exponential_lower
        prefactor_upper = inactive_upper * exponential_upper
        derivative_coefficient = weight_derivative(
            row, denominator, features, coordinate, dependent, box
        )
        gate_contribution = derivative_coefficient * (
            prefactor_lower * active_lower
            if derivative_coefficient >= 0
            else prefactor_upper * active_upper
        )
        gate_derivative_lower += gate_contribution
        expert_data.append(
            {
                "expert": expert,
                "first_cut": first_index,
                "weight": weight,
                "weight_lower": weight_lowers[expert],
                "weight_upper": weight_uppers[expert],
                "prefactor_lower": prefactor_lower,
                "prefactor_upper": prefactor_upper,
                "weight_derivative": derivative_coefficient,
                "gate_contribution": gate_contribution,
            }
        )

    response_proposals = proposal["first_cut_response_facet_lp_proposals"]
    response_certificates = {}
    response_lowers = {}
    for first_cut, polynomial in sorted(response_polynomials.items()):
        raw = response_proposals[str(first_cut)]
        certificate = facet_certificate(
            polynomial,
            proposal_multipliers(raw),
            slacks,
            dimension,
            minimum_degrees,
        )
        lower_bound = Fraction(certificate["minimum_adjusted_bernstein_lower"])
        response_certificates[str(first_cut)] = certificate
        response_lowers[first_cut] = lower_bound

    aggregate = constant(0, dimension)
    negative_correction = Fraction(0)
    serialized_experts = []
    for value in expert_data:
        first_cut = value["first_cut"]
        response = response_polynomials[first_cut]
        aggregate = add(
            aggregate,
            scale(
                multiply(value["weight"], response), value["prefactor_lower"]
            ),
        )
        response_lower = response_lowers[first_cut]
        correction = Fraction(0)
        if response_lower < 0:
            correction = (
                (value["prefactor_upper"] - value["prefactor_lower"])
                * value["weight_upper"]
                * response_lower
            )
            negative_correction += correction
        serialized_experts.append(
            {
                "expert": value["expert"],
                "first_cut": first_cut,
                "weight_lower": str(value["weight_lower"]),
                "weight_upper": str(value["weight_upper"]),
                "prefactor_lower": str(value["prefactor_lower"]),
                "prefactor_upper": str(value["prefactor_upper"]),
                "response_lower": str(response_lower),
                "negative_response_correction": str(correction),
                "weight_derivative": str(value["weight_derivative"]),
                "gate_derivative_contribution": str(value["gate_contribution"]),
            }
        )
    aggregate_proposal = proposal["sign_aware_prefactor_facet_lp_proposal"]
    aggregate_certificate = facet_certificate(
        aggregate,
        proposal_multipliers(aggregate_proposal),
        slacks,
        dimension,
        minimum_degrees,
    )
    aggregate_lower = Fraction(
        aggregate_certificate["minimum_adjusted_bernstein_lower"]
    )
    total_lower = aggregate_lower + negative_correction + gate_derivative_lower
    payload = {
        "status": "exact Arb sign-aware message-derivative certificate",
        "rigorous": True,
        "precision_bits": ctx.prec,
        "leaf": leaf_name,
        "gate": gate_reference,
        "coordinate": coordinate,
        "dependent_message_coordinate": dependent_message_coordinate,
        "fixed": proposal["fixed"],
        "half_restrictions": proposal["half_restrictions"],
        "dependent_coordinates": dependent,
        "variables": variables,
        "elevated_coordinates": args.elevate_coordinate,
        "box": serialized_box(box),
        "entropy_envelope": {
            "q0_interval": [str(q0_lower), str(q0_upper)],
            "qj_interval": [str(qj_lower), str(qj_upper)],
            "tangent_point": str(tangent) if tangent is not None else None,
            "kind": entropy_kind,
        },
        "weight_vertices": weight_vertices,
        "response_certificates": response_certificates,
        "aggregate_certificate": aggregate_certificate,
        "aggregate_response_lower": str(aggregate_lower),
        "negative_response_correction_lower": str(negative_correction),
        "gate_derivative_lower": str(gate_derivative_lower),
        "total_derivative_lower": str(total_lower),
        "experts": serialized_experts,
        "closed": total_lower > 0,
        "source_sha256": {
            "manifest": sha256(args.manifest),
            "partitions": sha256(args.partitions),
            "proposal": sha256(args.proposal),
            "gate": sha256(gate_path),
            "checkpoint": sha256(checkpoint_path),
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
                "aggregate_response_lower": payload["aggregate_response_lower"],
                "negative_response_correction_lower": payload[
                    "negative_response_correction_lower"
                ],
                "gate_derivative_lower": payload["gate_derivative_lower"],
                "total_derivative_lower": payload["total_derivative_lower"],
                "aggregate_response_lower_approx": float(aggregate_lower),
                "negative_response_correction_lower_approx": float(
                    negative_correction
                ),
                "gate_derivative_lower_approx": float(gate_derivative_lower),
                "total_derivative_lower_approx": float(total_lower),
                "output": str(args.output) if args.output is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["closed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
