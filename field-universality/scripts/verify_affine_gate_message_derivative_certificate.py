#!/usr/bin/env python3
"""Independently replay an exact sign-aware message-derivative certificate."""

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

from explore_affine_face_interval import FaceBox, tighten_simplex_box
from rationalize_affine_cover import (
    chamber_bounds,
    collect_leaves,
    exact_simplex_vertices,
    rational,
)


class VerificationError(ValueError):
    pass


Polynomial = dict[tuple[int, ...], arb]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Fraction)


def q(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def ball(value) -> arb:
    value = q(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value: fmpq) -> Fraction:
    return Fraction(str(value))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def const(value, dimension: int) -> Polynomial:
    return {(0,) * dimension: value if isinstance(value, arb) else ball(value)}


def plus(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, arb(0)) + coefficient
    return result


def times_scalar(polynomial: Polynomial, value) -> Polynomial:
    value = value if isinstance(value, arb) else ball(value)
    return {exponent: coefficient * value for exponent, coefficient in polynomial.items()}


def product(left: Polynomial, right: Polynomial) -> Polynomial:
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


def differentiate(polynomial: Polynomial, position: int) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        power = exponent[position]
        if power == 0:
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
    result: list[Polynomial | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        polynomial = const(box.lower[coordinate], dimension)
        if coordinate in positions:
            width = box.upper[coordinate] - box.lower[coordinate]
            if width:
                exponent = tuple(
                    1 if index == positions[coordinate] else 0
                    for index in range(dimension)
                )
                polynomial = plus(polynomial, {exponent: ball(width)})
        result[coordinate] = polynomial
    for start in (0, 5, 10):
        chosen = next(value for value in dependent if start <= value < start + 5)
        polynomial = const(1, dimension)
        for coordinate in range(start, start + 5):
            if coordinate != chosen:
                polynomial = plus(polynomial, times_scalar(result[coordinate], -1))
        result[chosen] = polynomial
    return [value for value in result if value is not None]


def cut(raw: dict):
    target = [ball(value) for value in raw["target_potential"]]
    increment = ball(raw["increment_support_potential"])
    state = ball(raw["state_support_potential"])
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


def stage(
    left: list[Polynomial], right: list[Polynomial], data, dimension: int
):
    target, increment, state, kernel = data
    partition: Polynomial = {}
    for difference in range(5):
        for source in range(5):
            partition = plus(
                partition,
                times_scalar(
                    product(left[difference], right[source]),
                    kernel[difference][source],
                ),
            )
    linear: Polynomial = {}
    for index in range(5):
        linear = plus(linear, times_scalar(right[index], target[index]))
    for value in left[1:]:
        linear = plus(linear, times_scalar(value, increment))
    for value in right[1:]:
        linear = plus(linear, times_scalar(value, state))
    return partition, linear


def evaluate_vertex(polynomial: Polynomial, vertex: tuple[int, ...]) -> arb:
    value = arb(0)
    for exponent, coefficient in polynomial.items():
        if all(power == 0 or vertex[index] for index, power in enumerate(exponent)):
            value += coefficient
    return value


def vertex_bounds(polynomial: Polynomial, vertices: list[tuple[int, ...]]):
    if any(any(power > 1 for power in exponent) for exponent in polynomial):
        raise VerificationError("vertex bound requested for non-multiaffine polynomial")
    values = [evaluate_vertex(polynomial, vertex) for vertex in vertices]
    return (
        min(endpoint(value.lower().fmpq()) for value in values),
        max(endpoint(value.upper().fmpq()) for value in values),
    )


def independent_bernstein_minimum(
    polynomial: Polynomial, degrees: list[int]
) -> tuple[Fraction, tuple[int, ...], int]:
    """Transform one axis at a time with explicit slice loops."""

    dimension = len(degrees)
    maximum = [
        max((exponent[position] for exponent in polynomial), default=0)
        for position in range(dimension)
    ]
    if any(found > allowed for found, allowed in zip(maximum, degrees, strict=True)):
        raise VerificationError("saved Bernstein degree is too small")
    shape = tuple(degree + 1 for degree in degrees)
    values = np.empty(shape, dtype=object)
    values.fill(arb(0))
    for exponent, coefficient in polynomial.items():
        values[exponent] = values[exponent] + coefficient
    for axis, degree in enumerate(degrees):
        transformed = np.empty(shape, dtype=object)
        transformed.fill(arb(0))
        other_shape = tuple(shape[index] for index in range(dimension) if index != axis)
        for other_index in np.ndindex(other_shape):
            base = list(other_index)
            for bernstein_index in range(degree + 1):
                total = arb(0)
                for power_degree in range(bernstein_index + 1):
                    full = list(base)
                    full.insert(axis, power_degree)
                    total += values[tuple(full)] * arb(
                        fmpq(
                            math.comb(bernstein_index, power_degree),
                            math.comb(degree, power_degree),
                        )
                    )
                full = list(base)
                full.insert(axis, bernstein_index)
                transformed[tuple(full)] = total
        values = transformed
    minimum = None
    minimum_index = None
    for flat_index, value in enumerate(values.flat):
        lower = endpoint(value.lower().fmpq())
        if minimum is None or lower < minimum:
            minimum = lower
            minimum_index = tuple(int(x) for x in np.unravel_index(flat_index, shape))
    assert minimum is not None and minimum_index is not None
    return minimum, minimum_index, int(values.size)


def slacks(
    points: list[Polynomial], variables: list[int], dependent: list[int], box: FaceBox
) -> dict[str, Polynomial]:
    dimension = len(variables)
    result = {}
    for position, coordinate in enumerate(variables):
        exponent = tuple(1 if index == position else 0 for index in range(dimension))
        variable = {exponent: ball(1)}
        result[f"box-{coordinate}-lower"] = variable
        result[f"box-{coordinate}-upper"] = plus(
            const(1, dimension), times_scalar(variable, -1)
        )
    for coordinate in dependent:
        result[f"coordinate-{coordinate}-lower"] = plus(
            points[coordinate], const(-box.lower[coordinate], dimension)
        )
        result[f"coordinate-{coordinate}-upper"] = plus(
            const(box.upper[coordinate], dimension),
            times_scalar(points[coordinate], -1),
        )
    first = const(0, dimension)
    second = const(0, dimension)
    for coordinate in range(1, 5):
        first = plus(first, times_scalar(points[coordinate], -1))
    for coordinate in range(6, 10):
        first = plus(first, times_scalar(points[coordinate], 2))
        second = plus(second, times_scalar(points[coordinate], -1))
    for coordinate in range(11, 15):
        second = plus(second, times_scalar(points[coordinate], 2))
    result["first-support"] = first
    result["second-support"] = second
    for coordinate in range(2, 5):
        result[f"message-order-1-ge-{coordinate}"] = plus(
            points[1], times_scalar(points[coordinate], -1)
        )
    return result


def verify_facet(
    polynomial: Polynomial,
    saved: dict,
    named_slacks: dict[str, Polynomial],
) -> Fraction:
    adjusted = dict(polynomial)
    for raw in saved["multipliers"]:
        name = str(raw["slack"])
        value = Fraction(raw["value"])
        if value < 0 or name not in named_slacks:
            raise VerificationError("invalid saved facet multiplier")
        adjusted = plus(adjusted, times_scalar(named_slacks[name], -value))
    replay, index, count = independent_bernstein_minimum(
        adjusted, [int(value) for value in saved["degrees"]]
    )
    stored = Fraction(saved["minimum_adjusted_bernstein_lower"])
    if replay < stored:
        raise VerificationError(f"replay Bernstein lower {replay} is below {stored}")
    if count != int(saved["coefficients"]):
        raise VerificationError("Bernstein coefficient count changed")
    return stored


def weight_ranges(record: dict, box: FaceBox):
    blocks = [
        exact_simplex_vertices(list(box.lower), list(box.upper), start)
        for start in (0, 5, 10)
    ]
    points = [
        (*triple[0], *triple[1], *triple[2])
        for triple in itertools.product(*blocks)
    ]
    denominator = int(record["coefficient_denominator"])
    features = [int(value) for value in record["feature_coordinates"]]
    lowers = []
    uppers = []
    for row in record["coefficient_numerators"]:
        values = [
            (
                int(row[0])
                + sum(
                    (
                        int(coefficient) * point[feature]
                        for coefficient, feature in zip(
                            row[1:], features, strict=True
                        )
                    ),
                    Fraction(0),
                )
            )
            / denominator
            for point in points
        ]
        lowers.append(min(values))
        uppers.append(max(values))
    return lowers, uppers, len(points)


def apply_box_restrictions(
    lower: list[Fraction],
    upper: list[Fraction],
    fixed_specs: list[str],
    half_specs: list[str],
    dependent: list[int],
    derivative_coordinate: int,
):
    fixed = set()
    for specification in fixed_specs:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in dependent or coordinate == derivative_coordinate:
            raise VerificationError("invalid fixed coordinate")
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            raise VerificationError("invalid fixed side")
    for specification in half_specs:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in dependent or coordinate in fixed:
            raise VerificationError("invalid half coordinate")
        midpoint = (lower[coordinate] + upper[coordinate]) / 2
        if side == "lower":
            upper[coordinate] = midpoint
        elif side == "upper":
            lower[coordinate] = midpoint
        else:
            raise VerificationError("invalid half side")
    return fixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("proposal", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision-bits", type=int, default=320)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision_bits < 256:
        parser.error("replay precision must be at least 256 bits")
    ctx.prec = args.precision_bits
    manifest = load(args.manifest)
    partitions = load(args.partitions)
    proposal = load(args.proposal)
    certificate = load(args.certificate)
    if not certificate.get("rigorous") or not certificate.get("closed"):
        raise VerificationError("certificate is not marked rigorous and closed")
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaf_name = str(certificate["leaf"])
    coordinate = int(certificate["coordinate"])
    dependent = [int(value) for value in certificate["dependent_coordinates"]]
    dependent_message_coordinate = next(
        value for value in dependent if 0 <= value < 5
    )
    if (
        proposal["leaf"] != leaf_name
        or int(proposal["coordinate"]) != coordinate
        or int(certificate["dependent_message_coordinate"])
        != dependent_message_coordinate
        or proposal.get("gate") != certificate.get("gate")
        or proposal["fixed"] != certificate["fixed"]
        or proposal["half_restrictions"] != certificate["half_restrictions"]
    ):
        raise VerificationError("proposal/certificate topology mismatch")
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[leaf_name]
    record = next(value for value in partitions["leaves"] if value["leaf"] == leaf_name)
    gate_reference = str(
        certificate.get("gate", record.get("gate", leaf["gate"]))
    )
    gate_path = root / gate_reference.replace("\\", "/")
    checkpoint_path = root / manifest["global_checkpoint"]
    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    expected_hashes = {
        "manifest": digest(args.manifest),
        "partitions": digest(args.partitions),
        "proposal": digest(args.proposal),
        "gate": digest(gate_path),
        "checkpoint": digest(checkpoint_path),
        "dual_bank": digest(bank_path),
    }
    if certificate["source_sha256"] != expected_hashes:
        raise VerificationError("source hash mismatch")
    gate = load(gate_path)
    checkpoint = load(checkpoint_path)
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    fixed = apply_box_restrictions(
        lower,
        upper,
        list(certificate["fixed"]),
        list(certificate["half_restrictions"]),
        dependent,
        coordinate,
    )
    variables = [
        value for value in range(15) if value not in fixed and value not in dependent
    ]
    if variables != [int(value) for value in certificate["variables"]]:
        raise VerificationError("free-coordinate list mismatch")
    box = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if box is None:
        raise VerificationError("replayed box is empty")
    dimension = len(variables)
    position = variables.index(coordinate)
    vertices = list(itertools.product((0, 1), repeat=dimension))
    points = point_polynomials(box, variables, dependent)
    named_slacks = slacks(points, variables, dependent, box)

    q0_lower = box.lower[dependent_message_coordinate]
    q0_upper = box.upper[dependent_message_coordinate]
    qj_lower, qj_upper = box.lower[coordinate], box.upper[coordinate]
    entropy_kind = certificate["entropy_envelope"]["kind"]
    if entropy_kind == "constant log(q0_lower/qj_upper)":
        entropy_slope = const(
            (ball(q0_lower).log() - ball(qj_upper).log())
            * ball((box.upper[coordinate] - box.lower[coordinate]) / 3),
            dimension,
        )
    else:
        if entropy_kind != "secant(log q0) plus tangent(-log qj)":
            raise VerificationError("unknown entropy-envelope type")
        if q0_lower == q0_upper:
            log_q0 = const(ball(q0_lower).log(), dimension)
        else:
            slope = (ball(q0_upper).log() - ball(q0_lower).log()) / ball(
                q0_upper - q0_lower
            )
            intercept = ball(q0_lower).log() - slope * ball(q0_lower)
            log_q0 = plus(
                const(intercept, dimension),
                times_scalar(points[dependent_message_coordinate], slope),
            )
        tangent = (qj_lower + qj_upper) / 2
        negative_log_qj = plus(
            const(ball(1) - ball(tangent).log(), dimension),
            times_scalar(points[coordinate], -Fraction(1, 1) / tangent),
        )
        entropy_slope = times_scalar(
            plus(log_q0, negative_log_qj),
            (box.upper[coordinate] - box.lower[coordinate]) / 3,
        )

    bank = load(bank_path)
    first_cuts = [cut(value) for value in bank["first_cuts"]]
    second_cuts = [cut(value) for value in bank["second_cuts"]]
    first_cache = {}
    second_cache = {}
    responses = {}
    expert_data = []
    denominator = int(record["coefficient_denominator"])
    features = [int(value) for value in record["feature_coordinates"]]
    weight_lowers, weight_uppers, weight_vertex_count = weight_ranges(record, box)
    if weight_vertex_count != int(certificate["weight_vertices"]):
        raise VerificationError("weight vertex count mismatch")
    gate_lower = Fraction(0)
    for expert, (row, pair) in enumerate(
        zip(record["coefficient_numerators"], record["expert_pairs"], strict=True)
    ):
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        first_cache.setdefault(
            first_index,
            stage(points[0:5], points[5:10], first_cuts[first_index], dimension),
        )
        second_cache.setdefault(
            second_index,
            stage(points[5:10], points[10:15], second_cuts[second_index], dimension),
        )
        first_partition, first_linear = first_cache[first_index]
        second_partition, second_linear = second_cache[second_index]
        if first_index not in responses:
            response = plus(
                differentiate(first_partition, position),
                times_scalar(
                    first_partition,
                    -differentiate(first_linear, position).get(
                        (0,) * dimension, arb(0)
                    ),
                ),
            )
            responses[first_index] = plus(
                response, product(first_partition, entropy_slope)
            )
        weight = const(Fraction(int(row[0]), denominator), dimension)
        for numerator, feature in zip(row[1:], features, strict=True):
            weight = plus(
                weight,
                times_scalar(points[feature], Fraction(int(numerator), denominator)),
            )
        inactive_lower, inactive_upper = vertex_bounds(second_partition, vertices)
        active_lower, active_upper = vertex_bounds(first_partition, vertices)
        exponent = times_scalar(plus(first_linear, second_linear), -1)
        exponent_lower, exponent_upper = vertex_bounds(exponent, vertices)
        exp_lower = endpoint(ball(exponent_lower).exp().lower().fmpq())
        exp_upper = endpoint(ball(exponent_upper).exp().upper().fmpq())
        prefactor_lower = inactive_lower * exp_lower
        prefactor_upper = inactive_upper * exp_upper
        coefficients = {
            feature: Fraction(int(value), denominator)
            for feature, value in zip(features, row[1:], strict=True)
        }
        block_start = (coordinate // 5) * 5
        block_dependent = next(
            value for value in dependent if block_start <= value < block_start + 5
        )
        derivative_coefficient = (
            box.upper[coordinate] - box.lower[coordinate]
        ) * (
            coefficients.get(coordinate, Fraction(0))
            - coefficients.get(block_dependent, Fraction(0))
        )
        contribution = derivative_coefficient * (
            prefactor_lower * active_lower
            if derivative_coefficient >= 0
            else prefactor_upper * active_upper
        )
        gate_lower += contribution
        expert_data.append(
            (
                first_index,
                weight,
                weight_uppers[expert],
                prefactor_lower,
                prefactor_upper,
            )
        )

    response_lowers = {}
    for first_index, response in sorted(responses.items()):
        saved = certificate["response_certificates"][str(first_index)]
        response_lowers[first_index] = verify_facet(
            response, saved, named_slacks
        )
    aggregate = const(0, dimension)
    correction = Fraction(0)
    for first_index, weight, weight_upper, prefactor_lower, prefactor_upper in expert_data:
        aggregate = plus(
            aggregate,
            times_scalar(product(weight, responses[first_index]), prefactor_lower),
        )
        response_lower = response_lowers[first_index]
        if response_lower < 0:
            correction += (
                (prefactor_upper - prefactor_lower)
                * weight_upper
                * response_lower
            )
    aggregate_lower = verify_facet(
        aggregate, certificate["aggregate_certificate"], named_slacks
    )
    stored_aggregate = Fraction(certificate["aggregate_response_lower"])
    stored_correction = Fraction(certificate["negative_response_correction_lower"])
    stored_gate = Fraction(certificate["gate_derivative_lower"])
    stored_total = Fraction(certificate["total_derivative_lower"])
    if aggregate_lower != stored_aggregate:
        raise VerificationError("aggregate lower does not match its facet record")
    if correction < stored_correction or gate_lower < stored_gate:
        raise VerificationError(
            "higher-precision correction/gate replay is below the stored bound"
        )
    if stored_total != stored_aggregate + stored_correction + stored_gate:
        raise VerificationError("stored total is algebraically inconsistent")
    if stored_total <= 0:
        raise VerificationError("replayed total derivative is not positive")
    replay = {
        "status": "independent exact message-derivative replay",
        "verified": True,
        "precision_bits": ctx.prec,
        "certificate": str(args.certificate),
        "certificate_sha256": digest(args.certificate),
        "total_derivative_lower": str(stored_total),
        "total_derivative_lower_approx": float(stored_total),
        "response_certificates": len(response_lowers),
        "experts": len(expert_data),
        "source_sha256": expected_hashes,
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, replay)
    print(json.dumps(replay, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
