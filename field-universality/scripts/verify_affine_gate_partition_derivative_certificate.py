#!/usr/bin/env python3
"""Independently replay an exact partition-aware derivative certificate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

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


def load_exact(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Fraction)


def q(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def ball(value) -> arb:
    value = q(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value: fmpq) -> Fraction:
    return Fraction(str(value))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cut(raw: dict):
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


def stage_value(left: list[Fraction], right: list[Fraction], cut) -> tuple[arb, arb]:
    target, increment, state, kernel = cut
    partition = arb(0)
    for difference in range(5):
        for source in range(5):
            partition += ball(left[difference]) * kernel[difference][source] * ball(
                right[source]
            )
    linear = sum((target[index] * ball(right[index]) for index in range(5)), arb(0))
    linear += increment * ball(sum(left[1:], Fraction(0)))
    linear += state * ball(sum(right[1:], Fraction(0)))
    return partition, linear


def output_response(
    left: list[Fraction],
    right: list[Fraction],
    cut,
    coordinate: int,
    dependent_coordinate: int,
    width: Fraction,
    partition: arb,
) -> arb:
    target, _, state, kernel = cut
    coordinate_local = coordinate - 10
    dependent_local = dependent_coordinate - 10
    partition_derivative = ball(width) * sum(
        (
            ball(left[difference])
            * (
                kernel[difference][coordinate_local]
                - kernel[difference][dependent_local]
            )
            for difference in range(5)
        ),
        arb(0),
    )
    linear_derivative = ball(width) * (
        target[coordinate_local]
        - target[dependent_local]
        + state
        * (
            int(coordinate_local != 0)
            - int(dependent_local != 0)
        )
    )
    return partition_derivative - partition * linear_derivative


def composition_vertex(
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
        point[coordinate] = (
            box.upper[coordinate]
            if coordinate in positions and vertex[positions[coordinate]]
            else box.lower[coordinate]
        )
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        point[coordinate] = 1 - sum(
            (point[other] for other in range(start, start + 5) if other != coordinate),
            Fraction(0),
        )
    result = [value for value in point if value is not None]
    if min(result) < 0:
        raise VerificationError("Boolean extension contains a negative probability")
    return result


def exact_weight_ranges(record: dict, box: FaceBox):
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
    return lowers, uppers, len(points)


def exact_weight_derivative(
    row: list[int],
    denominator: int,
    coordinates: list[int],
    coordinate: int,
    dependent: int,
    width: Fraction,
) -> Fraction:
    coefficients = {
        feature: Fraction(value, denominator)
        for value, feature in zip(row[1:], coordinates, strict=True)
    }
    return width * (
        coefficients.get(coordinate, Fraction(0))
        - coefficients.get(dependent, Fraction(0))
    )


def simplex_minimum(
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
        raise VerificationError("weight ranges do not meet the simplex")
    return sum((w * c for w, c in zip(weights, costs, strict=True)), Fraction(0)), weights


def parse_box(raw: dict) -> FaceBox:
    return FaceBox(
        tuple(Fraction(value) for value in raw["lower"]),
        tuple(Fraction(value) for value in raw["upper"]),
    )


def verify_saved_algebra(certificate: dict) -> None:
    experts = certificate["experts"]
    gate = sum(
        (Fraction(value["gate_derivative_contribution_lower"]) for value in experts),
        Fraction(0),
    )
    if gate != Fraction(certificate["gate_derivative_lower"]):
        raise VerificationError("saved gate derivative sum is inconsistent")
    for value in experts:
        derivative = Fraction(value["weight_derivative"])
        expected_gate = derivative * Fraction(
            value[
                "positive_value_lower" if derivative >= 0 else "positive_value_upper"
            ]
        )
        if expected_gate != Fraction(value["gate_derivative_contribution_lower"]):
            raise VerificationError("saved expert gate contribution is inconsistent")
        response = Fraction(value["stage_response_lower"])
        scale_endpoint = (
            Fraction(value["inactive_partition_lower"])
            * Fraction(value["exponential_lower"])
            if response >= 0
            else Fraction(value["inactive_partition_upper"])
            * Fraction(value["exponential_upper"])
        )
        if response * scale_endpoint != Fraction(value["weighted_response_cost_lower"]):
            raise VerificationError("saved response cost is inconsistent")
    weights = [Fraction(value) for value in certificate["lp_weights"]]
    if sum(weights, Fraction(0)) != 1:
        raise VerificationError("saved LP weights do not sum to one")
    costs = [Fraction(value["weighted_response_cost_lower"]) for value in experts]
    for weight, value in zip(weights, experts, strict=True):
        if not Fraction(value["weight_lower"]) <= weight <= Fraction(value["weight_upper"]):
            raise VerificationError("saved LP weight violates its interval")
    objective = sum((w * c for w, c in zip(weights, costs, strict=True)), Fraction(0))
    if objective != Fraction(certificate["bounded_simplex_response_lower"]):
        raise VerificationError("saved LP objective is inconsistent")
    total = gate + objective
    if total != Fraction(certificate["total_derivative_lower"]) or total <= 0:
        raise VerificationError("saved total derivative is not strictly positive")


def verify(
    manifest_path: Path,
    partitions_path: Path,
    certificate_path: Path,
    precision_bits: int,
) -> dict:
    ctx.prec = precision_bits
    manifest_path = manifest_path.resolve()
    partitions_path = partitions_path.resolve()
    certificate_path = certificate_path.resolve()
    manifest = load_exact(manifest_path)
    partitions = load_exact(partitions_path)
    certificate = load_exact(certificate_path)
    if certificate.get("rigorous") is not True or certificate.get("closed") is not True:
        raise VerificationError("certificate is not marked rigorous and closed")
    root = (manifest_path.parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[certificate["leaf"]]
    record = next(
        value for value in partitions["leaves"] if value["leaf"] == certificate["leaf"]
    )
    gate_path = root / str(record.get("gate", leaf["gate"])).replace("\\", "/")
    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    expected_hashes = {
        "manifest": file_hash(manifest_path),
        "partitions": file_hash(partitions_path),
        "gate": file_hash(gate_path),
        "dual_bank": file_hash(bank_path),
    }
    if certificate.get("source_sha256") != expected_hashes:
        raise VerificationError("source hashes do not match")
    verify_saved_algebra(certificate)
    gate = load_exact(gate_path)
    checkpoint = load_exact(root / manifest["global_checkpoint"])
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    dependent = [int(value) for value in certificate["dependent_coordinates"]]
    fixed: set[int] = set()
    for specification in certificate["fixed"]:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            raise VerificationError("invalid fixed side")
    variables = [
        coordinate
        for coordinate in range(15)
        if coordinate not in fixed and coordinate not in dependent
    ]
    box = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if box is None or box != parse_box(certificate["box"]):
        raise VerificationError("certificate box does not reconstruct")
    if variables != [int(value) for value in certificate["variables"]]:
        raise VerificationError("certificate variable list does not reconstruct")
    coordinate = int(certificate["coordinate"])
    width = box.upper[coordinate] - box.lower[coordinate]
    output_dependent = next(value for value in dependent if 10 <= value < 15)
    vertices = list(itertools.product((0, 1), repeat=len(variables)))
    points = [composition_vertex(box, variables, dependent, vertex) for vertex in vertices]
    bank = load_exact(bank_path)
    first_cuts = [load_cut(value) for value in bank["first_cuts"]]
    second_cuts = [load_cut(value) for value in bank["second_cuts"]]
    weight_lowers, weight_uppers, weight_vertex_count = exact_weight_ranges(record, box)
    denominator = int(record["coefficient_denominator"])
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]
    gate_lower = Fraction(0)
    response_costs = []
    for expert, (row, pair) in enumerate(
        zip(record["coefficient_numerators"], record["expert_pairs"], strict=True)
    ):
        first_cut = first_cuts[int(pair["first_cut"])]
        second_cut = second_cuts[int(pair["second_cut"])]
        active_lowers = []
        active_uppers = []
        inactive_lowers = []
        inactive_uppers = []
        response_lowers = []
        response_uppers = []
        exponent_lowers = []
        exponent_uppers = []
        for point in points:
            first_partition, first_linear = stage_value(
                point[0:5], point[5:10], first_cut
            )
            second_partition, second_linear = stage_value(
                point[5:10], point[10:15], second_cut
            )
            response = output_response(
                point[5:10],
                point[10:15],
                second_cut,
                coordinate,
                output_dependent,
                width,
                second_partition,
            )
            exponent = -(first_linear + second_linear)
            active_lowers.append(endpoint(second_partition.lower().fmpq()))
            active_uppers.append(endpoint(second_partition.upper().fmpq()))
            inactive_lowers.append(endpoint(first_partition.lower().fmpq()))
            inactive_uppers.append(endpoint(first_partition.upper().fmpq()))
            response_lowers.append(endpoint(response.lower().fmpq()))
            response_uppers.append(endpoint(response.upper().fmpq()))
            exponent_lowers.append(endpoint(exponent.lower().fmpq()))
            exponent_uppers.append(endpoint(exponent.upper().fmpq()))
        active_lower = min(active_lowers)
        active_upper = max(active_uppers)
        inactive_lower = min(inactive_lowers)
        inactive_upper = max(inactive_uppers)
        response_lower = min(response_lowers)
        exponent_lower = min(exponent_lowers)
        exponent_upper = max(exponent_uppers)
        exponential_lower = endpoint(ball(exponent_lower).exp().lower().fmpq())
        exponential_upper = endpoint(ball(exponent_upper).exp().upper().fmpq())
        scale_lower = inactive_lower * exponential_lower
        scale_upper = inactive_upper * exponential_upper
        value_lower = scale_lower * active_lower
        value_upper = scale_upper * active_upper
        derivative = exact_weight_derivative(
            row,
            denominator,
            feature_coordinates,
            coordinate,
            output_dependent,
            width,
        )
        gate_lower += derivative * (value_lower if derivative >= 0 else value_upper)
        response_costs.append(
            response_lower * (scale_lower if response_lower >= 0 else scale_upper)
        )
    response_lower, _ = simplex_minimum(weight_lowers, weight_uppers, response_costs)
    total_lower = gate_lower + response_lower
    if total_lower <= 0:
        raise VerificationError(f"independent replay derivative lower is {total_lower}")
    return {
        "status": "independently replayed exact partition-aware derivative certificate",
        "precision_bits": precision_bits,
        "leaf": certificate["leaf"],
        "coordinate": coordinate,
        "fixed": certificate["fixed"],
        "Boolean_vertices": len(vertices),
        "weight_vertices": weight_vertex_count,
        "replay_gate_derivative_lower": str(gate_lower),
        "replay_bounded_simplex_response_lower": str(response_lower),
        "replay_total_derivative_lower": str(total_lower),
        "closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision-bits", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision_bits < 128:
        parser.error("precision must be at least 128 bits")
    try:
        summary = verify(
            args.manifest,
            args.partitions,
            args.certificate,
            args.precision_bits,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
