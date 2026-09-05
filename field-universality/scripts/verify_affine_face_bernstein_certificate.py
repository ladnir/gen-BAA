#!/usr/bin/env python3
"""Independently replay an exact Arb/Bernstein affine-face certificate.

This verifier does not import the certificate generator.  It reconstructs the
chamber face and its dyadic cover from the source artifacts, rechecks the exact
affine partition, rebuilds the exponential polynomial, and recomputes every
Bernstein/Taylor upper bound with fresh Arb balls.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from flint import arb, ctx, fmpq


Exponent = tuple[int, ...]
Polynomial = dict[Exponent, arb]


class VerificationError(ValueError):
    pass


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]


def load_exact(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Fraction)


def q(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def ball(value: Any) -> arb:
    value = q(value)
    return arb(fmpq(value.numerator, value.denominator))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_leaves(node: dict) -> list[dict]:
    if "leaf" in node:
        return [node]
    return [*collect_leaves(node["le"]), *collect_leaves(node["ge"])]


def chamber_bounds(
    lower: list[Fraction], upper: list[Fraction], restrictions: list[str]
) -> tuple[list[Fraction], list[Fraction]]:
    lower = list(lower)
    upper = list(upper)
    for restriction in restrictions:
        coordinate_text, side, threshold_text = restriction.split(":")
        coordinate = int(coordinate_text)
        threshold = Fraction(threshold_text)
        if side == "le":
            upper[coordinate] = min(upper[coordinate], threshold)
        elif side == "ge":
            lower[coordinate] = max(lower[coordinate], threshold)
        else:
            raise VerificationError(f"invalid chamber restriction {restriction}")
    if any(a > b for a, b in zip(lower, upper, strict=True)):
        raise VerificationError("chamber restriction produced an empty box")
    return lower, upper


def tighten(box: Box, variables: list[int], dependent: list[int]) -> Box | None:
    lower = list(box.lower)
    upper = list(box.upper)
    for start in (0, 5, 10):
        block_variables = [value for value in variables if start <= value < start + 5]
        block_dependent = [value for value in dependent if start <= value < start + 5]
        if len(block_dependent) != 1:
            raise VerificationError("each simplex block needs one dependent coordinate")
        active = [*block_variables, block_dependent[0]]
        fixed_sum = sum(
            (lower[index] for index in range(start, start + 5) if index not in active),
            Fraction(0),
        )
        target = 1 - fixed_sum
        changed = True
        while changed:
            changed = False
            if sum((lower[index] for index in active), Fraction(0)) > target:
                return None
            if sum((upper[index] for index in active), Fraction(0)) < target:
                return None
            for index in active:
                new_lower = max(
                    lower[index],
                    target
                    - sum(
                        (upper[other] for other in active if other != index),
                        Fraction(0),
                    ),
                )
                new_upper = min(
                    upper[index],
                    target
                    - sum(
                        (lower[other] for other in active if other != index),
                        Fraction(0),
                    ),
                )
                if new_lower > new_upper:
                    return None
                if new_lower != lower[index] or new_upper != upper[index]:
                    lower[index] = new_lower
                    upper[index] = new_upper
                    changed = True
    return Box(tuple(lower), tuple(upper))


def apply_face(
    lower: list[Fraction], upper: list[Fraction], face: dict
) -> tuple[Box, list[int], list[int]]:
    lower = list(lower)
    upper = list(upper)
    fixed: set[int] = set()
    for specification in face["fixed"]:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            raise VerificationError(f"invalid face side {side}")
    dependent = [int(value) for value in face["dependent_coordinates"]]
    if fixed.intersection(dependent):
        raise VerificationError("fixed and dependent coordinates overlap")
    variables = [
        index for index in range(15) if index not in fixed and index not in dependent
    ]
    result = tighten(Box(tuple(lower), tuple(upper)), variables, dependent)
    if result is None:
        raise VerificationError("claimed face is empty")
    return result, variables, dependent


def dyadic_cover(
    base: Box,
    variables: list[int],
    dependent: list[int],
    level: int,
    split_coordinates: list[int] | None = None,
    schedule: list[int] | None = None,
) -> list[Box]:
    boxes = [base]
    split_coordinates = variables if split_coordinates is None else split_coordinates
    if len(set(split_coordinates)) != len(split_coordinates):
        raise VerificationError("dyadic split coordinates are not distinct")
    if any(value not in variables for value in split_coordinates):
        raise VerificationError("dyadic split coordinate is not free on the face")
    schedule = split_coordinates * level if schedule is None else schedule
    if any(value not in variables for value in schedule):
        raise VerificationError("dyadic schedule coordinate is not free on the face")
    for coordinate in schedule:
        children: list[Box] = []
        for parent in boxes:
            midpoint = (parent.lower[coordinate] + parent.upper[coordinate]) / 2
            for left in (True, False):
                lower = list(parent.lower)
                upper = list(parent.upper)
                if left:
                    upper[coordinate] = midpoint
                else:
                    lower[coordinate] = midpoint
                child = tighten(Box(tuple(lower), tuple(upper)), variables, dependent)
                if child is not None:
                    children.append(child)
        boxes = children
    return boxes


def parse_box(raw: dict) -> Box:
    return Box(
        tuple(Fraction(value) for value in raw["lower"]),
        tuple(Fraction(value) for value in raw["upper"]),
    )


def simplex_vertices(
    lower: list[Fraction], upper: list[Fraction], start: int
) -> list[tuple[Fraction, ...]]:
    lo = lower[start : start + 5]
    hi = upper[start : start + 5]
    vertices: set[tuple[Fraction, ...]] = set()
    for free in range(5):
        fixed = [index for index in range(5) if index != free]
        for bits in itertools.product((0, 1), repeat=4):
            point: list[Fraction | None] = [None] * 5
            for bit, index in zip(bits, fixed, strict=True):
                point[index] = hi[index] if bit else lo[index]
            point[free] = 1 - sum((point[index] for index in fixed), Fraction(0))
            if lo[free] <= point[free] <= hi[free]:
                vertices.add(tuple(value for value in point if value is not None))
    if not vertices:
        raise VerificationError(f"simplex block {start} has no vertices")
    return sorted(vertices)


def verify_partition(
    record: dict,
    chamber_lower: list[Fraction],
    chamber_upper: list[Fraction],
) -> Fraction:
    denominator = int(record["coefficient_denominator"])
    coordinates = [int(value) for value in record["feature_coordinates"]]
    numerators = record["coefficient_numerators"]
    if denominator <= 0 or not numerators:
        raise VerificationError("malformed affine partition")
    if any(len(row) != len(coordinates) + 1 for row in numerators):
        raise VerificationError("malformed affine coefficient row")
    for column_index, column in enumerate(zip(*numerators, strict=True)):
        expected = denominator if column_index == 0 else 0
        if sum(column) != expected:
            raise VerificationError("affine partition identity failed")
    blocks = [
        simplex_vertices(chamber_lower, chamber_upper, start)
        for start in (0, 5, 10)
    ]
    minimum: Fraction | None = None
    for product in itertools.product(*blocks):
        point = (*product[0], *product[1], *product[2])
        for row in numerators:
            value = (
                row[0]
                + sum(
                    (coefficient * point[coordinate]
                     for coefficient, coordinate in zip(row[1:], coordinates, strict=True)),
                    Fraction(0),
                )
            ) / denominator
            minimum = value if minimum is None else min(minimum, value)
    if minimum is None or minimum < 0:
        raise VerificationError(f"affine partition has negative chamber weight {minimum}")
    if minimum != Fraction(record["minimum_exact_weight"]):
        raise VerificationError("saved affine partition minimum does not replay")
    return minimum


def const(value: Any, dimension: int) -> Polynomial:
    return {(0,) * dimension: value if isinstance(value, arb) else ball(value)}


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, arb(0)) + coefficient
    return result


def scale(polynomial: Polynomial, scalar: Any) -> Polynomial:
    scalar = scalar if isinstance(scalar, arb) else ball(scalar)
    return {exponent: scalar * coefficient for exponent, coefficient in polynomial.items()}


def mul(left: Polynomial, right: Polynomial) -> Polynomial:
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


def face_polynomials(box: Box, variables: list[int], dependent: list[int]) -> list[Polynomial]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    result: list[Polynomial | None] = [None] * 15
    for coordinate in range(15):
        if coordinate in dependent:
            continue
        polynomial = const(box.lower[coordinate], dimension)
        if coordinate in positions and box.lower[coordinate] != box.upper[coordinate]:
            exponent = [0] * dimension
            exponent[positions[coordinate]] = 1
            polynomial = add(
                polynomial,
                {tuple(exponent): ball(box.upper[coordinate] - box.lower[coordinate])},
            )
        result[coordinate] = polynomial
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        polynomial = const(1, dimension)
        for other in range(start, start + 5):
            if other != coordinate:
                polynomial = add(polynomial, scale(result[other], -1))
        result[coordinate] = polynomial
    return [value for value in result if value is not None]


def vertex_value(polynomial: Polynomial, vertex: tuple[int, ...]) -> arb:
    total = arb(0)
    for exponent, coefficient in polynomial.items():
        if all(power == 0 or vertex[index] == 1 for index, power in enumerate(exponent)):
            total += coefficient
    return total


def vertex_bounds(polynomial: Polynomial, dimension: int) -> tuple[fmpq, fmpq]:
    if any(any(power < 0 or power > 1 for power in exponent) for exponent in polynomial):
        raise VerificationError("vertex extremum requested for non-multiaffine polynomial")
    values = [
        vertex_value(polynomial, vertex)
        for vertex in itertools.product((0, 1), repeat=dimension)
    ]
    return (
        min(value.lower().fmpq() for value in values),
        max(value.upper().fmpq() for value in values),
    )


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


def stage(left: list[Polynomial], right: list[Polynomial], cut, dimension: int):
    target, increment, state, kernel = cut
    partition: Polynomial = {}
    for difference in range(5):
        for source in range(5):
            partition = add(
                partition,
                scale(mul(left[difference], right[source]), kernel[difference][source]),
            )
    linear: Polynomial = {}
    for index in range(5):
        linear = add(linear, scale(right[index], target[index]))
    for value in left[1:]:
        linear = add(linear, scale(value, increment))
    for value in right[1:]:
        linear = add(linear, scale(value, state))
    if any(sum(exponent) > 1 for exponent in linear):
        raise VerificationError("dual exponent is not affine")
    return partition, linear


def affine_center(polynomial: Polynomial, dimension: int) -> tuple[arb, list[arb]]:
    center = polynomial.get((0,) * dimension, arb(0))
    coefficients: list[arb] = []
    allowed = {(0,) * dimension}
    for position in range(dimension):
        exponent = tuple(1 if index == position else 0 for index in range(dimension))
        allowed.add(exponent)
        coefficient = polynomial.get(exponent, arb(0))
        coefficients.append(coefficient)
        center += coefficient / 2
    if set(polynomial).difference(allowed):
        raise VerificationError("claimed affine exponent has higher-order terms")
    return center, coefficients


def entropy_tangent_polynomial(
    message: list[Polynomial], dimension: int
) -> Polynomial:
    bounds = [vertex_bounds(value, dimension) for value in message]
    if any(lower <= 0 for lower, _ in bounds):
        raise VerificationError("message is not strictly positive on tangent box")
    centers = [affine_center(value, dimension)[0] for value in message]
    entropy = -sum((value * value.log() for value in centers), arb(0)) / 3
    gradients = [-(value.log() + 1) / 3 for value in centers]
    tangent = const(
        entropy
        - sum(
            (left * right for left, right in zip(gradients, centers, strict=True)),
            arb(0),
        ),
        dimension,
    )
    for slope, coordinate in zip(gradients, message, strict=True):
        tangent = add(tangent, scale(coordinate, slope))
    return tangent


def bernstein_coefficients(polynomial: Polynomial, degree: int, dimension: int):
    degrees = [
        max((exponent[position] for exponent in polynomial), default=0)
        for position in range(dimension)
    ]
    if max(degrees, default=0) > degree:
        raise VerificationError("Bernstein degree is too small")
    shape = (degree + 1,) * dimension
    coefficients = np.empty(shape, dtype=object)
    coefficients.fill(arb(0))
    for exponent, value in polynomial.items():
        coefficients[exponent] = coefficients[exponent] + value
    transform = np.empty((degree + 1, degree + 1), dtype=object)
    transform.fill(arb(0))
    for index in range(degree + 1):
        for power in range(index + 1):
            transform[index, power] = arb(
                fmpq(math.comb(index, power), math.comb(degree, power))
            )
    # Apply the separable transform as explicit object-matrix products.  This
    # is deliberately distinct from the generator's tensordot implementation.
    result = coefficients
    for axis in range(dimension):
        moved = np.moveaxis(result, axis, 0)
        converted = transform @ moved.reshape(degree + 1, -1)
        result = np.moveaxis(converted.reshape(moved.shape), 0, axis)
    return result, degrees


def replay_box(
    box: Box,
    variables: list[int],
    dependent: list[int],
    record: dict,
    first_cuts: list,
    second_cuts: list,
    taylor_degree: int,
    bernstein_degree: int,
    entropy_mode: str,
) -> dict:
    dimension = len(variables)
    point = face_polynomials(box, variables, dependent)
    if entropy_mode == "fixed-exact":
        message = [vertex_value(value, (0,) * dimension) for value in point[0:5]]
        for value in point[0:5]:
            if any(any(exponent) for exponent in value):
                raise VerificationError("message is not fixed on face")
        entropy_factor = (
            -sum((value * value.log() for value in message), arb(0)) / 3
        ).exp()
        entropy_exponent = const(0, dimension)
    elif entropy_mode == "concave-tangent-upper":
        entropy_factor = arb(1)
        entropy_exponent = entropy_tangent_polynomial(point[0:5], dimension)
    else:
        raise VerificationError(f"unknown entropy mode {entropy_mode}")
    denominator = int(record["coefficient_denominator"])
    coordinates = [int(value) for value in record["feature_coordinates"]]
    approximation: Polynomial = {}
    remainder = arb(0)
    for numerator_row, pair in zip(
        record["coefficient_numerators"], record["expert_pairs"], strict=True
    ):
        weight = const(Fraction(numerator_row[0], denominator), dimension)
        for numerator, coordinate in zip(numerator_row[1:], coordinates, strict=True):
            weight = add(weight, scale(point[coordinate], Fraction(numerator, denominator)))
        first_partition, first_linear = stage(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])], dimension
        )
        second_partition, second_linear = stage(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])], dimension
        )
        weight_lower, weight_upper = vertex_bounds(weight, dimension)
        first_lower, first_upper = vertex_bounds(first_partition, dimension)
        second_lower, second_upper = vertex_bounds(second_partition, dimension)
        if first_lower <= 0 or second_lower <= 0:
            raise VerificationError("stage partition is not strictly positive")
        exponent = add(
            scale(add(first_linear, second_linear), -1), entropy_exponent
        )
        center, slopes = affine_center(exponent, dimension)
        centered = const(-sum(slopes, arb(0)) / 2, dimension)
        for position, slope in enumerate(slopes):
            power = tuple(1 if index == position else 0 for index in range(dimension))
            centered = add(centered, {power: slope})
        exponential = const(1, dimension)
        power_polynomial = const(1, dimension)
        for order in range(1, taylor_degree + 1):
            power_polynomial = mul(power_polynomial, centered)
            exponential = add(
                exponential,
                scale(power_polynomial, Fraction(1, math.factorial(order))),
            )
        factor = mul(mul(weight, first_partition), second_partition)
        approximation = add(
            approximation,
            scale(mul(factor, exponential), entropy_factor * center.exp()),
        )
        rho = sum((abs(value) for value in slopes), arb(0)) / 2
        remainder += (
            entropy_factor
            * arb(max(-weight_lower, weight_upper))
            * arb(first_upper)
            * arb(second_upper)
            * (center + rho).exp()
            * rho ** (taylor_degree + 1)
            / math.factorial(taylor_degree + 1)
        )
    approximation = add(approximation, const(-1, dimension))
    bernstein, degrees = bernstein_coefficients(
        approximation, bernstein_degree, dimension
    )
    maximum: fmpq | None = None
    maximum_index: tuple[int, ...] | None = None
    for flat_index, value in enumerate(bernstein.flat):
        upper = value.upper().fmpq()
        if maximum is None or upper > maximum:
            maximum = upper
            maximum_index = tuple(
                int(index) for index in np.unravel_index(flat_index, bernstein.shape)
            )
    if maximum is None:
        raise VerificationError("empty Bernstein tensor")
    remainder_upper = remainder.upper().fmpq()
    return {
        "power_degrees": degrees,
        "power_terms": len(approximation),
        "bernstein_coefficients": int(bernstein.size),
        "maximum_index": maximum_index,
        "maximum_upper": maximum,
        "remainder_upper": remainder_upper,
        "total_upper": maximum + remainder_upper,
    }


def verify(
    manifest_path: Path,
    partitions_path: Path,
    faces_path: Path,
    certificate_path: Path,
    precision_bits: int,
) -> dict:
    ctx.prec = precision_bits
    manifest_path = manifest_path.resolve()
    partitions_path = partitions_path.resolve()
    faces_path = faces_path.resolve()
    certificate_path = certificate_path.resolve()
    manifest = load_exact(manifest_path)
    partitions = load_exact(partitions_path)
    faces = load_exact(faces_path)
    certificate = load_exact(certificate_path)
    if certificate.get("rigorous") is not True or certificate.get("closed") is not True:
        raise VerificationError("certificate is not marked rigorous and closed")
    root = (manifest_path.parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf_name = certificate["leaf"]
    leaf = leaves[leaf_name]
    record = next(value for value in partitions["leaves"] if value["leaf"] == leaf_name)
    if faces["leaf"] != leaf_name:
        raise VerificationError("face artifact belongs to another leaf")
    face = next(value for value in faces["faces"] if value["name"] == certificate["face"])
    gate_path = root / str(record.get("gate", leaf["gate"])).replace("\\", "/")
    gate = load_exact(gate_path)
    result_path = Path(str(record.get("result", leaf["result"])).replace("\\", "/"))
    if not result_path.is_absolute():
        result_path = root / result_path
    result = load_exact(result_path)
    expected_pairs = list(result["coefficient_expert_pairs"])
    prune_threshold = float(record.get("prune_coefficient_norm", 0.0))
    if prune_threshold > 0.0:
        source_coefficients = np.asarray(result["final_coefficients"], dtype=float)
        if len(source_coefficients) != len(expected_pairs):
            raise VerificationError("source optimizer coefficient/pair count differs")
        retained = np.max(np.abs(source_coefficients), axis=1) > prune_threshold
        expected_pairs = [
            pair for pair, keep in zip(expected_pairs, retained, strict=True) if keep
        ]
        if int(record.get("floating_experts_before_pruning", -1)) != len(retained):
            raise VerificationError("saved pre-pruning expert count differs")
        if int(record.get("pruned_floating_experts", -1)) != int(
            len(retained) - np.count_nonzero(retained)
        ):
            raise VerificationError("saved pruned expert count differs")
    if record["expert_pairs"] != expected_pairs:
        raise VerificationError("expert pairs do not match source optimizer result")
    if [int(value) for value in record["feature_coordinates"]] != [
        int(value) for value in gate["affine_gating"]["feature_coordinates"]
    ]:
        raise VerificationError("feature coordinates do not match gate")
    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    expected_hashes = {
        "manifest": file_hash(manifest_path),
        "partitions": file_hash(partitions_path),
        "faces": file_hash(faces_path),
        "dual_bank": file_hash(bank_path),
        "gate": file_hash(gate_path),
    }
    if certificate.get("source_sha256") != expected_hashes:
        raise VerificationError("certificate source hashes do not match current files")
    checkpoint = load_exact(root / manifest["global_checkpoint"])
    selected = sorted(checkpoint["queue"], key=lambda entry: entry["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    base_lower = [q(value) for value in selected["lower"]]
    base_upper = [q(value) for value in selected["upper"]]
    chamber_lower, chamber_upper = chamber_bounds(
        base_lower, base_upper, gate["restrictions"]
    )
    partition_lower = list(chamber_lower)
    partition_upper = list(chamber_upper)
    for specification in record.get("fixed", []):
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if side == "lower":
            partition_upper[coordinate] = partition_lower[coordinate]
        elif side == "upper":
            partition_lower[coordinate] = partition_upper[coordinate]
        else:
            raise VerificationError(f"invalid partition fixed side {side}")
    partition_minimum = verify_partition(record, partition_lower, partition_upper)
    base_box, variables, dependent = apply_face(chamber_lower, chamber_upper, face)
    if variables != [int(value) for value in certificate["variables"]]:
        raise VerificationError("free-coordinate list does not replay")
    if dependent != [int(value) for value in certificate["dependent_coordinates"]]:
        raise VerificationError("dependent-coordinate list does not replay")
    if parse_box(certificate["base_face_box"]) != base_box:
        raise VerificationError("base face box does not replay")
    split_coordinates = [
        int(value) for value in certificate.get("dyadic_coordinates", variables)
    ]
    boxes = dyadic_cover(
        base_box,
        variables,
        dependent,
        int(certificate["dyadic_level"]),
        split_coordinates,
        [int(value) for value in certificate["dyadic_schedule"]]
        if "dyadic_schedule" in certificate
        else None,
    )
    records = certificate["box_records"]
    if len(boxes) != int(certificate["total_cover_boxes"]) or len(records) != len(boxes):
        raise VerificationError("certificate does not contain the full dyadic cover")
    for expected, saved in zip(boxes, records, strict=True):
        if parse_box(saved["box"]) != expected:
            raise VerificationError("saved box differs from reconstructed dyadic cover")
        saved_total = Fraction(saved["mixture_minus_one_upper"])
        if saved_total != (
            Fraction(saved["maximum_bernstein_coefficient_upper"])
            + Fraction(saved["exponential_taylor_remainder_upper"])
        ):
            raise VerificationError("saved rational total is not the sum of its parts")
        if saved_total >= 0 or saved.get("closed") is not True:
            raise VerificationError("saved box is not strictly closed")
    bank = load_exact(bank_path)
    first_cuts = [load_cut(value) for value in bank["first_cuts"]]
    second_cuts = [load_cut(value) for value in bank["second_cuts"]]
    replayed = []
    entropy_mode = str(certificate.get("entropy_mode", "fixed-exact"))
    if entropy_mode not in ("fixed-exact", "concave-tangent-upper"):
        raise VerificationError("certificate has an unknown entropy mode")
    proof_mode = certificate.get("proof_preconditions", {}).get(
        "message_entropy_mode"
    )
    if proof_mode != entropy_mode:
        raise VerificationError("certificate entropy mode disagrees with preconditions")
    for index, (box, saved) in enumerate(zip(boxes, records, strict=True), start=1):
        result = replay_box(
            box,
            variables,
            dependent,
            record,
            first_cuts,
            second_cuts,
            int(certificate["taylor_degree"]),
            int(certificate["bernstein_degree"]),
            entropy_mode,
        )
        if result["power_degrees"] != [int(value) for value in saved["power_degrees"]]:
            raise VerificationError(f"box {index}: power degrees differ")
        if result["power_terms"] != int(saved["power_terms"]):
            raise VerificationError(f"box {index}: power term count differs")
        if result["bernstein_coefficients"] != int(saved["bernstein_coefficients"]):
            raise VerificationError(f"box {index}: Bernstein tensor size differs")
        if result["total_upper"] >= 0:
            raise VerificationError(
                f"box {index}: independent replay upper bound is {result['total_upper']}"
            )
        replayed.append(result)
        print(
            json.dumps(
                {
                    "box": index,
                    "boxes": len(boxes),
                    "replay_upper": str(result["total_upper"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    worst = max(replayed, key=lambda value: value["total_upper"])
    return {
        "status": "independently replayed exact Arb/Bernstein face certificate",
        "precision_bits": precision_bits,
        "leaf": leaf_name,
        "face": certificate["face"],
        "boxes": len(boxes),
        "partition_minimum": str(partition_minimum),
        "worst_replay_upper": str(worst["total_upper"]),
        "closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision-bits", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision_bits < 128:
        parser.error("verification precision must be at least 128 bits")
    try:
        summary = verify(
            args.manifest,
            args.partitions,
            args.faces,
            args.certificate,
            args.precision_bits,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
