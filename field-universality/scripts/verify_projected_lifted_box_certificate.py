#!/usr/bin/env python3
"""Replay one rational lifted projected-KL box certificate for F_5."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from mpmath import iv

import verify_perron_point_certificate as intervals
import verify_stationary_box_certificate as domain


ZERO = Fraction(0)
ONE = Fraction(1)
PRIME = 5
COMPOSITIONS = 15
MOMENT_PAIRS = tuple(
    (left, right)
    for left in range(COMPOSITIONS)
    for right in range(left, COMPOSITIONS)
)
MOMENT_OFFSETS = {pair: offset for offset, pair in enumerate(MOMENT_PAIRS)}
MOMENT_START = COMPOSITIONS
MOMENT_END = MOMENT_START + len(MOMENT_PAIRS)
ENTROPY = MOMENT_END
STAGE_ONE = MOMENT_END + 1
STAGE_TWO = MOMENT_END + 2
VARIABLES = MOMENT_END + 3


@dataclass(frozen=True)
class Constraint:
    name: str
    coefficients: dict[int, Fraction]
    target: Fraction


@dataclass(frozen=True)
class ProjectionTangent:
    stage: int
    target_potential: tuple[Fraction, ...]
    increment_support_potential: Fraction
    state_support_potential: Fraction
    kernel: tuple[tuple[Fraction, ...], ...]
    partition_reference: Fraction


@dataclass(frozen=True)
class StationaryTangent:
    stage: int
    cut: domain.RationalCut
    partition_reference: Fraction


def parse_fraction(value: object) -> Fraction:
    return intervals.parse_fraction(value)


def text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def moment(left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    return MOMENT_START + MOMENT_OFFSETS[(left, right)]


def product(stage: int, left: int, right: int) -> int:
    starts = ((0, 5), (5, 10))
    left_start, right_start = starts[stage]
    return moment(left_start + left, right_start + right)


def clean(coefficients: dict[int, Fraction]) -> dict[int, Fraction]:
    return {index: value for index, value in coefficients.items() if value != ZERO}


def add_equality(
    constraints: list[Constraint],
    name: str,
    coefficients: dict[int, Fraction],
    target: Fraction,
) -> None:
    coefficients = clean(coefficients)
    constraints.append(Constraint(f"{name}_upper", coefficients, target))
    constraints.append(
        Constraint(
            f"{name}_lower",
            {index: -value for index, value in coefficients.items()},
            -target,
        )
    )


def base_inequalities() -> list[tuple[str, dict[int, Fraction]]]:
    result = [
        (
            "message_to_middle_support",
            {
                **{index: ONE for index in range(1, 5)},
                **{index: -2 * ONE for index in range(6, 10)},
            },
        ),
        (
            "middle_to_output_support",
            {
                **{index: ONE for index in range(6, 10)},
                **{index: -2 * ONE for index in range(11, 15)},
            },
        ),
    ]
    for value in range(2, 5):
        result.append((f"canonical_message_{value}", {value: ONE, 1: -ONE}))
    return result


def add_mccormick(
    constraints: list[Constraint],
    name: str,
    x: int,
    y: int,
    z: int,
    lx: Fraction,
    ux: Fraction,
    ly: Fraction,
    uy: Fraction,
) -> None:
    # All rows use the verifier convention row <= target.
    def row(*terms: tuple[int, Fraction]) -> dict[int, Fraction]:
        coefficients: dict[int, Fraction] = {}
        for index, value in terms:
            coefficients[index] = coefficients.get(index, ZERO) + value
        return clean(coefficients)

    constraints.append(Constraint(f"{name}_lower_ll", row((x, ly), (y, lx), (z, -ONE)), lx * ly))
    constraints.append(Constraint(f"{name}_lower_uu", row((x, uy), (y, ux), (z, -ONE)), ux * uy))
    constraints.append(Constraint(f"{name}_upper_ul", row((x, -ly), (y, -ux), (z, ONE)), -ux * ly))
    constraints.append(Constraint(f"{name}_upper_lu", row((x, -uy), (y, -lx), (z, ONE)), -lx * uy))


def parse_entropy_reference(raw: object) -> tuple[Fraction, ...]:
    reference = intervals.parse_vector(raw, PRIME)
    if any(value <= ZERO for value in reference) or sum(reference, ZERO) != ONE:
        raise ValueError("entropy tangent reference is not an interior distribution")
    return reference


def parse_projection_tangent(raw: object, precision: int) -> ProjectionTangent:
    if not isinstance(raw, dict):
        raise ValueError("projection tangent must be an object")
    stage = raw.get("stage")
    if stage not in (0, 1):
        raise ValueError("invalid projection stage")
    target = intervals.parse_vector(raw.get("target_potential"), PRIME)
    increment_support = parse_fraction(raw.get("increment_support_potential"))
    state_support = parse_fraction(raw.get("state_support_potential"))
    kernel = intervals.parse_matrix(raw.get("kernel_upper"), PRIME)
    reference = parse_fraction(raw.get("partition_reference"))
    if reference <= ZERO:
        raise ValueError("partition tangent reference must be positive")
    iv.dps = precision
    for difference in range(PRIME):
        for source in range(PRIME):
            exponent = (
                target[(difference + source) % PRIME]
                + increment_support * int(difference != 0)
                + state_support * int(source != 0)
            )
            exact_upper = intervals.rational_interval(
                iv.exp(intervals.iv_fraction(exponent))
            ).upper
            if kernel[difference][source] < exact_upper:
                raise ValueError("projection kernel entry does not upper-bound exp")
    return ProjectionTangent(
        int(stage), target, increment_support, state_support, kernel, reference
    )


def parse_stationary_tangent(raw: object, precision: int) -> StationaryTangent:
    if not isinstance(raw, dict) or raw.get("stage") not in (0, 1):
        raise ValueError("invalid stationary tangent")
    reference = parse_fraction(raw.get("partition_reference"))
    if reference <= ZERO:
        raise ValueError("stationary partition reference must be positive")
    return StationaryTangent(
        int(raw["stage"]), domain.parse_cut(raw.get("cut"), precision), reference
    )


def build_constraints(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    entropy_references: tuple[tuple[Fraction, ...], ...],
    projection_tangents: tuple[ProjectionTangent, ...],
    covariance_cuts: tuple[tuple[tuple[Fraction, ...], Fraction], ...],
    precision: int,
    stationary_tangents: tuple[StationaryTangent, ...] = (),
) -> list[Constraint]:
    constraints: list[Constraint] = []
    for block, start in enumerate((0, 5, 10)):
        add_equality(
            constraints,
            f"normalize_{block}",
            {start + offset: ONE for offset in range(PRIME)},
            ONE,
        )
    inequalities = base_inequalities()
    for name, coefficients in inequalities:
        constraints.append(Constraint(name, coefficients, ZERO))

    for left, right in MOMENT_PAIRS:
        add_mccormick(
            constraints,
            f"moment_{left}_{right}",
            left,
            right,
            moment(left, right),
            lower[left],
            upper[left],
            lower[right],
            upper[right],
        )

    # E[(sum_j X_j) X_k] = E[X_k] for each simplex block.
    for block, start in enumerate((0, 5, 10)):
        for coordinate in range(COMPOSITIONS):
            coefficients = {coordinate: -ONE}
            for member in range(start, start + PRIME):
                index = moment(member, coordinate)
                coefficients[index] = coefficients.get(index, ZERO) + ONE
            add_equality(
                constraints,
                f"moment_simplex_{block}_{coordinate}",
                coefficients,
                ZERO,
            )

    # First-level RLT for each homogeneous domain inequality g(x) <= 0.
    for inequality_name, inequality in inequalities:
        for coordinate in range(COMPOSITIONS):
            lower_row: dict[int, Fraction] = {}
            upper_row: dict[int, Fraction] = {}
            for member, coefficient in inequality.items():
                product_index = moment(member, coordinate)
                lower_row[product_index] = lower_row.get(product_index, ZERO) + coefficient
                lower_row[member] = lower_row.get(member, ZERO) - lower[coordinate] * coefficient
                upper_row[member] = upper_row.get(member, ZERO) + upper[coordinate] * coefficient
                upper_row[product_index] = upper_row.get(product_index, ZERO) - coefficient
            constraints.append(Constraint(f"rlt_{inequality_name}_{coordinate}_lower", clean(lower_row), ZERO))
            constraints.append(Constraint(f"rlt_{inequality_name}_{coordinate}_upper", clean(upper_row), ZERO))

    for cut_index, (direction, center) in enumerate(covariance_cuts):
        if len(direction) != COMPOSITIONS:
            raise ValueError("covariance direction has the wrong dimension")
        coefficients = {
            index: 2 * center * value
            for index, value in enumerate(direction)
            if value != ZERO
        }
        for left in range(COMPOSITIONS):
            coefficients[moment(left, left)] = coefficients.get(moment(left, left), ZERO) - direction[left] ** 2
            for right in range(left + 1, COMPOSITIONS):
                index = moment(left, right)
                coefficients[index] = coefficients.get(index, ZERO) - 2 * direction[left] * direction[right]
        constraints.append(Constraint(f"covariance_{cut_index}", clean(coefficients), center**2))

    iv.dps = precision
    for reference_index, reference in enumerate(entropy_references):
        coefficients = {ENTROPY: ONE}
        for symbol, value in enumerate(reference):
            log_lower = intervals.log_interval(value).lower
            coefficients[symbol] = log_lower / 3
        constraints.append(Constraint(f"entropy_tangent_{reference_index}", clean(coefficients), ZERO))

    for tangent_index, tangent in enumerate(projection_tangents):
        increment_start, state_start = ((0, 5), (5, 10))[tangent.stage]
        epigraph = STAGE_ONE if tangent.stage == 0 else STAGE_TWO
        coefficients: dict[int, Fraction] = {epigraph: ONE}
        for left in range(PRIME):
            for right in range(PRIME):
                index = product(tangent.stage, left, right)
                coefficients[index] = coefficients.get(index, ZERO) - tangent.kernel[left][right] / tangent.partition_reference
        for symbol, potential in enumerate(tangent.target_potential):
            coefficients[state_start + symbol] = coefficients.get(state_start + symbol, ZERO) + potential
        for symbol in range(1, PRIME):
            coefficients[increment_start + symbol] = coefficients.get(increment_start + symbol, ZERO) + tangent.increment_support_potential
            coefficients[state_start + symbol] = coefficients.get(state_start + symbol, ZERO) + tangent.state_support_potential
        target = intervals.log_interval(tangent.partition_reference).upper - ONE
        constraints.append(Constraint(f"projection_tangent_{tangent_index}", clean(coefficients), target))

    for tangent_index, tangent in enumerate(stationary_tangents):
        increment_start, state_start = ((0, 5), (5, 10))[tangent.stage]
        epigraph = STAGE_ONE if tangent.stage == 0 else STAGE_TWO
        coefficients: dict[int, Fraction] = {epigraph: ONE}
        moments = domain.moment_coefficients(tangent.cut)
        for symbol in range(PRIME):
            coefficients[increment_start + symbol] = moments[symbol]
            coefficients[state_start + symbol] = moments[5 + symbol]
        for left in range(PRIME):
            for right in range(PRIME):
                index = product(tangent.stage, left, right)
                coefficients[index] = coefficients.get(index, ZERO) - tangent.cut.kernel[left][right] / tangent.partition_reference
        target = (
            tangent.cut.shift
            + intervals.log_interval(tangent.partition_reference).upper
            - ONE
        )
        constraints.append(Constraint(f"stationary_tangent_{tangent_index}", clean(coefficients), target))

    # Explicit bounds let the dual treat every primal variable as unrestricted.
    for index in range(VARIABLES):
        if index < COMPOSITIONS:
            left, right = lower[index], upper[index]
        elif index < MOMENT_END:
            left, right = ZERO, ONE
        else:
            left, right = Fraction(-1000), Fraction(1000)
        constraints.append(Constraint(f"variable_{index}_upper", {index: ONE}, right))
        constraints.append(Constraint(f"variable_{index}_lower", {index: -ONE}, -left))
    return constraints


def verify(raw: object, precision: int) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-projected-lifted-box-v1":
        raise ValueError("unsupported projected lifted box schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("unexpected code parameters")
    eta = parse_fraction(raw.get("eta"))
    delta = parse_fraction(raw.get("delta"))
    if eta != Fraction(1, 500) or delta != Fraction(11, 50):
        raise ValueError("unexpected support cutoff or target distance")
    lower, upper = domain.parse_box(raw.get("box"))
    domain.verify_domain_box(lower, upper, eta, delta)
    entropy_raw = raw.get("entropy_references")
    tangent_raw = raw.get("projection_tangents")
    covariance_raw = raw.get("covariance_cuts")
    stationary_raw = raw.get("stationary_tangents", [])
    if not isinstance(entropy_raw, list) or not entropy_raw:
        raise ValueError("missing entropy tangents")
    if not isinstance(tangent_raw, list) or not tangent_raw:
        raise ValueError("missing projection tangents")
    if not isinstance(covariance_raw, list):
        raise ValueError("missing covariance cut list")
    if not isinstance(stationary_raw, list):
        raise ValueError("stationary tangent list must be an array")
    entropy_references = tuple(parse_entropy_reference(value) for value in entropy_raw)
    tangents = tuple(parse_projection_tangent(value, precision) for value in tangent_raw)
    stationary_tangents = tuple(
        parse_stationary_tangent(value, precision) for value in stationary_raw
    )
    if not any(tangent.stage == 0 for tangent in tangents) or not any(tangent.stage == 1 for tangent in tangents):
        raise ValueError("both projected stages need a tangent")
    covariance_cuts = []
    for value in covariance_raw:
        if not isinstance(value, dict):
            raise ValueError("covariance cut must be an object")
        covariance_cuts.append(
            (
                intervals.parse_vector(value.get("direction"), COMPOSITIONS),
                parse_fraction(value.get("center")),
            )
        )
    constraints = build_constraints(
        lower,
        upper,
        entropy_references,
        tangents,
        tuple(covariance_cuts),
        precision,
        stationary_tangents,
    )
    by_name = {constraint.name: constraint for constraint in constraints}
    if len(by_name) != len(constraints):
        raise ValueError("constraint names are not unique")
    stored_dual = raw.get("dual")
    if not isinstance(stored_dual, dict):
        raise ValueError("missing rational LP dual")
    dual_left = [ZERO] * VARIABLES
    dual_value = ZERO
    nonzero = 0
    for name, raw_multiplier in stored_dual.items():
        if name not in by_name:
            raise ValueError(f"unknown LP constraint {name}")
        multiplier = parse_fraction(raw_multiplier)
        if multiplier < ZERO:
            raise ValueError("LP dual multiplier is negative")
        if multiplier == ZERO:
            continue
        nonzero += 1
        constraint = by_name[name]
        dual_value += multiplier * constraint.target
        for index, coefficient in constraint.coefficients.items():
            dual_left[index] += multiplier * coefficient
    objective = [ZERO] * VARIABLES
    objective[ENTROPY] = objective[STAGE_ONE] = objective[STAGE_TWO] = ONE
    for index, coefficient in enumerate(objective):
        if dual_left[index] != coefficient:
            raise ValueError(f"LP dual equality fails at variable {index}")
    if dual_value >= ZERO:
        raise ValueError("LP dual does not prove a negative lifted box upper")
    return {
        "status": "verified rational projected lifted box certificate",
        "box_upper": float(dual_value),
        "box_upper_numerator": str(dual_value.numerator),
        "box_upper_denominator": str(dual_value.denominator),
        "box_upper_is_exactly_negative": True,
        "variables": VARIABLES,
        "constraints_checked": len(constraints),
        "entropy_tangents_checked": len(entropy_references),
        "projection_tangents_checked": len(tangents),
        "stationary_tangents_checked": len(stationary_tangents),
        "covariance_cuts_checked": len(covariance_cuts),
        "nonzero_dual_multipliers": nonzero,
        "interval_precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    raw = json.loads(args.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(raw, args.precision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
