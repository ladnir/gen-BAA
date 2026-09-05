#!/usr/bin/env python3
"""Independently replay one rational stationary-dual box certificate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from mpmath import iv

import verify_perron_point_certificate as intervals


ZERO = Fraction(0)
ONE = Fraction(1)
THREE = Fraction(3)
PRIME = 5
COMPOSITIONS = 15
VARIABLES = 65


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    coefficients: dict[int, Fraction]
    target: Fraction


@dataclass(frozen=True)
class RationalCut:
    potentials: tuple[Fraction, ...]
    shift: Fraction
    kernel: tuple[tuple[Fraction, ...], ...]


def parse_fraction(value: object) -> Fraction:
    return intervals.parse_fraction(value)


def parse_vector(value: object, length: int) -> tuple[Fraction, ...]:
    return intervals.parse_vector(value, length)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def product(stage: int, left: int, right: int) -> int:
    return 15 + 25 * stage + 5 * left + right


def add_equality(
    constraints: list[LinearConstraint],
    name: str,
    coefficients: dict[int, Fraction],
    target: Fraction,
) -> None:
    constraints.append(LinearConstraint(f"{name}_upper", coefficients, target))
    constraints.append(
        LinearConstraint(
            f"{name}_lower",
            {index: -coefficient for index, coefficient in coefficients.items()},
            -target,
        )
    )


def parse_box(raw: object) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("missing composition box")
    lower = parse_vector(raw.get("lower"), COMPOSITIONS)
    upper = parse_vector(raw.get("upper"), COMPOSITIONS)
    for left, right in zip(lower, upper, strict=True):
        if left < ZERO or left > right or right > ONE:
            raise ValueError("invalid composition interval")
    for start in (0, 5, 10):
        if sum(lower[start : start + 5], ZERO) > ONE:
            raise ValueError("box lower bounds miss a composition simplex")
        if sum(upper[start : start + 5], ZERO) < ONE:
            raise ValueError("box upper bounds miss a composition simplex")
    return lower, upper


def verify_domain_box(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    eta: Fraction,
    delta: Fraction,
) -> None:
    support_intervals = (
        (0, eta, min(ONE, 4 * delta)),
        (5, eta / 2, min(ONE, 2 * delta)),
        (10, eta / 4, delta),
    )
    for start, support_lower, support_upper in support_intervals:
        if upper[start] > ONE - support_lower:
            raise ValueError("composition box escapes the support lower bound")
        if lower[start] < ONE - support_upper:
            raise ValueError("composition box escapes the support upper bound")
        for index in range(start + 1, start + 5):
            if upper[index] > support_upper:
                raise ValueError("composition box escapes a coordinate support bound")


def verify_composition_witness(
    raw: object,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
) -> None:
    witness = parse_vector(raw, COMPOSITIONS)
    for start in (0, 5, 10):
        if sum(witness[start : start + 5], ZERO) != ONE:
            raise ValueError("composition witness is not normalized")
    # This is a global tangent reference, not a claimed feasible composition.
    # Entropy and log tangents remain valid when the reference lies outside the
    # box, so containment and code-domain inequalities are intentionally not
    # imposed here.


def cut_log_kernel(potentials: tuple[Fraction, ...], difference: int, source: int) -> Fraction:
    if len(potentials) != 12:
        raise ValueError("a stationary cut needs twelve potentials")
    target = (difference + source) % PRIME
    value = ZERO
    if difference < 4:
        value += potentials[difference]
    if source in (0, 2, 3, 4):
        value += potentials[4 + (0 if source == 0 else source - 1)]
    if target < 4:
        value += potentials[8 + target]
    return value


def parse_cut(raw: object, precision: int) -> RationalCut:
    if not isinstance(raw, dict):
        raise ValueError("missing stationary cut")
    potentials = parse_vector(raw.get("potentials"), 12)
    shift = parse_fraction(raw.get("shift"))
    kernel = intervals.parse_matrix(raw.get("kernel_upper"), PRIME)
    iv.dps = precision
    for difference in range(PRIME):
        for source in range(PRIME):
            exponent = cut_log_kernel(potentials, difference, source) - shift
            exact_upper = intervals.rational_interval(
                iv.exp(intervals.iv_fraction(exponent))
            ).upper
            if kernel[difference][source] < exact_upper:
                raise ValueError(
                    f"kernel entry ({difference},{source}) does not upper-bound exp"
                )
    return RationalCut(potentials, shift, kernel)


def moment_coefficients(cut: RationalCut) -> tuple[Fraction, ...]:
    coefficients = [ZERO] * 10
    for difference in range(4):
        coefficients[difference] += cut.potentials[difference]
    for source in (0, 2, 3, 4):
        offset = 0 if source == 0 else source - 1
        coefficients[5 + source] += cut.potentials[4 + offset]
    for target in range(4):
        coefficients[5 + target] += cut.potentials[8 + target]
    return tuple(coefficients)


def build_constraints(
    lower: tuple[Fraction, ...], upper: tuple[Fraction, ...]
) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    for block, start in enumerate((0, 5, 10)):
        add_equality(
            constraints,
            f"normalize_{block}",
            {start + index: ONE for index in range(5)},
            ONE,
        )

    constraints.append(
        LinearConstraint(
            "message_to_middle_support",
            {
                **{index: ONE for index in range(1, 5)},
                **{5 + index: -Fraction(2) for index in range(1, 5)},
            },
            ZERO,
        )
    )
    constraints.append(
        LinearConstraint(
            "middle_to_output_support",
            {
                **{5 + index: ONE for index in range(1, 5)},
                **{10 + index: -Fraction(2) for index in range(1, 5)},
            },
            ZERO,
        )
    )
    for value in range(2, 5):
        constraints.append(
            LinearConstraint(
                f"canonical_message_{value}",
                {value: ONE, 1: -ONE},
                ZERO,
            )
        )

    for index in range(COMPOSITIONS):
        constraints.append(
            LinearConstraint(f"box_{index}_upper", {index: ONE}, upper[index])
        )
        constraints.append(
            LinearConstraint(f"box_{index}_lower", {index: -ONE}, -lower[index])
        )

    for stage, left_start, right_start in ((0, 0, 5), (1, 5, 10)):
        for left in range(PRIME):
            lx = lower[left_start + left]
            ux = upper[left_start + left]
            for right in range(PRIME):
                ly = lower[right_start + right]
                uy = upper[right_start + right]
                z = product(stage, left, right)
                x = left_start + left
                y = right_start + right
                # Convex envelope of z=x*y on [lx,ux] times [ly,uy].
                constraints.append(
                    LinearConstraint(
                        f"mc_{stage}_{left}_{right}_lower_ll",
                        {z: -ONE, x: ly, y: lx},
                        lx * ly,
                    )
                )
                constraints.append(
                    LinearConstraint(
                        f"mc_{stage}_{left}_{right}_lower_uu",
                        {z: -ONE, x: uy, y: ux},
                        ux * uy,
                    )
                )
                constraints.append(
                    LinearConstraint(
                        f"mc_{stage}_{left}_{right}_upper_ul",
                        {z: ONE, x: -ly, y: -ux},
                        -ux * ly,
                    )
                )
                constraints.append(
                    LinearConstraint(
                        f"mc_{stage}_{left}_{right}_upper_lu",
                        {z: ONE, x: -uy, y: -lx},
                        -lx * uy,
                    )
                )

    for index in range(VARIABLES):
        constraints.append(
            LinearConstraint(f"variable_{index}_upper", {index: ONE}, ONE)
        )
    return constraints


def objective_upper(
    raw: object,
    first: RationalCut,
    second: RationalCut,
    precision: int,
) -> tuple[Fraction, tuple[Fraction, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("invalid certificate")
    entropy_reference = parse_vector(raw.get("entropy_reference"), 5)
    if any(value <= ZERO for value in entropy_reference) or sum(entropy_reference, ZERO) != ONE:
        raise ValueError("entropy reference must be a positive distribution")
    partition_references = parse_vector(raw.get("partition_references"), 2)
    if any(value <= ZERO for value in partition_references):
        raise ValueError("partition references must be positive")
    iv.dps = precision
    coefficients = [ZERO] * VARIABLES
    constant = first.shift + second.shift - Fraction(2)
    for reference in partition_references:
        constant += intervals.log_interval(reference).upper
    for index, reference in enumerate(entropy_reference):
        coefficients[index] += intervals.log_interval(reference).scale(
            -Fraction(1, 3)
        ).upper

    for stage, cut, left_start, right_start, reference in (
        (0, first, 0, 5, partition_references[0]),
        (1, second, 5, 10, partition_references[1]),
    ):
        moments = moment_coefficients(cut)
        for index in range(5):
            coefficients[left_start + index] -= moments[index]
            coefficients[right_start + index] -= moments[5 + index]
        for left in range(PRIME):
            for right in range(PRIME):
                coefficients[product(stage, left, right)] += (
                    cut.kernel[left][right] / reference
                )
    return constant, tuple(coefficients)


def verify(raw: object, precision: int) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-stationary-box-v1":
        raise ValueError("unsupported stationary box certificate schema")
    if raw.get("prime") != 5 or raw.get("repetition") != 3:
        raise ValueError("unexpected code parameters")
    eta = parse_fraction(raw.get("eta"))
    delta = parse_fraction(raw.get("delta"))
    if eta != Fraction(1, 500) or delta != Fraction(11, 50):
        raise ValueError("unexpected support cutoff or target distance")
    lower, upper = parse_box(raw.get("box"))
    verify_domain_box(lower, upper, eta, delta)
    verify_composition_witness(raw.get("composition_witness"), lower, upper)
    first = parse_cut(raw.get("first_cut"), precision)
    second = parse_cut(raw.get("second_cut"), precision)
    constant, objective = objective_upper(raw, first, second, precision)
    constraints = build_constraints(lower, upper)
    by_name = {constraint.name: constraint for constraint in constraints}
    if len(by_name) != len(constraints):
        raise ValueError("constraint names are not unique")

    stored_dual = raw.get("dual")
    if not isinstance(stored_dual, dict):
        raise ValueError("missing LP dual")
    dual_left = [ZERO] * VARIABLES
    dual_value = constant
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
    for index, coefficient in enumerate(objective):
        if dual_left[index] < coefficient:
            raise ValueError(f"LP dual is infeasible at variable {index}")
    if dual_value >= ZERO:
        raise ValueError("LP dual does not prove a negative box upper")
    return {
        "status": "verified rational stationary-dual box certificate",
        "box_upper": float(dual_value),
        "box_upper_numerator": str(dual_value.numerator),
        "box_upper_denominator": str(dual_value.denominator),
        "box_upper_is_exactly_negative": True,
        "eta": fraction_text(eta),
        "delta": fraction_text(delta),
        "composition_reference_normalization_checked": True,
        "constraints_checked": len(constraints),
        "variables": VARIABLES,
        "nonzero_dual_multipliers": nonzero,
        "interval_precision": precision,
        "arithmetic": (
            "rational stationary potentials, directed exp/log intervals, "
            "McCormick products, and a rational LP dual"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    certificate = json.loads(arguments.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(certificate, arguments.precision), indent=2, sort_keys=True))
