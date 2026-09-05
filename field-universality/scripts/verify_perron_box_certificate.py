#!/usr/bin/env python3
"""Replay one rational box certificate for the quinary Perron bound."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import verify_perron_point_certificate as point


ZERO = Fraction(0)
ONE = Fraction(1)
VARIABLE_COUNT = 65


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    coefficients: dict[int, Fraction]
    target: Fraction


def component_start(name: str) -> int:
    return {"message": 0, "middle": 5, "output": 10}[name]


def transition_start(stage: int) -> int:
    return 15 if stage == 1 else 40


def edge(stage: int, source: int, target: int) -> int:
    return transition_start(stage) + 5 * source + target


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
            {index: -value for index, value in coefficients.items()},
            -target,
        )
    )


def parse_box(raw: object) -> dict[str, tuple[Fraction, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("missing composition box")
    result = {}
    for component in ("message", "middle"):
        lower = point.parse_vector(raw[f"{component}_lower"], 5)
        upper = point.parse_vector(raw[f"{component}_upper"], 5)
        if any(left < ZERO or left > right or right > ONE for left, right in zip(lower, upper, strict=True)):
            raise ValueError(f"invalid {component} box")
        if sum(lower, ZERO) > ONE or sum(upper, ZERO) < ONE:
            raise ValueError(f"{component} box misses the probability simplex")
        result[f"{component}_lower"] = lower
        result[f"{component}_upper"] = upper
    return result


def build_constraints(
    box: dict[str, tuple[Fraction, ...]], eta: Fraction, delta: Fraction
) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    for component in ("message", "middle", "output"):
        start = component_start(component)
        add_equality(
            constraints,
            f"normalize_{component}",
            {start + index: ONE for index in range(5)},
            ONE,
        )

    for stage, state_name, increment_name in (
        (1, "middle", "message"),
        (2, "output", "middle"),
    ):
        state_start = component_start(state_name)
        increment_start = component_start(increment_name)
        for source in range(5):
            coefficients = {edge(stage, source, target): ONE for target in range(5)}
            coefficients[state_start + source] = -ONE
            add_equality(constraints, f"stage{stage}_row_{source}", coefficients, ZERO)
        for target in range(5):
            coefficients = {edge(stage, source, target): ONE for source in range(5)}
            coefficients[state_start + target] = -ONE
            add_equality(constraints, f"stage{stage}_column_{target}", coefficients, ZERO)
        for difference in range(5):
            coefficients = {
                edge(stage, source, (source + difference) % 5): ONE
                for source in range(5)
            }
            coefficients[increment_start + difference] = -ONE
            add_equality(
                constraints,
                f"stage{stage}_increment_{difference}",
                coefficients,
                ZERO,
            )

    constraints.append(
        LinearConstraint("message_cutoff", {0: ONE}, ONE - eta)
    )
    constraints.append(
        LinearConstraint("output_weight", {10: -ONE}, -(ONE - delta))
    )
    for value in range(2, 5):
        constraints.append(
            LinearConstraint(
                f"canonical_message_{value}",
                {value: ONE, 1: -ONE},
                ZERO,
            )
        )

    for component in ("message", "middle"):
        start = component_start(component)
        lower = box[f"{component}_lower"]
        upper = box[f"{component}_upper"]
        for index in range(5):
            constraints.append(
                LinearConstraint(
                    f"box_{component}_{index}_upper",
                    {start + index: ONE},
                    upper[index],
                )
            )
            constraints.append(
                LinearConstraint(
                    f"box_{component}_{index}_lower",
                    {start + index: -ONE},
                    -lower[index],
                )
            )

    for index in range(VARIABLE_COUNT):
        constraints.append(
            LinearConstraint(f"variable_{index}_upper", {index: ONE}, ONE)
        )
    return constraints


def x_log_x_upper(value: Fraction) -> Fraction:
    if value == ZERO:
        return ZERO
    return point.log_interval(value).scale(value).upper


def entropy_secant(lower: Fraction, upper: Fraction) -> tuple[Fraction, Fraction]:
    if lower == upper:
        return ZERO, x_log_x_upper(lower)
    left = x_log_x_upper(lower)
    right = x_log_x_upper(upper)
    slope = (right - left) / (upper - lower)
    return slope, left - slope * lower


def affine_perron_upper(
    base: object,
    box: dict[str, tuple[Fraction, ...]],
    precision: int,
) -> tuple[Fraction, tuple[Fraction, ...]]:
    if not isinstance(base, dict):
        raise ValueError("invalid base certificate")
    point.iv.dps = precision
    delta = point.parse_fraction(base["delta"])
    first = base["first_cut"]
    second = base["contracted_second_cut"]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("base certificate is missing a cut")
    first_increment = point.parse_vector(first["increment_fugacity"], 5)
    first_state = point.parse_vector(first["state_fugacity"], 5)
    first_lambda = point.parse_fraction(first["lambda"])
    second_increment = point.parse_vector(second["increment_fugacity"], 5)
    q = point.parse_fraction(second["q"])
    second_lambda = point.parse_fraction(second["lambda"])

    constant_interval = point.log_interval(first_lambda) + point.log_interval(
        second_lambda
    )
    constant_interval += point.log_interval(q).scale(-delta)
    constant = constant_interval.upper
    coefficients = [ZERO] * VARIABLE_COUNT
    for index in range(5):
        coefficients[index] += point.log_interval(first_increment[index]).scale(
            -ONE
        ).upper
        coefficients[5 + index] += point.log_interval(first_state[index]).scale(
            -ONE
        ).upper
        coefficients[5 + index] += point.log_interval(
            second_increment[index]
        ).scale(-ONE).upper

    for component, entropy_coefficient in (
        ("message", Fraction(2, 3)),
        ("middle", ONE),
    ):
        start = component_start(component)
        lower = box[f"{component}_lower"]
        upper = box[f"{component}_upper"]
        for index in range(5):
            slope, intercept = entropy_secant(lower[index], upper[index])
            coefficients[start + index] += entropy_coefficient * slope
            constant += entropy_coefficient * intercept
    return constant, tuple(coefficients)


def verify(
    raw: object,
    base: object,
    precision: int,
) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-perron-box-v1":
        raise ValueError("unsupported Perron box certificate schema")
    point.verify(base, precision)
    eta = point.parse_fraction(raw["eta"])
    delta = point.parse_fraction(base["delta"])
    box = parse_box(raw["box"])
    exact_type = base.get("exact_type")
    if not isinstance(exact_type, dict):
        raise ValueError("base certificate is missing its exact type")
    for component in ("message", "middle"):
        witness = point.parse_vector(exact_type[component], 5)
        lower = box[f"{component}_lower"]
        upper = box[f"{component}_upper"]
        if any(
            value < left or value > right
            for value, left, right in zip(witness, lower, upper, strict=True)
        ):
            raise ValueError(f"box does not contain the exact {component} witness")
    constraints = build_constraints(box, eta, delta)
    by_name = {constraint.name: constraint for constraint in constraints}
    if len(by_name) != len(constraints):
        raise ValueError("constraint names are not unique")
    constant, objective = affine_perron_upper(base, box, precision)

    stored_dual = raw.get("dual")
    if not isinstance(stored_dual, dict):
        raise ValueError("missing LP dual")
    dual = {}
    for name, value in stored_dual.items():
        if name not in by_name:
            raise ValueError(f"unknown dual constraint {name}")
        multiplier = point.parse_fraction(value)
        if multiplier < ZERO:
            raise ValueError("LP dual multipliers must be nonnegative")
        dual[name] = multiplier

    dual_left = [ZERO] * VARIABLE_COUNT
    dual_value = constant
    for name, multiplier in dual.items():
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
        "status": "verified rational Perron box certificate",
        "box_upper": float(dual_value),
        "box_upper_numerator": str(dual_value.numerator),
        "box_upper_denominator": str(dual_value.denominator),
        "box_upper_is_exactly_negative": True,
        "box_contains_exact_feasible_type": True,
        "nonzero_dual_multipliers": len(dual),
        "constraints_checked": len(constraints),
        "variables": VARIABLE_COUNT,
        "interval_precision": precision,
        "arithmetic": "rational entropy secants, rational LP dual, and directed logarithm intervals",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        type=Path,
        default=Path("results/f5-perron-box-certificate.json"),
    )
    parser.add_argument("--precision", type=int, default=100)
    args = parser.parse_args()
    if args.precision < 50:
        parser.error("precision must be at least 50 decimal digits")
    return args


def main() -> None:
    args = parse_args()
    raw = json.loads(args.certificate.read_text(encoding="utf-8"))
    base_name = raw.get("base_certificate")
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise ValueError("base certificate must be a sibling filename")
    base = json.loads(
        args.certificate.with_name(base_name).read_text(encoding="utf-8")
    )
    print(json.dumps(verify(raw, base, args.precision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
