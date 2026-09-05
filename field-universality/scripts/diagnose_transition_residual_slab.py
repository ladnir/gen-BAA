#!/usr/bin/env python3
"""Floating full-transition LP bound on one signed residual slab.

The LP keeps both stationary 5-by-5 transition tables.  Entropy tangents give
global affine upper bounds for their two positive entropy terms, composition
boxes give secant upper bounds for the negative composition entropies, and a
McCormick envelope imposes the selected reference-residual slab.  The result
is mathematically sound apart from floating-point roundoff, but it is a
diagnostic rather than an exact certificate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import optimize_bulk_exponent as bulk


PRIME = 5
LAYOUT = bulk.Layout(PRIME)
ENTROPY_START = LAYOUT.size
ENTROPY_CELLS = 2 * PRIME * PRIME
PRODUCT1 = ENTROPY_START + ENTROPY_CELLS
PRODUCT2 = PRODUCT1 + 1
SIZE = PRODUCT2 + 1


@dataclass(frozen=True)
class ResidualSlab:
    stage: int
    coordinate: int
    sign: int
    threshold: float


def parse_slab(text: str, threshold: float) -> ResidualSlab:
    try:
        stage_text, coordinate_text, sign_text = text.split(":")
        stage = int(stage_text) - 1
        coordinate = int(coordinate_text)
        sign = {"+": 1, "-": -1}[sign_text]
    except (ValueError, KeyError) as error:
        raise argparse.ArgumentTypeError(
            "slab must have the form STAGE:COORDINATE:SIGN"
        ) from error
    if stage not in (0, 1) or coordinate not in range(PRIME):
        raise argparse.ArgumentTypeError(
            "stage must be 1 or 2 and coordinate must be in 0,...,4"
        )
    return ResidualSlab(stage, coordinate, sign, threshold)


def x_log_x(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def x_log_x_secant(lower: float, upper: float) -> tuple[float, float]:
    if upper <= lower + 1e-15:
        slope = np.log(max(lower, 1e-300)) + 1.0
        return slope, x_log_x(lower) - slope * lower
    slope = (x_log_x(upper) - x_log_x(lower)) / (upper - lower)
    return slope, x_log_x(lower) - slope * lower


def support_form(start: int) -> np.ndarray:
    form = np.zeros(SIZE)
    form[start + 1 : start + PRIME] = 1.0
    return form


def crossing_form(stage: int) -> np.ndarray:
    form = np.zeros(SIZE)
    layer = stage + 1
    for target in range(1, PRIME):
        form[LAYOUT.edge(layer, 0, target)] = 1.0
    return form


def linear_bounds(
    coefficients: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    low = float(
        np.dot(np.maximum(coefficients, 0.0), lower)
        + np.dot(np.minimum(coefficients, 0.0), upper)
    )
    high = float(
        np.dot(np.maximum(coefficients, 0.0), upper)
        + np.dot(np.minimum(coefficients, 0.0), lower)
    )
    return low, high


def add_mccormick(
    rows: list[np.ndarray],
    targets: list[float],
    product: int,
    first: np.ndarray,
    second: np.ndarray,
    first_bounds: tuple[float, float],
    second_bounds: tuple[float, float],
) -> None:
    first_lower, first_upper = first_bounds
    second_lower, second_upper = second_bounds
    for first_scale, second_scale, product_scale, target in (
        (second_lower, first_lower, -1.0, first_lower * second_lower),
        (second_upper, first_upper, -1.0, first_upper * second_upper),
        (-second_lower, -first_upper, 1.0, -first_upper * second_lower),
        (-second_upper, -first_lower, 1.0, -first_lower * second_upper),
    ):
        row = first_scale * first + second_scale * second
        row[product] += product_scale
        rows.append(row)
        targets.append(target)


def composition_support_bounds(
    lower: np.ndarray, upper: np.ndarray, start: int
) -> tuple[float, float]:
    return (
        max(0.0, 1.0 - upper[start], float(np.sum(lower[start + 1 : start + 5]))),
        min(1.0, 1.0 - lower[start], float(np.sum(upper[start + 1 : start + 5]))),
    )


def add_residual_slab(
    rows: list[np.ndarray],
    targets: list[float],
    slab: ResidualSlab,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    increment_start = 0 if slab.stage == 0 else 5
    state_start = increment_start + 5
    increment_support = support_form(increment_start)
    state_support = support_form(state_start)
    crossing = crossing_form(slab.stage)
    rho = slab.threshold

    if slab.coordinate == 0:
        # residual = x - A/5.
        row = slab.sign * (increment_support / 5.0 - crossing)
        rows.append(row)
        targets.append(-rho)
        return

    both_nonzero = increment_support - crossing
    state_coordinate = np.zeros(SIZE)
    state_coordinate[state_start + slab.coordinate] = 1.0
    a_bounds = composition_support_bounds(lower, upper, increment_start)
    b_bounds = composition_support_bounds(lower, upper, state_start)
    crossing_bounds = linear_bounds(crossing, lower, upper)
    both_bounds = (
        max(a_bounds[0] / 2.0, a_bounds[0] - crossing_bounds[1], 0.0),
        max(0.0, a_bounds[1] - crossing_bounds[0]),
    )
    coordinate_bounds = (
        lower[state_start + slab.coordinate],
        upper[state_start + slab.coordinate],
    )
    add_mccormick(
        rows,
        targets,
        PRODUCT1,
        both_nonzero,
        state_coordinate,
        both_bounds,
        coordinate_bounds,
    )
    add_mccormick(
        rows,
        targets,
        PRODUCT2,
        increment_support,
        state_support,
        a_bounds,
        b_bounds,
    )
    # B * residual = (A-x) beta_a - AB/5.
    row = np.zeros(SIZE)
    row[PRODUCT1] = -slab.sign
    row[PRODUCT2] = slab.sign / 5.0
    row += slab.sign * rho * state_support
    rows.append(row)
    targets.append(0.0)


def entropy_variable(stage: int, source: int, target: int) -> int:
    return ENTROPY_START + stage * PRIME * PRIME + source * PRIME + target


def selected_frontier_entry(path: Path, index: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        frontier = json.load(source)
    return sorted(frontier["queue"], key=lambda entry: entry["priority"])[index][
        "evaluated"
    ]


def solve(args: argparse.Namespace) -> dict[str, object]:
    entry = selected_frontier_entry(args.frontier, args.frontier_index)
    slab = parse_slab(args.slab, args.threshold)
    equality, equality_target, bulk_upper, bulk_targets = bulk.constraints(
        LAYOUT, args.eta, args.delta
    )
    extended_equality = np.zeros((len(equality), SIZE))
    extended_equality[:, : LAYOUT.size] = equality
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for source, target in zip(bulk_upper, bulk_targets, strict=True):
        row = np.zeros(SIZE)
        row[: LAYOUT.size] = source
        rows.append(row)
        targets.append(float(target))

    lower = np.zeros(SIZE)
    upper = np.ones(SIZE)
    lower[:15] = np.asarray(entry["lower"], dtype=float)
    upper[:15] = np.asarray(entry["upper"], dtype=float)
    lower[ENTROPY_START : ENTROPY_START + ENTROPY_CELLS] = 0.0
    upper[ENTROPY_START : ENTROPY_START + ENTROPY_CELLS] = 1.0 / np.e

    # The projected cover has already canonicalized the largest nonzero
    # message symbol to one.
    for value in range(2, PRIME):
        row = np.zeros(SIZE)
        row[value] = 1.0
        row[1] = -1.0
        rows.append(row)
        targets.append(0.0)

    add_residual_slab(rows, targets, slab, lower, upper)

    objective = np.zeros(SIZE)
    objective[ENTROPY_START : ENTROPY_START + ENTROPY_CELLS] = 1.0
    constant = 0.0
    for start, coefficient in ((0, 2.0 / 3.0), (5, 2.0), (10, 1.0)):
        for coordinate in range(start, start + PRIME):
            slope, intercept = x_log_x_secant(lower[coordinate], upper[coordinate])
            objective[coordinate] += coefficient * slope
            constant += coefficient * intercept

    entropy_rows: list[np.ndarray] = []
    entropy_targets: list[float] = []
    initial_grid = (args.entropy_minimum, 1.0 / (PRIME * PRIME), 1.0)
    for stage, layer in enumerate((1, 2)):
        for source in range(PRIME):
            for target in range(PRIME):
                edge = LAYOUT.edge(layer, source, target)
                epigraph = entropy_variable(stage, source, target)
                for reference in initial_grid:
                    # -x log x <= q + (-log q - 1)x.
                    row = np.zeros(SIZE)
                    row[epigraph] = 1.0
                    row[edge] = np.log(reference) + 1.0
                    entropy_rows.append(row)
                    entropy_targets.append(reference)

    def run_lp() -> object:
        return linprog(
            -objective,
            A_ub=np.stack((*rows, *entropy_rows)),
            b_ub=np.asarray((*targets, *entropy_targets)),
            A_eq=extended_equality,
            b_eq=equality_target,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )

    result = run_lp()
    if not result.success:
        return {
            "status": "full-transition slab relaxation is infeasible",
            "message": result.message,
        }
    history = []
    for iteration in range(args.entropy_iterations + 1):
        point = result.x
        transition_entropy = bulk.entropy(point[LAYOUT.tau1]) + bulk.entropy(
            point[LAYOUT.tau2]
        )
        epigraph_entropy = float(
            np.sum(point[ENTROPY_START : ENTROPY_START + ENTROPY_CELLS])
        )
        entropy_gap = epigraph_entropy - transition_entropy
        upper_bound = constant - float(result.fun)
        true_value = bulk.exponent(point[: LAYOUT.size], LAYOUT, 3)
        history.append(
            {
                "iteration": iteration,
                "upper": upper_bound,
                "entropy_gap": entropy_gap,
                "value_at_lp_transition": true_value,
            }
        )
        if entropy_gap <= args.entropy_tolerance or iteration == args.entropy_iterations:
            break
        for stage, layer in enumerate((1, 2)):
            for source in range(PRIME):
                for target in range(PRIME):
                    edge = LAYOUT.edge(layer, source, target)
                    epigraph = entropy_variable(stage, source, target)
                    reference = max(args.entropy_minimum, float(point[edge]))
                    row = np.zeros(SIZE)
                    row[epigraph] = 1.0
                    row[edge] = np.log(reference) + 1.0
                    entropy_rows.append(row)
                    entropy_targets.append(reference)
        result = run_lp()
        if not result.success:
            raise RuntimeError("coordinatewise entropy LP became infeasible")
    secant_scores = []
    for start, coefficient in ((0, 2.0 / 3.0), (5, 2.0), (10, 1.0)):
        for coordinate in range(start, start + PRIME):
            slope, intercept = x_log_x_secant(lower[coordinate], upper[coordinate])
            gap = slope * point[coordinate] + intercept - x_log_x(point[coordinate])
            secant_scores.append((coordinate, coefficient * max(0.0, gap)))
    split_dimension, maximum_secant_gap = max(
        secant_scores, key=lambda item: item[1]
    )
    return {
        "status": "floating full-transition residual-slab diagnostic",
        "frontier": str(args.frontier),
        "frontier_index": args.frontier_index,
        "projected_frontier_upper": float(entry["bound"]),
        "slab": {
            "stage": slab.stage + 1,
            "coordinate": slab.coordinate,
            "sign": "+" if slab.sign > 0 else "-",
            "threshold": slab.threshold,
        },
        "upper": upper_bound,
        "combined_upper": min(float(entry["bound"]), upper_bound),
        "entropy_gap": entropy_gap,
        "value_at_lp_transition": true_value,
        "split_dimension": split_dimension,
        "maximum_composition_secant_gap": maximum_secant_gap,
        "composition_secant_gap": upper_bound - entropy_gap - true_value,
        "entropy_inequalities": len(entropy_rows),
        "entropy_iterations": len(history) - 1,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=166)
    parser.add_argument("--slab", default="1:0:+")
    parser.add_argument("--threshold", type=float, default=1.0 / 3200.0)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--entropy-minimum", type=float, default=1e-12)
    parser.add_argument("--entropy-iterations", type=int, default=32)
    parser.add_argument("--entropy-tolerance", type=float, default=1e-8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(solve(parse_args()), indent=2, sort_keys=True))
