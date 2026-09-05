#!/usr/bin/env python3
"""Floating entropy-dual LP boxes for the exact F_5 composition exponent."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

import diagnose_lagrangian_entropy_dual as entropy_dual
import explore_perron_bound as perron
import optimize_bulk_exponent as bulk


PRIME = 5
LAYOUT = bulk.Layout(PRIME)
T1, T2, SIZE = LAYOUT.size, LAYOUT.size + 1, LAYOUT.size + 2
COMPONENTS = (LAYOUT.mu, LAYOUT.beta, LAYOUT.zeta)
ENTROPY_COEFFICIENTS = (2.0 / 3.0, 2.0, 1.0)


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


def initial_box(eta: float, delta: float) -> Box:
    lower = [0.0] * 15
    upper = [1.0] * 15
    for start, support_lower, support_upper in (
        (0, eta, min(1.0, 4.0 * delta)),
        (5, eta / 2.0, min(1.0, 2.0 * delta)),
        (10, eta / 4.0, delta),
    ):
        lower[start] = 1.0 - support_upper
        upper[start] = 1.0 - support_lower
        for index in range(start + 1, start + 5):
            upper[index] = support_upper
    return Box(tuple(lower), tuple(upper))


def tighten(raw: Box) -> Box | None:
    lower, upper = list(raw.lower), list(raw.upper)
    for start in (0, 5, 10):
        for _ in range(6):
            total_lower = sum(lower[start : start + 5])
            total_upper = sum(upper[start : start + 5])
            if total_lower > 1.0 + 1e-12 or total_upper < 1.0 - 1e-12:
                return None
            changed = False
            for offset in range(5):
                index = start + offset
                new_lower = max(lower[index], 1.0 - total_upper + upper[index])
                new_upper = min(upper[index], 1.0 - total_lower + lower[index])
                if new_lower > new_upper + 1e-12:
                    return None
                changed |= new_lower > lower[index] + 1e-15
                changed |= new_upper < upper[index] - 1e-15
                lower[index], upper[index] = new_lower, new_upper
            if not changed:
                break
    return Box(tuple(lower), tuple(upper))


def x_log_x(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def secant_x_log_x(lower: float, upper: float) -> tuple[float, float]:
    if upper <= lower + 1e-15:
        slope = np.log(max(lower, 1e-300)) + 1.0
        return slope, x_log_x(lower) - slope * lower
    slope = (x_log_x(upper) - x_log_x(lower)) / (upper - lower)
    return slope, x_log_x(lower) - slope * lower


def scale_cut(cut: entropy_dual.GroupedCut, multiplier: int) -> entropy_dual.GroupedCut:
    return entropy_dual.GroupedCut(
        cut.log_partition,
        perron.scale_symbol_vector(cut.row_potential, multiplier),
        perron.scale_symbol_vector(cut.column_potential, multiplier),
        perron.scale_symbol_vector(cut.increment_potential, multiplier),
        cut.value_at_reference,
        cut.gradient_residual,
        cut.optimizer_success,
    )


def add_orbit(
    cuts: list[entropy_dual.GroupedCut],
    cut: entropy_dual.GroupedCut,
    use_orbits: bool,
) -> None:
    cuts.extend(
        scale_cut(cut, multiplier)
        for multiplier in (range(1, PRIME) if use_orbits else (1,))
    )


def cut_value(
    cut: entropy_dual.GroupedCut, increment: np.ndarray, state: np.ndarray
) -> float:
    return float(
        cut.log_partition
        + np.dot(cut.increment_potential, increment)
        + np.dot(cut.row_potential + cut.column_potential, state)
    )


def evaluate(
    raw_box: Box,
    inherited_first: tuple[entropy_dual.GroupedCut, ...],
    inherited_second: tuple[entropy_dual.GroupedCut, ...],
    outer_iterations: int,
    use_orbits: bool,
) -> tuple[object, ...] | None:
    box = tighten(raw_box)
    if box is None:
        return None
    equality, equality_target, upper_rows, upper_targets = bulk.constraints(
        LAYOUT, 0.0, 1.0
    )
    equality_extended = np.zeros((len(equality), SIZE))
    equality_extended[:, : LAYOUT.size] = equality
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for source_row, target in zip(upper_rows, upper_targets, strict=True):
        row = np.zeros(SIZE)
        row[: LAYOUT.size] = source_row
        rows.append(row)
        targets.append(float(target))
    for value in range(2, PRIME):
        row = np.zeros(SIZE)
        row[LAYOUT.mu.start + value] = 1.0
        row[LAYOUT.mu.start + 1] = -1.0
        rows.append(row)
        targets.append(0.0)

    objective = np.zeros(SIZE)
    objective[T1] = objective[T2] = 1.0
    constant = 0.0
    for component, coefficient in zip(COMPONENTS, ENTROPY_COEFFICIENTS, strict=True):
        for index in range(component.start, component.stop):
            slope, intercept = secant_x_log_x(box.lower[index], box.upper[index])
            objective[index] += coefficient * slope
            constant += coefficient * intercept

    first_cuts, second_cuts = list(inherited_first), list(inherited_second)

    def add_cut_row(
        cut: entropy_dual.GroupedCut, increment: slice, state: slice, epigraph: int
    ) -> None:
        row = np.zeros(SIZE)
        row[epigraph] = 1.0
        row[increment] -= cut.increment_potential
        row[state] -= cut.row_potential + cut.column_potential
        rows.append(row)
        targets.append(cut.log_partition)

    for cut in first_cuts:
        add_cut_row(cut, LAYOUT.mu, LAYOUT.beta, T1)
    for cut in second_cuts:
        add_cut_row(cut, LAYOUT.beta, LAYOUT.zeta, T2)

    lower_bounds, upper_bounds = np.zeros(SIZE), np.ones(SIZE)
    lower_bounds[:15], upper_bounds[:15] = box.lower, box.upper
    lower_bounds[T1:], upper_bounds[T1:] = -100.0, 100.0

    def solve() -> object:
        return linprog(
            -objective,
            A_ub=np.stack(rows),
            b_ub=np.asarray(targets),
            A_eq=equality_extended,
            b_eq=equality_target,
            bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
            method="highs",
        )

    result = solve()
    if not result.success:
        return None
    for _ in range(outer_iterations):
        point = result.x[:15]
        local_first = entropy_dual.fit_grouped_cut(
            point[LAYOUT.mu], point[LAYOUT.beta], 0.0
        )
        local_second = entropy_dual.fit_grouped_cut(
            point[LAYOUT.beta], point[LAYOUT.zeta], 0.0
        )
        additions = 0
        for local, cuts, increment, state, epigraph in (
            (local_first, first_cuts, LAYOUT.mu, LAYOUT.beta, T1),
            (local_second, second_cuts, LAYOUT.beta, LAYOUT.zeta, T2),
        ):
            if result.x[epigraph] <= cut_value(local, point[increment], point[state]) + 1e-9:
                continue
            new_cuts: list[entropy_dual.GroupedCut] = []
            add_orbit(new_cuts, local, use_orbits)
            for cut in new_cuts:
                cuts.append(cut)
                add_cut_row(cut, increment, state, epigraph)
            additions += len(new_cuts)
        if additions == 0:
            break
        result = solve()
        if not result.success:
            return None

    point = result.x[:15]
    local_first = entropy_dual.fit_grouped_cut(point[LAYOUT.mu], point[LAYOUT.beta], 0.0)
    local_second = entropy_dual.fit_grouped_cut(point[LAYOUT.beta], point[LAYOUT.zeta], 0.0)
    first_value = cut_value(local_first, point[LAYOUT.mu], point[LAYOUT.beta])
    second_value = cut_value(local_second, point[LAYOUT.beta], point[LAYOUT.zeta])
    true_value = first_value + second_value + sum(
        coefficient * sum(x_log_x(float(value)) for value in point[component])
        for component, coefficient in zip(COMPONENTS, ENTROPY_COEFFICIENTS, strict=True)
    )
    dual_gap = max(0.0, result.x[T1] - first_value) + max(0.0, result.x[T2] - second_value)
    scores: list[tuple[int, float]] = []
    for component, coefficient in zip(COMPONENTS, ENTROPY_COEFFICIENTS, strict=True):
        for index in range(component.start, component.stop):
            slope, intercept = secant_x_log_x(box.lower[index], box.upper[index])
            gap = slope * point[index] + intercept - x_log_x(point[index])
            scores.append((index, coefficient * max(0.0, gap)))
    dimension = max(scores, key=lambda item: item[1])[0]
    return (
        constant - float(result.fun), dimension, point, box, true_value, dual_gap,
        tuple(first_cuts), tuple(second_cuts),
        max(local_first.gradient_residual, local_second.gradient_residual),
    )


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper, right_lower = list(box.upper), list(box.lower)
    left_upper[dimension], right_lower[dimension] = midpoint, midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def initial_cuts(args: argparse.Namespace) -> tuple[tuple[object, ...], tuple[object, ...]]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=PRIME, repetition=3, delta=args.delta, eta=args.eta,
            restarts=args.initial_restarts, seed=args.seed, max_iterations=3000,
            tolerance=1e-11, sign_symmetric=False,
        )
    )
    message, middle, output = map(
        np.asarray, (optimized["mu"], optimized["beta"], optimized["zeta"])
    )
    first: list[entropy_dual.GroupedCut] = []
    second: list[entropy_dual.GroupedCut] = []
    add_orbit(first, entropy_dual.fit_grouped_cut(message, middle, 0.0), args.cut_orbits)
    add_orbit(second, entropy_dual.fit_grouped_cut(middle, output, 0.0), args.cut_orbits)
    return tuple(first), tuple(second)


def cover(args: argparse.Namespace) -> dict[str, object]:
    first, second = initial_cuts(args)
    initial = evaluate(
        initial_box(args.eta, args.delta), first, second,
        args.outer_iterations, args.cut_orbits,
    )
    if initial is None:
        raise RuntimeError("initial full-composition box is infeasible")
    queue: list[tuple[float, int, tuple[object, ...]]] = [(-initial[0], 0, initial)]
    identifier, history = 1, [float(initial[0])]
    for split_index in range(args.max_boxes):
        _, _, current = heapq.heappop(queue)
        if current[0] <= args.target_upper:
            return {
                "status": "target floating full-composition cover reached",
                "upper": current[0], "splits": split_index,
                "boxes_remaining": len(queue) + 1, "history": history,
            }
        for child in split(current[3], current[1]):
            bounded = evaluate(
                child, current[6], current[7],
                args.outer_iterations, args.cut_orbits,
            )
            if bounded is None:
                continue
            if bounded[0] > current[0]:
                bounded = (current[0], *bounded[1:])
            heapq.heappush(queue, (-bounded[0], identifier, bounded))
            identifier += 1
        history.append(float(-queue[0][0]))
        completed = split_index + 1
        if args.progress_every > 0 and completed % args.progress_every == 0:
            print(json.dumps({"splits": completed, "upper": -queue[0][0], "boxes": len(queue)}, sort_keys=True), flush=True)
    worst = queue[0][2]
    return {
        "status": "floating full-composition cover incomplete",
        "upper": worst[0], "splits": args.max_boxes, "boxes_remaining": len(queue),
        "history": [{"splits": i, "upper": history[i]} for i in range(0, len(history), 10)],
        "worst_box": {
            "lower": list(worst[3].lower), "upper": list(worst[3].upper),
            "lp_composition": worst[2].tolist(), "true_composition_value": worst[4],
            "entropy_dual_gap": worst[5],
            "composition_secant_gap": worst[0] - worst[5] - worst[4],
            "first_cuts": len(worst[6]), "second_cuts": len(worst[7]),
            "dual_gradient_residual": worst[8],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--max-boxes", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--initial-restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--target-upper", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cut-orbits", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = cover(arguments)
    result["parameters"] = {
        "prime": PRIME, "repetition": 3, "eta": arguments.eta,
        "delta": arguments.delta, "outer_iterations": arguments.outer_iterations,
        "cut_orbits": arguments.cut_orbits,
    }
    result["scope"] = "valid entropy-dual LP relaxation; floating potentials and LP solves are not exact"
    print(json.dumps(result, indent=2, sort_keys=True))
