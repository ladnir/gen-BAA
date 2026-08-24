#!/usr/bin/env python3
"""Design a cross-entropy box cover for the full-symbol bulk exponent."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import optimize_bulk_exponent as bulk


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class Cut:
    first: np.ndarray
    second: np.ndarray


@dataclass(frozen=True)
class Bound:
    value: float
    box: Box
    cuts: tuple[Cut, ...]
    split_dimension: int


def xlogx(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def secant(lower: float, upper: float) -> tuple[float, float]:
    if upper - lower <= 1e-15:
        return 0.0, xlogx(lower)
    slope = (xlogx(upper) - xlogx(lower)) / (upper - lower)
    return slope, xlogx(lower) - slope * lower


def tighten_simplex(
    lower: list[float], upper: list[float], start: int, dimension: int
) -> bool:
    for _ in range(dimension + 1):
        total_lower = sum(lower[start : start + dimension])
        total_upper = sum(upper[start : start + dimension])
        if total_lower > 1.0 + 1e-13 or total_upper < 1.0 - 1e-13:
            return False
        changed = False
        for offset in range(dimension):
            index = start + offset
            new_lower = max(lower[index], 1.0 - (total_upper - upper[index]))
            new_upper = min(upper[index], 1.0 - (total_lower - lower[index]))
            if new_lower > new_upper + 1e-13:
                return False
            new_lower = max(0.0, new_lower)
            new_upper = min(1.0, new_upper)
            changed |= abs(new_lower - lower[index]) > 1e-15
            changed |= abs(new_upper - upper[index]) > 1e-15
            lower[index] = new_lower
            upper[index] = new_upper
        if not changed:
            break
    return True


def tighten(raw_box: Box, prime: int) -> Box | None:
    lower = list(raw_box.lower)
    upper = list(raw_box.upper)
    for _ in range(8):
        before = (tuple(lower), tuple(upper))
        for start in (0, prime, 2 * prime):
            if not tighten_simplex(lower, upper, start, prime):
                return None
        for increment_zero, state_zero in ((0, prime), (prime, 2 * prime)):
            upper[state_zero] = min(
                upper[state_zero], (1.0 + upper[increment_zero]) / 2.0
            )
            lower[increment_zero] = max(
                lower[increment_zero], 2.0 * lower[state_zero] - 1.0
            )
            if lower[increment_zero] > upper[increment_zero] + 1e-13:
                return None
            if lower[state_zero] > upper[state_zero] + 1e-13:
                return None
        if before == (tuple(lower), tuple(upper)):
            break
    return Box(tuple(lower), tuple(upper))


def variable_bounds(box: Box, layout: bulk.Layout) -> list[tuple[float, float]]:
    bounds = [
        (box.lower[index], box.upper[index])
        for index in range(3 * layout.prime)
    ]
    bounds.extend((0.0, 1.0) for _ in range(layout.size - len(bounds)))
    return bounds


def feasible_reference(
    box: Box, layout: bulk.Layout, equality: np.ndarray, target: np.ndarray
) -> np.ndarray | None:
    result = linprog(
        np.zeros(layout.size),
        A_eq=equality,
        b_eq=target,
        bounds=variable_bounds(box, layout),
        method="highs",
    )
    return result.x if result.success else None


def cut_from_point(point: np.ndarray, layout: bulk.Layout) -> Cut:
    stages = []
    for transition in (layout.tau1, layout.tau2):
        distribution = np.maximum(point[transition], 0.0)
        distribution /= distribution.sum()
        distribution = (
            (1.0 - 1e-12) * distribution + 1e-12 / distribution.size
        )
        stages.append(-np.log(distribution))
    return Cut(stages[0], stages[1])


def evaluate(
    box: Box,
    cuts: tuple[Cut, ...],
    layout: bulk.Layout,
    equality: np.ndarray,
    target: np.ndarray,
    repetition: int,
) -> tuple[float, int, np.ndarray] | None:
    extended_size = layout.size + 2
    linear = np.zeros(extended_size)
    linear[-2:] = 1.0
    constant = 0.0
    entropy_coefficients = (
        [1.0 - 1.0 / repetition] * layout.prime
        + [2.0] * layout.prime
        + [1.0] * layout.prime
    )
    for index, coefficient in enumerate(entropy_coefficients):
        slope, intercept = secant(box.lower[index], box.upper[index])
        linear[index] += coefficient * slope
        constant += coefficient * intercept

    upper_rows = []
    for cut in cuts:
        first = np.zeros(extended_size)
        first[-2] = 1.0
        first[layout.tau1] = -cut.first
        upper_rows.append(first)
        second = np.zeros(extended_size)
        second[-1] = 1.0
        second[layout.tau2] = -cut.second
        upper_rows.append(second)
    result = linprog(
        -linear,
        A_ub=np.stack(upper_rows),
        b_ub=np.zeros(len(upper_rows)),
        A_eq=np.pad(equality, ((0, 0), (0, 2))),
        b_eq=target,
        bounds=variable_bounds(box, layout) + [(None, None), (None, None)],
        method="highs",
    )
    if not result.success:
        return None
    composition = result.x[: 3 * layout.prime]
    gaps = []
    for index, coefficient in enumerate(entropy_coefficients):
        slope, intercept = secant(box.lower[index], box.upper[index])
        gaps.append(
            max(
                0.0,
                coefficient
                * (
                    slope * composition[index]
                    + intercept
                    - xlogx(composition[index])
                ),
            )
        )
    return constant - float(result.fun), int(np.argmax(gaps)), result.x[: layout.size]


def box_bound(
    raw_box: Box,
    inherited: tuple[Cut, ...],
    layout: bulk.Layout,
    equality: np.ndarray,
    target: np.ndarray,
    repetition: int,
    maximum_cut_pairs: int,
) -> Bound | None:
    box = tighten(raw_box, layout.prime)
    if box is None:
        return None
    reference = feasible_reference(box, layout, equality, target)
    if reference is None:
        return None
    cuts = (inherited + (cut_from_point(reference, layout),))[
        -maximum_cut_pairs:
    ]
    first = evaluate(box, cuts, layout, equality, target, repetition)
    if first is None:
        return None
    _, _, witness = first
    cuts = (cuts + (cut_from_point(witness, layout),))[-maximum_cut_pairs:]
    second = evaluate(box, cuts, layout, equality, target, repetition)
    if second is None:
        return None
    value, dimension, _ = second
    return Bound(value, box, cuts, dimension)


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(args.prime)
    equality, target, _, _ = bulk.constraints(layout, args.eta, args.delta)
    lower = [0.0] * (3 * args.prime)
    upper = [1.0] * (3 * args.prime)
    upper[layout.mu.start] = 1.0 - args.eta
    lower[layout.zeta.start] = 1.0 - args.delta
    initial = box_bound(
        Box(tuple(lower), tuple(upper)),
        (),
        layout,
        equality,
        target,
        args.repetition,
        args.maximum_cut_pairs,
    )
    if initial is None:
        raise RuntimeError("initial full-symbol domain is infeasible")
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < 0.0:
            return {
                "status": "negative floating-point full-symbol cover found",
                "upper_bound": current.value,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
            }
        dimension = current.split_dimension
        if current.box.upper[dimension] - current.box.lower[dimension] <= 1e-14:
            widths = np.array(current.box.upper) - np.array(current.box.lower)
            dimension = int(np.argmax(widths))
        for child in split(current.box, dimension):
            bounded = box_bound(
                child,
                current.cuts,
                layout,
                equality,
                target,
                args.repetition,
                args.maximum_cut_pairs,
            )
            if bounded is not None:
                heapq.heappush(queue, (-bounded.value, identifier, bounded))
                identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    return {
        "status": "full-symbol cover search incomplete",
        "current_upper_bound": -queue[0][0] if queue else None,
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument("--maximum-cut-pairs", type=int, default=32)
    parser.add_argument("--max-boxes", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = search(args)
    result.update(
        {
            "prime": args.prime,
            "repetition": args.repetition,
            "delta": args.delta,
            "eta": args.eta,
            "maximum_cut_pairs": args.maximum_cut_pairs,
            "arithmetic": "floating-point certificate design",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
