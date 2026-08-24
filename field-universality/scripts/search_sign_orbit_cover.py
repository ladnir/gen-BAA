#!/usr/bin/env python3
"""Design a box certificate for the sign-orbit bulk relaxation."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import optimize_sign_orbit_bulk as orbit


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
            changed |= new_lower > lower[index] + 1e-15
            changed |= new_upper < upper[index] - 1e-15
            lower[index] = max(0.0, new_lower)
            upper[index] = min(1.0, new_upper)
        if not changed:
            break
    return True


def tighten(box: Box, orbit_count: int) -> Box | None:
    lower = list(box.lower)
    upper = list(box.upper)
    for start in (0, orbit_count, 2 * orbit_count):
        if not tighten_simplex(lower, upper, start, orbit_count):
            return None
    return Box(tuple(lower), tuple(upper))


def variable_bounds(box: Box, layout: orbit.Layout) -> list[tuple[float, float]]:
    bounds = [
        (box.lower[index], box.upper[index])
        for index in range(3 * layout.orbit_count)
    ]
    bounds.extend((0.0, 1.0) for _ in range(layout.size - len(bounds)))
    return bounds


def feasible_reference(
    box: Box, layout: orbit.Layout, equality: np.ndarray, target: np.ndarray
) -> np.ndarray | None:
    size = layout.size
    objective = np.zeros(size + 1)
    objective[-1] = -1.0
    equality_with_slack = np.pad(equality, ((0, 0), (0, 1)))
    gamma_indices = list(range(layout.gamma1.start, layout.gamma2.stop))
    upper = np.zeros((len(gamma_indices), size + 1))
    for row, index in enumerate(gamma_indices):
        upper[row, index] = -1.0
        upper[row, -1] = 1.0
    bounds = variable_bounds(box, layout) + [(0.0, 1.0)]
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(len(gamma_indices)),
        A_eq=equality_with_slack,
        b_eq=target,
        bounds=bounds,
        method="highs",
    )
    return result.x[:size] if result.success else None


def cut_from_point(point: np.ndarray, layout: orbit.Layout) -> Cut:
    coefficients = []
    for gamma in (layout.gamma1, layout.gamma2):
        pair = orbit.pair_distribution(point, layout, gamma)
        pair = np.maximum(pair, 0.0)
        pair /= pair.sum()
        pair = (1.0 - 1e-12) * pair + 1e-12 / pair.size
        coefficients.append(
            np.array(
                [-np.log(pair[increment, source]) for increment, source, _ in layout.triples]
            )
        )
    return Cut(coefficients[0], coefficients[1])


def evaluate(
    box: Box,
    cuts: tuple[Cut, ...],
    layout: orbit.Layout,
    equality: np.ndarray,
    target: np.ndarray,
    repetition: int,
) -> tuple[float, int, np.ndarray] | None:
    extended_size = layout.size + 2
    linear = np.zeros(extended_size)
    linear[-2:] = 1.0
    internal_entropy = np.log(2.0) / repetition
    linear[layout.mu.start + 1 : layout.mu.stop] = internal_entropy
    constant = 0.0
    entropy_coefficients = (
        [1.0 - 1.0 / repetition] * layout.orbit_count
        + [2.0] * layout.orbit_count
        + [1.0] * layout.orbit_count
    )
    for index, coefficient in enumerate(entropy_coefficients):
        slope, intercept = secant(box.lower[index], box.upper[index])
        linear[index] += coefficient * slope
        constant += coefficient * intercept

    upper_rows = []
    for cut in cuts:
        first = np.zeros(extended_size)
        first[-2] = 1.0
        first[layout.gamma1] = -cut.first
        upper_rows.append(first)
        second = np.zeros(extended_size)
        second[-1] = 1.0
        second[layout.gamma2] = -cut.second
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
    composition = result.x[: 3 * layout.orbit_count]
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
    layout: orbit.Layout,
    equality: np.ndarray,
    target: np.ndarray,
    repetition: int,
    maximum_cut_pairs: int,
) -> Bound | None:
    box = tighten(raw_box, layout.orbit_count)
    if box is None:
        return None
    reference = feasible_reference(box, layout, equality, target)
    if reference is None:
        return None
    cuts = inherited + (cut_from_point(reference, layout),)
    if maximum_cut_pairs:
        cuts = cuts[-maximum_cut_pairs:]
    evaluated = evaluate(box, cuts, layout, equality, target, repetition)
    if evaluated is None:
        return None
    value, dimension, witness = evaluated
    cuts += (cut_from_point(witness, layout),)
    if maximum_cut_pairs:
        cuts = cuts[-maximum_cut_pairs:]
    tightened = evaluate(box, cuts, layout, equality, target, repetition)
    if tightened is not None:
        value, dimension, _ = tightened
    return Bound(value, box, cuts, dimension)


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = orbit.Layout(args.prime)
    equality, target, _, _ = orbit.constraints(layout, args.eta, args.delta)
    lower = [0.0] * (3 * layout.orbit_count)
    upper = [1.0] * (3 * layout.orbit_count)
    upper[0] = 1.0 - args.eta
    lower[2 * layout.orbit_count] = 1.0 - args.delta
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
        raise RuntimeError("initial orbit domain is infeasible")
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_cuts = len(initial.cuts)
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < 0.0:
            return {
                "status": "negative floating-point orbit cover found",
                "upper_bound": current.value,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_cuts_per_box": maximum_cuts,
            }
        dimension = current.split_dimension
        if current.box.upper[dimension] - current.box.lower[dimension] <= 1e-14:
            dimension = int(
                np.argmax(np.array(current.box.upper) - np.array(current.box.lower))
            )
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
                maximum_cuts = max(maximum_cuts, len(bounded.cuts))
                heapq.heappush(queue, (-bounded.value, identifier, bounded))
                identifier += 1
        splits += 1
    return {
        "status": "orbit cover search incomplete",
        "current_upper_bound": -queue[0][0] if queue else -np.inf,
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_cuts_per_box": maximum_cuts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.0001)
    parser.add_argument("--max-boxes", type=int, default=10_000)
    parser.add_argument(
        "--maximum-cut-pairs",
        type=int,
        default=0,
        help="retain at most this many recent cut pairs; zero keeps all cuts",
    )
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
            "arithmetic": "floating-point certificate design",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
