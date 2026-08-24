#!/usr/bin/env python3
"""Search for a box cover proving a negative general bulk exponent.

This program uses floating-point linear and convex optimization to design a
certificate.  Its output is diagnostic until a separate interval checker
verifies every leaf bound and the cover tree.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import optimize_bulk_exponent as bulk
from diagnose_entropy_dual import optimize_dual


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class Bound:
    value: float
    box: Box
    minimum_transition_entry: float
    dual_residual: float
    duals: tuple["DualPair", ...]
    split_dimension: int


@dataclass(frozen=True)
class DualPair:
    first_constant: float
    second_constant: float
    first_increment: np.ndarray
    first_state: np.ndarray
    second_increment: np.ndarray
    second_state: np.ndarray
    residual: float
    reference_compositions: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    value: float
    split_dimension: int
    witness: np.ndarray


def tighten_simplex(
    lower: list[float], upper: list[float], start: int, prime: int
) -> bool:
    for _ in range(prime + 1):
        changed = False
        total_lower = sum(lower[start : start + prime])
        total_upper = sum(upper[start : start + prime])
        if total_lower > 1.0 + 1e-13 or total_upper < 1.0 - 1e-13:
            return False
        for symbol in range(prime):
            index = start + symbol
            new_lower = max(lower[index], 1.0 - (total_upper - upper[index]))
            new_upper = min(upper[index], 1.0 - (total_lower - lower[index]))
            if new_lower > new_upper + 1e-13:
                return False
            new_lower = max(0.0, new_lower)
            new_upper = min(1.0, new_upper)
            changed |= new_lower > lower[index] + 1e-15
            changed |= new_upper < upper[index] - 1e-15
            lower[index] = new_lower
            upper[index] = new_upper
        if not changed:
            break
    return True


def tighten(box: Box, prime: int) -> Box | None:
    lower = list(box.lower)
    upper = list(box.upper)
    for start in (0, prime, 2 * prime):
        if not tighten_simplex(lower, upper, start, prime):
            return None
    return Box(tuple(lower), tuple(upper))


def feasible_reference(
    box: Box, layout: bulk.Layout, equality: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float] | None:
    size = layout.size
    objective = np.zeros(size + 1)
    objective[-1] = -1.0
    equality_with_slack = np.pad(equality, ((0, 0), (0, 1)))
    transition_indices = list(range(layout.tau1.start, layout.tau2.stop))
    upper = np.zeros((len(transition_indices), size + 1))
    for row, index in enumerate(transition_indices):
        upper[row, index] = -1.0
        upper[row, -1] = 1.0
    bounds = [
        (box.lower[index], box.upper[index]) for index in range(3 * layout.prime)
    ]
    bounds.extend((0.0, 1.0) for _ in range(size - 3 * layout.prime))
    bounds.append((0.0, 1.0))
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(len(transition_indices)),
        A_eq=equality_with_slack,
        b_eq=target,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    return result.x[:size], result.x[-1]


def dual_at_reference(reference: np.ndarray, layout: bulk.Layout) -> DualPair:
    mu = reference[layout.mu]
    beta = reference[layout.beta]
    zeta = reference[layout.zeta]
    first_value, u1, v1, w1, first_residual = optimize_dual(mu, beta)
    second_value, u2, v2, w2, second_residual = optimize_dual(beta, zeta)
    first_constant = (
        first_value - float(np.dot(w1, mu)) - float(np.dot(u1 + v1, beta))
    )
    second_constant = (
        second_value - float(np.dot(w2, beta)) - float(np.dot(u2 + v2, zeta))
    )
    return DualPair(
        first_constant,
        second_constant,
        w1,
        u1 + v1,
        w2,
        u2 + v2,
        max(first_residual, second_residual),
        np.concatenate((mu, beta, zeta)),
    )


def xlogx(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def secant(lower: float, upper: float) -> tuple[float, float]:
    if upper - lower <= 1e-15:
        return 0.0, xlogx(lower)
    slope = (xlogx(upper) - xlogx(lower)) / (upper - lower)
    return slope, xlogx(lower) - slope * lower


def binary_xlogx(value: float) -> float:
    return xlogx(value) + xlogx(1.0 - value)


def binary_secant(lower: float, upper: float) -> tuple[float, float]:
    if upper - lower <= 1e-15:
        return 0.0, binary_xlogx(lower)
    slope = (binary_xlogx(upper) - binary_xlogx(lower)) / (upper - lower)
    return slope, binary_xlogx(lower) - slope * lower


def evaluate_duals(
    box: Box,
    duals: tuple[DualPair, ...],
    layout: bulk.Layout,
    repetition: int,
    equality: np.ndarray,
    equality_target: np.ndarray,
    entropy_relaxation: str,
) -> Evaluation:
    extended_size = layout.size + 2
    linear = np.zeros(extended_size)
    linear[-2:] = 1.0
    constant = 0.0
    entropy_coefficients = np.array(
        [1.0 - 1.0 / repetition] * layout.prime
        + [2.0] * layout.prime
        + [1.0] * layout.prime
    )
    if entropy_relaxation == "full":
        for index, coefficient in enumerate(entropy_coefficients):
            slope, intercept = secant(box.lower[index], box.upper[index])
            linear[index] = coefficient * slope
            constant += coefficient * intercept
    else:
        for index, coefficient in (
            (0, 1.0 - 1.0 / repetition),
            (layout.prime, 2.0),
            (2 * layout.prime, 1.0),
        ):
            slope, intercept = binary_secant(box.lower[index], box.upper[index])
            linear[index] = coefficient * slope
            constant += coefficient * intercept
    upper_rows = []
    upper_targets = []
    for dual in duals:
        first = np.zeros(extended_size)
        first[-2] = 1.0
        first[layout.mu] = -dual.first_increment
        first[layout.beta] = -dual.first_state
        upper_rows.append(first)
        upper_targets.append(dual.first_constant)

        second = np.zeros(extended_size)
        second[-1] = 1.0
        second[layout.beta] = -dual.second_increment
        second[layout.zeta] = -dual.second_state
        upper_rows.append(second)
        upper_targets.append(dual.second_constant)
    bounds = [
        (box.lower[index], box.upper[index]) for index in range(3 * layout.prime)
    ]
    bounds.extend((0.0, 1.0) for _ in range(layout.size - 3 * layout.prime))
    bounds.extend(((None, None), (None, None)))
    extended_equality = np.pad(equality, ((0, 0), (0, 2)))
    result = linprog(
        -linear,
        A_ub=np.stack(upper_rows),
        b_ub=np.array(upper_targets),
        A_eq=extended_equality,
        b_eq=equality_target,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return Evaluation(-np.inf, 0, np.zeros(3 * layout.prime))
    composition = result.x[: 3 * layout.prime]
    gaps = np.zeros(3 * layout.prime)
    if entropy_relaxation == "full":
        for index, coefficient in enumerate(entropy_coefficients):
            slope, intercept = secant(box.lower[index], box.upper[index])
            gap = coefficient * (
                slope * composition[index]
                + intercept
                - xlogx(composition[index])
            )
            gaps[index] = max(0.0, gap)
    else:
        for index, coefficient in (
            (0, 1.0 - 1.0 / repetition),
            (layout.prime, 2.0),
            (2 * layout.prime, 1.0),
        ):
            slope, intercept = binary_secant(box.lower[index], box.upper[index])
            gap = coefficient * (
                slope * composition[index]
                + intercept
                - binary_xlogx(composition[index])
            )
            gaps[index] = max(0.0, gap)
    if max(gaps, default=0.0) > 1e-12:
        split_dimension = int(np.argmax(gaps))
    else:
        references = np.stack([dual.reference_compositions for dual in duals])
        distances = np.min(np.abs(references - composition), axis=0)
        if entropy_relaxation == "scalar":
            allowed = np.zeros(3 * layout.prime)
            allowed[[0, layout.prime, 2 * layout.prime]] = 1.0
            distances *= allowed
        widths = np.array(box.upper) - np.array(box.lower)
        split_dimension = int(np.argmax(distances * (widths > 1e-14)))
    return Evaluation(constant - float(result.fun), split_dimension, composition)


def box_bound(
    raw_box: Box,
    layout: bulk.Layout,
    equality: np.ndarray,
    equality_target: np.ndarray,
    repetition: int,
    entropy_relaxation: str,
    inherited: tuple[DualPair, ...] = (),
) -> Bound | None:
    box = tighten(raw_box, layout.prime)
    if box is None:
        return None
    feasible = feasible_reference(box, layout, equality, equality_target)
    if feasible is None:
        return None
    reference, minimum_entry = feasible
    new_dual = dual_at_reference(reference, layout)
    duals = inherited + (new_dual,)
    evaluation = evaluate_duals(
        box,
        duals,
        layout,
        repetition,
        equality,
        equality_target,
        entropy_relaxation,
    )
    for _ in range(1):
        witness_reference = np.zeros(layout.size)
        witness_reference[: 3 * layout.prime] = evaluation.witness
        witness_dual = dual_at_reference(witness_reference, layout)
        duals += (witness_dual,)
        tightened = evaluate_duals(
            box,
            duals,
            layout,
            repetition,
            equality,
            equality_target,
            entropy_relaxation,
        )
        if evaluation.value - tightened.value < 1e-9:
            evaluation = tightened
            break
        evaluation = tightened
    return Bound(
        evaluation.value,
        box,
        minimum_entry,
        max(dual.residual for dual in duals),
        duals,
        evaluation.split_dimension,
    )


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def choose_dimension(
    box: Box,
) -> int:
    widths = np.array(box.upper) - np.array(box.lower)
    dimension = int(np.argmax(widths))
    if widths[dimension] <= 1e-14:
        raise ArithmeticError("nonnegative box has no splittable dimension")
    return dimension


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(args.prime)
    equality, equality_target, _, _ = bulk.constraints(layout, args.eta, args.delta)
    lower = [0.0] * (3 * args.prime)
    upper = [1.0] * (3 * args.prime)
    upper[0] = 1.0 - args.eta
    lower[2 * args.prime] = 1.0 - args.delta
    initial = box_bound(
        Box(tuple(lower), tuple(upper)),
        layout,
        equality,
        equality_target,
        args.repetition,
        args.entropy_relaxation,
    )
    if initial is None:
        raise RuntimeError("initial bulk domain is infeasible")
    queue: list[tuple[float, int, Bound]] = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    smallest_entry = initial.minimum_transition_entry
    largest_residual = initial.dual_residual
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < 0.0:
            return {
                "status": "negative floating-point cover found",
                "upper_bound": current.value,
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "smallest_reference_transition_entry": smallest_entry,
                "largest_dual_residual": largest_residual,
            }
        dimension = current.split_dimension
        if current.box.upper[dimension] - current.box.lower[dimension] <= 1e-14:
            dimension = choose_dimension(current.box)
        for child in split(current.box, dimension):
            bounded = box_bound(
                child,
                layout,
                equality,
                equality_target,
                args.repetition,
                args.entropy_relaxation,
                inherited=current.duals,
            )
            if bounded is None:
                continue
            smallest_entry = min(smallest_entry, bounded.minimum_transition_entry)
            largest_residual = max(largest_residual, bounded.dual_residual)
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
    largest = -queue[0][0] if queue else -np.inf
    return {
        "status": "cover search incomplete",
        "current_upper_bound": largest,
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "smallest_reference_transition_entry": smallest_entry,
        "largest_dual_residual": largest_residual,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.196)
    parser.add_argument("--eta", type=float, default=0.0001)
    parser.add_argument("--max-boxes", type=int, default=1000)
    parser.add_argument(
        "--entropy-relaxation", choices=("scalar", "full"), default="scalar"
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
            "entropy_relaxation": args.entropy_relaxation,
            "arithmetic": "floating-point diagnostic",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
