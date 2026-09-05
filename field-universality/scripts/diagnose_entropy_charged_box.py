#!/usr/bin/env python3
"""Bound one frontier box with entropy-charged product couplings.

The LP uses two coupling tables with the three composition laws as marginals.
Positive entropy and logarithm terms receive global tangents.  Convex negative
marginal-entropy terms receive coordinate secants on the input box.  The
reported LP is mathematically valid but floating point, not a certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected


COMPOSITIONS = 15
FIRST_COUPLING = 15
SECOND_COUPLING = 40
FIRST_ENTROPY = 65
SECOND_ENTROPY = 66
FIRST_CUT = 67
SECOND_CUT = 68
SIZE = 69


def coupling_index(stage: int, left: int, right: int) -> int:
    return (FIRST_COUPLING if stage == 0 else SECOND_COUPLING) + 5 * left + right


def xlogx(value: float) -> float:
    return 0.0 if value <= 0.0 else value * np.log(value)


def convex_secant(lower: float, upper: float) -> tuple[float, float]:
    if upper - lower <= 1e-14:
        value = max(upper, 1e-300)
        slope = np.log(value) + 1.0
        return slope, xlogx(value) - slope * value
    lower_value = xlogx(lower)
    upper_value = xlogx(upper)
    slope = (upper_value - lower_value) / (upper - lower)
    return slope, lower_value - slope * lower


def entropy(distribution: np.ndarray) -> float:
    positive = distribution > 0.0
    return -float(np.dot(distribution[positive], np.log(distribution[positive])))


def conditional_entropy(coupling: np.ndarray, paid_side: str) -> float:
    if paid_side == "left":
        conditioning = coupling.sum(axis=0)
        conditionals = coupling / np.maximum(conditioning[None, :], 1e-300)
    else:
        conditioning = coupling.sum(axis=1)
        conditionals = coupling / np.maximum(conditioning[:, None], 1e-300)
    positive = coupling > 0.0
    return -float(np.sum(coupling[positive] * np.log(conditionals[positive])))


def evaluate_box(
    box: projected.Box,
    first_cuts: list[object],
    second_cuts: list[object],
    first_kernels: tuple[np.ndarray, ...],
    second_kernels: tuple[np.ndarray, ...],
    multiplier: float,
    outer_iterations: int,
    second_paid_side: str,
) -> dict[str, object]:
    tightened = projected.tighten(box)
    if tightened is None:
        raise ValueError("frontier box is infeasible")
    box = tightened
    objective = np.zeros(SIZE)
    objective[FIRST_ENTROPY] = multiplier
    objective[SECOND_ENTROPY] = multiplier
    objective[FIRST_CUT] = 1.0
    objective[SECOND_CUT] = 1.0
    objective_constant = 0.0
    entropy_weights = (
        multiplier - 1.0 / 3.0,
        multiplier if second_paid_side == "left" else 0.0,
        multiplier if second_paid_side == "right" else 0.0,
    )
    for block, start in enumerate((0, 5, 10)):
        for coordinate in range(start, start + 5):
            slope, intercept = convex_secant(
                box.lower[coordinate], box.upper[coordinate]
            )
            objective[coordinate] += entropy_weights[block] * slope
            objective_constant += entropy_weights[block] * intercept

    equality_rows = []
    equality_targets = []
    for start in (0, 5, 10):
        row = np.zeros(SIZE)
        row[start : start + 5] = 1.0
        equality_rows.append(row)
        equality_targets.append(1.0)
    for stage, left_start, right_start in ((0, 0, 5), (1, 5, 10)):
        for left in range(5):
            row = np.zeros(SIZE)
            for right in range(5):
                row[coupling_index(stage, left, right)] = 1.0
            row[left_start + left] = -1.0
            equality_rows.append(row)
            equality_targets.append(0.0)
        for right in range(4):
            row = np.zeros(SIZE)
            for left in range(5):
                row[coupling_index(stage, left, right)] = 1.0
            row[right_start + right] = -1.0
            equality_rows.append(row)
            equality_targets.append(0.0)

    rows = []
    targets = []
    support_first = np.zeros(SIZE)
    support_first[1:5] = 1.0
    support_first[6:10] = -2.0
    rows.append(support_first)
    targets.append(0.0)
    support_second = np.zeros(SIZE)
    support_second[6:10] = 1.0
    support_second[11:15] = -2.0
    rows.append(support_second)
    targets.append(0.0)
    for value in range(2, 5):
        canonical = np.zeros(SIZE)
        canonical[value] = 1.0
        canonical[1] = -1.0
        rows.append(canonical)
        targets.append(0.0)

    paid_sides = ("left", second_paid_side)

    def add_entropy_tangent(stage: int, reference: np.ndarray) -> None:
        reference = np.maximum(reference, 1e-15)
        reference /= reference.sum()
        if paid_sides[stage] == "left":
            conditioning = reference.sum(axis=0)
            conditionals = reference / conditioning[None, :]
        else:
            conditioning = reference.sum(axis=1)
            conditionals = reference / conditioning[:, None]
        row = np.zeros(SIZE)
        row[FIRST_ENTROPY if stage == 0 else SECOND_ENTROPY] = 1.0
        start = FIRST_COUPLING if stage == 0 else SECOND_COUPLING
        row[start : start + 25] = np.log(conditionals.ravel())
        rows.append(row)
        targets.append(0.0)

    def add_cut_tangent(
        stage: int,
        cut: object,
        kernel: np.ndarray,
        partition_reference: float,
    ) -> None:
        left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
        row = np.zeros(SIZE)
        row[FIRST_CUT if stage == 0 else SECOND_CUT] = 1.0
        coupling_start = FIRST_COUPLING if stage == 0 else SECOND_COUPLING
        row[coupling_start : coupling_start + 25] -= (
            kernel.ravel() / partition_reference
        )
        row[right_start : right_start + 5] += cut.target_potential
        row[left_start + 1 : left_start + 5] += cut.increment_support_potential
        row[right_start + 1 : right_start + 5] += cut.state_support_potential
        rows.append(row)
        targets.append(np.log(partition_reference) - 1.0)

    references = [
        np.outer(projected.reference(box, 0), projected.reference(box, 5)),
        np.outer(projected.reference(box, 5), projected.reference(box, 10)),
    ]
    add_entropy_tangent(0, references[0])
    add_entropy_tangent(1, references[1])
    for stage, cuts, kernels in (
        (0, first_cuts, first_kernels),
        (1, second_cuts, second_kernels),
    ):
        for cut, kernel in zip(cuts, kernels, strict=True):
            add_cut_tangent(
                stage, cut, kernel, float(np.sum(kernel * references[stage]))
            )

    lower_bounds = np.zeros(SIZE)
    upper_bounds = np.ones(SIZE)
    lower_bounds[:COMPOSITIONS] = box.lower
    upper_bounds[:COMPOSITIONS] = box.upper
    for stage, left_start, right_start in ((0, 0, 5), (1, 5, 10)):
        for left in range(5):
            for right in range(5):
                index = coupling_index(stage, left, right)
                upper_bounds[index] = min(
                    box.upper[left_start + left], box.upper[right_start + right]
                )
    lower_bounds[FIRST_ENTROPY:] = -1000.0
    upper_bounds[FIRST_ENTROPY:] = 1000.0

    def solve() -> object:
        return linprog(
            -objective,
            A_ub=np.stack(rows),
            b_ub=np.asarray(targets),
            A_eq=np.stack(equality_rows),
            b_eq=np.asarray(equality_targets),
            bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
            method="highs",
        )

    result = solve()
    if not result.success:
        raise RuntimeError(result.message)
    for _ in range(outer_iterations):
        additions = 0
        for stage, entropy_index, coupling_start in (
            (0, FIRST_ENTROPY, FIRST_COUPLING),
            (1, SECOND_ENTROPY, SECOND_COUPLING),
        ):
            coupling = result.x[coupling_start : coupling_start + 25].reshape((5, 5))
            actual_entropy = conditional_entropy(coupling, paid_sides[stage])
            if result.x[entropy_index] > actual_entropy + 1e-9:
                add_entropy_tangent(stage, coupling)
                additions += 1
        point = result.x[:COMPOSITIONS]
        for stage, cuts, kernels, cut_index, coupling_start, left_start, right_start in (
            (0, first_cuts, first_kernels, FIRST_CUT, FIRST_COUPLING, 0, 5),
            (1, second_cuts, second_kernels, SECOND_CUT, SECOND_COUPLING, 5, 10),
        ):
            coupling = result.x[coupling_start : coupling_start + 25].reshape((5, 5))
            values = []
            for cut, kernel in zip(cuts, kernels, strict=True):
                values.append(
                    np.log(float(np.sum(kernel * coupling)))
                    - float(np.dot(cut.target_potential, point[right_start : right_start + 5]))
                    - cut.increment_support_potential * float(point[left_start + 1 : left_start + 5].sum())
                    - cut.state_support_potential * float(point[right_start + 1 : right_start + 5].sum())
                )
            active = int(np.argmin(values))
            if result.x[cut_index] > values[active] + 1e-9:
                add_cut_tangent(
                    stage,
                    cuts[active],
                    kernels[active],
                    float(np.sum(kernels[active] * coupling)),
                )
                additions += 1
        if additions == 0:
            break
        result = solve()
        if not result.success:
            raise RuntimeError(result.message)

    point = projected.sanitize_composition(result.x[:COMPOSITIONS])
    couplings = [
        result.x[FIRST_COUPLING : FIRST_COUPLING + 25].reshape((5, 5)),
        result.x[SECOND_COUPLING : SECOND_COUPLING + 25].reshape((5, 5)),
    ]
    informations = []
    for coupling, left, right in (
        (couplings[0], point[0:5], point[5:10]),
        (couplings[1], point[5:10], point[10:15]),
    ):
        product = np.outer(left, right)
        informations.append(
            float(np.sum(coupling * np.log(np.maximum(coupling, 1e-300) / product)))
        )
    exact_marginal = 0.0
    secant_marginal = objective_constant
    for block, start in enumerate((0, 5, 10)):
        exact_marginal += entropy_weights[block] * sum(
            xlogx(float(value)) for value in point[start : start + 5]
        )
        secant_marginal += float(
            np.dot(
                objective[start : start + 5],
                point[start : start + 5],
            )
        )
    actual_conditionals = [
        conditional_entropy(couplings[stage], paid_sides[stage])
        for stage in range(2)
    ]
    actual_cut_values = []
    for stage, cuts, kernels, coupling, left_start, right_start in (
        (0, first_cuts, first_kernels, couplings[0], 0, 5),
        (1, second_cuts, second_kernels, couplings[1], 5, 10),
    ):
        values = [
            np.log(float(np.sum(kernel * coupling)))
            - float(np.dot(cut.target_potential, point[right_start : right_start + 5]))
            - cut.increment_support_potential * float(point[left_start + 1 : left_start + 5].sum())
            - cut.state_support_potential * float(point[right_start + 1 : right_start + 5].sum())
            for cut, kernel in zip(cuts, kernels, strict=True)
        ]
        actual_cut_values.append(min(values))
    nonlinear_value = (
        exact_marginal
        + multiplier * sum(actual_conditionals)
        + sum(actual_cut_values)
    )
    return {
        "upper": -float(result.fun) + objective_constant,
        "nonlinear_value_at_lp_solution": nonlinear_value,
        "relaxation_gap_at_lp_solution": (
            -float(result.fun) + objective_constant - nonlinear_value
        ),
        "marginal_secant_gap": secant_marginal - exact_marginal,
        "conditional_tangent_gap": multiplier
        * (
            result.x[FIRST_ENTROPY]
            + result.x[SECOND_ENTROPY]
            - sum(actual_conditionals)
        ),
        "cut_tangent_gap": (
            result.x[FIRST_CUT]
            + result.x[SECOND_CUT]
            - sum(actual_cut_values)
        ),
        "point": point.tolist(),
        "coupling_informations": informations,
        "paid_sides": list(paid_sides),
        "constraints": len(rows),
        "outer_iterations": outer_iterations,
        "status": "valid floating LP upper; not an exact certificate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument("--multiplier", type=float, default=64.0)
    parser.add_argument("--outer-iterations", type=int, default=32)
    parser.add_argument(
        "--second-paid-side", choices=("left", "right"), default="left"
    )
    args = parser.parse_args()
    bank_args = SimpleNamespace(
        eta=0.002,
        delta=0.22,
        bank_iterations=20,
        bank_restarts=12,
        seed=11,
        bank_cache=str(args.bank),
        rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    saved = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(saved["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    box = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))
    result = evaluate_box(
        box,
        first_cuts,
        second_cuts,
        first_kernels,
        second_kernels,
        args.multiplier,
        args.outer_iterations,
        args.second_paid_side,
    )
    result.update(
        {
            "frontier_index": args.frontier_index,
            "multiplier": args.multiplier,
            "projected_moment_upper": selected["bound"],
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
