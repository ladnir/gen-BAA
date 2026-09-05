#!/usr/bin/env python3
"""Floating LP boxes for a fixed projected-KL dual bank.

The relaxation respects the upper-bound quantifiers: entropy and logarithms
use global tangents, and products use valid moment relaxations.  The reported
numbers are not exact certificates because cut construction, transcendental
constants, and LP solves use floating arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import projected_kl_dual as dual
import search_projected_kl_dual_envelope as envelope
import exact_stationary_dual as stationary


COMPOSITION_SIZE = 15
MOMENT_PAIRS = tuple(
    (left, right)
    for left in range(COMPOSITION_SIZE)
    for right in range(left, COMPOSITION_SIZE)
)
MOMENT_OFFSETS = {pair: offset for offset, pair in enumerate(MOMENT_PAIRS)}
MOMENT_START = COMPOSITION_SIZE
MOMENT_END = MOMENT_START + len(MOMENT_PAIRS)


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"), sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def encode_box(box: Box) -> dict[str, object]:
    return {"lower": list(box.lower), "upper": list(box.upper)}


def load_or_initialize_split_log(
    path: Path | None,
    splits_completed: int,
    queue: list[tuple[float, int, tuple[object, ...]]],
) -> dict[str, object] | None:
    if path is None:
        return None
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("version") != 1:
            raise ValueError("unexpected projected split-log schema")
        saved.setdefault("outward_margin_units", 10000)
        records = [
            record
            for record in saved["records"]
            if int(record["after_split"]) <= splits_completed
        ]
        if records and int(records[-1]["after_split"]) != splits_completed:
            raise ValueError("projected checkpoint is ahead of its split log")
        if not records and int(saved["base_splits"]) != splits_completed:
            raise ValueError("projected checkpoint does not match split-log base")
        saved["records"] = records
        return saved
    return {
        "version": 1,
        "base_splits": splits_completed,
        "base_identifiers": sorted(identifier for _, identifier, _ in queue),
        "outward_margin_units": 10000,
        "records": [],
    }


def save_split_log(path: Path | None, value: dict[str, object] | None) -> None:
    if path is not None and value is not None:
        write_json_atomic(path, value)


def encode_cut(cut: dual.ProjectionDual) -> dict[str, object]:
    return {
        "target_potential": cut.target_potential.tolist(),
        "increment_support_potential": cut.increment_support_potential,
        "state_support_potential": cut.state_support_potential,
        "value": cut.value,
        "gradient_residual": cut.gradient_residual,
        "optimizer_success": cut.optimizer_success,
    }


def decode_cut(value: dict[str, object]) -> dual.ProjectionDual:
    return dual.ProjectionDual(
        np.asarray(value["target_potential"], dtype=float),
        float(value["increment_support_potential"]),
        float(value["state_support_potential"]),
        float(value["value"]),
        float(value["gradient_residual"]),
        bool(value["optimizer_success"]),
    )


def bank_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "eta": args.eta,
        "delta": args.delta,
        "iterations": args.bank_iterations,
        "restarts": args.bank_restarts,
        "tolerance": 1e-7,
        "seed": args.seed,
    }


def bank_digest(
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> str:
    digest = hashlib.sha256()
    for label, cuts in ((b"first", first_cuts), (b"second", second_cuts)):
        digest.update(label)
        for cut in cuts:
            digest.update(np.asarray(cut.target_potential, dtype="<f8").tobytes())
            digest.update(
                np.asarray(
                    [
                        cut.increment_support_potential,
                        cut.state_support_potential,
                    ],
                    dtype="<f8",
                ).tobytes()
            )
    return digest.hexdigest()


def save_bank_cache(
    path: Path,
    parameters: dict[str, object],
    result: dict[str, object],
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> None:
    write_json_atomic(
        path,
        {
            "version": 1,
            "parameters": parameters,
            "result": result,
            "first_cuts": [encode_cut(cut) for cut in first_cuts],
            "second_cuts": [encode_cut(cut) for cut in second_cuts],
        },
    )


def load_or_generate_bank(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dual.ProjectionDual], list[dual.ProjectionDual]]:
    cache_path = Path(args.bank_cache)
    parameters = bank_parameters(args)
    if cache_path.exists() and not args.rebuild_bank:
        with cache_path.open("r", encoding="utf-8") as source:
            cached = json.load(source)
        if cached.get("parameters") != parameters:
            raise RuntimeError(
                f"bank cache parameters do not match: {cache_path}"
            )
        first_cuts = [decode_cut(value) for value in cached["first_cuts"]]
        second_cuts = [decode_cut(value) for value in cached["second_cuts"]]
        return cached["result"], first_cuts, second_cuts
    bank_args = argparse.Namespace(**parameters)
    result, first_cuts, second_cuts = envelope.generate_bank(bank_args)
    save_bank_cache(
        cache_path,
        parameters,
        result,
        first_cuts,
        second_cuts,
    )
    return result, first_cuts, second_cuts


def bulk_entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def sanitize_composition(point: np.ndarray) -> np.ndarray:
    sanitized = np.maximum(np.asarray(point, dtype=float), 0.0).copy()
    for start in (0, 5, 10):
        total = float(sanitized[start : start + 5].sum())
        if total <= 0.0:
            raise ValueError("composition block has zero mass")
        sanitized[start : start + 5] /= total
    return sanitized


def initial_box(eta: float, delta: float) -> Box:
    lower = [0.0] * 15
    upper = [1.0] * 15
    intervals = (
        (0, eta, min(1.0, 4.0 * delta)),
        (5, eta / 2.0, min(1.0, 2.0 * delta)),
        (10, eta / 4.0, delta),
    )
    for start, support_lower, support_upper in intervals:
        lower[start] = 1.0 - support_upper
        upper[start] = 1.0 - support_lower
        for index in range(start + 1, start + 5):
            upper[index] = support_upper
    return Box(tuple(lower), tuple(upper))


def tighten_simplex(
    lower: list[float], upper: list[float], start: int
) -> bool:
    for _ in range(6):
        total_lower = sum(lower[start : start + 5])
        total_upper = sum(upper[start : start + 5])
        if total_lower > 1.0 + 1e-12 or total_upper < 1.0 - 1e-12:
            return False
        changed = False
        for offset in range(5):
            index = start + offset
            new_lower = max(lower[index], 1.0 - (total_upper - upper[index]))
            new_upper = min(upper[index], 1.0 - (total_lower - lower[index]))
            if new_lower > new_upper + 1e-12:
                return False
            changed |= new_lower > lower[index] + 1e-15
            changed |= new_upper < upper[index] - 1e-15
            lower[index] = new_lower
            upper[index] = new_upper
        if not changed:
            break
    return True


def tighten(raw: Box) -> Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    for start in (0, 5, 10):
        if not tighten_simplex(lower, upper, start):
            return None
    return Box(tuple(lower), tuple(upper))


def reference(box: Box, start: int) -> np.ndarray:
    midpoint = np.array(
        [
            max(1e-15, (box.lower[index] + box.upper[index]) / 2.0)
            for index in range(start, start + 5)
        ]
    )
    return midpoint / midpoint.sum()


def simplex_box_vertices(box: Box, start: int) -> np.ndarray:
    """Enumerate the vertices of a five-coordinate boxed simplex."""
    lower = np.asarray(box.lower[start : start + 5], dtype=float)
    upper = np.asarray(box.upper[start : start + 5], dtype=float)
    vertices: list[np.ndarray] = []
    for free in range(5):
        fixed = [index for index in range(5) if index != free]
        for mask in range(1 << len(fixed)):
            point = np.empty(5)
            for offset, index in enumerate(fixed):
                point[index] = (
                    upper[index] if mask & (1 << offset) else lower[index]
                )
            point[free] = 1.0 - float(point[fixed].sum())
            if (
                point[free] >= lower[free] - 1e-12
                and point[free] <= upper[free] + 1e-12
            ):
                point[free] = min(
                    upper[free], max(lower[free], point[free])
                )
                vertices.append(point)
    if not vertices:
        raise RuntimeError("boxed simplex has no vertices")
    rounded = {
        tuple(float(value) for value in np.round(point, decimals=15))
        for point in vertices
    }
    return np.asarray(sorted(rounded), dtype=float)


def bilinear_remainder_upper(
    matrix: np.ndarray,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    left_vertices: np.ndarray,
    right_vertices: np.ndarray,
) -> float:
    """Maximize the centered bilinear remainder over two boxed simplices."""
    left_differences = left_vertices - left_reference
    right_differences = right_vertices - right_reference
    maximum = float(
        np.max(left_differences @ matrix @ right_differences.T)
    )
    return max(0.0, maximum)


def moment_index(left: int, right: int) -> int:
    pair = (left, right) if left <= right else (right, left)
    return MOMENT_START + MOMENT_OFFSETS[pair]


def product_index(stage: int, left: int, right: int) -> int:
    left_start, right_start = ((0, 5), (5, 10))[stage]
    return moment_index(left_start + left, right_start + right)


def nonzero_residual_slab_inequality(
    stage: int,
    coordinate: int,
    sign: int,
    threshold: float,
) -> tuple[np.ndarray, float]:
    """Return a lifted necessary condition for one nonzero residual slab.

    The true residual is r_a=(A-x) beta_a/B with 0<=x<=A/2.
    Positive and negative slabs therefore imply, respectively,

      -A beta_a + AB/5 + rho B <= 0,
       A beta_a/2 - AB/5 + rho B <= 0.

    Products are represented by the evaluator's lifted moment variables.
    """
    if stage not in (0, 1):
        raise ValueError("stage must be zero or one")
    if coordinate not in range(1, 5):
        raise ValueError("coordinate must be nonzero")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    increment_start, state_start = ((0, 5), (5, 10))[stage]
    state_coordinate = state_start + coordinate
    row = np.zeros(MOMENT_END)
    for increment_coordinate in range(increment_start + 1, increment_start + 5):
        row[moment_index(increment_coordinate, state_coordinate)] += (
            -1.0 if sign > 0 else 0.5
        )
        for state_nonzero in range(state_start + 1, state_start + 5):
            row[moment_index(increment_coordinate, state_nonzero)] += (
                0.2 if sign > 0 else -0.2
            )
    row[state_start + 1 : state_start + 5] += threshold
    return row, 0.0


def add_mccormick(
    rows: list[np.ndarray],
    targets: list[float],
    size: int,
    x: int,
    y: int,
    z: int,
    lx: float,
    ux: float,
    ly: float,
    uy: float,
) -> None:
    # z >= lx*y + ly*x - lx*ly
    row = np.zeros(size)
    row[x] += ly
    row[y] += lx
    row[z] -= 1.0
    rows.append(row)
    targets.append(lx * ly)
    # z >= ux*y + uy*x - ux*uy
    row = np.zeros(size)
    row[x] += uy
    row[y] += ux
    row[z] -= 1.0
    rows.append(row)
    targets.append(ux * uy)
    # z <= ux*y + ly*x - ux*ly
    row = np.zeros(size)
    row[x] -= ly
    row[y] -= ux
    row[z] += 1.0
    rows.append(row)
    targets.append(-ux * ly)
    # z <= lx*y + uy*x - lx*uy
    row = np.zeros(size)
    row[x] -= uy
    row[y] -= lx
    row[z] += 1.0
    rows.append(row)
    targets.append(-lx * uy)


def stage_upper_with_kernel(
    increment: np.ndarray,
    state: np.ndarray,
    cut: dual.ProjectionDual,
    matrix: np.ndarray,
) -> float:
    partition = float(increment @ matrix @ state)
    return float(
        np.log(partition)
        - np.dot(cut.target_potential, state)
        - cut.increment_support_potential * increment[1:].sum()
        - cut.state_support_potential * state[1:].sum()
    )


def evaluate(
    raw_box: Box,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
    first_kernels: tuple[np.ndarray, ...] | None = None,
    second_kernels: tuple[np.ndarray, ...] | None = None,
    outer_iterations: int = 8,
    affine_bilinear_cuts: int = 0,
    exact_first_cuts: tuple[stationary.StationaryDualCut, ...] = (),
    exact_second_cuts: tuple[stationary.StationaryDualCut, ...] = (),
    additional_composition_inequalities: tuple[np.ndarray, ...] = (),
    additional_lifted_inequalities: tuple[tuple[np.ndarray, float], ...] = (),
    stage_objective_weight: float = 1.0,
    additional_weighted_first_cuts: tuple[
        tuple[float, dual.ProjectionDual, np.ndarray], ...
    ] = (),
    additional_weighted_second_cuts: tuple[
        tuple[float, dual.ProjectionDual, np.ndarray], ...
    ] = (),
) -> tuple[float, int, np.ndarray, Box, float, float] | None:
    if stage_objective_weight <= 0.0:
        raise ValueError("the stage objective weight must be positive")
    if stage_objective_weight != 1.0 and (
        exact_first_cuts or exact_second_cuts
    ):
        raise ValueError("weighted projected stages do not support exact stationary cuts")
    if stage_objective_weight != 1.0 and (
        additional_weighted_first_cuts or additional_weighted_second_cuts
    ):
        raise ValueError("weighted stages cannot be combined with extra weighted cuts")
    if first_kernels is None:
        first_kernels = tuple(envelope.kernel(cut) for cut in first_cuts)
    if second_kernels is None:
        second_kernels = tuple(envelope.kernel(cut) for cut in second_cuts)
    box = tighten(raw_box)
    if box is None:
        return None
    size = MOMENT_END + 3
    entropy_epigraph, t1, t2 = MOMENT_END, MOMENT_END + 1, MOMENT_END + 2
    objective = np.zeros(size)
    message_reference = reference(box, 0)
    objective[entropy_epigraph] = 1.0
    objective[t1] = stage_objective_weight
    objective[t2] = stage_objective_weight
    equality_rows = []
    equality_targets = []
    for component, start in enumerate((0, 5, 10)):
        row = np.zeros(size)
        row[start : start + 5] = 1.0
        equality_rows.append(row)
        equality_targets.append(1.0)
    rows = []
    targets = []
    composition_inequalities = []

    def add_entropy_tangent(entropy_reference: np.ndarray) -> None:
        row = np.zeros(size)
        row[entropy_epigraph] = 1.0
        row[0:5] = np.log(np.maximum(entropy_reference, 1e-300)) / 3.0
        rows.append(row)
        targets.append(0.0)

    entropy_references = [message_reference]
    add_entropy_tangent(message_reference)
    support_first = np.zeros(size)
    support_first[1:5] = 1.0
    support_first[6:10] = -2.0
    rows.append(support_first)
    targets.append(0.0)
    composition_inequalities.append(support_first)
    support_second = np.zeros(size)
    support_second[6:10] = 1.0
    support_second[11:15] = -2.0
    rows.append(support_second)
    targets.append(0.0)
    composition_inequalities.append(support_second)
    for value in range(2, 5):
        canonical = np.zeros(size)
        canonical[value] = 1.0
        canonical[1] = -1.0
        rows.append(canonical)
        targets.append(0.0)
        composition_inequalities.append(canonical)
    for supplied in additional_composition_inequalities:
        supplied = np.asarray(supplied, dtype=float)
        if supplied.shape != (COMPOSITION_SIZE,):
            raise ValueError("a composition inequality must have length 15")
        inequality = np.zeros(size)
        inequality[:COMPOSITION_SIZE] = supplied
        rows.append(inequality)
        targets.append(0.0)
        composition_inequalities.append(inequality)
    for supplied, target in additional_lifted_inequalities:
        supplied = np.asarray(supplied, dtype=float)
        if supplied.shape != (MOMENT_END,):
            raise ValueError(
                "a lifted inequality must cover compositions and moments"
            )
        inequality = np.zeros(size)
        inequality[:MOMENT_END] = supplied
        rows.append(inequality)
        targets.append(float(target))
    for left, right in MOMENT_PAIRS:
        add_mccormick(
            rows,
            targets,
            size,
            left,
            right,
            moment_index(left, right),
            box.lower[left],
            box.upper[left],
            box.lower[right],
            box.upper[right],
        )

    # Multiply each simplex identity by every composition coordinate.  These
    # equations give the moment matrix the correct block marginals.
    for simplex_start in (0, 5, 10):
        for coordinate in range(COMPOSITION_SIZE):
            row = np.zeros(size)
            for member in range(simplex_start, simplex_start + 5):
                row[moment_index(member, coordinate)] += 1.0
            row[coordinate] -= 1.0
            equality_rows.append(row)
            equality_targets.append(0.0)

    # First-level RLT constraints.  If g(x) <= 0 on the composition domain,
    # then both g(x)(x_k-l_k) and g(x)(u_k-x_k) are nonpositive.  Their lifted
    # forms rule out mixtures whose average satisfies g although individual
    # product components do not.
    for inequality in composition_inequalities:
        for coordinate in range(COMPOSITION_SIZE):
            lower_row = np.zeros(size)
            upper_row = np.zeros(size)
            for member, coefficient in enumerate(inequality[:COMPOSITION_SIZE]):
                if coefficient == 0.0:
                    continue
                lower_row[moment_index(member, coordinate)] += coefficient
                lower_row[member] -= box.lower[coordinate] * coefficient
                upper_row[member] += box.upper[coordinate] * coefficient
                upper_row[moment_index(member, coordinate)] -= coefficient
            rows.extend((lower_row, upper_row))
            targets.extend((0.0, 0.0))

    def add_cut_tangent(
        cut: dual.ProjectionDual,
        matrix: np.ndarray,
        stage: int,
        increment_start: int,
        state_start: int,
        epigraph: int,
        partition_reference: float,
        weight: float = 1.0,
    ) -> None:
        row = np.zeros(size)
        row[epigraph] = 1.0
        for left in range(5):
            for right in range(5):
                row[product_index(stage, left, right)] -= (
                    weight * matrix[left, right] / partition_reference
                )
        row[state_start : state_start + 5] += weight * cut.target_potential
        row[increment_start + 1 : increment_start + 5] += (
            weight * cut.increment_support_potential
        )
        row[state_start + 1 : state_start + 5] += (
            weight * cut.state_support_potential
        )
        # log(z) <= log(z0) + z/z0 - 1.
        rows.append(row)
        targets.append(weight * (np.log(partition_reference) - 1.0))

    def add_exact_cut_tangent(
        cut: stationary.StationaryDualCut,
        stage: int,
        increment_start: int,
        state_start: int,
        epigraph: int,
        partition_reference: float,
    ) -> None:
        row = np.zeros(size)
        row[epigraph] = 1.0
        for left in range(5):
            for right in range(5):
                row[product_index(stage, left, right)] -= (
                    cut.kernel[left, right] / partition_reference
                )
        row[increment_start : increment_start + 5] += (
            cut.moment_coefficients[:5]
        )
        row[state_start : state_start + 5] += (
            cut.moment_coefficients[5:]
        )
        rows.append(row)
        targets.append(
            np.log(partition_reference) - 1.0 - cut.moment_constant
        )

    def add_affine_bilinear_cut(
        cut: dual.ProjectionDual,
        matrix: np.ndarray,
        increment_start: int,
        state_start: int,
        epigraph: int,
        increment_reference: np.ndarray,
        state_reference: np.ndarray,
        increment_vertices: np.ndarray,
        state_vertices: np.ndarray,
    ) -> None:
        partition_reference = float(
            increment_reference @ matrix @ state_reference
        )
        remainder = bilinear_remainder_upper(
            matrix,
            increment_reference,
            state_reference,
            increment_vertices,
            state_vertices,
        )
        row = np.zeros(size)
        row[epigraph] = 1.0
        row[increment_start : increment_start + 5] -= (
            matrix @ state_reference
        ) / partition_reference
        row[state_start : state_start + 5] -= (
            matrix.T @ increment_reference
        ) / partition_reference
        row[state_start : state_start + 5] += cut.target_potential
        row[increment_start + 1 : increment_start + 5] += (
            cut.increment_support_potential
        )
        row[state_start + 1 : state_start + 5] += (
            cut.state_support_potential
        )
        rows.append(row)
        targets.append(
            np.log(partition_reference)
            - 2.0
            + remainder / partition_reference
        )

    def add_exact_affine_bilinear_cut(
        cut: stationary.StationaryDualCut,
        increment_start: int,
        state_start: int,
        epigraph: int,
        increment_reference: np.ndarray,
        state_reference: np.ndarray,
        increment_vertices: np.ndarray,
        state_vertices: np.ndarray,
    ) -> None:
        partition_reference = float(
            increment_reference @ cut.kernel @ state_reference
        )
        remainder = bilinear_remainder_upper(
            cut.kernel,
            increment_reference,
            state_reference,
            increment_vertices,
            state_vertices,
        )
        row = np.zeros(size)
        row[epigraph] = 1.0
        row[increment_start : increment_start + 5] -= (
            cut.kernel @ state_reference
        ) / partition_reference
        row[state_start : state_start + 5] -= (
            cut.kernel.T @ increment_reference
        ) / partition_reference
        row[increment_start : increment_start + 5] += (
            cut.moment_coefficients[:5]
        )
        row[state_start : state_start + 5] += (
            cut.moment_coefficients[5:]
        )
        rows.append(row)
        targets.append(
            np.log(partition_reference)
            - 2.0
            + remainder / partition_reference
            - cut.moment_constant
        )

    first_references: list[list[float]] = []
    second_references: list[list[float]] = []
    for cut, matrix in zip(first_cuts, first_kernels, strict=True):
        left_reference = reference(box, 0)
        right_reference = reference(box, 5)
        partition_reference = float(
            left_reference @ matrix @ right_reference
        )
        first_references.append([partition_reference])
        add_cut_tangent(cut, matrix, 0, 0, 5, t1, partition_reference)
    for cut, matrix in zip(second_cuts, second_kernels, strict=True):
        left_reference = reference(box, 5)
        right_reference = reference(box, 10)
        partition_reference = float(
            left_reference @ matrix @ right_reference
        )
        second_references.append([partition_reference])
        add_cut_tangent(cut, matrix, 1, 5, 10, t2, partition_reference)
    weighted_references: list[list[list[float]]] = [
        [[] for _ in additional_weighted_first_cuts],
        [[] for _ in additional_weighted_second_cuts],
    ]
    for stage, weighted_cuts, left_start, right_start, epigraph in (
        (0, additional_weighted_first_cuts, 0, 5, t1),
        (1, additional_weighted_second_cuts, 5, 10, t2),
    ):
        left_reference = reference(box, left_start)
        right_reference = reference(box, right_start)
        for index, (weight, cut, matrix) in enumerate(weighted_cuts):
            partition_reference = float(
                left_reference @ matrix @ right_reference
            )
            weighted_references[stage][index].append(partition_reference)
            add_cut_tangent(
                cut,
                matrix,
                stage,
                left_start,
                right_start,
                epigraph,
                partition_reference,
                weight,
            )
    exact_first_references: list[list[float]] = []
    exact_second_references: list[list[float]] = []
    for cut in exact_first_cuts:
        left_reference = reference(box, 0)
        right_reference = reference(box, 5)
        partition_reference = float(
            left_reference @ cut.kernel @ right_reference
        )
        exact_first_references.append([partition_reference])
        add_exact_cut_tangent(cut, 0, 0, 5, t1, partition_reference)
    for cut in exact_second_cuts:
        left_reference = reference(box, 5)
        right_reference = reference(box, 10)
        partition_reference = float(
            left_reference @ cut.kernel @ right_reference
        )
        exact_second_references.append([partition_reference])
        add_exact_cut_tangent(cut, 1, 5, 10, t2, partition_reference)
    block_vertices: list[np.ndarray] = []
    affine_references: list[dict[int, list[tuple[np.ndarray, np.ndarray]]]] = [
        {},
        {},
    ]
    exact_affine_references: list[
        dict[int, list[tuple[np.ndarray, np.ndarray]]]
    ] = [{}, {}]
    if affine_bilinear_cuts > 0:
        block_references = [reference(box, start) for start in (0, 5, 10)]
        block_vertices = [
            simplex_box_vertices(box, start) for start in (0, 5, 10)
        ]
        for stage, cuts, kernels, increment_block, state_block, epigraph in (
            (0, first_cuts, first_kernels, 0, 1, t1),
            (1, second_cuts, second_kernels, 1, 2, t2),
        ):
            increment_reference = block_references[increment_block]
            state_reference = block_references[state_block]
            center_values = [
                stage_upper_with_kernel(
                    increment_reference,
                    state_reference,
                    cut,
                    matrix,
                )
                for cut, matrix in zip(cuts, kernels, strict=True)
            ]
            selected = np.argsort(center_values)[
                : min(affine_bilinear_cuts, len(cuts))
            ]
            for active in selected:
                active = int(active)
                add_affine_bilinear_cut(
                    cuts[active],
                    kernels[active],
                    increment_block * 5,
                    state_block * 5,
                    epigraph,
                    increment_reference,
                    state_reference,
                    block_vertices[increment_block],
                    block_vertices[state_block],
                )
                affine_references[stage].setdefault(active, []).append(
                    (increment_reference.copy(), state_reference.copy())
                )
        for stage, cuts, increment_block, state_block, epigraph in (
            (0, exact_first_cuts, 0, 1, t1),
            (1, exact_second_cuts, 1, 2, t2),
        ):
            increment_reference = block_references[increment_block]
            state_reference = block_references[state_block]
            compositions = np.concatenate(
                (increment_reference, state_reference)
            )
            center_values = [
                np.log(
                    float(
                        increment_reference @ cut.kernel @ state_reference
                    )
                )
                - cut.moment_constant
                - float(np.dot(cut.moment_coefficients, compositions))
                for cut in cuts
            ]
            selected = np.argsort(center_values)[
                : min(affine_bilinear_cuts, len(cuts))
            ]
            for active in selected:
                active = int(active)
                add_exact_affine_bilinear_cut(
                    cuts[active],
                    increment_block * 5,
                    state_block * 5,
                    epigraph,
                    increment_reference,
                    state_reference,
                    block_vertices[increment_block],
                    block_vertices[state_block],
                )
                exact_affine_references[stage].setdefault(active, []).append(
                    (increment_reference.copy(), state_reference.copy())
                )
    lower_bounds = np.zeros(size)
    upper_bounds = np.ones(size)
    lower_bounds[0:COMPOSITION_SIZE] = box.lower
    upper_bounds[0:COMPOSITION_SIZE] = box.upper
    lower_bounds[entropy_epigraph : t2 + 1] = -1000.0
    upper_bounds[entropy_epigraph : t2 + 1] = 1000.0

    def solve() -> object:
        return linprog(
            -objective,
            A_ub=np.stack(rows),
            b_ub=np.array(targets),
            A_eq=np.stack(equality_rows),
            b_eq=np.array(equality_targets),
            bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
            method="highs",
        )

    def extract_products(solution: object) -> np.ndarray:
        return np.array(
            [
                [
                    [
                        solution.x[product_index(stage, left, right)]
                        for right in range(5)
                    ]
                    for left in range(5)
                ]
                for stage in range(2)
            ]
        )

    def extract_moment_matrix(solution: object) -> np.ndarray:
        matrix = np.empty((COMPOSITION_SIZE, COMPOSITION_SIZE))
        for left in range(COMPOSITION_SIZE):
            for right in range(COMPOSITION_SIZE):
                matrix[left, right] = solution.x[moment_index(left, right)]
        return matrix

    def add_covariance_cut(
        direction: np.ndarray, center: float
    ) -> None:
        # E[(v.X-c)^2] >= 0, written as a linear inequality in (x,Y).
        row = np.zeros(size)
        row[0:COMPOSITION_SIZE] = 2.0 * center * direction
        for left in range(COMPOSITION_SIZE):
            row[moment_index(left, left)] -= direction[left] ** 2
            for right in range(left + 1, COMPOSITION_SIZE):
                row[moment_index(left, right)] -= (
                    2.0 * direction[left] * direction[right]
                )
        rows.append(row)
        targets.append(center**2)

    def exact_relaxed_stage_values(
        cuts: list[dual.ProjectionDual],
        kernels: tuple[np.ndarray, ...],
        stage: int,
        increment_start: int,
        state_start: int,
        point: np.ndarray,
        products: np.ndarray,
    ) -> np.ndarray:
        values = []
        for cut, matrix in zip(cuts, kernels, strict=True):
            # HiGHS may return a nominally nonnegative moment at roughly
            # -1e-12.  Flooring the scalar partition avoids an invalid log in
            # diagnostics without inflating it by clipping each weighted cell.
            partition = max(
                float(np.sum(matrix * products[stage])),
                1e-300,
            )
            values.append(
                np.log(partition)
                - float(
                    np.dot(
                        cut.target_potential,
                        point[state_start : state_start + 5],
                    )
                )
                - cut.increment_support_potential
                * float(point[increment_start + 1 : increment_start + 5].sum())
                - cut.state_support_potential
                * float(point[state_start + 1 : state_start + 5].sum())
            )
        return np.array(values)

    def exact_stationary_relaxed_values(
        cuts: tuple[stationary.StationaryDualCut, ...],
        stage: int,
        increment_start: int,
        state_start: int,
        point: np.ndarray,
        products: np.ndarray,
    ) -> np.ndarray:
        values = []
        compositions = np.concatenate(
            (
                point[increment_start : increment_start + 5],
                point[state_start : state_start + 5],
            )
        )
        for cut in cuts:
            partition = max(
                float(np.sum(cut.kernel * products[stage])),
                1e-300,
            )
            values.append(
                np.log(partition)
                - cut.moment_constant
                - float(np.dot(cut.moment_coefficients, compositions))
            )
        return np.array(values)

    def weighted_relaxed_stage_values(
        weighted_cuts: tuple[
            tuple[float, dual.ProjectionDual, np.ndarray], ...
        ],
        stage: int,
        increment_start: int,
        state_start: int,
        point: np.ndarray,
        products: np.ndarray,
    ) -> np.ndarray:
        values = []
        for weight, cut, matrix in weighted_cuts:
            partition = max(float(np.sum(matrix * products[stage])), 1e-300)
            values.append(
                weight
                * (
                    np.log(partition)
                    - float(
                        np.dot(
                            cut.target_potential,
                            point[state_start : state_start + 5],
                        )
                    )
                    - cut.increment_support_potential
                    * float(point[increment_start + 1 : increment_start + 5].sum())
                    - cut.state_support_potential
                    * float(point[state_start + 1 : state_start + 5].sum())
                )
            )
        return np.asarray(values)

    result = solve()
    if not result.success:
        return None
    # Outer approximation of the three concave terms.  Each added tangent is
    # a global upper bound, so early termination remains sound.
    for _ in range(outer_iterations):
        point = result.x[0:COMPOSITION_SIZE]
        products = extract_products(result)
        additions = 0
        moment_matrix = extract_moment_matrix(result)
        covariance = moment_matrix - np.outer(point, point)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        for eigen_index in np.flatnonzero(eigenvalues < -1e-8):
            direction = eigenvectors[:, eigen_index]
            center = float(np.dot(direction, point))
            add_covariance_cut(direction, center)
            additions += 1
        entropy_value = bulk_entropy(point[0:5]) / 3.0
        if result.x[entropy_epigraph] > entropy_value + 1e-9:
            entropy_reference = np.maximum(point[0:5], 1e-12)
            entropy_reference /= entropy_reference.sum()
            add_entropy_tangent(entropy_reference)
            entropy_references.append(entropy_reference)
            additions += 1
        for cuts, kernels, references, stage, increment_start, state_start, epigraph in (
            (first_cuts, first_kernels, first_references, 0, 0, 5, t1),
            (second_cuts, second_kernels, second_references, 1, 5, 10, t2),
        ):
            values = exact_relaxed_stage_values(
                cuts,
                kernels,
                stage,
                increment_start,
                state_start,
                point,
                products,
            )
            if affine_bilinear_cuts > 0:
                increment_reference = point[
                    increment_start : increment_start + 5
                ].copy()
                state_reference = point[state_start : state_start + 5].copy()
                for active in np.argsort(values)[
                    : min(3, affine_bilinear_cuts, len(cuts))
                ]:
                    active = int(active)
                    old_references = affine_references[stage].get(active, [])
                    if any(
                        float(np.abs(increment_reference - old_increment).sum())
                        + float(np.abs(state_reference - old_state).sum())
                        <= 1e-7
                        for old_increment, old_state in old_references
                    ):
                        continue
                    add_affine_bilinear_cut(
                        cuts[active],
                        kernels[active],
                        increment_start,
                        state_start,
                        epigraph,
                        increment_reference,
                        state_reference,
                        block_vertices[increment_start // 5],
                        block_vertices[state_start // 5],
                    )
                    affine_references[stage].setdefault(active, []).append(
                        (increment_reference.copy(), state_reference.copy())
                    )
                    additions += 1
            for active in np.argsort(values)[:3]:
                active = int(active)
                if result.x[epigraph] <= values[active] + 1e-9:
                    continue
                matrix = kernels[active]
                partition_reference = max(
                    float(
                        np.sum(matrix * np.maximum(products[stage], 0.0))
                    ),
                    1e-300,
                )
                if all(
                    abs(np.log(partition_reference / old_reference)) > 1e-10
                    for old_reference in references[active]
                ):
                    references[active].append(partition_reference)
                    add_cut_tangent(
                        cuts[active],
                        matrix,
                        stage,
                        increment_start,
                        state_start,
                        epigraph,
                        partition_reference,
                    )
                    additions += 1
        for stage, weighted_cuts, increment_start, state_start, epigraph in (
            (0, additional_weighted_first_cuts, 0, 5, t1),
            (1, additional_weighted_second_cuts, 5, 10, t2),
        ):
            if not weighted_cuts:
                continue
            values = weighted_relaxed_stage_values(
                weighted_cuts,
                stage,
                increment_start,
                state_start,
                point,
                products,
            )
            for active in np.argsort(values)[:3]:
                active = int(active)
                if result.x[epigraph] <= values[active] + 1e-9:
                    continue
                weight, cut, matrix = weighted_cuts[active]
                partition_reference = max(
                    float(np.sum(matrix * np.maximum(products[stage], 0.0))),
                    1e-300,
                )
                references = weighted_references[stage][active]
                if all(
                    abs(np.log(partition_reference / old_reference)) > 1e-10
                    for old_reference in references
                ):
                    references.append(partition_reference)
                    add_cut_tangent(
                        cut,
                        matrix,
                        stage,
                        increment_start,
                        state_start,
                        epigraph,
                        partition_reference,
                        weight,
                    )
                    additions += 1
        for cuts, references, stage, increment_start, state_start, epigraph in (
            (exact_first_cuts, exact_first_references, 0, 0, 5, t1),
            (exact_second_cuts, exact_second_references, 1, 5, 10, t2),
        ):
            if not cuts:
                continue
            values = exact_stationary_relaxed_values(
                cuts,
                stage,
                increment_start,
                state_start,
                point,
                products,
            )
            if affine_bilinear_cuts > 0:
                increment_reference = point[
                    increment_start : increment_start + 5
                ].copy()
                state_reference = point[state_start : state_start + 5].copy()
                for active in np.argsort(values)[
                    : min(3, affine_bilinear_cuts, len(cuts))
                ]:
                    active = int(active)
                    old_references = exact_affine_references[stage].get(
                        active, []
                    )
                    if any(
                        float(
                            np.abs(increment_reference - old_increment).sum()
                        )
                        + float(np.abs(state_reference - old_state).sum())
                        <= 1e-7
                        for old_increment, old_state in old_references
                    ):
                        continue
                    add_exact_affine_bilinear_cut(
                        cuts[active],
                        increment_start,
                        state_start,
                        epigraph,
                        increment_reference,
                        state_reference,
                        block_vertices[increment_start // 5],
                        block_vertices[state_start // 5],
                    )
                    exact_affine_references[stage].setdefault(
                        active, []
                    ).append(
                        (increment_reference.copy(), state_reference.copy())
                    )
                    additions += 1
            for active in np.argsort(values)[:3]:
                active = int(active)
                if result.x[epigraph] <= values[active] + 1e-9:
                    continue
                cut = cuts[active]
                partition_reference = max(
                    float(
                        np.sum(
                            cut.kernel
                            * np.maximum(products[stage], 0.0)
                        )
                    ),
                    1e-300,
                )
                if all(
                    abs(np.log(partition_reference / old_reference)) > 1e-10
                    for old_reference in references[active]
                ):
                    references[active].append(partition_reference)
                    add_exact_cut_tangent(
                        cut,
                        stage,
                        increment_start,
                        state_start,
                        epigraph,
                        partition_reference,
                    )
                    additions += 1
        if additions == 0:
            break
        result = solve()
        if not result.success:
            return None

    point = result.x[0:COMPOSITION_SIZE]
    widths = np.array(box.upper) - np.array(box.lower)
    products = extract_products(result)
    first_relaxed = exact_relaxed_stage_values(
        first_cuts, first_kernels, 0, 0, 5, point, products
    )
    second_relaxed = exact_relaxed_stage_values(
        second_cuts, second_kernels, 1, 5, 10, point, products
    )
    active_first = int(np.argmin(first_relaxed))
    active_second = int(np.argmin(second_relaxed))
    active_matrices = [
        first_kernels[active_first],
        second_kernels[active_second],
    ]
    if exact_first_cuts:
        exact_values = exact_stationary_relaxed_values(
            exact_first_cuts, 0, 0, 5, point, products
        )
        exact_active = int(np.argmin(exact_values))
        if exact_values[exact_active] < first_relaxed[active_first]:
            active_matrices[0] = exact_first_cuts[exact_active].kernel
    if exact_second_cuts:
        exact_values = exact_stationary_relaxed_values(
            exact_second_cuts, 1, 5, 10, point, products
        )
        exact_active = int(np.argmin(exact_values))
        if exact_values[exact_active] < second_relaxed[active_second]:
            active_matrices[1] = exact_second_cuts[exact_active].kernel
    coordinate_scores = np.zeros(15)
    for stage, matrix, left_start, right_start in (
        (0, active_matrices[0], 0, 5),
        (1, active_matrices[1], 5, 10),
    ):
        partition_reference = max(
            float(np.sum(matrix * np.maximum(products[stage], 0.0))),
            1e-300,
        )
        for left in range(5):
            for right in range(5):
                product_gap = max(
                    0.0,
                    products[stage, left, right]
                    - point[left_start + left] * point[right_start + right],
                )
                weighted = (
                    stage_objective_weight
                    * matrix[left, right]
                    * product_gap
                    / partition_reference
                )
                coordinate_scores[left_start + left] += weighted * widths[
                    left_start + left
                ]
                coordinate_scores[right_start + right] += weighted * widths[
                    right_start + right
                ]
    entropy_gap = (
        float(np.dot(point[0:5], np.log(np.maximum(point[0:5], 1e-300) / message_reference)))
        / 3.0
    )
    coordinate_scores[0:5] += entropy_gap * widths[0:5] / max(
        float(widths[0:5].sum()), 1e-300
    )
    if float(np.max(coordinate_scores)) <= 1e-14:
        coordinate_scores = widths / np.maximum(0.01, point)
    dimension = int(np.argmax(coordinate_scores))
    diagnostic_point = sanitize_composition(point)
    exact_bank = (
        bulk_entropy(diagnostic_point[0:5]) / 3.0
        + stage_objective_weight * min(
            stage_upper_with_kernel(
                diagnostic_point[0:5], diagnostic_point[5:10], cut, matrix
            )
            for cut, matrix in zip(first_cuts, first_kernels, strict=True)
        )
        + stage_objective_weight * min(
            stage_upper_with_kernel(
                diagnostic_point[5:10], diagnostic_point[10:15], cut, matrix
            )
            for cut, matrix in zip(second_cuts, second_kernels, strict=True)
        )
    )
    relaxed_rank_bank = (
        bulk_entropy(point[0:5]) / 3.0
        + stage_objective_weight * float(first_relaxed[active_first])
        + stage_objective_weight * float(second_relaxed[active_second])
    )
    return -float(result.fun), dimension, point, box, exact_bank, relaxed_rank_bank


def split(
    box: Box,
    dimension: int,
    point: np.ndarray | None = None,
    rule: str = "midpoint",
) -> tuple[Box, Box]:
    midpoint = split_threshold(box, dimension, point, rule)
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return (
        Box(box.lower, tuple(left_upper)),
        Box(tuple(right_lower), box.upper),
    )


def split_threshold(
    box: Box,
    dimension: int,
    point: np.ndarray | None = None,
    rule: str = "midpoint",
) -> float:
    lower_value = box.lower[dimension]
    upper_value = box.upper[dimension]
    midpoint = (lower_value + upper_value) / 2.0
    if rule == "lp_clipped":
        if point is None:
            raise ValueError("lp_clipped splitting needs an LP point")
        quarter = (upper_value - lower_value) / 4.0
        midpoint = min(
            upper_value - quarter,
            max(lower_value + quarter, float(point[dimension])),
        )
    elif rule != "midpoint":
        raise ValueError(f"unknown split rule: {rule}")
    return midpoint


def branch_candidates(
    evaluated: tuple[object, ...], count: int
) -> list[int]:
    suggested = int(evaluated[1])
    box = evaluated[3]
    point = np.asarray(evaluated[2])
    widths = np.asarray(box.upper) - np.asarray(box.lower)
    # Wide coordinates with appreciable mass control the largest unresolved
    # products.  Keep the evaluator's active-kernel choice first, then use
    # this cheap proxy for strong-branch alternatives.
    proxy = widths * np.sqrt(np.maximum(point, 1e-6))
    ordered = [suggested]
    # On the positive F_5 frontier, the largest one-step rank-gap reductions
    # usually come from a nonzero coordinate of the shared middle or output
    # composition.  Put those eight coordinates in the probe set before the
    # generic width proxy.  The default one-candidate path is unchanged.
    ordered.extend(
        index
        for index in (*range(6, 10), *range(11, 15))
        if index != suggested
    )
    ordered.extend(
        int(index)
        for index in np.argsort(-proxy)
        if int(index) not in ordered
    )
    return ordered[: max(1, count)]


def encode_evaluated(value: tuple[object, ...]) -> dict[str, object]:
    return {
        "bound": float(value[0]),
        "dimension": int(value[1]),
        "point": value[2].tolist(),
        "lower": list(value[3].lower),
        "upper": list(value[3].upper),
        "exact_bank": float(value[4]),
        "rank_relaxed_bank": float(value[5]),
    }


def decode_evaluated(value: dict[str, object]) -> tuple[object, ...]:
    return (
        float(value["bound"]),
        int(value["dimension"]),
        np.asarray(value["point"], dtype=float),
        Box(tuple(value["lower"]), tuple(value["upper"])),
        float(value["exact_bank"]),
        float(value["rank_relaxed_bank"]),
    )


def checkpoint_signature(
    args: argparse.Namespace,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> dict[str, object]:
    signature = {
        "eta": args.eta,
        "delta": args.delta,
        "bank_digest": bank_digest(first_cuts, second_cuts),
    }
    if args.affine_bilinear_cuts > 0:
        signature["affine_bilinear_cuts"] = args.affine_bilinear_cuts
    return signature


def save_checkpoint(
    path: Path,
    signature: dict[str, object],
    splits_completed: int,
    identifier: int,
    history: list[float],
    queue: list[tuple[float, int, tuple[object, ...]]],
) -> None:
    write_json_atomic(
        path,
        {
            "version": 1,
            "signature": signature,
            "splits_completed": splits_completed,
            "identifier": identifier,
            "history": [float(value) for value in history],
            "queue": [
                {
                    "priority": float(priority),
                    "identifier": entry_identifier,
                    "evaluated": encode_evaluated(evaluated),
                }
                for priority, entry_identifier, evaluated in queue
            ],
        },
    )


def load_checkpoint(
    path: Path, signature: dict[str, object]
) -> tuple[
    int,
    int,
    list[float],
    list[tuple[float, int, tuple[object, ...]]],
]:
    with path.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    if saved.get("version") != 1 or saved.get("signature") != signature:
        raise RuntimeError(f"checkpoint does not match this run: {path}")
    queue = [
        (
            float(entry["priority"]),
            int(entry["identifier"]),
            decode_evaluated(entry["evaluated"]),
        )
        for entry in saved["queue"]
    ]
    heapq.heapify(queue)
    return (
        int(saved["splits_completed"]),
        int(saved["identifier"]),
        [float(value) for value in saved["history"]],
        queue,
    )


def migrate_checkpoint_to_stronger_bank(
    path: Path,
    old_signature: dict[str, object],
    new_signature: dict[str, object],
) -> None:
    with path.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    if saved.get("signature") != old_signature:
        raise RuntimeError(f"cannot migrate unexpected checkpoint: {path}")
    saved["signature"] = new_signature
    saved["inherited_from_weaker_bank"] = True
    write_json_atomic(path, saved)


def enrich_bank_at_composition(
    args: argparse.Namespace,
    point: np.ndarray,
    bank_result: dict[str, object],
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> dict[str, object]:
    point = sanitize_composition(point)
    old_first = len(first_cuts)
    old_second = len(second_cuts)
    old_value = composition_bank_upper(point, first_cuts, second_cuts)
    fitted_first = dual.fit_projection_dual(point[0:5], point[5:10])
    fitted_second = dual.fit_projection_dual(point[5:10], point[10:15])
    fitted_upper = (
        bulk_entropy(point[0:5]) / 3.0
        - fitted_first.value
        - fitted_second.value
    )
    added = old_value - fitted_upper > 1e-7
    if added:
        first_cuts.extend(envelope.cut_orbit(fitted_first))
        second_cuts.extend(envelope.cut_orbit(fitted_second))
    refinement = {
        "point": point.tolist(),
        "old_bank_upper": float(old_value),
        "locally_fitted_upper": float(fitted_upper),
        "first_gradient_residual": fitted_first.gradient_residual,
        "second_gradient_residual": fitted_second.gradient_residual,
        "first_cuts_before": old_first,
        "first_cuts_after": len(first_cuts),
        "second_cuts_before": old_second,
        "second_cuts_after": len(second_cuts),
        "added": added,
    }
    bank_result.setdefault("global_cover_refinements", []).append(refinement)
    return refinement


def composition_bank_upper(
    point: np.ndarray,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
    first_kernels: tuple[np.ndarray, ...] | None = None,
    second_kernels: tuple[np.ndarray, ...] | None = None,
) -> float:
    point = sanitize_composition(point)
    if first_kernels is None:
        first_kernels = tuple(envelope.kernel(cut) for cut in first_cuts)
    if second_kernels is None:
        second_kernels = tuple(envelope.kernel(cut) for cut in second_cuts)
    return float(
        bulk_entropy(point[0:5]) / 3.0
        + min(
            stage_upper_with_kernel(point[0:5], point[5:10], cut, matrix)
            for cut, matrix in zip(first_cuts, first_kernels, strict=True)
        )
        + min(
            stage_upper_with_kernel(point[5:10], point[10:15], cut, matrix)
            for cut, matrix in zip(second_cuts, second_kernels, strict=True)
        )
    )


def enrich_bank_from_frontier(
    args: argparse.Namespace,
    bank_result: dict[str, object],
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
) -> list[dict[str, object]]:
    checkpoint_path = Path(args.checkpoint)
    old_signature = checkpoint_signature(args, first_cuts, second_cuts)
    with checkpoint_path.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    if saved.get("signature") != old_signature:
        raise RuntimeError("frontier and bank do not match")
    candidates = []
    first_kernels = tuple(envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(envelope.kernel(cut) for cut in second_cuts)
    for entry in saved["queue"]:
        point = np.asarray(entry["evaluated"]["point"], dtype=float)
        candidates.append(
            (
                composition_bank_upper(
                    point,
                    first_cuts,
                    second_cuts,
                    first_kernels,
                    second_kernels,
                ),
                point,
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    target_upper = getattr(args, "enrich_target_upper", None)
    selected = []
    refinements = []
    for candidate_upper, point in candidates:
        if target_upper is not None and candidate_upper <= target_upper:
            break
        if all(float(np.abs(point - old).sum()) >= 0.05 for old in selected):
            selected.append(point)
            refinement = enrich_bank_at_composition(
                args, point, bank_result, first_cuts, second_cuts
            )
            refinements.append(refinement)
        if sum(bool(refinement["added"]) for refinement in refinements) >= args.enrich_frontier:
            break
    save_bank_cache(
        Path(args.bank_cache),
        bank_parameters(args),
        bank_result,
        first_cuts,
        second_cuts,
    )
    new_signature = checkpoint_signature(args, first_cuts, second_cuts)
    migrate_checkpoint_to_stronger_bank(
        checkpoint_path, old_signature, new_signature
    )
    return refinements


def cover(
    args: argparse.Namespace,
    first_cuts: list[dual.ProjectionDual],
    second_cuts: list[dual.ProjectionDual],
    exact_first_cuts: tuple[stationary.StationaryDualCut, ...] = (),
    exact_second_cuts: tuple[stationary.StationaryDualCut, ...] = (),
) -> dict[str, object]:
    first_kernels = tuple(envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(envelope.kernel(cut) for cut in second_cuts)
    checkpoint_path = Path(args.checkpoint)
    signature = checkpoint_signature(args, first_cuts, second_cuts)
    if args.resume and checkpoint_path.exists():
        splits_completed, identifier, history, queue = load_checkpoint(
            checkpoint_path, signature
        )
        print(
            json.dumps(
                {
                    "resumed_splits": splits_completed,
                    "upper": -queue[0][0],
                    "boxes": len(queue),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if splits_completed > args.max_boxes:
            raise RuntimeError(
                "checkpoint already exceeds requested --max-boxes"
            )
        if args.refresh_frontier > 0:
            refreshed_entries = []
            refresh_count = min(args.refresh_frontier, len(queue))
            for refresh_index in range(refresh_count):
                _, entry_identifier, current = heapq.heappop(queue)
                refreshed = evaluate(
                    current[3],
                    first_cuts,
                    second_cuts,
                    first_kernels,
                    second_kernels,
                    args.outer_iterations,
                    args.affine_bilinear_cuts,
                    exact_first_cuts,
                    exact_second_cuts,
                )
                if refreshed is None:
                    continue
                if refreshed[0] > current[0]:
                    refreshed = (current[0], *refreshed[1:])
                refreshed_entries.append(
                    (-refreshed[0], entry_identifier, refreshed)
                )
                if (
                    args.progress_every > 0
                    and (refresh_index + 1) % args.progress_every == 0
                ):
                    print(
                        json.dumps(
                            {
                                "frontier_refreshed": refresh_index + 1,
                                "frontier_target": refresh_count,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            queue.extend(refreshed_entries)
            heapq.heapify(queue)
            history[-1] = min(history[-1], -queue[0][0])
            save_checkpoint(
                checkpoint_path,
                signature,
                splits_completed,
                identifier,
                history,
                queue,
            )
            print(
                json.dumps(
                    {
                        "frontier_refreshed": refresh_count,
                        "upper": -queue[0][0],
                        "boxes": len(queue),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    else:
        initial = evaluate(
            initial_box(args.eta, args.delta),
            first_cuts,
            second_cuts,
            first_kernels,
            second_kernels,
            args.outer_iterations,
            args.affine_bilinear_cuts,
            exact_first_cuts,
            exact_second_cuts,
        )
        if initial is None:
            raise RuntimeError("initial composition box is infeasible")
        queue = [(-initial[0], 0, initial)]
        identifier = 1
        history = [initial[0]]
        splits_completed = 0
    split_log_path_value = getattr(args, "split_log", None)
    split_log_path = (
        Path(split_log_path_value) if split_log_path_value is not None else None
    )
    split_log = load_or_initialize_split_log(
        split_log_path, splits_completed, queue
    )
    for split_index in range(splits_completed, args.max_boxes):
        _, parent_identifier, current = heapq.heappop(queue)
        if current[0] < args.target_upper:
            save_checkpoint(
                checkpoint_path,
                signature,
                split_index,
                identifier,
                history,
                queue + [(-current[0], parent_identifier, current)],
            )
            save_split_log(split_log_path, split_log)
            return {
                "status": "target floating LP box cover reached",
                "upper": current[0],
                "splits": split_index,
                "boxes_remaining": len(queue) + 1,
                "history": history,
            }
        use_strong_branching = (
            args.strong_branch_candidates > 1
            and args.strong_branch_every > 0
            and split_index % args.strong_branch_every == 0
        )
        candidate_count = (
            args.strong_branch_candidates if use_strong_branching else 1
        )
        best_children = None
        best_child_upper = float("inf")
        best_dimension = None
        for dimension in branch_candidates(current, candidate_count):
            children = []
            for side, child in enumerate(split(
                current[3], dimension, current[2], getattr(args, "split_rule", "midpoint")
            )):
                bounded = evaluate(
                    child,
                    first_cuts,
                    second_cuts,
                    first_kernels,
                    second_kernels,
                    (
                        args.strong_branch_probe_iterations
                        if use_strong_branching
                        and args.strong_branch_probe_iterations >= 0
                        else args.outer_iterations
                    ),
                    args.affine_bilinear_cuts,
                    exact_first_cuts,
                    exact_second_cuts,
                )
                if bounded is not None:
                    # The parent's bound remains valid on each child.
                    if bounded[0] > current[0]:
                        bounded = (current[0], *bounded[1:])
                    children.append((side, bounded))
            if not children:
                continue
            child_upper = max(child[1][0] for child in children)
            if child_upper < best_child_upper:
                best_child_upper = child_upper
                best_children = children
                best_dimension = dimension
        if best_children is None:
            raise RuntimeError("every candidate split was infeasible")
        if (
            use_strong_branching
            and args.strong_branch_probe_iterations >= 0
        ):
            assert best_dimension is not None
            best_children = []
            for side, child in enumerate(split(
                current[3],
                best_dimension,
                current[2],
                getattr(args, "split_rule", "midpoint"),
            )):
                bounded = evaluate(
                    child,
                    first_cuts,
                    second_cuts,
                    first_kernels,
                    second_kernels,
                    args.outer_iterations,
                    args.affine_bilinear_cuts,
                    exact_first_cuts,
                    exact_second_cuts,
                )
                if bounded is not None:
                    if bounded[0] > current[0]:
                        bounded = (current[0], *bounded[1:])
                    best_children.append((side, bounded))
            if not best_children:
                raise RuntimeError("the selected probe split was infeasible")
        child_records = [
            {
                "side": side,
                "identifier": None,
                "raw_box": encode_box(child),
            }
            for side, child in enumerate(
                split(
                    current[3],
                    best_dimension,
                    current[2],
                    getattr(args, "split_rule", "midpoint"),
                )
            )
        ]
        for side, bounded in best_children:
            child_records[side]["identifier"] = identifier
            child_records[side]["feasible_box"] = encode_box(bounded[3])
            heapq.heappush(queue, (-bounded[0], identifier, bounded))
            identifier += 1
        history.append(-queue[0][0])
        completed = split_index + 1
        if split_log is not None:
            split_log["records"].append(
                {
                    "after_split": completed,
                    "parent_identifier": parent_identifier,
                    "parent_box": encode_box(current[3]),
                    "dimension": best_dimension,
                    "threshold": split_threshold(
                        current[3],
                        best_dimension,
                        current[2],
                        getattr(args, "split_rule", "midpoint"),
                    ),
                    "rule": getattr(args, "split_rule", "midpoint"),
                    "children": child_records,
                }
            )
        if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
            save_split_log(split_log_path, split_log)
            save_checkpoint(
                checkpoint_path,
                signature,
                completed,
                identifier,
                history,
                queue,
            )
        if (
            args.progress_every > 0
            and completed % args.progress_every == 0
        ):
            print(
                json.dumps(
                    {
                        "splits": completed,
                        "upper": -queue[0][0],
                        "boxes": len(queue),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    save_split_log(split_log_path, split_log)
    save_checkpoint(
        checkpoint_path,
        signature,
        args.max_boxes,
        identifier,
        history,
        queue,
    )
    worst = queue[0][2]
    compact_history = [
        {"splits": index, "upper": history[index]}
        for index in range(0, len(history), 25)
    ]
    if compact_history[-1]["splits"] != len(history) - 1:
        compact_history.append(
            {"splits": len(history) - 1, "upper": history[-1]}
        )
    return {
        "status": "floating LP box cover incomplete",
        "upper": worst[0],
        "splits": args.max_boxes,
        "boxes_remaining": len(queue),
        "history": compact_history,
        "worst_box": {
            "lower": list(worst[3].lower),
            "upper": list(worst[3].upper),
            "split_dimension": worst[1],
            "lp_composition": worst[2].tolist(),
            "exact_bank_upper_at_lp_composition": worst[4],
            "rank_relaxed_bank_upper_at_lp_composition": worst[5],
            "lp_relaxation_gap": worst[0] - worst[4],
            "rank_one_relaxation_gap": worst[5] - worst[4],
            "tangent_relaxation_gap": worst[0] - worst[5],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--max-boxes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--split-rule", choices=("midpoint", "lp_clipped"), default="midpoint")
    parser.add_argument("--strong-branch-candidates", type=int, default=1)
    parser.add_argument("--strong-branch-every", type=int, default=0)
    parser.add_argument(
        "--strong-branch-probe-iterations",
        type=int,
        default=-1,
        help=(
            "when nonnegative, rank candidate splits with this many outer "
            "rounds and fully reevaluate only the selected split"
        ),
    )
    parser.add_argument("--refresh-frontier", type=int, default=0)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=0)
    parser.add_argument("--target-upper", type=float, default=0.0)
    parser.add_argument(
        "--bank-cache",
        default="results/f5-projected-kl-dual-bank.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/f5-projected-kl-box-cover.checkpoint.json",
    )
    parser.add_argument("--split-log", type=Path)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rebuild-bank", action="store_true")
    parser.add_argument("--enrich-bank", action="store_true")
    parser.add_argument("--enrich-frontier", type=int, default=0)
    parser.add_argument(
        "--enrich-target-upper",
        type=float,
        default=None,
        help="stop frontier enrichment once the current bank is below this value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank_result, first_cuts, second_cuts = load_or_generate_bank(args)
    frontier_refinements = []
    if args.enrich_frontier > 0:
        frontier_refinements = enrich_bank_from_frontier(
            args, bank_result, first_cuts, second_cuts
        )
    result = cover(args, first_cuts, second_cuts)
    if frontier_refinements:
        result["frontier_bank_refinements"] = frontier_refinements
    if args.enrich_bank and "worst_box" in result:
        old_signature = checkpoint_signature(args, first_cuts, second_cuts)
        point = np.asarray(result["worst_box"]["lp_composition"], dtype=float)
        refinement = enrich_bank_at_composition(
            args, point, bank_result, first_cuts, second_cuts
        )
        save_bank_cache(
            Path(args.bank_cache),
            bank_parameters(args),
            bank_result,
            first_cuts,
            second_cuts,
        )
        new_signature = checkpoint_signature(args, first_cuts, second_cuts)
        migrate_checkpoint_to_stronger_bank(
            Path(args.checkpoint), old_signature, new_signature
        )
        result["bank_refinement"] = refinement
    result["bank_envelope_upper"] = bank_result["final_envelope_upper"]
    result["first_cuts"] = len(first_cuts)
    result["second_cuts"] = len(second_cuts)
    result["scope"] = (
        "mathematically valid LP relaxation; floating solver output is not an exact certificate"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
