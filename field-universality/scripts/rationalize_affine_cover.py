#!/usr/bin/env python3
"""Rationalize affine-gate partitions and certify their linear invariants.

This script proves only that every saved affine weight is nonnegative on the
boxed product of simplices and that the weights sum identically to one.  It
does not certify the exponential mixture inequality over the chamber.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected


def rational(value: Any) -> Fraction:
    return Fraction(str(value))


def exact_simplex_vertices(
    lower: list[Fraction], upper: list[Fraction], start: int
) -> list[tuple[Fraction, ...]]:
    block_lower = lower[start : start + 5]
    block_upper = upper[start : start + 5]
    vertices: set[tuple[Fraction, ...]] = set()
    for free in range(5):
        fixed = [index for index in range(5) if index != free]
        for mask in range(1 << 4):
            point: list[Fraction | None] = [None] * 5
            for offset, index in enumerate(fixed):
                point[index] = (
                    block_upper[index] if mask & (1 << offset) else block_lower[index]
                )
            point[free] = Fraction(1) - sum(
                (point[index] for index in fixed), start=Fraction(0)
            )
            if block_lower[free] <= point[free] <= block_upper[free]:
                vertices.add(tuple(value for value in point if value is not None))
    if not vertices:
        raise ValueError(f"boxed simplex block {start} has no exact vertices")
    return sorted(vertices)


def balanced_round_column(
    values: np.ndarray, target: int, denominator: int
) -> list[int]:
    """Round one coefficient column while preserving its exact target sum."""

    normalized = values + (target / denominator - float(np.sum(values))) / len(values)
    scaled = normalized * denominator
    floors = np.floor(scaled).astype(object)
    integers = [int(value) for value in floors]
    residual = target - sum(integers)
    fractions = scaled - np.floor(scaled)
    if residual >= 0:
        order = np.argsort(-fractions)
        sign = 1
    else:
        order = np.argsort(fractions)
        sign = -1
    for offset in range(abs(residual)):
        integers[int(order[offset % len(order)])] += sign
    if sum(integers) != target:
        raise AssertionError("balanced rounding did not preserve the target sum")
    return integers


def collect_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    if "leaf" in node:
        return [node]
    return [*collect_leaves(node["le"]), *collect_leaves(node["ge"])]


def chamber_bounds(
    base_lower: list[Fraction],
    base_upper: list[Fraction],
    restrictions: list[str],
) -> tuple[list[Fraction], list[Fraction]]:
    lower = list(base_lower)
    upper = list(base_upper)
    for restriction in restrictions:
        dimension_text, side, threshold_text = restriction.split(":")
        dimension = int(dimension_text)
        threshold = rational(threshold_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], threshold)
        elif side == "ge":
            lower[dimension] = max(lower[dimension], threshold)
        else:
            raise ValueError(f"invalid restriction {restriction!r}")
    return lower, upper


def feature_rows(
    blocks: list[list[tuple[Fraction, ...]]], coordinates: list[int]
) -> tuple[np.ndarray, int]:
    rows = []
    for product in itertools.product(*blocks):
        point = (*product[0], *product[1], *product[2])
        rows.append([1.0, *(float(point[index]) for index in coordinates)])
    return np.asarray(rows), len(rows)


def fit_robust_partition(
    original: np.ndarray, features: np.ndarray, margin: float
) -> tuple[np.ndarray, float]:
    experts, feature_count = original.shape
    coefficient_count = experts * feature_count
    # Variables are flattened coefficients followed by the L-infinity adjustment.
    nonnegative_blocks = []
    for expert in range(experts):
        left = sparse.csr_matrix((len(features), expert * feature_count))
        middle = sparse.csr_matrix(-features)
        right = sparse.csr_matrix(
            (len(features), coefficient_count - (expert + 1) * feature_count + 1)
        )
        nonnegative_blocks.append(sparse.hstack((left, middle, right), format="csr"))
    nonnegative = sparse.vstack(nonnegative_blocks, format="csr")
    identity = sparse.eye(coefficient_count, format="csr")
    adjustment_upper = sparse.hstack(
        (identity, -np.ones((coefficient_count, 1))), format="csr"
    )
    adjustment_lower = sparse.hstack(
        (-identity, -np.ones((coefficient_count, 1))), format="csr"
    )
    upper = sparse.vstack(
        (nonnegative, adjustment_upper, adjustment_lower), format="csr"
    )
    target = np.concatenate(
        (
            np.full(nonnegative.shape[0], -margin),
            original.ravel(),
            -original.ravel(),
        )
    )
    equality = np.zeros((feature_count, coefficient_count + 1))
    for feature in range(feature_count):
        equality[feature, feature:coefficient_count:feature_count] = 1.0
    equality_target = np.zeros(feature_count)
    equality_target[0] = 1.0
    objective = np.zeros(coefficient_count + 1)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=target,
        A_eq=equality,
        b_eq=equality_target,
        bounds=[(None, None)] * coefficient_count + [(0.0, None)],
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-10,
            "primal_feasibility_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:-1].reshape(experts, feature_count), float(result.x[-1])


def exact_minimum_weight(
    numerators: list[list[int]],
    denominator: int,
    blocks: list[list[tuple[Fraction, ...]]],
    coordinates: list[int],
) -> Fraction:
    coordinate_positions = {coordinate: position + 1 for position, coordinate in enumerate(coordinates)}
    overall = None
    for row in numerators:
        minimum = Fraction(row[0], denominator)
        for block_index, start in enumerate((0, 5, 10)):
            block_values = []
            for vertex in blocks[block_index]:
                value = Fraction(0)
                for offset, coordinate in enumerate(range(start, start + 5)):
                    position = coordinate_positions.get(coordinate)
                    if position is not None:
                        value += Fraction(row[position], denominator) * vertex[offset]
                block_values.append(value)
            minimum += min(block_values)
        overall = minimum if overall is None else min(overall, minimum)
    if overall is None:
        raise ValueError("partition has no experts")
    return overall


def effective_gate_reference(
    leaf: dict[str, Any], result: dict[str, Any], *, result_override: bool
) -> str:
    """Choose the chamber descriptor that actually produced the coefficients."""

    if result_override:
        if not result.get("gate"):
            raise ValueError("overridden continuous result does not identify its gate")
        return str(result["gate"])
    return str(leaf["gate"])


def effective_feature_coordinates(result: dict[str, Any], gate: dict[str, Any]) -> list[int]:
    """Recover the feature basis that produced the saved coefficients."""

    raw = result.get(
        "feature_coordinates", gate["affine_gating"]["feature_coordinates"]
    )
    coordinates = [int(value) for value in raw]
    if len(set(coordinates)) != len(coordinates) or any(
        not 0 <= value < 15 for value in coordinates
    ):
        raise ValueError("invalid affine-gate feature coordinates")
    return coordinates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--leaf", action="append", default=[])
    parser.add_argument(
        "--result-override",
        type=Path,
        help="use this continuous-gate result when exactly one leaf is selected",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="fix a coordinate to its chamber lower or upper endpoint",
    )
    parser.add_argument("--denominator", type=int, default=10**12)
    parser.add_argument("--robust-margin", type=float, default=1e-8)
    parser.add_argument(
        "--prune-coefficient-norm",
        type=float,
        default=0.0,
        help="drop floating expert rows whose maximum absolute coefficient is at most this threshold",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.denominator <= 0:
        parser.error("--denominator must be positive")
    if not math.isfinite(args.robust_margin) or args.robust_margin <= 0.0:
        parser.error("--robust-margin must be finite and positive")
    if (
        not math.isfinite(args.prune_coefficient_norm)
        or args.prune_coefficient_norm < 0.0
    ):
        parser.error("--prune-coefficient-norm must be finite and nonnegative")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / manifest.get("artifact_root", "..")).resolve()
    frontier_path = root / manifest["global_checkpoint"]
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    base_lower = [rational(value) for value in selected["lower"]]
    base_upper = [rational(value) for value in selected["upper"]]

    wanted = set(args.leaf)
    leaves = [
        leaf for leaf in collect_leaves(manifest["tree"])
        if not wanted or leaf["leaf"] in wanted
    ]
    if wanted != {leaf["leaf"] for leaf in leaves} and wanted:
        missing = sorted(wanted - {leaf["leaf"] for leaf in leaves})
        parser.error(f"unknown leaves: {', '.join(missing)}")
    if args.result_override is not None and len(leaves) != 1:
        parser.error("--result-override requires exactly one selected leaf")
    records = []
    for leaf in leaves:
        result_path = (
            args.result_override.resolve()
            if args.result_override is not None
            else root / str(leaf["result"]).replace("\\", "/")
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        gate_reference = effective_gate_reference(
            leaf, result, result_override=args.result_override is not None
        )
        gate_path = root / gate_reference.replace("\\", "/")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        coefficients = np.asarray(result["final_coefficients"], dtype=float)
        pairs = result["coefficient_expert_pairs"]
        if len(coefficients) != len(pairs):
            raise ValueError(f"{leaf['leaf']}: coefficient/expert length mismatch")
        original_experts = len(pairs)
        if args.prune_coefficient_norm > 0.0:
            retained = np.max(np.abs(coefficients), axis=1) > args.prune_coefficient_norm
            coefficients = coefficients[retained]
            pairs = [pair for pair, keep in zip(pairs, retained, strict=True) if keep]
            if not pairs:
                raise ValueError(f"{leaf['leaf']}: pruning removed every expert")
        coordinates = effective_feature_coordinates(result, gate)
        lower, upper = chamber_bounds(base_lower, base_upper, gate["restrictions"])
        for specification in args.fix:
            coordinate_text, side = specification.split(":")
            coordinate = int(coordinate_text)
            if not 0 <= coordinate < 15:
                parser.error("--fix coordinate must be between 0 and 14")
            if side == "lower":
                upper[coordinate] = lower[coordinate]
            elif side == "upper":
                lower[coordinate] = upper[coordinate]
            else:
                parser.error("--fix side must be lower or upper")
        blocks = [exact_simplex_vertices(lower, upper, start) for start in (0, 5, 10)]
        features, vertex_count = feature_rows(blocks, coordinates)
        fitted, adjustment = fit_robust_partition(
            coefficients, features, args.robust_margin
        )
        columns = []
        for feature in range(fitted.shape[1]):
            columns.append(
                balanced_round_column(
                    fitted[:, feature],
                    args.denominator if feature == 0 else 0,
                    args.denominator,
                )
            )
        numerators = [list(row) for row in zip(*columns, strict=True)]
        minimum = exact_minimum_weight(
            numerators, args.denominator, blocks, coordinates
        )
        if minimum < 0:
            raise RuntimeError(
                f"{leaf['leaf']}: rationalized partition has negative weight {minimum}"
            )
        record = {
            "leaf": leaf["leaf"],
            "gate": gate_reference,
            "result": str(result_path),
            "fixed": args.fix,
            "expert_pairs": pairs,
            "feature_coordinates": coordinates,
            "coefficient_denominator": str(args.denominator),
            "coefficient_numerators": numerators,
            "product_vertex_count": vertex_count,
            "maximum_coefficient_adjustment": adjustment,
            "minimum_exact_weight": f"{minimum.numerator}/{minimum.denominator}",
            "partition_identity_exact": True,
            "nonnegative_on_product_box_exact": True,
            "floating_experts_before_pruning": original_experts,
            "pruned_floating_experts": original_experts - len(pairs),
            "prune_coefficient_norm": args.prune_coefficient_norm,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "leaf": leaf["leaf"],
                    "experts": len(pairs),
                    "vertices": vertex_count,
                    "adjustment": adjustment,
                    "minimum_exact_weight": record["minimum_exact_weight"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "status": "exact rational affine partition only; analytic mixture bound pending",
        "manifest": str(args.manifest),
        "coefficient_denominator": str(args.denominator),
        "robust_margin": args.robust_margin,
        "fixed": args.fix,
        "prune_coefficient_norm": args.prune_coefficient_norm,
        "leaves": records,
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "leaves": len(records),
                "worst_adjustment": max(record["maximum_coefficient_adjustment"] for record in records),
                "minimum_exact_weight": min(
                    rational(record["minimum_exact_weight"]) for record in records
                ).__str__(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
