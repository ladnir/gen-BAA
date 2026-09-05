#!/usr/bin/env python3
"""Audit whether one floating affine cover transfers to other frontier cells.

The audit reuses the saved affine coefficients and expert pairs.  It checks
only product vertices of each induced chamber.  A positive value disproves
direct transfer of that saved gate; a negative value is necessary but not
sufficient for continuous closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import explore_projected_kl_box_cover as projected
from search_continuous_affine_gate import entropy_rows, stage_upper_rows


def coefficient_array(raw: Any) -> np.ndarray:
    """Decode both current numeric rows and legacy space-separated rows."""

    if not isinstance(raw, list) or not raw:
        raise ValueError("affine coefficients must be a nonempty list")
    if isinstance(raw[0], str):
        return np.asarray([[float(value) for value in row.split()] for row in raw])
    return np.asarray(raw, dtype=float)


def induced_leaves(
    node: dict[str, Any], lower: np.ndarray, upper: np.ndarray
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    """Route a cell through a split tree, assigning split boundaries once."""

    if "leaf" in node:
        return [(node, lower, upper)]
    split = node["split"]
    dimension = int(split["dimension"])
    numerator, denominator = str(split["threshold"]).split("/")
    threshold = int(numerator) / int(denominator)
    if upper[dimension] <= threshold:
        return induced_leaves(node["le"], lower, upper)
    if lower[dimension] >= threshold:
        return induced_leaves(node["ge"], lower, upper)
    le_upper = upper.copy()
    le_upper[dimension] = threshold
    ge_lower = lower.copy()
    ge_lower[dimension] = threshold
    return [
        *induced_leaves(node["le"], lower.copy(), le_upper),
        *induced_leaves(node["ge"], ge_lower, upper.copy()),
    ]


def product_vertices(box: projected.Box) -> np.ndarray:
    blocks = [projected.simplex_box_vertices(box, start) for start in (0, 5, 10)]
    counts = [len(block) for block in blocks]
    message = np.repeat(blocks[0], counts[1] * counts[2], axis=0)
    middle = np.tile(np.repeat(blocks[1], counts[2], axis=0), (counts[0], 1))
    output = np.tile(blocks[2], (counts[0] * counts[1], 1))
    return np.concatenate((message, middle, output), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
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
    parser.add_argument("--cells", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells <= 0:
        parser.error("--cells must be positive")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / manifest.get("artifact_root", "..")).resolve()
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    ordered = sorted(frontier["queue"], key=lambda entry: entry["priority"])
    if args.cells > len(ordered):
        parser.error("--cells exceeds the saved frontier size")

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
    gate_cache: dict[str, dict[str, Any]] = {}
    records = []

    for frontier_index, entry in enumerate(ordered[: args.cells]):
        evaluated = entry["evaluated"]
        lower = np.asarray(evaluated["lower"], dtype=float)
        upper = np.asarray(evaluated["upper"], dtype=float)
        leaves = induced_leaves(manifest["tree"], lower, upper)
        leaf_records = []
        infeasible_leaves = 0
        cell_worst = float("-inf")
        cell_minimum_weight = float("inf")
        cell_partition_residual = 0.0
        for leaf, leaf_lower, leaf_upper in leaves:
            gate_relative = str(leaf["gate"]).replace("\\", "/")
            if gate_relative not in gate_cache:
                gate_path = root / gate_relative
                gate_cache[gate_relative] = json.loads(
                    gate_path.read_text(encoding="utf-8")
                )
            gate = gate_cache[gate_relative]
            affine = gate["affine_gating"]
            pairs = affine["expert_pairs"]
            coefficients = coefficient_array(affine["coefficients"])
            coordinates = [int(value) for value in affine["feature_coordinates"]]
            try:
                points = product_vertices(
                    projected.Box(tuple(leaf_lower), tuple(leaf_upper))
                )
            except RuntimeError as exc:
                if str(exc) != "boxed simplex has no vertices":
                    raise
                infeasible_leaves += 1
                continue
            features = np.concatenate(
                (np.ones((len(points), 1)), points[:, coordinates]), axis=1
            )
            weights = features @ coefficients.T
            first_indices = [int(pair["first_cut"]) for pair in pairs]
            second_indices = [int(pair["second_cut"]) for pair in pairs]
            first_values = stage_upper_rows(
                points[:, 0:5],
                points[:, 5:10],
                [first_cuts[index] for index in first_indices],
                np.stack([first_kernels[index] for index in first_indices]),
            )
            second_values = stage_upper_rows(
                points[:, 5:10],
                points[:, 10:15],
                [second_cuts[index] for index in second_indices],
                np.stack([second_kernels[index] for index in second_indices]),
            )
            exponentials = np.exp(
                entropy_rows(points[:, 0:5])[:, None] / 3.0
                + first_values
                + second_values
            )
            mixture = np.sum(weights * exponentials, axis=1)
            maximum = float(np.max(mixture))
            log_maximum = float(np.log(maximum)) if maximum > 0.0 else float("-inf")
            minimum_weight = float(np.min(weights))
            partition_residual = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
            cell_worst = max(cell_worst, log_maximum)
            cell_minimum_weight = min(cell_minimum_weight, minimum_weight)
            cell_partition_residual = max(cell_partition_residual, partition_residual)
            leaf_records.append(
                {
                    "leaf": leaf["leaf"],
                    "vertices": len(points),
                    "log_maximum_vertex_mixture": log_maximum,
                    "minimum_gating_weight": minimum_weight,
                    "maximum_partition_residual": partition_residual,
                }
            )
        record = {
            "frontier_index": frontier_index,
            "source_identifier": int(entry["identifier"]),
            "induced_leaves": len(leaves) - infeasible_leaves,
            "infeasible_leaves_pruned": infeasible_leaves,
            "worst_log_maximum_vertex_mixture": cell_worst,
            "minimum_gating_weight": cell_minimum_weight,
            "maximum_partition_residual": cell_partition_residual,
            "passes_vertex_margin": (
                cell_worst
                <= -float(manifest["target_margin"].split("/")[0])
                / float(manifest["target_margin"].split("/")[1])
                and cell_minimum_weight >= -1e-8
                and cell_partition_residual <= 1e-8
            ),
            "leaves": leaf_records,
        }
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "leaves"}, sort_keys=True), flush=True)

    payload = {
        "status": "floating transferred-gate vertex audit; negative is not continuous closure",
        "manifest": str(args.manifest),
        "cells": args.cells,
        "passing_cells": sum(record["passes_vertex_margin"] for record in records),
        "worst_log_maximum_vertex_mixture": max(
            record["worst_log_maximum_vertex_mixture"] for record in records
        ),
        "records": records,
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
