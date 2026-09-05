#!/usr/bin/env python3
"""Sample one tangent Hessian row of a rational affine-gate log mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_projected_kl_box_cover as projected
from diagnose_affine_gate_monotonicity import (
    log_mixture_gradients,
    random_product_points,
    tangent_directions,
)
from optimize_affine_exponential_gate import domain_matrices
from rationalize_affine_cover import collect_leaves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--coordinate", type=int, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--step", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--fix", action="append", default=[], metavar="DIM:SIDE")
    parser.add_argument("--dependent-coordinates", default="3,5,12")
    parser.add_argument(
        "--bank", type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--frontier", type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.step <= 0.0:
        parser.error("sample counts and step must be positive")
    dependent = [int(value) for value in args.dependent_coordinates.split(",")]
    if args.coordinate in dependent or not 0 <= args.coordinate < 15:
        parser.error("--coordinate must select an explicit simplex coordinate")
    target_dependent = dependent[args.coordinate // 5]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[args.leaf]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(value for value in partitions["leaves"] if value["leaf"] == args.leaf)
    denominator = float(int(record["coefficient_denominator"]))
    coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]
    pair_first = [int(pair["first_cut"]) for pair in record["expert_pairs"]]
    pair_second = [int(pair["second_cut"]) for pair in record["expert_pairs"]]

    bank_args = SimpleNamespace(
        eta=0.002, delta=0.22, bank_iterations=20, bank_restarts=12,
        seed=11, bank_cache=str(args.bank), rebuild_bank=False,
    )
    _, all_first, all_second = projected.load_or_generate_bank(bank_args)
    all_first_kernels = tuple(projected.envelope.kernel(cut) for cut in all_first)
    all_second_kernels = tuple(projected.envelope.kernel(cut) for cut in all_second)
    first_cuts = [all_first[index] for index in pair_first]
    second_cuts = [all_second[index] for index in pair_second]
    first_kernels = np.stack([all_first_kernels[index] for index in pair_first])
    second_kernels = np.stack([all_second_kernels[index] for index in pair_second])

    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    lower = np.asarray(selected["lower"], dtype=float)
    upper = np.asarray(selected["upper"], dtype=float)
    gate_path = args.gate if args.gate is not None else Path(str(record.get("gate", leaf["gate"])))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for restriction in gate["restrictions"]:
        dimension_text, side, threshold_text = restriction.split(":")
        dimension = int(dimension_text)
        threshold = float(threshold_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], threshold)
        elif side == "ge":
            lower[dimension] = max(lower[dimension], threshold)
        else:
            raise ValueError(f"invalid gate restriction {restriction!r}")
    for specification in args.fix:
        dimension_text, side = specification.split(":")
        dimension = int(dimension_text)
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            parser.error("--fix side must be lower or upper")
    directions = [
        direction
        for direction in tangent_directions(dependent)
        if lower[direction] < upper[direction]
    ]
    box = projected.Box(tuple(lower), tuple(upper))
    block_vertices = [projected.simplex_box_vertices(box, start) for start in (0, 5, 10)]
    _, _, domain_upper, domain_target = domain_matrices()

    minima = np.full(len(directions), np.inf)
    maxima = np.full(len(directions), -np.inf)
    minimum_points: list[list[float] | None] = [None] * len(directions)
    maximum_points: list[list[float] | None] = [None] * len(directions)
    evaluated = np.zeros(len(directions), dtype=int)
    random = np.random.default_rng(args.seed)
    remaining = args.samples
    while remaining:
        count = min(args.batch_size, remaining)
        points = random_product_points(block_vertices, count, random)
        points = points[
            np.all(domain_upper @ points.T <= domain_target[:, None] + 1e-12, axis=0)
        ]
        for offset, direction in enumerate(directions):
            direction_dependent = dependent[direction // 5]
            vector = np.zeros(15)
            vector[direction] = 1.0
            vector[direction_dependent] = -1.0
            coordinate_margin = np.minimum.reduce(
                (
                    points[:, direction] - lower[direction],
                    upper[direction] - points[:, direction],
                    points[:, direction_dependent] - lower[direction_dependent],
                    upper[direction_dependent] - points[:, direction_dependent],
                )
            )
            domain_slack = domain_target[None, :] - points @ domain_upper.T
            domain_slope = np.abs(domain_upper @ vector)
            active = domain_slope > 0.0
            if np.any(active):
                domain_margin = np.min(
                    domain_slack[:, active] / domain_slope[active][None, :], axis=1
                )
                margin = np.minimum(coordinate_margin, domain_margin)
            else:
                margin = coordinate_margin
            steps = np.minimum(args.step, margin / 4.0)
            keep = steps > args.step * 1e-3
            if not np.any(keep):
                continue
            selected_points = points[keep]
            selected_steps = steps[keep]
            plus = selected_points + selected_steps[:, None] * vector[None, :]
            minus = selected_points - selected_steps[:, None] * vector[None, :]
            plus_gradients, _ = log_mixture_gradients(
                plus, coefficients, feature_coordinates, first_cuts, second_cuts,
                first_kernels, second_kernels,
            )
            minus_gradients, _ = log_mixture_gradients(
                minus, coefficients, feature_coordinates, first_cuts, second_cuts,
                first_kernels, second_kernels,
            )
            plus_target = plus_gradients[:, args.coordinate] - plus_gradients[:, target_dependent]
            minus_target = minus_gradients[:, args.coordinate] - minus_gradients[:, target_dependent]
            values = (plus_target - minus_target) / (2.0 * selected_steps)
            low_index = int(np.argmin(values))
            high_index = int(np.argmax(values))
            if values[low_index] < minima[offset]:
                minima[offset] = float(values[low_index])
                minimum_points[offset] = selected_points[low_index].tolist()
            if values[high_index] > maxima[offset]:
                maxima[offset] = float(values[high_index])
                maximum_points[offset] = selected_points[high_index].tolist()
            evaluated[offset] += len(values)
        remaining -= count

    payload = {
        "status": "floating central-difference tangent Hessian diagnostic; not a proof",
        "leaf": args.leaf,
        "gate": str(gate_path),
        "fixed_faces": args.fix,
        "dependent_coordinates": dependent,
        "directional_coordinate": args.coordinate,
        "directional_dependent_coordinate": target_dependent,
        "requested_samples": args.samples,
        "central_step_cap": args.step,
        "directions": [
            {
                "coordinate": direction,
                "evaluated_samples": int(evaluated[offset]),
                "minimum_mixed_derivative": float(minima[offset]),
                "maximum_mixed_derivative": float(maxima[offset]),
                "sampled_fixed_sign": bool(minima[offset] > 0.0 or maxima[offset] < 0.0),
                "minimum_point": minimum_points[offset],
                "maximum_point": maximum_points[offset],
            }
            for offset, direction in enumerate(directions)
        ],
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
