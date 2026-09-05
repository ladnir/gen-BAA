#!/usr/bin/env python3
"""Sample tangent-gradient signs of one rational affine gate efficiently."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_projected_kl_box_cover as projected
from optimize_affine_exponential_gate import domain_matrices


def tangent_directions(dependent: list[int]) -> list[int]:
    if (
        len(dependent) != 3
        or any(
            not start <= coordinate < start + 5
            for start, coordinate in zip((0, 5, 10), dependent, strict=True)
        )
    ):
        raise ValueError("choose one dependent coordinate from each simplex block")
    return [
        coordinate
        for start, eliminated in zip((0, 5, 10), dependent, strict=True)
        for coordinate in range(start, start + 5)
        if coordinate != eliminated
    ]
from rationalize_affine_cover import collect_leaves
from search_continuous_affine_gate import entropy_rows, stage_upper_rows


def random_product_points(
    block_vertices: list[np.ndarray], samples: int, random: np.random.Generator
) -> np.ndarray:
    blocks = []
    for vertices in block_vertices:
        weights = random.dirichlet(np.ones(len(vertices)), size=samples)
        blocks.append(weights @ vertices)
    return np.concatenate(blocks, axis=1)


def stage_log_gradients(
    left: np.ndarray, right: np.ndarray, cuts: list, kernels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ambient gradients for one stage, shaped (points, experts, 10)."""

    partitions = np.einsum("ni,eij,nj->ne", left, kernels, right, optimize=True)
    left_gradient = np.einsum("eij,nj->nei", kernels, right, optimize=True)
    left_gradient /= partitions[:, :, None]
    right_gradient = np.einsum("eij,ni->nej", kernels, left, optimize=True)
    right_gradient /= partitions[:, :, None]
    target = np.stack([cut.target_potential for cut in cuts])
    increment_support = np.asarray([cut.increment_support_potential for cut in cuts])
    state_support = np.asarray([cut.state_support_potential for cut in cuts])
    left_gradient[:, :, 1:5] -= increment_support[None, :, None]
    right_gradient -= target[None, :, :]
    right_gradient[:, :, 1:5] -= state_support[None, :, None]
    return left_gradient, right_gradient


def log_mixture_gradients(
    points: np.ndarray,
    coefficients: np.ndarray,
    coordinates: list[int],
    first_cuts: list,
    second_cuts: list,
    first_kernels: np.ndarray,
    second_kernels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate((np.ones((len(points), 1)), points[:, coordinates]), axis=1)
    weights = features @ coefficients.T
    first_values = stage_upper_rows(
        points[:, 0:5], points[:, 5:10], first_cuts, first_kernels
    )
    second_values = stage_upper_rows(
        points[:, 5:10], points[:, 10:15], second_cuts, second_kernels
    )
    logs = entropy_rows(points[:, 0:5])[:, None] / 3.0 + first_values + second_values
    logs -= np.max(logs, axis=1, keepdims=True)
    exponentials = np.exp(logs)
    ambient_weight_gradients = np.zeros((len(coefficients), 15))
    for feature, coordinate in enumerate(coordinates, start=1):
        ambient_weight_gradients[:, coordinate] = coefficients[:, feature]
    gradients = np.zeros((len(points), len(coefficients), 15))
    gradients[:, :, 0:5] = -(np.log(points[:, None, 0:5]) + 1.0) / 3.0
    first_left, first_right = stage_log_gradients(
        points[:, 0:5], points[:, 5:10], first_cuts, first_kernels
    )
    second_left, second_right = stage_log_gradients(
        points[:, 5:10], points[:, 10:15], second_cuts, second_kernels
    )
    gradients[:, :, 0:5] += first_left
    gradients[:, :, 5:10] += first_right + second_left
    gradients[:, :, 10:15] += second_right
    terms = weights * exponentials
    mixture = np.sum(terms, axis=1)
    expert_numerators = exponentials[:, :, None] * (
        ambient_weight_gradients[None, :, :] + weights[:, :, None] * gradients
    )
    numerator = np.sum(expert_numerators, axis=1)
    return numerator / mixture[:, None], expert_numerators


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument(
        "--gate",
        type=Path,
        help="override the rational partition record's gate descriptor",
    )
    parser.add_argument(
        "--include-product-vertices",
        action="store_true",
        help="also evaluate every feasible product-simplex vertex",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="restrict a coordinate to its current lower or upper bound",
    )
    parser.add_argument(
        "--dependent-coordinates",
        default="0,5,10",
        help="comma-separated simplex coordinates eliminated in tangent derivatives",
    )
    parser.add_argument(
        "--bank", type=Path, default=Path("results/f5-projected-kl-dual-bank-0.22.json")
    )
    parser.add_argument(
        "--frontier", type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("sample counts must be positive")
    dependent = [int(value) for value in args.dependent_coordinates.split(",")]
    try:
        directions = tangent_directions(dependent)
    except ValueError as error:
        parser.error(str(error))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[args.leaf]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(record for record in partitions["leaves"] if record["leaf"] == args.leaf)
    denominator = float(int(record["coefficient_denominator"]))
    coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
    coordinates = [int(value) for value in record["feature_coordinates"]]
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
    lower = list(selected["lower"])
    upper = list(selected["upper"])
    gate_path = (
        args.gate
        if args.gate is not None
        else Path(str(record.get("gate", leaf["gate"])))
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for restriction in gate["restrictions"]:
        dimension_text, side, threshold_text = restriction.split(":")
        dimension = int(dimension_text)
        threshold = float(threshold_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], threshold)
        else:
            lower[dimension] = max(lower[dimension], threshold)
    for specification in args.fix:
        dimension_text, side = specification.split(":")
        dimension = int(dimension_text)
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            parser.error("--fix side must be lower or upper")
    box = projected.Box(tuple(lower), tuple(upper))
    block_vertices = [projected.simplex_box_vertices(box, start) for start in (0, 5, 10)]
    _, _, domain_upper, domain_target = domain_matrices()
    minima = np.full(len(directions), np.inf)
    maxima = np.full(len(directions), -np.inf)
    pairwise_minima = np.full((len(directions), len(directions)), np.inf)
    pairwise_maxima = np.full((len(directions), len(directions)), -np.inf)
    expert_minima = np.full((len(coefficients), len(directions)), np.inf)
    expert_maxima = np.full((len(coefficients), len(directions)), -np.inf)
    feasible = 0
    random = np.random.default_rng(args.seed)
    remaining = args.samples
    while remaining:
        count = min(args.batch_size, remaining)
        points = random_product_points(block_vertices, count, random)
        points = points[np.all(domain_upper @ points.T <= domain_target[:, None] + 1e-12, axis=0)]
        if len(points):
            gradients, expert_numerators = log_mixture_gradients(
                points, coefficients, coordinates, first_cuts, second_cuts,
                first_kernels, second_kernels,
            )
            tangent = np.stack(
                [
                    gradients[:, dimension]
                    - gradients[:, dependent[dimension // 5]]
                    for dimension in directions
                ],
                axis=1,
            )
            expert_tangent = np.stack(
                [
                    expert_numerators[:, :, dimension]
                    - expert_numerators[:, :, dependent[dimension // 5]]
                    for dimension in directions
                ],
                axis=2,
            )
            expert_minima = np.minimum(expert_minima, np.min(expert_tangent, axis=0))
            expert_maxima = np.maximum(expert_maxima, np.max(expert_tangent, axis=0))
            minima = np.minimum(minima, np.min(tangent, axis=0))
            maxima = np.maximum(maxima, np.max(tangent, axis=0))
            differences = tangent[:, :, None] - tangent[:, None, :]
            pairwise_minima = np.minimum(pairwise_minima, np.min(differences, axis=0))
            pairwise_maxima = np.maximum(pairwise_maxima, np.max(differences, axis=0))
            feasible += len(points)
        remaining -= count
    product_vertex_ranges = None
    if args.include_product_vertices:
        product_points = np.asarray(
            [
                np.concatenate(product)
                for product in itertools.product(*block_vertices)
            ],
            dtype=float,
        )
        product_points = product_points[
            np.all(
                domain_upper @ product_points.T
                <= domain_target[:, None] + 1e-12,
                axis=0,
            )
        ]
        product_gradients, _ = log_mixture_gradients(
            product_points,
            coefficients,
            coordinates,
            first_cuts,
            second_cuts,
            first_kernels,
            second_kernels,
        )
        product_tangent = np.stack(
            [
                product_gradients[:, dimension]
                - product_gradients[:, dependent[dimension // 5]]
                for dimension in directions
            ],
            axis=1,
        )
        product_vertex_ranges = [
            {
                "coordinate": dimension,
                "minimum_tangent_derivative": float(np.min(product_tangent[:, offset])),
                "maximum_tangent_derivative": float(np.max(product_tangent[:, offset])),
                "minimum_point": product_points[
                    int(np.argmin(product_tangent[:, offset]))
                ].tolist(),
                "maximum_point": product_points[
                    int(np.argmax(product_tangent[:, offset]))
                ].tolist(),
            }
            for offset, dimension in enumerate(directions)
        ]
    payload = {
        "status": "floating sampled tangent-gradient diagnostic; not a sign proof",
        "leaf": args.leaf,
        "gate": str(gate_path),
        "fixed_faces": args.fix,
        "dependent_coordinates": dependent,
        "requested_samples": args.samples,
        "feasible_samples": feasible,
        "feasible_product_vertices": (
            len(product_points) if args.include_product_vertices else None
        ),
        "product_vertex_directions": product_vertex_ranges,
        "directions": [
            {
                "coordinate": dimension,
                "minimum_tangent_derivative": float(minimum),
                "maximum_tangent_derivative": float(maximum),
                "sampled_fixed_sign": bool(minimum > 0.0 or maximum < 0.0),
            }
            for dimension, minimum, maximum in zip(directions, minima, maxima, strict=True)
        ],
        "expert_term_directions": [
            {
                "coordinate": dimension,
                "minimum_over_samples_and_experts": float(np.min(expert_minima[:, offset])),
                "maximum_over_samples_and_experts": float(np.max(expert_maxima[:, offset])),
                "experts_sampled_positive": int(np.sum(expert_minima[:, offset] > 0.0)),
                "experts_sampled_negative": int(np.sum(expert_maxima[:, offset] < 0.0)),
                "experts": len(coefficients),
                "nonpositive_experts": [
                    int(index)
                    for index in range(len(coefficients))
                    if expert_minima[index, offset] <= 0.0
                ],
            }
            for offset, dimension in enumerate(directions)
        ],
        "sampled_fixed_pairwise_orderings": [
            {
                "larger_coordinate": directions[left] if pairwise_minima[left, right] > 0.0 else directions[right],
                "smaller_coordinate": directions[right] if pairwise_minima[left, right] > 0.0 else directions[left],
                "minimum_derivative_gap": float(
                    pairwise_minima[left, right]
                    if pairwise_minima[left, right] > 0.0
                    else -pairwise_maxima[left, right]
                ),
            }
            for block_start in (0, 4, 8)
            for left in range(block_start, block_start + 4)
            for right in range(left + 1, block_start + 4)
            if pairwise_minima[left, right] > 0.0 or pairwise_maxima[left, right] < 0.0
        ],
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
