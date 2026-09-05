#!/usr/bin/env python3
"""Probe joint concavity of a fixed projected-dual pair on one frontier box.

This script is a floating diagnostic.  It tests the Hessian only at sampled
points and therefore does not certify concavity on the complete box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_projected_kl_box_cover as projected


def simplex_tangent_basis(
    fixed_support_blocks: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    blocks = []
    for fixed_support in fixed_support_blocks:
        if fixed_support:
            block = np.zeros((5, 3))
            for column in range(3):
                block[1, column] = -1.0
                block[column + 2, column] = 1.0
        else:
            block = np.zeros((5, 4))
            for column in range(4):
                block[0, column] = -1.0
                block[column + 1, column] = 1.0
        blocks.append(block)
    basis = np.zeros((15, sum(block.shape[1] for block in blocks)))
    column_start = 0
    for block_index, block in enumerate(blocks):
        column_stop = column_start + block.shape[1]
        basis[
            5 * block_index : 5 * (block_index + 1),
            column_start:column_stop,
        ] = block
        column_start = column_stop
    return basis


def add_log_bilinear_hessian(
    hessian: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    kernel: np.ndarray,
    left_start: int,
    right_start: int,
) -> None:
    partition = float(left @ kernel @ right)
    left_gradient = kernel @ right / partition
    right_gradient = kernel.T @ left / partition
    hessian[
        left_start : left_start + 5,
        left_start : left_start + 5,
    ] -= np.outer(left_gradient, left_gradient)
    hessian[
        right_start : right_start + 5,
        right_start : right_start + 5,
    ] -= np.outer(right_gradient, right_gradient)
    crossing = kernel / partition - np.outer(left_gradient, right_gradient)
    hessian[
        left_start : left_start + 5,
        right_start : right_start + 5,
    ] += crossing
    hessian[
        right_start : right_start + 5,
        left_start : left_start + 5,
    ] += crossing.T


def restricted_hessian(
    point: np.ndarray,
    first_kernel: np.ndarray,
    second_kernel: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    hessian = np.zeros((15, 15))
    hessian[0:5, 0:5] -= np.diag(1.0 / (3.0 * point[0:5]))
    add_log_bilinear_hessian(
        hessian, point[0:5], point[5:10], first_kernel, 0, 5
    )
    add_log_bilinear_hessian(
        hessian, point[5:10], point[10:15], second_kernel, 5, 10
    )
    return basis.T @ hessian @ basis


def random_boxed_simplex_point(
    vertices: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    blocks = []
    for block_vertices in vertices:
        weights = rng.dirichlet(np.ones(len(block_vertices)))
        blocks.append(weights @ block_vertices)
    return np.concatenate(blocks)


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
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--candidate-cuts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=71)
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
    point = projected.sanitize_composition(np.asarray(selected["point"]))
    first_values = [
        projected.stage_upper_with_kernel(
            point[0:5], point[5:10], cut, kernel
        )
        for cut, kernel in zip(first_cuts, first_kernels, strict=True)
    ]
    second_values = [
        projected.stage_upper_with_kernel(
            point[5:10], point[10:15], cut, kernel
        )
        for cut, kernel in zip(second_cuts, second_kernels, strict=True)
    ]
    first_index = int(np.argmin(first_values))
    second_index = int(np.argmin(second_values))
    bases = {
        "free_supports": simplex_tangent_basis(),
        "fixed_output_support": simplex_tangent_basis((False, False, True)),
        "fixed_all_supports": simplex_tangent_basis((True, True, True)),
    }
    basis = bases["free_supports"]
    first_candidates = np.argsort(first_values)[: args.candidate_cuts]
    second_candidates = np.argsort(second_values)[: args.candidate_cuts]
    pair_candidates = []
    for first_rank, candidate_first in enumerate(first_candidates):
        for second_rank, candidate_second in enumerate(second_candidates):
            pair_hessian = restricted_hessian(
                point,
                first_kernels[int(candidate_first)],
                second_kernels[int(candidate_second)],
                basis,
            )
            pair_candidates.append(
                {
                    "first_index": int(candidate_first),
                    "second_index": int(candidate_second),
                    "first_rank": first_rank,
                    "second_rank": second_rank,
                    "fixed_pair_upper": (
                        projected.bulk_entropy(point[0:5]) / 3.0
                        + first_values[int(candidate_first)]
                        + second_values[int(candidate_second)]
                    ),
                    "maximum_hessian_eigenvalue": float(
                        np.linalg.eigvalsh(pair_hessian)[-1]
                    ),
                }
            )
    negative_pairs = [
        candidate
        for candidate in pair_candidates
        if candidate["fixed_pair_upper"] < 0.0
    ]
    best_curvature_pair = min(
        negative_pairs,
        key=lambda candidate: candidate["maximum_hessian_eigenvalue"],
    )

    point_eigenvalues, point_eigenvectors = np.linalg.eigh(
        restricted_hessian(
            point,
            first_kernels[first_index],
            second_kernels[second_index],
            basis,
        )
    )
    positive_direction = basis @ point_eigenvectors[:, -1]
    positive_direction /= np.max(np.abs(positive_direction))
    vertices = [
        projected.simplex_box_vertices(box, start) for start in (0, 5, 10)
    ]
    rng = np.random.default_rng(args.seed)
    maxima = {name: [] for name in bases}
    worst_sample = None
    for _ in range(args.samples):
        sample = random_boxed_simplex_point(vertices, rng)
        for name, current_basis in bases.items():
            maximum = float(
                np.linalg.eigvalsh(
                    restricted_hessian(
                        sample,
                        first_kernels[first_index],
                        second_kernels[second_index],
                        current_basis,
                    )
                )[-1]
            )
            maxima[name].append(maximum)
        if worst_sample is None or maxima["free_supports"][-1] > worst_sample[0]:
            worst_sample = (maxima["free_supports"][-1], sample.tolist())

    subspace_results = {}
    for name, current_basis in bases.items():
        eigenvalues = np.linalg.eigvalsh(
            restricted_hessian(
                point,
                first_kernels[first_index],
                second_kernels[second_index],
                current_basis,
            )
        )
        subspace_results[name] = {
            "dimension": current_basis.shape[1],
            "maximum_eigenvalue_at_lp_composition": float(eigenvalues[-1]),
            "maximum_sampled_eigenvalue": max(maxima[name]),
            "minimum_sampled_maximum_eigenvalue": min(maxima[name]),
            "positive_sample_fraction": float(
                np.mean(np.asarray(maxima[name]) > 1e-10)
            ),
        }

    print(
        json.dumps(
            {
                "frontier_index": args.frontier_index,
                "box_upper": selected["bound"],
                "first_cut_index": first_index,
                "second_cut_index": second_index,
                "fixed_pair_upper_at_lp_composition": (
                    projected.bulk_entropy(point[0:5]) / 3.0
                    + first_values[first_index]
                    + second_values[second_index]
                ),
                "candidate_pair_search": {
                    "cuts_per_stage": args.candidate_cuts,
                    "negative_pairs": len(negative_pairs),
                    "nonpositive_hessian_pairs": sum(
                        candidate["maximum_hessian_eigenvalue"] <= 1e-10
                        for candidate in negative_pairs
                    ),
                    "best_curvature_pair": best_curvature_pair,
                },
                "simplex_vertex_counts": [len(value) for value in vertices],
                "maximum_hessian_eigenvalue_at_lp_composition": float(
                    point_eigenvalues[-1]
                ),
                "minimum_hessian_eigenvalue_at_lp_composition": float(
                    point_eigenvalues[0]
                ),
                "hessian_eigenvalues_at_lp_composition": (
                    point_eigenvalues.tolist()
                ),
                "positive_hessian_dimensions_at_lp_composition": int(
                    np.sum(point_eigenvalues > 1e-10)
                ),
                "positive_hessian_direction_at_lp_composition": (
                    positive_direction.tolist()
                ),
                "samples": args.samples,
                "subspaces": subspace_results,
                "worst_sample": worst_sample[1],
                "status": "floating Hessian diagnostic; sampled points do not certify a box",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
