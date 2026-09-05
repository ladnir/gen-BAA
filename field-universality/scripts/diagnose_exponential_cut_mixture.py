#!/usr/bin/env python3
"""Search for a fixed exponential mixture of projected cut pairs.

For cut-pair upper bounds G_ij(x),

  exp(min_ij G_ij(x)) <= sum_ij w_ij exp(G_ij(x)).

This diagnostic minimizes the right-hand side uniformly over all vertex
products of one frontier box.  A value below one is only a vertex diagnostic;
the continuous box still requires a separate analytic bound.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

import explore_projected_kl_box_cover as projected


def entropy_rows(distributions: np.ndarray) -> np.ndarray:
    positive = distributions > 0.0
    terms = np.zeros_like(distributions)
    terms[positive] = -distributions[positive] * np.log(distributions[positive])
    return np.sum(terms, axis=1)


def stage_cut_values(
    compositions: np.ndarray,
    cuts: list[object],
    kernels: tuple[np.ndarray, ...],
    stage: int,
) -> np.ndarray:
    left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
    left = compositions[:, left_start : left_start + 5]
    right = compositions[:, right_start : right_start + 5]
    values = []
    for cut, kernel in zip(cuts, kernels, strict=True):
        partition = np.sum((left @ kernel) * right, axis=1)
        linear = (
            right @ cut.target_potential
            + cut.increment_support_potential * left[:, 1:5].sum(axis=1)
            + cut.state_support_potential * right[:, 1:5].sum(axis=1)
        )
        values.append(np.log(partition) - linear)
    return np.stack(values, axis=1)


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
    parser.add_argument(
        "--restriction",
        action="append",
        default=[],
        metavar="DIM:SIDE:VALUE",
    )
    parser.add_argument(
        "--candidate-cuts",
        type=int,
        default=0,
        help="also retain this many best cuts by worst vertex value",
    )
    parser.add_argument("--affine-gating", action="store_true")
    parser.add_argument(
        "--affine-active-experts",
        type=int,
        default=0,
        help="add this many frequent vertex-minimizing cut pairs",
    )
    parser.add_argument(
        "--max-lp-coefficient",
        type=float,
        default=1e12,
        help=(
            "exclude fixed-mixture columns whose largest vertex coefficient "
            "exceeds this discovery-only scaling guard"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.max_lp_coefficient) or args.max_lp_coefficient <= 0.0:
        parser.error("--max-lp-coefficient must be finite and positive")

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
    kernels = [
        tuple(projected.envelope.kernel(cut) for cut in cuts)
        for cuts in (first_cuts, second_cuts)
    ]
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    lower = list(selected["lower"])
    upper = list(selected["upper"])
    for specification in args.restriction:
        dimension_text, side, value_text = specification.split(":")
        dimension = int(dimension_text)
        value = float(value_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], value)
        elif side == "ge":
            lower[dimension] = max(lower[dimension], value)
        else:
            raise ValueError("restriction side must be le or ge")
    box = projected.Box(tuple(lower), tuple(upper))
    vertices = [
        projected.simplex_box_vertices(box, start) for start in (0, 5, 10)
    ]
    message_vertices, middle_vertices, output_vertices = vertices
    counts = [len(value) for value in vertices]
    triple_count = counts[0] * counts[1] * counts[2]
    message = np.repeat(message_vertices, counts[1] * counts[2], axis=0)
    middle = np.tile(
        np.repeat(middle_vertices, counts[2], axis=0), (counts[0], 1)
    )
    output = np.tile(output_vertices, (counts[0] * counts[1], 1))
    compositions = np.concatenate((message, middle, output), axis=1)
    stage_values = [
        stage_cut_values(compositions, cuts, stage_kernels, stage)
        for stage, (cuts, stage_kernels) in enumerate(
            ((first_cuts, kernels[0]), (second_cuts, kernels[1]))
        )
    ]
    selected_cuts = []
    for values in stage_values:
        active = set(int(index) for index in np.argmin(values, axis=1))
        if args.candidate_cuts > 0:
            best_worst = np.argsort(np.max(values, axis=0))[
                : args.candidate_cuts
            ]
            active.update(int(index) for index in best_worst)
        selected_cuts.append(sorted(active))

    first_values = stage_values[0][:, selected_cuts[0]]
    second_values = stage_values[1][:, selected_cuts[1]]
    message_entropy = entropy_rows(message) / 3.0
    log_exponentials = (
        message_entropy[:, None, None]
        + first_values[:, :, None]
        + second_values[:, None, :]
    ).reshape(triple_count, -1)
    exponentials = np.exp(log_exponentials)
    pair_count = exponentials.shape[1]
    safe_columns = np.flatnonzero(
        np.max(log_exponentials, axis=0) <= np.log(args.max_lp_coefficient)
    )
    if len(safe_columns) == 0:
        raise RuntimeError("the LP scaling guard excluded every cut-pair column")
    fixed_exponentials = exponentials[:, safe_columns]
    fixed_pair_count = fixed_exponentials.shape[1]
    objective = np.zeros(fixed_pair_count + 1)
    objective[-1] = 1.0
    upper = np.concatenate(
        (fixed_exponentials, -np.ones((triple_count, 1))), axis=1
    )
    equality = np.zeros((1, fixed_pair_count + 1))
    equality[0, :fixed_pair_count] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(triple_count),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * fixed_pair_count + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"{result.message}; log coefficient range "
            f"[{np.nanmin(log_exponentials):.9g}, "
            f"{np.nanmax(log_exponentials):.9g}], coefficient range "
            f"[{np.nanmin(exponentials):.9g}, {np.nanmax(exponentials):.9g}], "
            f"finite={bool(np.all(np.isfinite(exponentials)))}"
        )
    flat_weights = np.zeros(pair_count)
    flat_weights[safe_columns] = result.x[:fixed_pair_count]
    weights = flat_weights.reshape(
        len(selected_cuts[0]), len(selected_cuts[1])
    )
    support = []
    support_positions = np.argwhere(weights > 1e-9)
    for first_position, second_position in support_positions:
        support.append(
            {
                "first_cut": selected_cuts[0][int(first_position)],
                "second_cut": selected_cuts[1][int(second_position)],
                "weight": float(weights[first_position, second_position]),
            }
        )
    mixture_values = exponentials @ flat_weights
    affine_payload = None
    if args.affine_gating:
        support_columns = [
            int(first_position) * len(selected_cuts[1]) + int(second_position)
            for first_position, second_position in support_positions
        ]
        first_positions = {
            cut: position for position, cut in enumerate(selected_cuts[0])
        }
        second_positions = {
            cut: position for position, cut in enumerate(selected_cuts[1])
        }
        active_pairs = Counter(
            zip(
                np.argmin(stage_values[0], axis=1).tolist(),
                np.argmin(stage_values[1], axis=1).tolist(),
                strict=True,
            )
        )
        for (first_cut, second_cut), _ in active_pairs.most_common(
            args.affine_active_experts
        ):
            column = (
                first_positions[first_cut] * len(selected_cuts[1])
                + second_positions[second_cut]
            )
            if column not in support_columns:
                support_columns.append(column)
        support_exponentials = exponentials[:, support_columns]
        independent_coordinates = [
            *range(1, 5),
            *range(6, 10),
            *range(11, 15),
        ]
        features = np.concatenate(
            (
                np.ones((triple_count, 1)),
                compositions[:, independent_coordinates],
            ),
            axis=1,
        )
        feature_count = features.shape[1]
        expert_count = len(support_columns)
        nonnegative = -sparse.kron(
            sparse.eye(expert_count, format="csr"),
            sparse.csr_matrix(features),
            format="csr",
        )
        nonnegative = sparse.hstack(
            (nonnegative, sparse.csr_matrix((nonnegative.shape[0], 1))),
            format="csr",
        )
        mixture_blocks = [
            sparse.csr_matrix(features * support_exponentials[:, [expert]])
            for expert in range(expert_count)
        ]
        mixture_rows = sparse.hstack(
            (*mixture_blocks, -np.ones((triple_count, 1))), format="csr"
        )
        affine_upper = sparse.vstack(
            (nonnegative, mixture_rows), format="csr"
        )
        affine_targets = np.zeros(affine_upper.shape[0])
        affine_equality = np.zeros(
            (feature_count, expert_count * feature_count + 1)
        )
        for feature in range(feature_count):
            for expert in range(expert_count):
                affine_equality[
                    feature, expert * feature_count + feature
                ] = 1.0
        affine_equality_target = np.zeros(feature_count)
        affine_equality_target[0] = 1.0
        affine_objective = np.zeros(expert_count * feature_count + 1)
        affine_objective[-1] = 1.0
        affine_result = linprog(
            affine_objective,
            A_ub=affine_upper,
            b_ub=affine_targets,
            A_eq=affine_equality,
            b_eq=affine_equality_target,
            bounds=[(None, None)] * (expert_count * feature_count)
            + [(None, None)],
            method="highs",
        )
        if not affine_result.success:
            raise RuntimeError(affine_result.message)
        coefficients = affine_result.x[:-1].reshape(
            expert_count, feature_count
        )
        gating_weights = features @ coefficients.T
        affine_values = np.sum(
            gating_weights * support_exponentials, axis=1
        )
        affine_payload = {
            "experts": expert_count,
            "expert_pairs": [
                {
                    "first_cut": selected_cuts[0][
                        column // len(selected_cuts[1])
                    ],
                    "second_cut": selected_cuts[1][
                        column % len(selected_cuts[1])
                    ],
                }
                for column in support_columns
            ],
            "feature_coordinates": independent_coordinates,
            "maximum_vertex_mixture": float(np.max(affine_values)),
            "log_maximum_vertex_mixture": float(
                np.log(np.max(affine_values))
            ),
            "minimum_gating_weight": float(np.min(gating_weights)),
            "maximum_partition_residual": float(
                np.max(np.abs(np.sum(gating_weights, axis=1) - 1.0))
            ),
            "coefficients": coefficients.tolist(),
        }
    payload = {
                "status": "floating exponential cut-mixture vertex diagnostic",
                "frontier_index": args.frontier_index,
                "restrictions": args.restriction,
                "vertex_counts": counts,
                "vertex_triples": triple_count,
                "first_candidate_cuts": selected_cuts[0],
                "second_candidate_cuts": selected_cuts[1],
                "cut_pairs": pair_count,
                "fixed_lp_pairs": fixed_pair_count,
                "fixed_lp_excluded_pairs": pair_count - fixed_pair_count,
                "max_lp_coefficient": args.max_lp_coefficient,
                "mixture_support": support,
                "affine_gating": affine_payload,
                "maximum_vertex_mixture": float(np.max(mixture_values)),
                "log_maximum_vertex_mixture": float(
                    np.log(np.max(mixture_values))
                ),
                "maximum_vertex_exact_exponential": float(
                    np.max(
                        np.exp(
                            message_entropy
                            + np.min(stage_values[0], axis=1)
                            + np.min(stage_values[1], axis=1)
                        )
                    )
                ),
            }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
