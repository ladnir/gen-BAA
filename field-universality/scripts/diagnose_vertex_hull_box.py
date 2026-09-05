#!/usr/bin/env python3
"""Bound one projected frontier box over its explicit vertex-product hull.

The three boxed simplexes have finitely many vertices.  A rank-one product of
three compositions lies in the convex hull of the products of vertex triples.
This LP optimizes the projected cut envelope over that hull.  Cross-block
domain inequalities are imposed only on the average composition, so the hull
is a sound outer approximation when an individual vertex triple violates one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected


def entropy(distribution: np.ndarray) -> float:
    positive = distribution > 0.0
    return -float(np.dot(distribution[positive], np.log(distribution[positive])))


def cut_values_at_composition(
    composition: np.ndarray,
    cuts: tuple[object, ...],
    kernels: tuple[np.ndarray, ...],
    stage: int,
) -> np.ndarray:
    """Evaluate every projected cut at one rank-one composition."""
    left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
    left = composition[left_start : left_start + 5]
    right = composition[right_start : right_start + 5]
    values = []
    for cut, kernel in zip(cuts, kernels, strict=True):
        partition = float(left @ kernel @ right)
        values.append(
            np.log(partition)
            - float(np.dot(cut.target_potential, right))
            - cut.increment_support_potential * float(left[1:5].sum())
            - cut.state_support_potential * float(right[1:5].sum())
        )
    return np.asarray(values)


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
    parser.add_argument("--outer-iterations", type=int, default=32)
    parser.add_argument("--all-single-stage-regimes", action="store_true")
    parser.add_argument("--cut-regime-splits", action="store_true")
    parser.add_argument(
        "--first-cut-indices",
        default="",
        help="comma-separated first-stage cuts used by the hull LP",
    )
    parser.add_argument(
        "--second-cut-indices",
        default="",
        help="comma-separated second-stage cuts used by the hull LP",
    )
    parser.add_argument(
        "--restriction",
        action="append",
        default=[],
        metavar="DIM:SIDE:VALUE",
        help="restrict the selected box with SIDE equal to le or ge",
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
    lp_cut_indices = [
        (
            [int(value) for value in specification.split(",")]
            if specification
            else list(range(len(cuts)))
        )
        for specification, cuts in (
            (args.first_cut_indices, first_cuts),
            (args.second_cut_indices, second_cuts),
        )
    ]
    saved = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(saved["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    lower = list(selected["lower"])
    upper = list(selected["upper"])
    applied_restrictions = []
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
        if lower[dimension] > upper[dimension]:
            raise ValueError("restriction makes the box empty")
        applied_restrictions.append(
            {"dimension": dimension, "side": side, "value": value}
        )
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
    entropy_index = triple_count
    first_index = triple_count + 1
    second_index = triple_count + 2
    size = triple_count + 3
    objective = np.zeros(size)
    objective[entropy_index] = 1.0 / 3.0
    objective[first_index] = 1.0
    objective[second_index] = 1.0
    rows = []
    targets = []

    def add_entropy_tangent(reference: np.ndarray) -> None:
        reference = np.maximum(reference, 1e-15)
        reference /= reference.sum()
        row = np.zeros(size)
        row[entropy_index] = 1.0
        row[:triple_count] = message @ np.log(reference)
        rows.append(row)
        targets.append(0.0)

    def partition_values(kernel: np.ndarray, stage: int) -> np.ndarray:
        left, right = ((message, middle) if stage == 0 else (middle, output))
        return np.sum((left @ kernel) * right, axis=1)

    partition_cache = [
        [partition_values(kernel, 0) for kernel in first_kernels],
        [partition_values(kernel, 1) for kernel in second_kernels],
    ]

    def add_cut_tangent(
        stage: int,
        cut: object,
        partition: np.ndarray,
        reference: float,
    ) -> None:
        left_start, right_start, epigraph = (
            (0, 5, first_index) if stage == 0 else (5, 10, second_index)
        )
        linear = (
            compositions[:, right_start : right_start + 5]
            @ cut.target_potential
            + cut.increment_support_potential
            * compositions[:, left_start + 1 : left_start + 5].sum(axis=1)
            + cut.state_support_potential
            * compositions[:, right_start + 1 : right_start + 5].sum(axis=1)
        )
        row = np.zeros(size)
        row[epigraph] = 1.0
        row[:triple_count] = linear - partition / reference
        rows.append(row)
        targets.append(np.log(reference) - 1.0)

    message_reference = projected.reference(box, 0)
    middle_reference = projected.reference(box, 5)
    output_reference = projected.reference(box, 10)
    add_entropy_tangent(message_reference)
    for stage, cuts, kernels, left_reference, right_reference in (
        (0, first_cuts, first_kernels, message_reference, middle_reference),
        (1, second_cuts, second_kernels, middle_reference, output_reference),
    ):
        for cut_index in lp_cut_indices[stage]:
            cut = cuts[cut_index]
            kernel = kernels[cut_index]
            add_cut_tangent(
                stage,
                cut,
                partition_cache[stage][cut_index],
                float(left_reference @ kernel @ right_reference),
            )

    domain_inequalities = []
    support_first = np.zeros(15)
    support_first[1:5] = 1.0
    support_first[6:10] = -2.0
    domain_inequalities.append(support_first)
    support_second = np.zeros(15)
    support_second[6:10] = 1.0
    support_second[11:15] = -2.0
    domain_inequalities.append(support_second)
    for value in range(2, 5):
        canonical = np.zeros(15)
        canonical[value] = 1.0
        canonical[1] = -1.0
        domain_inequalities.append(canonical)
    for inequality in domain_inequalities:
        row = np.zeros(size)
        row[:triple_count] = compositions @ inequality
        rows.append(row)
        targets.append(0.0)

    equality = np.zeros((1, size))
    equality[0, :triple_count] = 1.0
    base_bounds = [(0.0, 1.0)] * triple_count + [(-1000.0, 1000.0)] * 3

    def solve(
        allowed: np.ndarray | None = None,
        halfspace: tuple[np.ndarray, float, bool] | None = None,
    ) -> object:
        bounds = base_bounds
        if allowed is not None:
            bounds = [
                (0.0, 1.0) if bool(value) else (0.0, 0.0)
                for value in allowed
            ] + [(-1000.0, 1000.0)] * 3
        active_rows = rows
        active_targets = targets
        if halfspace is not None:
            normal, threshold, less_equal = halfspace
            halfspace_row = np.zeros(size)
            halfspace_row[:triple_count] = compositions @ normal
            halfspace_target = threshold
            if not less_equal:
                halfspace_row = -halfspace_row
                halfspace_target = -halfspace_target
            active_rows = rows + [halfspace_row]
            active_targets = targets + [halfspace_target]
        return linprog(
            -objective,
            A_ub=np.stack(active_rows),
            b_ub=np.asarray(active_targets),
            A_eq=equality,
            b_eq=np.ones(1),
            bounds=bounds,
            method="highs",
        )

    result = solve()
    if not result.success:
        raise RuntimeError(result.message)
    for _ in range(args.outer_iterations):
        weights = result.x[:triple_count]
        point = weights @ compositions
        additions = 0
        actual_entropy = entropy(point[0:5])
        if result.x[entropy_index] > actual_entropy + 1e-9:
            add_entropy_tangent(point[0:5])
            additions += 1
        for stage, cuts, kernels, epigraph in (
            (0, first_cuts, first_kernels, first_index),
            (1, second_cuts, second_kernels, second_index),
        ):
            values = []
            for cut_index in lp_cut_indices[stage]:
                cut = cuts[cut_index]
                partition = float(
                    np.dot(weights, partition_cache[stage][cut_index])
                )
                left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
                values.append(
                    np.log(partition)
                    - float(np.dot(cut.target_potential, point[right_start : right_start + 5]))
                    - cut.increment_support_potential * float(point[left_start + 1 : left_start + 5].sum())
                    - cut.state_support_potential * float(point[right_start + 1 : right_start + 5].sum())
                )
            for active_position in np.argsort(values)[:3]:
                active_position = int(active_position)
                active = lp_cut_indices[stage][active_position]
                if result.x[epigraph] <= values[active_position] + 1e-9:
                    continue
                reference = float(
                    np.dot(weights, partition_cache[stage][active])
                )
                add_cut_tangent(
                    stage,
                    cuts[active],
                    partition_cache[stage][active],
                    reference,
                )
                additions += 1
        if additions == 0:
            break
        result = solve()
        if not result.success:
            raise RuntimeError(result.message)

    weights = result.x[:triple_count]
    point = projected.sanitize_composition(weights @ compositions)
    exact_bank = projected.composition_bank_upper(point, first_cuts, second_cuts)
    mixed_cut_values = []
    point_cut_values = []
    for stage, cuts, kernels in (
        (0, first_cuts, first_kernels),
        (1, second_cuts, second_kernels),
    ):
        left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
        mixed_values = []
        for cut_index, cut in enumerate(cuts):
            partition = float(np.dot(weights, partition_cache[stage][cut_index]))
            mixed_values.append(
                np.log(partition)
                - float(np.dot(cut.target_potential, point[right_start : right_start + 5]))
                - cut.increment_support_potential * float(point[left_start + 1 : left_start + 5].sum())
                - cut.state_support_potential * float(point[right_start + 1 : right_start + 5].sum())
            )
        mixed_values = np.asarray(mixed_values)
        rank_one_values = cut_values_at_composition(point, cuts, kernels, stage)
        mixed_cut_values.append(mixed_values)
        point_cut_values.append(rank_one_values)
    vertex_active_cuts = []
    vertex_cut_value_tables = []
    for stage, cuts in ((0, first_cuts), (1, second_cuts)):
        left_start, right_start = ((0, 5) if stage == 0 else (5, 10))
        values = []
        for cut_index, cut in enumerate(cuts):
            linear = (
                compositions[:, right_start : right_start + 5]
                @ cut.target_potential
                + cut.increment_support_potential
                * compositions[:, left_start + 1 : left_start + 5].sum(axis=1)
                + cut.state_support_potential
                * compositions[:, right_start + 1 : right_start + 5].sum(axis=1)
            )
            values.append(np.log(partition_cache[stage][cut_index]) - linear)
        value_table = np.stack(values)
        vertex_cut_value_tables.append(value_table)
        vertex_active_cuts.append(np.argmin(value_table, axis=0))

    message_entropies = np.asarray(
        [entropy(composition[0:5]) / 3.0 for composition in compositions]
    )
    fixed_cut_vertex_summaries = []
    fixed_cut_vertex_violations = []
    greedy_vertex_covers = []
    for stage in range(2):
        other_stage = 1 - stage
        base = message_entropies + np.min(
            vertex_cut_value_tables[other_stage], axis=0
        )
        maxima = np.max(base[None, :] + vertex_cut_value_tables[stage], axis=1)
        best_cut = int(np.argmin(maxima))
        fixed_cut_vertex_summaries.append(
            {
                "stage": stage + 1,
                "best_cut": best_cut,
                "best_worst_vertex_value": float(maxima[best_cut]),
                "cuts_negative_on_every_vertex": int(np.sum(maxima < 0.0)),
            }
        )
        violation_indices = np.flatnonzero(
            base + vertex_cut_value_tables[stage][best_cut] >= 0.0
        )
        fixed_cut_vertex_violations.append(
            {
                "stage": stage + 1,
                "cut": best_cut,
                "violations": [
                    {
                        "flat_index": int(index),
                        "value": float(
                            base[index]
                            + vertex_cut_value_tables[stage][best_cut, index]
                        ),
                        "composition": compositions[index].tolist(),
                        "active_cut": int(vertex_active_cuts[stage][index]),
                    }
                    for index in violation_indices[:20]
                ],
                "violation_count": int(len(violation_indices)),
            }
        )
        covered = np.zeros(triple_count, dtype=bool)
        cover = []
        negative = base[None, :] + vertex_cut_value_tables[stage] < 0.0
        while not np.all(covered):
            gains = np.sum(negative & ~covered[None, :], axis=1)
            cut_index = int(np.argmax(gains))
            gain = int(gains[cut_index])
            if gain == 0:
                break
            newly_covered = negative[cut_index] & ~covered
            covered |= negative[cut_index]
            cover.append(
                {
                    "cut": cut_index,
                    "new_vertices": int(np.sum(newly_covered)),
                    "total_vertices": int(np.sum(covered)),
                }
            )
        greedy_vertex_covers.append(
            {
                "stage": stage + 1,
                "covered_vertices": int(np.sum(covered)),
                "total_vertices": triple_count,
                "cuts": cover,
            }
        )

    active_flat_indices = np.flatnonzero(weights > 1e-9)
    active_triples = []
    for flat_index in active_flat_indices:
        message_index = int(flat_index // (counts[1] * counts[2]))
        remainder = int(flat_index % (counts[1] * counts[2]))
        middle_index = remainder // counts[2]
        output_index = remainder % counts[2]
        composition = compositions[flat_index]
        active_cuts = []
        active_values = []
        for stage, cuts, kernels in (
            (0, first_cuts, first_kernels),
            (1, second_cuts, second_kernels),
        ):
            values = cut_values_at_composition(composition, cuts, kernels, stage)
            active = int(np.argmin(values))
            active_cuts.append(active)
            active_values.append(float(values[active]))
        active_triples.append(
            {
                "weight": float(weights[flat_index]),
                "indices": [message_index, middle_index, output_index],
                "composition": composition.tolist(),
                "exact_bank_upper": projected.composition_bank_upper(
                    composition, first_cuts, second_cuts
                ),
                "active_cuts": active_cuts,
                "active_cut_values": active_values,
            }
        )

    regime_diagnostics = {}
    for name, identifiers in (
        ("first_stage", vertex_active_cuts[0][:, None]),
        ("second_stage", vertex_active_cuts[1][:, None]),
        ("cut_pair", np.stack(vertex_active_cuts, axis=1)),
    ):
        selected_identifiers = sorted(
            {
                tuple(int(value) for value in identifiers[index])
                for index in active_flat_indices
            }
        )
        entries = []
        for identifier in selected_identifiers:
            allowed = np.all(identifiers == np.asarray(identifier), axis=1)
            restricted = solve(allowed)
            entries.append(
                {
                    "identifier": list(identifier),
                    "vertex_triples": int(np.sum(allowed)),
                    "upper_without_new_separation": (
                        -float(restricted.fun) if restricted.success else None
                    ),
                }
            )
        regime_diagnostics[name] = entries

    all_single_stage_regimes = {}
    if args.all_single_stage_regimes:
        for name, active_identifiers in (
            ("first_stage", vertex_active_cuts[0]),
            ("second_stage", vertex_active_cuts[1]),
        ):
            entries = []
            for identifier in np.unique(active_identifiers):
                allowed = active_identifiers == identifier
                restricted = solve(allowed)
                entries.append(
                    {
                        "identifier": int(identifier),
                        "vertex_triples": int(np.sum(allowed)),
                        "upper_without_new_separation": (
                            -float(restricted.fun) if restricted.success else None
                        ),
                    }
                )
            all_single_stage_regimes[name] = entries

    cut_regime_splits = []
    if args.cut_regime_splits:
        candidates = []

        def add_candidate(name: str, normal: np.ndarray, threshold: float) -> None:
            scale = float(np.max(np.abs(normal)))
            if scale <= 1e-12:
                return
            normal = normal / scale
            threshold = float(threshold / scale)
            candidates.append((name, normal, threshold))

        # Axis-aligned candidates provide the reference value for the same LP.
        for dimension in range(15):
            normal = np.zeros(15)
            normal[dimension] = 1.0
            add_candidate(
                f"coordinate_{dimension}",
                normal,
                0.5 * (box.lower[dimension] + box.upper[dimension]),
            )

        # Separate each active first-stage regime from the remaining vertices.
        selected_first = sorted(
            {int(vertex_active_cuts[0][index]) for index in active_flat_indices}
        )
        for identifier in selected_first:
            selected_mask = vertex_active_cuts[0] == identifier
            selected_centroid = compositions[selected_mask].mean(axis=0)
            other_centroid = compositions[~selected_mask].mean(axis=0)
            normal = selected_centroid - other_centroid
            threshold = 0.5 * float(normal @ (selected_centroid + other_centroid))
            add_candidate(f"first_regime_{identifier}_vs_rest", normal, threshold)

        # Pairwise centroid planes between regimes that occur in the active mix.
        regime_centroids = {
            identifier: compositions[vertex_active_cuts[0] == identifier].mean(axis=0)
            for identifier in selected_first
        }
        for offset, first_identifier in enumerate(selected_first):
            for second_identifier in selected_first[offset + 1 :]:
                first_centroid = regime_centroids[first_identifier]
                second_centroid = regime_centroids[second_identifier]
                normal = first_centroid - second_centroid
                threshold = 0.5 * float(normal @ (first_centroid + second_centroid))
                add_candidate(
                    f"first_regimes_{first_identifier}_vs_{second_identifier}",
                    normal,
                    threshold,
                )

        # Linearize equal-value surfaces between the active and mixed cuts.
        comparison_cuts = sorted(
            set(selected_first)
            | {int(np.argmin(mixed_cut_values[0]))}
            | {int(np.argmin(point_cut_values[0]))}
        )

        def cut_gradient(cut_index: int) -> np.ndarray:
            cut = first_cuts[cut_index]
            kernel = first_kernels[cut_index]
            left = point[0:5]
            right = point[5:10]
            partition = float(left @ kernel @ right)
            gradient = np.zeros(15)
            gradient[0:5] = kernel @ right / partition
            gradient[5:10] = kernel.T @ left / partition - cut.target_potential
            gradient[1:5] -= cut.increment_support_potential
            gradient[6:10] -= cut.state_support_potential
            return gradient

        gradients = {index: cut_gradient(index) for index in comparison_cuts}
        for offset, first_index in enumerate(comparison_cuts):
            for second_index in comparison_cuts[offset + 1 :]:
                normal = gradients[first_index] - gradients[second_index]
                value_difference = (
                    point_cut_values[0][first_index]
                    - point_cut_values[0][second_index]
                )
                threshold = float(normal @ point - value_difference)
                add_candidate(
                    f"linearized_cuts_{first_index}_vs_{second_index}",
                    normal,
                    threshold,
                )

        for name, normal, threshold in candidates:
            lower_child = solve(halfspace=(normal, threshold, True))
            upper_child = solve(halfspace=(normal, threshold, False))
            if not lower_child.success or not upper_child.success:
                continue
            lower_upper = -float(lower_child.fun)
            upper_upper = -float(upper_child.fun)
            cut_regime_splits.append(
                {
                    "name": name,
                    "threshold": threshold,
                    "normal": normal.tolist(),
                    "lower_child_upper": lower_upper,
                    "upper_child_upper": upper_upper,
                    "maximum_child_upper": max(lower_upper, upper_upper),
                }
            )
        cut_regime_splits.sort(key=lambda value: value["maximum_child_upper"])
    print(
        json.dumps(
            {
                "frontier_index": args.frontier_index,
                "lp_cut_indices": lp_cut_indices,
                "applied_restrictions": applied_restrictions,
                "vertex_counts": counts,
                "vertex_triples": triple_count,
                "original_moment_upper": selected["bound"],
                "vertex_hull_upper": -float(result.fun),
                "exact_bank_upper_at_lp_composition": exact_bank,
                "relaxation_gap": -float(result.fun) - exact_bank,
                "active_vertex_triples": int(np.sum(weights > 1e-10)),
                "active_triple_details": active_triples,
                "mixed_hull_cut_summary": [
                    {
                        "stage": stage + 1,
                        "minimum_cut": int(np.argmin(mixed_cut_values[stage])),
                        "minimum_value": float(np.min(mixed_cut_values[stage])),
                        "five_smallest": [
                            {
                                "cut": int(index),
                                "value": float(mixed_cut_values[stage][index]),
                            }
                            for index in np.argsort(mixed_cut_values[stage])[:5]
                        ],
                    }
                    for stage in range(2)
                ],
                "rank_one_lp_composition_cut_summary": [
                    {
                        "stage": stage + 1,
                        "minimum_cut": int(np.argmin(point_cut_values[stage])),
                        "minimum_value": float(np.min(point_cut_values[stage])),
                    }
                    for stage in range(2)
                ],
                "active_support_regime_diagnostics": regime_diagnostics,
                "all_vertex_regime_counts": {
                    "first_stage": int(len(np.unique(vertex_active_cuts[0]))),
                    "second_stage": int(len(np.unique(vertex_active_cuts[1]))),
                    "cut_pair": int(
                        len(
                            np.unique(
                                np.stack(vertex_active_cuts, axis=1), axis=0
                            )
                        )
                    ),
                },
                "fixed_cut_vertex_summaries": fixed_cut_vertex_summaries,
                "fixed_cut_vertex_violations": fixed_cut_vertex_violations,
                "greedy_vertex_covers": greedy_vertex_covers,
                "all_single_stage_regime_diagnostics": all_single_stage_regimes,
                "cut_regime_split_diagnostics": cut_regime_splits[:20],
                "cut_regime_split_candidates": len(cut_regime_splits),
                "constraints": len(rows),
                "lp_composition": point.tolist(),
                "status": "valid floating vertex-hull LP; not an exact certificate",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
