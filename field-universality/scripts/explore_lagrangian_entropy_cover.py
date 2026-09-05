#!/usr/bin/env python3
"""Floating box cover for the grouped internal-gap entropy dual."""

from __future__ import annotations

import argparse
import heapq
import json

import numpy as np

import diagnose_lagrangian_entropy_dual as grouped
import explore_perron_bound as perron
import optimize_bulk_exponent as bulk
import search_internal_information_gap as gap_search


def scale_grouped_cut(
    cut: grouped.GroupedCut, multiplier: int
) -> grouped.GroupedCut:
    return grouped.GroupedCut(
        cut.log_partition,
        perron.scale_symbol_vector(cut.row_potential, multiplier),
        perron.scale_symbol_vector(cut.column_potential, multiplier),
        perron.scale_symbol_vector(cut.increment_potential, multiplier),
        cut.value_at_reference,
        cut.gradient_residual,
        cut.optimizer_success,
    )


def linearized_cut(
    box: perron.CompositionBox,
    first: grouped.GroupedCut,
    second: grouped.GroupedCut,
    multiplier: float,
) -> tuple[float, np.ndarray]:
    layout = bulk.Layout(5)
    coefficients = np.zeros(layout.size)
    coefficients[layout.mu] = first.increment_potential
    coefficients[layout.beta] = (
        first.row_potential
        + first.column_potential
        + second.increment_potential
    )
    coefficients[layout.zeta] = second.row_potential + second.column_potential
    constant = first.log_partition + second.log_partition

    entropy_data = (
        (
            layout.mu,
            (1.0 - box.support_upper[0], *box.lower[0:4]),
            (1.0 - box.support_lower[0], *box.upper[0:4]),
            (1.0 + multiplier) * 2.0 / 3.0,
        ),
        (
            layout.beta,
            (1.0 - box.support_upper[1], *box.lower[4:8]),
            (1.0 - box.support_lower[1], *box.upper[4:8]),
            2.0 * (1.0 + multiplier),
        ),
        (
            layout.zeta,
            (1.0 - box.support_upper[2], *box.lower[8:12]),
            (1.0 - box.support_lower[2], *box.upper[8:12]),
            1.0 + multiplier,
        ),
    )
    for component, lowers, uppers, coefficient in entropy_data:
        for offset, (lower, upper) in enumerate(zip(lowers, uppers, strict=True)):
            slope, intercept = perron.secant_x_log_x(lower, upper)
            coefficients[component.start + offset] += coefficient * slope
            constant += coefficient * intercept

    support_data = (
        (
            box.support_lower[0],
            box.support_upper[0],
            multiplier * 2.0 / 3.0,
            range(layout.mu.start + 1, layout.mu.stop),
        ),
        (
            box.support_lower[1],
            box.support_upper[1],
            2.0 * multiplier,
            range(layout.beta.start + 1, layout.beta.stop),
        ),
        (
            box.support_lower[2],
            box.support_upper[2],
            multiplier,
            range(layout.zeta.start + 1, layout.zeta.stop),
        ),
    )
    for lower, upper, coefficient, indices in support_data:
        midpoint = min(max((lower + upper) / 2.0, 1e-12), 1.0 - 1e-12)
        slope = np.log((1.0 - midpoint) / midpoint)
        intercept = perron.binary_entropy(midpoint) - slope * midpoint
        constant += coefficient * intercept
        for index in indices:
            coefficients[index] += coefficient * slope
    return constant, coefficients


def values_at(
    point: np.ndarray, affine: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    return np.array(
        [constant + float(np.dot(coefficients, point)) for constant, coefficients in affine]
    )


def robust_mixture(
    box: perron.CompositionBox,
    args: argparse.Namespace,
    pairs: list[tuple[grouped.GroupedCut, grouped.GroupedCut]],
    seeds: list[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray, int]:
    affine = [linearized_cut(box, first, second, args.gap_multiplier) for first, second in pairs]
    rows = [values_at(seed, affine) for seed in seeds]
    maximizer = seeds[0]
    upper = float("inf")
    weights = np.full(len(pairs), 1.0 / len(pairs))
    for iteration in range(1, 101):
        _, weights = perron.common_mixture_bound(np.stack(rows))
        constant = sum(
            weight * cut[0] for weight, cut in zip(weights, affine, strict=True)
        )
        coefficients = sum(
            (weight * cut[1] for weight, cut in zip(weights, affine, strict=True)),
            np.zeros(bulk.Layout(5).size),
        )
        upper, maximizer = perron.maximize_affine_on_box(
            box, args.eta, args.delta, constant, coefficients
        )
        row = values_at(maximizer, affine)
        lower = float(np.dot(weights, row))
        if upper <= lower + 1e-9:
            return upper, weights, maximizer, iteration
        rows.append(row)
    return upper, weights, maximizer, 100


def exact_cut_value(
    point: np.ndarray,
    pair: tuple[grouped.GroupedCut, grouped.GroupedCut],
    multiplier: float,
) -> float:
    return grouped.lagrangian_cut_value(
        point, bulk.Layout(5), 3, multiplier, pair[0], pair[1]
    )


def split_dimension(
    box: perron.CompositionBox,
    point: np.ndarray,
    multiplier: float,
) -> int:
    layout = bulk.Layout(5)
    scores = []
    entropy_coefficients = (
        (1.0 + multiplier) * 2.0 / 3.0,
        2.0 * (1.0 + multiplier),
        1.0 + multiplier,
    )
    for component_index, component in enumerate((layout.mu, layout.beta, layout.zeta)):
        for offset in range(4):
            dimension = 4 * component_index + offset
            lower = box.lower[dimension]
            upper = box.upper[dimension]
            slope, intercept = perron.secant_x_log_x(lower, upper)
            value = point[component.start + 1 + offset]
            gap = slope * value + intercept - perron.x_log_x(value)
            scores.append(entropy_coefficients[component_index] * max(0.0, gap))
    support_coefficients = (
        multiplier * 2.0 / 3.0,
        2.0 * multiplier,
        multiplier,
    )
    for component_index, component in enumerate((layout.mu, layout.beta, layout.zeta)):
        lower = box.support_lower[component_index]
        upper = box.support_upper[component_index]
        midpoint = min(max((lower + upper) / 2.0, 1e-12), 1.0 - 1e-12)
        slope = np.log((1.0 - midpoint) / midpoint)
        intercept = perron.binary_entropy(midpoint) - slope * midpoint
        value = float(point[component][1:].sum())
        gap = slope * value + intercept - perron.binary_entropy(value)
        scores.append(support_coefficients[component_index] * max(0.0, gap))
    return int(np.argmax(scores))


def split_box(
    box: perron.CompositionBox, dimension: int
) -> tuple[perron.CompositionBox, perron.CompositionBox]:
    if dimension < 12:
        midpoint = (box.lower[dimension] + box.upper[dimension]) / 2.0
        left_upper = list(box.upper)
        left_upper[dimension] = midpoint
        right_lower = list(box.lower)
        right_lower[dimension] = midpoint
        return (
            perron.CompositionBox(box.lower, tuple(left_upper), box.support_lower, box.support_upper),
            perron.CompositionBox(tuple(right_lower), box.upper, box.support_lower, box.support_upper),
        )
    support = dimension - 12
    midpoint = (box.support_lower[support] + box.support_upper[support]) / 2.0
    left_upper = list(box.support_upper)
    left_upper[support] = midpoint
    right_lower = list(box.support_lower)
    right_lower[support] = midpoint
    return (
        perron.CompositionBox(box.lower, box.upper, box.support_lower, tuple(left_upper)),
        perron.CompositionBox(box.lower, box.upper, tuple(right_lower), box.support_upper),
    )


def evaluate_box(
    box: perron.CompositionBox, args: argparse.Namespace, seed: int
) -> dict[str, object]:
    feasible = perron.feasible_box_points(
        box, args.eta, args.delta, args.cuts * 4, seed
    )
    if not feasible:
        return {"feasible": False}
    references = perron.reference_points(feasible, args.cuts, seed + 1)
    layout = bulk.Layout(5)
    first = [
        grouped.fit_grouped_cut(
            reference[layout.mu], reference[layout.beta], args.gap_multiplier
        )
        for reference in references
    ]
    second = [
        grouped.fit_grouped_cut(
            reference[layout.beta], reference[layout.zeta], args.gap_multiplier
        )
        for reference in references
    ]
    if args.symmetric_reference_cuts:
        first.extend(
            grouped.fit_grouped_cut(
                perron.uniform_nonzero_composition(reference[layout.mu]),
                perron.uniform_nonzero_composition(reference[layout.beta]),
                args.gap_multiplier,
            )
            for reference in references
        )
        second.extend(
            grouped.fit_grouped_cut(
                perron.uniform_nonzero_composition(reference[layout.beta]),
                perron.uniform_nonzero_composition(reference[layout.zeta]),
                args.gap_multiplier,
            )
            for reference in references
        )

    def paired_cuts() -> list[tuple[grouped.GroupedCut, grouped.GroupedCut]]:
        expanded_first = (
            [
                scale_grouped_cut(cut, multiplier)
                for cut in first
                for multiplier in range(1, 5)
            ]
            if args.cut_orbits
            else first
        )
        expanded_second = (
            [
                scale_grouped_cut(cut, multiplier)
                for cut in second
                for multiplier in range(1, 5)
            ]
            if args.cut_orbits
            else second
        )
        return [
            (left, right)
            for left in expanded_first
            for right in expanded_second
        ]
    maximizer = references[0]
    iterations = 0
    for adaptive in range(args.adaptive_cuts + 1):
        pairs = paired_cuts()
        upper, weights, maximizer, used = robust_mixture(box, args, pairs, feasible)
        iterations += used
        if adaptive == args.adaptive_cuts:
            break
        first.append(
            grouped.fit_grouped_cut(
                maximizer[layout.mu], maximizer[layout.beta], args.gap_multiplier
            )
        )
        second.append(
            grouped.fit_grouped_cut(
                maximizer[layout.beta], maximizer[layout.zeta], args.gap_multiplier
            )
        )
    pairs = paired_cuts()
    mixture_exact = sum(
        weight * exact_cut_value(maximizer, pair, args.gap_multiplier)
        for weight, pair in zip(weights, pairs, strict=True)
    )
    local_pair = (
        grouped.fit_grouped_cut(
            maximizer[layout.mu], maximizer[layout.beta], args.gap_multiplier
        ),
        grouped.fit_grouped_cut(
            maximizer[layout.beta], maximizer[layout.zeta], args.gap_multiplier
        ),
    )
    local = exact_cut_value(maximizer, local_pair, args.gap_multiplier)
    exponent = bulk.exponent(maximizer, layout, 3)
    gap = gap_search.gap_and_gradient(maximizer, layout, 3)[0]
    true = exponent - args.gap_multiplier * gap
    return {
        "feasible": True,
        "upper": upper,
        "split_dimension": split_dimension(box, maximizer, args.gap_multiplier),
        "first_cuts": len(first),
        "second_cuts": len(second),
        "active_pairs": int(np.count_nonzero(weights > 1e-9)),
        "mixture_iterations": iterations,
        "worst_true_lagrangian": true,
        "worst_local_grouped_upper": local,
        "grouped_duality_gap": local - true,
        "cut_bank_gap": mixture_exact - local,
        "entropy_linearization_gap": upper - mixture_exact,
        "message": maximizer[layout.mu].tolist(),
        "middle": maximizer[layout.beta].tolist(),
        "output": maximizer[layout.zeta].tolist(),
    }


def search(args: argparse.Namespace) -> dict[str, object]:
    initial_box = perron.initial_composition_box(args.eta, args.delta)
    initial = evaluate_box(initial_box, args, args.seed + 10_000)
    queue = [(-float(initial["upper"]), 0, initial_box, initial)]
    identifier = 1
    history = [float(initial["upper"])]
    for split_index in range(args.max_boxes):
        negated, _, box, diagnostic = heapq.heappop(queue)
        parent = -negated
        dimension = int(diagnostic["split_dimension"])
        for child_index, child in enumerate(split_box(box, dimension)):
            child_diagnostic = evaluate_box(
                child, args, args.seed + 20_000 + 2 * split_index + child_index
            )
            if not child_diagnostic["feasible"]:
                continue
            local = float(child_diagnostic["upper"])
            value = min(parent, local)
            child_diagnostic["local_upper"] = local
            child_diagnostic["upper"] = value
            heapq.heappush(queue, (-value, identifier, child, child_diagnostic))
            identifier += 1
        history.append(-queue[0][0])
        if -queue[0][0] < -args.gap_multiplier * args.maximum_gap:
            return {
                "status": "numerically closes the low-gap proof target",
                "splits": split_index + 1,
                "current_upper": -queue[0][0],
                "required_upper": -args.gap_multiplier * args.maximum_gap,
                "history": history,
            }
    worst = queue[0]
    return {
        "status": "split budget exhausted",
        "splits": args.max_boxes,
        "boxes_remaining": len(queue),
        "current_upper": -worst[0],
        "required_upper": -args.gap_multiplier * args.maximum_gap,
        "history": history,
        "worst_box": {
            "lower": list(worst[2].lower),
            "upper": list(worst[2].upper),
            "support_lower": list(worst[2].support_lower),
            "support_upper": list(worst[2].support_upper),
            "diagnostic": worst[3],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--gap-multiplier", type=float, default=1.5)
    parser.add_argument("--maximum-gap", type=float, default=0.0001)
    parser.add_argument("--cuts", type=int, default=2)
    parser.add_argument("--adaptive-cuts", type=int, default=1)
    parser.add_argument("--cut-orbits", action="store_true")
    parser.add_argument("--symmetric-reference-cuts", action="store_true")
    parser.add_argument("--max-boxes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.cuts < 1 or args.adaptive_cuts < 0 or args.max_boxes < 1:
        parser.error("cut and box counts are invalid")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(search(arguments), indent=2, sort_keys=True))
