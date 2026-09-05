#!/usr/bin/env python3
"""Verify and evaluate the first-round run-profile peeling bound."""

from __future__ import annotations

import argparse
import itertools
import json
import math


def run_profile(deletion: tuple[int, ...], length: int) -> tuple[int, int]:
    runs = 1 + sum(right != left + 1 for left, right in zip(deletion, deletion[1:]))
    return runs, int(deletion[-1] == length - 1)


def profile_count(length: int, size: int, runs: int, terminal: int | None) -> int:
    compositions = math.comb(size - 1, runs - 1)
    if terminal == 1:
        return compositions * math.comb(length - size, runs - 1)
    if terminal == 0:
        return compositions * math.comb(length - size, runs)
    return compositions * math.comb(length - size + 1, runs)


def verify_counts(maximum_length: int) -> int:
    checked = 0
    for length in range(1, maximum_length + 1):
        for size in range(1, length + 1):
            observed: dict[tuple[int, int], int] = {}
            for deletion in itertools.combinations(range(length), size):
                profile = run_profile(deletion, length)
                observed[profile] = observed.get(profile, 0) + 1
            for (runs, terminal), count in observed.items():
                expected = profile_count(length, size, runs, terminal)
                if count != expected:
                    raise AssertionError(
                        f"N={length}, d={size}, r={runs}, tau={terminal}: "
                        f"observed {count}, expected {expected}"
                    )
                checked += 1
            by_runs: dict[int, int] = {}
            for (runs, _), count in observed.items():
                by_runs[runs] = by_runs.get(runs, 0) + count
            for runs, count in by_runs.items():
                expected = profile_count(length, size, runs, None)
                if count != expected:
                    raise AssertionError(
                        f"N={length}, d={size}, r={runs}: "
                        f"observed {count}, expected {expected}"
                    )
                checked += 1
    return checked


def binary_entropy(value: float) -> float:
    if value in (0.0, 1.0):
        return 0.0
    return -value * math.log2(value) - (1.0 - value) * math.log2(1.0 - value)


def asymptotic_profile_entropy(alpha: float, rho: float) -> float:
    return alpha * binary_entropy(rho / alpha) + (1.0 - alpha) * binary_entropy(
        rho / (1.0 - alpha)
    )


def evaluate_bound(
    length: int,
    size: int,
    runs: int,
    terminal: int,
    expansion_slack: float,
    pairing_slack: float,
) -> dict[str, float | int]:
    support_before_second_interleaver = size + runs - terminal
    expected_expanded_rows = (
        2 * support_before_second_interleaver
        - support_before_second_interleaver**2 / length
    )
    expanded_row_floor = expected_expanded_rows - expansion_slack
    singleton_row_floor = max(0.0, 2 * expanded_row_floor - 4 * size)
    zero_row_floor = max(0, length - 2 * support_before_second_interleaver)
    candidate_check_mean_floor = (
        4 * singleton_row_floor * zero_row_floor / (3 * (length - 1))
        if length > 1
        else 0.0
    )
    peeled_column_floor = max(
        0, math.floor((candidate_check_mean_floor - pairing_slack) / 8)
    )
    expansion_failure_bound = math.exp(-(expansion_slack**2) / (72 * length))
    pairing_failure_bound = math.exp(-(pairing_slack**2) / (32 * length))
    return {
        "length": length,
        "deletion_size": size,
        "runs": runs,
        "terminal": terminal,
        "initial_active_rows": support_before_second_interleaver,
        "expected_expanded_rows": expected_expanded_rows,
        "expanded_row_floor": expanded_row_floor,
        "singleton_row_floor": singleton_row_floor,
        "zero_row_floor": zero_row_floor,
        "candidate_check_mean_floor": candidate_check_mean_floor,
        "peeled_column_floor": peeled_column_floor,
        "peeled_fraction_of_length": peeled_column_floor / length,
        "peeled_fraction_of_deletion": peeled_column_floor / size,
        "expansion_failure_bound": expansion_failure_bound,
        "pairing_failure_bound": pairing_failure_bound,
        "combined_failure_bound": expansion_failure_bound + pairing_failure_bound,
    }


def evaluate_two_round_bound(
    one_round: dict[str, float | int],
    first_round_triple_fraction: float,
    stopping_cap_fraction: float,
    category_slack_fraction: float,
    matching_tail: float,
) -> dict[str, float | int]:
    length = int(one_round["length"])
    deletion_size = int(one_round["deletion_size"])
    singleton_floor = float(one_round["singleton_row_floor"])
    zero_floor = float(one_round["zero_row_floor"])
    if length % 3 != 0:
        raise ValueError("two-round bound requires length divisible by three")
    blocks = length // 3
    first_matching_edges = round(first_round_triple_fraction * blocks)
    second_matching_edges = blocks - first_matching_edges
    effective_triple_fraction = first_matching_edges / blocks
    exposed_rows = 2 * first_matching_edges
    unexposed_rows = length - exposed_rows
    matching_slack = math.sqrt(8 * length * math.log(1 / matching_tail))
    category_slack = category_slack_fraction * length
    stopping_cap = math.floor(stopping_cap_fraction * length)
    first_candidate_mean_floor = (
        first_matching_edges
        * 2
        * singleton_floor
        * zero_floor
        / (length * (length - 1))
    )
    first_peeled_floor = max(
        0, math.floor((first_candidate_mean_floor - matching_slack) / 4)
    )
    second_singleton_floor = max(
        0.0,
        unexposed_rows / length * singleton_floor
        - category_slack
        - 4 * stopping_cap,
    )
    second_zero_floor = max(
        0.0, unexposed_rows / length * zero_floor - category_slack
    )
    second_candidate_mean_floor = (
        second_matching_edges
        * 2
        * second_singleton_floor
        * second_zero_floor
        / (unexposed_rows * (unexposed_rows - 1))
    )
    second_peeled_floor = max(
        0, math.floor((second_candidate_mean_floor - matching_slack) / 4)
    )
    total_peeled_floor = min(
        stopping_cap, first_peeled_floor + second_peeled_floor
    )
    category_tail = math.exp(-(category_slack**2) / (2 * length))
    second_matching_tail = math.exp(
        -(matching_slack**2) / (8 * unexposed_rows)
    )
    combined_failure_bound = (
        float(one_round["expansion_failure_bound"])
        + matching_tail
        + second_matching_tail
        + 2 * category_tail
    )
    return {
        "requested_first_round_triple_fraction": first_round_triple_fraction,
        "effective_first_round_triple_fraction": effective_triple_fraction,
        "first_matching_edges": first_matching_edges,
        "second_matching_edges": second_matching_edges,
        "exposed_rows": exposed_rows,
        "unexposed_rows": unexposed_rows,
        "matching_slack": matching_slack,
        "category_slack": category_slack,
        "stopping_cap": stopping_cap,
        "first_candidate_mean_floor": first_candidate_mean_floor,
        "first_peeled_floor": first_peeled_floor,
        "second_singleton_floor": second_singleton_floor,
        "second_zero_floor": second_zero_floor,
        "second_candidate_mean_floor": second_candidate_mean_floor,
        "second_peeled_floor": second_peeled_floor,
        "total_peeled_floor": math.floor(total_peeled_floor),
        "total_fraction_of_length": total_peeled_floor / length,
        "total_fraction_of_deletion": total_peeled_floor / deletion_size,
        "first_matching_tail": matching_tail,
        "second_matching_tail": second_matching_tail,
        "category_tail_per_type": category_tail,
        "combined_failure_bound": combined_failure_bound,
    }


def evaluate_multi_round_bound(
    one_round: dict[str, float | int], rounds: int, tail_per_event: float
) -> dict[str, object]:
    length = int(one_round["length"])
    deletion_size = int(one_round["deletion_size"])
    if length % 3 != 0:
        raise ValueError("multi-round bound requires length divisible by three")
    blocks = length // 3
    if not 1 <= rounds <= blocks:
        raise ValueError("round count is outside 1..N/3")
    reservoir_sizes = [blocks // rounds] * rounds
    for index in range(blocks % rounds):
        reservoir_sizes[index] += 1

    remaining_rows = length
    singleton_floor = float(one_round["singleton_row_floor"])
    zero_floor = float(one_round["zero_row_floor"])
    total_peeled = 0
    trace = []
    for index, matching_edges in enumerate(reservoir_sizes):
        matching_slack = math.sqrt(
            8 * remaining_rows * math.log(1 / tail_per_event)
        )
        candidate_mean = (
            matching_edges
            * 2
            * singleton_floor
            * zero_floor
            / (remaining_rows * (remaining_rows - 1))
        )
        peeled = max(0, math.floor((candidate_mean - matching_slack) / 4))
        total_peeled += peeled
        round_result: dict[str, float | int] = {
            "round": index + 1,
            "matching_edges": matching_edges,
            "remaining_rows_before": remaining_rows,
            "singleton_floor_before": singleton_floor,
            "zero_floor_before": zero_floor,
            "candidate_mean_floor": candidate_mean,
            "matching_slack": matching_slack,
            "peeled_columns": peeled,
        }
        if index + 1 < rounds:
            next_remaining_rows = remaining_rows - 2 * matching_edges
            category_slack = math.sqrt(
                2 * remaining_rows * math.log(1 / tail_per_event)
            )
            singleton_floor = max(
                0.0,
                next_remaining_rows / remaining_rows * singleton_floor
                - category_slack
                - 4 * peeled,
            )
            zero_floor = max(
                0.0,
                next_remaining_rows / remaining_rows * zero_floor
                - category_slack,
            )
            remaining_rows = next_remaining_rows
            round_result["category_slack"] = category_slack
        trace.append(round_result)

    combined_failure_bound = (
        float(one_round["expansion_failure_bound"])
        + rounds * tail_per_event
        + 2 * (rounds - 1) * tail_per_event
    )
    return {
        "rounds": rounds,
        "tail_per_event": tail_per_event,
        "total_peeled_columns": total_peeled,
        "total_fraction_of_length": total_peeled / length,
        "total_fraction_of_deletion": total_peeled / deletion_size,
        "combined_failure_bound": combined_failure_bound,
        "trace": trace,
    }


def zero_slack_continuum_limit(
    one_round: dict[str, float | int]
) -> dict[str, float]:
    length = int(one_round["length"])
    deletion_size = int(one_round["deletion_size"])
    singleton_density = float(one_round["singleton_row_floor"]) / length
    zero_density = float(one_round["zero_row_floor"]) / length
    final_unexposed_density = 1 / 3
    blocks = length // 3
    recurrence_ceiling = (
        blocks
        * singleton_density
        * zero_density
        / 2
        * (blocks / (blocks - 1))
    )
    peeled_density = (
        zero_density
        * singleton_density
        / (4 * (zero_density + 1))
        * (1 - final_unexposed_density ** (zero_density + 1))
    )
    return {
        "all_partition_recurrence_ceiling_columns": recurrence_ceiling,
        "all_partition_recurrence_ceiling_fraction_of_deletion": (
            recurrence_ceiling / deletion_size
        ),
        "peeled_fraction_of_length": peeled_density,
        "peeled_fraction_of_deletion": peeled_density
        / (deletion_size / length),
        "interpretation": (
            "Large-N fluid limit of the present worst-case recurrence for a "
            "vanishing equal-reservoir mesh, with floors and concentration "
            "slacks removed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1_048_575)
    parser.add_argument("--size", type=int, default=138_726)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--terminal", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expansion-slack-fraction", type=float, default=0.02)
    parser.add_argument("--pairing-slack-fraction", type=float, default=0.02)
    parser.add_argument("--first-round-triple-fraction", type=float, default=0.5)
    parser.add_argument("--two-round-stopping-cap-fraction", type=float, default=0.017)
    parser.add_argument("--category-slack-fraction", type=float, default=0.005)
    parser.add_argument("--matching-tail", type=float, default=0.001)
    parser.add_argument("--multi-rounds", type=int, default=4)
    parser.add_argument("--multi-tail", type=float, default=0.0001)
    parser.add_argument("--verify-max-n", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.size <= args.length:
        raise SystemExit("--size must lie in 1..--length")
    expected_runs = args.size * (args.length - args.size + 1) / args.length
    runs = args.runs if args.runs is not None else round(expected_runs)
    if not 1 <= runs <= min(args.size, args.length - args.size + 1):
        raise SystemExit("--runs is infeasible")
    checked = verify_counts(args.verify_max_n)
    alpha = args.size / args.length
    rho = runs / args.length
    critical_sigma = 1 - math.sqrt(1 - 2 * alpha)
    one_round = evaluate_bound(
        args.length,
        args.size,
        runs,
        args.terminal,
        args.expansion_slack_fraction * args.length,
        args.pairing_slack_fraction * args.length,
    )
    result = {
        "status": "rigorous bound evaluated with floating-point arithmetic",
        "verified_finite_count_profiles": checked,
        "expected_runs_for_uniform_fixed_size_deletion": expected_runs,
        "profile_entropy_bits_per_coordinate": asymptotic_profile_entropy(alpha, rho),
        "all_deletions_entropy_bits_per_coordinate": binary_entropy(alpha),
        "positive_singleton_threshold_run_fraction": critical_sigma - alpha,
        "bound": one_round,
        "two_round_bound": evaluate_two_round_bound(
            one_round,
            args.first_round_triple_fraction,
            args.two_round_stopping_cap_fraction,
            args.category_slack_fraction,
            args.matching_tail,
        ),
        "multi_round_bound": evaluate_multi_round_bound(
            one_round, args.multi_rounds, args.multi_tail
        ),
        "zero_slack_continuum_limit": zero_slack_continuum_limit(one_round),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
