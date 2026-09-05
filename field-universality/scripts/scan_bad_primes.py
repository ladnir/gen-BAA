#!/usr/bin/env python3
"""Enumerate characteristics that lower a small encoder below its binary distance.

This is an exact small-instance diagnostic.  For every deletion set smaller than
the binary minimum distance, it computes the top determinantal divisor of the
retained integer generator.  A zero divisor records a characteristic-zero
dependency; the odd prime divisors of a nonzero divisor are precisely the
characteristics in which that deletion causes an additional rank loss.

For repetition three at length six, ``--exhaustive-interleavers`` instead uses
the dimension-two projective-line description to enumerate every ordered pair
of interleavers.  This specialized path is substantially faster than scanning
all deletion sets separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sympy import factorint

MASK64 = (1 << 64) - 1
SAMPLE_MULTIPLIER = 0xD6E8FEB86659FD93


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def permutation(self, length: int) -> list[int]:
        result = list(range(length))
        for index in range(length - 1, 0, -1):
            other = self.next_u64() % (index + 1)
            result[index], result[other] = result[other], result[index]
        return result


def sample_seed(master_seed: int, sample: int) -> int:
    random = SplitMix64(master_seed ^ ((sample * SAMPLE_MULTIPLIER) & MASK64))
    return random.next_u64()


def repetition_three(blocks: int) -> list[list[int]]:
    length = 3 * blocks
    matrix = [[0] * length for _ in range(blocks)]
    for block in range(blocks):
        matrix[block][3 * block : 3 * block + 3] = [1, 1, 1]
    return matrix


def permute_columns(
    matrix: list[list[int]], permutation: tuple[int, ...] | list[int]
) -> list[list[int]]:
    return [[row[source] for source in permutation] for row in matrix]


def accumulate(matrix: list[list[int]]) -> None:
    for row in matrix:
        for column in range(1, len(row)):
            row[column] += row[column - 1]


def build_repetition_three(blocks: int, schedule: str, seed: int) -> list[list[int]]:
    matrix = repetition_three(blocks)
    length = 3 * blocks
    random = SplitMix64(seed)
    for operation in schedule:
        if operation == "P":
            permutation = random.permutation(length)
            matrix = permute_columns(matrix, permutation)
        elif operation == "A":
            accumulate(matrix)
        else:
            raise ValueError("schedule must use only P and A")
    return matrix


def binary_distance(matrix: list[list[int]]) -> int:
    dimension = len(matrix)
    length = len(matrix[0])
    best = length + 1
    for message in range(1, 1 << dimension):
        weight = 0
        for column in range(length):
            symbol = 0
            for row in range(dimension):
                if message & (1 << row):
                    symbol ^= matrix[row][column] & 1
            weight += symbol
        best = min(best, weight)
    return best


def determinant(matrix: list[list[int]], columns: tuple[int, ...]) -> int:
    dimension = len(matrix)
    work = [[matrix[row][column] for column in columns] for row in range(dimension)]
    if dimension == 0:
        return 1
    sign = 1
    previous_pivot = 1
    for pivot_index in range(dimension):
        pivot_row = next(
            (row for row in range(pivot_index, dimension) if work[row][pivot_index]),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            sign = -sign
        if pivot_index + 1 == dimension:
            return sign * work[pivot_index][pivot_index]
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, dimension):
            leading = work[row][pivot_index]
            for column in range(pivot_index + 1, dimension):
                numerator = work[row][column] * pivot - leading * work[pivot_index][column]
                work[row][column] = numerator // previous_pivot
            work[row][pivot_index] = 0
        previous_pivot = pivot
    raise AssertionError("the final pivot returns")


def top_divisor(matrix: list[list[int]], retained: tuple[int, ...]) -> int:
    divisor = 0
    for columns in itertools.combinations(retained, len(matrix)):
        divisor = math.gcd(divisor, abs(determinant(matrix, columns)))
        if divisor == 1:
            break
    return divisor


@lru_cache(maxsize=None)
def odd_prime_factors(value: int) -> tuple[int, ...]:
    value = abs(value)
    while value and value % 2 == 0:
        value //= 2
    factors: list[int] = []
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 2
    if value > 1:
        factors.append(value)
    return tuple(factors)


def dimension_two_distance(matrix: list[list[int]], prime: int | None) -> int:
    """Return the exact distance of a full-rank two-row matrix over Q or F_p."""

    if len(matrix) != 2:
        raise ValueError("dimension-two distance requires exactly two rows")
    columns = list(zip(matrix[0], matrix[1], strict=True))
    if prime is not None:
        columns = [(left % prime, right % prime) for left, right in columns]
    zero_columns = sum(left == 0 and right == 0 for left, right in columns)
    largest_direction = 0
    for left_index, (left, right) in enumerate(columns):
        if left == 0 and right == 0:
            continue
        count = 0
        for other_left, other_right in columns[left_index:]:
            if other_left == 0 and other_right == 0:
                continue
            determinant_value = left * other_right - right * other_left
            if prime is not None:
                determinant_value %= prime
            count += determinant_value == 0
        # Include earlier columns on the same projective line.  This second
        # short loop avoids allocating normalized projective representatives.
        for other_left, other_right in columns[:left_index]:
            if other_left == 0 and other_right == 0:
                continue
            determinant_value = left * other_right - right * other_left
            if prime is not None:
                determinant_value %= prime
            count += determinant_value == 0
        largest_direction = max(largest_direction, count)
    return len(columns) - zero_columns - largest_direction


def candidate_odd_primes(matrix: list[list[int]]) -> set[int]:
    """Return every odd characteristic where projective column data can change."""

    columns = list(zip(matrix[0], matrix[1], strict=True))
    candidates: set[int] = set()
    for left, right in columns:
        candidates.update(odd_prime_factors(math.gcd(left, right)))
    for first, (left, right) in enumerate(columns):
        for other_left, other_right in columns[first + 1 :]:
            candidates.update(
                odd_prime_factors(left * other_right - right * other_left)
            )
    return candidates


def apply_accumulators(matrix: list[list[int]], count: int) -> None:
    for _ in range(count):
        accumulate(matrix)


def exhaustive_length_six(schedule: str) -> dict[str, object]:
    if schedule.count("P") != 2:
        raise ValueError("exhaustive interleavers require exactly two P operations")
    base = repetition_three(2)
    before_first, between, after_second = schedule.split("P")
    if set(before_first + between + after_second) - {"A"}:
        raise ValueError("schedule must use only P and A")
    apply_accumulators(base, len(before_first))
    permutations = list(itertools.permutations(range(6)))
    prime_counts: Counter[int] = Counter()
    binary_distances: Counter[int] = Counter()
    rational_distances: Counter[int] = Counter()
    prime_examples: dict[int, dict[str, object]] = {}
    structural_pairs = 0
    structural_example: dict[str, object] | None = None
    largest_candidate_prime = 1
    candidate_primes_seen: set[int] = set()
    maximum_abs_determinant = 0
    pair_index = 0
    for first in permutations:
        after_first = permute_columns(base, first)
        apply_accumulators(after_first, len(between))
        for second in permutations:
            matrix = permute_columns(after_first, second)
            apply_accumulators(matrix, len(after_second))
            binary = dimension_two_distance(matrix, 2)
            rational = dimension_two_distance(matrix, None)
            binary_distances[binary] += 1
            rational_distances[rational] += 1
            if rational < binary:
                structural_pairs += 1
                if structural_example is None:
                    structural_example = {
                        "pair_index": pair_index,
                        "binary_distance": binary,
                        "rational_distance": rational,
                        "first_permutation": first,
                        "second_permutation": second,
                        "encoder": matrix,
                    }
            columns = list(zip(matrix[0], matrix[1], strict=True))
            for left, right in columns:
                maximum_abs_determinant = max(
                    maximum_abs_determinant, abs(math.gcd(left, right))
                )
            for column, (left, right) in enumerate(columns):
                for other_left, other_right in columns[column + 1 :]:
                    maximum_abs_determinant = max(
                        maximum_abs_determinant,
                        abs(left * other_right - right * other_left),
                    )
            candidates = candidate_odd_primes(matrix)
            candidate_primes_seen.update(candidates)
            for prime in candidates:
                largest_candidate_prime = max(largest_candidate_prime, prime)
                distance = dimension_two_distance(matrix, prime)
                if distance >= binary:
                    continue
                prime_counts[prime] += 1
                prime_examples.setdefault(
                    prime,
                    {
                        "pair_index": pair_index,
                        "binary_distance": binary,
                        "prime_distance": distance,
                        "rational_distance": rational,
                        "first_permutation": first,
                        "second_permutation": second,
                        "encoder": matrix,
                    },
                )
            pair_index += 1
    return {
        "status": "complete enumeration of every ordered interleaver pair",
        "blocks": 2,
        "length": 6,
        "schedule": schedule,
        "interleavers_per_stage": len(permutations),
        "ordered_interleaver_pairs": pair_index,
        "binary_distances": dict(sorted(binary_distances.items())),
        "rational_distances": dict(sorted(rational_distances.items())),
        "characteristic_zero_distance_beating_pairs": structural_pairs,
        "characteristic_zero_example": structural_example,
        "bad_odd_prime_pair_counts": dict(sorted(prime_counts.items())),
        "bad_prime_examples": dict(sorted(prime_examples.items())),
        "candidate_odd_primes": sorted(candidate_primes_seen),
        "largest_candidate_odd_prime": largest_candidate_prime,
        "maximum_abs_projective_determinant_or_content": maximum_abs_determinant,
    }


def scan(blocks: int, samples: int, schedule: str, seed: int) -> dict[str, object]:
    length = 3 * blocks
    prime_counts: Counter[int] = Counter()
    prime_sample_counts: Counter[int] = Counter()
    divisor_counts: Counter[int] = Counter()
    deletion_sets = 0
    torsion_sets = 0
    structural_sets = 0
    structural_samples = 0
    binary_distances: Counter[int] = Counter()
    prime_examples: dict[int, dict[str, object]] = {}
    structural_example: dict[str, object] | None = None
    largest_divisor = 1
    for sample in range(samples):
        encoder = build_repetition_three(blocks, schedule, sample_seed(seed, sample))
        distance = binary_distance(encoder)
        binary_distances[distance] += 1
        sample_bad_primes: set[int] = set()
        sample_has_structural_dependency = False
        coordinates = range(length)
        for deleted_size in range(distance):
            for deleted in itertools.combinations(coordinates, deleted_size):
                deleted_set = set(deleted)
                retained = tuple(index for index in coordinates if index not in deleted_set)
                divisor = top_divisor(encoder, retained)
                deletion_sets += 1
                if divisor == 0:
                    structural_sets += 1
                    sample_has_structural_dependency = True
                    if structural_example is None:
                        structural_example = {
                            "sample": sample,
                            "sample_seed": sample_seed(seed, sample),
                            "binary_distance": distance,
                            "deleted": list(deleted),
                        }
                    continue
                if divisor == 1:
                    continue
                torsion_sets += 1
                divisor_counts[divisor] += 1
                largest_divisor = max(largest_divisor, divisor)
                for prime in factorint(divisor):
                    if prime != 2:
                        odd_prime = int(prime)
                        prime_counts[odd_prime] += 1
                        sample_bad_primes.add(odd_prime)
                        prime_examples.setdefault(
                            odd_prime,
                            {
                                "sample": sample,
                                "sample_seed": sample_seed(seed, sample),
                                "binary_distance": distance,
                                "deleted": list(deleted),
                                "top_divisor": divisor,
                            },
                        )
        prime_sample_counts.update(sample_bad_primes)
        structural_samples += sample_has_structural_dependency
    return {
        "blocks": blocks,
        "length": length,
        "samples": samples,
        "schedule": schedule,
        "seed": seed,
        "deletion_sets": deletion_sets,
        "torsion_sets": torsion_sets,
        "characteristic_zero_dependency_sets": structural_sets,
        "characteristic_zero_dependency_samples": structural_samples,
        "characteristic_zero_example": structural_example,
        "binary_distances": dict(sorted(binary_distances.items())),
        "top_divisors": dict(sorted(divisor_counts.items())),
        "bad_odd_primes": dict(sorted(prime_counts.items())),
        "bad_odd_prime_samples": dict(sorted(prime_sample_counts.items())),
        "bad_prime_examples": dict(sorted(prime_examples.items())),
        "largest_top_divisor": largest_divisor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--schedule", default="PAPA")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exhaustive-interleavers", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.blocks <= 0 or args.samples <= 0:
        parser.error("blocks and samples must be positive")
    if args.exhaustive_interleavers:
        if args.blocks != 2:
            parser.error("--exhaustive-interleavers requires --blocks 2")
        result = exhaustive_length_six(args.schedule)
    else:
        result = scan(args.blocks, args.samples, args.schedule, args.seed)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
