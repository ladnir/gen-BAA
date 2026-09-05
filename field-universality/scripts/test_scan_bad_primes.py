from __future__ import annotations

import itertools
import unittest

import scan_bad_primes as scan


def brute_distance(matrix: list[list[int]], prime: int) -> int:
    dimension = len(matrix)
    length = len(matrix[0])
    return min(
        sum(
            sum(message[row] * matrix[row][column] for row in range(dimension))
            % prime
            != 0
            for column in range(length)
        )
        for message in itertools.product(range(prime), repeat=dimension)
        if any(message)
    )


class ScanBadPrimesTests(unittest.TestCase):
    def test_dimension_two_distance_matches_message_enumeration(self) -> None:
        for schedule in ("PAPA", "APAP"):
            for seed in (0, 1, 17):
                matrix = scan.build_repetition_three(2, schedule, seed)
                for prime in (2, 3, 5, 7):
                    self.assertEqual(
                        scan.dimension_two_distance(matrix, prime),
                        brute_distance(matrix, prime),
                    )

    def test_candidate_primes_cover_every_change_from_rational_data(self) -> None:
        test_primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
        for schedule in ("PAPA", "APAP"):
            for seed in (0, 1, 17):
                matrix = scan.build_repetition_three(2, schedule, seed)
                rational_distance = scan.dimension_two_distance(matrix, None)
                candidates = scan.candidate_odd_primes(matrix)
                for prime in test_primes:
                    if prime not in candidates:
                        self.assertEqual(
                            scan.dimension_two_distance(matrix, prime),
                            rational_distance,
                        )

    def test_zero_top_divisor_records_rational_rank_loss(self) -> None:
        matrix = [[1, 1, 0], [0, 0, 1]]
        self.assertEqual(scan.top_divisor(matrix, (0, 1)), 0)

    def test_sample_scan_reports_support_and_encoder_counts(self) -> None:
        result = scan.scan(blocks=2, samples=5, schedule="PAPA", seed=1)
        support_counts = result["bad_odd_primes"]
        sample_counts = result["bad_odd_prime_samples"]
        self.assertEqual(set(support_counts), set(sample_counts))
        for prime, samples in sample_counts.items():
            self.assertLessEqual(samples, support_counts[prime])

    def test_apap_length_twelve_p13_counterexample(self) -> None:
        matrix = scan.build_repetition_three(
            4, "APAP", 12380526508519931293
        )
        self.assertEqual(brute_distance(matrix, 2), 5)
        self.assertEqual(brute_distance(matrix, 13), 4)
        message = (0, 0, 1, 4)
        codeword = [
            sum(message[row] * matrix[row][column] for row in range(4)) % 13
            for column in range(12)
        ]
        self.assertEqual(
            codeword,
            [3, 0, 0, 0, 0, 0, 6, 2, 0, 0, 2, 0],
        )


if __name__ == "__main__":
    unittest.main()
