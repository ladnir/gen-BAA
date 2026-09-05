#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import diagnose_fourier_information_bound as projection
import explore_projected_kl_box_cover as cover
import projected_kl_dual as dual
import search_projected_kl_dual_envelope as envelope


def kl(actual: np.ndarray, comparison: np.ndarray) -> float:
    positive = actual > 0.0
    return float(
        np.dot(actual[positive], np.log(actual[positive] / comparison[positive]))
    )


class ProjectedKLDualTests(unittest.TestCase):
    def test_split_log_round_trip_and_ahead_trim(self) -> None:
        evaluated = (
            -0.1,
            0,
            np.full(15, 0.2),
            cover.Box(tuple([0.0] * 15), tuple([1.0] * 15)),
            -0.2,
            -0.15,
        )
        queue = [(-evaluated[0], 7, evaluated), (-evaluated[0], 11, evaluated)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            log = cover.load_or_initialize_split_log(path, 100, queue)
            self.assertEqual(log["base_identifiers"], [7, 11])
            log["records"].extend(
                [
                    {"after_split": 101},
                    {"after_split": 102},
                ]
            )
            cover.save_split_log(path, log)
            resumed = cover.load_or_initialize_split_log(path, 101, queue)
            self.assertEqual(
                [record["after_split"] for record in resumed["records"]],
                [101],
            )

    def test_lp_clipped_split_uses_middle_half(self) -> None:
        box = cover.Box(tuple([0.0] * 15), tuple([1.0] * 15))
        low_point = np.zeros(15)
        high_point = np.ones(15)
        left, right = cover.split(box, 3, low_point, "lp_clipped")
        self.assertEqual(left.upper[3], 0.25)
        self.assertEqual(right.lower[3], 0.25)
        left, right = cover.split(box, 3, high_point, "lp_clipped")
        self.assertEqual(left.upper[3], 0.75)
        self.assertEqual(right.lower[3], 0.75)

    def test_boxed_simplex_bilinear_remainder(self) -> None:
        lower = tuple([0.0] * 15)
        upper = tuple([0.6] * 15)
        box = cover.Box(lower, upper)
        left_vertices = cover.simplex_box_vertices(box, 0)
        right_vertices = cover.simplex_box_vertices(box, 5)
        self.assertTrue(
            np.allclose(left_vertices.sum(axis=1), 1.0, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(right_vertices.sum(axis=1), 1.0, atol=1e-12)
        )
        left_reference = cover.reference(box, 0)
        right_reference = cover.reference(box, 5)
        random = np.random.default_rng(7)
        matrix = np.exp(random.normal(size=(5, 5)))
        remainder = cover.bilinear_remainder_upper(
            matrix,
            left_reference,
            right_reference,
            left_vertices,
            right_vertices,
        )
        for _ in range(100):
            left_weights = random.dirichlet(np.ones(len(left_vertices)))
            right_weights = random.dirichlet(np.ones(len(right_vertices)))
            left = left_weights @ left_vertices
            right = right_weights @ right_vertices
            actual = float(left @ matrix @ right)
            affine_upper = float(
                left @ matrix @ right_reference
                + left_reference @ matrix @ right
                - left_reference @ matrix @ right_reference
                + remainder
            )
            self.assertLessEqual(actual, affine_upper + 1e-12)

    def test_branch_candidates_prioritize_rank_gap_blocks(self) -> None:
        point = np.full(15, 1.0 / 5.0)
        lower = np.zeros(15)
        upper = np.full(15, 0.1)
        upper[9] = 0.9
        evaluated = (
            0.125,
            4,
            point,
            cover.Box(tuple(lower), tuple(upper)),
            -0.25,
            0.0625,
        )
        candidates = cover.branch_candidates(evaluated, 3)
        self.assertEqual(candidates[0], 4)
        self.assertEqual(candidates[1:], [6, 7])
        self.assertEqual(len(set(candidates)), 3)

    def test_box_checkpoint_round_trip(self) -> None:
        evaluated = (
            0.125,
            7,
            np.linspace(0.0, 1.0, 15),
            cover.Box(tuple(np.zeros(15)), tuple(np.ones(15))),
            -0.25,
            0.0625,
        )
        signature = {"eta": 0.002, "delta": 0.1323, "bank_digest": "abc"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            cover.save_checkpoint(
                path,
                signature,
                19,
                23,
                [0.5, 0.25, 0.125],
                [(-0.125, 22, evaluated)],
            )
            splits, identifier, history, queue = cover.load_checkpoint(
                path, signature
            )
            stronger = dict(signature, bank_digest="def")
            cover.migrate_checkpoint_to_stronger_bank(path, signature, stronger)
            migrated = cover.load_checkpoint(path, stronger)
        self.assertEqual(splits, 19)
        self.assertEqual(identifier, 23)
        self.assertEqual(history, [0.5, 0.25, 0.125])
        self.assertEqual(queue[0][0:2], (-0.125, 22))
        restored = queue[0][2]
        self.assertEqual(restored[0:2], evaluated[0:2])
        np.testing.assert_array_equal(restored[2], evaluated[2])
        self.assertEqual(restored[3], evaluated[3])
        self.assertEqual(restored[4:6], evaluated[4:6])
        self.assertEqual(migrated[0:3], (splits, identifier, history))

    def test_dual_cut_is_below_every_projected_transition(self) -> None:
        random = np.random.default_rng(23)
        for _ in range(20):
            transition = random.dirichlet(np.ones(25)).reshape((5, 5))
            state = transition.sum(axis=1)
            target = transition.sum(axis=0)
            increment = np.array(
                [
                    sum(
                        transition[source, (source + difference) % 5]
                        for source in range(5)
                    )
                    for difference in range(5)
                ]
            )
            actual = np.zeros((2, 2, 5))
            for source in range(5):
                for output in range(5):
                    difference = (output - source) % 5
                    actual[
                        int(difference != 0), int(source != 0), output
                    ] += transition[source, output]
            comparison = projection.independent_support_sum_projection(
                increment, state
            )
            target_potential = np.array(
                [0.0, *random.normal(size=4)]
            )
            increment_support, state_support = random.normal(size=2)
            lower = dual.dual_value(
                increment,
                state,
                target,
                target_potential,
                float(increment_support),
                float(state_support),
            )
            self.assertLessEqual(lower, kl(actual, comparison) + 1e-12)

    def test_stage_gradient_matches_finite_difference(self) -> None:
        random = np.random.default_rng(29)
        increment = random.dirichlet(np.ones(5))
        state = random.dirichlet(np.ones(5))
        cut = dual.ProjectionDual(
            np.array([0.0, *random.normal(size=4)]),
            float(random.normal()),
            float(random.normal()),
            0.0,
            0.0,
            True,
        )
        _, increment_gradient, state_gradient = envelope.stage_upper_gradient(
            increment, state, cut
        )
        epsilon = 1e-7
        for vector, gradient, which in (
            (increment, increment_gradient, 0),
            (state, state_gradient, 1),
        ):
            for coordinate in range(5):
                plus = vector.copy()
                minus = vector.copy()
                plus[coordinate] += epsilon
                minus[coordinate] -= epsilon
                arguments_plus = (plus, state) if which == 0 else (increment, plus)
                arguments_minus = (minus, state) if which == 0 else (increment, minus)
                value_plus = envelope.stage_upper_gradient(
                    *arguments_plus, cut
                )[0]
                value_minus = envelope.stage_upper_gradient(
                    *arguments_minus, cut
                )[0]
                observed = (value_plus - value_minus) / (2.0 * epsilon)
                self.assertAlmostEqual(observed, gradient[coordinate], places=6)

    def test_multiplicative_orbit_preserves_stage_value(self) -> None:
        random = np.random.default_rng(31)
        increment = random.dirichlet(np.ones(5))
        state = random.dirichlet(np.ones(5))
        cut = dual.ProjectionDual(
            np.array([0.0, *random.normal(size=4)]),
            float(random.normal()),
            float(random.normal()),
            0.0,
            0.0,
            True,
        )
        expected = envelope.stage_upper_gradient(increment, state, cut)[0]
        for multiplier in range(1, 5):
            scaled_increment = dual.scale_symbol_vector(increment, multiplier)
            scaled_state = dual.scale_symbol_vector(state, multiplier)
            scaled_cut = dual.scale_projection_dual(cut, multiplier)
            observed = envelope.stage_upper_gradient(
                scaled_increment, scaled_state, scaled_cut
            )[0]
            self.assertAlmostEqual(observed, expected, places=12)


if __name__ == "__main__":
    unittest.main()
