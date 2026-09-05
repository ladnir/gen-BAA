#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import diagnose_joint_information_projection as joint_projection
import optimize_joint_projection_bound as joint_optimizer


class JointInformationProjectionTests(unittest.TestCase):
    def test_outer_constraints_do_not_choose_a_distinguished_nonzero_symbol(
        self,
    ) -> None:
        """Multiplication by a nonzero field element may relabel a maximizer."""
        point = np.array(
            [
                0.998, 0.0, 0.001, 0.001, 0.0,
                0.9972, 0.0001, 0.0013, 0.0013, 0.0001,
                0.78, 0.02, 0.09, 0.09, 0.02,
                0.001, 0.001,
            ]
        )
        equality, target, upper, upper_target = (
            joint_optimizer.outer_constraints(0.002, 0.22)
        )
        self.assertLess(np.max(np.abs(equality @ point - target)), 1e-12)
        self.assertLessEqual(np.max(upper @ point - upper_target), 1e-12)

    def test_outer_envelope_gradient(self) -> None:
        point = np.array(
            [
                0.7102533, 0.1252767, 0.0195967, 0.0195966, 0.1252767,
                0.7215541, 0.1245347, 0.0146882, 0.0146882, 0.1245348,
                0.7400000, 0.1080855, 0.0219146, 0.0219145, 0.1080855,
                0.1302999, 0.1213764,
            ]
        )
        direction = np.array(
            [
                -0.3, 0.2, -0.1, 0.05, 0.15,
                0.1, -0.2, 0.05, 0.15, -0.1,
                -0.2, 0.1, 0.05, -0.05, 0.1,
                0.07, -0.04,
            ]
        )
        value, gradient, diagnostic = joint_optimizer.objective_value_gradient(
            point
        )
        self.assertTrue(diagnostic["feasible"])
        step = 1e-5
        plus, _ = joint_optimizer.objective(point + step * direction)
        minus, _ = joint_optimizer.objective(point - step * direction)
        finite_difference = (plus - minus) / (2.0 * step)
        self.assertAlmostEqual(
            float(np.dot(gradient, direction)), finite_difference, places=5
        )
        self.assertTrue(np.isfinite(value))

    def test_independent_uniform_state_has_zero_payment(self) -> None:
        increment = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        state = np.full(5, 0.2)
        crossing = float(increment[1:].sum() * state[0])
        payment = joint_projection.stage_payment(increment, state, crossing)
        self.assertTrue(payment["projection"]["feasible"])
        self.assertAlmostEqual(payment["total_payment"], 0.0, places=9)
        self.assertLess(abs(payment["projection"]["duality_gap"]), 1e-9)

    def test_projection_payment_is_below_transition_information(self) -> None:
        prime = 5
        transition = np.zeros((prime, prime))
        for source in range(prime):
            transition[source, source] += 0.2 * 0.7
            transition[source, (2 * source) % prime] += 0.2 * 0.3
        joint = np.zeros_like(transition)
        for source in range(prime):
            for target in range(prime):
                joint[(target - source) % prime, source] += transition[
                    source, target
                ]
        increment = joint.sum(axis=1)
        state = joint.sum(axis=0)
        crossing = float(joint[1:, 0].sum())
        product = np.outer(increment, state)
        information = joint_projection.relative_entropy(
            joint.ravel(), product.ravel()
        )
        payment = joint_projection.stage_payment(increment, state, crossing)
        self.assertTrue(payment["projection"]["feasible"])
        self.assertLessEqual(payment["total_payment"], information + 1e-9)
        self.assertLess(abs(payment["projection"]["duality_gap"]), 1e-8)
        self.assertLess(
            payment["projection"]["dual"]["equality_residual"], 5e-8
        )


if __name__ == "__main__":
    unittest.main()
