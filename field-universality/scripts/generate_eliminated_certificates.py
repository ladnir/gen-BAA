#!/usr/bin/env python3
"""Generate numerical cut mixtures for the exact five-variable checker."""

from __future__ import annotations

import json

import numpy as np

from diagnose_dual_elimination import (
    branchwise_eliminated_upper,
    optimize_cut_mixture,
    order_multipliers,
    pure_cut_matrices,
)
from optimize_conditional_product_bound import entropy, support_information
from verify_conditional_product_bulk import default_cuts


REFERENCES = (
    ((0.002, 0.002813303962336616, 0.1323), (0.0009924297599807304, 0.0013463064097057693)),
    ((0.01, 0.01229912347945, 0.1323), (0.004920958852997632, 0.005794215928141168)),
    ((0.05, 0.0495582174099318, 0.1323), (0.02420947491599, 0.022886529490094412)),
    ((0.26, 0.1835336471409826, 0.1323), (0.12340366447329709, 0.08713621468109453)),
)


def generate() -> list[dict[str, object]]:
    cuts = default_cuts()
    records = []
    for supports, crossings in REFERENCES:
        first, second = pure_cut_matrices(cuts, supports, crossings)
        branchwise_value, branchwise = branchwise_eliminated_upper(
            first, second, supports[0]
        )
        value, mixture, result = optimize_cut_mixture(first, second, supports[0])
        support_term = entropy(np.array([1.0 - supports[0], supports[0]])) / 3.0
        support_term -= support_information(supports[0], supports[1], crossings[0])
        support_term -= support_information(supports[1], supports[2], crossings[1])
        cut_count = len(cuts)
        first_mix = np.tensordot(mixture[:cut_count], first, axes=(0, 0))
        records.append(
            {
                "supports": supports,
                "crossings": crossings,
                "conditional_upper": value,
                "support_term": support_term,
                "total_upper": support_term + value,
                "branchwise_conditional_upper": branchwise_value,
                "branchwise_total_upper": support_term + branchwise_value,
                "branchwise_certificates": branchwise,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "first_weights": [
                    [index, float(weight)]
                    for index, weight in enumerate(mixture[:cut_count])
                    if weight > 1e-9
                ],
                "second_weights": [
                    [index, float(weight)]
                    for index, weight in enumerate(mixture[cut_count : 2 * cut_count])
                    if weight > 1e-9
                ],
                "order_multipliers": [
                    order_multipliers(first_mix[:, middle], supports[0]).tolist()
                    for middle in range(4)
                ],
            }
        )
    return records


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
