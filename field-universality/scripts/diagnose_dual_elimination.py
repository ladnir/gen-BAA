#!/usr/bin/env python3
"""Test whether fixed KL dual cuts eliminate all conditional compositions."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import minimize

from optimize_conditional_product_bound import entropy, support_information
from verify_conditional_product_bulk import Cut, default_cuts, make_cut


def pure_cut_value(
    cut: Cut,
    difference: int,
    source: int,
    increment_support: float,
    state_support: float,
    crossing: float,
) -> float:
    cell = np.array([float(value) for value in cut.cell])
    target = np.array([float(value) for value in cut.target])
    correction = np.array(
        [
            [float(value.upper) for value in row]
            for row in cut.exponential_minus_one
        ]
    )
    joint = np.array(
        [
            1.0 - state_support - crossing,
            state_support - increment_support + crossing,
            crossing,
            increment_support - crossing,
        ]
    )
    image = (difference + source) % 5
    return float(
        cell @ joint
        + target[0] * (1.0 - state_support)
        + target[source] * state_support
        - correction[0, 0] * joint[0]
        - correction[1, source] * joint[1]
        - correction[2, difference] * joint[2]
        - correction[3, image] * joint[3]
    )


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def anchored_order_logsumexp(logits: np.ndarray) -> tuple[float, np.ndarray]:
    """Entropy conjugate on q[0] >= q[i] for every i."""
    pool = {0}
    while True:
        mean_logit = float(np.mean(logits[list(pool)]))
        violating = [
            index
            for index in range(len(logits))
            if index not in pool and logits[index] > mean_logit
        ]
        if not violating:
            break
        pool.add(max(violating, key=lambda index: logits[index]))
    mean_logit = float(np.mean(logits[list(pool)]))
    effective_logits = np.array(logits, copy=True)
    for index in pool:
        effective_logits[index] = mean_logit
    normalizer = logsumexp(effective_logits.tolist())
    probability = np.exp(effective_logits - normalizer)
    pooled_probability = sum(probability[index] for index in pool) / len(pool)
    for index in pool:
        probability[index] = pooled_probability
    return normalizer, probability


def eliminated_conditional_upper(
    first: Cut,
    second: Cut,
    supports: tuple[float, float, float],
    crossings: tuple[float, float],
) -> float:
    message, middle, output = supports
    first_crossing, second_crossing = crossings
    best = -math.inf
    for middle_symbol in range(1, 5):
        first_values = [
            pure_cut_value(
                first,
                difference,
                middle_symbol,
                message,
                middle,
                first_crossing,
            )
            for difference in range(1, 5)
        ]
        normalizer, _ = anchored_order_logsumexp(
            np.array([-3.0 * value / message for value in first_values])
        )
        message_value = message / 3.0 * normalizer
        for output_symbol in range(1, 5):
            second_value = pure_cut_value(
                second,
                middle_symbol,
                output_symbol,
                middle,
                output,
                second_crossing,
            )
            best = max(best, message_value - second_value)
    return best


def pure_cut_matrices(
    cuts: tuple[Cut, ...],
    supports: tuple[float, float, float],
    crossings: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    first = np.empty((len(cuts), 4, 4))
    second = np.empty((len(cuts), 4, 4))
    for cut_index, cut in enumerate(cuts):
        for difference in range(1, 5):
            for source in range(1, 5):
                first[cut_index, difference - 1, source - 1] = pure_cut_value(
                    cut,
                    difference,
                    source,
                    supports[0],
                    supports[1],
                    crossings[0],
                )
                second[cut_index, difference - 1, source - 1] = pure_cut_value(
                    cut,
                    difference,
                    source,
                    supports[1],
                    supports[2],
                    crossings[1],
                )
    return first, second


def optimize_cut_mixture(
    first: np.ndarray, second: np.ndarray, message_support: float
) -> tuple[float, np.ndarray, object]:
    cut_count = first.shape[0]
    variable_count = 2 * cut_count + 1

    def terms(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first_mix = np.tensordot(point[:cut_count], first, axes=(0, 0))
        second_mix = np.tensordot(
            point[cut_count : 2 * cut_count], second, axes=(0, 0)
        )
        values = np.empty((4, 4))
        gradients = np.zeros((4, 4, variable_count))
        for middle_symbol in range(4):
            logits = -3.0 * first_mix[:, middle_symbol] / message_support
            normalizer, probabilities = anchored_order_logsumexp(logits)
            message_value = message_support / 3.0 * normalizer
            first_gradient = -np.tensordot(
                first[:, :, middle_symbol], probabilities, axes=(1, 0)
            )
            for output_symbol in range(4):
                values[middle_symbol, output_symbol] = (
                    message_value - second_mix[middle_symbol, output_symbol]
                )
                gradients[middle_symbol, output_symbol, :cut_count] = (
                    first_gradient
                )
                gradients[
                    middle_symbol,
                    output_symbol,
                    cut_count : 2 * cut_count,
                ] = -second[:, middle_symbol, output_symbol]
        return values, gradients

    def inequalities(point: np.ndarray) -> np.ndarray:
        values, _ = terms(point)
        return point[-1] - values.ravel()

    def inequality_jacobian(point: np.ndarray) -> np.ndarray:
        _, gradients = terms(point)
        jacobian = -gradients.reshape(16, variable_count)
        jacobian[:, -1] = 1.0
        return jacobian

    initial = np.zeros(variable_count)
    initial[1] = 1.0
    initial[cut_count + 1] = 1.0
    values, _ = terms(initial)
    initial[-1] = float(np.max(values)) + 1e-6
    objective = np.zeros(variable_count)
    objective[-1] = 1.0
    equality = np.zeros((2, variable_count))
    equality[0, :cut_count] = 1.0
    equality[1, cut_count : 2 * cut_count] = 1.0
    result = minimize(
        lambda point: float(point[-1]),
        initial,
        jac=lambda point: objective,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * cut_count) + [(None, None)],
        constraints=(
            {
                "type": "eq",
                "fun": lambda point: equality @ point - 1.0,
                "jac": lambda point: equality,
            },
            {
                "type": "ineq",
                "fun": inequalities,
                "jac": inequality_jacobian,
            },
        ),
        options={"ftol": 1e-13, "maxiter": 3000},
    )
    final_values, _ = terms(result.x)
    return float(np.max(final_values)), result.x, result


def optimize_first_branch(
    first: np.ndarray, middle_symbol: int, message_support: float
) -> tuple[float, np.ndarray, object, np.ndarray]:
    cut_count = first.shape[0]

    def value_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        costs = np.tensordot(weights, first[:, :, middle_symbol], axes=(0, 0))
        logits = -3.0 * costs / message_support
        normalizer, probability = anchored_order_logsumexp(logits)
        value = message_support / 3.0 * normalizer
        gradient = -np.tensordot(
            first[:, :, middle_symbol], probability, axes=(1, 0)
        )
        return float(value), gradient, costs

    initial = np.full(cut_count, 1.0 / cut_count)
    result = minimize(
        lambda weights: value_and_gradient(weights)[0],
        initial,
        jac=lambda weights: value_and_gradient(weights)[1],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * cut_count,
        constraints=(
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda weights: np.ones(cut_count),
            },
        ),
        options={"ftol": 1e-13, "maxiter": 3000},
    )
    value, _, costs = value_and_gradient(result.x)
    return value, result.x, result, costs


def branchwise_eliminated_upper(
    first: np.ndarray, second: np.ndarray, message_support: float
) -> tuple[float, list[dict[str, object]]]:
    branches = []
    upper = -math.inf
    for middle_symbol in range(4):
        first_value, weights, result, costs = optimize_first_branch(
            first, middle_symbol, message_support
        )
        multipliers = order_multipliers(costs, message_support)
        for output_symbol in range(4):
            second_values = second[:, middle_symbol, output_symbol]
            second_cut = int(np.argmax(second_values))
            value = first_value - float(second_values[second_cut])
            upper = max(upper, value)
            branches.append(
                {
                    "middle_symbol": middle_symbol + 1,
                    "output_symbol": output_symbol + 1,
                    "conditional_upper": value,
                    "first_optimizer_success": bool(result.success),
                    "first_weights": [
                        [index, float(weight)]
                        for index, weight in enumerate(weights)
                        if weight > 1e-9
                    ],
                    "order_multipliers": multipliers.tolist(),
                    "second_cut": second_cut,
                }
            )
    return upper, branches


def mixture_adversary(
    first: np.ndarray,
    second: np.ndarray,
    mixture: np.ndarray,
    message_support: float,
) -> tuple[np.ndarray, int, int, float]:
    cut_count = first.shape[0]
    first_mix = np.tensordot(mixture[:cut_count], first, axes=(0, 0))
    second_mix = np.tensordot(
        mixture[cut_count : 2 * cut_count], second, axes=(0, 0)
    )
    best: tuple[float, np.ndarray, int, int] | None = None
    for middle_symbol in range(4):
        logits = -3.0 * first_mix[:, middle_symbol] / message_support
        normalizer, conditional_message = anchored_order_logsumexp(logits)
        message_value = message_support / 3.0 * normalizer
        for output_symbol in range(4):
            value = message_value - second_mix[middle_symbol, output_symbol]
            if best is None or value > best[0]:
                best = (
                    float(value),
                    conditional_message,
                    middle_symbol,
                    output_symbol,
                )
    if best is None:
        raise RuntimeError("no adversary")
    return best[1], best[2], best[3], best[0]


def smoothed_vertex(index: int, width: int, mixing: float = 1e-7) -> np.ndarray:
    vector = np.full(width, mixing / width)
    vector[index] += 1.0 - mixing
    return vector


def order_multipliers(costs: np.ndarray, message_support: float) -> np.ndarray:
    logits = -3.0 * costs / message_support
    pool = {0}
    while True:
        mean_logit = float(np.mean(logits[list(pool)]))
        violating = [
            index
            for index in range(len(logits))
            if index not in pool and logits[index] > mean_logit
        ]
        if not violating:
            break
        pool.add(max(violating, key=lambda index: logits[index]))
    mean_cost = float(np.mean(costs[list(pool)]))
    multipliers = np.zeros(len(costs))
    for index in pool:
        if index != 0:
            multipliers[index] = max(0.0, mean_cost - costs[index])
    return multipliers


def cutting_plane_mixture(
    initial_cuts: tuple[Cut, ...],
    supports: tuple[float, float, float],
    crossings: tuple[float, float],
    rounds: int = 8,
) -> tuple[float, tuple[Cut, ...], np.ndarray, object, list[dict[str, object]]]:
    cuts = list(initial_cuts)
    history: list[dict[str, object]] = []
    final_mixture: np.ndarray | None = None
    final_result: object | None = None
    final_value = math.inf
    for round_index in range(rounds + 1):
        first, second = pure_cut_matrices(tuple(cuts), supports, crossings)
        final_value, final_mixture, final_result = optimize_cut_mixture(
            first, second, supports[0]
        )
        message_conditional, middle_symbol, output_symbol, adversary_value = (
            mixture_adversary(
                first, second, final_mixture, supports[0]
            )
        )
        history.append(
            {
                "round": round_index,
                "cuts": len(cuts),
                "conditional_upper": final_value,
                "middle_symbol": middle_symbol + 1,
                "output_symbol": output_symbol + 1,
                "message_conditional": message_conditional.tolist(),
                "adversary_value": adversary_value,
            }
        )
        if round_index == rounds:
            break
        middle_conditional = smoothed_vertex(middle_symbol, 4)
        output_conditional = smoothed_vertex(output_symbol, 4)
        message = np.concatenate(
            ([1.0 - supports[0]], supports[0] * message_conditional)
        )
        middle = np.concatenate(
            ([1.0 - supports[1]], supports[1] * middle_conditional)
        )
        output = np.concatenate(
            ([1.0 - supports[2]], supports[2] * output_conditional)
        )
        cuts.append(make_cut(message, middle, crossings[0]))
        cuts.append(make_cut(middle, output, crossings[1]))
    if final_mixture is None or final_result is None:
        raise RuntimeError("cutting-plane mixture did not run")
    return final_value, tuple(cuts), final_mixture, final_result, history


def diagnose() -> dict[str, object]:
    mu = np.array(
        [
            0.9980000000000012,
            0.00010317395587027238,
            0.0006110650740864773,
            0.0011823776239572128,
            0.0001033833460848407,
        ]
    )
    beta = np.array(
        [
            0.9971866960376634,
            0.000024089511252901538,
            0.002364879746160249,
            0.00039798534996817093,
            0.000026349354955294586,
        ]
    )
    zeta = np.array(
        [
            0.8677000000000001,
            0.0006796742292351857,
            0.009602382386432106,
            0.1216227133119478,
            0.0003952300723848224,
        ]
    )
    crossings = (0.0009924297599807304, 0.0013463064097057693)
    supports = tuple(float(vector[1:].sum()) for vector in (mu, beta, zeta))
    message, middle, _ = supports
    phi = entropy(np.array([1.0 - message, message])) / 3.0
    phi -= support_information(message, middle, crossings[0])
    phi -= support_information(middle, supports[2], crossings[1])
    cuts = default_cuts()
    best: tuple[float, int, int] | None = None
    for first_index, first in enumerate(cuts):
        for second_index, second in enumerate(cuts):
            conditional = eliminated_conditional_upper(
                first, second, supports, crossings
            )
            value = phi + conditional
            if best is None or value < best[0]:
                best = (value, first_index, second_index)
    if best is None:
        raise RuntimeError("no cuts")
    mixture_value, enriched_cuts, mixture, mixture_result, history = (
        cutting_plane_mixture(cuts, supports, crossings, rounds=0)
    )
    cut_count = len(enriched_cuts)
    first_matrices, _ = pure_cut_matrices(
        enriched_cuts, supports, crossings
    )
    first_mix = np.tensordot(
        mixture[:cut_count], first_matrices, axes=(0, 0)
    )
    _, second_matrices = pure_cut_matrices(
        enriched_cuts, supports, crossings
    )
    branchwise_value, branchwise = branchwise_eliminated_upper(
        first_matrices, second_matrices, message
    )
    return {
        "support_term": phi,
        "supports": supports,
        "crossings": crossings,
        "fixed_cut_pair_upper": best[0],
        "first_cut": best[1],
        "second_cut": best[2],
        "mixed_cut_pair_upper": phi + mixture_value,
        "branchwise_cut_upper": phi + branchwise_value,
        "branchwise_certificates": branchwise,
        "mixture_optimizer_success": bool(mixture_result.success),
        "mixture_optimizer_message": str(mixture_result.message),
        "cutting_plane_history": history,
        "first_mixture": [
            [index, float(weight)]
            for index, weight in enumerate(mixture[:cut_count])
            if weight > 1e-8
        ],
        "second_mixture": [
            [index, float(weight)]
            for index, weight in enumerate(mixture[cut_count : 2 * cut_count])
            if weight > 1e-8
        ],
        "order_multipliers": [
            order_multipliers(first_mix[:, middle], message).tolist()
            for middle in range(4)
        ],
        "true_reduced_value_at_reference": -0.001461300855759,
    }


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(diagnose(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
