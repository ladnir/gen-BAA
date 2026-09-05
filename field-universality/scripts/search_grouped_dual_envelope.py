#!/usr/bin/env python3
"""Numerically maximize a cutting-plane envelope for the grouped dual.

This is a discovery tool, not a certificate.  Each grouped cut globally
upper-bounds the transition-eliminated Lagrangian.  The minimum of a finite
bank of cuts is therefore still an upper bound.  Constraint generation adds
a cut fitted at the current envelope maximizer.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import diagnose_lagrangian_entropy_dual as grouped
import optimize_bulk_exponent as bulk
import search_internal_information_gap as gap_search


def entropy_gradient(distribution: np.ndarray) -> np.ndarray:
    return -(np.log(np.maximum(distribution, 1e-300)) + 1.0)


def binary_entropy_derivative(value: float) -> float:
    clipped = min(max(value, 1e-300), 1.0 - 1e-15)
    return float(np.log((1.0 - clipped) / clipped))


def cut_value_gradient(
    point: np.ndarray,
    layout: bulk.Layout,
    repetition: int,
    multiplier: float,
    pair: tuple[grouped.GroupedCut, grouped.GroupedCut],
) -> tuple[float, np.ndarray]:
    first, second = pair
    value = grouped.lagrangian_cut_value(
        point, layout, repetition, multiplier, first, second
    )
    gradient = np.zeros(layout.size)
    gradient[layout.mu] += first.increment_potential
    gradient[layout.beta] += (
        first.row_potential
        + first.column_potential
        + second.increment_potential
    )
    gradient[layout.zeta] += second.row_potential + second.column_potential

    message = point[layout.mu]
    middle = point[layout.beta]
    output = point[layout.zeta]
    gradient[layout.mu] -= (
        (1.0 + multiplier)
        * (1.0 - 1.0 / repetition)
        * entropy_gradient(message)
    )
    gradient[layout.beta] -= (
        2.0 * (1.0 + multiplier) * entropy_gradient(middle)
    )
    gradient[layout.zeta] -= (
        (1.0 + multiplier) * entropy_gradient(output)
    )

    support_terms = (
        (
            layout.mu,
            multiplier * (1.0 - 1.0 / repetition),
            float(message[1:].sum()),
        ),
        (layout.beta, 2.0 * multiplier, float(middle[1:].sum())),
        (layout.zeta, multiplier, float(output[1:].sum())),
    )
    for component, coefficient, support in support_terms:
        gradient[component.start + 1 : component.stop] += (
            coefficient * binary_entropy_derivative(support)
        )
    return value, gradient


def fit_pair(
    point: np.ndarray, layout: bulk.Layout, multiplier: float
) -> tuple[grouped.GroupedCut, grouped.GroupedCut]:
    return (
        grouped.fit_grouped_cut(
            point[layout.mu], point[layout.beta], multiplier
        ),
        grouped.fit_grouped_cut(
            point[layout.beta], point[layout.zeta], multiplier
        ),
    )


def envelope_value_gradient(
    extended: np.ndarray,
    layout: bulk.Layout,
    repetition: int,
    multiplier: float,
    cuts: list[tuple[grouped.GroupedCut, grouped.GroupedCut]],
) -> tuple[np.ndarray, np.ndarray]:
    point = extended[:-1]
    threshold = extended[-1]
    values = np.empty(len(cuts))
    jacobian = np.zeros((len(cuts), layout.size + 1))
    for index, pair in enumerate(cuts):
        value, gradient = cut_value_gradient(
            point, layout, repetition, multiplier, pair
        )
        values[index] = value - threshold
        jacobian[index, :-1] = gradient
        jacobian[index, -1] = -1.0
    return values, jacobian


def maximize_envelope(
    args: argparse.Namespace,
    cuts: list[tuple[grouped.GroupedCut, grouped.GroupedCut]],
    starts: list[np.ndarray],
    equality: np.ndarray,
    equality_target: np.ndarray,
    upper: np.ndarray,
    upper_target: np.ndarray,
) -> object:
    layout = bulk.Layout(args.prime)
    extended_equality = np.pad(equality, ((0, 0), (0, 1)))
    extended_upper = np.pad(upper, ((0, 0), (0, 1)))

    def objective(extended: np.ndarray) -> float:
        return -float(extended[-1])

    def objective_gradient(extended: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(extended)
        gradient[-1] = -1.0
        return gradient

    constraints = [
        LinearConstraint(
            extended_equality, equality_target, equality_target
        ),
        LinearConstraint(extended_upper, -np.inf, upper_target),
        {
            "type": "ineq",
            "fun": lambda extended: envelope_value_gradient(
                extended,
                layout,
                args.repetition,
                args.gap_multiplier,
                cuts,
            )[0],
            "jac": lambda extended: envelope_value_gradient(
                extended,
                layout,
                args.repetition,
                args.gap_multiplier,
                cuts,
            )[1],
        },
    ]
    best = None
    for start in starts:
        values = [
            grouped.lagrangian_cut_value(
                start,
                layout,
                args.repetition,
                args.gap_multiplier,
                pair[0],
                pair[1],
            )
            for pair in cuts
        ]
        extended_start = np.concatenate((start, [min(values) - 1e-10]))
        result = minimize(
            objective,
            extended_start,
            jac=objective_gradient,
            method="SLSQP",
            bounds=Bounds(
                np.concatenate((np.zeros(layout.size), [-20.0])),
                np.concatenate((np.ones(layout.size), [20.0])),
            ),
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": 3000, "disp": False},
        )
        point = result.x[:-1]
        equality_error = float(
            np.max(np.abs(equality @ point - equality_target))
        )
        upper_error = float(np.max(upper @ point - upper_target))
        envelope_error = float(
            np.min(
                envelope_value_gradient(
                    result.x,
                    layout,
                    args.repetition,
                    args.gap_multiplier,
                    cuts,
                )[0]
            )
        )
        if (
            equality_error <= 1e-7
            and upper_error <= 1e-7
            and envelope_error >= -1e-7
            and (best is None or result.fun < best.fun)
        ):
            best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible envelope point")
    return best


def search(args: argparse.Namespace) -> dict[str, object]:
    layout = bulk.Layout(args.prime)
    equality, equality_target, upper, upper_target = bulk.constraints(
        layout, args.eta, args.delta
    )
    random = np.random.default_rng(args.seed)
    vertices = bulk.feasible_vertices(
        layout,
        equality,
        equality_target,
        upper,
        upper_target,
        max(64, 2 * args.restarts),
        random,
    )
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(
            len(vertices), size=min(16, len(vertices)), replace=True
        )
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(
                weight * vertices[index]
                for weight, index in zip(weights, chosen, strict=True)
            )
        )

    numerical = gap_search.search(
        SimpleNamespace(
            prime=args.prime,
            repetition=args.repetition,
            delta=args.delta,
            eta=args.eta,
            restarts=max(args.restarts, 16),
            seed=args.seed + 1000,
            max_message_weight=None,
            mode="lagrangian",
            maximum_gap=0.0,
            gap_multiplier=args.gap_multiplier,
        )
    )
    initial_point = np.zeros(layout.size)
    initial_point[layout.mu] = numerical["mu"]
    initial_point[layout.beta] = numerical["beta"]
    initial_point[layout.zeta] = numerical["zeta"]
    initial_point[layout.tau1] = np.asarray(numerical["tau1"]).reshape(-1)
    initial_point[layout.tau2] = np.asarray(numerical["tau2"]).reshape(-1)
    starts.insert(0, initial_point)
    cuts = [fit_pair(initial_point, layout, args.gap_multiplier)]
    history = []
    result = None
    for iteration in range(args.iterations):
        result = maximize_envelope(
            args,
            cuts,
            starts,
            equality,
            equality_target,
            upper,
            upper_target,
        )
        point = result.x[:-1]
        true_exponent = bulk.exponent(point, layout, args.repetition)
        gap = gap_search.gap_and_gradient(
            point, layout, args.repetition
        )[0]
        true_lagrangian = true_exponent - args.gap_multiplier * gap
        envelope = float(result.x[-1])
        history.append(
            {
                "iteration": iteration + 1,
                "cuts": len(cuts),
                "envelope_upper": envelope,
                "true_lagrangian_at_maximizer": true_lagrangian,
                "duality_gap": envelope - true_lagrangian,
                "message": point[layout.mu].tolist(),
                "middle": point[layout.beta].tolist(),
                "output": point[layout.zeta].tolist(),
                "optimizer_success": bool(result.success),
            }
        )
        if envelope - true_lagrangian <= args.tolerance:
            break
        cuts.append(fit_pair(point, layout, args.gap_multiplier))
        starts.insert(0, point)
    assert result is not None
    return {
        "prime": args.prime,
        "delta": args.delta,
        "eta": args.eta,
        "gap_multiplier": args.gap_multiplier,
        "iterations": len(history),
        "final_envelope_upper": history[-1]["envelope_upper"],
        "required_upper": -args.gap_multiplier * args.maximum_gap,
        "numerically_closes_target": (
            history[-1]["envelope_upper"]
            < -args.gap_multiplier * args.maximum_gap
        ),
        "history": history,
        "status": "numerical cutting-plane diagnostic only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--gap-multiplier", type=float, default=1.5)
    parser.add_argument("--maximum-gap", type=float, default=0.0001)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(search(parse_args()), indent=2, sort_keys=True))
