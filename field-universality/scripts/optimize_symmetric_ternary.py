#!/usr/bin/env python3
"""Global numerical search inside the sign-symmetric ternary type family."""

from __future__ import annotations

import argparse
import json
import math

from scipy.optimize import differential_evolution, minimize_scalar


def xlogx(value: float) -> float:
    return 0.0 if value <= 0.0 else value * math.log(value)


def ternary_composition_entropy(nonzero_fraction: float) -> float:
    zero = 1.0 - nonzero_fraction
    half = nonzero_fraction / 2.0
    return -xlogx(zero) - 2.0 * xlogx(half)


def transition_entries(increment_weight: float, state_weight: float, bridge: float) -> tuple[float, ...]:
    diagonal_zero = 1.0 - state_weight - 2.0 * bridge
    diagonal_nonzero = (state_weight - increment_weight + 2.0 * bridge) / 2.0
    off_diagonal_nonzero = increment_weight / 2.0 - 2.0 * bridge
    return diagonal_zero, bridge, diagonal_nonzero, off_diagonal_nonzero


def transition_rate(increment_weight: float, state_weight: float) -> tuple[float, float]:
    lower = max(0.0, (increment_weight - state_weight) / 2.0)
    upper = min((1.0 - state_weight) / 2.0, increment_weight / 4.0)
    if lower > upper + 1e-14:
        return -math.inf, math.nan

    def rate(bridge: float) -> float:
        diagonal_zero, bridge, diagonal_nonzero, off_diagonal_nonzero = transition_entries(
            increment_weight, state_weight, bridge
        )
        if min(diagonal_zero, bridge, diagonal_nonzero, off_diagonal_nonzero) < -1e-12:
            return -math.inf
        transition_entropy = (
            -xlogx(max(0.0, diagonal_zero))
            - 4.0 * xlogx(max(0.0, bridge))
            - 2.0 * xlogx(max(0.0, diagonal_nonzero))
            - 2.0 * xlogx(max(0.0, off_diagonal_nonzero))
        )
        return transition_entropy - ternary_composition_entropy(state_weight)

    if upper - lower <= 1e-14:
        return rate((lower + upper) / 2.0), (lower + upper) / 2.0
    result = minimize_scalar(
        lambda bridge: -rate(bridge),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-15, "maxiter": 1000},
    )
    candidates = [(rate(lower), lower), (rate(upper), upper), (rate(result.x), result.x)]
    return max(candidates)


def exponent(message_weight: float, middle_weight: float, output_weight: float, repetition: int) -> float:
    first, _ = transition_rate(message_weight, middle_weight)
    second, _ = transition_rate(middle_weight, output_weight)
    if not math.isfinite(first) or not math.isfinite(second):
        return -math.inf
    message_entropy = ternary_composition_entropy(message_weight)
    middle_entropy = ternary_composition_entropy(middle_weight)
    return message_entropy / repetition + first - message_entropy + second - middle_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.repetition < 1:
        parser.error("repetition must be positive")
    if not 0.0 < args.eta < 1.0 or not 0.0 < args.delta < 1.0:
        parser.error("eta and delta must lie in (0,1)")
    return args


def main() -> None:
    args = parse_args()
    result = differential_evolution(
        lambda point: -exponent(point[0], point[1], point[2], args.repetition),
        bounds=((args.eta, 1.0), (0.0, 1.0), (0.0, args.delta)),
        seed=args.seed,
        popsize=24,
        maxiter=args.max_iterations,
        tol=1e-11,
        polish=True,
        workers=1,
        updating="immediate",
    )
    message_weight, middle_weight, output_weight = result.x
    first_rate, first_bridge = transition_rate(message_weight, middle_weight)
    second_rate, second_bridge = transition_rate(middle_weight, output_weight)
    print(
        json.dumps(
            {
                "repetition": args.repetition,
                "delta": args.delta,
                "eta": args.eta,
                "exponent": -result.fun,
                "message_nonzero_fraction": message_weight,
                "middle_nonzero_fraction": middle_weight,
                "output_nonzero_fraction": output_weight,
                "first_transition_rate": first_rate,
                "second_transition_rate": second_rate,
                "first_bridge": first_bridge,
                "second_bridge": second_bridge,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
