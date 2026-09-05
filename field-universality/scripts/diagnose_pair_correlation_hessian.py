#!/usr/bin/env python3
"""Probe nonproduct pair-type directions around the F_5 product saddle.

This is a numerical diagnostic, not a certificate.  Every perturbation has
zero row and column sums, so both one-replica compositions remain fixed.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

import diagnose_pair_replica_payment as pair
import optimize_bulk_exponent as bulk


def correlation_direction(
    marginal: np.ndarray, rng: np.random.Generator, rank: int = 4
) -> np.ndarray:
    """Return a tangent with zero row/column sums and bounded relative size."""
    direction = np.zeros((pair.PRIME, pair.PRIME))
    for _ in range(rank):
        left = rng.standard_normal(pair.PRIME)
        right = rng.standard_normal(pair.PRIME)
        left -= np.dot(marginal, left)
        right -= np.dot(marginal, right)
        direction += np.outer(marginal * left, marginal * right)
    product = np.outer(marginal, marginal)
    relative_size = np.max(np.abs(direction) / product)
    if relative_size == 0.0:
        raise RuntimeError("degenerate random correlation direction")
    return direction / relative_size


def pair_exponent(
    message: np.ndarray, middle: np.ndarray, output: np.ndarray
) -> tuple[float, list[dict[str, object]]]:
    first = pair.pair_stationary_payment(message, middle)
    second = pair.pair_stationary_payment(middle, output)
    exponent = (
        bulk.entropy(message) / 3.0
        - float(first["payment"])
        - float(second["payment"])
    )
    return exponent, [first, second]


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    optimized = bulk.optimize(
        SimpleNamespace(
            prime=pair.PRIME,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=args.seed,
            max_iterations=3000,
            tolerance=1e-11,
            sign_symmetric=False,
        )
    )
    marginals = [
        np.asarray(optimized["mu"]),
        np.asarray(optimized["beta"]),
        np.asarray(optimized["zeta"]),
    ]
    products = [np.outer(value, value) for value in marginals]
    baseline, baseline_stages = pair_exponent(
        products[0].ravel(), products[1].ravel(), products[2].ravel()
    )

    rng = np.random.default_rng(args.seed + 1)
    probes = []
    for sample in range(args.samples):
        directions = [correlation_direction(value, rng) for value in marginals]
        exponents = []
        residuals = []
        for sign in (-1.0, 1.0):
            perturbed = [
                product + sign * args.step * direction
                for product, direction in zip(products, directions)
            ]
            if min(float(np.min(value)) for value in perturbed) <= 0.0:
                raise RuntimeError("step left the positive simplex")
            value, stages = pair_exponent(
                perturbed[0].ravel(),
                perturbed[1].ravel(),
                perturbed[2].ravel(),
            )
            exponents.append(value)
            residuals.append(
                max(float(stage["equality_residual"]) for stage in stages)
            )
        slope = (exponents[1] - exponents[0]) / (2.0 * args.step)
        curvature = (exponents[1] - 2.0 * baseline + exponents[0]) / (
            args.step * args.step
        )
        probes.append(
            {
                "sample": sample,
                "minus_exponent": exponents[0],
                "plus_exponent": exponents[1],
                "central_slope": slope,
                "central_curvature": curvature,
                "max_equality_residual": max(residuals),
            }
        )

    return {
        "delta": args.delta,
        "step": args.step,
        "samples": args.samples,
        "single_exponent": optimized["exponent"],
        "product_pair_exponent": baseline,
        "product_doubling_error": baseline - 2.0 * optimized["exponent"],
        "baseline_stage_residuals": [
            stage["equality_residual"] for stage in baseline_stages
        ],
        "maximum_sampled_slope_magnitude": max(
            abs(value["central_slope"]) for value in probes
        ),
        "maximum_sampled_curvature": max(
            value["central_curvature"] for value in probes
        ),
        "minimum_sampled_curvature": min(
            value["central_curvature"] for value in probes
        ),
        "probes": probes,
        "status": "local numerical diagnostic; not a global pair certificate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.261)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
