#!/usr/bin/env python3
"""Stress-test target-mismatch coercivity globally and near its zero set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analyze_zero_payment_surface as zero
import optimize_conditional_product_bound as bound


def optimize(
    projection: str,
    maximum_zero_defect: float | None,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    return bound.optimize(
        SimpleNamespace(
            prime=5,
            repetition=3,
            delta=args.delta,
            eta=args.eta,
            restarts=args.restarts,
            seed=seed,
            max_iterations=args.max_iterations,
            tolerance=1e-11,
            support_lower=None,
            support_upper=None,
            projection=projection,
            sign_symmetric=False,
            maximum_zero_defect=maximum_zero_defect,
        )
    )


def compact(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "projection",
        "maximum_upper_bound_found",
        "maximum_zero_payment_defect",
        "zero_payment_defect",
        "message_nonzero_fraction",
        "middle_nonzero_fraction",
        "output_nonzero_fraction",
        "first_crossing_fraction",
        "second_crossing_fraction",
        "first_projected_divergence",
        "second_projected_divergence",
        "first_target_divergence",
        "second_target_divergence",
        "first_stationary_divergence",
        "second_stationary_divergence",
        "first_cell_target_divergence",
        "second_cell_target_divergence",
        "optimizer_success",
        "optimizer_message",
    )
    compacted = {}
    for key in keys:
        value = result[key]
        if isinstance(value, float) and not np.isfinite(value):
            value = "infinity" if value > 0.0 else "negative-infinity"
        compacted[key] = value
    return compacted


def run(args: argparse.Namespace) -> dict[str, object]:
    global_results = {}
    projection_seeds = {
        "fourier-coercive": args.seed,
        "target-l2": args.seed + 1,
        "target": args.seed + 2,
        # This seed reaches the known two-orbit positive witness reliably.
        "two-level": 43,
    }
    for projection in ("fourier-coercive", "target-l2", "target", "two-level"):
        global_results[projection] = compact(
            optimize(projection, None, args, projection_seeds[projection])
        )
    tube_results = {}
    for offset, threshold in enumerate(args.thresholds):
        tube_results[format(threshold, ".17g")] = compact(
            optimize(
                "fourier-coercive",
                threshold,
                args,
                args.seed + 100 + offset,
            )
        )
    normalized_tube = compact(
        optimize(
            "two-level",
            args.normalized_tube_threshold,
            args,
            43,
        )
    )
    spectral_gap = 1.0 - np.cos(2.0 * np.pi / 5.0)
    report = {
        "prime": 5,
        "repetition": 3,
        "eta": args.eta,
        "delta": args.delta,
        "fourier_coercivity": {
            "spectral_gap": float(spectral_gap),
            "squared_coefficient": float(spectral_gap**2),
            "statement": "D(beta||K) >= (1-cos(2*pi/5))^2 * zero_payment_defect",
        },
        "global_searches": global_results,
        "tube_searches": tube_results,
        "normalized_tube_search": normalized_tube,
        "restarts_per_search": args.restarts,
        "status": "floating constrained multistart diagnostics; not global certificates",
    }
    zero.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(1e-8, 1e-7, 1e-6),
    )
    parser.add_argument("--normalized-tube-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-fourier-coercivity.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))
