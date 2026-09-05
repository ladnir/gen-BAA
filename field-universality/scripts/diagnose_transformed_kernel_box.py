#!/usr/bin/env python3
"""Test KL-transformed kernels in the projected moment box relaxation.

For c,t>0 and a positive kernel K,

  log E[K] <= c log E[exp(t K / c)] - log(t) - 1.

Scaling the transformed kernel absorbs the final constant.  Dividing a dual
cut's linear potentials by c then lets the existing evaluator optimize the
result with stage objective weight c.  This is a valid floating LP bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import explore_projected_kl_box_cover as projected
import projected_kl_dual as dual


def transform_cut(
    cut: dual.ProjectionDual,
    kernel: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    multiplier: float,
) -> tuple[dual.ProjectionDual, np.ndarray, float]:
    log_product = np.log(np.outer(left, right).ravel())
    flat_kernel = kernel.ravel()

    def objective(log_t: float) -> float:
        t = np.exp(log_t)
        return (
            multiplier
            * float(logsumexp(log_product + t * flat_kernel / multiplier))
            - log_t
            - 1.0
        )

    optimized = minimize_scalar(
        objective,
        bounds=(-12.0, 12.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not optimized.success:
        raise RuntimeError("failed to optimize transformed-kernel scale")
    t = float(np.exp(optimized.x))
    log_scale = (-np.log(t) - 1.0) / multiplier
    transformed_log_kernel = log_scale + t * kernel / multiplier
    normalization = float(np.max(transformed_log_kernel))
    transformed_kernel = np.exp(transformed_log_kernel - normalization)
    transformed_cut = dual.ProjectionDual(
        target_potential=(
            cut.target_potential / multiplier
            - normalization * np.ones_like(cut.target_potential)
        ),
        increment_support_potential=cut.increment_support_potential / multiplier,
        state_support_potential=cut.state_support_potential / multiplier,
        value=cut.value / multiplier,
        gradient_residual=cut.gradient_residual,
        optimizer_success=cut.optimizer_success,
    )
    return transformed_cut, transformed_kernel, t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument("--multiplier", type=float, default=64.0)
    parser.add_argument("--outer-iterations", type=int, default=32)
    args = parser.parse_args()
    bank_args = SimpleNamespace(
        eta=0.002,
        delta=0.22,
        bank_iterations=20,
        bank_restarts=12,
        seed=11,
        bank_cache=str(args.bank),
        rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    saved = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(saved["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    point = projected.sanitize_composition(np.asarray(selected["point"]))
    box = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))

    transformed_cuts = [[], []]
    transformed_kernels = [[], []]
    scales = [[], []]
    for stage, cuts, kernels, left, right in (
        (0, first_cuts, first_kernels, point[0:5], point[5:10]),
        (1, second_cuts, second_kernels, point[5:10], point[10:15]),
    ):
        for cut, kernel in zip(cuts, kernels, strict=True):
            transformed_cut, transformed_kernel, scale = transform_cut(
                cut, kernel, left, right, args.multiplier
            )
            transformed_cuts[stage].append(transformed_cut)
            transformed_kernels[stage].append(transformed_kernel)
            scales[stage].append(scale)

    evaluated = projected.evaluate(
        box,
        first_cuts,
        second_cuts,
        first_kernels,
        second_kernels,
        outer_iterations=args.outer_iterations,
        additional_weighted_first_cuts=tuple(
            (args.multiplier, cut, kernel)
            for cut, kernel in zip(
                transformed_cuts[0], transformed_kernels[0], strict=True
            )
        ),
        additional_weighted_second_cuts=tuple(
            (args.multiplier, cut, kernel)
            for cut, kernel in zip(
                transformed_cuts[1], transformed_kernels[1], strict=True
            )
        ),
    )
    if evaluated is None:
        raise RuntimeError("selected frontier box became infeasible")
    print(
        json.dumps(
            {
                "frontier_index": args.frontier_index,
                "multiplier": args.multiplier,
                "original_projected_upper": selected["bound"],
                "intersection_projected_upper": float(evaluated[0]),
                "original_upper_at_lp_composition": float(evaluated[4]),
                "rank_relaxed_upper_at_lp_composition": float(evaluated[5]),
                "split_dimension": int(evaluated[1]),
                "lp_composition": evaluated[2].tolist(),
                "first_t_range": [min(scales[0]), max(scales[0])],
                "second_t_range": [min(scales[1]), max(scales[1])],
                "status": "valid floating transformed-kernel LP; not an exact certificate",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
