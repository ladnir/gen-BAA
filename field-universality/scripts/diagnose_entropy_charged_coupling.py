#!/usr/bin/env python3
"""Test mutual-information penalties for the lifted product coupling.

For fixed marginals alpha and beta and a positive kernel K, the quantity

  max_P log(<K,P>) - c D(P || alpha tensor beta)

over couplings P is an upper bound on log(alpha^T K beta).  This script is a
floating point diagnostic and does not certify the inner maximization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

import explore_projected_kl_box_cover as projected


def charged_partition_upper(
    left: np.ndarray,
    right: np.ndarray,
    kernel: np.ndarray,
    multiplier: float,
) -> dict[str, object]:
    product = np.outer(left, right)
    equality = []
    targets = []
    for row in range(5):
        coefficients = np.zeros((5, 5))
        coefficients[row, :] = 1.0
        equality.append(coefficients.ravel())
        targets.append(left[row])
    for column in range(4):
        coefficients = np.zeros((5, 5))
        coefficients[:, column] = 1.0
        equality.append(coefficients.ravel())
        targets.append(right[column])

    def value_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coupling = flat.reshape((5, 5))
        partition = float(np.sum(kernel * coupling))
        ratio = np.maximum(coupling, 1e-300) / product
        information = float(np.sum(coupling * np.log(ratio)))
        value = np.log(partition) - multiplier * information
        gradient = (
            kernel / partition
            - multiplier * (np.log(ratio) + 1.0)
        )
        return -value, -gradient.ravel()

    result = minimize(
        value_gradient,
        product.ravel(),
        jac=True,
        method="SLSQP",
        bounds=[(1e-15, 1.0)] * 25,
        constraints={
            "type": "eq",
            "fun": lambda flat: np.asarray(equality) @ flat - np.asarray(targets),
            "jac": lambda _flat: np.asarray(equality),
        },
        options={"ftol": 1e-13, "maxiter": 5000},
    )
    coupling = result.x.reshape((5, 5))
    information = float(
        np.sum(coupling * np.log(np.maximum(coupling, 1e-300) / product))
    )
    return {
        "upper": -float(result.fun),
        "information": information,
        "partition": float(np.sum(kernel * coupling)),
        "product_partition": float(np.sum(kernel * product)),
        "marginal_residual": float(
            max(
                np.max(np.abs(coupling.sum(axis=1) - left)),
                np.max(np.abs(coupling.sum(axis=0) - right)),
            )
        ),
        "optimizer_success": bool(result.success),
    }


def transformed_partition_upper(
    left: np.ndarray,
    right: np.ndarray,
    kernel: np.ndarray,
    multiplier: float,
) -> dict[str, float | bool]:
    product = np.outer(left, right)
    log_product = np.log(product.ravel())
    flat_kernel = kernel.ravel()

    def objective(log_t: float) -> float:
        t = np.exp(log_t)
        return (
            multiplier
            * float(logsumexp(log_product + t * flat_kernel / multiplier))
            - log_t
            - 1.0
        )

    result = minimize_scalar(
        objective,
        bounds=(-12.0, 12.0),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return {
        "upper": float(result.fun),
        "t": float(np.exp(result.x)),
        "product_log_partition": float(np.log(np.sum(product * kernel))),
        "optimizer_success": bool(result.success),
    }


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
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    )
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
    first_values = [
        projected.stage_upper_with_kernel(
            point[0:5], point[5:10], cut, kernel
        )
        for cut, kernel in zip(first_cuts, first_kernels, strict=True)
    ]
    second_values = [
        projected.stage_upper_with_kernel(
            point[5:10], point[10:15], cut, kernel
        )
        for cut, kernel in zip(second_cuts, second_kernels, strict=True)
    ]
    first_index = int(np.argmin(first_values))
    second_index = int(np.argmin(second_values))
    records = []
    for multiplier in args.multipliers:
        first = charged_partition_upper(
            point[0:5], point[5:10], first_kernels[first_index], multiplier
        )
        second = charged_partition_upper(
            point[5:10], point[10:15], second_kernels[second_index], multiplier
        )
        first_transformed = transformed_partition_upper(
            point[0:5], point[5:10], first_kernels[first_index], multiplier
        )
        second_transformed = transformed_partition_upper(
            point[5:10], point[10:15], second_kernels[second_index], multiplier
        )
        first_linear = first_values[first_index] - np.log(
            first["product_partition"]
        )
        second_linear = second_values[second_index] - np.log(
            second["product_partition"]
        )
        records.append(
            {
                "multiplier": multiplier,
                "charged_fixed_pair_upper": (
                    projected.bulk_entropy(point[0:5]) / 3.0
                    + first_linear
                    + first["upper"]
                    + second_linear
                    + second["upper"]
                ),
                "transformed_fixed_pair_upper": (
                    projected.bulk_entropy(point[0:5]) / 3.0
                    + first_linear
                    + first_transformed["upper"]
                    + second_linear
                    + second_transformed["upper"]
                ),
                "first": first,
                "second": second,
                "first_transformed": first_transformed,
                "second_transformed": second_transformed,
            }
        )
    print(
        json.dumps(
            {
                "frontier_index": args.frontier_index,
                "floating_box_upper": selected["bound"],
                "exact_fixed_pair_upper": (
                    projected.bulk_entropy(point[0:5]) / 3.0
                    + first_values[first_index]
                    + second_values[second_index]
                ),
                "first_cut_index": first_index,
                "second_cut_index": second_index,
                "records": records,
                "status": "floating entropy-charged coupling diagnostic",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
