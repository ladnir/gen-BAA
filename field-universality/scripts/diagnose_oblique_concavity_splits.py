#!/usr/bin/env python3
"""Test oblique splits along positive Hessian modes of a frontier box.

This is a floating diagnostic.  The two homogeneous inequalities for each
mode cover the composition domain because every message block sums to one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diagnose_joint_cut_concavity as concavity
import explore_projected_kl_box_cover as projected


def homogeneous_halfspace(
    direction: np.ndarray, threshold: float, lower_side: bool
) -> np.ndarray:
    inequality = direction.copy() if lower_side else -direction.copy()
    inequality[0:5] += -threshold if lower_side else threshold
    return inequality


def direction_range(
    direction: np.ndarray, vertices: list[np.ndarray]
) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    for block, block_vertices in enumerate(vertices):
        values = block_vertices @ direction[5 * block : 5 * (block + 1)]
        lower += float(np.min(values))
        upper += float(np.max(values))
    return lower, upper


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
    parser.add_argument("--outer-iterations", type=int, default=20)
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
    box = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))
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
    basis = concavity.simplex_tangent_basis()
    hessian = concavity.restricted_hessian(
        point,
        first_kernels[first_index],
        second_kernels[second_index],
        basis,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    vertices = [
        projected.simplex_box_vertices(box, start) for start in (0, 5, 10)
    ]
    parent = projected.evaluate(
        box,
        first_cuts,
        second_cuts,
        first_kernels,
        second_kernels,
        outer_iterations=args.outer_iterations,
    )
    if parent is None:
        raise RuntimeError("selected frontier box is infeasible")

    probes = []
    for eigen_index in np.flatnonzero(eigenvalues > 1e-10):
        direction = basis @ eigenvectors[:, eigen_index]
        direction /= np.linalg.norm(direction)
        lower, upper = direction_range(direction, vertices)
        raw_threshold = float(np.dot(direction, point))
        threshold = min(
            lower + 0.75 * (upper - lower),
            max(lower + 0.25 * (upper - lower), raw_threshold),
        )
        children = []
        for lower_side in (True, False):
            inequality = homogeneous_halfspace(
                direction, threshold, lower_side
            )
            evaluated = projected.evaluate(
                box,
                first_cuts,
                second_cuts,
                first_kernels,
                second_kernels,
                outer_iterations=args.outer_iterations,
                additional_composition_inequalities=(inequality,),
            )
            if evaluated is not None:
                children.append(
                    {
                        "side": "lower" if lower_side else "upper",
                        "upper": float(evaluated[0]),
                        "exact_bank_upper_at_lp_composition": float(evaluated[4]),
                        "split_dimension": int(evaluated[1]),
                    }
                )
        probes.append(
            {
                "eigenvalue": float(eigenvalues[eigen_index]),
                "direction": direction.tolist(),
                "range": [lower, upper],
                "threshold": threshold,
                "children": children,
                "maximum_child_upper": max(
                    (child["upper"] for child in children),
                    default=float("-inf"),
                ),
            }
        )
    probes.sort(key=lambda probe: probe["maximum_child_upper"])
    print(
        json.dumps(
            {
                "frontier_index": args.frontier_index,
                "saved_parent_upper": selected["bound"],
                "reevaluated_parent_upper": float(parent[0]),
                "positive_modes": len(probes),
                "probes": probes,
                "status": "floating oblique-split diagnostic; not a cover certificate",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
