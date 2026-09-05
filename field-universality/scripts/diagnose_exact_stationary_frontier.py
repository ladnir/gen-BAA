#!/usr/bin/env python3
"""Add exact stationary dual cuts on a projected-cover frontier box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    bank_args = argparse.Namespace(
        eta=args.eta,
        delta=args.delta,
        bank_iterations=args.bank_iterations,
        bank_restarts=args.bank_restarts,
        seed=args.seed,
        bank_cache=args.bank_cache,
        rebuild_bank=False,
    )
    _, first_projected, second_projected = projected.load_or_generate_bank(
        bank_args
    )
    with Path(args.frontier).open("r", encoding="utf-8") as source:
        saved = json.load(source)
    entries = sorted(saved["queue"], key=lambda entry: entry["priority"])
    selected = entries[args.frontier_index]["evaluated"]
    box = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))

    first_exact: list[stationary.StationaryDualCut] = []
    second_exact: list[stationary.StationaryDualCut] = []
    history = []
    for iteration in range(args.iterations):
        evaluated = projected.evaluate(
            box,
            first_projected,
            second_projected,
            outer_iterations=args.outer_iterations,
            affine_bilinear_cuts=args.affine_bilinear_cuts,
            exact_first_cuts=tuple(first_exact),
            exact_second_cuts=tuple(second_exact),
        )
        if evaluated is None:
            raise RuntimeError("frontier box became infeasible")
        point = projected.sanitize_composition(np.asarray(evaluated[2]))
        first = stationary.fit_stationary_dual(point[0:5], point[5:10])
        second = stationary.fit_stationary_dual(point[5:10], point[10:15])
        exact_point_upper = (
            projected.bulk_entropy(point[0:5]) / 3.0
            - first.value
            - second.value
        )
        history.append(
            {
                "iteration": iteration,
                "box_upper": float(evaluated[0]),
                "exact_upper_at_lp_composition": exact_point_upper,
                "relaxation_gap": float(evaluated[0]) - exact_point_upper,
                "split_dimension": int(evaluated[1]),
                "point": point.tolist(),
                "first_residual": first.equality_residual,
                "second_residual": second.equality_residual,
            }
        )
        first_exact.append(first)
        second_exact.append(second)
        if evaluated[0] <= args.target:
            break
    child_probe = []
    if history and args.child_probe:
        dimension = int(history[-1]["split_dimension"])
        for child in projected.split(box, dimension):
            evaluated = projected.evaluate(
                child,
                first_projected,
                second_projected,
                outer_iterations=args.outer_iterations,
                affine_bilinear_cuts=args.affine_bilinear_cuts,
                exact_first_cuts=tuple(first_exact),
                exact_second_cuts=tuple(second_exact),
            )
            if evaluated is not None:
                child_probe.append(
                    {
                        "upper": float(evaluated[0]),
                        "split_dimension": int(evaluated[1]),
                        "lower": list(evaluated[3].lower),
                        "upper_box": list(evaluated[3].upper),
                    }
                )
    return {
        "delta": args.delta,
        "frontier": args.frontier,
        "frontier_index": args.frontier_index,
        "original_box_upper": float(selected["bound"]),
        "exact_cuts_per_stage": len(first_exact),
        "history": history,
        "child_probe": child_probe,
        "status": "floating diagnostic; exact cuts are sound but LP solves are not certified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument(
        "--frontier",
        default="results/f5-projected-kl-box-cover-0.22.checkpoint.json",
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument(
        "--bank-cache",
        default="results/f5-projected-kl-dual-bank-0.22.json",
    )
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument(
        "--child-probe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
