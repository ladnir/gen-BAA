#!/usr/bin/env python3
"""Measure lifted residual-slab constraints on a saved projected frontier box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    with args.frontier.open("r", encoding="utf-8") as source:
        frontier = json.load(source)
    entries = sorted(frontier["queue"], key=lambda entry: entry["priority"])
    selected = entries[args.frontier_index]["evaluated"]
    box = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))

    with args.exact_bank.open("r", encoding="utf-8") as source:
        bank = json.load(source)
    first_exact = tuple(
        stationary.decode_cut(value) for value in bank["first_cuts"]
    )
    second_exact = tuple(
        stationary.decode_cut(value) for value in bank["second_cuts"]
    )
    bank_args = argparse.Namespace(
        eta=args.eta,
        delta=args.delta,
        bank_iterations=20,
        bank_restarts=12,
        seed=11,
        bank_cache=str(args.projected_bank),
        rebuild_bank=False,
    )
    _, first_projected, second_projected = projected.load_or_generate_bank(
        bank_args
    )

    def evaluate(extra: tuple[tuple[object, float], ...] = ()) -> float | None:
        result = projected.evaluate(
            box,
            first_projected,
            second_projected,
            outer_iterations=args.outer_iterations,
            affine_bilinear_cuts=args.affine_bilinear_cuts,
            exact_first_cuts=first_exact,
            exact_second_cuts=second_exact,
            additional_lifted_inequalities=extra,
        )
        return None if result is None else float(result[0])

    records = []
    if not args.baseline_only:
        for stage in range(2):
            for coordinate in range(1, 5):
                for sign in (-1, 1):
                    inequality = projected.nonzero_residual_slab_inequality(
                        stage, coordinate, sign, args.threshold
                    )
                    records.append(
                        {
                            "stage": stage + 1,
                            "coordinate": coordinate,
                            "sign": "+" if sign > 0 else "-",
                            "upper": evaluate((inequality,)),
                        }
                    )
    return {
        "status": "floating lifted-slab frontier diagnostic",
        "frontier_index": args.frontier_index,
        "projected_frontier_upper": float(selected["bound"]),
        "baseline_with_stationary_bank": evaluate(),
        "threshold": args.threshold,
        "records": records,
        "largest_slab_upper": max(
            (record["upper"] for record in records if record["upper"] is not None),
            default=None,
        ),
        "scope": "floating LP diagnostic; not an exact certificate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument(
        "--exact-bank",
        type=Path,
        default=Path("results/f5-fixed-stationary-shared-bank.json"),
    )
    parser.add_argument(
        "--projected-bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--threshold", type=float, default=1.0 / 3200.0)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=8)
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2, sort_keys=True))
