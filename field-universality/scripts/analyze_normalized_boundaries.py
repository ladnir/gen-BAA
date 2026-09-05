#!/usr/bin/env python3
"""Audit the inequivalent increment-support strata of normalized coercivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_normalized_two_level as normalized
import analyze_zero_payment_surface as zero


SUPPORT_STRATA = {
    "singleton": (0,),
    "opposite_pair": (0, 3),
    "nonopposite_pair": (0, 1),
    "three_symbols": (0, 1, 2),
    "four_symbols": (0, 1, 2, 3),
}


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for ratio_index, support_ratio in enumerate(args.support_ratios):
        strata = {}
        for stratum_index, (name, support) in enumerate(SUPPORT_STRATA.items()):
            strata[name] = normalized.optimize_fixed_support_ratio(
                support_ratio,
                support,
                args.restarts,
                args.seed + 100 * ratio_index + stratum_index,
                args.maximum_iterations,
            )
        best_name = min(
            strata,
            key=lambda name: float(strata[name]["max_support_normalized_ratio"]),
        )
        rows.append(
            {
                "support_ratio": support_ratio,
                "best_stratum": best_name,
                "best_max_support_normalized_ratio": strata[best_name][
                    "max_support_normalized_ratio"
                ],
                "strata": strata,
            }
        )
    report = {
        "prime": 5,
        "normalization": "max(A,B)*L/R",
        "rows": rows,
        "minimum_found": min(
            float(row["best_max_support_normalized_ratio"]) for row in rows
        ),
        "status": (
            "floating multistart audit of inequivalent support strata; "
            "not a global certificate"
        ),
    }
    zero.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support-ratios",
        type=float,
        nargs="+",
        default=(0.95, 0.975, 1.0, 1.025, 1.05),
    )
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1771)
    parser.add_argument("--maximum-iterations", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-normalized-boundaries.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))
