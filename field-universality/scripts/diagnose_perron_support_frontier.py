#!/usr/bin/env python3
"""Diagnose Perron cuts on the nonnegative support frontier.

This is deliberately not a proof reduction.  The internal-information
quantity G can be negative, so terminal boxes with negative support exponent
cannot be discarded without another argument.  The selected boxes are still
useful stress tests for the full-symbol cuts.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

import explore_perron_bound as perron
import extract_support_frontier as frontier


def composition_box(box: object) -> perron.CompositionBox:
    support_lower = tuple(float(value) for value in box.lower)
    support_upper = tuple(float(value) for value in box.upper)
    coordinate_lower = (0.0,) * 12
    coordinate_upper = tuple(
        support_upper[component]
        for component in range(3)
        for _ in range(4)
    )
    first_crossing_upper = min(
        support_upper[0], 1.0 - support_lower[1]
    )
    second_crossing_upper = min(
        support_upper[1], 1.0 - support_lower[2]
    )
    return perron.CompositionBox(
        (*coordinate_lower, 0.0, 0.0),
        (*coordinate_upper, first_crossing_upper, second_crossing_upper),
        support_lower,
        support_upper,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    support_args = argparse.Namespace(
        prime=5,
        repetition=3,
        delta=Fraction(1323, 10_000),
        eta=Fraction(1, 500),
        cutoff=Fraction(1323, 2500),
        target=Fraction(1, 10_000),
        precision=args.precision,
        max_boxes=100_000,
        display=0,
    )
    _, _, entries = frontier.support_frontier_entries(support_args)
    diagnostics = []
    for index, (support_upper, support_box) in enumerate(
        entries[: args.frontier_boxes]
    ):
        bank: dict[str, object] = {}
        diagnostic = perron.evaluate_composition_box(
            composition_box(support_box),
            args,
            args.cuts,
            args.seed + 100 * index,
            include_projection=False,
            cut_bank_out=bank,
        )
        if not diagnostic["full_transition_box_feasible"]:
            value = float("-inf")
        else:
            value = float(diagnostic["common_mixture_transition_lp_upper"])
        diagnostics.append(
            {
                "index": index,
                "support_upper_bound": float(support_upper),
                "support_lower": [str(value) for value in support_box.lower],
                "support_upper": [str(value) for value in support_box.upper],
                "perron_upper": value,
                "diagnostic": diagnostic,
            }
        )
    diagnostics.sort(key=lambda item: item["perron_upper"], reverse=True)
    return {
        "frontier_boxes_total": len(entries),
        "frontier_boxes_tested": len(diagnostics),
        "boxes_numerically_negative": sum(
            item["perron_upper"] < 0.0 for item in diagnostics
        ),
        "maximum_perron_upper": (
            diagnostics[0]["perron_upper"] if diagnostics else None
        ),
        "worst_boxes": diagnostics[: args.display],
        "status": (
            "numerical stress test only; negative-support boxes are not discharged"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--frontier-boxes", type=int, default=94)
    parser.add_argument("--cuts", type=int, default=4)
    parser.add_argument("--adaptive-cuts", type=int, default=2)
    parser.add_argument("--independent-stage-cuts", action="store_true")
    parser.add_argument("--cut-orbits", action="store_true")
    parser.add_argument("--symmetric-reference-cuts", action="store_true")
    parser.add_argument("--retain-output-composition", action="store_true")
    parser.add_argument("--log-bound", type=float, default=40.0)
    parser.add_argument("--gap-multiplier", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--display", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
