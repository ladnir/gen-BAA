#!/usr/bin/env python3
"""Compare every coordinate split on the worst projected-KL frontier box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import explore_projected_kl_box_cover as projected


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
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument(
        "--restriction",
        action="append",
        default=[],
        metavar="DIM:SIDE:VALUE",
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
    _, first, second = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second)
    with args.frontier.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    selected = sorted(saved["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]
    current = projected.decode_evaluated(selected["evaluated"])
    lower = list(current[3].lower)
    upper = list(current[3].upper)
    for specification in args.restriction:
        dimension_text, side, value_text = specification.split(":")
        dimension = int(dimension_text)
        value = float(value_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], value)
        elif side == "ge":
            lower[dimension] = max(lower[dimension], value)
        else:
            raise ValueError("restriction side must be le or ge")
    restricted_box = projected.Box(tuple(lower), tuple(upper))
    restricted = projected.evaluate(
        restricted_box,
        first,
        second,
        first_kernels,
        second_kernels,
        args.outer_iterations,
    )
    if restricted is None:
        raise RuntimeError("restricted box is infeasible")
    current = restricted
    candidates = []
    for dimension in range(15):
        children = []
        for child in projected.split(current[3], dimension):
            evaluated = projected.evaluate(
                child,
                first,
                second,
                first_kernels,
                second_kernels,
                args.outer_iterations,
            )
            if evaluated is not None:
                children.append(min(float(current[0]), float(evaluated[0])))
        candidates.append(
            {
                "dimension": dimension,
                "child_upper": max(children, default=float("-inf")),
                "children": children,
            }
        )
    candidates.sort(key=lambda candidate: candidate["child_upper"])
    print(
        json.dumps(
            {
                "parent_upper": current[0],
                "frontier_index": args.frontier_index,
                "suggested_dimension": current[1],
                "restrictions": args.restriction,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
