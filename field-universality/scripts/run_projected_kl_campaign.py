#!/usr/bin/env python3
"""Run bounded enrich-and-split batches for the F_5 projected-KL cover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import explore_projected_kl_box_cover as projected


def frontier_state(
    checkpoint: Path,
    first_cuts: list[object],
    second_cuts: list[object],
) -> dict[str, object]:
    with checkpoint.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    worst = min(saved["queue"], key=lambda entry: entry["priority"])
    evaluated = worst["evaluated"]
    point = np.asarray(evaluated["point"], dtype=float)
    exact_upper = projected.composition_bank_upper(point, first_cuts, second_cuts)
    return {
        "splits": int(saved["splits_completed"]),
        "leaves": len(saved["queue"]),
        "upper": float(evaluated["bound"]),
        "worst_exact_bank_upper": exact_upper,
    }


def campaign(args: argparse.Namespace) -> dict[str, object]:
    bank_result, first_cuts, second_cuts = projected.load_or_generate_bank(args)
    cycles = []
    for cycle in range(args.batches):
        before = frontier_state(args.checkpoint, first_cuts, second_cuts)
        refinements = []
        if before["worst_exact_bank_upper"] > args.enrich_target_upper:
            refinements = projected.enrich_bank_from_frontier(
                args, bank_result, first_cuts, second_cuts
            )
        args.max_boxes = int(before["splits"]) + args.batch_splits
        result = projected.cover(args, first_cuts, second_cuts)
        after = frontier_state(args.checkpoint, first_cuts, second_cuts)
        record = {
            "cycle": cycle + 1,
            "before": before,
            "after": after,
            "refinements_added": sum(
                bool(refinement["added"]) for refinement in refinements
            ),
            "cuts_per_stage": len(first_cuts),
            "status": result["status"],
        }
        cycles.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if result["status"] == "target floating LP box cover reached":
            break
    final = frontier_state(args.checkpoint, first_cuts, second_cuts)
    return {
        "status": (
            "floating projected-KL campaign reached target"
            if final["upper"] < args.target_upper
            else "floating projected-KL campaign incomplete"
        ),
        "cycles": cycles,
        "final": final,
        "cuts_per_stage": len(first_cuts),
        "scope": "floating LP campaign; not an exact certificate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-splits", type=int, default=200)
    parser.add_argument("--enrich-frontier", type=int, default=8)
    parser.add_argument("--enrich-target-upper", type=float, default=-0.005)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--strong-branch-candidates", type=int, default=9)
    parser.add_argument("--strong-branch-every", type=int, default=0)
    parser.add_argument("--strong-branch-probe-iterations", type=int, default=0)
    parser.add_argument("--target-upper", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--split-rule", choices=("midpoint", "lp_clipped"), default="lp_clipped")
    parser.add_argument("--affine-bilinear-cuts", type=int, default=0)
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--bank-cache",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument(
        "--split-log",
        type=Path,
        default=Path("results/f5-projected-kl-split-log-0.22.json"),
    )
    args = parser.parse_args()
    if args.batches < 1 or args.batch_splits < 1 or args.enrich_frontier < 1:
        parser.error("batch and enrichment counts must be positive")
    # Attributes consumed by the shared bank and cover implementation.
    args.bank_cache = str(args.bank_cache)
    args.checkpoint = Path(args.checkpoint)
    args.split_log = Path(args.split_log)
    args.rebuild_bank = False
    args.resume = True
    args.refresh_frontier = 0
    args.max_boxes = 0
    return args


if __name__ == "__main__":
    print(json.dumps(campaign(parse_args()), indent=2, sort_keys=True))
