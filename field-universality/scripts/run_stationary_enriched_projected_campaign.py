#!/usr/bin/env python3
"""Test a stationary-dual cut bank on a saved projected-KL frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def frontier_points(path: Path, count: int) -> list[np.ndarray]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    entries = sorted(saved["queue"], key=lambda value: value["priority"])
    selected = []
    for entry in entries:
        point = projected.sanitize_composition(
            np.asarray(entry["evaluated"]["point"], dtype=float)
        )
        if all(float(np.abs(point - old).sum()) >= 0.03 for old in selected):
            selected.append(point)
        if len(selected) >= count:
            break
    return selected


def stationary_bank(
    points: list[np.ndarray],
) -> tuple[
    tuple[stationary.StationaryDualCut, ...],
    tuple[stationary.StationaryDualCut, ...],
]:
    first = []
    second = []
    for point in points:
        first.extend(
            stationary.stationary_cut_orbit(
                stationary.fit_stationary_dual(point[:5], point[5:10]),
                include_state_translations=False,
            )
        )
        second.extend(
            stationary.stationary_cut_orbit(
                stationary.fit_stationary_dual(point[5:10], point[10:15]),
                include_state_translations=False,
            )
        )
    return tuple(first), tuple(second)


def run(args: argparse.Namespace) -> dict[str, object]:
    _, first, second = projected.load_or_generate_bank(args)
    points = frontier_points(Path(args.checkpoint), args.stationary_points)
    exact_first, exact_second = stationary_bank(points)
    saved = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    before = {
        "splits": int(saved["splits_completed"]),
        "leaves": len(saved["queue"]),
        "upper": -min(float(entry["priority"]) for entry in saved["queue"]),
    }
    args.max_boxes = before["splits"] + args.batch_splits
    result = projected.cover(args, first, second, exact_first, exact_second)
    saved = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    return {
        "status": result["status"],
        "scope": "floating stationary-enriched projected campaign",
        "before": before,
        "after": {
            "splits": int(saved["splits_completed"]),
            "leaves": len(saved["queue"]),
            "upper": -min(float(entry["priority"]) for entry in saved["queue"]),
        },
        "stationary_reference_points": len(points),
        "stationary_cuts_per_stage": len(exact_first),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stationary-points", type=int, default=4)
    parser.add_argument("--batch-splits", type=int, default=100)
    parser.add_argument("--refresh-frontier", type=int, default=50)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--split-rule", choices=("midpoint", "lp_clipped"), default="lp_clipped")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-cache", default="results/f5-projected-kl-dual-bank-0.22.json")
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.rebuild_bank = False
    args.resume = True
    args.affine_bilinear_cuts = 0
    args.strong_branch_candidates = 1
    args.strong_branch_every = 0
    args.strong_branch_probe_iterations = 0
    args.target_upper = 0.0
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
