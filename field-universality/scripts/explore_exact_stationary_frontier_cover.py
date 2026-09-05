#!/usr/bin/env python3
"""Floating branch-and-cut cover of one projected-KL frontier box."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def cover(args: argparse.Namespace) -> dict[str, object]:
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
    root = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))

    first_exact: list[stationary.StationaryDualCut] = []
    second_exact: list[stationary.StationaryDualCut] = []

    def evaluate(box: projected.Box) -> tuple[object, ...] | None:
        return projected.evaluate(
            box,
            first_projected,
            second_projected,
            outer_iterations=args.outer_iterations,
            affine_bilinear_cuts=args.affine_bilinear_cuts,
            exact_first_cuts=tuple(first_exact),
            exact_second_cuts=tuple(second_exact),
        )

    initial = evaluate(root)
    if initial is None:
        raise RuntimeError("selected frontier box is infeasible")
    queue = [(-float(initial[0]), 0, initial)]
    identifier = 1
    history = [float(initial[0])]
    closed_boxes = 0
    maximum_point_residual = 0.0
    for split_index in range(args.max_splits):
        _, _, current = heapq.heappop(queue)
        point = projected.sanitize_composition(np.asarray(current[2]))
        first = stationary.fit_stationary_dual(point[0:5], point[5:10])
        second = stationary.fit_stationary_dual(point[5:10], point[10:15])
        maximum_point_residual = max(
            maximum_point_residual,
            first.equality_residual,
            second.equality_residual,
        )
        first_exact.append(first)
        second_exact.append(second)
        refreshed = evaluate(current[3])
        if refreshed is None:
            closed_boxes += 1
            continue
        if refreshed[0] > current[0]:
            refreshed = (current[0], *refreshed[1:])
        if refreshed[0] <= args.target:
            closed_boxes += 1
        else:
            for child in projected.split(refreshed[3], int(refreshed[1])):
                bounded = evaluate(child)
                if bounded is None:
                    continue
                if bounded[0] > refreshed[0]:
                    bounded = (refreshed[0], *bounded[1:])
                if bounded[0] <= args.target:
                    closed_boxes += 1
                    continue
                heapq.heappush(
                    queue, (-float(bounded[0]), identifier, bounded)
                )
                identifier += 1
        if not queue:
            return {
                "status": "floating frontier box closed",
                "delta": args.delta,
                "frontier_index": args.frontier_index,
                "original_upper": float(selected["bound"]),
                "upper": args.target,
                "splits": split_index + 1,
                "closed_boxes": closed_boxes,
                "exact_cuts_per_stage": len(first_exact),
                "maximum_point_residual": maximum_point_residual,
                "history": history,
            }
        history.append(-queue[0][0])
        if args.progress_every and (split_index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "splits": split_index + 1,
                        "upper": -queue[0][0],
                        "boxes": len(queue),
                        "cuts": len(first_exact),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    worst = queue[0][2]
    worst_point = projected.sanitize_composition(np.asarray(worst[2]))
    first = stationary.fit_stationary_dual(
        worst_point[0:5], worst_point[5:10]
    )
    second = stationary.fit_stationary_dual(
        worst_point[5:10], worst_point[10:15]
    )
    exact_point_upper = (
        projected.bulk_entropy(worst_point[0:5]) / 3.0
        - first.value
        - second.value
    )
    return {
        "status": "floating frontier box cover incomplete",
        "delta": args.delta,
        "frontier_index": args.frontier_index,
        "original_upper": float(selected["bound"]),
        "upper": -queue[0][0],
        "splits": args.max_splits,
        "boxes": len(queue),
        "closed_boxes": closed_boxes,
        "exact_cuts_per_stage": len(first_exact),
        "maximum_point_residual": maximum_point_residual,
        "worst_exact_upper_at_lp_composition": exact_point_upper,
        "history": history,
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
    parser.add_argument("--max-splits", type=int, default=40)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(cover(parse_args()), indent=2, sort_keys=True))
