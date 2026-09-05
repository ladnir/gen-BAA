#!/usr/bin/env python3
"""Floating McCormick cover of one saved frontier box using a fixed cut bank."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected


def encode_evaluation(value: tuple[object, ...]) -> dict[str, object]:
    box = value[3]
    return {
        "bound": float(value[0]),
        "split_dimension": int(value[1]),
        "point": np.asarray(value[2]).tolist(),
        "lower": list(box.lower),
        "upper": list(box.upper),
        "projected_at_point": float(value[4]),
        "rank_relaxed_at_point": float(value[5]),
    }


def decode_evaluation(value: dict[str, object]) -> tuple[object, ...]:
    return (
        float(value["bound"]),
        int(value["split_dimension"]),
        np.asarray(value["point"], dtype=float),
        projected.Box(tuple(value["lower"]), tuple(value["upper"])),
        float(value["projected_at_point"]),
        float(value["rank_relaxed_at_point"]),
    )


def cover(args: argparse.Namespace) -> dict[str, object]:
    with args.frontier.open("r", encoding="utf-8") as source:
        frontier = json.load(source)
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
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
        bank_iterations=args.bank_iterations,
        bank_restarts=args.bank_restarts,
        seed=args.seed,
        bank_cache=str(args.projected_bank),
        rebuild_bank=False,
    )
    _, first_projected, second_projected = projected.load_or_generate_bank(
        bank_args
    )

    def evaluate(box: projected.Box) -> tuple[object, ...] | None:
        return projected.evaluate(
            box,
            first_projected,
            second_projected,
            outer_iterations=args.outer_iterations,
            affine_bilinear_cuts=args.affine_bilinear_cuts,
            exact_first_cuts=first_exact,
            exact_second_cuts=second_exact,
        )

    signature = {
        "version": 1,
        "frontier": str(args.frontier),
        "frontier_index": args.frontier_index,
        "exact_bank": str(args.exact_bank),
        "projected_bank": str(args.projected_bank),
        "outer_iterations": args.outer_iterations,
        "affine_bilinear_cuts": args.affine_bilinear_cuts,
        "target": args.target,
    }
    queue: list[tuple[float, int, tuple[object, ...]]] = []
    identifier = 0
    completed = 0
    closed = 0
    history: list[float] = []
    if args.resume and args.checkpoint.exists():
        with args.checkpoint.open("r", encoding="utf-8") as source:
            saved = json.load(source)
        if saved.get("signature") != signature:
            raise RuntimeError("cover checkpoint signature does not match")
        identifier = int(saved["identifier"])
        completed = int(saved["completed"])
        closed = int(saved["closed"])
        history = [float(value) for value in saved["history"]]
        queue = [
            (
                float(entry["priority"]),
                int(entry["identifier"]),
                decode_evaluation(entry["evaluation"]),
            )
            for entry in saved["queue"]
        ]
        heapq.heapify(queue)
    else:
        root = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))
        evaluated = evaluate(root)
        if evaluated is None:
            raise RuntimeError("selected root box is infeasible")
        queue = [(-float(evaluated[0]), identifier, evaluated)]
        identifier += 1
        history = [float(evaluated[0])]

    def save() -> None:
        projected.write_json_atomic(
            args.checkpoint,
            {
                "signature": signature,
                "identifier": identifier,
                "completed": completed,
                "closed": closed,
                "history": history,
                "queue": [
                    {
                        "priority": priority,
                        "identifier": entry_identifier,
                        "evaluation": encode_evaluation(evaluation),
                    }
                    for priority, entry_identifier, evaluation in queue
                ],
            },
        )

    while queue and completed < args.max_splits:
        _, _, current = heapq.heappop(queue)
        dimensions = [int(current[1])]
        if (
            args.strong_branch_every > 0
            and completed % args.strong_branch_every == 0
        ):
            dimensions = list(range(15))[: args.strong_branch_candidates]
        best_children = None
        best_child_upper = float("inf")
        for dimension in dimensions:
            candidate_children = []
            for child in projected.split(current[3], dimension):
                evaluated = evaluate(child)
                if evaluated is None:
                    continue
                if evaluated[0] > current[0]:
                    evaluated = (current[0], *evaluated[1:])
                candidate_children.append(evaluated)
            candidate_upper = max(
                (float(value[0]) for value in candidate_children),
                default=-float("inf"),
            )
            if candidate_upper < best_child_upper:
                best_child_upper = candidate_upper
                best_children = candidate_children
        for evaluated in best_children or []:
            if evaluated[0] <= args.target:
                closed += 1
                continue
            heapq.heappush(queue, (-float(evaluated[0]), identifier, evaluated))
            identifier += 1
        completed += 1
        if queue:
            history.append(-queue[0][0])
        if args.checkpoint_every and completed % args.checkpoint_every == 0:
            save()
        if args.progress_every and completed % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "splits": completed,
                        "upper": None if not queue else -queue[0][0],
                        "boxes": len(queue),
                        "closed": closed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    save()
    return {
        "status": (
            "floating fixed-bank frontier box closed"
            if not queue
            else "floating fixed-bank frontier box cover incomplete"
        ),
        "frontier_index": args.frontier_index,
        "original_projected_upper": float(selected["bound"]),
        "upper": args.target if not queue else -queue[0][0],
        "splits": completed,
        "boxes_remaining": len(queue),
        "closed_boxes": closed,
        "cuts_per_stage": len(first_exact),
        "checkpoint": str(args.checkpoint),
        "scope": "floating LP relaxation; not a directed certificate",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=166)
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
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=8)
    parser.add_argument("--target", type=float, default=-1e-5)
    parser.add_argument("--max-splits", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--strong-branch-candidates", type=int, default=15)
    parser.add_argument("--strong-branch-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/f5-fixed-stationary-box166-cover.checkpoint.json"),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(cover(parse_args()), indent=2, sort_keys=True))
