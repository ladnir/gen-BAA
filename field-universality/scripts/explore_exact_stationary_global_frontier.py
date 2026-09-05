#!/usr/bin/env python3
"""Strengthen a saved projected-KL frontier with exact stationary cuts."""

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
    total_frontier = len(entries)
    if args.frontier_limit > 0:
        entries = entries[: args.frontier_limit]

    first_exact: list[stationary.StationaryDualCut] = []
    second_exact: list[stationary.StationaryDualCut] = []
    selected_points: list[np.ndarray] = []

    def usable(cut: stationary.StationaryDualCut) -> bool:
        return bool(
            cut.optimizer_success
            and cut.equality_residual <= args.maximum_cut_residual
            and cut.gradient_residual <= args.maximum_cut_residual
            and np.isfinite(cut.moment_constant)
            and np.all(np.isfinite(cut.moment_coefficients))
            and np.all(np.isfinite(cut.kernel))
        )

    def interior(point: np.ndarray) -> bool:
        return bool(np.min(point) >= args.minimum_cut_mass)

    for entry in entries:
        point = projected.sanitize_composition(
            np.asarray(entry["evaluated"]["point"], dtype=float)
        )
        if any(
            float(np.abs(point - old).sum()) < args.seed_separation
            for old in selected_points
        ):
            continue
        if not interior(point):
            continue
        first = stationary.fit_stationary_dual(point[0:5], point[5:10])
        second = stationary.fit_stationary_dual(point[5:10], point[10:15])
        if not usable(first) or not usable(second):
            continue
        selected_points.append(point)
        first_exact.append(first)
        second_exact.append(second)
        if len(selected_points) >= args.seed_cuts:
            break

    queue = []
    identifier = 0
    for entry in entries:
        evaluated = projected.decode_evaluated(entry["evaluated"])
        heapq.heappush(queue, (-float(evaluated[0]), identifier, evaluated))
        identifier += 1

    checkpoint_path = Path(args.checkpoint)
    signature = {
        "delta": args.delta,
        "eta": args.eta,
        "frontier": str(Path(args.frontier)),
        "frontier_limit": args.frontier_limit,
        "seed_cuts": args.seed_cuts,
        "seed_separation": args.seed_separation,
        "maximum_exact_cuts": args.maximum_exact_cuts,
        "outer_iterations": args.outer_iterations,
        "affine_bilinear_cuts": args.affine_bilinear_cuts,
        "target": args.target,
    }

    def save_checkpoint(
        processed: int,
        history: list[float],
        closed_boxes: int,
        maximum_point_residual: float,
    ) -> None:
        projected.write_json_atomic(
            checkpoint_path,
            {
                "version": 1,
                "signature": signature,
                "processed": processed,
                "identifier": identifier,
                "history": history,
                "closed_boxes": closed_boxes,
                "maximum_point_residual": maximum_point_residual,
                "first_exact": [
                    stationary.encode_cut(cut) for cut in first_exact
                ],
                "second_exact": [
                    stationary.encode_cut(cut) for cut in second_exact
                ],
                "queue": [
                    {
                        "priority": priority,
                        "identifier": entry_identifier,
                        "evaluated": projected.encode_evaluated(evaluated),
                    }
                    for priority, entry_identifier, evaluated in queue
                ],
            },
        )

    def active_cuts(
        cuts: list[stationary.StationaryDualCut],
    ) -> tuple[stationary.StationaryDualCut, ...]:
        if len(cuts) <= args.maximum_exact_cuts:
            return tuple(cuts)
        seed_count = min(args.seed_cuts, args.maximum_exact_cuts // 2)
        recent_count = args.maximum_exact_cuts - seed_count
        return tuple((*cuts[:seed_count], *cuts[-recent_count:]))

    def evaluate(box: projected.Box) -> tuple[object, ...] | None:
        return projected.evaluate(
            box,
            first_projected,
            second_projected,
            outer_iterations=args.outer_iterations,
            affine_bilinear_cuts=args.affine_bilinear_cuts,
            exact_first_cuts=active_cuts(first_exact),
            exact_second_cuts=active_cuts(second_exact),
        )

    def baseline_is_empty(box: projected.Box) -> bool:
        return (
            projected.evaluate(
                box,
                first_projected,
                second_projected,
                outer_iterations=1,
                affine_bilinear_cuts=0,
            )
            is None
        )

    history = [-queue[0][0]]
    closed_boxes = 0
    maximum_point_residual = 0.0
    processed_start = 0
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as source:
            checkpoint = json.load(source)
        if checkpoint.get("version") != 1 or checkpoint.get("signature") != signature:
            raise RuntimeError("combined frontier checkpoint does not match this run")
        processed_start = int(checkpoint["processed"])
        identifier = int(checkpoint["identifier"])
        history = [float(value) for value in checkpoint["history"]]
        closed_boxes = int(checkpoint["closed_boxes"])
        maximum_point_residual = float(checkpoint["maximum_point_residual"])
        first_exact = [
            stationary.decode_cut(value) for value in checkpoint["first_exact"]
        ]
        second_exact = [
            stationary.decode_cut(value) for value in checkpoint["second_exact"]
        ]
        queue = [
            (
                float(entry["priority"]),
                int(entry["identifier"]),
                projected.decode_evaluated(entry["evaluated"]),
            )
            for entry in checkpoint["queue"]
        ]
        heapq.heapify(queue)
        print(
            json.dumps(
                {
                    "resumed_processed": processed_start,
                    "upper": -queue[0][0],
                    "boxes": len(queue),
                    "stored_cuts": len(first_exact),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for split_index in range(processed_start, args.max_splits):
        _, _, current = heapq.heappop(queue)
        refreshed = evaluate(current[3])
        if refreshed is None:
            if baseline_is_empty(current[3]):
                closed_boxes += 1
                if not queue:
                    break
                continue
            raise RuntimeError("combined LP failed on a nonempty frontier box")
        if refreshed[0] > current[0]:
            refreshed = (current[0], *refreshed[1:])
        point = projected.sanitize_composition(np.asarray(refreshed[2]))
        first = stationary.fit_stationary_dual(point[0:5], point[5:10])
        second = stationary.fit_stationary_dual(point[5:10], point[10:15])
        if interior(point) and usable(first) and usable(second):
            maximum_point_residual = max(
                maximum_point_residual,
                first.equality_residual,
                second.equality_residual,
            )
            first_exact.append(first)
            second_exact.append(second)
        strengthened = evaluate(refreshed[3])
        if strengthened is None:
            raise RuntimeError("combined LP failed after adding a usable cut")
        else:
            if strengthened[0] > refreshed[0]:
                strengthened = (refreshed[0], *strengthened[1:])
            if strengthened[0] <= args.target:
                closed_boxes += 1
            else:
                for child in projected.split(
                    strengthened[3], int(strengthened[1])
                ):
                    bounded = evaluate(child)
                    if bounded is None:
                        if baseline_is_empty(child):
                            continue
                        raise RuntimeError(
                            "combined LP failed on a nonempty child box"
                        )
                    if bounded[0] > strengthened[0]:
                        bounded = (strengthened[0], *bounded[1:])
                    if bounded[0] <= args.target:
                        closed_boxes += 1
                        continue
                    heapq.heappush(
                        queue, (-float(bounded[0]), identifier, bounded)
                    )
                    identifier += 1
        if not queue:
            break
        history.append(-queue[0][0])
        completed = split_index + 1
        if args.checkpoint_every and completed % args.checkpoint_every == 0:
            save_checkpoint(
                completed, history, closed_boxes, maximum_point_residual
            )
        if args.progress_every and (split_index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": split_index + 1,
                        "upper": -queue[0][0],
                        "boxes": len(queue),
                        "closed": closed_boxes,
                        "stored_cuts": len(first_exact),
                        "active_cuts": len(active_cuts(first_exact)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    global_scope = len(entries) == total_frontier
    closed = not queue
    completed = min(args.max_splits, processed_start + max(0, len(history) - 1))
    if queue:
        save_checkpoint(
            args.max_splits, history, closed_boxes, maximum_point_residual
        )
    return {
        "status": (
            "floating global frontier closed"
            if closed and global_scope
            else "floating sampled frontier closed"
            if closed
            else "floating frontier cover incomplete"
        ),
        "delta": args.delta,
        "frontier_boxes_selected": len(entries),
        "frontier_boxes_total": total_frontier,
        "global_scope": global_scope,
        "upper": args.target if closed else -queue[0][0],
        "processed": args.max_splits if not closed else len(history) - 1,
        "boxes_remaining": len(queue),
        "closed_boxes": closed_boxes,
        "seed_cuts_per_stage": len(selected_points),
        "stored_exact_cuts_per_stage": len(first_exact),
        "active_exact_cuts_per_stage": len(active_cuts(first_exact)),
        "maximum_point_residual": maximum_point_residual,
        "history": history,
        "scope": "floating LP cover; exact cuts are sound but constants and LP solves require directed replay",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument(
        "--frontier",
        default="results/f5-projected-kl-box-cover-0.22.checkpoint.json",
    )
    parser.add_argument("--frontier-limit", type=int, default=0)
    parser.add_argument(
        "--bank-cache",
        default="results/f5-projected-kl-dual-bank-0.22.json",
    )
    parser.add_argument("--bank-iterations", type=int, default=20)
    parser.add_argument("--bank-restarts", type=int, default=12)
    parser.add_argument("--seed-cuts", type=int, default=24)
    parser.add_argument("--seed-separation", type=float, default=0.04)
    parser.add_argument("--maximum-exact-cuts", type=int, default=64)
    parser.add_argument("--maximum-cut-residual", type=float, default=1e-6)
    parser.add_argument("--minimum-cut-mass", type=float, default=1e-8)
    parser.add_argument("--max-splits", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=12)
    parser.add_argument("--affine-bilinear-cuts", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        default="results/f5-exact-stationary-global-frontier.checkpoint.json",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(cover(parse_args()), indent=2, sort_keys=True))
