#!/usr/bin/env python3
"""Recover an exact coverage tree from terminal floating frontier boxes."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import explore_projected_kl_box_cover as projected
import verify_projected_frontier_cover_map as checker
import verify_stationary_cover_certificate as tree


Box = checker.Box
ZERO = Fraction(0)


def text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_box(box: Box) -> dict[str, list[str]]:
    return {
        "lower": [text(value) for value in box[0]],
        "upper": [text(value) for value in box[1]],
    }


def decode_box(raw: object) -> Box:
    if not isinstance(raw, dict):
        raise ValueError("invalid saved reconstruction box")
    lower = tuple(Fraction(value) for value in raw["lower"])
    upper = tuple(Fraction(value) for value in raw["upper"])
    if len(lower) != 15 or len(upper) != 15:
        raise ValueError("saved reconstruction box has the wrong dimension")
    return lower, upper


def save_work(
    path: Path,
    source: dict[str, object],
    nodes: dict[str, object],
    stack: list[tuple[int, Box, tuple[int, ...], int]],
    next_id: int,
    max_depth: int,
    status: str,
) -> None:
    projected.write_json_atomic(
        path,
        {
            "schema": "f5-projected-frontier-cover-work-v1",
            "status": status,
            "source": source,
            "nodes": nodes,
            "stack": [
                {
                    "node_id": node_id,
                    "query": encode_box(query),
                    "candidates": list(candidates),
                    "depth": depth,
                }
                for node_id, query, candidates, depth in stack
            ],
            "next_id": next_id,
            "max_depth": max_depth,
        },
    )


def outward_box(
    raw: dict[str, object], denominator: int, root: Box, margin_units: int
) -> Box:
    lower = []
    upper = []
    for index, (left, right) in enumerate(
        zip(raw["lower"], raw["upper"], strict=True)
    ):
        lower.append(
            max(
                root[0][index],
                Fraction(round(float(left) * denominator) - margin_units, denominator),
            )
        )
        upper.append(
            min(
                root[1][index],
                Fraction(round(float(right) * denominator) + margin_units, denominator),
            )
        )
    return tuple(lower), tuple(upper)


def intersects(left: Box, right: Box) -> bool:
    return all(
        left[0][index] <= right[1][index]
        and right[0][index] <= left[1][index]
        for index in range(15)
    )


def choose_split(
    query: Box,
    candidates: tuple[int, ...],
    boxes: tuple[Box, ...],
    merge_tolerance: Fraction,
) -> tuple[int, Fraction]:
    best = None
    for dimension in range(15):
        boundaries = sorted(
            {
                value
                for index in candidates
                for value in (boxes[index][0][dimension], boxes[index][1][dimension])
                if query[0][dimension] < value < query[1][dimension]
            }
        )
        if not boundaries:
            continue
        clusters: list[list[Fraction]] = []
        for value in boundaries:
            if not clusters or value - clusters[-1][-1] > merge_tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        boundaries = [
            (cluster[0] + cluster[-1]) / 2 for cluster in clusters
        ]
        probes = boundaries if len(boundaries) <= 9 else [
            boundaries[round(position * (len(boundaries) - 1) / 8)]
            for position in range(9)
        ]
        for threshold in probes:
            left_count = sum(boxes[index][0][dimension] < threshold for index in candidates)
            right_count = sum(boxes[index][1][dimension] > threshold for index in candidates)
            score = (max(left_count, right_count), left_count + right_count, dimension)
            if best is None or score < best[0]:
                best = (score, dimension, threshold)
    if best is None:
        raise RuntimeError(
            "coverage gap has no frontier boundary to split: "
            f"candidates={len(candidates)}, "
            f"query_lower={[float(value) for value in query[0]]}, "
            f"query_upper={[float(value) for value in query[1]]}"
        )
    return best[1], best[2]


def reconstruct(args: argparse.Namespace) -> dict[str, object]:
    saved = json.loads(args.frontier.read_text(encoding="utf-8"))
    root = tree.root_box(Fraction(1, 500), Fraction(11, 50))
    boxes = tuple(
        outward_box(entry["evaluated"], args.denominator, root, args.margin_units)
        for entry in saved["queue"]
    )
    routing_boxes = tuple(
        outward_box(entry["evaluated"], args.denominator, root, 0)
        for entry in saved["queue"]
    )
    source = {
        "frontier_file": args.frontier.name,
        "frontier_splits": saved["splits_completed"],
        "frontier_leaves": len(boxes),
        "outward_denominator": args.denominator,
        "outward_margin_units": args.margin_units,
    }
    if args.resume:
        work = json.loads(args.work.read_text(encoding="utf-8"))
        if (
            work.get("schema") != "f5-projected-frontier-cover-work-v1"
            or work.get("source") != source
        ):
            raise ValueError("coverage reconstruction checkpoint does not match")
        nodes = work["nodes"]
        stack = [
            (
                int(value["node_id"]),
                decode_box(value["query"]),
                tuple(int(index) for index in value["candidates"]),
                int(value["depth"]),
            )
            for value in work["stack"]
        ]
        next_id = int(work["next_id"])
        max_depth = int(work["max_depth"])
    else:
        nodes: dict[str, object] = {}
        stack: list[tuple[int, Box, tuple[int, ...], int]] = [
            (0, root, tuple(range(len(boxes))), 0)
        ]
        next_id = 1
        max_depth = 0
    while stack:
        node_id, query, candidates, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if not candidates:
            candidates = tuple(
                index for index, box in enumerate(boxes) if intersects(box, query)
            )
        reason = tree.infeasibility_reason(query)
        tightened_query = checker.tighten_domain_box(query)
        if reason is None and tightened_query is None:
            reason = "domain_tightening"
        if reason is not None:
            nodes[str(node_id)] = {"kind": "infeasible", "reason": reason}
            continue
        containing = next(
            (
                index
                for index in candidates
                if checker.contains(boxes[index], tightened_query)
            ),
            None,
        )
        if containing is not None:
            nodes[str(node_id)] = {
                "kind": "frontier",
                "frontier_index": containing,
            }
            continue
        if not candidates:
            raise RuntimeError("frontier boxes leave an uncovered feasible cell")
        if len(nodes) + len(stack) >= args.max_nodes:
            stack.append((node_id, query, candidates, depth))
            save_work(
                args.work,
                source,
                nodes,
                stack,
                next_id,
                max_depth,
                "incomplete",
            )
            return {
                "status": "exact coverage reconstruction incomplete",
                "nodes_resolved": len(nodes),
                "pending": len(stack),
                "max_depth": max_depth,
                "work": args.work.name,
            }
        try:
            dimension, threshold = choose_split(
                query,
                candidates,
                routing_boxes,
                ZERO,
            )
        except RuntimeError:
            candidates = tuple(
                index for index, box in enumerate(boxes) if intersects(box, query)
            )
            containing = next(
                (
                    index
                    for index in candidates
                    if checker.contains(boxes[index], tightened_query)
                ),
                None,
            )
            if containing is not None:
                nodes[str(node_id)] = {
                    "kind": "frontier",
                    "frontier_index": containing,
                }
                continue
            dimension, threshold = choose_split(
                query,
                candidates,
                routing_boxes,
                ZERO,
            )
        left_id, right_id = next_id, next_id + 1
        next_id += 2
        nodes[str(node_id)] = {
            "kind": "split",
            "dimension": dimension,
            "threshold": text(threshold),
            "left": left_id,
            "right": right_id,
        }
        right = tree.child_box(query, dimension, threshold, False)
        left = tree.child_box(query, dimension, threshold, True)
        right_candidates = tuple(
            index
            for index in candidates
            if routing_boxes[index][1][dimension] > threshold
        )
        left_candidates = tuple(
            index
            for index in candidates
            if routing_boxes[index][0][dimension] < threshold
        )
        stack.append((right_id, right, right_candidates, depth + 1))
        stack.append((left_id, left, left_candidates, depth + 1))
        if args.progress_every > 0 and len(nodes) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "nodes_resolved": len(nodes),
                        "stack": len(stack),
                        "max_depth": max_depth,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    raw = {
        "schema": "f5-projected-frontier-cover-map-v1",
        "eta": "1/500",
        "delta": "11/50",
        "root_box": encode_box(root),
        "frontier_boxes": [encode_box(box) for box in boxes],
        "root": 0,
        "nodes": nodes,
        "frontier_identifiers": [
            int(entry["identifier"]) for entry in saved["queue"]
        ],
        "source": {**source, "max_depth": max_depth},
    }
    checker.verify(raw)
    projected.write_json_atomic(args.output, raw)
    save_work(
        args.work, source, nodes, [], next_id, max_depth, "complete"
    )
    return checker.verify(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", type=Path, default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"))
    parser.add_argument("--denominator", type=int, default=1_000_000_000_000)
    parser.add_argument("--margin-units", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work", type=Path, default=Path("results/f5-projected-frontier-cover-map-0.22.work.json"))
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results/f5-projected-frontier-cover-map-0.22.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(reconstruct(parse_args()), indent=2, sort_keys=True))
