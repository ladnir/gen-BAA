#!/usr/bin/env python3
"""Close one projected-KL frontier cell with an adaptive local split tree.

This is a floating discovery tool.  At each positive leaf it evaluates every
nondegenerate coordinate split, selects the split with the smallest maximum
child upper, and records both children.  The resulting tree is intended to
design a later rational disjunctive certificate.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import explore_projected_kl_box_cover as projected


def encode_box(box: projected.Box) -> dict[str, list[float]]:
    return {"lower": list(box.lower), "upper": list(box.upper)}


def decode_box(value: dict[str, list[float]]) -> projected.Box:
    return projected.Box(tuple(value["lower"]), tuple(value["upper"]))


def save_checkpoint(
    path: Path,
    source_identifier: int,
    queue: list[tuple[float, int, projected.Box, list[dict[str, float | int | str]]]],
    records: list[dict[str, object]],
    next_identifier: int,
) -> None:
    ordered = sorted(queue)
    payload = {
        "schema": "f5-projected-local-split-tree-v1",
        "source_identifier": source_identifier,
        "next_identifier": next_identifier,
        "records": records,
        "queue": [
            {
                "identifier": identifier,
                "bound": -priority,
                "box": encode_box(box),
                "path": node_path,
            }
            for priority, identifier, box, node_path in ordered
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/f5-projected-local-split-tree-0.22.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=0)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--max-splits", type=int, default=20)
    parser.add_argument(
        "--candidate-dimensions",
        default="",
        help="comma-separated coordinate subset; empty means all 15",
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
    saved_frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected_entry = sorted(
        saved_frontier["queue"], key=lambda entry: entry["priority"]
    )[args.frontier_index]
    source_identifier = int(selected_entry["identifier"])
    source = projected.decode_evaluated(selected_entry["evaluated"])

    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if saved.get("schema") != "f5-projected-local-split-tree-v1":
            raise ValueError("unexpected local split-tree schema")
        if int(saved["source_identifier"]) != source_identifier:
            raise ValueError("local split tree belongs to another frontier cell")
        records = list(saved["records"])
        next_identifier = int(saved["next_identifier"])
        queue = [
            (
                -float(entry["bound"]),
                int(entry["identifier"]),
                decode_box(entry["box"]),
                list(entry["path"]),
            )
            for entry in saved["queue"]
        ]
        heapq.heapify(queue)
    else:
        queue = [(-float(source[0]), 0, source[3], [])]
        records = []
        next_identifier = 1

    candidate_dimensions = (
        [int(value) for value in args.candidate_dimensions.split(",")]
        if args.candidate_dimensions
        else list(range(15))
    )
    if not candidate_dimensions or any(
        dimension < 0 or dimension >= 15 for dimension in candidate_dimensions
    ):
        raise ValueError("candidate dimensions must lie in [0,15)")
    target_split_count = len(records) + args.max_splits
    while queue and -queue[0][0] >= 0.0 and len(records) < target_split_count:
        neg_bound, parent_identifier, parent_box, parent_path = heapq.heappop(queue)
        parent_bound = -neg_bound
        best = None
        for dimension in candidate_dimensions:
            if parent_box.upper[dimension] - parent_box.lower[dimension] <= 1e-14:
                continue
            evaluated_children = []
            for side, child in zip(("le", "ge"), projected.split(parent_box, dimension), strict=True):
                evaluated = projected.evaluate(
                    child,
                    first,
                    second,
                    first_kernels,
                    second_kernels,
                    args.outer_iterations,
                )
                if evaluated is None:
                    continue
                evaluated_children.append(
                    (side, child, min(parent_bound, float(evaluated[0])))
                )
            child_maximum = max(
                (child[2] for child in evaluated_children), default=float("-inf")
            )
            candidate = (child_maximum, dimension, evaluated_children)
            if best is None or candidate[0:2] < best[0:2]:
                best = candidate
        if best is None:
            raise RuntimeError("positive leaf has no nondegenerate split")

        child_maximum, dimension, evaluated_children = best
        threshold = 0.5 * (
            parent_box.lower[dimension] + parent_box.upper[dimension]
        )
        child_records = []
        for side, child_box, child_bound in evaluated_children:
            identifier = next_identifier
            next_identifier += 1
            child_path = parent_path + [
                {"dimension": dimension, "side": side, "threshold": threshold}
            ]
            heapq.heappush(
                queue, (-child_bound, identifier, child_box, child_path)
            )
            child_records.append(
                {
                    "identifier": identifier,
                    "side": side,
                    "bound": child_bound,
                    "box": encode_box(child_box),
                }
            )
        records.append(
            {
                "sequence": len(records) + 1,
                "parent_identifier": parent_identifier,
                "parent_bound": parent_bound,
                "dimension": dimension,
                "threshold": threshold,
                "child_maximum": child_maximum,
                "children": child_records,
            }
        )
        save_checkpoint(
            args.checkpoint,
            source_identifier,
            queue,
            records,
            next_identifier,
        )
        positive = sum(1 for entry in queue if -entry[0] >= 0.0)
        print(
            json.dumps(
                {
                    "splits": len(records),
                    "leaves": len(queue),
                    "positive_leaves": positive,
                    "upper": -queue[0][0],
                    "split_dimension": dimension,
                    "split_child_maximum": child_maximum,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    save_checkpoint(
        args.checkpoint,
        source_identifier,
        queue,
        records,
        next_identifier,
    )
    print(
        json.dumps(
            {
                "status": (
                    "closed floating local split tree"
                    if queue and -queue[0][0] < 0.0
                    else "floating local split tree remains open"
                ),
                "splits": len(records),
                "leaves": len(queue),
                "positive_leaves": sum(1 for entry in queue if -entry[0] >= 0.0),
                "upper": -queue[0][0] if queue else float("-inf"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
