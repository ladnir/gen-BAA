#!/usr/bin/env python3
"""Import a verified local split tree into a projected global checkpoint."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import explore_projected_kl_box_cover as projected


def decode_box(value: dict[str, list[float]]) -> projected.Box:
    return projected.Box(tuple(value["lower"]), tuple(value["upper"]))


def boxes_equal(first: projected.Box, second: projected.Box) -> bool:
    return first.lower == second.lower and first.upper == second.upper


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("results/f5-projected-local-split-tree-0.22.json"),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path(
            "results/f5-projected-kl-box-cover-0.22-local-import.checkpoint.json"
        ),
    )
    parser.add_argument(
        "--output-split-log",
        type=Path,
        default=Path("results/f5-projected-kl-split-log-0.22-local-import.json"),
    )
    parser.add_argument("--outer-iterations", type=int, default=20)
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    split_log = json.loads(args.split_log.read_text(encoding="utf-8"))
    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    if checkpoint.get("version") != 1 or split_log.get("version") != 1:
        raise ValueError("unsupported projected checkpoint or split log")
    if tree.get("schema") != "f5-projected-local-split-tree-v1":
        raise ValueError("unsupported local split tree")
    if len(split_log["records"]) + int(split_log["base_splits"]) != int(
        checkpoint["splits_completed"]
    ):
        raise ValueError("global checkpoint and split log are out of sync")

    source_identifier = int(tree["source_identifier"])
    source_entries = [
        entry
        for entry in checkpoint["queue"]
        if int(entry["identifier"]) == source_identifier
    ]
    if len(source_entries) != 1:
        raise ValueError("local source is absent or duplicated")
    source = projected.decode_evaluated(source_entries[0]["evaluated"])

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

    global_queue = [
        (
            float(entry["priority"]),
            int(entry["identifier"]),
            projected.decode_evaluated(entry["evaluated"]),
        )
        for entry in checkpoint["queue"]
        if int(entry["identifier"]) != source_identifier
    ]
    heapq.heapify(global_queue)
    rest_upper = -global_queue[0][0]
    history = [float(value) for value in checkpoint["history"]]
    global_records = list(split_log["records"])
    next_global_identifier = int(checkpoint["identifier"])
    local_to_global = {0: source_identifier}
    active_boxes = {0: source[3]}
    active_bounds = {0: float(source[0])}
    starting_splits = int(checkpoint["splits_completed"])

    for offset, record in enumerate(tree["records"], start=1):
        local_parent = int(record["parent_identifier"])
        if local_parent not in active_boxes:
            raise ValueError("local tree parent is not active during import")
        parent_box = active_boxes.pop(local_parent)
        active_bounds.pop(local_parent)
        dimension = int(record["dimension"])
        expected_children = projected.split(parent_box, dimension)
        imported_children = []
        for child, expected_box in zip(
            record["children"], expected_children, strict=True
        ):
            local_child = int(child["identifier"])
            child_box = decode_box(child["box"])
            if not boxes_equal(child_box, expected_box):
                raise ValueError("local child does not match the replayed split")
            global_child = next_global_identifier
            next_global_identifier += 1
            local_to_global[local_child] = global_child
            active_boxes[local_child] = child_box
            active_bounds[local_child] = float(child["bound"])
            imported_children.append(
                {
                    "side": 0 if child["side"] == "le" else 1,
                    "identifier": global_child,
                    "raw_box": projected.encode_box(child_box),
                    "feasible_box": projected.encode_box(child_box),
                }
            )
        completed = starting_splits + offset
        global_records.append(
            {
                "after_split": completed,
                "parent_identifier": local_to_global[local_parent],
                "parent_box": projected.encode_box(parent_box),
                "dimension": dimension,
                "threshold": float(record["threshold"]),
                "rule": "midpoint",
                "children": imported_children,
            }
        )
        history.append(max(rest_upper, max(active_bounds.values())))

    terminal_by_identifier = {
        int(entry["identifier"]): entry for entry in tree["queue"]
    }
    if set(terminal_by_identifier) != set(active_boxes):
        raise ValueError("local terminal set does not match replayed active boxes")
    terminal_upper = float("-inf")
    for local_identifier, box in active_boxes.items():
        evaluated = projected.evaluate(
            box,
            first,
            second,
            first_kernels,
            second_kernels,
            args.outer_iterations,
        )
        if evaluated is None:
            continue
        terminal_upper = max(terminal_upper, float(evaluated[0]))
        global_identifier = local_to_global[local_identifier]
        heapq.heappush(
            global_queue, (-float(evaluated[0]), global_identifier, evaluated)
        )

    output_split_log = dict(split_log)
    output_split_log["records"] = global_records
    projected.save_split_log(args.output_split_log, output_split_log)
    projected.save_checkpoint(
        args.output_checkpoint,
        checkpoint["signature"],
        starting_splits + len(tree["records"]),
        next_global_identifier,
        history,
        global_queue,
    )
    print(
        json.dumps(
            {
                "status": "imported projected local split tree",
                "source_identifier": source_identifier,
                "imported_splits": len(tree["records"]),
                "terminal_upper": terminal_upper,
                "global_splits": starting_splits + len(tree["records"]),
                "global_leaves": len(global_queue),
                "global_upper": -global_queue[0][0],
                "next_identifier": next_global_identifier,
                "output_checkpoint": str(args.output_checkpoint),
                "output_split_log": str(args.output_split_log),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
