#!/usr/bin/env python3
"""Replay and optionally re-evaluate a projected local split tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import explore_projected_kl_box_cover as projected


def boxes_close(first: projected.Box, second: projected.Box) -> bool:
    return all(
        abs(left - right) <= 1e-14
        for left, right in zip(
            first.lower + first.upper,
            second.lower + second.upper,
            strict=True,
        )
    )


def decode_box(value: dict[str, list[float]]) -> projected.Box:
    return projected.Box(tuple(value["lower"]), tuple(value["upper"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tree",
        type=Path,
        nargs="?",
        default=Path("results/f5-projected-local-split-tree-0.22.json"),
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--recheck-bounds", action="store_true")
    args = parser.parse_args()

    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    if tree.get("schema") != "f5-projected-local-split-tree-v1":
        raise ValueError("unexpected local split-tree schema")
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    source_identifier = int(tree["source_identifier"])
    matches = [
        entry
        for entry in frontier["queue"]
        if int(entry["identifier"]) == source_identifier
    ]
    if len(matches) > 1:
        raise ValueError("source frontier cell is duplicated")
    if matches:
        source_box = projected.decode_evaluated(matches[0]["evaluated"])[3]
    else:
        if not tree["records"]:
            raise ValueError("cannot reconstruct an imported unsplit source")
        first_record = tree["records"][0]
        first_children = first_record["children"]
        if len(first_children) != 2:
            raise ValueError("cannot reconstruct the imported source box")
        left = decode_box(first_children[0]["box"])
        right = decode_box(first_children[1]["box"])
        dimension = int(first_record["dimension"])
        if abs(left.upper[dimension] - right.lower[dimension]) > 1e-14:
            raise ValueError("imported source children do not meet")
        source_box = projected.Box(left.lower, right.upper)
    active = {0: source_box}

    for sequence, record in enumerate(tree["records"], start=1):
        if int(record["sequence"]) != sequence:
            raise ValueError("split-tree sequence is not contiguous")
        parent_identifier = int(record["parent_identifier"])
        if parent_identifier not in active:
            raise ValueError("split-tree parent is not active")
        parent = active.pop(parent_identifier)
        dimension = int(record["dimension"])
        expected_threshold = 0.5 * (
            parent.lower[dimension] + parent.upper[dimension]
        )
        if abs(float(record["threshold"]) - expected_threshold) > 1e-14:
            raise ValueError("split-tree threshold is not the parent midpoint")
        expected_children = projected.split(parent, dimension)
        children = record["children"]
        if len(children) != 2:
            raise ValueError("split-tree record does not contain two children")
        for child_record, expected_side, expected_box in zip(
            children, ("le", "ge"), expected_children, strict=True
        ):
            if child_record["side"] != expected_side:
                raise ValueError("split-tree child side is invalid")
            child_identifier = int(child_record["identifier"])
            if child_identifier in active:
                raise ValueError("split-tree reuses an active identifier")
            child_box = decode_box(child_record["box"])
            if not boxes_close(child_box, expected_box):
                raise ValueError("split-tree child box does not match its split")
            active[child_identifier] = child_box

    queue = tree["queue"]
    queue_identifiers = {int(entry["identifier"]) for entry in queue}
    if queue_identifiers != set(active):
        raise ValueError("terminal identifiers do not match replayed leaves")
    for entry in queue:
        identifier = int(entry["identifier"])
        if not boxes_close(decode_box(entry["box"]), active[identifier]):
            raise ValueError("terminal box does not match replayed leaf")

    rechecked = []
    if args.recheck_bounds:
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
        for entry in queue:
            evaluated = projected.evaluate(
                decode_box(entry["box"]),
                first,
                second,
                first_kernels,
                second_kernels,
                args.outer_iterations,
            )
            if evaluated is None:
                continue
            rechecked.append(float(evaluated[0]))

    stored_bounds = [float(entry["bound"]) for entry in queue]
    result = {
        "status": "verified floating projected local split tree",
        "source_identifier": source_identifier,
        "splits": len(tree["records"]),
        "leaves": len(queue),
        "stored_upper": max(stored_bounds, default=float("-inf")),
        "stored_positive_leaves": sum(bound >= 0.0 for bound in stored_bounds),
    }
    if args.recheck_bounds:
        result.update(
            {
                "rechecked_nonempty_leaves": len(rechecked),
                "rechecked_upper": max(rechecked, default=float("-inf")),
                "rechecked_positive_leaves": sum(
                    bound >= 0.0 for bound in rechecked
                ),
                "outer_iterations": args.outer_iterations,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
