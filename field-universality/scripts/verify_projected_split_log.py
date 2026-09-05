#!/usr/bin/env python3
"""Verify incremental projected frontier splits against an exact base map."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import verify_projected_frontier_cover_map as map_checker
import verify_stationary_cover_certificate as tree


Box = map_checker.Box
OptionalBox = Box | None


def nearest(value: object, denominator: int) -> Fraction:
    return Fraction(round(float(value) * denominator), denominator)


def outward_box(
    raw: object,
    denominator: int,
    margin: int,
    root: Box,
) -> Box:
    if not isinstance(raw, dict):
        raise ValueError("split log box must be an object")
    lower = []
    upper = []
    for index, (left, right) in enumerate(
        zip(raw.get("lower", ()), raw.get("upper", ()), strict=True)
    ):
        lower.append(
            max(
                root[0][index],
                Fraction(round(float(left) * denominator) - margin, denominator),
            )
        )
        upper.append(
            min(
                root[1][index],
                Fraction(round(float(right) * denominator) + margin, denominator),
            )
        )
    if len(lower) != 15 or any(left > right for left, right in zip(lower, upper, strict=True)):
        raise ValueError("invalid split log box")
    return tuple(lower), tuple(upper)


def split_exact_cell(
    parent: OptionalBox,
    dimension: int,
    threshold: Fraction,
    lower_side: bool,
) -> OptionalBox:
    """Intersect one exact partition cell with a logged floating half-space."""
    if parent is None:
        return None
    lower = parent[0][dimension]
    upper = parent[1][dimension]
    if lower_side:
        if threshold < lower:
            return None
        if threshold >= upper:
            return parent
    else:
        if threshold > upper:
            return None
        if threshold <= lower:
            return parent
    return tree.child_box(parent, dimension, threshold, lower_side)


def verify(base: object, log: object) -> dict[str, object]:
    base_result = map_checker.verify(base)
    if not isinstance(base, dict) or not isinstance(log, dict):
        raise ValueError("invalid split-log inputs")
    if log.get("version") != 1:
        raise ValueError("unexpected split-log schema")
    source = base.get("source")
    if not isinstance(source, dict):
        raise ValueError("base coverage map lacks source metadata")
    denominator = int(source["outward_denominator"])
    margin = int(source["outward_margin_units"])
    split_margin = int(log.get("outward_margin_units", 10000))
    if split_margin < margin:
        raise ValueError("split-log outward margin is smaller than the base margin")
    root = tree.root_box(Fraction(1, 500), Fraction(11, 50))
    frontier = map_checker.parse_frontier_boxes(base["frontier_boxes"])
    identifiers = tuple(int(value) for value in base["frontier_identifiers"])
    if sorted(log.get("base_identifiers", ())) != sorted(identifiers):
        raise ValueError("split-log base identifiers do not match the coverage map")
    if int(log.get("base_splits")) != int(source["frontier_splits"]):
        raise ValueError("split-log base count does not match the coverage map")
    active = dict(zip(identifiers, frontier, strict=True))
    records = log.get("records")
    if not isinstance(records, list):
        raise ValueError("split-log record list is missing")
    expected_split = int(log["base_splits"]) + 1
    infeasible_children = 0
    for record in records:
        if not isinstance(record, dict) or int(record.get("after_split")) != expected_split:
            raise ValueError("split-log sequence is not contiguous")
        expected_split += 1
        parent_identifier = int(record["parent_identifier"])
        if parent_identifier not in active:
            raise ValueError("split-log parent is not active")
        parent = active.pop(parent_identifier)
        stored_parent = outward_box(
            record.get("parent_box"), denominator, split_margin, root
        )
        if parent is not None and not map_checker.contains(stored_parent, parent):
            raise ValueError("split-log parent box does not contain its active box")
        dimension = record.get("dimension")
        if not isinstance(dimension, int) or dimension not in range(15):
            raise ValueError("invalid split-log dimension")
        threshold = nearest(record.get("threshold"), denominator)
        if not stored_parent[0][dimension] <= threshold <= stored_parent[1][dimension]:
            raise ValueError(
                "split-log threshold lies outside its logged parent: "
                f"after_split={record['after_split']}, dimension={dimension}, "
                f"threshold={threshold}, "
                f"parent=({stored_parent[0][dimension]}, {stored_parent[1][dimension]})"
            )
        children = record.get("children")
        if not isinstance(children, list) or len(children) != 2:
            raise ValueError("split log must contain two child records")
        by_side = {int(value["side"]): value for value in children}
        if set(by_side) != {0, 1}:
            raise ValueError("split-log child sides are invalid")
        for side in (0, 1):
            exact_child = split_exact_cell(parent, dimension, threshold, side == 0)
            raw_child = outward_box(
                by_side[side].get("raw_box"), denominator, split_margin, root
            )
            tightened = (
                None
                if exact_child is None
                else map_checker.tighten_domain_box(exact_child)
            )
            identifier = by_side[side].get("identifier")
            if identifier is None:
                if tightened is not None:
                    raise ValueError("split log drops a feasible exact child")
                infeasible_children += 1
                continue
            feasible = outward_box(
                by_side[side].get("feasible_box"), denominator, split_margin, root
            )
            if exact_child is not None and not map_checker.contains(raw_child, exact_child):
                raise ValueError("raw logged child does not contain its exact split cell")
            if tightened is not None and not map_checker.contains(feasible, tightened):
                violations = [
                    index
                    for index in range(15)
                    if feasible[0][index] > tightened[0][index]
                    or feasible[1][index] < tightened[1][index]
                ]
                raise ValueError(
                    "logged feasible child misses its tightened exact cell: "
                    f"after_split={record['after_split']}, side={side}, "
                    f"dimensions={violations}, "
                    f"feasible={[(str(feasible[0][i]), str(feasible[1][i])) for i in violations]}, "
                    f"tightened={[(str(tightened[0][i]), str(tightened[1][i])) for i in violations]}"
                )
            identifier = int(identifier)
            if identifier in active:
                raise ValueError("split log reuses an active identifier")
            active[identifier] = tightened
    nonempty_active = sum(box is not None for box in active.values())
    return {
        "status": "verified exact incremental projected split log",
        "base_map_nodes_checked": base_result["nodes_checked"],
        "records_checked": len(records),
        "base_splits": int(log["base_splits"]),
        "final_splits": expected_split - 1,
        "active_boxes": len(active),
        "nonempty_exact_cells": nonempty_active,
        "empty_floating_cells": len(active) - nonempty_active,
        "infeasible_children_checked": infeasible_children,
        "outward_denominator": denominator,
        "outward_margin_units": margin,
        "split_outward_margin_units": split_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_map", type=Path)
    parser.add_argument("split_log", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base_map.read_text(encoding="utf-8"))
    log = json.loads(args.split_log.read_text(encoding="utf-8"))
    print(json.dumps(verify(base, log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
