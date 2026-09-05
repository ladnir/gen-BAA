#!/usr/bin/env python3
"""Verify that saved frontier boxes cover the exact F_5 bulk domain."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import verify_stationary_box_certificate as box_parser
import verify_stationary_cover_certificate as tree


Box = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]


def contains(outer: Box, inner: Box) -> bool:
    return all(
        outer[0][index] <= inner[0][index]
        and inner[1][index] <= outer[1][index]
        for index in range(15)
    )


def tighten_simplex_box(box: Box) -> Box | None:
    lower = list(box[0])
    upper = list(box[1])
    changed = True
    while changed:
        changed = False
        for start in (0, 5, 10):
            if sum(lower[start : start + 5], Fraction(0)) > 1:
                return None
            if sum(upper[start : start + 5], Fraction(0)) < 1:
                return None
            for offset in range(5):
                index = start + offset
                new_lower = max(
                    lower[index],
                    1
                    - sum(
                        (upper[start + other] for other in range(5) if other != offset),
                        Fraction(0),
                    ),
                )
                new_upper = min(
                    upper[index],
                    1
                    - sum(
                        (lower[start + other] for other in range(5) if other != offset),
                        Fraction(0),
                    ),
                )
                if new_lower > new_upper:
                    return None
                changed |= new_lower != lower[index] or new_upper != upper[index]
                lower[index], upper[index] = new_lower, new_upper
    return tuple(lower), tuple(upper)


def tighten_domain_box(box: Box) -> Box | None:
    current = tighten_simplex_box(box)
    if current is None:
        return None
    inequalities = [
        {
            **{index: Fraction(1) for index in range(1, 5)},
            **{index: Fraction(-2) for index in range(6, 10)},
        },
        {
            **{index: Fraction(1) for index in range(6, 10)},
            **{index: Fraction(-2) for index in range(11, 15)},
        },
        *({value: Fraction(1), 1: Fraction(-1)} for value in range(2, 5)),
    ]
    changed = True
    while changed:
        changed = False
        lower, upper = list(current[0]), list(current[1])
        for inequality in inequalities:
            for index, coefficient in inequality.items():
                minimum_other = sum(
                    (
                        value * (lower[member] if value > 0 else upper[member])
                        for member, value in inequality.items()
                        if member != index
                    ),
                    Fraction(0),
                )
                boundary = -minimum_other / coefficient
                if coefficient > 0 and boundary < upper[index]:
                    upper[index] = boundary
                    changed = True
                elif coefficient < 0 and boundary > lower[index]:
                    lower[index] = boundary
                    changed = True
                if lower[index] > upper[index]:
                    return None
        tightened = tighten_simplex_box((tuple(lower), tuple(upper)))
        if tightened is None:
            return None
        changed |= tightened != current
        current = tightened
    return current


def parse_frontier_boxes(raw: object) -> tuple[Box, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("frontier box list is empty")
    return tuple(box_parser.parse_box(value) for value in raw)


def verify_node(
    nodes: dict[str, object],
    node_id: int,
    expected: Box,
    frontier: tuple[Box, ...],
    counts: dict[str, int],
) -> None:
    raw = nodes.get(str(node_id))
    if not isinstance(raw, dict):
        raise ValueError("coverage map references a missing node")
    kind = raw.get("kind")
    if kind == "frontier":
        index = raw.get("frontier_index")
        if not isinstance(index, int) or index not in range(len(frontier)):
            raise ValueError("invalid frontier index")
        tightened = tighten_domain_box(expected)
        if tightened is None or not contains(frontier[index], tightened):
            raise ValueError("frontier terminal does not contain its exact cell")
        counts["frontier"] += 1
        return
    if kind == "infeasible":
        reason = tree.infeasibility_reason(expected)
        if reason is None and tighten_domain_box(expected) is None:
            reason = "domain_tightening"
        if reason is None or raw.get("reason") != reason:
            raise ValueError("invalid infeasible coverage cell")
        counts["infeasible"] += 1
        return
    if kind != "split":
        raise ValueError("unknown coverage node kind")
    dimension = raw.get("dimension")
    if not isinstance(dimension, int) or dimension not in range(15):
        raise ValueError("invalid coverage split dimension")
    threshold = box_parser.parse_fraction(raw.get("threshold"))
    if not expected[0][dimension] < threshold < expected[1][dimension]:
        raise ValueError("coverage split lies outside its cell")
    counts["splits"] += 1
    verify_node(
        nodes,
        int(raw["left"]),
        tree.child_box(expected, dimension, threshold, True),
        frontier,
        counts,
    )
    verify_node(
        nodes,
        int(raw["right"]),
        tree.child_box(expected, dimension, threshold, False),
        frontier,
        counts,
    )


def verify(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or raw.get("schema") != "f5-projected-frontier-cover-map-v1":
        raise ValueError("unsupported frontier coverage schema")
    if raw.get("eta") != "1/500" or raw.get("delta") != "11/50":
        raise ValueError("unexpected bulk domain")
    root = tree.root_box(Fraction(1, 500), Fraction(11, 50))
    if box_parser.parse_box(raw.get("root_box")) != root:
        raise ValueError("coverage map has the wrong root box")
    frontier = parse_frontier_boxes(raw.get("frontier_boxes"))
    identifiers = raw.get("frontier_identifiers")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) != len(frontier)
        or any(not isinstance(value, int) for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("frontier identifiers are missing or not unique")
    nodes = raw.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("coverage node table is missing")
    root_id = raw.get("root")
    if not isinstance(root_id, int):
        raise ValueError("coverage root id is invalid")
    counts = {"splits": 0, "frontier": 0, "infeasible": 0}
    verify_node(nodes, root_id, root, frontier, counts)
    reachable = 2 * counts["splits"] + 1
    if reachable != len(nodes):
        raise ValueError("coverage map contains unreachable nodes")
    return {
        "status": "verified exact frontier coverage map",
        "splits_checked": counts["splits"],
        "frontier_cells_checked": counts["frontier"],
        "infeasible_cells_checked": counts["infeasible"],
        "frontier_boxes": len(frontier),
        "frontier_identifiers_checked": len(identifiers),
        "nodes_checked": reachable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(raw), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
