#!/usr/bin/env python3
"""Extract boxes with nonnegative support-indicator upper bounds.

This replays the exact support cover but retains its terminal queue.  The
result is a heuristic diagnostic only: the internal-information quantity G
can be negative, so a negative support exponent Phi does not by itself imply
a negative full exponent Psi = Phi - G.  Consequently, the displayed
positive frontier is not a proof-theoretic partition of the full domain.
"""

from __future__ import annotations

import argparse
import heapq
import json
from fractions import Fraction

from mpmath import iv

import verify_support_indicator_bulk as support


def support_frontier_entries(
    args: argparse.Namespace,
) -> tuple[int, list[tuple[Fraction, support.Box]], list[tuple[Fraction, support.Box]]]:
    iv.dps = args.precision
    initial_box = support.Box(
        (args.eta, args.eta / 2, args.eta / 4),
        (args.cutoff, 2 * args.delta, args.delta),
    )
    initial = support.box_bound(
        initial_box, args.prime, args.repetition, internal_values_paid=True
    )
    if initial is None:
        raise RuntimeError("initial support box is infeasible")
    queue = [(-initial[0], 0, initial[1], initial[2])]
    identifier = 1
    splits = 0
    while queue and -queue[0][0] >= args.target:
        if splits >= args.max_boxes:
            raise RuntimeError("support frontier split budget exhausted")
        _, _, dimension, box = heapq.heappop(queue)
        for child in support.split(box, dimension):
            result = support.box_bound(
                child,
                args.prime,
                args.repetition,
                internal_values_paid=True,
            )
            if result is None:
                continue
            bound, child_dimension, tightened = result
            heapq.heappush(
                queue,
                (-bound, identifier, child_dimension, tightened),
            )
            identifier += 1
        splits += 1

    terminal = [(-entry[0], entry[3]) for entry in queue]
    terminal.sort(key=lambda entry: entry[0], reverse=True)
    open_boxes = [entry for entry in terminal if entry[0] >= 0]
    return splits, terminal, open_boxes


def frontier(args: argparse.Namespace) -> dict[str, object]:
    splits, terminal, open_boxes = support_frontier_entries(args)
    return {
        "splits": splits,
        "terminal_boxes": len(terminal),
        "positive_frontier_boxes": len(open_boxes),
        "maximum_terminal_upper": float(terminal[0][0]),
        "minimum_positive_upper": (
            float(min(entry[0] for entry in open_boxes))
            if open_boxes
            else None
        ),
        "positive_frontier": [
            {
                "upper_bound": float(bound),
                "lower": [str(value) for value in box.lower],
                "upper": [str(value) for value in box.upper],
            }
            for bound, box in open_boxes[: args.display]
        ],
        "displayed_boxes": min(len(open_boxes), args.display),
        "target": str(args.target),
        "status": (
            "exact rational support boxes, but heuristic-only as a full-symbol frontier"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=Fraction, default=Fraction(1323, 10_000))
    parser.add_argument("--eta", type=Fraction, default=Fraction(1, 500))
    parser.add_argument("--cutoff", type=Fraction, default=Fraction(1323, 2500))
    parser.add_argument("--target", type=Fraction, default=Fraction(1, 10_000))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--max-boxes", type=int, default=100_000)
    parser.add_argument("--display", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(frontier(parse_args()), indent=2, sort_keys=True))
