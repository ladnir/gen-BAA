#!/usr/bin/env python3
"""Adaptively propose a partition-aware derivative-sign cover.

The evaluator is run sequentially.  This discovery script deliberately emits
only a floating topology; exact Arb generation and independent replay are
separate acceptance steps.
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(order=True)
class Pending:
    priority: float
    identifier: int
    path: tuple[str, ...] = field(compare=False)
    bound: float = field(compare=False)


def evaluate(
    evaluator: Path,
    manifest: Path,
    partitions: Path,
    leaf: str,
    coordinate: int,
    fixed: list[str],
    path: tuple[str, ...],
) -> float:
    command = [
        sys.executable,
        str(evaluator),
        str(manifest),
        str(partitions),
        "--leaf",
        leaf,
        "--coordinate",
        str(coordinate),
        "--partition-aware-only",
        "--quiet",
    ]
    for specification in fixed:
        command.extend(("--fix", specification))
    for specification in path:
        command.extend(("--half", specification))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("derivative evaluator returned no summary")
    payload = json.loads(lines[-1])
    return float(payload["partition_aware_aggregate_lower_proposal"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--coordinate", type=int, required=True)
    parser.add_argument("--fix", action="append", default=[])
    parser.add_argument("--candidate", action="append", type=int, required=True)
    parser.add_argument("--initial-half", action="append", default=[])
    parser.add_argument("--max-splits", type=int, default=40)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("scripts/diagnose_affine_gate_expert_derivative_bernstein.py"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_splits < 0:
        parser.error("--max-splits must be nonnegative")
    candidates = list(dict.fromkeys(args.candidate))
    if args.coordinate in args.candidate:
        # Splitting the differentiated coordinate is allowed and often useful.
        pass

    if args.resume is None:
        initial_path = tuple(args.initial_half)
        initial_bound = evaluate(
            args.evaluator,
            args.manifest,
            args.partitions,
            args.leaf,
            args.coordinate,
            args.fix,
            initial_path,
        )
        pending: list[Pending] = []
        identifier = 0
        if initial_bound <= 0.0:
            heapq.heappush(
                pending, Pending(initial_bound, identifier, initial_path, initial_bound)
            )
            identifier += 1
        closed = []
        if initial_bound > 0.0:
            closed.append({"path": list(initial_path), "bound": initial_bound})
        splits = []
    else:
        resumed = json.loads(args.resume.read_text(encoding="utf-8"))
        if (
            resumed.get("leaf") != args.leaf
            or int(resumed.get("coordinate")) != args.coordinate
            or resumed.get("fixed") != args.fix
            or resumed.get("candidates") != candidates
        ):
            parser.error("resume artifact does not match the requested campaign")
        initial_path = tuple(resumed["initial_path"])
        initial_bound = float(resumed["initial_bound"])
        closed = list(resumed["closed_boxes"])
        splits = list(resumed["splits"])
        pending = []
        identifier = 0
        for value in resumed["pending_boxes"]:
            path = tuple(value["path"])
            bound = float(value["bound"])
            heapq.heappush(pending, Pending(bound, identifier, path, bound))
            identifier += 1
    while pending and len(splits) < args.max_splits:
        parent = heapq.heappop(pending)
        choices = []
        for split_coordinate in candidates:
            children = []
            for side in ("lower", "upper"):
                child_path = (*parent.path, f"{split_coordinate}:{side}")
                child_bound = evaluate(
                    args.evaluator,
                    args.manifest,
                    args.partitions,
                    args.leaf,
                    args.coordinate,
                    args.fix,
                    child_path,
                )
                children.append((child_path, child_bound))
            choices.append((min(value for _, value in children), split_coordinate, children))
        score, split_coordinate, children = max(choices, key=lambda value: value[0])
        split_record = {
            "parent_path": list(parent.path),
            "parent_bound": parent.bound,
            "coordinate": split_coordinate,
            "score": score,
            "children": [
                {"path": list(path), "bound": bound} for path, bound in children
            ],
        }
        splits.append(split_record)
        for path, bound in children:
            if bound > 0.0:
                closed.append({"path": list(path), "bound": bound})
            else:
                heapq.heappush(pending, Pending(bound, identifier, path, bound))
                identifier += 1
        print(
            json.dumps(
                {
                    "split": len(splits),
                    "coordinate": split_coordinate,
                    "parent_bound": parent.bound,
                    "child_bounds": [value for _, value in children],
                    "closed": len(closed),
                    "pending": len(pending),
                    "worst_pending": pending[0].bound if pending else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "status": (
            "floating partition-aware derivative cover proposal closed"
            if not pending
            else "floating partition-aware derivative cover proposal remains open"
        ),
        "rigorous": False,
        "leaf": args.leaf,
        "coordinate": args.coordinate,
        "fixed": args.fix,
        "candidates": candidates,
        "initial_path": list(initial_path),
        "initial_bound": initial_bound,
        "splits": splits,
        "split_count": len(splits),
        "closed_boxes": closed,
        "pending_boxes": [
            {"path": list(value.path), "bound": value.bound}
            for value in sorted(pending)
        ],
        "closed": not pending,
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "splits": payload["split_count"],
                "closed_boxes": len(closed),
                "pending_boxes": len(pending),
                "worst_pending": pending[0].bound if pending else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
