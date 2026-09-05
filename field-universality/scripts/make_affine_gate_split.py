#!/usr/bin/env python3
"""Create two affine-gate child descriptors at an exact decimal split."""

from __future__ import annotations

import argparse
import copy
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def split_gate(
    parent: dict[str, Any], dimension: int, threshold: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0 <= dimension < 15:
        raise ValueError("split dimension must lie in [0, 15)")
    try:
        parsed = Decimal(threshold)
    except InvalidOperation as error:
        raise ValueError("split threshold must be an exact decimal") from error
    if not parsed.is_finite():
        raise ValueError("split threshold must be finite")
    rendered = format(parsed, "f")
    children = []
    for side in ("le", "ge"):
        child = copy.deepcopy(parent)
        child.setdefault("restrictions", []).append(
            f"{dimension}:{side}:{rendered}"
        )
        child["status"] = "floating affine-gate child descriptor; analytic bound pending"
        child["split_parent_status"] = parent.get("status")
        child["split_dimension"] = dimension
        child["split_side"] = side
        child["split_threshold"] = rendered
        children.append(child)
    return children[0], children[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("dimension", type=int)
    parser.add_argument("threshold")
    parser.add_argument("lower_child", type=Path)
    parser.add_argument("upper_child", type=Path)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    lower, upper = split_gate(parent, args.dimension, args.threshold)
    lower["split_parent"] = str(args.parent)
    upper["split_parent"] = str(args.parent)
    write_json(args.lower_child, lower)
    write_json(args.upper_child, upper)
    print(
        json.dumps(
            {
                "status": "created exact affine-gate split descriptors",
                "dimension": args.dimension,
                "threshold": lower["split_threshold"],
                "lower_child": str(args.lower_child),
                "upper_child": str(args.upper_child),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
