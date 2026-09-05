#!/usr/bin/env python3
"""Check an exact split tree against floating affine-gate artifacts.

This checker deliberately does not promote floating optimization output to a
proof.  It verifies that the rational chamber tree is a complete binary cover
and that every referenced numerical result satisfies one uniform acceptance
rule.  Directed evaluation of the analytic inequalities is a separate step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


class CoverError(ValueError):
    """Raised when a cover manifest or one of its artifacts is inconsistent."""


Bounds = dict[int, dict[str, Fraction]]


@dataclass(frozen=True)
class CoverSummary:
    leaves: int
    splits: int
    worst_log_upper: float
    weakest_leaf: str


def _fraction(value: Any, context: str) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise CoverError(f"{context}: invalid rational {value!r}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CoverError(f"missing artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoverError(f"expected a JSON object in {path}")
    return value


def _copy_bounds(bounds: Bounds) -> Bounds:
    return {dimension: dict(sides) for dimension, sides in bounds.items()}


def _add_bound(bounds: Bounds, dimension: int, side: str, threshold: Fraction) -> None:
    if side not in {"le", "ge"}:
        raise CoverError(f"dimension {dimension}: invalid side {side!r}")
    sides = bounds.setdefault(dimension, {})
    old = sides.get(side)
    if old is None or (side == "ge" and threshold > old) or (side == "le" and threshold < old):
        sides[side] = threshold
    if sides.get("ge", threshold) > sides.get("le", threshold):
        raise CoverError(
            f"dimension {dimension}: empty chamber "
            f"[{sides.get('ge')}, {sides.get('le')}]"
        )


def _parse_gate_bounds(restrictions: Any, context: str) -> Bounds:
    if not isinstance(restrictions, list):
        raise CoverError(f"{context}: restrictions must be a list")
    bounds: Bounds = {}
    for restriction in restrictions:
        if not isinstance(restriction, str):
            raise CoverError(f"{context}: non-string restriction {restriction!r}")
        fields = restriction.split(":")
        if len(fields) != 3:
            raise CoverError(f"{context}: malformed restriction {restriction!r}")
        try:
            dimension = int(fields[0])
        except ValueError as exc:
            raise CoverError(f"{context}: malformed dimension in {restriction!r}") from exc
        _add_bound(bounds, dimension, fields[1], _fraction(fields[2], context))
    return bounds


def _resolve(root: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CoverError(f"{context}: expected a nonempty artifact path")
    return (root / Path(relative.replace("\\", "/"))).resolve()


def verify_manifest(manifest_path: Path) -> CoverSummary:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("certificate_status") != "floating_only_not_a_proof":
        raise CoverError("manifest must explicitly identify itself as floating-only")
    frontier_index = manifest.get("frontier_index")
    if not isinstance(frontier_index, int) or frontier_index < 0:
        raise CoverError("frontier_index must be a nonnegative integer")
    target = _fraction(manifest.get("target_margin"), "target_margin")
    if target <= 0:
        raise CoverError("target_margin must be positive")
    artifact_root = manifest.get("artifact_root", "..")
    root = (manifest_path.parent / str(artifact_root)).resolve()

    leaf_names: set[str] = set()
    split_count = 0
    worst_upper = float("-inf")
    weakest_leaf = ""

    def visit(node: Any, bounds: Bounds, context: str) -> None:
        nonlocal split_count, worst_upper, weakest_leaf
        if not isinstance(node, dict):
            raise CoverError(f"{context}: node must be an object")
        is_leaf = "leaf" in node
        is_split = "split" in node
        if is_leaf == is_split:
            raise CoverError(f"{context}: node must contain exactly one of leaf or split")

        if is_split:
            if set(node) != {"split", "le", "ge"}:
                raise CoverError(f"{context}: split node must contain exactly split, le, and ge")
            split = node["split"]
            if not isinstance(split, dict) or set(split) != {"dimension", "threshold"}:
                raise CoverError(f"{context}: malformed split descriptor")
            dimension = split["dimension"]
            if not isinstance(dimension, int) or dimension < 0:
                raise CoverError(f"{context}: split dimension must be a nonnegative integer")
            threshold = _fraction(split["threshold"], f"{context}.split.threshold")
            split_count += 1
            for side in ("le", "ge"):
                child_bounds = _copy_bounds(bounds)
                _add_bound(child_bounds, dimension, side, threshold)
                visit(node[side], child_bounds, f"{context}.{side}")
            return

        if set(node) != {"leaf", "gate", "result"}:
            raise CoverError(f"{context}: leaf node must contain exactly leaf, gate, and result")
        leaf = node["leaf"]
        if not isinstance(leaf, str) or not leaf:
            raise CoverError(f"{context}: leaf name must be nonempty")
        if leaf in leaf_names:
            raise CoverError(f"duplicate leaf name: {leaf}")
        leaf_names.add(leaf)

        gate_path = _resolve(root, node["gate"], f"{leaf}.gate")
        result_path = _resolve(root, node["result"], f"{leaf}.result")
        gate = _read_json(gate_path)
        result = _read_json(result_path)
        if gate.get("frontier_index") != frontier_index:
            raise CoverError(f"{leaf}: gate has wrong frontier_index")
        gate_bounds = _parse_gate_bounds(gate.get("restrictions"), leaf)
        if gate_bounds != bounds:
            raise CoverError(
                f"{leaf}: effective gate restrictions do not match tree chamber; "
                f"gate={gate_bounds}, tree={bounds}"
            )
        result_gate_path = _resolve(root, result.get("gate"), f"{leaf}.result.gate")
        if result_gate_path != gate_path:
            raise CoverError(f"{leaf}: result points to {result_gate_path}, expected {gate_path}")
        if result.get("closed") is not True or result.get("complete") is not True:
            raise CoverError(f"{leaf}: result is not closed and complete")
        if result.get("termination") != "target margin reached":
            raise CoverError(f"{leaf}: result did not terminate by reaching the target margin")
        result_target = _fraction(result.get("target_margin"), f"{leaf}.target_margin")
        if result_target < target:
            raise CoverError(f"{leaf}: result target {result_target} is weaker than {target}")
        history = result.get("history")
        if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
            raise CoverError(f"{leaf}: result has no final history entry")
        upper = history[-1].get("continuous_log_upper")
        if not isinstance(upper, (int, float)):
            raise CoverError(f"{leaf}: final continuous_log_upper is missing")
        if _fraction(upper, f"{leaf}.continuous_log_upper") > -target:
            raise CoverError(f"{leaf}: final upper {upper} does not reach margin {target}")
        if float(upper) > worst_upper:
            worst_upper = float(upper)
            weakest_leaf = leaf

    tree = manifest.get("tree")
    visit(tree, {}, "tree")
    if split_count != len(leaf_names) - 1:
        raise CoverError(
            f"non-binary topology: {split_count} splits for {len(leaf_names)} leaves"
        )
    return CoverSummary(
        leaves=len(leaf_names),
        splits=split_count,
        worst_log_upper=worst_upper,
        weakest_leaf=weakest_leaf,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_manifest(args.manifest)
    except CoverError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "verified floating topology; not a proof",
                "leaves": summary.leaves,
                "splits": summary.splits,
                "weakest_leaf": summary.weakest_leaf,
                "worst_log_upper": summary.worst_log_upper,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
