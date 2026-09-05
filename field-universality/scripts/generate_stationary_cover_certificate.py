#!/usr/bin/env python3
"""Build a resumable exact stationary-box split cover for delta=0.22."""

from __future__ import annotations

import argparse
import heapq
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import explore_projected_kl_box_cover as projected
import generate_stationary_box_certificate as leaf_generator
import search_projected_kl_dual_envelope as envelope
import verify_stationary_box_certificate as leaf_checker
import verify_stationary_cover_certificate as cover_checker


def encode_box(box: tuple[tuple[Fraction, ...], tuple[Fraction, ...]]) -> dict[str, list[str]]:
    return {
        "lower": [cover_checker.text(value) for value in box[0]],
        "upper": [cover_checker.text(value) for value in box[1]],
    }


def decode_box(raw: object) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("missing work box")
    lower = leaf_checker.parse_vector(raw.get("lower"), 15)
    upper = leaf_checker.parse_vector(raw.get("upper"), 15)
    if any(left < 0 or left > right or right > 1 for left, right in zip(lower, upper, strict=True)):
        raise ValueError("invalid work box interval")
    return lower, upper


def load_bank(path: Path) -> tuple[list[object], list[object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return (
        [projected.decode_cut(value) for value in raw["first_cuts"]],
        [projected.decode_cut(value) for value in raw["second_cuts"]],
    )


def floating_evaluate(
    box: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    first: list[object],
    second: list[object],
    first_kernels: tuple[np.ndarray, ...],
    second_kernels: tuple[np.ndarray, ...],
    outer_iterations: int,
) -> tuple[object, ...] | None:
    raw = projected.Box(
        tuple(float(value) for value in box[0]),
        tuple(float(value) for value in box[1]),
    )
    return projected.evaluate(
        raw, first, second, first_kernels, second_kernels, outer_iterations
    )


def interior_reference(
    reference: np.ndarray,
    box: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
) -> np.ndarray:
    """Move a boundary LP witness minimally into the positive orthant."""
    lower, upper = box
    for floor in (1e-7, 1e-9, 1e-11, 1e-13):
        candidate = np.asarray(reference, dtype=float).copy()
        for start in (0, 5, 10):
            candidate[start : start + 5] = np.maximum(
                candidate[start : start + 5], floor
            )
            candidate[start : start + 5] /= candidate[start : start + 5].sum()
        if all(
            float(left) - 1e-12 <= value <= float(right) + 1e-12
            for value, left, right in zip(candidate, lower, upper, strict=True)
        ):
            message = candidate[1:5].sum()
            middle = candidate[6:10].sum()
            output = candidate[11:15].sum()
            if (
                message <= 2 * middle + 1e-12
                and middle <= 2 * output + 1e-12
                and np.all(candidate[2:5] <= candidate[1] + 1e-12)
            ):
                return candidate
    return np.asarray(reference, dtype=float)


def materialize(nodes: dict[str, object], node_id: int) -> dict[str, object]:
    node = nodes[str(node_id)]
    assert isinstance(node, dict)
    kind = node["kind"]
    if kind == "leaf":
        return {"kind": "leaf", "certificate": node["certificate"]}
    if kind == "infeasible":
        return {"kind": "infeasible", "reason": node["reason"]}
    if kind == "split":
        return {
            "kind": "split",
            "dimension": node["dimension"],
            "threshold": node["threshold"],
            "left": materialize(nodes, int(node["left"])),
            "right": materialize(nodes, int(node["right"])),
        }
    raise ValueError("cannot materialize an unfinished cover")


def save(path: Path, state: dict[str, object]) -> None:
    projected.write_json_atomic(path, state)


def run(args: argparse.Namespace) -> dict[str, object]:
    eta = Fraction(1, 500)
    delta = Fraction(11, 50)
    if args.resume and args.checkpoint.exists():
        state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if state.get("schema") != "f5-stationary-cover-work-v1":
            raise ValueError("unexpected cover checkpoint schema")
    else:
        root = cover_checker.root_box(eta, delta)
        state = {
            "schema": "f5-stationary-cover-work-v1",
            "eta": "1/500",
            "delta": "11/50",
            "root_box": encode_box(root),
            "root": 0,
            "next_id": 1,
            "splits": 0,
            "nodes": {"0": {"kind": "pending", "box": encode_box(root), "priority": 1.0}},
        }
    nodes = state["nodes"]
    assert isinstance(nodes, dict)
    first, second = load_bank(args.bank_cache)
    first_kernels = tuple(envelope.kernel(cut) for cut in first)
    second_kernels = tuple(envelope.kernel(cut) for cut in second)
    queue = []
    for raw_id, raw_node in nodes.items():
        if isinstance(raw_node, dict) and raw_node.get("kind") == "pending":
            heapq.heappush(queue, (-float(raw_node.get("priority", 1.0)), int(raw_id)))

    while queue and int(state["splits"]) < args.max_splits:
        _, node_id = heapq.heappop(queue)
        node = nodes[str(node_id)]
        assert isinstance(node, dict)
        box = decode_box(node["box"])
        evaluated = floating_evaluate(
            box, first, second, first_kernels, second_kernels, args.outer_iterations
        )
        forced_dimension = None
        forced_priority = None
        if evaluated is None:
            reason = cover_checker.infeasibility_reason(box)
            if reason is not None:
                nodes[str(node_id)] = {"kind": "infeasible", "reason": reason}
                continue
            widths = [right - left for left, right in zip(*box, strict=True)]
            forced_dimension = max(range(len(widths)), key=widths.__getitem__)
            forced_priority = float(node.get("priority", 1.0))
        if evaluated is None:
            reference = None
        else:
            reference = interior_reference(np.asarray(evaluated[2], dtype=float), box)
        certificate = None
        closed = False
        try:
            if reference is None:
                raise RuntimeError("no floating reference")
            certificate = leaf_generator.generate_box(
                box[0],
                box[1],
                reference,
                args,
                {"reference_source": "projected_lp_point", "node_id": node_id},
                require_negative=False,
            )
            leaf_checker.verify(certificate, args.precision)
            closed = True
        except RuntimeError as error:
            if "floating stationary fit failed" not in str(error) and "no floating reference" not in str(error):
                raise
        except ValueError as error:
            if "does not prove a negative box upper" not in str(error):
                raise
        if closed:
            assert certificate is not None
            nodes[str(node_id)] = {"kind": "leaf", "certificate": certificate}
            continue
        dimension = forced_dimension if forced_dimension is not None else int(evaluated[1])
        lower, upper = box
        threshold = (lower[dimension] + upper[dimension]) / 2
        left_id = int(state["next_id"])
        right_id = left_id + 1
        state["next_id"] = right_id + 1
        left = cover_checker.child_box(box, dimension, threshold, True)
        right = cover_checker.child_box(box, dimension, threshold, False)
        priority = forced_priority if forced_priority is not None else float(evaluated[0])
        nodes[str(node_id)] = {
            "kind": "split",
            "dimension": dimension,
            "threshold": cover_checker.text(threshold),
            "left": left_id,
            "right": right_id,
        }
        for child_id, child in ((left_id, left), (right_id, right)):
            nodes[str(child_id)] = {
                "kind": "pending",
                "box": encode_box(child),
                "priority": priority,
            }
            heapq.heappush(queue, (-priority, child_id))
        state["splits"] = int(state["splits"]) + 1
        if args.checkpoint_every > 0 and (
            int(state["splits"]) % args.checkpoint_every == 0
        ):
            save(args.checkpoint, state)
    save(args.checkpoint, state)
    pending = sum(
        isinstance(node, dict) and node.get("kind") == "pending"
        for node in nodes.values()
    )
    leaves = sum(
        isinstance(node, dict) and node.get("kind") == "leaf"
        for node in nodes.values()
    )
    infeasible = sum(
        isinstance(node, dict) and node.get("kind") == "infeasible"
        for node in nodes.values()
    )
    result = {
        "status": "exact stationary cover incomplete" if pending else "exact stationary cover generated",
        "splits": int(state["splits"]),
        "leaves": leaves,
        "infeasible": infeasible,
        "pending": pending,
        "nodes": len(nodes),
    }
    if not pending:
        final = {
            "schema": "f5-stationary-cover-v1",
            "prime": 5,
            "repetition": 3,
            "eta": "1/500",
            "delta": "11/50",
            "root_box": state["root_box"],
            "tree": materialize(nodes, int(state["root"])),
        }
        save(args.output, final)
        result["verification"] = cover_checker.verify(final, args.precision)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-splits", type=int, default=20)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--precision", type=int, default=100)
    parser.add_argument("--potential-denominator", type=int, default=100_000_000)
    parser.add_argument("--reference-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--dual-denominator", type=int, default=1_000_000_000)
    parser.add_argument("--bank-cache", type=Path, default=Path("results/f5-projected-kl-dual-bank-0.22.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/f5-stationary-cover-0.22.checkpoint.json"))
    parser.add_argument("--output", type=Path, default=Path("results/f5-stationary-cover-0.22.json"))
    args = parser.parse_args()
    if args.max_splits < 0 or args.precision < 50:
        parser.error("invalid split limit or interval precision")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
