#!/usr/bin/env python3
"""Numerically maximize one affine-gated exponential cut mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import explore_projected_kl_box_cover as projected


def domain_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equality = np.zeros((3, 15))
    for row, start in enumerate((0, 5, 10)):
        equality[row, start : start + 5] = 1.0
    upper = []
    support_first = np.zeros(15)
    support_first[1:5] = 1.0
    support_first[6:10] = -2.0
    upper.append(support_first)
    support_second = np.zeros(15)
    support_second[6:10] = 1.0
    support_second[11:15] = -2.0
    upper.append(support_second)
    for value in range(2, 5):
        canonical = np.zeros(15)
        canonical[value] = 1.0
        canonical[1] = -1.0
        upper.append(canonical)
    return equality, np.ones(3), np.stack(upper), np.zeros(len(upper))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    parser.add_argument(
        "--rational-partitions",
        type=Path,
        help="override the saved gate with one exact rational partition record",
    )
    parser.add_argument(
        "--leaf",
        help="leaf name in --rational-partitions (defaults to the gate basename match)",
    )
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
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="fix a coordinate to its current chamber lower or upper bound",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    saved_gate = json.loads(args.gate.read_text(encoding="utf-8"))
    affine = saved_gate.get("affine_gating")
    if not isinstance(affine, dict):
        raise ValueError("gate file does not contain affine coefficients")
    bank_args = SimpleNamespace(
        eta=0.002,
        delta=0.22,
        bank_iterations=20,
        bank_restarts=12,
        seed=11,
        bank_cache=str(args.bank),
        rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        int(saved_gate["frontier_index"])
    ]["evaluated"]
    lower = list(selected["lower"])
    upper = list(selected["upper"])
    for specification in saved_gate["restrictions"]:
        dimension_text, side, value_text = specification.split(":")
        dimension = int(dimension_text)
        value = float(value_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], value)
        else:
            lower[dimension] = max(lower[dimension], value)
    for specification in args.fix:
        dimension_text, side = specification.split(":")
        dimension = int(dimension_text)
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            parser.error("--fix side must be lower or upper")
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)

    expert_pairs = affine["expert_pairs"]
    coefficients = np.asarray(affine["coefficients"])
    rational_record = None
    if args.rational_partitions is not None:
        saved_partitions = json.loads(
            args.rational_partitions.read_text(encoding="utf-8")
        )
        candidates = saved_partitions.get("leaves", [])
        if args.leaf is not None:
            candidates = [entry for entry in candidates if entry.get("leaf") == args.leaf]
        else:
            normalized_gate = str(args.gate).replace("\\", "/")
            candidates = [
                entry
                for entry in candidates
                if str(entry.get("gate", "")).replace("\\", "/") == normalized_gate
            ]
        if len(candidates) != 1:
            parser.error("rational partition selection did not identify exactly one leaf")
        rational_record = candidates[0]
        expert_pairs = rational_record["expert_pairs"]
        denominator = float(int(rational_record["coefficient_denominator"]))
        coefficients = (
            np.asarray(rational_record["coefficient_numerators"], dtype=float)
            / denominator
        )
    feature_coordinates = [
        int(value)
        for value in (
            rational_record["feature_coordinates"]
            if rational_record is not None
            else affine["feature_coordinates"]
        )
    ]
    equality, equality_target, domain_upper, domain_target = domain_matrices()
    linear_constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(domain_upper, -np.inf, domain_target),
    ]
    bounds = Bounds(lower_array, upper_array)

    def value(point: np.ndarray) -> tuple[float, np.ndarray]:
        features = np.concatenate(([1.0], point[feature_coordinates]))
        gating_weights = coefficients @ features
        message_entropy = projected.bulk_entropy(point[0:5]) / 3.0
        exponentials = []
        for pair in expert_pairs:
            first_index = int(pair["first_cut"])
            second_index = int(pair["second_cut"])
            exponent = message_entropy
            exponent += projected.stage_upper_with_kernel(
                point[0:5],
                point[5:10],
                first_cuts[first_index],
                first_kernels[first_index],
            )
            exponent += projected.stage_upper_with_kernel(
                point[5:10],
                point[10:15],
                second_cuts[second_index],
                second_kernels[second_index],
            )
            exponentials.append(np.exp(exponent))
        mixture = float(np.dot(gating_weights, exponentials))
        return float(np.log(max(mixture, 1e-300))), gating_weights

    random = np.random.default_rng(args.seed)
    starts = []
    for _ in range(max(args.restarts * 3, 64)):
        direction = random.normal(size=15)
        vertex = linprog(
            direction,
            A_ub=domain_upper,
            b_ub=domain_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=list(zip(lower_array, upper_array, strict=True)),
            method="highs",
        )
        if vertex.success:
            starts.append(vertex.x)
        if len(starts) >= args.restarts:
            break
    if not starts:
        raise RuntimeError("failed to construct a feasible restart")

    best = None
    successful = 0
    for start in starts:
        result = minimize(
            lambda point: -value(point)[0],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=linear_constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        point = result.x
        equality_error = float(np.max(np.abs(equality @ point - equality_target)))
        domain_error = float(np.max(domain_upper @ point - domain_target))
        if equality_error <= 1e-7 and domain_error <= 1e-7:
            successful += 1
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("no restart returned a feasible maximizer")
    maximum, gating_weights = value(best.x)
    payload = {
        "status": "floating continuous affine-gate optimization",
        "fixed": args.fix,
                "gate": str(args.gate),
                "rational_partitions": (
                    str(args.rational_partitions)
                    if args.rational_partitions is not None
                    else None
                ),
                "rational_leaf": (
                    rational_record.get("leaf") if rational_record is not None else None
                ),
                "restarts": len(starts),
                "successful_restarts": successful,
                "maximum_log_mixture": maximum,
                "vertex_log_mixture": affine["log_maximum_vertex_mixture"],
                "point": best.x.tolist(),
                "optimizer_success": bool(best.success),
                "optimizer_message": best.message,
                "minimum_gating_weight": float(np.min(gating_weights)),
                "gating_sum_residual": float(abs(np.sum(gating_weights) - 1.0)),
                "equality_residual": float(
                    np.max(np.abs(equality @ best.x - equality_target))
                ),
                "domain_residual": float(
                    np.max(domain_upper @ best.x - domain_target)
                ),
            }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
