#!/usr/bin/env python3
"""Refine one projected frontier box with crossings and a residual slab.

This is a floating diagnostic, not an exact certificate.  The selected
projected-KL frontier bound is a valid cap for the joint objective.  Each
retained box is enlarged only by necessary interval consequences of the
requested raw-residual slab, so its joint-projection upper bound remains
sound apart from floating-point roundoff.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_joint_projection_box_cover as joint


@dataclass(frozen=True)
class ResidualSlab:
    stage: int
    coordinate: int
    sign: int
    threshold: float


def parse_slab(text: str, threshold: float) -> ResidualSlab:
    try:
        stage_text, coordinate_text, sign_text = text.split(":")
        stage = int(stage_text) - 1
        coordinate = int(coordinate_text)
        sign = {"+": 1, "-": -1}[sign_text]
    except (ValueError, KeyError) as error:
        raise argparse.ArgumentTypeError(
            "slab must have the form STAGE:COORDINATE:SIGN"
        ) from error
    if stage not in (0, 1) or coordinate not in range(5):
        raise argparse.ArgumentTypeError(
            "stage must be 1 or 2 and coordinate must be in 0,...,4"
        )
    return ResidualSlab(stage, coordinate, sign, threshold)


def projected_support_bounds(
    lower: list[float], upper: list[float], start: int
) -> tuple[float, float]:
    return (
        max(0.0, 1.0 - upper[start], sum(lower[start + 1 : start + 5])),
        min(1.0, 1.0 - lower[start], sum(upper[start + 1 : start + 5])),
    )


def crossing_bounds(
    increment: tuple[float, float], state: tuple[float, float]
) -> tuple[float, float]:
    a_lower, a_upper = increment
    b_lower, b_upper = state
    return (
        max(0.0, a_lower - b_upper),
        min(a_upper / 2.0, 1.0 - b_lower),
    )


def frontier_box(entry: dict[str, object]) -> joint.Box:
    lower = [float(value) for value in entry["lower"]]
    upper = [float(value) for value in entry["upper"]]
    supports = [
        projected_support_bounds(lower, upper, start) for start in (0, 5, 10)
    ]
    first_crossing = crossing_bounds(supports[0], supports[1])
    second_crossing = crossing_bounds(supports[1], supports[2])
    lower.extend((first_crossing[0], second_crossing[0]))
    upper.extend((first_crossing[1], second_crossing[1]))
    return joint.Box(
        tuple(lower),
        tuple(upper),
        tuple(value[0] for value in supports),
        tuple(value[1] for value in supports),
    )


def tighten(raw: joint.Box, slab: ResidualSlab) -> joint.Box | None:
    lower = list(raw.lower)
    upper = list(raw.upper)
    support_lower = list(raw.support_lower)
    support_upper = list(raw.support_upper)
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if slab.stage == 0 else (5, 10, 16)
    )
    support_index = slab.stage
    state_support_index = slab.stage + 1

    for _ in range(12):
        before = (tuple(lower), tuple(upper), tuple(support_lower), tuple(support_upper))
        candidate = joint.Box(
            tuple(lower), tuple(upper), tuple(support_lower), tuple(support_upper)
        )
        for component, start in enumerate((0, 5, 10)):
            simple_lower, simple_upper = joint.simplex_intervals(candidate, start)
            if float(np.sum(simple_lower)) > 1.0 + 1e-12:
                return None
            if float(np.sum(simple_upper)) < 1.0 - 1e-12:
                return None
            lower[start : start + 5] = np.maximum(
                np.asarray(lower[start : start + 5]), simple_lower
            ).tolist()
            upper[start : start + 5] = np.minimum(
                np.asarray(upper[start : start + 5]), simple_upper
            ).tolist()
            support_lower[component] = max(
                support_lower[component],
                1.0 - upper[start],
                sum(lower[start + 1 : start + 5]),
            )
            support_upper[component] = min(
                support_upper[component],
                1.0 - lower[start],
                sum(upper[start + 1 : start + 5]),
            )

        a_lower = support_lower[support_index]
        a_upper = support_upper[support_index]
        b_lower = support_lower[state_support_index]
        b_upper = support_upper[state_support_index]
        lower[crossing_index] = max(lower[crossing_index], 0.0, a_lower - b_upper)
        upper[crossing_index] = min(
            upper[crossing_index], a_upper / 2.0, 1.0 - b_lower
        )

        rho = slab.threshold
        if slab.coordinate == 0:
            if slab.sign > 0:
                # x - A/5 >= rho, with A = 1 - increment[0].
                lower[crossing_index] = max(
                    lower[crossing_index], a_lower / 5.0 + rho
                )
                lower[increment_start] = max(
                    lower[increment_start],
                    1.0 + 5.0 * rho - 5.0 * upper[crossing_index],
                )
            else:
                # x - A/5 <= -rho.
                upper[crossing_index] = min(
                    upper[crossing_index], a_upper / 5.0 - rho
                )
                upper[increment_start] = min(
                    upper[increment_start],
                    1.0 - 5.0 * rho - 5.0 * lower[crossing_index],
                )
        else:
            state_coordinate = state_start + slab.coordinate
            denominator_upper = a_upper - lower[crossing_index]
            denominator_lower = max(a_lower / 2.0, a_lower - upper[crossing_index])
            if slab.sign > 0:
                if denominator_upper <= 0.0:
                    return None
                necessary_lower = (
                    b_lower * (a_lower / 5.0 + rho) / denominator_upper
                )
                lower[state_coordinate] = max(
                    lower[state_coordinate], necessary_lower
                )
            else:
                numerator_upper = a_upper / 5.0 - rho
                if numerator_upper < 0.0:
                    return None
                if denominator_lower > 0.0:
                    necessary_upper = b_upper * numerator_upper / denominator_lower
                    upper[state_coordinate] = min(
                        upper[state_coordinate], necessary_upper
                    )

        if any(left > right + 1e-14 for left, right in zip(lower, upper, strict=True)):
            return None
        if any(
            left > right + 1e-14
            for left, right in zip(support_lower, support_upper, strict=True)
        ):
            return None
        after = (tuple(lower), tuple(upper), tuple(support_lower), tuple(support_upper))
        if after == before:
            break

    return joint.Box(
        tuple(lower), tuple(upper), tuple(support_lower), tuple(support_upper)
    )


def residual_interval(box: joint.Box, slab: ResidualSlab) -> tuple[float, float]:
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if slab.stage == 0 else (5, 10, 16)
    )
    a_lower, a_upper = joint.support_interval(box, increment_start)
    if slab.coordinate == 0:
        return (
            box.lower[crossing_index] - a_upper / 5.0,
            box.upper[crossing_index] - a_lower / 5.0,
        )
    b_lower, b_upper = joint.support_interval(box, state_start)
    beta_lower, beta_upper = joint.simplex_intervals(box, state_start)
    coordinate_lower = beta_lower[slab.coordinate]
    coordinate_upper = beta_upper[slab.coordinate]
    both_lower = max(a_lower / 2.0, a_lower - box.upper[crossing_index])
    both_upper = a_upper - box.lower[crossing_index]
    lower = -a_upper / 5.0
    upper = -a_lower / 5.0
    if b_upper > 0.0:
        lower += both_lower * coordinate_lower / b_upper
    if b_lower > 0.0:
        upper += both_upper * coordinate_upper / b_lower
    elif both_upper * coordinate_upper > 0.0:
        upper = float("inf")
    return lower, upper


def split_dimension(
    box: joint.Box, diagnostic: dict[str, object], slab: ResidualSlab, root: joint.Box
) -> int:
    increment_start, state_start, crossing_index = (
        (0, 5, 15) if slab.stage == 0 else (5, 10, 16)
    )
    candidates = [
        increment_start,
        state_start,
        crossing_index,
        joint.VARIABLES + slab.stage,
        joint.VARIABLES + slab.stage + 1,
    ]
    if slab.coordinate > 0:
        candidates.append(state_start + slab.coordinate)
    widths = []
    for dimension in candidates:
        if dimension < joint.VARIABLES:
            width = box.upper[dimension] - box.lower[dimension]
            scale = max(root.upper[dimension] - root.lower[dimension], 1e-12)
        else:
            index = dimension - joint.VARIABLES
            width = box.support_upper[index] - box.support_lower[index]
            scale = max(
                root.support_upper[index] - root.support_lower[index], 1e-12
            )
        widths.append(width / scale)
    if max(widths, default=0.0) > 0.08:
        return candidates[int(np.argmax(widths))]
    return int(diagnostic["split_dimension"])


def run(args: argparse.Namespace) -> dict[str, object]:
    with args.frontier.open("r", encoding="utf-8") as source:
        frontier = json.load(source)
    entries = sorted(frontier["queue"], key=lambda entry: entry["priority"])
    selected = entries[args.frontier_index]["evaluated"]
    slab = parse_slab(args.slab, args.threshold)
    root = tighten(frontier_box(selected), slab)
    if root is None:
        return {"status": "slab is infeasible on the selected frontier box"}

    evaluator_args = SimpleNamespace(
        eta=args.eta,
        delta=args.delta,
        vertices=args.vertices,
        cut_orbits=args.cut_orbits,
        cut_depth=args.cut_depth,
    )
    evaluated = joint.evaluate(root, evaluator_args, args.seed)
    if evaluated is None:
        return {"status": "slab relaxation is infeasible"}
    diagnostic, first, second = evaluated
    diagnostic["upper"] = min(float(selected["bound"]), float(diagnostic["upper"]))
    queue = [(-float(diagnostic["upper"]), 0, root, diagnostic, first, second)]
    history = [float(diagnostic["upper"])]
    identifier = 1

    for index in range(args.max_splits):
        _, _, box, parent, first, second = heapq.heappop(queue)
        dimension = split_dimension(box, parent, slab, root)
        for child_index, raw_child in enumerate(joint.split(box, dimension)):
            child = tighten(raw_child, slab)
            if child is None:
                continue
            child_result = joint.evaluate(
                child,
                evaluator_args,
                args.seed + 2 * index + child_index + 1,
                first,
                second,
            )
            if child_result is None:
                continue
            child_diagnostic, child_first, child_second = child_result
            child_diagnostic["upper"] = min(
                float(parent["upper"]), float(child_diagnostic["upper"])
            )
            heapq.heappush(
                queue,
                (
                    -float(child_diagnostic["upper"]),
                    identifier,
                    child,
                    child_diagnostic,
                    child_first,
                    child_second,
                ),
            )
            identifier += 1
        if not queue:
            return {
                "status": "slab is infeasible on the selected frontier box",
                "splits": index + 1,
            }
        history.append(-queue[0][0])
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {"splits": index + 1, "upper": -queue[0][0], "boxes": len(queue)}
                ),
                flush=True,
            )
        if -queue[0][0] < args.target:
            break

    worst = queue[0]
    return {
        "status": (
            "floating joint slab cover reached target"
            if -worst[0] < args.target
            else "floating joint slab cover incomplete"
        ),
        "frontier": str(args.frontier),
        "frontier_index": args.frontier_index,
        "projected_frontier_upper": float(selected["bound"]),
        "slab": {
            "stage": slab.stage + 1,
            "coordinate": slab.coordinate,
            "sign": "+" if slab.sign > 0 else "-",
            "threshold": slab.threshold,
        },
        "upper": -worst[0],
        "splits": len(history) - 1,
        "boxes": len(queue),
        "history": history,
        "worst_residual_interval": residual_interval(worst[2], slab),
        "worst_box": {
            "lower": list(worst[2].lower),
            "upper": list(worst[2].upper),
            "support_lower": list(worst[2].support_lower),
            "support_upper": list(worst[2].support_upper),
            "diagnostic": worst[3],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=166)
    parser.add_argument("--slab", default="1:0:+")
    parser.add_argument("--threshold", type=float, default=1.0 / 3200.0)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--max-splits", type=int, default=100)
    parser.add_argument("--vertices", type=int, default=8)
    parser.add_argument("--cut-depth", type=int, default=64)
    parser.add_argument(
        "--cut-orbits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
