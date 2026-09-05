#!/usr/bin/env python3
"""Cover the two-dimensional F_5 zero-payment support surface.

The structural zero set fixes x_1=A/5, x_2=B/5 and makes the middle and
output nonzero laws uniform.  Monotonicity in the output support fixes
C=delta when delta<4/5.  The remaining upper exponent is a function of
(A,B) on a triangle.  This script bounds it with entropy tangents and
secants.  The cover uses floating arithmetic and is therefore a diagnostic,
not an exact certificate.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


@dataclass(frozen=True)
class Box:
    a_lower: float
    a_upper: float
    b_lower: float
    b_upper: float
    depth: int = 0


def rational_text(value: float) -> str:
    # Every cover endpoint is obtained by bisection from eta and delta.  The
    # denominator cap recovers those small intended rationals from their
    # binary floating representations.
    rational = Fraction(float(value)).limit_denominator(1 << 40)
    return f"{rational.numerator}/{rational.denominator}"


def binary_entropy(value: float) -> float:
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return -value * math.log(value) - (1.0 - value) * math.log(1.0 - value)


def support_cells(increment: float, state: float) -> np.ndarray:
    """Return the support table at crossing increment/5."""
    crossing = increment / 5.0
    return np.array(
        [
            1.0 - state - crossing,
            state - increment + crossing,
            crossing,
            increment - crossing,
        ],
        dtype=float,
    )


def entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def support_information(increment: float, state: float) -> float:
    return (
        binary_entropy(increment)
        + binary_entropy(state)
        - entropy(support_cells(increment, state))
    )


def zero_surface_value(
    message_support: float,
    middle_support: float,
    output_support: float,
) -> float:
    message_entropy = binary_entropy(message_support) + message_support * math.log(4.0)
    return (
        message_entropy / 3.0
        - support_information(message_support, middle_support)
        - support_information(middle_support, output_support)
    )


def output_derivative(
    middle_support: float, output_support: float
) -> float:
    numerator = output_support * (
        1.0 - output_support - middle_support / 5.0
    )
    denominator = (1.0 - output_support) * (
        output_support - 4.0 * middle_support / 5.0
    )
    if denominator <= 0.0:
        return float("inf")
    return math.log(numerator / denominator)


def polygon_vertices(box: Box) -> np.ndarray:
    """Intersect a rectangle with B >= 4A/5."""
    inequalities = np.array(
        [
            [-1.0, 0.0, -box.a_lower],
            [1.0, 0.0, box.a_upper],
            [0.0, -1.0, -box.b_lower],
            [0.0, 1.0, box.b_upper],
            [4.0 / 5.0, -1.0, 0.0],
        ]
    )
    vertices: list[np.ndarray] = []
    for first, second in combinations(inequalities, 2):
        matrix = np.stack((first[:2], second[:2]))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1e-15:
            continue
        point = np.linalg.solve(matrix, np.array([first[2], second[2]]))
        if np.all(inequalities[:, :2] @ point <= inequalities[:, 2] + 1e-11):
            vertices.append(point)
    if not vertices:
        return np.empty((0, 2))
    unique = {
        tuple(float(value) for value in np.round(point, decimals=15))
        for point in vertices
    }
    return np.asarray(sorted(unique), dtype=float)


def entropy_secant(lower: float, upper: float) -> tuple[float, float]:
    """Return slope and intercept of a lower chord for binary entropy."""
    if upper - lower <= 1e-15:
        derivative = math.log((1.0 - lower) / lower)
        return derivative, binary_entropy(lower) - derivative * lower
    lower_value = binary_entropy(lower)
    upper_value = binary_entropy(upper)
    slope = (upper_value - lower_value) / (upper - lower)
    return slope, lower_value - slope * lower


def add_cross_entropy_tangent(
    constant: float,
    coefficients: np.ndarray,
    cell_affine: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, np.ndarray]:
    logs = -np.log(np.maximum(reference, 1e-300))
    constant += float(np.dot(logs, cell_affine[:, 0]))
    coefficients += logs @ cell_affine[:, 1:]
    return constant, coefficients


def box_upper(box: Box, delta: float) -> tuple[float, np.ndarray] | None:
    vertices = polygon_vertices(box)
    if len(vertices) < 3:
        return None
    reference = vertices.mean(axis=0)
    first_reference = support_cells(reference[0], reference[1])
    second_reference = support_cells(reference[1], delta)
    if min(first_reference.min(), second_reference.min()) <= 0.0:
        return None

    # F = -(2/3)h(A)+(A/3)log(4)-2h(B)-h(delta)
    #     +H(J_1)+H(J_2).
    constant = -binary_entropy(delta)
    coefficients = np.array([math.log(4.0) / 3.0, 0.0])
    for coordinate, factor, lower, upper in (
        (0, 2.0 / 3.0, box.a_lower, box.a_upper),
        (1, 2.0, box.b_lower, box.b_upper),
    ):
        slope, intercept = entropy_secant(lower, upper)
        constant -= factor * intercept
        coefficients[coordinate] -= factor * slope

    first_affine = np.array(
        [
            [1.0, -1.0 / 5.0, -1.0],
            [0.0, -4.0 / 5.0, 1.0],
            [0.0, 1.0 / 5.0, 0.0],
            [0.0, 4.0 / 5.0, 0.0],
        ]
    )
    second_affine = np.array(
        [
            [1.0 - delta, 0.0, -1.0 / 5.0],
            [delta, 0.0, -4.0 / 5.0],
            [0.0, 0.0, 1.0 / 5.0],
            [0.0, 0.0, 4.0 / 5.0],
        ]
    )
    constant, coefficients = add_cross_entropy_tangent(
        constant, coefficients, first_affine, first_reference
    )
    constant, coefficients = add_cross_entropy_tangent(
        constant, coefficients, second_affine, second_reference
    )
    values = constant + vertices @ coefficients
    active = int(np.argmax(values))
    return float(values[active]), vertices[active]


def split_box(box: Box, coordinate: int) -> tuple[Box, Box]:
    if coordinate == 0:
        midpoint = (box.a_lower + box.a_upper) / 2.0
        return (
            Box(box.a_lower, midpoint, box.b_lower, box.b_upper, box.depth + 1),
            Box(midpoint, box.a_upper, box.b_lower, box.b_upper, box.depth + 1),
        )
    midpoint = (box.b_lower + box.b_upper) / 2.0
    return (
        Box(box.a_lower, box.a_upper, box.b_lower, midpoint, box.depth + 1),
        Box(box.a_lower, box.a_upper, midpoint, box.b_upper, box.depth + 1),
    )


def evaluated_children(
    box: Box, coordinate: int, delta: float
) -> list[tuple[float, Box, np.ndarray]]:
    result = []
    for child in split_box(box, coordinate):
        evaluated = box_upper(child, delta)
        if evaluated is not None:
            result.append((evaluated[0], child, evaluated[1]))
    return result


def cover(
    eta: float,
    delta: float,
    target: float,
    maximum_boxes: int,
) -> dict[str, object]:
    if not 0.0 < eta < delta < 4.0 / 5.0:
        raise ValueError("require 0 < eta < delta < 4/5")
    # Feasibility of the second zero-payment transition gives B<=5C/4.
    # The first transition then gives A<=5B/4.
    b_upper = 5.0 * delta / 4.0
    a_upper = 5.0 * b_upper / 4.0
    root = Box(eta, a_upper, 4.0 * eta / 5.0, b_upper)
    root_evaluation = box_upper(root, delta)
    if root_evaluation is None:
        raise RuntimeError("zero-payment root is empty")

    queue: list[tuple[float, int, Box, np.ndarray]] = []
    serial = 0
    heapq.heappush(
        queue,
        (-root_evaluation[0], serial, root, root_evaluation[1]),
    )
    boxes_split = 0
    boxes_pruned = 0
    maximum_depth = 0
    closed_boxes: list[dict[str, object]] = []
    while queue and -queue[0][0] > target and boxes_split < maximum_boxes:
        _, _, active, active_vertex = heapq.heappop(queue)
        widths = np.array(
            [
                active.a_upper - active.a_lower,
                active.b_upper - active.b_lower,
            ]
        )
        # Entropy curvature scales as 1/(x(1-x)).  A one-step lookahead can
        # repeatedly split a nearly resolved coordinate for a tiny immediate
        # gain while leaving the small-support coordinate broad.  The
        # curvature-weighted width targets the source of the secant error and
        # avoids that pathology.
        curvature_scale = np.maximum(
            active_vertex * (1.0 - active_vertex), 1e-8
        )
        coordinate = int(np.argmax(widths**2 / curvature_scale))
        chosen = evaluated_children(active, coordinate, delta)
        boxes_split += 1
        for upper, child, point in chosen:
            maximum_depth = max(maximum_depth, child.depth)
            if upper <= target:
                boxes_pruned += 1
                closed_boxes.append(
                    {
                        "a_lower": child.a_lower,
                        "a_upper": child.a_upper,
                        "b_lower": child.b_lower,
                        "b_upper": child.b_upper,
                        "depth": child.depth,
                        "upper": upper,
                        "bounds_rational": [
                            rational_text(child.a_lower),
                            rational_text(child.a_upper),
                            rational_text(child.b_lower),
                            rational_text(child.b_upper),
                        ],
                    }
                )
                continue
            serial += 1
            heapq.heappush(queue, (-upper, serial, child, point))

    final_upper = (
        max(float(box["upper"]) for box in closed_boxes)
        if not queue
        else -queue[0][0]
    )
    witness = None if not queue else queue[0][3]
    return {
        "root_upper": root_evaluation[0],
        "final_upper": final_upper,
        "target": target,
        "closed": final_upper <= target,
        "boxes_split": boxes_split,
        "boxes_pruned": boxes_pruned,
        "active_boxes": len(queue),
        "maximum_depth": maximum_depth,
        "closed_boxes": closed_boxes,
        "worst_box": None
        if not queue
        else {
            "a_lower": queue[0][2].a_lower,
            "a_upper": queue[0][2].a_upper,
            "b_lower": queue[0][2].b_lower,
            "b_upper": queue[0][2].b_upper,
            "upper": -queue[0][0],
        },
        "upper_witness": None if witness is None else witness.tolist(),
    }


def floating_maximum(eta: float, delta: float, seed: int) -> dict[str, object]:
    b_upper = 5.0 * delta / 4.0
    a_upper = 5.0 * b_upper / 4.0

    def objective(point: np.ndarray) -> float:
        a, b = point
        if b < 4.0 * a / 5.0:
            return 1e3 + 4.0 * a / 5.0 - b
        return -zero_surface_value(a, b, delta)

    result = differential_evolution(
        objective,
        ((eta, a_upper), (4.0 * eta / 5.0, b_upper)),
        seed=seed,
        workers=1,
        polish=True,
        tol=1e-12,
    )
    return {
        "value": -float(result.fun),
        "message_support": float(result.x[0]),
        "middle_support": float(result.x[1]),
        "output_support": delta,
        "optimizer_success": bool(result.success),
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> dict[str, object]:
    result = {
        "prime": 5,
        "repetition": 3,
        "eta": args.eta,
        "delta": args.delta,
        "reduction": {
            "first_crossing": "A/5",
            "second_crossing": "B/5",
            "middle_nonzero_law": "uniform on F_5^*",
            "output_nonzero_law": "uniform on F_5^*",
            "output_monotonicity": "dF/dC > 0 for C < 4/5",
            "fixed_output_support": args.delta,
            "reduced_domain": [
                "eta <= A <= 25*delta/16",
                "4*A/5 <= B <= 5*delta/4",
            ],
        },
        "floating_maximum": floating_maximum(args.eta, args.delta, args.seed),
        "floating_cover": cover(
            args.eta,
            args.delta,
            args.target,
            args.maximum_boxes,
        ),
        "status": "floating entropy-tangent/secant cover; not an exact certificate",
    }
    write_json_atomic(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument("--target", type=float, default=-0.005)
    parser.add_argument("--maximum-boxes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-zero-payment-surface.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = run(arguments)
    print(json.dumps(report, indent=2, sort_keys=True))
