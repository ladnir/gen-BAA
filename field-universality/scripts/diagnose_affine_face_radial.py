#!/usr/bin/env python3
"""Sample radial derivatives toward a candidate affine-face maximizer."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

import explore_projected_kl_box_cover as projected
from diagnose_affine_face_curvature import (
    TANGENT_COORDINATES,
    point_from_variables,
)
from diagnose_affine_gate_concavity import mixture_log_derivatives
from explore_affine_face_interval import apply_face
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("--face", required=True)
    parser.add_argument("--center-from", type=Path, required=True)
    parser.add_argument("--local-radius-fraction", type=float, default=0.04)
    parser.add_argument(
        "--local-box-from", type=Path,
        help="closed face result whose initial box constrains an optimized radial target",
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument(
        "--bank", type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or not 0.0 < args.local_radius_fraction < 1.0:
        parser.error("samples must be positive and local radius must lie in (0,1)")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    face_file = json.loads(args.faces.read_text(encoding="utf-8"))
    leaf_name = face_file["leaf"]
    leaf = leaves[leaf_name]
    matches = [face for face in face_file["faces"] if face["name"] == args.face]
    if len(matches) != 1:
        parser.error("face selection did not identify exactly one face")
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(record for record in partitions["leaves"] if record["leaf"] == leaf_name)
    denominator = float(int(record["coefficient_denominator"]))
    coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
    coordinates = [int(value) for value in record["feature_coordinates"]]

    frontier = json.loads((root / manifest["global_checkpoint"]).read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate = json.loads((root / str(leaf["gate"]).replace("\\", "/")).read_text(encoding="utf-8"))
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    box, variables, dependent = apply_face(lower, upper, matches[0])

    center_payload = json.loads(args.center_from.read_text(encoding="utf-8"))
    center = np.asarray(center_payload["point"], dtype=float)
    local_lower = np.asarray([float(value) for value in box.lower])
    local_upper = np.asarray([float(value) for value in box.upper])
    for coordinate in variables:
        radius = args.local_radius_fraction * float(
            box.upper[coordinate] - box.lower[coordinate]
        )
        local_lower[coordinate] = max(
            local_lower[coordinate], center[coordinate] - radius
        )
        local_upper[coordinate] = min(
            local_upper[coordinate], center[coordinate] + radius
        )

    bank_args = SimpleNamespace(
        eta=0.002, delta=0.22, bank_iterations=20, bank_restarts=12,
        seed=11, bank_cache=str(args.bank), rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    face_tangent = np.zeros((len(TANGENT_COORDINATES), len(variables)))
    for column, coordinate in enumerate(variables):
        block = coordinate // 5
        base = 5 * block
        dependent_coordinate = dependent[block]
        if coordinate != base:
            face_tangent[TANGENT_COORDINATES.index(coordinate), column] += 1.0
        if dependent_coordinate != base:
            face_tangent[TANGENT_COORDINATES.index(dependent_coordinate), column] -= 1.0

    def radial_components(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tangent_gradient, _ = mixture_log_derivatives(
            point,
            coefficients,
            coordinates,
            record["expert_pairs"],
            first_cuts,
            second_cuts,
            first_kernels,
            second_kernels,
        )
        gradient = face_tangent.T @ tangent_gradient
        displacement = np.asarray(
            [point[coordinate] - center[coordinate] for coordinate in variables]
        )
        return displacement * gradient, gradient

    random = np.random.default_rng(args.seed)
    candidates: list[np.ndarray] = []
    variable_lower = [float(box.lower[index]) for index in variables]
    variable_upper = [float(box.upper[index]) for index in variables]
    for bits in itertools.product((0, 1), repeat=len(variables)):
        values = [
            upper_value if bit else lower_value
            for bit, lower_value, upper_value in zip(
                bits, variable_lower, variable_upper, strict=True
            )
        ]
        point = point_from_variables(box, variables, dependent, values)
        if point is not None:
            candidates.append(point)
    while len(candidates) < args.samples + 32:
        point = point_from_variables(
            box,
            variables,
            dependent,
            [
                random.uniform(lower_value, upper_value)
                for lower_value, upper_value in zip(
                    variable_lower, variable_upper, strict=True
                )
            ],
        )
        if point is not None:
            candidates.append(point)

    outside = [
        point
        for point in candidates
        if any(
            point[index] < local_lower[index] or point[index] > local_upper[index]
            for index in variables
        )
    ]
    component_and_gradients = [radial_components(point) for point in outside]
    component_values = [value[0] for value in component_and_gradients]
    gradients = [value[1] for value in component_and_gradients]
    values = [float(np.sum(value)) for value in component_values]
    worst_index = int(np.argmax(values))
    optimized_target = None
    if args.local_box_from is not None:
        local_payload = json.loads(args.local_box_from.read_text(encoding="utf-8"))
        initial = local_payload["initial_box"]
        target_lower = np.asarray(
            [
                float(Fraction(initial["lower"][coordinate]))
                for coordinate in variables
            ]
        )
        target_upper = np.asarray(
            [
                float(Fraction(initial["upper"][coordinate]))
                for coordinate in variables
            ]
        )
        objective = np.zeros(len(variables) + 1)
        objective[-1] = -1.0
        constraint_rows = []
        constraint_upper = []
        for point, gradient in zip(outside, gradients, strict=True):
            row = np.zeros(len(variables) + 1)
            row[:-1] = -gradient
            row[-1] = 1.0
            displacement_origin = np.asarray(
                [point[coordinate] for coordinate in variables]
            )
            constraint_rows.append(row)
            constraint_upper.append(-float(displacement_origin @ gradient))
        for block_start in (0, 5, 10):
            block_positions = [
                position
                for position, coordinate in enumerate(variables)
                if block_start <= coordinate < block_start + 5
            ]
            if not block_positions:
                continue
            dependent_coordinate = next(
                value
                for value in dependent
                if block_start <= value < block_start + 5
            )
            fixed_sum = sum(
                center[coordinate]
                for coordinate in range(block_start, block_start + 5)
                if coordinate != dependent_coordinate
                and coordinate not in variables
            )
            upper_row = np.zeros(len(variables) + 1)
            upper_row[block_positions] = 1.0
            constraint_rows.append(upper_row)
            constraint_upper.append(
                1.0
                - fixed_sum
                - float(
                    Fraction(initial["lower"][dependent_coordinate])
                )
            )
            lower_row = -upper_row
            constraint_rows.append(lower_row)
            constraint_upper.append(
                -(
                    1.0
                    - fixed_sum
                    - float(
                        Fraction(initial["upper"][dependent_coordinate])
                    )
                )
            )
        fit = linprog(
            objective,
            A_ub=np.asarray(constraint_rows),
            b_ub=np.asarray(constraint_upper),
            bounds=[
                (float(lower), float(upper))
                for lower, upper in zip(target_lower, target_upper, strict=True)
            ]
            + [(None, None)],
            method="highs",
        )
        if not fit.success:
            raise RuntimeError(f"radial-target LP failed: {fit.message}")
        full_target = center.copy()
        for coordinate, value in zip(variables, fit.x[:-1], strict=True):
            full_target[coordinate] = value
        for block_start in (0, 5, 10):
            dependent_coordinate = next(
                value
                for value in dependent
                if block_start <= value < block_start + 5
            )
            full_target[dependent_coordinate] = 1.0 - sum(
                full_target[coordinate]
                for coordinate in range(block_start, block_start + 5)
                if coordinate != dependent_coordinate
            )
        optimized_values = [
            float(
                np.asarray(
                    [point[coordinate] - full_target[coordinate] for coordinate in variables]
                )
                @ gradient
            )
            for point, gradient in zip(outside, gradients, strict=True)
        ]
        optimized_target = {
            "local_box_source": str(args.local_box_from),
            "point": full_target.tolist(),
            "certified_sample_margin_from_lp": float(fit.x[-1]),
            "maximum_sampled_outward_derivative": max(optimized_values),
            "minimum_sampled_outward_derivative": min(optimized_values),
        }

    payload = {
        "status": "floating sampled radial-derivative diagnostic; not a proof",
        "leaf": leaf_name,
        "face": args.face,
        "variables": variables,
        "center_source": str(args.center_from),
        "local_radius_fraction": args.local_radius_fraction,
        "sampled_points_outside_local_box": len(outside),
        "positive_radial_derivative_samples": sum(value > 0.0 for value in values),
        "maximum_outward_radial_derivative": values[worst_index],
        "minimum_outward_radial_derivative": min(values),
        "maximum_point": outside[worst_index].tolist(),
        "optimized_radial_target": optimized_target,
        "coordinate_outward_derivatives": [
            {
                "coordinate": coordinate,
                "positive_samples": int(
                    sum(value[position] > 0.0 for value in component_values)
                ),
                "maximum": max(float(value[position]) for value in component_values),
                "minimum": min(float(value[position]) for value in component_values),
            }
            for position, coordinate in enumerate(variables)
        ],
        "interpretation": (
            "A proof that this outward derivative is negative outside the local "
            "box would reduce global maximality to the existing local certificate."
        ),
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
