#!/usr/bin/env python3
"""Sample curvature and Hessian variation on one affine-gate face."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_projected_kl_box_cover as projected
from diagnose_affine_gate_concavity import mixture_hessian
from explore_affine_face_interval import apply_face
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


TANGENT_COORDINATES = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]


def point_from_variables(box, variables: list[int], dependent: list[int], values) -> np.ndarray | None:
    point = np.asarray([float(value) for value in box.lower], dtype=float)
    for coordinate, value in zip(variables, values, strict=True):
        point[coordinate] = value
    for start in (0, 5, 10):
        coordinate = next(value for value in dependent if start <= value < start + 5)
        point[coordinate] = 1.0 - sum(
            point[other] for other in range(start, start + 5) if other != coordinate
        )
    lower = np.asarray([float(value) for value in box.lower])
    upper = np.asarray([float(value) for value in box.upper])
    if np.any(point < lower - 1e-13) or np.any(point > upper + 1e-13):
        return None
    return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("--face", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--difference-step", type=float, default=1e-6)
    parser.add_argument(
        "--center-from",
        type=Path,
        help="sample a local box centered at the point in this optimizer result",
    )
    parser.add_argument(
        "--local-radius-fraction",
        type=float,
        default=1.0,
        help="local radius as a fraction of each full face-coordinate width",
    )
    parser.add_argument(
        "--bank", type=Path, default=Path("results/f5-projected-kl-dual-bank-0.22.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.samples <= 0
        or args.difference_step <= 0.0
        or args.local_radius_fraction <= 0.0
    ):
        parser.error("sample count, difference step, and radius fraction must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    face_file = json.loads(args.faces.read_text(encoding="utf-8"))
    leaf_name = face_file["leaf"]
    face = next(face for face in face_file["faces"] if face["name"] == args.face)
    leaf = leaves[leaf_name]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(record for record in partitions["leaves"] if record["leaf"] == leaf_name)
    denominator = float(int(record["coefficient_denominator"]))
    coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
    coordinates = [int(value) for value in record["feature_coordinates"]]

    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    frontier = json.loads((root / manifest["global_checkpoint"]).read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate = json.loads((root / str(leaf["gate"]).replace("\\", "/")).read_text(encoding="utf-8"))
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    box, variables, dependent = apply_face(lower, upper, face)

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
            face_tangent[
                TANGENT_COORDINATES.index(dependent_coordinate), column
            ] -= 1.0

    sampling_lower = {
        coordinate: float(box.lower[coordinate]) for coordinate in variables
    }
    sampling_upper = {
        coordinate: float(box.upper[coordinate]) for coordinate in variables
    }
    center_point = None
    if args.center_from is not None:
        center_payload = json.loads(args.center_from.read_text(encoding="utf-8"))
        center_point = np.asarray(center_payload["point"], dtype=float)
        for coordinate in variables:
            width = float(box.upper[coordinate] - box.lower[coordinate])
            radius = args.local_radius_fraction * width
            sampling_lower[coordinate] = max(
                sampling_lower[coordinate], center_point[coordinate] - radius
            )
            sampling_upper[coordinate] = min(
                sampling_upper[coordinate], center_point[coordinate] + radius
            )

    random = np.random.default_rng(args.seed)
    candidates = []
    for bits in itertools.product((0, 1), repeat=len(variables)):
        values = [
            sampling_upper[coordinate] if bit else sampling_lower[coordinate]
            for coordinate, bit in zip(variables, bits, strict=True)
        ]
        point = point_from_variables(box, variables, dependent, values)
        if point is not None:
            candidates.append(point)
    if center_point is not None:
        candidates.append(center_point)
    while len(candidates) < args.samples + 32:
        values = [
            random.uniform(sampling_lower[coordinate], sampling_upper[coordinate])
            for coordinate in variables
        ]
        point = point_from_variables(box, variables, dependent, values)
        if point is not None:
            candidates.append(point)
    candidates = candidates[: args.samples + 32]

    maximum_eigenvalue = -np.inf
    minimum_eigenvalue = np.inf
    positive = 0
    maximum_point = None
    derivative_norms = np.zeros(len(variables))
    derivative_samples = np.zeros(len(variables), dtype=int)
    for point in candidates:
        full = mixture_hessian(
            point, coefficients, coordinates, record["expert_pairs"],
            first_cuts, second_cuts, first_kernels, second_kernels,
        )
        hessian = face_tangent.T @ full @ face_tangent
        eigenvalues = np.linalg.eigvalsh(hessian)
        if eigenvalues[-1] > maximum_eigenvalue:
            maximum_eigenvalue = float(eigenvalues[-1])
            maximum_point = point.tolist()
        minimum_eigenvalue = min(minimum_eigenvalue, float(eigenvalues[0]))
        positive += int(eigenvalues[-1] > 0.0)
        for index, coordinate in enumerate(variables):
            block = coordinate // 5
            dependent_coordinate = dependent[block]
            slack = min(
                point[coordinate] - float(box.lower[coordinate]),
                float(box.upper[coordinate]) - point[coordinate],
                point[dependent_coordinate] - float(box.lower[dependent_coordinate]),
                float(box.upper[dependent_coordinate]) - point[dependent_coordinate],
            )
            step = min(args.difference_step, slack / 2.0)
            if step <= 1e-12:
                continue
            plus = point.copy()
            minus = point.copy()
            plus[coordinate] += step
            plus[dependent_coordinate] -= step
            minus[coordinate] -= step
            minus[dependent_coordinate] += step
            plus_full = mixture_hessian(
                plus, coefficients, coordinates, record["expert_pairs"],
                first_cuts, second_cuts, first_kernels, second_kernels,
            )
            minus_full = mixture_hessian(
                minus, coefficients, coordinates, record["expert_pairs"],
                first_cuts, second_cuts, first_kernels, second_kernels,
            )
            plus_hessian = face_tangent.T @ plus_full @ face_tangent
            minus_hessian = face_tangent.T @ minus_full @ face_tangent
            derivative = (plus_hessian - minus_hessian) / (2.0 * step)
            derivative_norms[index] = max(
                derivative_norms[index], float(np.linalg.norm(derivative, ord=2))
            )
            derivative_samples[index] += 1

    payload = {
        "status": "floating sampled face-curvature diagnostic; not a proof",
        "leaf": leaf_name,
        "face": args.face,
        "variables": variables,
        "center_source": str(args.center_from) if args.center_from is not None else None,
        "local_radius_fraction": args.local_radius_fraction,
        "sampled_points": len(candidates),
        "positive_maximum_eigenvalue_samples": positive,
        "minimum_hessian_eigenvalue": minimum_eigenvalue,
        "maximum_hessian_eigenvalue": maximum_eigenvalue,
        "maximum_eigenvalue_point": maximum_point,
        "coordinate_hessian_derivative_norm_estimates": [
            {
                "coordinate": coordinate,
                "samples": int(count),
                "maximum_spectral_norm": float(value),
            }
            for coordinate, count, value in zip(
                variables, derivative_samples, derivative_norms, strict=True
            )
        ],
        "sum_radius_times_derivative_norm_estimate": float(
            sum(
                (sampling_upper[coordinate] - sampling_lower[coordinate])
                * derivative_norms[index]
                / 2.0
                for index, coordinate in enumerate(variables)
            )
        ),
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
