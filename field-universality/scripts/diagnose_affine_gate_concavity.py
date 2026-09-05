#!/usr/bin/env python3
"""Sample Hessians of a rational affine-gated exponential mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_projected_kl_box_cover as projected
from diagnose_joint_cut_concavity import add_log_bilinear_hessian, simplex_tangent_basis
from rationalize_affine_cover import collect_leaves


def simplex_tangent_basis_for(dependent: list[int]) -> tuple[np.ndarray, list[int]]:
    if (
        len(dependent) != 3
        or any(
            not start <= coordinate < start + 5
            for start, coordinate in zip((0, 5, 10), dependent, strict=True)
        )
    ):
        raise ValueError("choose one dependent coordinate from each simplex block")
    directions = [
        coordinate
        for start, eliminated in zip((0, 5, 10), dependent, strict=True)
        for coordinate in range(start, start + 5)
        if coordinate != eliminated
    ]
    basis = np.zeros((15, 12))
    for column, coordinate in enumerate(directions):
        eliminated = dependent[coordinate // 5]
        basis[coordinate, column] = 1.0
        basis[eliminated, column] = -1.0
    return basis, directions


def add_stage_gradient(
    gradient: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cut,
    kernel: np.ndarray,
    left_start: int,
    right_start: int,
) -> None:
    partition = float(left @ kernel @ right)
    gradient[left_start : left_start + 5] += kernel @ right / partition
    gradient[right_start : right_start + 5] += kernel.T @ left / partition
    gradient[right_start : right_start + 5] -= cut.target_potential
    gradient[left_start + 1 : left_start + 5] -= cut.increment_support_potential
    gradient[right_start + 1 : right_start + 5] -= cut.state_support_potential


def expert_log_derivatives(point, first_cut, second_cut, first_kernel, second_kernel):
    value = projected.bulk_entropy(point[0:5]) / 3.0
    value += projected.stage_upper_with_kernel(
        point[0:5], point[5:10], first_cut, first_kernel
    )
    value += projected.stage_upper_with_kernel(
        point[5:10], point[10:15], second_cut, second_kernel
    )
    gradient = np.zeros(15)
    gradient[0:5] = -(np.log(point[0:5]) + 1.0) / 3.0
    add_stage_gradient(
        gradient, point[0:5], point[5:10], first_cut, first_kernel, 0, 5
    )
    add_stage_gradient(
        gradient, point[5:10], point[10:15], second_cut, second_kernel, 5, 10
    )
    hessian = np.zeros((15, 15))
    hessian[0:5, 0:5] -= np.diag(1.0 / (3.0 * point[0:5]))
    add_log_bilinear_hessian(hessian, point[0:5], point[5:10], first_kernel, 0, 5)
    add_log_bilinear_hessian(hessian, point[5:10], point[10:15], second_kernel, 5, 10)
    return float(value), gradient, hessian


def mixture_log_derivatives(
    point: np.ndarray,
    coefficients: np.ndarray,
    coordinates: list[int],
    pairs: list[dict],
    first_cuts,
    second_cuts,
    first_kernels,
    second_kernels,
    dependent: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    basis = (
        simplex_tangent_basis()
        if dependent is None
        else simplex_tangent_basis_for(dependent)[0]
    )
    features = np.concatenate(([1.0], point[coordinates]))
    weights = coefficients @ features
    weight_gradients = np.zeros((len(weights), 15))
    for feature, coordinate in enumerate(coordinates, start=1):
        weight_gradients[:, coordinate] = coefficients[:, feature]
    values = []
    gradients = []
    hessians = []
    for pair in pairs:
        first = int(pair["first_cut"])
        second = int(pair["second_cut"])
        value, gradient, hessian = expert_log_derivatives(
            point,
            first_cuts[first],
            second_cuts[second],
            first_kernels[first],
            second_kernels[second],
        )
        values.append(value)
        gradients.append(gradient)
        hessians.append(hessian)
    values = np.asarray(values)
    gradients = np.asarray(gradients)
    hessians = np.asarray(hessians)
    exponentials = np.exp(values - np.max(values))
    terms = weights * exponentials
    mixture = float(np.sum(terms))
    mixture_gradient = np.sum(
        exponentials[:, None]
        * (weight_gradients + weights[:, None] * gradients),
        axis=0,
    )
    mixture_hessian_ambient = np.zeros((15, 15))
    for expert in range(len(weights)):
        gradient = gradients[expert]
        weight_gradient = weight_gradients[expert]
        mixture_hessian_ambient += exponentials[expert] * (
            weights[expert]
            * (hessians[expert] + np.outer(gradient, gradient))
            + np.outer(weight_gradient, gradient)
            + np.outer(gradient, weight_gradient)
        )
    log_hessian = (
        mixture_hessian_ambient / mixture
        - np.outer(mixture_gradient, mixture_gradient) / mixture**2
    )
    log_gradient = mixture_gradient / mixture
    return basis.T @ log_gradient, basis.T @ log_hessian @ basis


def mixture_hessian(
    point: np.ndarray,
    coefficients: np.ndarray,
    coordinates: list[int],
    pairs: list[dict],
    first_cuts,
    second_cuts,
    first_kernels,
    second_kernels,
    dependent: list[int] | None = None,
) -> np.ndarray:
    return mixture_log_derivatives(
        point,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        first_kernels,
        second_kernels,
        dependent,
    )[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="fix a coordinate to its chamber lower or upper endpoint",
    )
    parser.add_argument(
        "--filter-domain",
        action="store_true",
        help="reject samples that violate the two support constraints or canonical message order",
    )
    parser.add_argument(
        "--dependent-coordinates",
        default="0,5,10",
        help="comma-separated simplex coordinates eliminated in the tangent basis",
    )
    parser.add_argument(
        "--bank", type=Path, default=Path("results/f5-projected-kl-dual-bank-0.22.json")
    )
    parser.add_argument(
        "--frontier", type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    dependent = [int(value) for value in args.dependent_coordinates.split(",")]
    try:
        _, reported_directions = simplex_tangent_basis_for(dependent)
    except ValueError as error:
        parser.error(str(error))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    if args.leaf not in leaves:
        parser.error("unknown manifest leaf")
    leaf = leaves[args.leaf]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    matches = [record for record in partitions["leaves"] if record["leaf"] == args.leaf]
    if len(matches) != 1:
        parser.error("partition selection did not identify exactly one leaf")
    record = matches[0]
    denominator = float(int(record["coefficient_denominator"]))
    coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
    coordinates = [int(value) for value in record["feature_coordinates"]]

    bank_args = SimpleNamespace(
        eta=0.002, delta=0.22, bank_iterations=20, bank_restarts=12,
        seed=11, bank_cache=str(args.bank), rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    lower = list(selected["lower"])
    upper = list(selected["upper"])
    gate = json.loads(
        Path(str(record.get("gate", leaf["gate"]))).read_text(encoding="utf-8")
    )
    for restriction in gate["restrictions"]:
        dimension_text, side, threshold_text = restriction.split(":")
        dimension = int(dimension_text)
        threshold = float(threshold_text)
        if side == "le":
            upper[dimension] = min(upper[dimension], threshold)
        else:
            lower[dimension] = max(lower[dimension], threshold)
    for specification in args.fix:
        dimension_text, side = specification.split(":")
        dimension = int(dimension_text)
        if not 0 <= dimension < 15:
            parser.error("--fix coordinate must be between 0 and 14")
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            parser.error("--fix side must be lower or upper")
    box = projected.Box(tuple(lower), tuple(upper))
    block_vertices = [projected.simplex_box_vertices(box, start) for start in (0, 5, 10)]
    random = np.random.default_rng(args.seed)
    maximum = float("-inf")
    minimum = float("inf")
    positive = 0
    worst_point = None
    block_minimum_eigenvalues = np.full(3, np.inf)
    block_maximum_eigenvalues = np.full(3, -np.inf)
    leading_group_minimum = np.inf
    leading_group_maximum = -np.inf
    direction_minimum_curvatures = np.full(12, np.inf)
    direction_maximum_curvatures = np.full(12, -np.inf)
    sample_count = 0
    trials = 0
    while sample_count < args.samples and trials < 100 * args.samples:
        # The loop terminates by the explicit accepted-sample count below.  The
        # trial cap prevents a malformed or nearly empty filtered domain from
        # spinning forever.
        trials += 1
        point = np.concatenate(
            [random.dirichlet(np.ones(len(vertices))) @ vertices for vertices in block_vertices]
        )
        if args.filter_domain:
            if np.sum(point[1:5]) > 2.0 * np.sum(point[6:10]) + 1e-12:
                continue
            if np.sum(point[6:10]) > 2.0 * np.sum(point[11:15]) + 1e-12:
                continue
            if np.max(point[2:5] - point[1]) > 1e-12:
                continue
        hessian = mixture_hessian(
            point, coefficients, coordinates, record["expert_pairs"],
            first_cuts, second_cuts, first_kernels, second_kernels,
            dependent,
        )
        eigenvalues = np.linalg.eigvalsh(hessian)
        direction_minimum_curvatures = np.minimum(
            direction_minimum_curvatures, np.diag(hessian)
        )
        direction_maximum_curvatures = np.maximum(
            direction_maximum_curvatures, np.diag(hessian)
        )
        leading_eigenvalues = np.linalg.eigvalsh(hessian[0:8, 0:8])
        leading_group_minimum = min(leading_group_minimum, leading_eigenvalues[0])
        leading_group_maximum = max(leading_group_maximum, leading_eigenvalues[-1])
        for block in range(3):
            block_eigenvalues = np.linalg.eigvalsh(
                hessian[4 * block : 4 * (block + 1), 4 * block : 4 * (block + 1)]
            )
            block_minimum_eigenvalues[block] = min(
                block_minimum_eigenvalues[block], block_eigenvalues[0]
            )
            block_maximum_eigenvalues[block] = max(
                block_maximum_eigenvalues[block], block_eigenvalues[-1]
            )
        value = float(eigenvalues[-1])
        minimum = min(minimum, float(eigenvalues[0]))
        if value > maximum:
            maximum = value
            worst_point = point.tolist()
        if value > 1e-9:
            positive += 1
        sample_count += 1
        if sample_count >= args.samples:
            break
    if sample_count != args.samples:
        raise RuntimeError(
            f"accepted only {sample_count} of {args.samples} requested samples after {trials} trials"
        )
    payload = {
        "status": "floating sampled Hessian diagnostic; not a concavity proof",
        "leaf": args.leaf,
        "samples": args.samples,
        "sampling_trials": trials,
        "fixed": args.fix,
        "dependent_coordinates": dependent,
        "filtered_to_domain": args.filter_domain,
        "positive_maximum_eigenvalue_samples": positive,
        "maximum_hessian_eigenvalue": maximum,
        "minimum_hessian_eigenvalue": minimum,
        "block_hessian_eigenvalue_ranges": [
            {
                "block": block,
                "minimum_eigenvalue": float(block_minimum_eigenvalues[block]),
                "maximum_eigenvalue": float(block_maximum_eigenvalues[block]),
            }
            for block in range(3)
        ],
        "message_middle_hessian_eigenvalue_range": {
            "minimum_eigenvalue": float(leading_group_minimum),
            "maximum_eigenvalue": float(leading_group_maximum),
        },
        "coordinate_direction_curvature_ranges": [
            {
                "coordinate": coordinate,
                "minimum_second_derivative": float(minimum_value),
                "maximum_second_derivative": float(maximum_value),
            }
            for coordinate, minimum_value, maximum_value in zip(
                reported_directions,
                direction_minimum_curvatures,
                direction_maximum_curvatures,
                strict=True,
            )
        ],
        "worst_point": worst_point,
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
