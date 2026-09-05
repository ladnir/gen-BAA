#!/usr/bin/env python3
"""Constraint-generate an affine exponential gate over a continuous chamber."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import explore_projected_kl_box_cover as projected
from optimize_affine_exponential_gate import domain_matrices


def entropy_rows(distributions: np.ndarray) -> np.ndarray:
    positive = distributions > 0.0
    terms = np.zeros_like(distributions)
    terms[positive] = -distributions[positive] * np.log(distributions[positive])
    return np.sum(terms, axis=1)


def stage_upper_rows(
    increments: np.ndarray,
    states: np.ndarray,
    cuts: list,
    kernels: np.ndarray,
) -> np.ndarray:
    """Evaluate one projected-stage upper bound per row and expert."""

    partitions = np.einsum(
        "ni,eij,nj->ne", increments, kernels, states, optimize=True
    )
    target_potentials = np.stack([cut.target_potential for cut in cuts])
    increment_potentials = np.asarray(
        [cut.increment_support_potential for cut in cuts]
    )
    state_potentials = np.asarray(
        [cut.state_support_potential for cut in cuts]
    )
    return (
        np.log(partitions)
        - states @ target_potentials.T
        - increments[:, 1:].sum(axis=1, keepdims=True)
        * increment_potentials[None, :]
        - states[:, 1:].sum(axis=1, keepdims=True)
        * state_potentials[None, :]
    )


def message_directional_exponential_derivatives(
    points: np.ndarray,
    exponentials: np.ndarray,
    cuts: list,
    kernels: np.ndarray,
    coordinate: int,
    dependent: int = 0,
) -> np.ndarray:
    """Differentiate expert exponentials along e_dependent - e_coordinate."""

    if not (0 <= dependent < 5 and 0 <= coordinate < 5 and dependent != coordinate):
        raise ValueError("message direction must use distinct coordinates in [0, 5)")
    partitions = np.einsum(
        "ni,eij,nj->ne",
        points[:, 0:5],
        kernels,
        points[:, 5:10],
        optimize=True,
    )
    partition_gradients = np.einsum(
        "eij,nj->nei", kernels, points[:, 5:10], optimize=True
    )
    partition_gradients /= partitions[:, :, None]
    increment_support = np.asarray(
        [cut.increment_support_potential for cut in cuts]
    )
    log_derivative = (
        partition_gradients[:, :, dependent]
        - partition_gradients[:, :, coordinate]
    )
    if dependent:
        log_derivative -= increment_support[None, :]
    if coordinate:
        log_derivative += increment_support[None, :]
    log_derivative += (
        np.log(points[:, coordinate] / points[:, dependent])[:, None] / 3.0
    )
    return exponentials * log_derivative


def affine_feature_direction(
    feature_coordinates: list[int], coordinate: int, dependent: int = 0
) -> np.ndarray:
    """Derivative of [1, selected coordinates] along e_dependent-e_coordinate."""

    result = np.zeros(len(feature_coordinates) + 1)
    for position, feature in enumerate(feature_coordinates, start=1):
        if feature == dependent:
            result[position] += 1.0
        if feature == coordinate:
            result[position] -= 1.0
    return result


def distinct_points(
    points: list[np.ndarray], tolerance: float = 1e-9
) -> list[np.ndarray]:
    """Retain the first representative of each coordinatewise point cluster."""

    result: list[np.ndarray] = []
    for point in points:
        if all(np.max(np.abs(point - old)) > tolerance for old in result):
            result.append(point)
    return result


def point_in_box(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Return whether a saved feasible point belongs to a child box."""

    return bool(
        np.all(point >= lower - tolerance) and np.all(point <= upper + tolerance)
    )


def initial_expert_pairs(
    affine: dict,
    saved_state: dict | None,
    *,
    seed_points_only: bool,
) -> list[dict]:
    """Choose the initial expert bank for a fresh, seeded, or resumed run."""

    if saved_state is not None and not seed_points_only:
        return list(saved_state.get("expert_pairs", affine["expert_pairs"]))
    return list(affine["expert_pairs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
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
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--points-per-iteration", type=int, default=4)
    parser.add_argument("--generate-experts", action="store_true")
    parser.add_argument(
        "--constant-gate",
        action="store_true",
        help="fit one constant convex weight per expert instead of affine weights",
    )
    parser.add_argument(
        "--gate-features",
        help="comma-separated affine gate coordinates, overriding the saved feature list",
    )
    parser.add_argument(
        "--enforce-message-direction",
        type=int,
        choices=range(1, 5),
        help="require a nonnegative derivative along q0 minus this message coordinate",
    )
    parser.add_argument(
        "--derivative-margin",
        type=float,
        default=0.0,
        help="strict sampled lower margin for --enforce-message-direction",
    )
    parser.add_argument(
        "--gate-weight-margin",
        type=float,
        default=0.0,
        help="minimum fitted affine weight on every product-simplex vertex",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="fix a coordinate to its chamber lower or upper endpoint",
    )
    parser.add_argument(
        "--historical-restarts",
        type=int,
        default=16,
        help="maximum saved interior maximizers retained as extra restarts",
    )
    parser.add_argument(
        "--target-margin",
        type=float,
        default=0.0,
        help="require continuous log upper at most the negative of this value",
    )
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--resume",
        type=Path,
        help="resume expert and separation state from an earlier output",
    )
    state_group.add_argument(
        "--seed-from",
        type=Path,
        help=(
            "seed experts and in-domain training points from a parent chamber; "
            "prior history is not counted as evidence for the child"
        ),
    )
    parser.add_argument(
        "--seed-points-only",
        action="store_true",
        help=(
            "with --seed-from, inherit in-domain training points but restart "
            "from the child gate's local expert bank"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()
    if args.historical_restarts < 0:
        parser.error("--historical-restarts must be nonnegative")
    if args.target_margin < 0.0:
        parser.error("--target-margin must be nonnegative")
    if args.derivative_margin < 0.0:
        parser.error("--derivative-margin must be nonnegative")
    if args.gate_weight_margin < 0.0:
        parser.error("--gate-weight-margin must be nonnegative")
    if args.seed_points_only and args.seed_from is None:
        parser.error("--seed-points-only requires --seed-from")
    if args.constant_gate and args.gate_features is not None:
        parser.error("--constant-gate and --gate-features are mutually exclusive")

    saved_gate = json.loads(args.gate.read_text(encoding="utf-8"))
    affine = saved_gate["affine_gating"]
    state_path = args.resume if args.resume is not None else args.seed_from
    resumed = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path is not None
        else None
    )
    expert_pairs = initial_expert_pairs(
        affine,
        resumed,
        seed_points_only=args.seed_points_only,
    )
    if args.constant_gate:
        feature_coordinates = []
    elif args.gate_features is not None:
        feature_coordinates = [
            int(value) for value in args.gate_features.split(",") if value
        ]
        if len(set(feature_coordinates)) != len(feature_coordinates) or any(
            not 0 <= value < 15 for value in feature_coordinates
        ):
            parser.error("--gate-features must be distinct coordinates in [0, 15)")
    else:
        feature_coordinates = [int(value) for value in affine["feature_coordinates"]]
    direction_features = (
        affine_feature_direction(
            feature_coordinates, args.enforce_message_direction
        )
        if args.enforce_message_direction is not None
        else None
    )
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
    candidate_first_indices = [int(value) for value in saved_gate["first_candidate_cuts"]]
    candidate_second_indices = [
        int(value) for value in saved_gate["second_candidate_cuts"]
    ]
    candidate_first_cuts = [first_cuts[index] for index in candidate_first_indices]
    candidate_second_cuts = [
        second_cuts[index] for index in candidate_second_indices
    ]
    candidate_first_kernels = np.stack(
        [first_kernels[index] for index in candidate_first_indices]
    )
    candidate_second_kernels = np.stack(
        [second_kernels[index] for index in candidate_second_indices]
    )

    def prepare_experts() -> tuple[list, list, np.ndarray, np.ndarray]:
        return (
            [first_cuts[int(pair["first_cut"])] for pair in expert_pairs],
            [second_cuts[int(pair["second_cut"])] for pair in expert_pairs],
            np.stack(
                [first_kernels[int(pair["first_cut"])] for pair in expert_pairs]
            ),
            np.stack(
                [second_kernels[int(pair["second_cut"])] for pair in expert_pairs]
            ),
        )

    (
        expert_first_cuts,
        expert_second_cuts,
        expert_first_kernels,
        expert_second_kernels,
    ) = prepare_experts()
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
        if not 0 <= dimension < 15:
            parser.error("--fix coordinate must be between 0 and 14")
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            parser.error("--fix side must be lower or upper")
    box = projected.Box(tuple(lower), tuple(upper))
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    block_vertices = [
        projected.simplex_box_vertices(box, start) for start in (0, 5, 10)
    ]
    counts = [len(value) for value in block_vertices]
    message = np.repeat(block_vertices[0], counts[1] * counts[2], axis=0)
    middle = np.tile(
        np.repeat(block_vertices[1], counts[2], axis=0), (counts[0], 1)
    )
    output = np.tile(block_vertices[2], (counts[0] * counts[1], 1))
    product_vertices = np.concatenate((message, middle, output), axis=1)

    def features(points: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                np.ones((len(points), 1)),
                points[:, feature_coordinates],
            ),
            axis=1,
        )

    def expert_exponentials(points: np.ndarray) -> np.ndarray:
        entropies = entropy_rows(points[:, 0:5]) / 3.0
        first_values = stage_upper_rows(
            points[:, 0:5],
            points[:, 5:10],
            expert_first_cuts,
            expert_first_kernels,
        )
        second_values = stage_upper_rows(
            points[:, 5:10],
            points[:, 10:15],
            expert_second_cuts,
            expert_second_kernels,
        )
        return np.exp(entropies[:, None] + first_values + second_values)

    vertex_features = features(product_vertices)
    feature_count = vertex_features.shape[1]
    affine_equality_target = np.zeros(feature_count)
    affine_equality_target[0] = 1.0

    equality, equality_target, domain_upper, domain_target = domain_matrices()
    constraints = [
        LinearConstraint(equality, equality_target, equality_target),
        LinearConstraint(domain_upper, -np.inf, domain_target),
    ]
    bounds = Bounds(lower_array, upper_array)
    random = np.random.default_rng(args.seed)
    base_starts = []
    for _ in range(max(args.restarts * 3, 64)):
        vertex = linprog(
            random.normal(size=15),
            A_ub=domain_upper,
            b_ub=domain_target,
            A_eq=equality,
            b_eq=equality_target,
            bounds=list(zip(lower_array, upper_array, strict=True)),
            method="highs",
        )
        if vertex.success:
            base_starts.append(vertex.x)
        if len(base_starts) >= args.restarts:
            break
    saved_additions: list[np.ndarray] = []
    history = (
        list(resumed.get("history", []))
        if resumed is not None and args.resume is not None
        else []
    )
    if resumed is not None:
        raw_additions = resumed.get("training_additions")
        if raw_additions is None:
            raw_additions = []
            for entry in history:
                entry_additions = entry.get("added_points")
                if entry_additions is None:
                    entry_additions = [entry["point"]]
                raw_additions.extend(entry_additions)
        saved_additions = distinct_points(
            [
                point
                for point in (
                    np.asarray(saved_point, dtype=float)
                    for saved_point in raw_additions
                )
                if point_in_box(point, lower_array, upper_array)
            ]
        )
    training = (
        np.vstack((product_vertices, *saved_additions))
        if saved_additions
        else product_vertices.copy()
    )
    historical_starts = (
        saved_additions[-args.historical_restarts :]
        if args.historical_restarts
        else []
    )
    coefficients = None
    final_local_maxima = []
    closed = False
    termination = "iteration limit"
    iteration_offset = len(history)
    for local_iteration in range(args.iterations):
        iteration = iteration_offset + local_iteration
        expert_count = len(expert_pairs)
        nonnegative = -sparse.kron(
            sparse.eye(expert_count, format="csr"),
            sparse.csr_matrix(vertex_features),
            format="csr",
        )
        nonnegative = sparse.hstack(
            (nonnegative, sparse.csr_matrix((nonnegative.shape[0], 1))),
            format="csr",
        )
        affine_equality = np.zeros(
            (feature_count, expert_count * feature_count + 1)
        )
        for feature in range(feature_count):
            for expert in range(expert_count):
                affine_equality[
                    feature, expert * feature_count + feature
                ] = 1.0
        affine_objective = np.zeros(expert_count * feature_count + 1)
        affine_objective[-1] = 1.0
        training_features = features(training)
        training_exponentials = expert_exponentials(training)
        mixture_blocks = [
            sparse.csr_matrix(
                training_features * training_exponentials[:, [expert]]
            )
            for expert in range(expert_count)
        ]
        mixture_rows = sparse.hstack(
            (*mixture_blocks, -np.ones((len(training), 1))), format="csr"
        )
        upper_blocks = [nonnegative, mixture_rows]
        upper_targets = [
            np.full(nonnegative.shape[0], -args.gate_weight_margin),
            np.zeros(mixture_rows.shape[0]),
        ]
        if args.enforce_message_direction is not None:
            training_derivatives = message_directional_exponential_derivatives(
                training,
                training_exponentials,
                expert_first_cuts,
                expert_first_kernels,
                args.enforce_message_direction,
            )
            derivative_blocks = [
                sparse.csr_matrix(
                    -(
                        training_features * training_derivatives[:, [expert]]
                        + training_exponentials[:, [expert]]
                        * direction_features[None, :]
                    )
                )
                for expert in range(expert_count)
            ]
            derivative_rows = sparse.hstack(
                (*derivative_blocks, np.zeros((len(training), 1))), format="csr"
            )
            upper_blocks.append(derivative_rows)
            upper_targets.append(
                np.full(len(training), -args.derivative_margin)
            )
        affine_upper = sparse.vstack(upper_blocks, format="csr")
        affine_upper_target = np.concatenate(upper_targets)
        fit = linprog(
            affine_objective,
            A_ub=affine_upper,
            b_ub=affine_upper_target,
            A_eq=affine_equality,
            b_eq=affine_equality_target,
            bounds=[(None, None)] * (expert_count * feature_count)
            + [(None, None)],
            method="highs",
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        coefficients = fit.x[:-1].reshape(expert_count, feature_count)

        def log_mixture(point: np.ndarray) -> float:
            point_features = np.concatenate(
                ([1.0], point[feature_coordinates])
            )
            weights = coefficients @ point_features
            exponentials = expert_exponentials(point[None, :])[0]
            return float(np.log(max(float(weights @ exponentials), 1e-300)))

        def message_directional_derivative(point: np.ndarray) -> float:
            if args.enforce_message_direction is None:
                return float("inf")
            point_features = np.concatenate(
                ([1.0], point[feature_coordinates])
            )
            weights = coefficients @ point_features
            exponentials = expert_exponentials(point[None, :])
            derivatives = message_directional_exponential_derivatives(
                point[None, :],
                exponentials,
                expert_first_cuts,
                expert_first_kernels,
                args.enforce_message_direction,
            )[0]
            weight_derivatives = coefficients @ direction_features
            return float(weights @ derivatives + weight_derivatives @ exponentials[0])

        optimized_points = []
        search_starts = distinct_points(historical_starts + base_starts)
        for start in search_starts:
            optimized = minimize(
                lambda point: -log_mixture(point),
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
            )
            if optimized.success:
                optimized_points.append(
                    (-float(optimized.fun), optimized.x.copy())
                )
        if not optimized_points:
            raise RuntimeError("continuous maximization found no successful restart")
        optimized_points.sort(key=lambda value: value[0], reverse=True)
        distinct = []
        for value, point in optimized_points:
            if all(np.max(np.abs(point - other)) > 1e-8 for _, other in distinct):
                distinct.append((value, point))
        final_local_maxima = [
            {"continuous_log_value": value, "point": point.tolist()}
            for value, point in distinct
        ]
        continuous_upper, best_point = distinct[0]
        derivative_minimum = float("inf")
        derivative_point = None
        derivative_distinct = []
        if args.enforce_message_direction is not None:
            derivative_optimized = []
            for start in search_starts:
                optimized = minimize(
                    message_directional_derivative,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
                )
                if optimized.success:
                    derivative_optimized.append(
                        (float(optimized.fun), optimized.x.copy())
                    )
            if not derivative_optimized:
                raise RuntimeError("derivative minimization found no successful restart")
            derivative_optimized.sort(key=lambda value: value[0])
            for value, point in derivative_optimized:
                if all(
                    np.max(np.abs(point - other)) > 1e-8
                    for _, other in derivative_distinct
                ):
                    derivative_distinct.append((value, point))
            derivative_minimum, derivative_point = derivative_distinct[0]
        entry = {
            "iteration": iteration + 1,
            "training_points": len(training),
            "training_log_upper": float(np.log(fit.x[-1])),
            "continuous_log_upper": continuous_upper,
            "point": best_point.tolist(),
            "distinct_local_maxima": len(distinct),
            "search_restarts": len(search_starts),
            "experts": expert_count,
            "continuous_derivative_lower": derivative_minimum,
            "derivative_point": (
                derivative_point.tolist() if derivative_point is not None else None
            ),
        }
        history.append(entry)
        print(
            f"iteration {iteration + 1}: training={np.log(fit.x[-1]):.9g} "
            f"continuous={continuous_upper:.9g} derivative={derivative_minimum:.9g} "
            f"local_maxima={len(distinct)}",
            flush=True,
        )
        value_closed = continuous_upper <= -args.target_margin
        derivative_closed = derivative_minimum >= args.derivative_margin
        if value_closed and derivative_closed:
            closed = True
            termination = "value and derivative margins reached"
            additions = []
            violating_points = []
        else:
            value_violations = [
                point
                for value, point in distinct
                if value > -args.target_margin
            ][: args.points_per_iteration]
            derivative_violations = [
                point
                for value, point in derivative_distinct
                if value < args.derivative_margin
            ][: args.points_per_iteration]
            violating_points = distinct_points(
                value_violations + derivative_violations
            )[: args.points_per_iteration * 2]
            additions = [
                point
                for point in violating_points
                if all(
                    np.max(np.abs(point - old)) > 1e-9
                    for old in training
                )
            ]
        entry["added_points"] = [point.tolist() for point in additions]
        if additions:
            training = np.vstack((training, *additions))
            saved_additions.extend(additions)
            historical_starts = distinct_points(
                additions + historical_starts
            )[: args.historical_restarts]
        added_experts = 0
        if args.generate_experts:
            existing = {
                (int(pair["first_cut"]), int(pair["second_cut"]))
                for pair in expert_pairs
            }
            for point in violating_points:
                first_values = stage_upper_rows(
                    point[None, 0:5],
                    point[None, 5:10],
                    candidate_first_cuts,
                    candidate_first_kernels,
                )[0]
                second_values = stage_upper_rows(
                    point[None, 5:10],
                    point[None, 10:15],
                    candidate_second_cuts,
                    candidate_second_kernels,
                )[0]
                pair = (
                    candidate_first_indices[int(np.argmin(first_values))],
                    candidate_second_indices[int(np.argmin(second_values))],
                )
                if pair not in existing:
                    expert_pairs.append(
                        {"first_cut": pair[0], "second_cut": pair[1]}
                    )
                    existing.add(pair)
                    added_experts += 1
            (
                expert_first_cuts,
                expert_second_cuts,
                expert_first_kernels,
                expert_second_kernels,
            ) = prepare_experts()
        entry["added_experts"] = added_experts
        if not closed and not additions and added_experts == 0:
            termination = "no new violating point or expert"

        checkpoint = {
            "status": "floating continuous affine-gate constraint generation",
            "fixed": args.fix,
            "gate": str(args.gate),
            "feature_coordinates": feature_coordinates,
            "enforced_message_direction": args.enforce_message_direction,
            "derivative_margin": args.derivative_margin,
            "gate_weight_margin": args.gate_weight_margin,
            "state_source": str(state_path) if state_path is not None else None,
            "state_mode": (
                "resume"
                if args.resume is not None
                else "seed_points"
                if args.seed_points_only
                else "seed"
                if args.seed_from is not None
                else "fresh"
            ),
            "experts": len(expert_pairs),
            "expert_pairs": expert_pairs,
            "coefficient_expert_pairs": expert_pairs[:expert_count],
            "history": history,
            "training_additions": [point.tolist() for point in saved_additions],
            "final_coefficients": coefficients.tolist(),
            "final_local_maxima": final_local_maxima,
            "closed": closed,
            "target_margin": args.target_margin,
            "termination": termination,
            "complete": closed,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if closed or (not additions and added_experts == 0):
            break

    payload = {
        "status": "floating continuous affine-gate constraint generation",
        "fixed": args.fix,
        "gate": str(args.gate),
        "feature_coordinates": feature_coordinates,
        "enforced_message_direction": args.enforce_message_direction,
        "derivative_margin": args.derivative_margin,
        "gate_weight_margin": args.gate_weight_margin,
        "state_source": str(state_path) if state_path is not None else None,
        "state_mode": (
            "resume"
            if args.resume is not None
            else "seed_points"
            if args.seed_points_only
            else "seed"
            if args.seed_from is not None
            else "fresh"
        ),
        "experts": len(expert_pairs),
        "expert_pairs": expert_pairs,
        "coefficient_expert_pairs": expert_pairs[: len(coefficients)]
        if coefficients is not None
        else [],
        "history": history,
        "final_coefficients": coefficients.tolist()
        if coefficients is not None
        else None,
        "final_local_maxima": final_local_maxima,
        "training_additions": [point.tolist() for point in saved_additions],
        "closed": closed,
        "target_margin": args.target_margin,
        "termination": termination,
        "complete": closed,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
