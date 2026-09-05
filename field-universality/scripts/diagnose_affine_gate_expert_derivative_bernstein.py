#!/usr/bin/env python3
"""Float Bernstein proposals for termwise affine-gate derivative signs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from diagnose_affine_face_bernstein import (
    add,
    constant,
    cut_data,
    multiply,
    point_polynomials,
    power_to_bernstein,
    scale,
    stage_polynomials,
)
from explore_affine_face_interval import (
    FaceBox,
    bounded_simplex_linear_extreme,
    tighten_simplex_box,
)
from rationalize_affine_cover import (
    chamber_bounds,
    collect_leaves,
    exact_simplex_vertices,
    rational,
)


def derivative(polynomial: dict, position: int) -> dict:
    result = {}
    for exponent, coefficient in polynomial.items():
        power = exponent[position]
        if not power:
            continue
        reduced = list(exponent)
        reduced[position] -= 1
        key = tuple(reduced)
        result[key] = result.get(key, 0.0) + power * coefficient
    return result


def values_on_boolean_vertices(polynomial: dict, vertices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(vertices), dtype=np.float64)
    for exponent, coefficient in polynomial.items():
        support = np.flatnonzero(exponent)
        if len(support):
            result += coefficient * np.prod(vertices[:, support], axis=1)
        else:
            result += coefficient
    return result


def handelman_margin(
    polynomial: dict,
    slacks: list[tuple[str, dict]],
    dimension: int,
    degree: int,
) -> dict:
    """Propose P = margin + a nonnegative degree-d slack combination."""

    if degree <= 0:
        raise ValueError("Handelman degree must be positive")
    products: list[tuple[tuple[int, ...], dict]] = []
    for order in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(
            range(len(slacks)), order
        ):
            value = constant(1.0, dimension)
            for index in indices:
                value = multiply(value, slacks[index][1])
            products.append((indices, value))
    exponent_set = set(polynomial)
    exponent_set.add((0,) * dimension)
    for _, value in products:
        exponent_set.update(value)
    exponents = sorted(exponent_set, key=lambda value: (sum(value), value))
    row_for_exponent = {value: index for index, value in enumerate(exponents)}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for column, (_, value) in enumerate(products):
        for exponent, coefficient in value.items():
            if coefficient:
                rows.append(row_for_exponent[exponent])
                columns.append(column)
                data.append(float(coefficient))
    margin_column = len(products)
    rows.append(row_for_exponent[(0,) * dimension])
    columns.append(margin_column)
    data.append(1.0)
    matrix = csc_matrix(
        (data, (rows, columns)),
        shape=(len(exponents), len(products) + 1),
    )
    target = np.asarray([float(polynomial.get(value, 0.0)) for value in exponents])
    objective = np.zeros(len(products) + 1)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_eq=matrix,
        b_eq=target,
        bounds=[(0.0, None)] * len(products) + [(None, None)],
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        return {
            "degree": degree,
            "slacks": len(slacks),
            "products": len(products),
            "monomials": len(exponents),
            "error": result.message,
        }
    residual = matrix @ result.x - target
    active = [
        (float(value), indices)
        for value, (indices, _) in zip(result.x[:-1], products, strict=True)
        if value > 1e-10
    ]
    active.sort(reverse=True)
    return {
        "degree": degree,
        "slacks": len(slacks),
        "products": len(products),
        "monomials": len(exponents),
        "margin_proposal": float(result.x[-1]),
        "maximum_equality_residual": float(np.max(np.abs(residual))),
        "active_products": len(active),
        "largest_multipliers": [
            {
                "value": value,
                "factors": [slacks[index][0] for index in indices],
            }
            for value, indices in active[:20]
        ],
    }


def facet_bernstein_margin(
    polynomial: dict,
    slacks: list[tuple[str, dict]],
    dimension: int,
) -> dict:
    degree = max((max(exponent) for exponent in polynomial), default=0)
    bernstein, _ = power_to_bernstein(polynomial, degree, dimension)
    flat = bernstein.ravel()
    slack_bernstein = np.stack(
        [
            power_to_bernstein(slack, degree, dimension)[0].ravel()
            for _, slack in slacks
        ],
        axis=1,
    )
    objective = np.zeros(len(slacks) + 1)
    objective[-1] = -1.0
    constraints = np.concatenate(
        (slack_bernstein, np.ones((len(flat), 1))), axis=1
    )
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=flat,
        bounds=[(0.0, None)] * len(slacks) + [(None, None)],
        method="highs",
    )
    if not result.success:
        return {"degree": degree, "error": result.message}
    multipliers = result.x[:-1]
    adjusted = flat - slack_bernstein @ multipliers
    return {
        "degree": degree,
        "minimum_adjusted_bernstein_coefficient_proposal": float(
            np.min(adjusted)
        ),
        "reported_lp_margin_proposal": float(result.x[-1]),
        "multipliers": [
            {"slack": name, "value": float(value)}
            for (name, _), value in zip(slacks, multipliers, strict=True)
            if value > 1e-12
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--leaf", required=True)
    parser.add_argument(
        "--gate",
        type=Path,
        help="override the rational partition record's gate descriptor",
    )
    parser.add_argument(
        "--coordinate",
        type=int,
        required=True,
        choices=(*range(0, 5), *range(11, 15)),
    )
    parser.add_argument("--fix", action="append", default=[], metavar="DIM:SIDE")
    parser.add_argument(
        "--dependent-coordinates",
        default="0,5,10",
        help="comma-separated choice of one dependent coordinate in each simplex block",
    )
    parser.add_argument(
        "--half",
        action="append",
        default=[],
        metavar="DIM:SIDE",
        help="restrict a remaining coordinate to its lower or upper midpoint half",
    )
    parser.add_argument("--partition-aware-only", action="store_true")
    parser.add_argument("--combined-response-bernstein", action="store_true")
    parser.add_argument(
        "--entropy-affine-lower",
        action="store_true",
        help="retain q0/qj coupling with an affine lower envelope for the entropy slope",
    )
    parser.add_argument(
        "--handelman-degree",
        type=int,
        default=0,
        help="try a floating feasible-polytope Handelman certificate of this degree",
    )
    parser.add_argument(
        "--sign-aware-prefactors",
        action="store_true",
        help="use response signs to select one-sided expert prefactor bounds",
    )
    parser.add_argument(
        "--skip-polynomial-facets",
        action="store_true",
        help="compute cheap diagnostics but skip tensor Bernstein and facet LPs",
    )
    parser.add_argument(
        "--facet-slack",
        action="append",
        default=[],
        help="restrict facet LPs to a named valid slack (repeatable)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dependent = [int(value) for value in args.dependent_coordinates.split(",")]
    if len(dependent) != 3 or any(
        len([value for value in dependent if start <= value < start + 5]) != 1
        for start in (0, 5, 10)
    ):
        parser.error("--dependent-coordinates must choose one coordinate per block")
    if args.coordinate in dependent:
        parser.error("the differentiated coordinate cannot be dependent")
    is_message_direction = args.coordinate < 5
    dependent_message_coordinate = next(
        value for value in dependent if 0 <= value < 5
    )
    if is_message_direction and not args.partition_aware_only:
        parser.error("message derivatives currently require --partition-aware-only")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    leaf = leaves[args.leaf]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(value for value in partitions["leaves"] if value["leaf"] == args.leaf)
    denominator = int(record["coefficient_denominator"])
    coefficient_rows = [
        [float(Fraction(int(value), denominator)) for value in row]
        for row in record["coefficient_numerators"]
    ]
    feature_coordinates = [int(value) for value in record["feature_coordinates"]]
    checkpoint = json.loads((root / manifest["global_checkpoint"]).read_text(encoding="utf-8"))
    selected = sorted(checkpoint["queue"], key=lambda value: value["priority"])[
        int(manifest["frontier_index"])
    ]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate_reference = (
        str(args.gate)
        if args.gate is not None
        else str(record.get("gate", leaf["gate"]))
    )
    gate_path = root / gate_reference.replace("\\", "/")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    fixed: set[int] = set()
    for specification in args.fix:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in dependent:
            parser.error("dependent coordinates cannot be fixed")
        if coordinate == args.coordinate:
            parser.error("the differentiated coordinate cannot also be fixed")
        fixed.add(coordinate)
        if side == "lower":
            upper[coordinate] = lower[coordinate]
        elif side == "upper":
            lower[coordinate] = upper[coordinate]
        else:
            parser.error("--fix side must be lower or upper")
    for specification in args.half:
        coordinate_text, side = specification.split(":")
        coordinate = int(coordinate_text)
        if coordinate in fixed or coordinate in dependent:
            parser.error("--half coordinate must be a free nondependent coordinate")
        midpoint = (lower[coordinate] + upper[coordinate]) / 2
        if side == "lower":
            upper[coordinate] = midpoint
        elif side == "upper":
            lower[coordinate] = midpoint
        else:
            parser.error("--half side must be lower or upper")
    variables = [
        coordinate
        for coordinate in range(15)
        if coordinate not in fixed and coordinate not in dependent
    ]
    box = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if box is None:
        raise ValueError("chamber is empty")
    dimension = len(variables)
    boolean_vertices = (
        (np.arange(1 << dimension, dtype=np.uint32)[:, None]
         >> np.arange(dimension, dtype=np.uint32)[None, :])
        & 1
    ).astype(np.float64)
    position = variables.index(args.coordinate)
    point = point_polynomials(box, variables, dependent)
    point_vertex_values = np.stack(
        [values_on_boolean_vertices(value, boolean_vertices) for value in point],
        axis=1,
    )
    block_vertices = [
        exact_simplex_vertices(list(box.lower), list(box.upper), start)
        for start in (0, 5, 10)
    ]
    product_points = np.asarray(
        [
            (*product[0], *product[1], *product[2])
            for product in itertools.product(*block_vertices)
        ],
        dtype=np.float64,
    )
    domain_mask = (
        np.sum(product_points[:, 1:5], axis=1)
        <= 2.0 * np.sum(product_points[:, 6:10], axis=1) + 1e-12
    )
    domain_mask &= (
        np.sum(product_points[:, 6:10], axis=1)
        <= 2.0 * np.sum(product_points[:, 11:15], axis=1) + 1e-12
    )
    for coordinate in range(2, 5):
        domain_mask &= product_points[:, coordinate] <= product_points[:, 1] + 1e-12
    feasible_product_points = product_points[domain_mask]
    variable_lower = np.asarray([float(box.lower[index]) for index in variables])
    variable_width = np.asarray(
        [float(box.upper[index] - box.lower[index]) for index in variables]
    )
    normalized_feasible_product_points = (
        feasible_product_points[:, variables] - variable_lower[None, :]
    ) / variable_width[None, :]
    feature_matrix = np.concatenate(
        (
            np.ones((len(product_points), 1)),
            product_points[:, feature_coordinates],
        ),
        axis=1,
    )
    weight_vertex_values = feature_matrix @ np.asarray(coefficient_rows).T
    chamber_weight_lowers = np.min(weight_vertex_values, axis=0)
    chamber_weight_uppers = np.max(weight_vertex_values, axis=0)
    if float(np.min(chamber_weight_lowers)) < -1e-12:
        raise ValueError("affine gate is negative on a boxed-simplex vertex")
    entropy_tangent_lower = 0.0
    entropy_tangent_values = np.zeros(len(boolean_vertices), dtype=np.float64)
    entropy_tangent_polynomial = constant(0.0, dimension)
    entropy_affine_lower_metadata = None
    if is_message_direction:
        dependent_message_values = values_on_boolean_vertices(
            point[dependent_message_coordinate], boolean_vertices
        )
        message_values = values_on_boolean_vertices(
            point[args.coordinate], boolean_vertices
        )
        dependent_message_minimum = float(np.min(dependent_message_values))
        message_maximum = float(np.max(message_values))
        if dependent_message_minimum <= 0.0 or message_maximum <= 0.0:
            raise ValueError("message entropy derivative ratio is not positive")
        entropy_tangent_lower = (
            np.log(dependent_message_minimum / message_maximum) / 3.0
            * float(box.upper[args.coordinate] - box.lower[args.coordinate])
        )
        entropy_tangent_values = (
            np.log(dependent_message_values / message_values) / 3.0
            * float(box.upper[args.coordinate] - box.lower[args.coordinate])
        )
        entropy_tangent_polynomial = constant(entropy_tangent_lower, dimension)
        if args.entropy_affine_lower:
            q0_lower = float(box.lower[dependent_message_coordinate])
            q0_upper = float(box.upper[dependent_message_coordinate])
            qj_lower = float(box.lower[args.coordinate])
            qj_upper = float(box.upper[args.coordinate])
            if q0_lower <= 0.0 or qj_lower <= 0.0:
                raise ValueError("affine entropy envelope requires positive coordinates")
            if q0_upper == q0_lower:
                log_q0_lower = constant(math.log(q0_lower), dimension)
                q0_slope = 0.0
                q0_intercept = math.log(q0_lower)
            else:
                q0_slope = (
                    math.log(q0_upper) - math.log(q0_lower)
                ) / (q0_upper - q0_lower)
                q0_intercept = math.log(q0_lower) - q0_slope * q0_lower
                log_q0_lower = add(
                    constant(q0_intercept, dimension),
                    scale(point[dependent_message_coordinate], q0_slope),
                )
            tangent_point = (qj_lower + qj_upper) / 2.0
            negative_log_qj_lower = add(
                constant(1.0 - math.log(tangent_point), dimension),
                scale(point[args.coordinate], -1.0 / tangent_point),
            )
            entropy_tangent_polynomial = scale(
                add(log_q0_lower, negative_log_qj_lower),
                float(box.upper[args.coordinate] - box.lower[args.coordinate]) / 3.0,
            )
            entropy_affine_values = values_on_boolean_vertices(
                entropy_tangent_polynomial, boolean_vertices
            )
            entropy_tangent_lower = float(np.min(entropy_affine_values))
            entropy_affine_lower_metadata = {
                "q0_secant_slope": q0_slope,
                "q0_secant_intercept": q0_intercept,
                "negative_log_qj_tangent_point": tangent_point,
                "minimum_boolean_value": entropy_tangent_lower,
                "maximum_boolean_gap_to_actual": float(
                    np.max(entropy_tangent_values - entropy_affine_values)
                ),
                "minimum_boolean_gap_to_actual": float(
                    np.min(entropy_tangent_values - entropy_affine_values)
                ),
            }
    bank = json.loads((root / "results/f5-projected-kl-dual-bank-0.22.json").read_text(encoding="utf-8"))
    first_cuts = [cut_data(value) for value in bank["first_cuts"]]
    second_cuts = [cut_data(value) for value in bank["second_cuts"]]
    records = []
    combined_response = constant(0.0, dimension)
    direct_surrogate_vertex_values = np.zeros(len(boolean_vertices), dtype=np.float64)
    direct_actual_derivative_vertex_values = np.zeros(
        len(boolean_vertices), dtype=np.float64
    )
    direct_mixture_vertex_values = np.zeros(len(boolean_vertices), dtype=np.float64)
    direct_actual_derivative_product_vertex_values = np.zeros(
        len(feasible_product_points), dtype=np.float64
    )
    direct_surrogate_derivative_product_vertex_values = np.zeros(
        len(feasible_product_points), dtype=np.float64
    )
    product_group_surrogate_values: dict[int, np.ndarray] = {}
    direct_mixture_product_vertex_values = np.zeros(
        len(feasible_product_points), dtype=np.float64
    )
    product_entropy_tangent_values = np.zeros(
        len(feasible_product_points), dtype=np.float64
    )
    if is_message_direction:
        product_entropy_tangent_values = (
            np.log(
                feasible_product_points[:, dependent_message_coordinate]
                / feasible_product_points[:, args.coordinate]
            )
            / 3.0
            * float(box.upper[args.coordinate] - box.lower[args.coordinate])
        )
    reference_scaled_bracket = constant(0.0, dimension)
    reference_scale_error = 0.0
    sign_aware_response = constant(0.0, dimension)
    sign_aware_negative_correction = 0.0
    sign_aware_gate_derivative_lower = 0.0
    response_polynomials: dict[int, dict] = {}
    sign_aware_expert_data = []
    for expert, (row, pair) in enumerate(
        zip(coefficient_rows, record["expert_pairs"], strict=True)
    ):
        weight = constant(row[0], dimension)
        for coefficient, coordinate in zip(row[1:], feature_coordinates, strict=True):
            weight = add(weight, scale(point[coordinate], coefficient))
        exact_weight_minimum = Fraction(record["minimum_exact_weight"])
        first_partition, first_linear = stage_polynomials(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])], dimension
        )
        second_partition, second_linear = stage_polynomials(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])], dimension
        )
        weight_derivative = derivative(weight, position)
        if is_message_direction:
            active_partition = first_partition
            active_linear = first_linear
            inactive_partition = second_partition
        else:
            active_partition = second_partition
            active_linear = second_linear
            inactive_partition = first_partition
        partition_derivative = derivative(active_partition, position)
        linear_derivative = derivative(active_linear, position)
        if any(any(exponent) for exponent in linear_derivative):
            raise ValueError("stage linear derivative is not constant")
        response = add(
            partition_derivative,
            scale(
                active_partition,
                -linear_derivative.get((0,) * dimension, 0.0),
            ),
        )
        augmented_response = (
            add(response, multiply(active_partition, entropy_tangent_polynomial))
            if is_message_direction
            else response
        )
        response_polynomials.setdefault(int(pair["first_cut"]), augmented_response)
        if args.combined_response_bernstein:
            combined_response = add(
                combined_response, multiply(weight, augmented_response)
            )
        response_degree = max((max(exponent) for exponent in response), default=0)
        partition_degree = max((max(exponent) for exponent in active_partition), default=0)
        if args.partition_aware_only:
            if response_degree > 1 or partition_degree > 1:
                raise ValueError("partition-aware fast path requires multi-affine factors")
            response_values = values_on_boolean_vertices(response, boolean_vertices)
            augmented_response_values = values_on_boolean_vertices(
                augmented_response, boolean_vertices
            )
            partition_values = values_on_boolean_vertices(
                active_partition, boolean_vertices
            )
            response_minimum = float(np.min(response_values))
            augmented_response_minimum = float(
                np.min(augmented_response_values)
            )
            augmented_response_maximum = float(
                np.max(augmented_response_values)
            )
            partition_minimum = float(np.min(partition_values))
            partition_maximum = float(np.max(partition_values))
        else:
            response_bernstein, _ = power_to_bernstein(
                response, response_degree, dimension
            )
            partition_bernstein, _ = power_to_bernstein(
                active_partition, partition_degree, dimension
            )
            response_minimum = float(np.min(response_bernstein))
            augmented_response_minimum = response_minimum
            augmented_response_maximum = float(np.max(response_bernstein))
            partition_minimum = float(np.min(partition_bernstein))
            partition_maximum = float(np.max(partition_bernstein))
        if args.partition_aware_only:
            degree = None
            power_degrees = None
            bernstein_size = None
            minimum = None
            maximum = None
            weight_bernstein_minimum = None
        else:
            bracket = add(
                multiply(weight_derivative, active_partition),
                multiply(weight, response),
            )
            degree = max((max(exponent) for exponent in bracket), default=0)
            bernstein, power_degrees = power_to_bernstein(bracket, degree, dimension)
            bernstein_size = int(bernstein.size)
            minimum = float(np.min(bernstein))
            maximum = float(np.max(bernstein))
            weight_degree = max((max(exponent) for exponent in weight), default=0)
            weight_bernstein, _ = power_to_bernstein(weight, weight_degree, dimension)
            weight_bernstein_minimum = float(np.min(weight_bernstein))
        weight_derivative_value = weight_derivative.get((0,) * dimension, 0.0)
        decomposed_lower = (
            weight_derivative_value
            * (partition_minimum if weight_derivative_value >= 0.0 else partition_maximum)
            + float(exact_weight_minimum) * response_minimum
        )
        selected_bracket_lower = (
            decomposed_lower if minimum is None else max(minimum, decomposed_lower)
        )
        total_linear = add(first_linear, second_linear)
        inactive_values = values_on_boolean_vertices(
            inactive_partition, boolean_vertices
        )
        exponent_values = -values_on_boolean_vertices(total_linear, boolean_vertices)
        if args.combined_response_bernstein:
            active_values = values_on_boolean_vertices(
                active_partition, boolean_vertices
            )
            weight_values = values_on_boolean_vertices(weight, boolean_vertices)
            augmented_values = values_on_boolean_vertices(
                augmented_response, boolean_vertices
            )
            direct_surrogate_vertex_values += (
                np.exp(exponent_values)
                * inactive_values
                * (
                    weight_derivative_value * active_values
                    + weight_values * augmented_values
                )
            )
            direct_actual_derivative_vertex_values += (
                np.exp(exponent_values)
                * inactive_values
                * (
                    weight_derivative_value * active_values
                    + weight_values
                    * (
                        values_on_boolean_vertices(response, boolean_vertices)
                        + active_values * entropy_tangent_values
                    )
                )
            )
            direct_mixture_vertex_values += (
                weight_values
                * active_values
                * inactive_values
                * np.exp(exponent_values)
            )
            product_active_values = values_on_boolean_vertices(
                active_partition, normalized_feasible_product_points
            )
            product_inactive_values = values_on_boolean_vertices(
                inactive_partition, normalized_feasible_product_points
            )
            product_weight_values = values_on_boolean_vertices(
                weight, normalized_feasible_product_points
            )
            product_response_values = values_on_boolean_vertices(
                response, normalized_feasible_product_points
            )
            product_entropy_lower_values = values_on_boolean_vertices(
                entropy_tangent_polynomial, normalized_feasible_product_points
            )
            product_augmented_response_values = (
                product_response_values
                + product_active_values * product_entropy_lower_values
            )
            feasible_augmented_response_minimum = float(
                np.min(product_augmented_response_values)
            )
            product_exponent_values = -values_on_boolean_vertices(
                total_linear, normalized_feasible_product_points
            )
            direct_actual_derivative_product_vertex_values += (
                np.exp(product_exponent_values)
                * product_inactive_values
                * (
                    weight_derivative_value * product_active_values
                    + product_weight_values
                    * (
                        product_response_values
                        + product_active_values * product_entropy_tangent_values
                    )
                )
            )
            product_surrogate_contribution = (
                np.exp(product_exponent_values)
                * product_inactive_values
                * (
                    weight_derivative_value * product_active_values
                    + product_weight_values
                    * product_augmented_response_values
                )
            )
            direct_surrogate_derivative_product_vertex_values += (
                product_surrogate_contribution
            )
            first_cut_index = int(pair["first_cut"])
            if first_cut_index not in product_group_surrogate_values:
                product_group_surrogate_values[first_cut_index] = np.zeros(
                    len(feasible_product_points), dtype=np.float64
                )
            product_group_surrogate_values[first_cut_index] += (
                product_surrogate_contribution
            )
            direct_mixture_product_vertex_values += (
                product_weight_values
                * product_active_values
                * product_inactive_values
                * np.exp(product_exponent_values)
            )
        scale_lower = float(np.min(inactive_values)) * np.exp(
            float(np.min(exponent_values))
        )
        scale_upper = float(np.max(inactive_values)) * np.exp(
            float(np.max(exponent_values))
        )
        response_factor_lower = augmented_response_minimum * (
            scale_lower if augmented_response_minimum >= 0.0 else scale_upper
        )
        scaled_derivative_lower = selected_bracket_lower * (
            scale_lower if selected_bracket_lower >= 0.0 else scale_upper
        )
        full_bracket = add(
            multiply(weight_derivative, active_partition),
            multiply(weight, augmented_response),
        )
        reference_scale = (scale_lower + scale_upper) / 2.0
        reference_scaled_bracket = add(
            reference_scaled_bracket,
            scale(full_bracket, reference_scale),
        )
        if args.entropy_affine_lower:
            entropy_lower_values = values_on_boolean_vertices(
                entropy_tangent_polynomial, boolean_vertices
            )
            entropy_lower_minimum = float(np.min(entropy_lower_values))
            entropy_lower_maximum = float(np.max(entropy_lower_values))
            entropy_partition_products = (
                partition_minimum * entropy_lower_minimum,
                partition_minimum * entropy_lower_maximum,
                partition_maximum * entropy_lower_minimum,
                partition_maximum * entropy_lower_maximum,
            )
            augmented_absolute_upper = max(
                abs(response_minimum + min(entropy_partition_products)),
                abs(float(np.max(response_values)) + max(entropy_partition_products)),
            )
        else:
            augmented_absolute_upper = max(
                abs(augmented_response_minimum),
                abs(augmented_response_maximum),
            )
        bracket_absolute_upper = (
            abs(weight_derivative_value) * partition_maximum
            + float(chamber_weight_uppers[expert]) * augmented_absolute_upper
        )
        reference_scale_error += (
            (scale_upper - scale_lower) / 2.0 * bracket_absolute_upper
        )
        if args.sign_aware_prefactors:
            if not args.combined_response_bernstein:
                raise ValueError(
                    "--sign-aware-prefactors requires --combined-response-bernstein"
                )
            sign_aware_response = add(
                sign_aware_response,
                scale(
                    multiply(weight, augmented_response),
                    scale_lower,
                ),
            )
            if feasible_augmented_response_minimum < 0.0:
                sign_aware_negative_correction += (
                    (scale_upper - scale_lower)
                    * float(chamber_weight_uppers[expert])
                    * feasible_augmented_response_minimum
                )
            sign_aware_gate_derivative_lower += weight_derivative_value * (
                scale_lower * partition_minimum
                if weight_derivative_value >= 0.0
                else scale_upper * partition_maximum
            )
            sign_aware_expert_data.append(
                {
                    "first_cut": int(pair["first_cut"]),
                    "weight": weight,
                    "augmented_response": augmented_response,
                    "scale_lower": scale_lower,
                    "scale_upper": scale_upper,
                    "weight_upper": float(chamber_weight_uppers[expert]),
                }
            )
        records.append(
            {
                "expert": expert,
                "first_cut": int(pair["first_cut"]),
                "second_cut": int(pair["second_cut"]),
                "power_degrees": power_degrees,
                "bernstein_degree": degree,
                "coefficients": bernstein_size,
                "minimum_bernstein_coefficient": minimum,
                "maximum_bernstein_coefficient": maximum,
                "weight_derivative_constant": weight_derivative_value,
                "exact_chamber_weight_minimum": str(exact_weight_minimum),
                "chamber_weight_lower_proposal": float(
                    chamber_weight_lowers[expert]
                ),
                "chamber_weight_upper_proposal": float(
                    chamber_weight_uppers[expert]
                ),
                "decomposed_chamber_lower_proposal": decomposed_lower,
                "decomposed_chamber_positive_proposal": decomposed_lower > 0.0,
                "selected_chamber_bracket_lower_proposal": selected_bracket_lower,
                "positive_prefactor_lower_proposal": scale_lower,
                "positive_prefactor_upper_proposal": scale_upper,
                "positive_prefactor_reference_proposal": reference_scale,
                "positive_prefactor_reference_error_contribution_proposal": (
                    (scale_upper - scale_lower) / 2.0 * bracket_absolute_upper
                ),
                "partition_factor_lower_proposal": partition_minimum,
                "partition_factor_upper_proposal": partition_maximum,
                "weighted_response_factor_lower_proposal": (
                    response_factor_lower
                ),
                "gate_derivative_factor_lower_proposal": (
                    weight_derivative_value
                    * (
                        scale_lower * partition_minimum
                        if weight_derivative_value >= 0.0
                        else scale_upper * partition_maximum
                    )
                ),
                "scaled_derivative_lower_proposal": scaled_derivative_lower,
                "minimum_weight_bernstein_coefficient": weight_bernstein_minimum,
                "minimum_partition_bernstein_coefficient": partition_minimum,
                "maximum_partition_bernstein_coefficient": partition_maximum,
                "minimum_stage_response_bernstein_coefficient": response_minimum,
                "minimum_entropy_augmented_response_proposal": (
                    augmented_response_minimum
                ),
                "maximum_entropy_augmented_response_proposal": (
                    augmented_response_maximum
                ),
                "minimum_entropy_augmented_response_on_feasible_product_vertices_proposal": (
                    float(np.min(product_augmented_response_values))
                    if args.combined_response_bernstein
                    else None
                ),
                "termwise_positive_proposal": minimum is not None and minimum > 0.0,
                "termwise_nonnegative_proposal": minimum is not None and minimum >= 0.0,
            }
        )
        if not args.quiet:
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    response_costs = [
        value["weighted_response_factor_lower_proposal"] for value in records
    ]
    bounded_simplex_response_lower = bounded_simplex_linear_extreme(
        chamber_weight_lowers.tolist(),
        chamber_weight_uppers.tolist(),
        response_costs,
        maximize=False,
    )
    affine_gate_response_lower = float(
        np.min(weight_vertex_values @ np.asarray(response_costs))
    )
    selected_response_lower = max(
        bounded_simplex_response_lower, affine_gate_response_lower
    )
    gate_derivative_lower = sum(
        value["gate_derivative_factor_lower_proposal"] for value in records
    )
    combined_response_minimum = None
    combined_response_degree = None
    combined_response_perturbation_lower = None
    combined_response_reference = None
    combined_response_minimum_index = None
    combined_response_facet_lp = None
    combined_response_handelman = None
    reference_scaled_bracket_facet_lp = None
    sign_aware_prefactor_facet_lp = None
    response_facet_lps = None
    if args.combined_response_bernstein and not args.skip_polynomial_facets:
        message_vertex_values = point_vertex_values[:, 0:5]
        entropy_factor_vertex_values = np.exp(
            -np.sum(message_vertex_values * np.log(message_vertex_values), axis=1) / 3.0
        )
        direct_mixture_vertex_values *= entropy_factor_vertex_values
        direct_actual_derivative_vertex_values *= entropy_factor_vertex_values
        product_message_values = feasible_product_points[:, 0:5]
        product_entropy_factor_values = np.exp(
            -np.sum(product_message_values * np.log(product_message_values), axis=1)
            / 3.0
        )
        direct_mixture_product_vertex_values *= product_entropy_factor_values
        direct_actual_derivative_product_vertex_values *= (
            product_entropy_factor_values
        )
        direct_surrogate_derivative_product_vertex_values *= (
            product_entropy_factor_values
        )
        for values in product_group_surrogate_values.values():
            values *= product_entropy_factor_values
        combined_response_degree = max(
            (max(exponent) for exponent in combined_response), default=0
        )
        combined_bernstein, _ = power_to_bernstein(
            combined_response, combined_response_degree, dimension
        )
        combined_response_minimum = float(np.min(combined_bernstein))
        combined_response_minimum_index = [
            int(value)
            for value in np.unravel_index(
                np.argmin(combined_bernstein), combined_bernstein.shape
            )
        ]
        slack_polynomials: list[tuple[str, dict]] = []
        for local_coordinate in range(dimension):
            variable = {
                tuple(1 if index == local_coordinate else 0 for index in range(dimension)): 1.0
            }
            slack_polynomials.extend(
                (
                    (f"box-{variables[local_coordinate]}-lower", variable),
                    (
                        f"box-{variables[local_coordinate]}-upper",
                        add(constant(1.0, dimension), scale(variable, -1.0)),
                    ),
                )
            )
        for coordinate in dependent:
            slack_polynomials.extend(
                (
                    (
                        f"coordinate-{coordinate}-lower",
                        add(
                            point[coordinate],
                            constant(-float(box.lower[coordinate]), dimension),
                        ),
                    ),
                    (
                        f"coordinate-{coordinate}-upper",
                        add(
                            constant(float(box.upper[coordinate]), dimension),
                            scale(point[coordinate], -1.0),
                        ),
                    ),
                )
            )
        first_support_slack = constant(0.0, dimension)
        second_support_slack = constant(0.0, dimension)
        for coordinate in range(1, 5):
            first_support_slack = add(
                first_support_slack, scale(point[coordinate], -1.0)
            )
        for coordinate in range(6, 10):
            first_support_slack = add(
                first_support_slack, scale(point[coordinate], 2.0)
            )
            second_support_slack = add(
                second_support_slack, scale(point[coordinate], -1.0)
            )
        for coordinate in range(11, 15):
            second_support_slack = add(
                second_support_slack, scale(point[coordinate], 2.0)
            )
        slack_polynomials.extend(
            (
                ("first-support", first_support_slack),
                ("second-support", second_support_slack),
            )
        )
        for coordinate in range(2, 5):
            slack_polynomials.append(
                (
                    f"message-order-1-ge-{coordinate}",
                    add(point[1], scale(point[coordinate], -1.0)),
                )
            )
        if args.facet_slack:
            requested_slacks = set(args.facet_slack)
            known_slacks = {name for name, _ in slack_polynomials}
            missing_slacks = requested_slacks - known_slacks
            if missing_slacks:
                raise ValueError(
                    f"unknown --facet-slack values: {sorted(missing_slacks)}"
                )
            slack_polynomials = [
                value for value in slack_polynomials if value[0] in requested_slacks
            ]
        slack_bernstein = np.stack(
            [
                power_to_bernstein(polynomial, combined_response_degree, dimension)[0].ravel()
                for _, polynomial in slack_polynomials
            ],
            axis=1,
        )
        combined_flat = combined_bernstein.ravel()
        # Maximize t subject to every Bernstein coefficient of
        # P - sum_j lambda_j g_j being at least t, with lambda_j >= 0.
        # Each g_j is nonnegative on the true feasible polytope.
        objective = np.zeros(len(slack_polynomials) + 1)
        objective[-1] = -1.0
        constraint_matrix = np.concatenate(
            (slack_bernstein, np.ones((len(combined_flat), 1))), axis=1
        )
        facet_result = linprog(
            objective,
            A_ub=constraint_matrix,
            b_ub=combined_flat,
            bounds=[(0.0, None)] * len(slack_polynomials) + [(None, None)],
            method="highs",
        )
        if facet_result.success:
            multipliers = facet_result.x[:-1]
            adjusted = combined_flat - slack_bernstein @ multipliers
            combined_response_facet_lp = {
                "minimum_adjusted_bernstein_coefficient_proposal": float(
                    np.min(adjusted)
                ),
                "reported_lp_margin_proposal": float(facet_result.x[-1]),
                "multipliers": [
                    {
                        "slack": name,
                        "value": float(value),
                    }
                    for (name, _), value in zip(
                        slack_polynomials, multipliers, strict=True
                    )
                    if value > 1e-12
                ],
            }
        else:
            combined_response_facet_lp = {
                "error": facet_result.message,
            }
        if args.sign_aware_prefactors:
            response_facet_lps = {
                str(first_cut): facet_bernstein_margin(
                    polynomial, slack_polynomials, dimension
                )
                for first_cut, polynomial in sorted(response_polynomials.items())
            }
            certified_response_margins = {
                int(first_cut): float(
                    result["minimum_adjusted_bernstein_coefficient_proposal"]
                )
                for first_cut, result in response_facet_lps.items()
                if "minimum_adjusted_bernstein_coefficient_proposal" in result
            }
            if len(certified_response_margins) != len(response_polynomials):
                raise RuntimeError("a first-cut response facet LP failed")
            sign_aware_response = constant(0.0, dimension)
            sign_aware_negative_correction = 0.0
            for value in sign_aware_expert_data:
                sign_aware_response = add(
                    sign_aware_response,
                    scale(
                        multiply(value["weight"], value["augmented_response"]),
                        value["scale_lower"],
                    ),
                )
                response_margin = certified_response_margins[value["first_cut"]]
                if response_margin < 0.0:
                    sign_aware_negative_correction += (
                        (value["scale_upper"] - value["scale_lower"])
                        * value["weight_upper"]
                        * response_margin
                    )
        reference_degree = max(
            (max(exponent) for exponent in reference_scaled_bracket), default=0
        )
        if not args.sign_aware_prefactors:
            reference_bernstein, _ = power_to_bernstein(
                reference_scaled_bracket, reference_degree, dimension
            )
            if reference_degree == combined_response_degree:
                reference_slack_bernstein = slack_bernstein
            else:
                reference_slack_bernstein = np.stack(
                    [
                        power_to_bernstein(polynomial, reference_degree, dimension)[0].ravel()
                        for _, polynomial in slack_polynomials
                    ],
                    axis=1,
                )
            reference_flat = reference_bernstein.ravel()
            reference_objective = np.zeros(len(slack_polynomials) + 1)
            reference_objective[-1] = -1.0
            reference_constraints = np.concatenate(
                (
                    reference_slack_bernstein,
                    np.ones((len(reference_flat), 1)),
                ),
                axis=1,
            )
            reference_result = linprog(
                reference_objective,
                A_ub=reference_constraints,
                b_ub=reference_flat,
                bounds=[(0.0, None)] * len(slack_polynomials) + [(None, None)],
                method="highs",
            )
            if reference_result.success:
                reference_multipliers = reference_result.x[:-1]
                reference_adjusted = (
                    reference_flat
                    - reference_slack_bernstein @ reference_multipliers
                )
                reference_margin = float(np.min(reference_adjusted))
                reference_scaled_bracket_facet_lp = {
                    "degree": reference_degree,
                    "minimum_adjusted_bernstein_coefficient_proposal": reference_margin,
                    "reported_lp_margin_proposal": float(reference_result.x[-1]),
                    "prefactor_interval_error_upper_proposal": reference_scale_error,
                    "derivative_lower_proposal": reference_margin
                    - reference_scale_error,
                    "multipliers": [
                        {"slack": name, "value": float(value)}
                        for (name, _), value in zip(
                            slack_polynomials, reference_multipliers, strict=True
                        )
                        if value > 1e-12
                    ],
                }
            else:
                reference_scaled_bracket_facet_lp = {
                    "degree": reference_degree,
                    "error": reference_result.message,
                    "prefactor_interval_error_upper_proposal": reference_scale_error,
                }
        else:
            sign_degree = max(
                (max(exponent) for exponent in sign_aware_response), default=0
            )
            sign_bernstein, _ = power_to_bernstein(
                sign_aware_response, sign_degree, dimension
            )
            if sign_degree == combined_response_degree:
                sign_slack_bernstein = slack_bernstein
            else:
                sign_slack_bernstein = np.stack(
                    [
                        power_to_bernstein(polynomial, sign_degree, dimension)[0].ravel()
                        for _, polynomial in slack_polynomials
                    ],
                    axis=1,
                )
            sign_flat = sign_bernstein.ravel()
            sign_objective = np.zeros(len(slack_polynomials) + 1)
            sign_objective[-1] = -1.0
            sign_constraints = np.concatenate(
                (sign_slack_bernstein, np.ones((len(sign_flat), 1))), axis=1
            )
            sign_result = linprog(
                sign_objective,
                A_ub=sign_constraints,
                b_ub=sign_flat,
                bounds=[(0.0, None)] * len(slack_polynomials) + [(None, None)],
                method="highs",
            )
            if sign_result.success:
                sign_multipliers = sign_result.x[:-1]
                sign_adjusted = sign_flat - sign_slack_bernstein @ sign_multipliers
                sign_margin = float(np.min(sign_adjusted))
                sign_aware_prefactor_facet_lp = {
                    "degree": sign_degree,
                    "minimum_adjusted_bernstein_coefficient_proposal": sign_margin,
                    "negative_response_correction_lower_proposal": sign_aware_negative_correction,
                    "gate_derivative_lower_proposal": sign_aware_gate_derivative_lower,
                    "derivative_lower_proposal": sign_margin
                    + sign_aware_negative_correction
                    + sign_aware_gate_derivative_lower,
                    "multipliers": [
                        {"slack": name, "value": float(value)}
                        for (name, _), value in zip(
                            slack_polynomials, sign_multipliers, strict=True
                        )
                        if value > 1e-12
                    ],
                }
            else:
                sign_aware_prefactor_facet_lp = {
                    "degree": sign_degree,
                    "error": sign_result.message,
                }
        if args.handelman_degree:
            combined_response_handelman = handelman_margin(
                combined_response,
                slack_polynomials,
                dimension,
                args.handelman_degree,
            )
        best = None
        for reference in np.linspace(0.5, 1.5, 2001):
            error = max(
                max(
                    abs(value["positive_prefactor_lower_proposal"] - reference),
                    abs(value["positive_prefactor_upper_proposal"] - reference),
                )
                * max(
                    abs(value["minimum_entropy_augmented_response_proposal"]),
                    abs(value["maximum_entropy_augmented_response_proposal"]),
                )
                for value in records
            )
            lower_bound = reference * combined_response_minimum - error
            if best is None or lower_bound > best[0]:
                best = (lower_bound, float(reference))
        combined_response_perturbation_lower, combined_response_reference = best
    payload = {
        "status": "floating termwise derivative Bernstein proposal; not a proof",
        "leaf": args.leaf,
        "gate": gate_reference,
        "coordinate": args.coordinate,
        "dependent_message_coordinate": dependent_message_coordinate,
        "fixed": args.fix,
        "dependent_coordinates": dependent,
        "half_restrictions": args.half,
        "experts": len(records),
        "positive_experts": sum(value["termwise_positive_proposal"] for value in records),
        "nonnegative_experts": sum(value["termwise_nonnegative_proposal"] for value in records),
        "worst_minimum_bernstein_coefficient": (
            None
            if args.partition_aware_only
            else min(value["minimum_bernstein_coefficient"] for value in records)
        ),
        "aggregate_derivative_lower_proposal": sum(
            value["scaled_derivative_lower_proposal"] for value in records
        ),
        "aggregate_derivative_positive_proposal": bool(
            sum(value["scaled_derivative_lower_proposal"] for value in records) > 0.0
        ),
        "partition_aware_only": args.partition_aware_only,
        "combined_response_bernstein_degree": combined_response_degree,
        "combined_response_bernstein_minimum_proposal": combined_response_minimum,
        "combined_response_bernstein_minimum_index": (
            combined_response_minimum_index
        ),
        "combined_response_reference_proposal": combined_response_reference,
        "combined_response_perturbation_lower_proposal": (
            combined_response_perturbation_lower
        ),
        "combined_response_facet_lp_proposal": combined_response_facet_lp,
        "combined_response_handelman_proposal": combined_response_handelman,
        "reference_scaled_bracket_facet_lp_proposal": (
            reference_scaled_bracket_facet_lp
        ),
        "sign_aware_prefactor_facet_lp_proposal": (
            sign_aware_prefactor_facet_lp
        ),
        "first_cut_response_facet_lp_proposals": response_facet_lps,
        "sign_aware_negative_correction_lower_proposal": (
            sign_aware_negative_correction
        ),
        "sign_aware_gate_derivative_lower_proposal": (
            sign_aware_gate_derivative_lower
        ),
        "direct_surrogate_minimum_vertex_proposal": (
            float(np.min(direct_surrogate_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_surrogate_minimum_vertex_index": (
            [
                int(value)
                for value in np.unravel_index(
                    np.argmin(direct_surrogate_vertex_values),
                    (2,) * dimension,
                )
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_vertex_proposal": (
            float(np.min(direct_actual_derivative_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_vertex_index": (
            [
                int(value)
                for value in np.unravel_index(
                    np.argmin(direct_actual_derivative_vertex_values),
                    (2,) * dimension,
                )
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_point": (
            [
                float(value)
                for value in point_vertex_values[
                    int(np.argmin(direct_actual_derivative_vertex_values))
                ]
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_point_domain_residuals": (
            [
                float(
                    np.sum(
                        point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            1:5,
                        ]
                    )
                    - 2.0
                    * np.sum(
                        point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            6:10,
                        ]
                    )
                ),
                float(
                    np.sum(
                        point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            6:10,
                        ]
                    )
                    - 2.0
                    * np.sum(
                        point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            11:15,
                        ]
                    )
                ),
                *[
                    float(
                        point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            coordinate,
                        ]
                        - point_vertex_values[
                            int(np.argmin(direct_actual_derivative_vertex_values)),
                            1,
                        ]
                    )
                    for coordinate in range(2, 5)
                ],
            ]
            if args.combined_response_bernstein
            else None
        ),
        "feasible_product_vertices": (
            int(len(feasible_product_points))
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_product_vertex_proposal": (
            float(np.min(direct_actual_derivative_product_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_maximum_product_vertex_proposal": (
            float(np.max(direct_actual_derivative_product_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_surrogate_derivative_minimum_product_vertex_proposal": (
            float(np.min(direct_surrogate_derivative_product_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_surrogate_derivative_maximum_product_vertex_proposal": (
            float(np.max(direct_surrogate_derivative_product_vertex_values))
            if args.combined_response_bernstein
            else None
        ),
        "direct_surrogate_first_cut_group_product_vertex_ranges_proposal": (
            {
                str(first_cut): {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                for first_cut, values in sorted(product_group_surrogate_values.items())
            }
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_minimum_product_vertex_point": (
            [
                float(value)
                for value in feasible_product_points[
                    int(np.argmin(direct_actual_derivative_product_vertex_values))
                ]
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_actual_derivative_maximum_product_vertex_point": (
            [
                float(value)
                for value in feasible_product_points[
                    int(np.argmax(direct_actual_derivative_product_vertex_values))
                ]
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_mixture_minus_one_maximum_product_vertex_proposal": (
            float(np.max(direct_mixture_product_vertex_values) - 1.0)
            if args.combined_response_bernstein
            else None
        ),
        "direct_mixture_minus_one_maximum_vertex_proposal": (
            float(np.max(direct_mixture_vertex_values) - 1.0)
            if args.combined_response_bernstein
            else None
        ),
        "direct_mixture_minus_one_maximum_vertex_index": (
            [
                int(value)
                for value in np.unravel_index(
                    np.argmax(direct_mixture_vertex_values),
                    (2,) * dimension,
                )
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_mixture_minus_one_maximum_point": (
            [
                float(value)
                for value in point_vertex_values[
                    int(np.argmax(direct_mixture_vertex_values))
                ]
            ]
            if args.combined_response_bernstein
            else None
        ),
        "direct_mixture_minus_one_maximum_point_domain_residuals": (
            [
                float(
                    np.sum(
                        point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), 1:5
                        ]
                    )
                    - 2.0
                    * np.sum(
                        point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), 6:10
                        ]
                    )
                ),
                float(
                    np.sum(
                        point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), 6:10
                        ]
                    )
                    - 2.0
                    * np.sum(
                        point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), 11:15
                        ]
                    )
                ),
                *[
                    float(
                        point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), coordinate
                        ]
                        - point_vertex_values[
                            int(np.argmax(direct_mixture_vertex_values)), 1
                        ]
                    )
                    for coordinate in range(2, 5)
                ],
            ]
            if args.combined_response_bernstein
            else None
        ),
        "entropy_tangent_lower_proposal": entropy_tangent_lower,
        "entropy_affine_lower_proposal": entropy_affine_lower_metadata,
        "reference_scale_error_upper_proposal": reference_scale_error,
        "partition_aware_gate_derivative_lower_proposal": gate_derivative_lower,
        "partition_aware_weighted_response_lower_proposal": min(
            response_costs
        ),
        "bounded_simplex_weighted_response_lower_proposal": (
            bounded_simplex_response_lower
        ),
        "affine_gate_weighted_response_lower_proposal": affine_gate_response_lower,
        "selected_weighted_response_lower_proposal": selected_response_lower,
        "partition_aware_aggregate_lower_proposal": gate_derivative_lower
        + selected_response_lower,
        "records": records,
    }
    if args.combined_response_bernstein:
        first_cut_product_minima = {}
        for value in records:
            key = str(value["first_cut"])
            proposal = value[
                "minimum_entropy_augmented_response_on_feasible_product_vertices_proposal"
            ]
            first_cut_product_minima[key] = (
                proposal
                if key not in first_cut_product_minima
                else min(first_cut_product_minima[key], proposal)
            )
        payload[
            "first_cut_minimum_entropy_augmented_response_on_feasible_product_vertices_proposal"
        ] = first_cut_product_minima
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    summary = {key: value for key, value in payload.items() if key != "records"}
    if args.quiet:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
