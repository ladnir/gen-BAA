#!/usr/bin/env python3
"""Audit exact stationary symmetries and their effect on the weak F_5 box.

The reported optimization values are floating diagnostics.  The group actions
are exact identities of the stationary information problem.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import exact_stationary_dual as stationary
import explore_projected_kl_box_cover as projected
import optimize_bulk_exponent as bulk


def cut_key(cut: stationary.StationaryDualCut) -> tuple[float, ...]:
    values = np.concatenate(
        (
            np.array([cut.moment_constant]),
            cut.moment_coefficients,
            cut.kernel.ravel(),
        )
    )
    return tuple(float(value) for value in np.round(values, decimals=12))


def unique_cuts(
    cuts: list[stationary.StationaryDualCut],
) -> tuple[stationary.StationaryDualCut, ...]:
    unique: dict[tuple[float, ...], stationary.StationaryDualCut] = {}
    for cut in cuts:
        unique.setdefault(cut_key(cut), cut)
    return tuple(unique.values())


def expand_bank(
    cuts: tuple[stationary.StationaryDualCut, ...], family: str
) -> tuple[stationary.StationaryDualCut, ...]:
    expanded: list[stationary.StationaryDualCut] = []
    if family == "original":
        return cuts
    for cut in cuts:
        if family == "scaling":
            parameters = (
                (multiplier, 0, False) for multiplier in range(1, 5)
            )
        elif family == "edge":
            parameters = (
                (multiplier, 0, reverse)
                for multiplier in range(1, 5)
                for reverse in (False, True)
            )
        elif family == "affine":
            parameters = (
                (multiplier, translation, False)
                for multiplier in range(1, 5)
                for translation in range(5)
            )
        elif family == "full":
            parameters = (
                (multiplier, translation, reverse)
                for multiplier in range(1, 5)
                for translation in range(5)
                for reverse in (False, True)
            )
        else:
            raise ValueError(f"unknown cut family {family}")
        expanded.extend(
            stationary.transform_stationary_cut(cut, *parameter)
            for parameter in parameters
        )
    return unique_cuts(expanded)


def entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def stationary_exponent(point: np.ndarray) -> tuple[float, float]:
    first = stationary.fit_stationary_dual(point[0:5], point[5:10])
    second = stationary.fit_stationary_dual(point[5:10], point[10:15])
    residual = max(
        first.equality_residual,
        first.gradient_residual,
        second.equality_residual,
        second.gradient_residual,
    )
    if residual > 1e-6:
        raise RuntimeError(
            f"stationary solver residual {residual} is too large"
        )
    return entropy(point[0:5]) / 3.0 - first.value - second.value, residual


def fixed_envelope(
    point: np.ndarray,
    first: tuple[stationary.StationaryDualCut, ...],
    second: tuple[stationary.StationaryDualCut, ...],
) -> float:
    return (
        entropy(point[0:5]) / 3.0
        - max(
            stationary.fixed_cut_payment(point[0:5], point[5:10], cut)
            for cut in first
        )
        - max(
            stationary.fixed_cut_payment(point[5:10], point[10:15], cut)
            for cut in second
        )
    )


def sign_average(distribution: np.ndarray) -> np.ndarray:
    opposite = (-np.arange(5)) % 5
    return (distribution + distribution[opposite]) / 2.0


def multiplicative_average(distribution: np.ndarray) -> np.ndarray:
    return sum(
        stationary.affine_symbol_pushforward(distribution, multiplier)
        for multiplier in range(1, 5)
    ) / 4.0


def global_group() -> tuple[tuple[int, int, int], ...]:
    signs = (1, -1)
    return tuple(
        sorted(
            {
                tuple((multiplier * sign) % 5 for sign in choices)
                for multiplier in range(1, 5)
                for choices in itertools.product(signs, repeat=3)
            }
        )
    )


def transform_compositions(
    point: np.ndarray, multipliers: tuple[int, int, int]
) -> np.ndarray:
    transformed = np.empty(15)
    for block, multiplier in enumerate(multipliers):
        start = 5 * block
        transformed[start : start + 5] = stationary.affine_symbol_pushforward(
            point[start : start + 5], multiplier
        )
    return transformed


def random_feasible_compositions(
    random: np.random.Generator, eta: float, delta: float, count: int
) -> list[np.ndarray]:
    layout = bulk.Layout(5)
    equality, equality_target, upper, upper_target = bulk.constraints(
        layout, eta, delta
    )
    vertices = bulk.feasible_vertices(
        layout,
        equality,
        equality_target,
        upper,
        upper_target,
        max(48, count),
        random,
    )
    points = []
    for _ in range(count):
        chosen = random.choice(
            len(vertices), size=min(16, len(vertices)), replace=True
        )
        weights = random.dirichlet(np.ones(len(chosen)))
        full_point = sum(
            weight * vertices[index]
            for weight, index in zip(weights, chosen, strict=True)
        )
        point = np.concatenate(
            (
                full_point[layout.mu],
                full_point[layout.beta],
                full_point[layout.zeta],
            )
        )
        largest = 1 + int(np.argmax(point[1:5]))
        canonical_multiplier = pow(largest, -1, 5)
        points.append(
            transform_compositions(
                point,
                (
                    canonical_multiplier,
                    canonical_multiplier,
                    canonical_multiplier,
                ),
            )
        )
    return points


def symmetry_averaging_audit(
    reference: np.ndarray,
    samples: int,
    seed: int,
    eta: float,
    delta: float,
) -> dict[str, object]:
    random = np.random.default_rng(seed)
    points = [reference]
    points.extend(
        0.98 * point + 0.02 * reference
        for point in random_feasible_compositions(
            random, eta, delta, samples
        )
    )
    records = []
    maximum_residual = 0.0
    for index, point in enumerate(points):
        before, residual = stationary_exponent(point)
        sign_point = np.concatenate(
            tuple(sign_average(point[start : start + 5]) for start in (0, 5, 10))
        )
        multiplicative_point = np.concatenate(
            tuple(
                multiplicative_average(point[start : start + 5])
                for start in (0, 5, 10)
            )
        )
        after_sign, sign_residual = stationary_exponent(sign_point)
        after_multiplicative, multiplicative_residual = stationary_exponent(
            multiplicative_point
        )
        maximum_residual = max(
            maximum_residual,
            residual,
            sign_residual,
            multiplicative_residual,
        )
        records.append(
            {
                "index": index,
                "exponent": before,
                "sign_gap": after_sign - before,
                "multiplicative_gap": after_multiplicative - before,
                "point": point.tolist(),
            }
        )
    sign_worst = min(records, key=lambda record: float(record["sign_gap"]))
    multiplicative_worst = min(
        records, key=lambda record: float(record["multiplicative_gap"])
    )
    near_thresholds = {}
    for threshold in (-0.1, -0.02, -0.01, -0.005, -0.002, 0.0):
        eligible = [
            record for record in records if float(record["exponent"]) >= threshold
        ]
        near_thresholds[str(threshold)] = (
            None
            if not eligible
            else min(float(record["sign_gap"]) for record in eligible)
        )
    return {
        "samples": len(records),
        "minimum_sign_gap": float(sign_worst["sign_gap"]),
        "minimum_sign_gap_exponent": float(sign_worst["exponent"]),
        "minimum_sign_gap_point": sign_worst["point"],
        "minimum_multiplicative_gap": float(
            multiplicative_worst["multiplicative_gap"]
        ),
        "minimum_multiplicative_gap_exponent": float(
            multiplicative_worst["exponent"]
        ),
        "minimum_sign_gap_by_exponent_floor": near_thresholds,
        "maximum_stationary_solver_residual": maximum_residual,
        "scope": "floating random audit; a negative gap refutes averaging only at the sampled point",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[
        args.frontier_index
    ]["evaluated"]
    root = projected.Box(tuple(selected["lower"]), tuple(selected["upper"]))
    raw_bank = json.loads(args.exact_bank.read_text(encoding="utf-8"))
    original_first = tuple(
        stationary.decode_cut(value) for value in raw_bank["first_cuts"]
    )
    original_second = tuple(
        stationary.decode_cut(value) for value in raw_bank["second_cuts"]
    )
    reference = projected.sanitize_composition(
        np.asarray(raw_bank["maximizer"], dtype=float)
    )
    bank_args = argparse.Namespace(
        eta=args.eta,
        delta=args.delta,
        bank_iterations=20,
        bank_restarts=12,
        seed=args.bank_seed,
        bank_cache=str(args.projected_bank),
        rebuild_bank=False,
    )
    _, projected_first, projected_second = projected.load_or_generate_bank(
        bank_args
    )

    base_value = stationary.fit_stationary_dual(
        reference[0:5], reference[5:10]
    ).value
    one_stage_values = []
    one_stage_residuals = []
    for multiplier in range(1, 5):
        for translation in range(5):
            for reverse in (False, True):
                increment_multiplier = (
                    -multiplier if reverse else multiplier
                ) % 5
                transformed = stationary.fit_stationary_dual(
                    stationary.affine_symbol_pushforward(
                        reference[0:5], increment_multiplier
                    ),
                    stationary.affine_symbol_pushforward(
                        reference[5:10], multiplier, translation
                    ),
                )
                one_stage_values.append(transformed.value)
                one_stage_residuals.extend(
                    (transformed.equality_residual, transformed.gradient_residual)
                )

    beta_chamber = np.zeros(15)
    beta_chamber[9] = 1.0
    beta_chamber[6] = -1.0
    output_chamber = np.zeros(15)
    output_chamber[14] = 1.0
    output_chamber[11] = -1.0
    chamber = (beta_chamber, output_chamber)

    families = {}
    for family in args.families:
        first = expand_bank(original_first, family)
        second = expand_bank(original_second, family)
        evaluations = {}
        for sector, extra in (("unrestricted", ()), ("residual_chamber", chamber)):
            evaluated = projected.evaluate(
                root,
                projected_first,
                projected_second,
                outer_iterations=args.outer_iterations,
                affine_bilinear_cuts=0,
                exact_first_cuts=first,
                exact_second_cuts=second,
                additional_composition_inequalities=extra,
            )
            if evaluated is None:
                evaluations[sector] = None
                continue
            point = projected.sanitize_composition(np.asarray(evaluated[2]))
            evaluations[sector] = {
                "lp_upper": float(evaluated[0]),
                "fixed_envelope_at_lp_composition": fixed_envelope(
                    point, first, second
                ),
                "lp_composition": point.tolist(),
            }
        orbit_values = [
            fixed_envelope(
                transform_compositions(reference, multipliers), first, second
            )
            for multipliers in global_group()
        ]
        families[family] = {
            "first_cuts": len(first),
            "second_cuts": len(second),
            "reference_envelope": fixed_envelope(reference, first, second),
            "global_orbit_envelope_spread": max(orbit_values) - min(orbit_values),
            "weak_box": evaluations,
        }

    report = {
        "prime": 5,
        "repetition": 3,
        "eta": args.eta,
        "delta": args.delta,
        "frontier_index": args.frontier_index,
        "exact_symmetries": {
            "one_stage_group_order": 40,
            "one_stage_value": base_value,
            "one_stage_maximum_value_deviation": max(
                abs(value - base_value) for value in one_stage_values
            ),
            "one_stage_maximum_solver_residual": max(one_stage_residuals),
            "two_stage_weight_preserving_group_order": len(global_group()),
            "residual_group_order_after_message_canonicalization": 4,
        },
        "cut_families": families,
        "averaging": symmetry_averaging_audit(
            reference,
            args.random_samples,
            args.seed,
            args.eta,
            args.delta,
        ),
        "status": "exact group audit with floating optimization diagnostics",
    }
    projected.write_json_atomic(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.22)
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--frontier-index", type=int, default=166)
    parser.add_argument(
        "--exact-bank",
        type=Path,
        default=Path("results/f5-fixed-stationary-shared-bank.json"),
    )
    parser.add_argument(
        "--projected-bank",
        type=Path,
        default=Path("results/f5-projected-kl-dual-bank-0.22.json"),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("original", "scaling", "edge", "affine", "full"),
        default=("original", "scaling", "edge", "affine", "full"),
    )
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--random-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/f5-stationary-symmetry-audit.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
