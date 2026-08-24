#!/usr/bin/env python3
"""Optimize the exact globally sign-symmetric BAA bulk problem.

This is a numerical diagnostic, not a certificate.  It stores one mass for
each orbit under simultaneous negation.  Thus an odd-prime transition uses
(p^2+1)/2 variables instead of p^2 variables, without relaxing any constraint
inside the globally sign-symmetric subproblem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import verify_sign_orbit_bulk as exact


LOG_TWO = float(np.log(2.0))


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1 if divisor == 2 else 2
    return True


def entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(-np.dot(positive, np.log(positive)))


def entropy_gradient(values: np.ndarray) -> np.ndarray:
    return -(np.log(np.maximum(values, 1e-300)) + 1.0)


def full_orbit_entropy(values: np.ndarray) -> float:
    return entropy(values) + LOG_TWO * float(values[1:].sum())


def full_orbit_entropy_gradient(values: np.ndarray) -> np.ndarray:
    gradient = entropy_gradient(values)
    gradient[1:] += LOG_TWO
    return gradient


def full_pair_entropy(
    values: np.ndarray, layout: exact.Layout
) -> float:
    nontrivial_mass = sum(
        value
        for value, representative in zip(
            values, layout.pair_representatives, strict=True
        )
        if representative != (0, 0)
    )
    return entropy(values) + LOG_TWO * float(nontrivial_mass)


def full_pair_entropy_gradient(
    values: np.ndarray, layout: exact.Layout
) -> np.ndarray:
    gradient = entropy_gradient(values)
    for index, representative in enumerate(layout.pair_representatives):
        if representative != (0, 0):
            gradient[index] += LOG_TWO
    return gradient


def support_entropy_and_gradient(
    values: np.ndarray,
) -> tuple[float, np.ndarray]:
    pushed = np.array([values[0], values[1:].sum()])
    pushed_gradient = entropy_gradient(pushed)
    gradient = np.full_like(values, pushed_gradient[1])
    gradient[0] = pushed_gradient[0]
    return entropy(pushed), gradient


def support_joint_and_gradient(
    values: np.ndarray, layout: exact.Layout
) -> tuple[float, np.ndarray, np.ndarray]:
    pushed = np.zeros(4)
    categories = np.empty(len(layout.triples), dtype=int)
    for index, (increment, source, _) in enumerate(layout.triples):
        category = 2 * int(increment != 0) + int(source != 0)
        categories[index] = category
        pushed[category] += values[index]
    pushed_gradient = entropy_gradient(pushed)
    return entropy(pushed), pushed_gradient[categories], pushed


def orbit_transition_decomposition(
    values: np.ndarray,
    increment: np.ndarray,
    state: np.ndarray,
    layout: exact.Layout,
) -> dict[str, float]:
    pair = np.zeros((layout.orbit_count, layout.orbit_count))
    both_nonzero = 0.0
    for value, (increment_orbit, state_orbit, _) in zip(
        values, layout.triples, strict=True
    ):
        pair[increment_orbit, state_orbit] += value
        if increment_orbit != 0 and state_orbit != 0:
            both_nonzero += float(value)
    orbit_information = (
        entropy(increment) + entropy(state) - entropy(pair.ravel())
    )
    branch_entropy = entropy(values) - entropy(pair.ravel())
    sign_toll = both_nonzero * LOG_TWO - branch_entropy
    support_increment, _ = support_entropy_and_gradient(increment)
    support_state, _ = support_entropy_and_gradient(state)
    _, _, support_joint = support_joint_and_gradient(values, layout)
    support_information = (
        support_increment + support_state - entropy(support_joint)
    )
    return {
        "orbit_information": orbit_information,
        "magnitude_information_beyond_support": (
            orbit_information - support_information
        ),
        "both_nonzero_mass": both_nonzero,
        "relative_sign_branch_entropy": branch_entropy,
        "relative_sign_information_toll": sign_toll,
        "support_joint": support_joint.tolist(),
    }


def exponent_and_gradient(
    point: np.ndarray, layout: exact.Layout, repetition: int
) -> tuple[float, np.ndarray]:
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    tau1 = point[layout.gamma1]
    tau2 = point[layout.gamma2]
    coefficients = (
        (layout.mu, -float(repetition - 1) / repetition),
        (layout.beta, -2.0),
        (layout.zeta, -1.0),
    )
    value = full_pair_entropy(tau1, layout) + full_pair_entropy(tau2, layout)
    gradient = np.zeros(layout.size)
    gradient[layout.gamma1] = full_pair_entropy_gradient(tau1, layout)
    gradient[layout.gamma2] = full_pair_entropy_gradient(tau2, layout)
    for component, coefficient in coefficients:
        values = point[component]
        value += coefficient * full_orbit_entropy(values)
        gradient[component] += coefficient * full_orbit_entropy_gradient(values)
    return value, gradient


def information_decomposition(
    point: np.ndarray, layout: exact.Layout, repetition: int
) -> dict[str, float]:
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    full_entropies = {
        "mu": full_orbit_entropy(mu),
        "beta": full_orbit_entropy(beta),
        "zeta": full_orbit_entropy(zeta),
        "tau1": full_pair_entropy(point[layout.gamma1], layout),
        "tau2": full_pair_entropy(point[layout.gamma2], layout),
    }
    support_mu, _ = support_entropy_and_gradient(mu)
    support_beta, _ = support_entropy_and_gradient(beta)
    support_zeta, _ = support_entropy_and_gradient(zeta)
    support_tau1, _, _ = support_joint_and_gradient(
        point[layout.gamma1], layout
    )
    support_tau2, _, _ = support_joint_and_gradient(
        point[layout.gamma2], layout
    )
    full_first = (
        full_entropies["mu"]
        + full_entropies["beta"]
        - full_entropies["tau1"]
    )
    full_second = (
        full_entropies["beta"]
        + full_entropies["zeta"]
        - full_entropies["tau2"]
    )
    support_first = support_mu + support_beta - support_tau1
    support_second = support_beta + support_zeta - support_tau2
    internal_message = full_entropies["mu"] - support_mu
    gap = (
        full_first
        - support_first
        + full_second
        - support_second
        - internal_message / repetition
    )
    phi = support_mu / repetition - support_first - support_second
    first_orbit = orbit_transition_decomposition(
        point[layout.gamma1], mu, beta, layout
    )
    second_orbit = orbit_transition_decomposition(
        point[layout.gamma2], beta, zeta, layout
    )
    return {
        "full_first_stage_information": full_first,
        "full_second_stage_information": full_second,
        "support_first_stage_information": support_first,
        "support_second_stage_information": support_second,
        "internal_message_entropy": internal_message,
        "support_exponent_phi": phi,
        "internal_information_gap": gap,
        "first_stage_orbit_decomposition": first_orbit,
        "second_stage_orbit_decomposition": second_orbit,
        "reconstructed_exponent": phi - gap,
    }


def gap_and_gradient(
    point: np.ndarray, layout: exact.Layout, repetition: int
) -> tuple[float, np.ndarray]:
    mu = point[layout.mu]
    beta = point[layout.beta]
    zeta = point[layout.zeta]
    tau1 = point[layout.gamma1]
    tau2 = point[layout.gamma2]

    full_mu_gradient = full_orbit_entropy_gradient(mu)
    full_beta_gradient = full_orbit_entropy_gradient(beta)
    full_zeta_gradient = full_orbit_entropy_gradient(zeta)
    full_tau1_gradient = full_pair_entropy_gradient(tau1, layout)
    full_tau2_gradient = full_pair_entropy_gradient(tau2, layout)
    support_mu, support_mu_gradient = support_entropy_and_gradient(mu)
    support_beta, support_beta_gradient = support_entropy_and_gradient(beta)
    support_zeta, support_zeta_gradient = support_entropy_and_gradient(zeta)
    support_tau1, support_tau1_gradient, _ = support_joint_and_gradient(
        tau1, layout
    )
    support_tau2, support_tau2_gradient, _ = support_joint_and_gradient(
        tau2, layout
    )

    gradient = np.zeros(layout.size)
    gradient[layout.gamma1] += -full_tau1_gradient + support_tau1_gradient
    gradient[layout.gamma2] += -full_tau2_gradient + support_tau2_gradient
    gradient[layout.mu] += (
        full_mu_gradient
        - support_mu_gradient
        - (full_mu_gradient - support_mu_gradient) / repetition
    )
    gradient[layout.beta] += 2.0 * (
        full_beta_gradient - support_beta_gradient
    )
    gradient[layout.zeta] += full_zeta_gradient - support_zeta_gradient

    decomposition = information_decomposition(point, layout, repetition)
    return decomposition["internal_information_gap"], gradient


def feasible_vertices(
    equality: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    random: np.random.Generator,
) -> list[np.ndarray]:
    vertices: list[np.ndarray] = []
    for _ in range(count):
        direction = random.normal(size=lower.size)
        result = linprog(
            direction,
            A_eq=equality,
            b_eq=target,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if result.success:
            vertices.append(result.x)
    if not vertices:
        raise RuntimeError("could not find a feasible sign-symmetric point")
    return vertices


@dataclass(frozen=True)
class SearchResult:
    point: np.ndarray
    objective: float
    success: bool
    message: str


def optimize(args: argparse.Namespace) -> dict[str, object]:
    layout = exact.make_layout(args.prime, sign_symmetric=True)
    equality_fraction, target_fraction = exact.structural_equalities(layout)
    equality = np.array(equality_fraction, dtype=float)
    target = np.array(target_fraction, dtype=float)
    lower = np.zeros(layout.size)
    upper = np.ones(layout.size)
    upper[layout.mu.start] = 1.0 - args.eta
    lower[layout.zeta.start] = 1.0 - args.delta
    if args.max_message_weight is not None:
        lower[layout.mu.start] = 1.0 - args.max_message_weight

    random = np.random.default_rng(args.seed)
    vertices = feasible_vertices(
        equality,
        target,
        lower,
        upper,
        max(32, args.restarts * 4),
        random,
    )
    starts = [np.mean(vertices, axis=0)]
    for _ in range(args.restarts - 1):
        chosen = random.choice(len(vertices), size=min(24, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        starts.append(
            sum(weight * vertices[index] for weight, index in zip(weights, chosen))
        )

    constraint = LinearConstraint(equality, target, target)
    best: SearchResult | None = None
    for start in starts:
        if args.objective == "exponent":
            value_and_gradient = lambda point: exponent_and_gradient(
                point, layout, args.repetition
            )
        else:
            def value_and_gradient(point: np.ndarray) -> tuple[float, np.ndarray]:
                exponent, exponent_gradient = exponent_and_gradient(
                    point, layout, args.repetition
                )
                gap, gap_gradient = gap_and_gradient(
                    point, layout, args.repetition
                )
                return (
                    exponent - args.gap_multiplier * gap,
                    exponent_gradient - args.gap_multiplier * gap_gradient,
                )

        result = minimize(
            lambda point: -value_and_gradient(point)[0],
            start,
            jac=lambda point: -value_and_gradient(point)[1],
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=(constraint,),
            options={
                "ftol": args.tolerance,
                "maxiter": args.max_iterations,
                "disp": False,
            },
        )
        equality_error = float(np.max(np.abs(equality @ result.x - target)))
        if equality_error > 1e-7:
            continue
        candidate = SearchResult(
            result.x,
            -float(result.fun),
            bool(result.success),
            str(result.message),
        )
        if best is None or candidate.objective > best.objective:
            best = candidate
    if best is None:
        raise RuntimeError("no restart returned a feasible point")

    exponent, _ = exponent_and_gradient(best.point, layout, args.repetition)
    decomposition = information_decomposition(best.point, layout, args.repetition)
    return {
        "prime": args.prime,
        "repetition": args.repetition,
        "delta": args.delta,
        "eta": args.eta,
        "max_message_weight": args.max_message_weight,
        "objective": args.objective,
        "gap_multiplier": (
            args.gap_multiplier if args.objective == "lagrangian" else None
        ),
        "objective_found": best.objective,
        "exponent_at_result": exponent,
        "message_nonzero_fraction": 1.0 - best.point[layout.mu.start],
        "middle_nonzero_fraction": 1.0 - best.point[layout.beta.start],
        "output_nonzero_fraction": 1.0 - best.point[layout.zeta.start],
        "mu_orbits": best.point[layout.mu].tolist(),
        "beta_orbits": best.point[layout.beta].tolist(),
        "zeta_orbits": best.point[layout.zeta].tolist(),
        "transition_variables_per_stage": len(layout.triples),
        "optimizer_success": best.success,
        "optimizer_message": best.message,
        **decomposition,
        "status": "numerical exact sign-symmetric subproblem only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--max-message-weight", type=float)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--objective", choices=("exponent", "lagrangian"), default="exponent"
    )
    parser.add_argument("--gap-multiplier", type=float, default=1.5)
    args = parser.parse_args()
    if args.prime < 3 or not is_prime(args.prime):
        parser.error("prime must be odd")
    if args.repetition != 3:
        parser.error("the current ladder study is specialized to repetition three")
    if not 0.0 < args.eta < 1.0:
        parser.error("eta must lie between zero and one")
    if not 0.0 < args.delta < 0.5:
        parser.error("delta must lie between zero and one half")
    if (
        args.max_message_weight is not None
        and not args.eta <= args.max_message_weight <= 1.0
    ):
        parser.error("max-message-weight must lie between eta and one")
    return args


def main() -> None:
    print(json.dumps(optimize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
