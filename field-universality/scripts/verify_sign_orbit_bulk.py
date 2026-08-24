#!/usr/bin/env python3
"""Verify a quinary sign-orbit bulk bound with exact LP dual checks.

The search uses floating-point linear programming only to propose witnesses
and dual variables.  Every retained upper bound is checked with rational
arithmetic.  Logarithms enter only through directed mpmath intervals.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from mpmath import iv
from scipy.optimize import linprog


ZERO = Fraction(0)
ONE = Fraction(1)
THREE = Fraction(3)
COEFFICIENT_BITS = 56
DUAL_BITS = 36


@dataclass(frozen=True)
class Box:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]


@dataclass(frozen=True)
class Cut:
    first: tuple[Fraction, ...]
    second: tuple[Fraction, ...]


@dataclass(frozen=True)
class Bound:
    value: Fraction
    box: Box
    cuts: tuple[Cut, ...]
    split_dimension: int


@dataclass(frozen=True)
class Layout:
    prime: int
    orbits: tuple[tuple[int, ...], ...]
    triples: tuple[tuple[int, int, int], ...]

    @property
    def orbit_count(self) -> int:
        return len(self.orbits)

    @property
    def mu(self) -> slice:
        return slice(0, self.orbit_count)

    @property
    def beta(self) -> slice:
        return slice(self.orbit_count, 2 * self.orbit_count)

    @property
    def zeta(self) -> slice:
        return slice(2 * self.orbit_count, 3 * self.orbit_count)

    @property
    def gamma1(self) -> slice:
        start = 3 * self.orbit_count
        return slice(start, start + len(self.triples))

    @property
    def gamma2(self) -> slice:
        return slice(self.gamma1.stop, self.gamma1.stop + len(self.triples))

    @property
    def size(self) -> int:
        return self.gamma2.stop


def make_layout(prime: int) -> Layout:
    seen: set[int] = set()
    orbits = []
    for value in range(prime):
        if value in seen:
            continue
        current = tuple(sorted({value, (-value) % prime}))
        seen.update(current)
        orbits.append(current)
    triples = []
    for increment, increment_orbit in enumerate(orbits):
        for source, source_orbit in enumerate(orbits):
            for target, target_orbit in enumerate(orbits):
                if any(
                    (state + difference) % prime in target_orbit
                    for difference in increment_orbit
                    for state in source_orbit
                ):
                    triples.append((increment, source, target))
    return Layout(prime, tuple(orbits), tuple(triples))


def mpf_fraction(raw: tuple[int, int, int, int]) -> Fraction:
    sign, mantissa, exponent, _ = raw
    value = Fraction(mantissa)
    if exponent >= 0:
        value *= 2**exponent
    else:
        value /= 2 ** (-exponent)
    return -value if sign else value


def interval(value: Fraction) -> object:
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def interval_upper(value: object) -> Fraction:
    exact_upper = mpf_fraction(value._mpi_[1])
    scale = 1 << COEFFICIENT_BITS
    scaled = exact_upper * scale
    numerator = -((-scaled.numerator) // scaled.denominator)
    return Fraction(numerator, scale)


@lru_cache(maxsize=None)
def log_upper(value: Fraction) -> Fraction:
    if value <= ZERO:
        raise ValueError("logarithm argument must be positive")
    return interval_upper(iv.log(interval(value)))


@lru_cache(maxsize=None)
def negative_log_upper(value: Fraction) -> Fraction:
    if value <= ZERO:
        raise ValueError("logarithm argument must be positive")
    return interval_upper(-iv.log(interval(value)))


@lru_cache(maxsize=None)
def xlogx_upper(value: Fraction) -> Fraction:
    if value == ZERO:
        return ZERO
    point = interval(value)
    return interval_upper(point * iv.log(point))


def secant_upper(lower: Fraction, upper: Fraction) -> tuple[Fraction, Fraction]:
    lower_value = xlogx_upper(lower)
    if lower == upper:
        return ZERO, lower_value
    upper_value = xlogx_upper(upper)
    slope = (upper_value - lower_value) / (upper - lower)
    return slope, lower_value - slope * lower


def independent_equalities(
    rows: list[list[Fraction]], targets: list[Fraction]
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    basis: dict[int, tuple[list[Fraction], Fraction]] = {}
    selected = []
    for index, (raw_row, raw_target) in enumerate(zip(rows, targets, strict=True)):
        row = list(raw_row)
        target = raw_target
        for pivot in sorted(basis):
            basis_row, basis_target = basis[pivot]
            factor = row[pivot]
            if factor:
                row = [a - factor * b for a, b in zip(row, basis_row, strict=True)]
                target -= factor * basis_target
        pivot = next((column for column, value in enumerate(row) if value), None)
        if pivot is None:
            if target:
                raise ValueError("inconsistent structural equalities")
            continue
        scale = row[pivot]
        row = [value / scale for value in row]
        target /= scale
        basis[pivot] = (row, target)
        selected.append(index)
    return (
        tuple(tuple(rows[index]) for index in selected),
        tuple(targets[index] for index in selected),
    )


def structural_equalities(
    layout: Layout,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    rows: list[list[Fraction]] = []
    targets: list[Fraction] = []
    for component in (layout.mu, layout.beta, layout.zeta):
        row = [ZERO] * layout.size
        for index in range(component.start, component.stop):
            row[index] = ONE
        rows.append(row)
        targets.append(ONE)
    for gamma, increment_component, state_component in (
        (layout.gamma1, layout.mu, layout.beta),
        (layout.gamma2, layout.beta, layout.zeta),
    ):
        for orbit_index in range(layout.orbit_count):
            increment_row = [ZERO] * layout.size
            source_row = [ZERO] * layout.size
            target_row = [ZERO] * layout.size
            for offset, (increment, source, target) in enumerate(layout.triples):
                variable = gamma.start + offset
                if increment == orbit_index:
                    increment_row[variable] = ONE
                if source == orbit_index:
                    source_row[variable] = ONE
                if target == orbit_index:
                    target_row[variable] = ONE
            increment_row[increment_component.start + orbit_index] -= ONE
            source_row[state_component.start + orbit_index] -= ONE
            target_row[state_component.start + orbit_index] -= ONE
            rows.extend((increment_row, source_row, target_row))
            targets.extend((ZERO, ZERO, ZERO))
    return independent_equalities(rows, targets)


def tighten_simplex(
    lower: list[Fraction], upper: list[Fraction], start: int, dimension: int
) -> bool:
    for _ in range(dimension + 1):
        total_lower = sum(lower[start : start + dimension], ZERO)
        total_upper = sum(upper[start : start + dimension], ZERO)
        if total_lower > ONE or total_upper < ONE:
            return False
        changed = False
        for offset in range(dimension):
            index = start + offset
            new_lower = max(lower[index], ONE - (total_upper - upper[index]))
            new_upper = min(upper[index], ONE - (total_lower - lower[index]))
            if new_lower > new_upper:
                return False
            changed |= new_lower != lower[index] or new_upper != upper[index]
            lower[index] = max(ZERO, new_lower)
            upper[index] = min(ONE, new_upper)
        if not changed:
            break
    return True


def tighten(raw_box: Box, orbit_count: int) -> Box | None:
    lower = list(raw_box.lower)
    upper = list(raw_box.upper)
    for _ in range(8):
        before = (tuple(lower), tuple(upper))
        for start in (0, orbit_count, 2 * orbit_count):
            if not tighten_simplex(lower, upper, start, orbit_count):
                return None
        # Every accumulator satisfies wt(increment) <= 2 wt(state).
        for increment_zero, state_zero in (
            (0, orbit_count),
            (orbit_count, 2 * orbit_count),
        ):
            upper[state_zero] = min(
                upper[state_zero], (ONE + upper[increment_zero]) / 2
            )
            lower[increment_zero] = max(
                lower[increment_zero], 2 * lower[state_zero] - ONE
            )
            if lower[increment_zero] > upper[increment_zero]:
                return None
            if lower[state_zero] > upper[state_zero]:
                return None
        if before == (tuple(lower), tuple(upper)):
            break
    return Box(tuple(lower), tuple(upper))


def pair_distribution(point: np.ndarray, layout: Layout, gamma: slice) -> np.ndarray:
    pair = np.zeros((layout.orbit_count, layout.orbit_count))
    for value, (increment, source, _) in zip(point[gamma], layout.triples, strict=True):
        pair[increment, source] += value
    pair = np.maximum(pair, 0.0)
    total = pair.sum()
    return pair / total if total > 0.0 else np.full_like(pair, 1.0 / pair.size)


def rational_distribution(values: np.ndarray, bits: int) -> tuple[Fraction, ...]:
    scale = 1 << bits
    weights = [max(1, int(round(float(value) * scale))) for value in values.flat]
    total = sum(weights)
    return tuple(Fraction(weight, total) for weight in weights)


def cut_from_point(point: np.ndarray, layout: Layout, bits: int) -> Cut:
    stages = []
    for gamma in (layout.gamma1, layout.gamma2):
        pair = rational_distribution(pair_distribution(point, layout, gamma), bits)
        coefficients = []
        for increment, source, _ in layout.triples:
            probability = pair[increment * layout.orbit_count + source]
            coefficients.append(negative_log_upper(probability))
        stages.append(tuple(coefficients))
    return Cut(stages[0], stages[1])


def uniform_cut(layout: Layout) -> Cut:
    probability = Fraction(1, layout.orbit_count**2)
    coefficient = negative_log_upper(probability)
    stage = tuple(coefficient for _ in layout.triples)
    return Cut(stage, stage)


def variable_bounds(box: Box, layout: Layout) -> tuple[tuple[Fraction, Fraction], ...]:
    bounds = [
        (box.lower[index], box.upper[index])
        for index in range(3 * layout.orbit_count)
    ]
    bounds.extend((ZERO, ONE) for _ in range(layout.size - len(bounds)))
    bounds.extend(((ZERO, THREE), (ZERO, THREE)))
    return tuple(bounds)


def as_float_matrix(rows: tuple[tuple[Fraction, ...], ...]) -> np.ndarray:
    return np.array([[float(value) for value in row] for row in rows])


def as_float_vector(values: tuple[Fraction, ...]) -> np.ndarray:
    return np.array([float(value) for value in values])


def relaxed_float_bounds(
    bounds: tuple[tuple[Fraction, Fraction], ...], slack: float = 1e-5
) -> list[tuple[float, float]]:
    relaxed = []
    for lower, upper in bounds:
        if upper == THREE:
            relaxed.append((-1.0, 4.0))
        else:
            relaxed.append((float(lower) - slack, float(upper) + slack))
    return relaxed


def bounded_fraction(value: float) -> Fraction:
    if not np.isfinite(value):
        raise ValueError("nonfinite LP dual value")
    scale = 1 << DUAL_BITS
    return Fraction(round(float(value) * scale), scale)


def exact_dual_upper_bound(
    objective: tuple[Fraction, ...],
    constant: Fraction,
    upper_rows: tuple[tuple[Fraction, ...], ...],
    upper_targets: tuple[Fraction, ...],
    equality: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
    bounds: tuple[tuple[Fraction, Fraction], ...],
    result: object,
) -> Fraction:
    inequality_dual = tuple(
        max(ZERO, -bounded_fraction(value)) for value in result.ineqlin.marginals
    )
    equality_dual = tuple(-bounded_fraction(value) for value in result.eqlin.marginals)
    base = [ZERO] * len(objective)
    for multiplier, row in zip(inequality_dual, upper_rows, strict=True):
        for index, coefficient in enumerate(row):
            base[index] += multiplier * coefficient
    for multiplier, row in zip(equality_dual, equality, strict=True):
        for index, coefficient in enumerate(row):
            base[index] += multiplier * coefficient
    upper_multipliers = []
    lower_multipliers = []
    for coefficient, represented in zip(objective, base, strict=True):
        residual = coefficient - represented
        upper_multipliers.append(max(ZERO, residual))
        lower_multipliers.append(max(ZERO, -residual))
    dual_value = constant
    dual_value += sum(
        multiplier * value
        for multiplier, value in zip(inequality_dual, upper_targets, strict=True)
    )
    dual_value += sum(
        multiplier * value
        for multiplier, value in zip(equality_dual, target, strict=True)
    )
    dual_value += sum(
        upper * multiplier - lower * correction
        for (lower, upper), multiplier, correction in zip(
            bounds, upper_multipliers, lower_multipliers, strict=True
        )
    )
    return dual_value


def phase_one_reference(
    box: Box,
    layout: Layout,
    equality: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
) -> tuple[np.ndarray | None, bool]:
    """Return a proposed point or an exact certificate of infeasibility."""
    size = layout.size + 1
    objective = [ZERO] * size
    objective[-1] = -ONE
    upper_rows = []
    upper_targets = []
    for row, value in zip(equality, target, strict=True):
        positive = list(row) + [-ONE]
        negative = [-coefficient for coefficient in row] + [-ONE]
        upper_rows.extend((tuple(positive), tuple(negative)))
        upper_targets.extend((value, -value))
    upper_rows_tuple = tuple(upper_rows)
    upper_targets_tuple = tuple(upper_targets)
    bounds = variable_bounds(box, layout)[: layout.size] + ((ZERO, THREE),)
    result = linprog(
        -as_float_vector(tuple(objective)),
        A_ub=as_float_matrix(upper_rows_tuple),
        b_ub=as_float_vector(upper_targets_tuple),
        bounds=relaxed_float_bounds(bounds),
        method="highs",
        options={
            "presolve": False,
            "primal_feasibility_tolerance": 1e-5,
            "dual_feasibility_tolerance": 1e-5,
        },
    )
    if not result.success:
        raise RuntimeError(f"phase-I LP failed: {result.message}")
    upper = exact_dual_upper_bound(
        tuple(objective),
        ZERO,
        upper_rows_tuple,
        upper_targets_tuple,
        (),
        (),
        bounds,
        result,
    )
    if upper < ZERO:
        return None, True
    return result.x[: layout.size], False


def evaluate(
    box: Box,
    cuts: tuple[Cut, ...],
    layout: Layout,
    equality_core: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
    repetition: int,
) -> tuple[Fraction, int, np.ndarray] | None:
    extended_size = layout.size + 2
    objective = [ZERO] * extended_size
    objective[-2] = ONE
    objective[-1] = ONE
    ln2_over_r = log_upper(Fraction(2)) / repetition
    for index in range(layout.mu.start + 1, layout.mu.stop):
        objective[index] += ln2_over_r
    constant = ZERO
    entropy_coefficients = (
        [Fraction(repetition - 1, repetition)] * layout.orbit_count
        + [Fraction(2)] * layout.orbit_count
        + [ONE] * layout.orbit_count
    )
    for index, coefficient in enumerate(entropy_coefficients):
        slope, intercept = secant_upper(box.lower[index], box.upper[index])
        objective[index] += coefficient * slope
        constant += coefficient * intercept

    upper_rows = []
    for cut in cuts:
        first = [ZERO] * extended_size
        first[-2] = ONE
        for offset, coefficient in enumerate(cut.first):
            first[layout.gamma1.start + offset] = -coefficient
        upper_rows.append(tuple(first))
        second = [ZERO] * extended_size
        second[-1] = ONE
        for offset, coefficient in enumerate(cut.second):
            second[layout.gamma2.start + offset] = -coefficient
        upper_rows.append(tuple(second))
    upper_rows_tuple = tuple(upper_rows)
    upper_targets = tuple(ZERO for _ in upper_rows_tuple)
    equality = tuple(row + (ZERO, ZERO) for row in equality_core)
    bounds = variable_bounds(box, layout)
    result = linprog(
        -as_float_vector(tuple(objective)),
        A_ub=as_float_matrix(upper_rows_tuple),
        b_ub=as_float_vector(upper_targets),
        A_eq=as_float_matrix(equality),
        b_eq=as_float_vector(target),
        bounds=relaxed_float_bounds(bounds),
        method="highs",
        options={
            "presolve": False,
            "primal_feasibility_tolerance": 1e-5,
            "dual_feasibility_tolerance": 1e-5,
        },
    )
    if not result.success:
        if result.status == 2:
            _, certified_infeasible = phase_one_reference(
                box, layout, equality_core, target
            )
            if certified_infeasible:
                return None
            raise RuntimeError(
                "floating objective LP rejected a box without an exact "
                f"infeasibility certificate: {result.message}"
            )
        raise RuntimeError(f"floating LP failed: {result.message}")
    upper = exact_dual_upper_bound(
        tuple(objective),
        constant,
        upper_rows_tuple,
        upper_targets,
        equality,
        target,
        bounds,
        result,
    )
    composition = result.x[: 3 * layout.orbit_count]
    gaps = []
    for index, coefficient in enumerate(entropy_coefficients):
        slope, intercept = secant_upper(box.lower[index], box.upper[index])
        point = max(0.0, composition[index])
        actual = 0.0 if point == 0.0 else point * np.log(point)
        gaps.append(
            float(coefficient)
            * max(0.0, float(slope) * point + float(intercept) - actual)
        )
    return upper, int(np.argmax(gaps)), result.x[: layout.size]


def box_bound(
    raw_box: Box,
    inherited: tuple[Cut, ...],
    layout: Layout,
    equality: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
    repetition: int,
    maximum_cut_pairs: int,
    cut_bits: int,
) -> Bound | None:
    box = tighten(raw_box, layout.orbit_count)
    if box is None:
        return None
    reference, certified_infeasible = phase_one_reference(
        box, layout, equality, target
    )
    if certified_infeasible:
        return None
    if reference is None:
        raise RuntimeError("phase-I returned neither a point nor a certificate")
    cuts = (inherited + (cut_from_point(reference, layout, cut_bits),))[
        -maximum_cut_pairs:
    ]
    first = evaluate(box, cuts, layout, equality, target, repetition)
    if first is None:
        return None
    _, _, witness = first
    cuts = (cuts + (cut_from_point(witness, layout, cut_bits),))[
        -maximum_cut_pairs:
    ]
    second = evaluate(box, cuts, layout, equality, target, repetition)
    if second is None:
        return None
    value, dimension, _ = second
    return Bound(value, box, cuts, dimension)


def split(box: Box, dimension: int) -> tuple[Box, Box]:
    midpoint = (box.lower[dimension] + box.upper[dimension]) / 2
    left_upper = list(box.upper)
    left_upper[dimension] = midpoint
    right_lower = list(box.lower)
    right_lower[dimension] = midpoint
    return Box(box.lower, tuple(left_upper)), Box(tuple(right_lower), box.upper)


def search(args: argparse.Namespace) -> dict[str, object]:
    iv.dps = args.precision
    layout = make_layout(args.prime)
    equality, target = structural_equalities(layout)
    lower = [ZERO] * (3 * layout.orbit_count)
    upper = [ONE] * (3 * layout.orbit_count)
    upper[layout.mu.start] = ONE - args.eta
    lower[layout.zeta.start] = ONE - args.delta
    initial = box_bound(
        Box(tuple(lower), tuple(upper)),
        (uniform_cut(layout),),
        layout,
        equality,
        target,
        args.repetition,
        args.maximum_cut_pairs,
        args.cut_bits,
    )
    if initial is None:
        raise RuntimeError("initial orbit domain is infeasible")
    queue = [(-initial.value, 0, initial)]
    identifier = 1
    splits = 0
    maximum_queue = 1
    infeasible_boxes = 0
    while queue and splits < args.max_boxes:
        _, _, current = heapq.heappop(queue)
        if current.value < ZERO:
            return {
                "status": "verified negative sign-orbit cover",
                "upper_bound": float(current.value),
                "upper_bound_numerator": str(current.value.numerator),
                "upper_bound_denominator": str(current.value.denominator),
                "upper_bound_is_exactly_negative": True,
                "upper_bound_numerator_bits": current.value.numerator.bit_length(),
                "upper_bound_denominator_bits": current.value.denominator.bit_length(),
                "boxes_split": splits,
                "boxes_remaining": len(queue) + 1,
                "maximum_queue_size": maximum_queue,
                "infeasible_boxes_checked_exactly": infeasible_boxes,
                "maximum_cut_pairs": args.maximum_cut_pairs,
            }
        dimension = current.split_dimension
        if current.box.lower[dimension] == current.box.upper[dimension]:
            widths = [
                upper - lower
                for lower, upper in zip(
                    current.box.lower, current.box.upper, strict=True
                )
            ]
            dimension = max(range(len(widths)), key=widths.__getitem__)
        for child in split(current.box, dimension):
            bounded = box_bound(
                child,
                current.cuts,
                layout,
                equality,
                target,
                args.repetition,
                args.maximum_cut_pairs,
                args.cut_bits,
            )
            if bounded is None:
                infeasible_boxes += 1
                continue
            heapq.heappush(queue, (-bounded.value, identifier, bounded))
            identifier += 1
        splits += 1
        maximum_queue = max(maximum_queue, len(queue))
    return {
        "status": "exact cover search incomplete",
        "current_upper_bound": float(-queue[0][0]) if queue else None,
        "current_upper_bound_is_exact": bool(queue),
        "boxes_split": splits,
        "boxes_remaining": len(queue),
        "maximum_queue_size": maximum_queue,
        "infeasible_boxes_checked_exactly": infeasible_boxes,
        "maximum_cut_pairs": args.maximum_cut_pairs,
    }


def parse_fraction(value: str) -> Fraction:
    return Fraction(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=3)
    parser.add_argument("--delta", type=parse_fraction, default=Fraction(1, 20))
    parser.add_argument("--eta", type=parse_fraction, default=Fraction(1, 72))
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--cut-bits", type=int, default=30)
    parser.add_argument("--maximum-cut-pairs", type=int, default=16)
    parser.add_argument("--max-boxes", type=int, default=10_000)
    args = parser.parse_args()
    if args.prime != 5:
        parser.error("the current sign-orbit theorem is specialized to p=5")
    if args.repetition != 3:
        parser.error("the current verifier is specialized to repetition three")
    if not ZERO < args.delta < Fraction(1, 2):
        parser.error("delta must lie between zero and one half")
    if not ZERO < args.eta < ONE:
        parser.error("eta must lie between zero and one")
    if args.maximum_cut_pairs < 1:
        parser.error("maximum-cut-pairs must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = search(args)
    result.update(
        {
            "prime": args.prime,
            "repetition": args.repetition,
            "delta": str(args.delta),
            "eta": str(args.eta),
            "interval_precision": args.precision,
            "cut_denominator_bits": args.cut_bits,
            "arithmetic": "directed intervals and exact rational LP duals",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
