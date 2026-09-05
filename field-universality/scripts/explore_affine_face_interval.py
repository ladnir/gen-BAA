#!/usr/bin/env python3
"""Prototype directed interval subdivision on low-dimensional affine-gate faces."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from mpmath import iv

from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


ZERO = Fraction(0)
ONE = Fraction(1)
LOG_CONCAVITY_STATS = {
    "evaluations": 0,
    "negative_curvature": 0,
    "fallback": 0,
    "minimum_lambda_upper": math.inf,
    "maximum_lambda_upper": -math.inf,
    "minimum_mixture_lower": math.inf,
    "maximum_mixture_upper": -math.inf,
    "maximum_raw_hessian_magnitude": 0.0,
}
CENTERED_CURVATURE_STATS = {
    "evaluations": 0,
    "negative_curvature": 0,
    "fallback": 0,
    "minimum_lambda_upper": math.inf,
    "maximum_lambda_upper": -math.inf,
    "maximum_third_derivative_magnitude": 0.0,
    "maximum_affine_hessian_residual": 0.0,
}
CENTERED_CURVATURE_CERTIFICATES: list[dict[str, Any]] = []
DIRECTED_TAYLOR_CERTIFICATES: list[dict[str, Any]] = []


@dataclass(frozen=True)
class FaceBox:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]


class Dual:
    """Interval value with first derivatives in the face variables."""

    __slots__ = ("value", "gradient")

    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

    def __add__(self, other):
        other = as_dual(other, len(self.gradient))
        return Dual(
            self.value + other.value,
            [left + right for left, right in zip(self.gradient, other.gradient, strict=True)],
        )

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, [-value for value in self.gradient])

    def __sub__(self, other):
        return self + (-as_dual(other, len(self.gradient)))

    def __rsub__(self, other):
        return as_dual(other, len(self.gradient)) - self

    def __mul__(self, other):
        other = as_dual(other, len(self.gradient))
        return Dual(
            self.value * other.value,
            [
                left * other.value + self.value * right
                for left, right in zip(self.gradient, other.gradient, strict=True)
            ],
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_dual(other, len(self.gradient))
        denominator = other.value * other.value
        return Dual(
            self.value / other.value,
            [
                (left * other.value - self.value * right) / denominator
                for left, right in zip(
                    self.gradient, other.gradient, strict=True
                )
            ],
        )

    def __rtruediv__(self, other):
        return as_dual(other, len(self.gradient)) / self


class DirectionalDual:
    """A dual value together with one selected directional derivative.

    Both fields are ordinary ``Dual`` values.  The gradient of ``base`` is
    the full first derivative, while the gradient of ``directional`` is the
    selected Hessian row.  This costs O(d) per arithmetic operation instead
    of the O(d^2) full ``Jet`` used by the generic face explorer.
    """

    __slots__ = ("base", "directional")

    def __init__(self, base: Dual, directional: Dual):
        self.base = base
        self.directional = directional

    def coerce(self, other) -> "DirectionalDual":
        if isinstance(other, DirectionalDual):
            return other
        dimension = len(self.base.gradient)
        return DirectionalDual(
            as_dual(other, dimension), as_dual(iv.mpf(0), dimension)
        )

    def __add__(self, other):
        other = self.coerce(other)
        return DirectionalDual(
            self.base + other.base,
            self.directional + other.directional,
        )

    __radd__ = __add__

    def __neg__(self):
        return DirectionalDual(-self.base, -self.directional)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return DirectionalDual(
            self.base * other.base,
            self.directional * other.base + self.base * other.directional,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = as_dual(iv.mpf(1), len(self.base.gradient)) / self.base
        return DirectionalDual(
            inverse,
            -self.directional * inverse * inverse,
        )

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) / self


def directional_exp(value: DirectionalDual) -> DirectionalDual:
    result = dual_exp(value.base)
    return DirectionalDual(result, result * value.directional)


def directional_log(value: DirectionalDual) -> DirectionalDual:
    return DirectionalDual(
        dual_log(value.base),
        value.directional / value.base,
    )


class AffineInterval:
    """Directed affine form in normalized box variables plus interval error."""

    __slots__ = ("center", "coefficients", "error")

    def __init__(self, center, coefficients, error=None):
        self.center = center
        self.coefficients = coefficients
        self.error = iv.mpf(0) if error is None else error

    @classmethod
    def constant(cls, value, dimension: int) -> "AffineInterval":
        return cls(value, [iv.mpf(0) for _ in range(dimension)])

    @classmethod
    def variable(cls, center, radius, position: int, dimension: int) -> "AffineInterval":
        coefficients = [iv.mpf(0) for _ in range(dimension)]
        coefficients[position] = radius
        return cls(center, coefficients)

    def coerce(self, other) -> "AffineInterval":
        if isinstance(other, AffineInterval):
            if len(other.coefficients) != len(self.coefficients):
                raise ValueError("affine-form dimensions do not match")
            return other
        return AffineInterval.constant(other, len(self.coefficients))

    def radius(self):
        return sum((abs(value) for value in self.coefficients), iv.mpf(0)) + self.error

    def interval(self):
        radius = upper_float(self.radius())
        return self.center + iv.mpf([str(-radius), str(radius)])

    def __add__(self, other):
        other = self.coerce(other)
        return AffineInterval(
            self.center + other.center,
            [
                left + right
                for left, right in zip(
                    self.coefficients, other.coefficients, strict=True
                )
            ],
            self.error + other.error,
        )

    __radd__ = __add__

    def __neg__(self):
        return AffineInterval(
            -self.center,
            [-value for value in self.coefficients],
            self.error,
        )

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        error = (
            abs(self.center) * other.error
            + abs(other.center) * self.error
            + self.radius() * other.radius()
        )
        return AffineInterval(
            self.center * other.center,
            [
                self.center * right + other.center * left
                for left, right in zip(
                    self.coefficients, other.coefficients, strict=True
                )
            ],
            error,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        full = self.interval()
        if lower_float(full) <= 0.0 <= upper_float(full):
            raise ZeroDivisionError("affine reciprocal interval contains zero")
        center = iv.mpf(1) / self.center
        slope = -iv.mpf(1) / (self.center * self.center)
        minimum_absolute = min(abs(lower_float(full)), abs(upper_float(full)))
        second_derivative = iv.mpf(str(2.0 / minimum_absolute**3))
        radius = self.radius()
        error = (
            abs(slope) * self.error
            + second_derivative * radius * radius / 2
        )
        return AffineInterval(
            center,
            [slope * value for value in self.coefficients],
            error,
        )

    def __truediv__(self, other):
        other = self.coerce(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) / self


class FastAffine:
    """Non-rigorous float affine form used only to propose subdivisions."""

    __slots__ = ("center", "coefficients", "error")

    def __init__(self, center: float, coefficients: list[float], error: float = 0.0):
        self.center = center
        self.coefficients = coefficients
        self.error = error

    @staticmethod
    def scalar_bounds(value) -> tuple[float, float]:
        if isinstance(value, FastAffine):
            return value.lower(), value.upper()
        try:
            return lower_float(value), upper_float(value)
        except AttributeError:
            scalar = float(value)
            return scalar, scalar

    @classmethod
    def constant(cls, value, dimension: int) -> "FastAffine":
        lower, upper = cls.scalar_bounds(value)
        return cls(
            (lower + upper) / 2.0,
            [0.0 for _ in range(dimension)],
            (upper - lower) / 2.0,
        )

    @classmethod
    def variable(cls, center, radius, position: int, dimension: int) -> "FastAffine":
        center_lower, center_upper = cls.scalar_bounds(center)
        radius_lower, radius_upper = cls.scalar_bounds(radius)
        coefficients = [0.0 for _ in range(dimension)]
        coefficients[position] = max(abs(radius_lower), abs(radius_upper))
        return cls(
            (center_lower + center_upper) / 2.0,
            coefficients,
            (center_upper - center_lower) / 2.0,
        )

    def coerce(self, other) -> "FastAffine":
        if isinstance(other, FastAffine):
            if len(other.coefficients) != len(self.coefficients):
                raise ValueError("fast affine-form dimensions do not match")
            return other
        return FastAffine.constant(other, len(self.coefficients))

    def radius(self) -> float:
        return sum(abs(value) for value in self.coefficients) + self.error

    def lower(self) -> float:
        return self.center - self.radius()

    def upper(self) -> float:
        return self.center + self.radius()

    def interval(self):
        return iv.mpf([str(self.lower()), str(self.upper())])

    def __add__(self, other):
        other = self.coerce(other)
        return FastAffine(
            self.center + other.center,
            [
                left + right
                for left, right in zip(
                    self.coefficients, other.coefficients, strict=True
                )
            ],
            self.error + other.error,
        )

    __radd__ = __add__

    def __neg__(self):
        return FastAffine(
            -self.center,
            [-value for value in self.coefficients],
            self.error,
        )

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        diagonal = [
            left * right
            for left, right in zip(
                self.coefficients, other.coefficients, strict=True
            )
        ]
        center_shift = sum(diagonal) / 2.0
        quadratic_error = sum(abs(value) for value in diagonal) / 2.0
        for i in range(len(self.coefficients)):
            for j in range(i + 1, len(self.coefficients)):
                quadratic_error += abs(
                    self.coefficients[i] * other.coefficients[j]
                    + self.coefficients[j] * other.coefficients[i]
                )
        affine_radius_self = sum(abs(value) for value in self.coefficients)
        affine_radius_other = sum(abs(value) for value in other.coefficients)
        error = (
            abs(self.center) * other.error
            + abs(other.center) * self.error
            + affine_radius_self * other.error
            + affine_radius_other * self.error
            + self.error * other.error
            + quadratic_error
        )
        return FastAffine(
            self.center * other.center + center_shift,
            [
                self.center * right + other.center * left
                for left, right in zip(
                    self.coefficients, other.coefficients, strict=True
                )
            ],
            error,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        lower = self.lower()
        upper = self.upper()
        if lower <= 0.0 <= upper:
            raise ZeroDivisionError("fast affine reciprocal contains zero")
        if lower == upper:
            return FastAffine.constant(1.0 / lower, len(self.coefficients))
        slope = ((1.0 / upper) - (1.0 / lower)) / (upper - lower)
        critical = math.sqrt(-1.0 / slope)
        if upper < 0.0:
            critical = -critical
        return fast_affine_secant(
            self, slope, lambda value: 1.0 / value, [critical]
        )

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) / self


def affine_exp(value: AffineInterval) -> AffineInterval:
    full = value.interval()
    center = iv.exp(value.center)
    slope = center
    second_derivative = iv.exp(iv.mpf(str(upper_float(full))))
    radius = value.radius()
    error = abs(slope) * value.error + second_derivative * radius * radius / 2
    return AffineInterval(
        center,
        [slope * coefficient for coefficient in value.coefficients],
        error,
    )


def affine_log(value: AffineInterval) -> AffineInterval:
    full = value.interval()
    lower = lower_float(full)
    if lower <= 0.0:
        raise ValueError("affine logarithm interval is not positive")
    center = iv.log(value.center)
    slope = iv.mpf(1) / value.center
    second_derivative = iv.mpf(str(1.0 / lower**2))
    radius = value.radius()
    error = abs(slope) * value.error + second_derivative * radius * radius / 2
    return AffineInterval(
        center,
        [slope * coefficient for coefficient in value.coefficients],
        error,
    )


def fast_affine_exp(value: FastAffine) -> FastAffine:
    lower = value.lower()
    upper = value.upper()
    if lower == upper:
        return FastAffine.constant(math.exp(lower), len(value.coefficients))
    slope = (math.exp(upper) - math.exp(lower)) / (upper - lower)
    return fast_affine_secant(value, slope, math.exp, [math.log(slope)])


def fast_affine_log(value: FastAffine) -> FastAffine:
    lower = value.lower()
    upper = value.upper()
    if lower <= 0.0:
        raise ValueError("fast affine logarithm is not positive")
    if lower == upper:
        return FastAffine.constant(math.log(lower), len(value.coefficients))
    slope = (math.log(upper) - math.log(lower)) / (upper - lower)
    return fast_affine_secant(value, slope, math.log, [1.0 / slope])


def fast_affine_secant(
    value: FastAffine,
    slope: float,
    function,
    critical_points: list[float],
) -> FastAffine:
    """Minimax-center a scalar secant residual over the form's interval."""

    lower = value.lower()
    upper = value.upper()
    candidates = [lower, upper]
    candidates.extend(
        point for point in critical_points if lower <= point <= upper
    )
    residuals = [function(point) - slope * point for point in candidates]
    residual_lower = min(residuals)
    residual_upper = max(residuals)
    residual_center = (residual_lower + residual_upper) / 2.0
    residual_error = (residual_upper - residual_lower) / 2.0
    rounding_pad = 8.0 * math.ulp(
        max(1.0, abs(residual_center), abs(residual_error))
    )
    return FastAffine(
        slope * value.center + residual_center,
        [slope * coefficient for coefficient in value.coefficients],
        abs(slope) * value.error + residual_error + rounding_pad,
    )


def scalar_exp(value):
    if isinstance(value, AffineInterval):
        return affine_exp(value)
    if isinstance(value, FastAffine):
        return fast_affine_exp(value)
    return iv.exp(value)


def scalar_log(value):
    if isinstance(value, AffineInterval):
        return affine_log(value)
    if isinstance(value, FastAffine):
        return fast_affine_log(value)
    return iv.log(value)


def scalar_interval(value):
    return (
        value.interval()
        if isinstance(value, (AffineInterval, FastAffine))
        else value
    )


def third_component(value, i: int, j: int, k: int):
    return (
        value.component(i, j, k)
        if isinstance(value, SymmetricThirdJet)
        else value.third[i][j][k]
    )


class Jet:
    """Interval value with first and second face derivatives."""

    __slots__ = ("value", "gradient", "hessian")

    def __init__(self, value, gradient, hessian):
        self.value = value
        self.gradient = gradient
        self.hessian = hessian

    def __add__(self, other):
        other = as_jet(other, len(self.gradient))
        return Jet(
            self.value + other.value,
            [left + right for left, right in zip(self.gradient, other.gradient, strict=True)],
            [
                [left + right for left, right in zip(left_row, right_row, strict=True)]
                for left_row, right_row in zip(self.hessian, other.hessian, strict=True)
            ],
        )

    __radd__ = __add__

    def __neg__(self):
        return Jet(
            -self.value,
            [-value for value in self.gradient],
            [[-value for value in row] for row in self.hessian],
        )

    def __sub__(self, other):
        return self + (-as_jet(other, len(self.gradient)))

    def __rsub__(self, other):
        return as_jet(other, len(self.gradient)) - self

    def __mul__(self, other):
        other = as_jet(other, len(self.gradient))
        dimension = len(self.gradient)
        return Jet(
            self.value * other.value,
            [
                self.gradient[index] * other.value
                + self.value * other.gradient[index]
                for index in range(dimension)
            ],
            [
                [
                    self.hessian[row][column] * other.value
                    + self.gradient[row] * other.gradient[column]
                    + self.gradient[column] * other.gradient[row]
                    + self.value * other.hessian[row][column]
                    for column in range(dimension)
                ]
                for row in range(dimension)
            ],
        )

    __rmul__ = __mul__


class ThirdJet:
    """Interval value with derivatives through order three."""

    __slots__ = ("value", "gradient", "hessian", "third")

    def __init__(self, value, gradient, hessian, third):
        self.value = value
        self.gradient = gradient
        self.hessian = hessian
        self.third = third

    def __add__(self, other):
        other = as_third_jet(other, len(self.gradient))
        dimension = len(self.gradient)
        return ThirdJet(
            self.value + other.value,
            [self.gradient[i] + other.gradient[i] for i in range(dimension)],
            [
                [self.hessian[i][j] + other.hessian[i][j] for j in range(dimension)]
                for i in range(dimension)
            ],
            [
                [
                    [
                        self.third[i][j][k] + other.third[i][j][k]
                        for k in range(dimension)
                    ]
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ],
        )

    __radd__ = __add__

    def __neg__(self):
        return ThirdJet(
            -self.value,
            [-value for value in self.gradient],
            [[-value for value in row] for row in self.hessian],
            [[[-value for value in line] for line in plane] for plane in self.third],
        )

    def __sub__(self, other):
        return self + (-as_third_jet(other, len(self.gradient)))

    def __rsub__(self, other):
        return as_third_jet(other, len(self.gradient)) - self

    def __mul__(self, other):
        other = as_third_jet(other, len(self.gradient))
        dimension = len(self.gradient)
        return ThirdJet(
            self.value * other.value,
            [
                self.gradient[i] * other.value
                + self.value * other.gradient[i]
                for i in range(dimension)
            ],
            [
                [
                    self.hessian[i][j] * other.value
                    + self.gradient[i] * other.gradient[j]
                    + self.gradient[j] * other.gradient[i]
                    + self.value * other.hessian[i][j]
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ],
            [
                [
                    [
                        self.third[i][j][k] * other.value
                        + self.hessian[i][j] * other.gradient[k]
                        + self.hessian[i][k] * other.gradient[j]
                        + self.hessian[j][k] * other.gradient[i]
                        + self.gradient[i] * other.hessian[j][k]
                        + self.gradient[j] * other.hessian[i][k]
                        + self.gradient[k] * other.hessian[i][j]
                        + self.value * other.third[i][j][k]
                        for k in range(dimension)
                    ]
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ],
        )

    __rmul__ = __mul__


class SymmetricThirdJet:
    """Float-affine third jet storing only symmetric tensor entries."""

    __slots__ = ("value", "gradient", "hessian", "third", "triples", "positions")

    def __init__(self, value, gradient, hessian, third, triples):
        self.value = value
        self.gradient = gradient
        self.hessian = hessian
        self.third = third
        self.triples = triples
        self.positions = {triple: index for index, triple in enumerate(triples)}

    def component(self, i: int, j: int, k: int):
        return self.third[self.positions[tuple(sorted((i, j, k)))]]

    def coerce(self, other):
        if isinstance(other, SymmetricThirdJet):
            return other
        return symmetric_third_constant(other, len(self.gradient))

    def __add__(self, other):
        other = self.coerce(other)
        dimension = len(self.gradient)
        return SymmetricThirdJet(
            self.value + other.value,
            [self.gradient[i] + other.gradient[i] for i in range(dimension)],
            [
                [self.hessian[i][j] + other.hessian[i][j] for j in range(dimension)]
                for i in range(dimension)
            ],
            [
                left + right
                for left, right in zip(self.third, other.third, strict=True)
            ],
            self.triples,
        )

    __radd__ = __add__

    def __neg__(self):
        return SymmetricThirdJet(
            -self.value,
            [-value for value in self.gradient],
            [[-value for value in row] for row in self.hessian],
            [-value for value in self.third],
            self.triples,
        )

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        dimension = len(self.gradient)
        third = []
        for i, j, k in self.triples:
            third.append(
                self.component(i, j, k) * other.value
                + self.hessian[i][j] * other.gradient[k]
                + self.hessian[i][k] * other.gradient[j]
                + self.hessian[j][k] * other.gradient[i]
                + self.gradient[i] * other.hessian[j][k]
                + self.gradient[j] * other.hessian[i][k]
                + self.gradient[k] * other.hessian[i][j]
                + self.value * other.component(i, j, k)
            )
        return SymmetricThirdJet(
            self.value * other.value,
            [
                self.gradient[i] * other.value
                + self.value * other.gradient[i]
                for i in range(dimension)
            ],
            [
                [
                    self.hessian[i][j] * other.value
                    + self.gradient[i] * other.gradient[j]
                    + self.gradient[j] * other.gradient[i]
                    + self.value * other.hessian[i][j]
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ],
            third,
            self.triples,
        )

    __rmul__ = __mul__


def symmetric_triples(dimension: int) -> list[tuple[int, int, int]]:
    return [
        (i, j, k)
        for i in range(dimension)
        for j in range(i, dimension)
        for k in range(j, dimension)
    ]


def symmetric_third_constant(value, dimension: int) -> SymmetricThirdJet:
    zero = FastAffine.constant(0.0, dimension)
    triples = symmetric_triples(dimension)
    return SymmetricThirdJet(
        FastAffine.constant(value, dimension),
        [zero for _ in range(dimension)],
        [[zero for _ in range(dimension)] for _ in range(dimension)],
        [zero for _ in triples],
        triples,
    )


def symmetric_third_exp(value: SymmetricThirdJet) -> SymmetricThirdJet:
    result = fast_affine_exp(value.value)
    dimension = len(value.gradient)
    third = []
    for i, j, k in value.triples:
        third.append(
            result
            * (
                value.component(i, j, k)
                + value.hessian[i][j] * value.gradient[k]
                + value.hessian[i][k] * value.gradient[j]
                + value.hessian[j][k] * value.gradient[i]
                + value.gradient[i] * value.gradient[j] * value.gradient[k]
            )
        )
    return SymmetricThirdJet(
        result,
        [result * value.gradient[i] for i in range(dimension)],
        [
            [
                result
                * (value.hessian[i][j] + value.gradient[i] * value.gradient[j])
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
        third,
        value.triples,
    )


def symmetric_third_log(value: SymmetricThirdJet) -> SymmetricThirdJet:
    dimension = len(value.gradient)
    square = value.value * value.value
    cube = square * value.value
    third = []
    for i, j, k in value.triples:
        third.append(
            value.component(i, j, k) / value.value
            - (
                value.hessian[i][j] * value.gradient[k]
                + value.hessian[i][k] * value.gradient[j]
                + value.hessian[j][k] * value.gradient[i]
            )
            / square
            + 2
            * value.gradient[i]
            * value.gradient[j]
            * value.gradient[k]
            / cube
        )
    return SymmetricThirdJet(
        fast_affine_log(value.value),
        [value.gradient[i] / value.value for i in range(dimension)],
        [
            [
                value.hessian[i][j] / value.value
                - value.gradient[i] * value.gradient[j] / square
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
        third,
        value.triples,
    )


def as_dual(value, dimension: int) -> Dual:
    if isinstance(value, Dual):
        return value
    return Dual(value, [iv.mpf(0) for _ in range(dimension)])


def dual_exp(value: Dual) -> Dual:
    result = scalar_exp(value.value)
    return Dual(result, [result * gradient for gradient in value.gradient])


def dual_log(value: Dual) -> Dual:
    return Dual(
        value=scalar_log(value.value),
        gradient=[gradient / value.value for gradient in value.gradient],
    )


def as_jet(value, dimension: int) -> Jet:
    if isinstance(value, Jet):
        return value
    return Jet(
        value,
        [iv.mpf(0) for _ in range(dimension)],
        [[iv.mpf(0) for _ in range(dimension)] for _ in range(dimension)],
    )


def as_third_jet(value, dimension: int) -> ThirdJet:
    if isinstance(value, ThirdJet):
        return value
    return ThirdJet(
        value,
        [iv.mpf(0) for _ in range(dimension)],
        [[iv.mpf(0) for _ in range(dimension)] for _ in range(dimension)],
        [
            [
                [iv.mpf(0) for _ in range(dimension)]
                for _ in range(dimension)
            ]
            for _ in range(dimension)
        ],
    )


def third_exp(value: ThirdJet) -> ThirdJet:
    result = scalar_exp(value.value)
    dimension = len(value.gradient)
    return ThirdJet(
        result,
        [result * value.gradient[i] for i in range(dimension)],
        [
            [
                result
                * (value.hessian[i][j] + value.gradient[i] * value.gradient[j])
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
        [
            [
                [
                    result
                    * (
                        value.third[i][j][k]
                        + value.hessian[i][j] * value.gradient[k]
                        + value.hessian[i][k] * value.gradient[j]
                        + value.hessian[j][k] * value.gradient[i]
                        + value.gradient[i] * value.gradient[j] * value.gradient[k]
                    )
                    for k in range(dimension)
                ]
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
    )


def third_log(value: ThirdJet) -> ThirdJet:
    dimension = len(value.gradient)
    square = value.value * value.value
    cube = square * value.value
    return ThirdJet(
        scalar_log(value.value),
        [value.gradient[i] / value.value for i in range(dimension)],
        [
            [
                value.hessian[i][j] / value.value
                - value.gradient[i] * value.gradient[j] / square
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
        [
            [
                [
                    value.third[i][j][k] / value.value
                    - (
                        value.hessian[i][j] * value.gradient[k]
                        + value.hessian[i][k] * value.gradient[j]
                        + value.hessian[j][k] * value.gradient[i]
                    )
                    / square
                    + 2
                    * value.gradient[i]
                    * value.gradient[j]
                    * value.gradient[k]
                    / cube
                    for k in range(dimension)
                ]
                for j in range(dimension)
            ]
            for i in range(dimension)
        ],
    )


def jet_exp(value: Jet) -> Jet:
    result = scalar_exp(value.value)
    dimension = len(value.gradient)
    return Jet(
        result,
        [result * value.gradient[index] for index in range(dimension)],
        [
            [
                result
                * (
                    value.hessian[row][column]
                    + value.gradient[row] * value.gradient[column]
                )
                for column in range(dimension)
            ]
            for row in range(dimension)
        ],
    )


def jet_log(value: Jet) -> Jet:
    dimension = len(value.gradient)
    return Jet(
        scalar_log(value.value),
        [value.gradient[index] / value.value for index in range(dimension)],
        [
            [
                value.hessian[row][column] / value.value
                - value.gradient[row] * value.gradient[column] / (value.value * value.value)
                for column in range(dimension)
            ]
            for row in range(dimension)
        ],
    )


def tighten_simplex_box(box: FaceBox, variables: list[int], dependent: list[int]) -> FaceBox | None:
    lower = list(box.lower)
    upper = list(box.upper)
    for start in (0, 5, 10):
        block_variables = [index for index in variables if start <= index < start + 5]
        block_dependent = [index for index in dependent if start <= index < start + 5]
        if len(block_dependent) != 1:
            raise ValueError("each simplex block must have one dependent coordinate")
        active = [*block_variables, block_dependent[0]]
        fixed_sum = sum(
            (lower[index] for index in range(start, start + 5) if index not in active),
            ZERO,
        )
        target = ONE - fixed_sum
        changed = True
        while changed:
            changed = False
            if sum((lower[index] for index in active), ZERO) > target:
                return None
            if sum((upper[index] for index in active), ZERO) < target:
                return None
            for index in active:
                new_lower = max(
                    lower[index],
                    target - sum((upper[other] for other in active if other != index), ZERO),
                )
                new_upper = min(
                    upper[index],
                    target - sum((lower[other] for other in active if other != index), ZERO),
                )
                if new_lower > new_upper:
                    return None
                if new_lower != lower[index] or new_upper != upper[index]:
                    lower[index] = new_lower
                    upper[index] = new_upper
                    changed = True
    return FaceBox(tuple(lower), tuple(upper))


def interval(lower: Fraction, upper: Fraction):
    return iv.mpf(
        [
            f"{lower.numerator}/{lower.denominator}",
            f"{upper.numerator}/{upper.denominator}",
        ]
    )


def upper_float(value) -> float:
    return float(value.b)


def lower_float(value) -> float:
    return float(value.a)


def outward_fraction_bounds(value) -> tuple[Fraction, Fraction]:
    import numpy as np

    lower = float(np.nextafter(lower_float(value), -np.inf))
    upper = float(np.nextafter(upper_float(value), np.inf))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("cannot rationalize a nonfinite interval endpoint")
    return Fraction.from_float(lower), Fraction.from_float(upper)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def exact_ldl_pivots(matrix: list[list[Fraction]]) -> list[Fraction] | None:
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError("LDL matrix must be square")
    lower = [[ZERO for _ in range(dimension)] for _ in range(dimension)]
    pivots = [ZERO for _ in range(dimension)]
    for i in range(dimension):
        lower[i][i] = ONE
        pivot = matrix[i][i] - sum(
            (lower[i][k] * lower[i][k] * pivots[k] for k in range(i)),
            ZERO,
        )
        if pivot <= ZERO:
            return None
        pivots[i] = pivot
        for j in range(i + 1, dimension):
            numerator = matrix[j][i] - sum(
                (
                    lower[j][k] * lower[i][k] * pivots[k]
                    for k in range(i)
                ),
                ZERO,
            )
            lower[j][i] = numerator / pivot
    return pivots


def exact_cut(raw: dict[str, Any]) -> dict[str, Any]:
    target = [iv.mpf(str(value)) for value in raw["target_potential"]]
    increment = iv.mpf(str(raw["increment_support_potential"]))
    state = iv.mpf(str(raw["state_support_potential"]))
    kernel = []
    for difference in range(5):
        row = []
        for source in range(5):
            exponent = target[(difference + source) % 5]
            if difference:
                exponent += increment
            if source:
                exponent += state
            row.append(iv.exp(exponent))
        kernel.append(row)
    return {"target": target, "increment": increment, "state": state, "kernel": kernel}


def stage_factor(left: list, right: list, cut: dict[str, Any]):
    partition = iv.mpf(0)
    for difference in range(5):
        for source in range(5):
            partition += left[difference] * cut["kernel"][difference][source] * right[source]
    linear = sum((cut["target"][index] * right[index] for index in range(5)), iv.mpf(0))
    linear += cut["increment"] * sum(left[1:], iv.mpf(0))
    linear += cut["state"] * sum(right[1:], iv.mpf(0))
    return partition * iv.exp(-linear)


def face_intervals(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list:
    values = [None] * 15
    for index in range(15):
        if index not in dependent:
            values[index] = interval(box.lower[index], box.upper[index])
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = iv.mpf(1)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return values


def face_center(
    box: FaceBox, dependent: list[int]
) -> list[Fraction]:
    values: list[Fraction | None] = [None] * 15
    for index in range(15):
        if index not in dependent:
            values[index] = (box.lower[index] + box.upper[index]) / 2
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        values[index] = ONE - sum(
            (
                values[other]
                for other in range(start, start + 5)
                if other != index
            ),
            ZERO,
        )
    if any(value is None for value in values):
        raise AssertionError("face center left an unset coordinate")
    return [value for value in values if value is not None]


def face_duals(box: FaceBox, variables: list[int], dependent: list[int]) -> list[Dual]:
    dimension = len(variables)
    values: list[Dual | None] = [None] * 15
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    for index in range(15):
        if index in dependent:
            continue
        gradient = [iv.mpf(0) for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = iv.mpf(1)
        values[index] = Dual(interval(box.lower[index], box.upper[index]), gradient)
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = as_dual(iv.mpf(1), dimension)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def face_directional_duals(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
) -> list[DirectionalDual]:
    dimension = len(variables)
    values: list[DirectionalDual | None] = [None] * 15
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    if derivative_coordinate not in positions:
        raise ValueError("the derivative coordinate must be a free face variable")
    selected_position = positions[derivative_coordinate]
    for index in range(15):
        if index in dependent:
            continue
        gradient = [iv.mpf(0) for _ in range(dimension)]
        directional_value = iv.mpf(0)
        if index in positions:
            position = positions[index]
            gradient[position] = iv.mpf(1)
            if position == selected_position:
                directional_value = iv.mpf(1)
        values[index] = DirectionalDual(
            Dual(interval(box.lower[index], box.upper[index]), gradient),
            Dual(directional_value, [iv.mpf(0) for _ in range(dimension)]),
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = DirectionalDual(
            as_dual(iv.mpf(1), dimension), as_dual(iv.mpf(0), dimension)
        )
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def face_jets(box: FaceBox, variables: list[int], dependent: list[int]) -> list[Jet]:
    dimension = len(variables)
    values: list[Jet | None] = [None] * 15
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    for index in range(15):
        if index in dependent:
            continue
        gradient = [iv.mpf(0) for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = iv.mpf(1)
        values[index] = Jet(
            interval(box.lower[index], box.upper[index]),
            gradient,
            [[iv.mpf(0) for _ in range(dimension)] for _ in range(dimension)],
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = as_jet(iv.mpf(1), dimension)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def face_third_jets(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list[ThirdJet]:
    dimension = len(variables)
    values: list[ThirdJet | None] = [None] * 15
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    for index in range(15):
        if index in dependent:
            continue
        gradient = [iv.mpf(0) for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = iv.mpf(1)
        values[index] = ThirdJet(
            interval(box.lower[index], box.upper[index]),
            gradient,
            [[iv.mpf(0) for _ in range(dimension)] for _ in range(dimension)],
            [
                [
                    [iv.mpf(0) for _ in range(dimension)]
                    for _ in range(dimension)
                ]
                for _ in range(dimension)
            ],
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = as_third_jet(iv.mpf(1), dimension)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def mixture_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> float:
    point = face_intervals(box, variables, dependent)
    entropy = iv.mpf(0)
    for value in point[0:5]:
        entropy -= value * iv.log(value)
    entropy_factor = iv.exp(entropy / 3)
    mixture = iv.mpf(0)
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = iv.mpf(f"{row[0].numerator}/{row[0].denominator}")
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            ) * point[coordinate]
        first = stage_factor(point[0:5], point[5:10], first_cuts[int(pair["first_cut"])])
        second = stage_factor(point[5:10], point[10:15], second_cuts[int(pair["second_cut"])])
        mixture += weight * entropy_factor * first * second
    return upper_float(mixture)


def bounded_simplex_linear_extreme(
    lower: list[float],
    upper: list[float],
    costs: list[float],
    *,
    maximize: bool,
) -> float:
    """Solve a box-constrained simplex linear program by greedy filling."""

    weights = [max(0.0, value) for value in lower]
    caps = [min(1.0, value) for value in upper]
    if any(left > right for left, right in zip(weights, caps, strict=True)):
        raise ValueError("inconsistent affine-weight bounds")
    remaining = 1.0 - sum(weights)
    if remaining < -1e-10:
        raise ValueError("affine-weight lower bounds exceed one")
    order = sorted(
        range(len(costs)), key=costs.__getitem__, reverse=maximize
    )
    for index in order:
        take = min(max(remaining, 0.0), caps[index] - weights[index])
        weights[index] += take
        remaining -= take
    if remaining > 1e-10:
        raise ValueError("affine-weight upper bounds do not reach one")
    return sum(weight * cost for weight, cost in zip(weights, costs, strict=True))


def mixture_convex_bounds(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, float]:
    """Bound the mixture using weight boxes plus the exact simplex identity.

    The caller must establish that the affine weights are nonnegative and sum
    to one throughout the chamber.  The remaining bounded-simplex relaxation
    is a linear program and is solved exactly in structure (with outward float
    padding here; the final verifier will replay it over rationals/intervals).
    """

    point = face_intervals(box, variables, dependent)
    entropy = iv.mpf(0)
    for value in point[0:5]:
        entropy -= value * iv.log(value)
    entropy_factor = iv.exp(entropy / 3)
    weight_lower = []
    weight_upper = []
    expert_lower = []
    expert_upper = []
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = iv.mpf(f"{row[0].numerator}/{row[0].denominator}")
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            ) * point[coordinate]
        first = stage_factor(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        expert = entropy_factor * first * second
        weight_lower.append(np_next_down(lower_float(weight)))
        weight_upper.append(np_next_up(upper_float(weight)))
        expert_lower.append(np_next_down(lower_float(expert)))
        expert_upper.append(np_next_up(upper_float(expert)))
    lower = bounded_simplex_linear_extreme(
        weight_lower, weight_upper, expert_lower, maximize=False
    )
    upper = bounded_simplex_linear_extreme(
        weight_lower, weight_upper, expert_upper, maximize=True
    )
    lower = np_next_down(lower)
    upper = np_next_up(upper)
    if not math.isfinite(lower) or lower <= 0.0 or upper < lower:
        raise ValueError("mixture bounds must be finite, positive, and ordered")
    return lower, upper


def np_next_down(value: float) -> float:
    import numpy as np

    return float(np.nextafter(value, -np.inf))


def np_next_up(value: float) -> float:
    import numpy as np

    return float(np.nextafter(value, np.inf))


def stage_factor_dual(left: list[Dual], right: list[Dual], cut: dict[str, Any]) -> Dual:
    dimension = len(left[0].gradient)
    partition = as_dual(iv.mpf(0), dimension)
    for difference in range(5):
        for source in range(5):
            partition += left[difference] * cut["kernel"][difference][source] * right[source]
    linear = as_dual(iv.mpf(0), dimension)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(left[1:], as_dual(iv.mpf(0), dimension)) * cut["increment"]
    linear += sum(right[1:], as_dual(iv.mpf(0), dimension)) * cut["state"]
    return partition * dual_exp(-linear)


def mixture_dual(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> Dual:
    point = face_duals(box, variables, dependent)
    dimension = len(variables)
    entropy = as_dual(iv.mpf(0), dimension)
    for value in point[0:5]:
        entropy -= value * dual_log(value)
    entropy_factor = dual_exp(entropy * (iv.mpf(1) / 3))
    mixture = as_dual(iv.mpf(0), dimension)
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = as_dual(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"), dimension
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_dual(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_dual(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        mixture += weight * entropy_factor * first * second
    return mixture


def stage_factor_directional(
    left: list[DirectionalDual],
    right: list[DirectionalDual],
    cut: dict[str, Any],
) -> DirectionalDual:
    dimension = len(left[0].base.gradient)
    zero = DirectionalDual(
        as_dual(iv.mpf(0), dimension), as_dual(iv.mpf(0), dimension)
    )
    partition = zero
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = zero
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(left[1:], zero) * cut["increment"]
    linear += sum(right[1:], zero) * cut["state"]
    return partition * directional_exp(-linear)


def mixture_directional_dual(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> DirectionalDual:
    point = face_directional_duals(
        box, variables, dependent, derivative_coordinate
    )
    dimension = len(variables)
    zero = DirectionalDual(
        as_dual(iv.mpf(0), dimension), as_dual(iv.mpf(0), dimension)
    )
    entropy = zero
    for value in point[0:5]:
        entropy -= value * directional_log(value)
    entropy_factor = directional_exp(entropy * (iv.mpf(1) / 3))
    mixture = zero
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = DirectionalDual(
            as_dual(
                iv.mpf(f"{row[0].numerator}/{row[0].denominator}"), dimension
            ),
            as_dual(iv.mpf(0), dimension),
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_directional(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_directional(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        mixture += weight * entropy_factor * first * second
    return mixture


def stage_factor_jet(left: list[Jet], right: list[Jet], cut: dict[str, Any]) -> Jet:
    dimension = len(left[0].gradient)
    partition = as_jet(iv.mpf(0), dimension)
    for difference in range(5):
        for source in range(5):
            partition += left[difference] * cut["kernel"][difference][source] * right[source]
    linear = as_jet(iv.mpf(0), dimension)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(left[1:], as_jet(iv.mpf(0), dimension)) * cut["increment"]
    linear += sum(right[1:], as_jet(iv.mpf(0), dimension)) * cut["state"]
    return partition * jet_exp(-linear)


def mixture_jet(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> Jet:
    point = face_jets(box, variables, dependent)
    dimension = len(variables)
    entropy = as_jet(iv.mpf(0), dimension)
    for value in point[0:5]:
        entropy -= value * jet_log(value)
    entropy_factor = jet_exp(entropy * (iv.mpf(1) / 3))
    weights = []
    experts = []
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = as_jet(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"), dimension
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_jet(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_jet(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        weights.append(weight)
        experts.append(entropy_factor * first * second)
    if not experts:
        raise ValueError("affine gate must contain at least one expert")

    weight_lower = [np_next_down(lower_float(weight.value)) for weight in weights]
    weight_upper = [np_next_up(upper_float(weight.value)) for weight in weights]

    def convex_interval(values: list) -> Any:
        lower = bounded_simplex_linear_extreme(
            weight_lower,
            weight_upper,
            [np_next_down(lower_float(value)) for value in values],
            maximize=False,
        )
        upper = bounded_simplex_linear_extreme(
            weight_lower,
            weight_upper,
            [np_next_up(upper_float(value)) for value in values],
            maximize=True,
        )
        return iv.mpf([str(np_next_down(lower)), str(np_next_up(upper))])

    value = convex_interval([expert.value for expert in experts])
    gradient = []
    for coordinate in range(dimension):
        weighted = convex_interval(
            [expert.gradient[coordinate] for expert in experts]
        )
        weight_derivative = sum(
            (
                weight.gradient[coordinate] * expert.value
                for weight, expert in zip(weights, experts, strict=True)
            ),
            iv.mpf(0),
        )
        gradient.append(weighted + weight_derivative)

    hessian = []
    for row in range(dimension):
        hessian_row = []
        for column in range(dimension):
            weighted = convex_interval(
                [expert.hessian[row][column] for expert in experts]
            )
            weight_derivative = sum(
                (
                    weight.gradient[row] * expert.gradient[column]
                    + weight.gradient[column] * expert.gradient[row]
                    for weight, expert in zip(weights, experts, strict=True)
                ),
                iv.mpf(0),
            )
            hessian_row.append(weighted + weight_derivative)
        hessian.append(hessian_row)
    return Jet(value, gradient, hessian)


def mixture_jet_direct(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> Jet:
    """Evaluate the affine mixture and two derivatives without LP relaxation."""

    point = face_jets(box, variables, dependent)
    dimension = len(variables)
    entropy = as_jet(iv.mpf(0), dimension)
    for value in point[0:5]:
        entropy -= value * jet_log(value)
    entropy_factor = jet_exp(entropy * (iv.mpf(1) / 3))
    mixture = as_jet(iv.mpf(0), dimension)
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = as_jet(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"), dimension
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_jet(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_jet(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        mixture += weight * entropy_factor * first * second
    return mixture


def verify_affine_partition_identity(coefficients: list[list[Fraction]]) -> None:
    if not coefficients:
        raise ValueError("affine partition must contain at least one expert")
    width = len(coefficients[0])
    if any(len(row) != width for row in coefficients):
        raise ValueError("affine partition rows have inconsistent widths")
    expected = [ONE, *([ZERO] * (width - 1))]
    actual = [
        sum((row[column] for row in coefficients), ZERO)
        for column in range(width)
    ]
    if actual != expected:
        raise ValueError("affine expert weights do not sum identically to one")


def stage_factor_third(
    left: list[ThirdJet], right: list[ThirdJet], cut: dict[str, Any]
) -> ThirdJet:
    dimension = len(left[0].gradient)
    partition = as_third_jet(iv.mpf(0), dimension)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = as_third_jet(iv.mpf(0), dimension)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(left[1:], as_third_jet(iv.mpf(0), dimension)) * cut["increment"]
    linear += sum(right[1:], as_third_jet(iv.mpf(0), dimension)) * cut["state"]
    return partition * third_exp(-linear)


def mixture_third_jet(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> ThirdJet:
    point = face_third_jets(box, variables, dependent)
    dimension = len(variables)
    entropy = as_third_jet(iv.mpf(0), dimension)
    for value in point[0:5]:
        entropy -= value * third_log(value)
    entropy_factor = third_exp(entropy * (iv.mpf(1) / 3))
    weights = []
    experts = []
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = as_third_jet(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"), dimension
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_third(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_third(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        weights.append(weight)
        experts.append(entropy_factor * first * second)
    if not experts:
        raise ValueError("affine gate must contain at least one expert")

    weight_lower = [np_next_down(lower_float(weight.value)) for weight in weights]
    weight_upper = [np_next_up(upper_float(weight.value)) for weight in weights]

    def convex_interval(values: list) -> Any:
        lower = bounded_simplex_linear_extreme(
            weight_lower,
            weight_upper,
            [np_next_down(lower_float(value)) for value in values],
            maximize=False,
        )
        upper = bounded_simplex_linear_extreme(
            weight_lower,
            weight_upper,
            [np_next_up(upper_float(value)) for value in values],
            maximize=True,
        )
        return iv.mpf([str(np_next_down(lower)), str(np_next_up(upper))])

    value = convex_interval([expert.value for expert in experts])
    gradient = []
    for i in range(dimension):
        gradient.append(
            convex_interval([expert.gradient[i] for expert in experts])
            + sum(
                (
                    weight.gradient[i] * expert.value
                    for weight, expert in zip(weights, experts, strict=True)
                ),
                iv.mpf(0),
            )
        )
    hessian = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            row.append(
                convex_interval([expert.hessian[i][j] for expert in experts])
                + sum(
                    (
                        weight.gradient[i] * expert.gradient[j]
                        + weight.gradient[j] * expert.gradient[i]
                        for weight, expert in zip(weights, experts, strict=True)
                    ),
                    iv.mpf(0),
                )
            )
        hessian.append(row)
    third = []
    for i in range(dimension):
        plane = []
        for j in range(dimension):
            line = []
            for k in range(dimension):
                line.append(
                    convex_interval(
                        [expert.third[i][j][k] for expert in experts]
                    )
                    + sum(
                        (
                            weight.gradient[i] * expert.hessian[j][k]
                            + weight.gradient[j] * expert.hessian[i][k]
                            + weight.gradient[k] * expert.hessian[i][j]
                            for weight, expert in zip(weights, experts, strict=True)
                        ),
                        iv.mpf(0),
                    )
                )
            plane.append(line)
        third.append(plane)
    return ThirdJet(value, gradient, hessian, third)


def affine_third_constant(
    value, dimension: int, form_type=AffineInterval
) -> ThirdJet:
    zero = form_type.constant(iv.mpf(0), dimension)
    return ThirdJet(
        form_type.constant(value, dimension),
        [zero for _ in range(dimension)],
        [[zero for _ in range(dimension)] for _ in range(dimension)],
        [
            [[zero for _ in range(dimension)] for _ in range(dimension)]
            for _ in range(dimension)
        ],
    )


def face_affine_third_jets(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    form_type=AffineInterval,
) -> list[ThirdJet]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    values: list[ThirdJet | None] = [None] * 15
    zero = form_type.constant(iv.mpf(0), dimension)
    one = form_type.constant(iv.mpf(1), dimension)
    for index in range(15):
        if index in dependent:
            continue
        center = (box.lower[index] + box.upper[index]) / 2
        radius = (box.upper[index] - box.lower[index]) / 2
        if index in positions:
            scalar = form_type.variable(
                iv.mpf(f"{center.numerator}/{center.denominator}"),
                iv.mpf(f"{radius.numerator}/{radius.denominator}"),
                positions[index],
                dimension,
            )
        else:
            scalar = form_type.constant(
                iv.mpf(f"{center.numerator}/{center.denominator}"), dimension
            )
        gradient = [zero for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = one
        values[index] = ThirdJet(
            scalar,
            gradient,
            [[zero for _ in range(dimension)] for _ in range(dimension)],
            [
                [[zero for _ in range(dimension)] for _ in range(dimension)]
                for _ in range(dimension)
            ],
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = affine_third_constant(iv.mpf(1), dimension, form_type)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def affine_dual_constant(value, dimension: int, form_type=AffineInterval) -> Dual:
    zero = form_type.constant(iv.mpf(0), dimension)
    return Dual(
        form_type.constant(value, dimension),
        [zero for _ in range(dimension)],
    )


def face_affine_duals(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    form_type=AffineInterval,
) -> list[Dual]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    values: list[Dual | None] = [None] * 15
    zero = form_type.constant(iv.mpf(0), dimension)
    one = form_type.constant(iv.mpf(1), dimension)
    for index in range(15):
        if index in dependent:
            continue
        center = (box.lower[index] + box.upper[index]) / 2
        radius = (box.upper[index] - box.lower[index]) / 2
        scalar = (
            form_type.variable(
                iv.mpf(f"{center.numerator}/{center.denominator}"),
                iv.mpf(f"{radius.numerator}/{radius.denominator}"),
                positions[index],
                dimension,
            )
            if index in positions
            else form_type.constant(
                iv.mpf(f"{center.numerator}/{center.denominator}"), dimension
            )
        )
        gradient = [zero for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = one
        values[index] = Dual(scalar, gradient)
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = affine_dual_constant(iv.mpf(1), dimension, form_type)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def stage_factor_affine_dual(
    left: list[Dual],
    right: list[Dual],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> Dual:
    dimension = len(left[0].gradient)
    partition = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_dual_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_dual_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition * dual_exp(-linear)


def stage_log_components_affine_dual(
    left: list[Dual],
    right: list[Dual],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> tuple[Dual, Dual]:
    dimension = len(left[0].gradient)
    partition = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_dual_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_dual_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition, linear


def stage_log_from_components(components: tuple[Dual, Dual]) -> Dual:
    partition, linear = components
    return dual_log(partition) - linear


def stage_relative_log_from_components(
    components: tuple[Dual, Dual],
    reference: tuple[Dual, Dual],
) -> Dual:
    partition, linear = components
    reference_partition, reference_linear = reference
    dimension = len(partition.gradient)
    zero = partition.value.coerce(0)
    one = Dual(
        partition.value.coerce(1), [zero for _ in range(dimension)]
    )
    relative_partition = (partition - reference_partition) / reference_partition
    return dual_log(one + relative_partition) - (linear - reference_linear)


def stage_log_affine_dual(
    left: list[Dual],
    right: list[Dual],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> Dual:
    return stage_log_from_components(
        stage_log_components_affine_dual(left, right, cut, form_type)
    )


def relative_mixture_affine_dual(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> tuple[Dual, Dual, list[Dual]]:
    """Return log-reference L0 and S with mixture = exp(L0) S."""

    point = face_affine_duals(box, variables, dependent, form_type)
    dimension = len(variables)
    entropy = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * dual_log(value)
    entropy *= iv.mpf(1) / 3
    weights = []
    logarithms = []
    pair_indices = []
    first_components: dict[int, tuple[Dual, Dual]] = {}
    second_components: dict[int, tuple[Dual, Dual]] = {}
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_dual_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_components:
            first_components[first_index] = stage_log_components_affine_dual(
                point[0:5], point[5:10], first_cuts[first_index], form_type
            )
        if second_index not in second_components:
            second_components[second_index] = stage_log_components_affine_dual(
                point[5:10], point[10:15], second_cuts[second_index], form_type
            )
        first = stage_log_from_components(first_components[first_index])
        second = stage_log_from_components(second_components[second_index])
        weights.append(weight)
        logarithms.append(entropy + first + second)
        pair_indices.append((first_index, second_index))
    if not logarithms:
        raise ValueError("affine gate must contain at least one expert")
    radii = [
        float((box.upper[coordinate] - box.lower[coordinate]) / 2)
        for coordinate in variables
    ]

    def midpoint(form) -> float:
        if isinstance(form, FastAffine):
            return form.center
        if isinstance(form, AffineInterval):
            return (lower_float(form.center) + upper_float(form.center)) / 2.0
        return (lower_float(form) + upper_float(form)) / 2.0

    vectors = [
        [midpoint(logarithm.value)]
        + [
            midpoint(gradient) * radius
            for gradient, radius in zip(logarithm.gradient, radii, strict=True)
        ]
        for logarithm in logarithms
    ]
    reference_index = min(
        range(len(logarithms)),
        key=lambda candidate: max(
            sum(
                (left - right) ** 2
                for left, right in zip(
                    vectors[candidate], vector, strict=True
                )
            )
            for vector in vectors
        ),
    )
    reference = logarithms[reference_index]
    reference_first, reference_second = pair_indices[reference_index]
    one = affine_dual_constant(iv.mpf(1), dimension, form_type)
    zero = affine_dual_constant(iv.mpf(0), dimension, form_type)
    relative_mixture = one
    for weight, (first_index, second_index) in zip(
        weights, pair_indices, strict=True
    ):
        relative_logarithm = zero
        if first_index != reference_first:
            relative_logarithm += (
                stage_log_from_components(first_components[first_index])
                - stage_log_from_components(first_components[reference_first])
            )
        if second_index != reference_second:
            relative_logarithm += (
                stage_log_from_components(second_components[second_index])
                - stage_log_from_components(second_components[reference_second])
            )
        relative_mixture += weight * (dual_exp(relative_logarithm) - one)
    return reference, relative_mixture, point


def mixture_affine_dual(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> tuple[Dual, list[Dual]]:
    point = face_affine_duals(box, variables, dependent, form_type)
    dimension = len(variables)
    entropy = affine_dual_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * dual_log(value)
    entropy_factor = dual_exp(entropy * (iv.mpf(1) / 3))
    weights = []
    experts = []
    first_factors: dict[int, Dual] = {}
    second_factors: dict[int, Dual] = {}
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_dual_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_factors:
            first_factors[first_index] = stage_factor_affine_dual(
                point[0:5], point[5:10], first_cuts[first_index], form_type
            )
        if second_index not in second_factors:
            second_factors[second_index] = stage_factor_affine_dual(
                point[5:10], point[10:15], second_cuts[second_index], form_type
            )
        first = first_factors[first_index]
        second = second_factors[second_index]
        weights.append(weight)
        experts.append(entropy_factor * first * second)
    if not experts:
        raise ValueError("affine gate must contain at least one expert")
    reference = experts[0]
    mixture = reference
    for weight, expert in zip(weights, experts, strict=True):
        mixture += weight * (expert - reference)
    return mixture, point


def mixture_log_affine_dual(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> tuple[Dual, list[Dual]]:
    mixture, point = mixture_affine_dual(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        form_type,
    )
    return dual_log(mixture), point


def affine_radial_form(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    target: list[Fraction],
    form_type=AffineInterval,
):
    reference, relative_mixture, point = relative_mixture_affine_dual(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        form_type,
    )
    dimension = len(variables)
    reference_radial = form_type.constant(iv.mpf(0), dimension)
    mixture_radial = form_type.constant(iv.mpf(0), dimension)
    for position, coordinate in enumerate(variables):
        target_value = target[coordinate]
        displacement = point[coordinate].value - form_type.constant(
            iv.mpf(f"{target_value.numerator}/{target_value.denominator}"),
            dimension,
        )
        reference_radial += displacement * reference.gradient[position]
        mixture_radial += displacement * relative_mixture.gradient[position]
    return relative_mixture.value * reference_radial + mixture_radial


def affine_jet_constant(value, dimension: int, form_type=AffineInterval) -> Jet:
    zero = form_type.constant(iv.mpf(0), dimension)
    return Jet(
        form_type.constant(value, dimension),
        [zero for _ in range(dimension)],
        [[zero for _ in range(dimension)] for _ in range(dimension)],
    )


def face_affine_jets(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    form_type=AffineInterval,
) -> list[Jet]:
    """Second-order jets whose scalars retain box-wide affine correlations."""

    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    values: list[Jet | None] = [None] * 15
    zero = form_type.constant(iv.mpf(0), dimension)
    one = form_type.constant(iv.mpf(1), dimension)
    for index in range(15):
        if index in dependent:
            continue
        center = (box.lower[index] + box.upper[index]) / 2
        radius = (box.upper[index] - box.lower[index]) / 2
        scalar = (
            form_type.variable(
                iv.mpf(f"{center.numerator}/{center.denominator}"),
                iv.mpf(f"{radius.numerator}/{radius.denominator}"),
                positions[index],
                dimension,
            )
            if index in positions
            else form_type.constant(
                iv.mpf(f"{center.numerator}/{center.denominator}"), dimension
            )
        )
        gradient = [zero for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = one
        values[index] = Jet(
            scalar,
            gradient,
            [[zero for _ in range(dimension)] for _ in range(dimension)],
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = affine_jet_constant(iv.mpf(1), dimension, form_type)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def stage_factor_affine_jet(
    left: list[Jet],
    right: list[Jet],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> Jet:
    dimension = len(left[0].gradient)
    partition = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_jet_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_jet_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition * jet_exp(-linear)


def stage_log_components_affine_jet(
    left: list[Jet],
    right: list[Jet],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> tuple[Jet, Jet]:
    dimension = len(left[0].gradient)
    partition = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_jet_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_jet_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition, linear


def stage_log_from_jet_components(components: tuple[Jet, Jet]) -> Jet:
    partition, linear = components
    return jet_log(partition) - linear


def relative_log_mixture_affine_jet(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> Jet:
    """Return log mixture through order two using expert log-ratios."""

    point = face_affine_jets(box, variables, dependent, form_type)
    dimension = len(variables)
    entropy = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * jet_log(value)
    entropy *= iv.mpf(1) / 3
    weights: list[Jet] = []
    logarithms: list[Jet] = []
    pair_indices: list[tuple[int, int]] = []
    first_components: dict[int, tuple[Jet, Jet]] = {}
    second_components: dict[int, tuple[Jet, Jet]] = {}
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_jet_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_components:
            first_components[first_index] = stage_log_components_affine_jet(
                point[0:5], point[5:10], first_cuts[first_index], form_type
            )
        if second_index not in second_components:
            second_components[second_index] = stage_log_components_affine_jet(
                point[5:10], point[10:15], second_cuts[second_index], form_type
            )
        first = stage_log_from_jet_components(first_components[first_index])
        second = stage_log_from_jet_components(second_components[second_index])
        weights.append(weight)
        logarithms.append(entropy + first + second)
        pair_indices.append((first_index, second_index))
    if not logarithms:
        raise ValueError("affine gate must contain at least one expert")

    radii = [
        float((box.upper[coordinate] - box.lower[coordinate]) / 2)
        for coordinate in variables
    ]

    def midpoint(form) -> float:
        if isinstance(form, FastAffine):
            return form.center
        return (lower_float(form.center) + upper_float(form.center)) / 2.0

    vectors = [
        [midpoint(logarithm.value)]
        + [
            midpoint(gradient) * radius
            for gradient, radius in zip(logarithm.gradient, radii, strict=True)
        ]
        for logarithm in logarithms
    ]
    reference_index = min(
        range(len(logarithms)),
        key=lambda candidate: max(
            sum(
                (left - right) ** 2
                for left, right in zip(vectors[candidate], vector, strict=True)
            )
            for vector in vectors
        ),
    )
    reference = logarithms[reference_index]
    reference_first, reference_second = pair_indices[reference_index]
    one = affine_jet_constant(iv.mpf(1), dimension, form_type)
    zero = affine_jet_constant(iv.mpf(0), dimension, form_type)
    relative_mixture = one
    for weight, (first_index, second_index) in zip(
        weights, pair_indices, strict=True
    ):
        relative_logarithm = zero
        if first_index != reference_first:
            relative_logarithm += (
                stage_log_from_jet_components(first_components[first_index])
                - stage_log_from_jet_components(first_components[reference_first])
            )
        if second_index != reference_second:
            relative_logarithm += (
                stage_log_from_jet_components(second_components[second_index])
                - stage_log_from_jet_components(second_components[reference_second])
            )
        relative_mixture += weight * (jet_exp(relative_logarithm) - one)
    return reference + jet_log(relative_mixture)


def affine_directional_constant(
    value, dimension: int, form_type=AffineInterval
) -> DirectionalDual:
    return DirectionalDual(
        affine_dual_constant(value, dimension, form_type),
        affine_dual_constant(iv.mpf(0), dimension, form_type),
    )


def face_affine_directional_duals(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    form_type=AffineInterval,
) -> list[DirectionalDual]:
    base = face_affine_duals(box, variables, dependent, form_type)
    if derivative_coordinate in dependent:
        raise ValueError("the selected derivative coordinate cannot be dependent")
    if not 0 <= derivative_coordinate < 15:
        raise ValueError("the selected derivative coordinate is out of range")
    dimension = len(variables)
    zero = form_type.constant(iv.mpf(0), dimension)
    one = form_type.constant(iv.mpf(1), dimension)
    minus_one = form_type.constant(iv.mpf(-1), dimension)
    block_start = 5 * (derivative_coordinate // 5)
    block_dependent = next(
        index for index in dependent if block_start <= index < block_start + 5
    )
    return [
        DirectionalDual(
            value,
            Dual(
                one
                if index == derivative_coordinate
                else minus_one
                if index == block_dependent
                else zero,
                [zero for _ in range(dimension)],
            ),
        )
        for index, value in enumerate(base)
    ]


def stage_log_components_affine_directional(
    left: list[DirectionalDual],
    right: list[DirectionalDual],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> tuple[DirectionalDual, DirectionalDual]:
    dimension = len(left[0].base.gradient)
    partition = affine_directional_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_directional_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_directional_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_directional_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition, linear


def stage_log_from_directional_components(
    components: tuple[DirectionalDual, DirectionalDual]
) -> DirectionalDual:
    partition, linear = components
    return directional_log(partition) - linear


def relative_log_mixture_affine_directional(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> DirectionalDual:
    """Return a relative log mixture with one selected affine Hessian row."""

    point = face_affine_directional_duals(
        box, variables, dependent, derivative_coordinate, form_type
    )
    dimension = len(variables)
    entropy = affine_directional_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * directional_log(value)
    entropy *= iv.mpf(1) / 3
    weights: list[DirectionalDual] = []
    logarithms: list[DirectionalDual] = []
    pair_indices: list[tuple[int, int]] = []
    first_components: dict[
        int, tuple[DirectionalDual, DirectionalDual]
    ] = {}
    second_components: dict[
        int, tuple[DirectionalDual, DirectionalDual]
    ] = {}
    first_logs: dict[int, DirectionalDual] = {}
    second_logs: dict[int, DirectionalDual] = {}
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_directional_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_components:
            first_components[first_index] = stage_log_components_affine_directional(
                point[0:5], point[5:10], first_cuts[first_index], form_type
            )
        if second_index not in second_components:
            second_components[second_index] = stage_log_components_affine_directional(
                point[5:10], point[10:15], second_cuts[second_index], form_type
            )
        if first_index not in first_logs:
            first_logs[first_index] = stage_log_from_directional_components(
                first_components[first_index]
            )
        if second_index not in second_logs:
            second_logs[second_index] = stage_log_from_directional_components(
                second_components[second_index]
            )
        first = first_logs[first_index]
        second = second_logs[second_index]
        weights.append(weight)
        logarithms.append(entropy + first + second)
        pair_indices.append((first_index, second_index))
    if not logarithms:
        raise ValueError("affine gate must contain at least one expert")

    radii = [
        float((box.upper[coordinate] - box.lower[coordinate]) / 2)
        for coordinate in variables
    ]

    def midpoint(form) -> float:
        if isinstance(form, FastAffine):
            return form.center
        return (lower_float(form.center) + upper_float(form.center)) / 2.0

    vectors = [
        [midpoint(logarithm.base.value)]
        + [
            midpoint(gradient) * radius
            for gradient, radius in zip(
                logarithm.base.gradient, radii, strict=True
            )
        ]
        for logarithm in logarithms
    ]
    reference_index = min(
        range(len(logarithms)),
        key=lambda candidate: max(
            sum(
                (left - right) ** 2
                for left, right in zip(vectors[candidate], vector, strict=True)
            )
            for vector in vectors
        ),
    )
    reference = logarithms[reference_index]
    reference_first, reference_second = pair_indices[reference_index]
    one = affine_directional_constant(iv.mpf(1), dimension, form_type)
    zero = affine_directional_constant(iv.mpf(0), dimension, form_type)
    relative_mixture = one
    for weight, (first_index, second_index) in zip(
        weights, pair_indices, strict=True
    ):
        relative_logarithm = zero
        if first_index != reference_first:
            relative_logarithm += first_logs[first_index] - first_logs[reference_first]
        if second_index != reference_second:
            relative_logarithm += (
                second_logs[second_index] - second_logs[reference_second]
            )
        relative_mixture += weight * (
            directional_exp(relative_logarithm) - one
        )
    return reference + directional_log(relative_mixture)


def mixture_log_affine_jet(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> Jet:
    point = face_affine_jets(box, variables, dependent, form_type)
    dimension = len(variables)
    entropy = affine_jet_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * jet_log(value)
    entropy_factor = jet_exp(entropy * (iv.mpf(1) / 3))
    mixture = affine_jet_constant(iv.mpf(0), dimension, form_type)
    first_factors: dict[int, Jet] = {}
    second_factors: dict[int, Jet] = {}
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_jet_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first_index = int(pair["first_cut"])
        second_index = int(pair["second_cut"])
        if first_index not in first_factors:
            first_factors[first_index] = stage_factor_affine_jet(
                point[0:5], point[5:10], first_cuts[first_index], form_type
            )
        if second_index not in second_factors:
            second_factors[second_index] = stage_factor_affine_jet(
                point[5:10], point[10:15], second_cuts[second_index], form_type
            )
        first = first_factors[first_index]
        second = second_factors[second_index]
        mixture += weight * entropy_factor * first * second
    return jet_log(mixture)


def stage_factor_affine_third(
    left: list[ThirdJet],
    right: list[ThirdJet],
    cut: dict[str, Any],
    form_type=AffineInterval,
) -> ThirdJet:
    dimension = len(left[0].gradient)
    partition = affine_third_constant(iv.mpf(0), dimension, form_type)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * cut["kernel"][difference][source]
                * right[source]
            )
    linear = affine_third_constant(iv.mpf(0), dimension, form_type)
    for index in range(5):
        linear += right[index] * cut["target"][index]
    linear += sum(
        left[1:], affine_third_constant(iv.mpf(0), dimension, form_type)
    ) * cut["increment"]
    linear += sum(
        right[1:], affine_third_constant(iv.mpf(0), dimension, form_type)
    ) * cut["state"]
    return partition * third_exp(-linear)


def mixture_log_affine_third(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> ThirdJet:
    point = face_affine_third_jets(box, variables, dependent, form_type)
    dimension = len(variables)
    entropy = affine_third_constant(iv.mpf(0), dimension, form_type)
    for value in point[0:5]:
        entropy -= value * third_log(value)
    entropy_factor = third_exp(entropy * (iv.mpf(1) / 3))
    mixture = affine_third_constant(iv.mpf(0), dimension, form_type)
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = affine_third_constant(
            iv.mpf(f"{row[0].numerator}/{row[0].denominator}"),
            dimension,
            form_type,
        )
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * iv.mpf(
                f"{coefficient.numerator}/{coefficient.denominator}"
            )
        first = stage_factor_affine_third(
            point[0:5],
            point[5:10],
            first_cuts[int(pair["first_cut"])],
            form_type,
        )
        second = stage_factor_affine_third(
            point[5:10],
            point[10:15],
            second_cuts[int(pair["second_cut"])],
            form_type,
        )
        mixture += weight * entropy_factor * first * second
    return third_log(mixture)


def face_symmetric_third_jets(
    box: FaceBox, variables: list[int], dependent: list[int]
) -> list[SymmetricThirdJet]:
    dimension = len(variables)
    positions = {coordinate: position for position, coordinate in enumerate(variables)}
    values: list[SymmetricThirdJet | None] = [None] * 15
    zero = FastAffine.constant(0.0, dimension)
    one = FastAffine.constant(1.0, dimension)
    triples = symmetric_triples(dimension)
    for index in range(15):
        if index in dependent:
            continue
        center = float((box.lower[index] + box.upper[index]) / 2)
        radius = float((box.upper[index] - box.lower[index]) / 2)
        scalar = (
            FastAffine.variable(center, radius, positions[index], dimension)
            if index in positions
            else FastAffine.constant(center, dimension)
        )
        gradient = [zero for _ in range(dimension)]
        if index in positions:
            gradient[positions[index]] = one
        values[index] = SymmetricThirdJet(
            scalar,
            gradient,
            [[zero for _ in range(dimension)] for _ in range(dimension)],
            [zero for _ in triples],
            triples,
        )
    for start in (0, 5, 10):
        index = next(value for value in dependent if start <= value < start + 5)
        derived = symmetric_third_constant(1.0, dimension)
        for other in range(start, start + 5):
            if other != index:
                derived -= values[other]
        values[index] = derived
    return [value for value in values if value is not None]


def stage_factor_symmetric_third(
    left: list[SymmetricThirdJet],
    right: list[SymmetricThirdJet],
    cut: dict[str, Any],
) -> SymmetricThirdJet:
    dimension = len(left[0].gradient)
    partition = symmetric_third_constant(0.0, dimension)
    for difference in range(5):
        for source in range(5):
            partition += (
                left[difference]
                * FastAffine.constant(cut["kernel"][difference][source], dimension)
                * right[source]
            )
    linear = symmetric_third_constant(0.0, dimension)
    for index in range(5):
        linear += right[index] * FastAffine.constant(cut["target"][index], dimension)
    linear += sum(left[1:], symmetric_third_constant(0.0, dimension)) * FastAffine.constant(
        cut["increment"], dimension
    )
    linear += sum(right[1:], symmetric_third_constant(0.0, dimension)) * FastAffine.constant(
        cut["state"], dimension
    )
    return partition * symmetric_third_exp(-linear)


def mixture_log_symmetric_third(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> SymmetricThirdJet:
    point = face_symmetric_third_jets(box, variables, dependent)
    dimension = len(variables)
    entropy = symmetric_third_constant(0.0, dimension)
    for value in point[0:5]:
        entropy -= value * symmetric_third_log(value)
    entropy_factor = symmetric_third_exp(entropy * FastAffine.constant(1.0 / 3.0, dimension))
    mixture = symmetric_third_constant(0.0, dimension)
    for row, pair in zip(coefficients, pairs, strict=True):
        weight = symmetric_third_constant(float(row[0]), dimension)
        for coefficient, coordinate in zip(row[1:], coordinates, strict=True):
            weight += point[coordinate] * FastAffine.constant(float(coefficient), dimension)
        first = stage_factor_symmetric_third(
            point[0:5], point[5:10], first_cuts[int(pair["first_cut"])]
        )
        second = stage_factor_symmetric_third(
            point[5:10], point[10:15], second_cuts[int(pair["second_cut"])]
        )
        mixture += weight * entropy_factor * first * second
    return symmetric_third_log(mixture)


def mean_value_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    center_lower = list(box.lower)
    center_upper = list(box.upper)
    radii = []
    for coordinate in variables:
        center = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center
        center_upper[coordinate] = center
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center_box = FaceBox(tuple(center_lower), tuple(center_upper))
    center_value = mixture_dual(
        center_box, variables, dependent, coefficients, coordinates, pairs,
        first_cuts, second_cuts,
    ).value
    enclosure = mixture_dual(
        box, variables, dependent, coefficients, coordinates, pairs,
        first_cuts, second_cuts,
    )
    bound = center_value
    contributions = []
    for gradient, radius in zip(enclosure.gradient, radii, strict=True):
        contribution = abs(gradient) * iv.mpf(
            f"{radius.numerator}/{radius.denominator}"
        )
        bound += contribution
        contributions.append(upper_float(contribution))
    return upper_float(bound), contributions


def taylor_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    center_lower = list(box.lower)
    center_upper = list(box.upper)
    radii = []
    for coordinate in variables:
        center = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center
        center_upper[coordinate] = center
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center = mixture_jet(
        FaceBox(tuple(center_lower), tuple(center_upper)),
        variables, dependent, coefficients, coordinates, pairs, first_cuts, second_cuts,
    )
    enclosure = mixture_jet(
        box, variables, dependent, coefficients, coordinates, pairs,
        first_cuts, second_cuts,
    )
    bound = center.value
    contributions = [iv.mpf(0) for _ in variables]
    radius_intervals = [
        iv.mpf(f"{radius.numerator}/{radius.denominator}") for radius in radii
    ]
    for row in range(len(variables)):
        linear = abs(center.gradient[row]) * radius_intervals[row]
        bound += linear
        contributions[row] += linear
        for column in range(len(variables)):
            quadratic = (
                abs(enclosure.hessian[row][column])
                * radius_intervals[row]
                * radius_intervals[column]
                / 2
            )
            bound += quadratic
            contributions[row] += quadratic
    return upper_float(bound), [upper_float(value) for value in contributions]


def directional_derivative_negative_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Upper-bound minus one face-coordinate derivative by centered mean value.

    A face coordinate represents the tangent direction obtained by increasing
    that coordinate and decreasing the dependent coordinate in the same
    simplex block.  Thus, for example, coordinate 0 with dependent coordinate
    3 is exactly the q0-minus-q3 derivative used by the message-monotonicity
    reduction.
    """

    if derivative_coordinate not in variables:
        raise ValueError("the derivative coordinate must be a free face variable")
    position = variables.index(derivative_coordinate)
    center_lower = list(box.lower)
    center_upper = list(box.upper)
    radii = []
    for coordinate in variables:
        center = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center
        center_upper[coordinate] = center
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center = mixture_directional_dual(
        FaceBox(tuple(center_lower), tuple(center_upper)),
        variables,
        dependent,
        derivative_coordinate,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
    )
    enclosure = mixture_directional_dual(
        box,
        variables,
        dependent,
        derivative_coordinate,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
    )
    # If D denotes the requested derivative, then
    #   -D(x) <= -D(center) + sum_j sup|d_j D| radius_j.
    # Every operation below is directed interval arithmetic.
    bound = -center.directional.value
    contributions = []
    for hessian_entry, radius in zip(
        enclosure.directional.gradient, radii, strict=True
    ):
        contribution = abs(hessian_entry) * iv.mpf(
            f"{radius.numerator}/{radius.denominator}"
        )
        bound += contribution
        contributions.append(upper_float(contribution))
    return upper_float(bound), contributions


def directional_derivative_natural_negative_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Naturally enclose minus one combined face-coordinate derivative."""

    enclosure = mixture_directional_dual(
        box,
        variables,
        dependent,
        derivative_coordinate,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
    )
    radii = [
        (box.upper[coordinate] - box.lower[coordinate]) / 2
        for coordinate in variables
    ]
    contributions = [
        upper_float(
            abs(hessian_entry)
            * iv.mpf(f"{radius.numerator}/{radius.denominator}")
        )
        for hessian_entry, radius in zip(
            enclosure.directional.gradient, radii, strict=True
        )
    ]
    return upper_float(-enclosure.directional.value), contributions


def directional_derivative_affine_negative_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Affine-form enclosure of minus one combined face derivative."""

    if derivative_coordinate not in variables:
        raise ValueError("the derivative coordinate must be a free face variable")
    position = variables.index(derivative_coordinate)
    mixture, _ = mixture_affine_dual(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        AffineInterval,
    )
    derivative_form = mixture.gradient[position]
    derivative_interval = derivative_form.interval()
    shared_error = upper_float(derivative_form.error) / max(len(variables), 1)
    contributions = [
        upper_float(abs(value)) + shared_error
        for value in derivative_form.coefficients
    ]
    return -lower_float(derivative_interval), contributions


def directional_derivative_fast_affine_negative_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Floating affine-form proposal for minus one combined derivative."""

    if derivative_coordinate not in variables:
        raise ValueError("the derivative coordinate must be a free face variable")
    position = variables.index(derivative_coordinate)
    mixture, _ = mixture_affine_dual(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        FastAffine,
    )
    derivative_form = mixture.gradient[position]
    shared_error = derivative_form.error / max(len(variables), 1)
    contributions = [
        abs(value) + shared_error for value in derivative_form.coefficients
    ]
    return -derivative_form.lower(), contributions


def directional_log_derivative_affine_negative_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
    expected_sign: str = "positive",
) -> tuple[float, list[float]]:
    """Affine-form upper bound on violation of a selected derivative sign."""

    if derivative_coordinate in dependent:
        raise ValueError("the selected derivative coordinate cannot be dependent")
    log_mixture = relative_log_mixture_affine_directional(
        box,
        variables,
        dependent,
        derivative_coordinate,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        form_type,
    )
    derivative_form = log_mixture.directional.value
    if form_type is FastAffine:
        shared_error = derivative_form.error / max(len(variables), 1)
        contributions = [
            abs(value) + shared_error for value in derivative_form.coefficients
        ]
        if expected_sign == "positive":
            return -derivative_form.lower(), contributions
        if expected_sign == "negative":
            return derivative_form.upper(), contributions
        raise ValueError("expected derivative sign must be positive or negative")
    derivative_interval = derivative_form.interval()
    shared_error = upper_float(derivative_form.error) / max(len(variables), 1)
    contributions = [
        upper_float(abs(value)) + shared_error
        for value in derivative_form.coefficients
    ]
    if expected_sign == "positive":
        return -lower_float(derivative_interval), contributions
    if expected_sign == "negative":
        return upper_float(derivative_interval), contributions
    raise ValueError("expected derivative sign must be positive or negative")


def log_hessian_sign_violation_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    derivative_coordinate: int,
    hessian_coordinate: int,
    expected_sign: str,
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    form_type=AffineInterval,
) -> tuple[float, list[float]]:
    """Upper-bound the violation of one expected log-Hessian sign."""

    if derivative_coordinate in dependent or hessian_coordinate not in variables:
        raise ValueError(
            "the selected derivative must be explicit and the Hessian coordinate free"
        )
    second_position = variables.index(hessian_coordinate)
    log_mixture = relative_log_mixture_affine_directional(
        box,
        variables,
        dependent,
        derivative_coordinate,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        form_type,
    )
    form = log_mixture.directional.gradient[second_position]
    if form_type is FastAffine:
        shared_error = form.error / max(len(variables), 1)
        contributions = [abs(value) + shared_error for value in form.coefficients]
        if expected_sign == "negative":
            return form.upper(), contributions
        return -form.lower(), contributions
    value_interval = form.interval()
    shared_error = upper_float(form.error) / max(len(variables), 1)
    contributions = [
        upper_float(abs(value)) + shared_error for value in form.coefficients
    ]
    if expected_sign == "negative":
        return upper_float(value_interval), contributions
    return -lower_float(value_interval), contributions


def log_concavity_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Hybrid prototype; the spectral step still needs exact replay."""

    import numpy as np

    center_lower = list(box.lower)
    center_upper = list(box.upper)
    radii = []
    for coordinate in variables:
        center = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center
        center_upper[coordinate] = center
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center = jet_log(
        mixture_jet(
            FaceBox(tuple(center_lower), tuple(center_upper)),
            variables, dependent, coefficients, coordinates, pairs,
            first_cuts, second_cuts,
        )
    )
    raw_enclosure = mixture_jet(
        box, variables, dependent, coefficients, coordinates, pairs,
        first_cuts, second_cuts,
    )
    positive_lower = lower_float(raw_enclosure.value)
    positive_upper = upper_float(raw_enclosure.value)
    if positive_lower <= 0.0:
        raise ValueError("structured mixture enclosure lost positivity")
    enclosure = jet_log(raw_enclosure)
    LOG_CONCAVITY_STATS["minimum_mixture_lower"] = min(
        LOG_CONCAVITY_STATS["minimum_mixture_lower"], positive_lower
    )
    LOG_CONCAVITY_STATS["maximum_mixture_upper"] = max(
        LOG_CONCAVITY_STATS["maximum_mixture_upper"], positive_upper
    )
    LOG_CONCAVITY_STATS["maximum_raw_hessian_magnitude"] = max(
        LOG_CONCAVITY_STATS["maximum_raw_hessian_magnitude"],
        max(
            upper_float(abs(value))
            for row in raw_enclosure.hessian
            for value in row
        ),
    )
    dimension = len(variables)
    midpoint = np.empty((dimension, dimension))
    error = np.empty((dimension, dimension))
    for row in range(dimension):
        for column in range(dimension):
            lower = lower_float(enclosure.hessian[row][column])
            upper = upper_float(enclosure.hessian[row][column])
            midpoint[row, column] = (lower + upper) / 2
            error[row, column] = (upper - lower) / 2
    midpoint = (midpoint + midpoint.T) / 2
    error = np.maximum(error, error.T)
    spectral_error = float(np.max(np.sum(error, axis=1)))
    lambda_upper = (
        float(np.linalg.eigvalsh(midpoint)[-1]) + spectral_error + 1e-12
    )
    LOG_CONCAVITY_STATS["evaluations"] += 1
    LOG_CONCAVITY_STATS["minimum_lambda_upper"] = min(
        LOG_CONCAVITY_STATS["minimum_lambda_upper"], lambda_upper
    )
    LOG_CONCAVITY_STATS["maximum_lambda_upper"] = max(
        LOG_CONCAVITY_STATS["maximum_lambda_upper"], lambda_upper
    )
    radius_floats = [float(value) for value in radii]
    if lambda_upper < 0.0:
        LOG_CONCAVITY_STATS["negative_curvature"] += 1
        gradient_bounds = [upper_float(abs(value)) for value in center.gradient]
        contributions = []
        for gradient, radius in zip(
            gradient_bounds, radius_floats, strict=True
        ):
            displacement = min(radius, gradient / (-lambda_upper))
            contributions.append(
                gradient * displacement
                + lambda_upper * displacement * displacement / 2.0
            )
        upper = upper_float(center.value) + sum(contributions) + 1e-14
        return upper, contributions

    LOG_CONCAVITY_STATS["fallback"] += 1
    bound = center.value
    contributions = [iv.mpf(0) for _ in variables]
    radius_intervals = [
        iv.mpf(f"{radius.numerator}/{radius.denominator}") for radius in radii
    ]
    for row in range(dimension):
        linear = abs(center.gradient[row]) * radius_intervals[row]
        bound += linear
        contributions[row] += linear
        for column in range(dimension):
            quadratic = (
                abs(enclosure.hessian[row][column])
                * radius_intervals[row]
                * radius_intervals[column]
                / 2
            )
            bound += quadratic
            contributions[row] += quadratic
    return upper_float(bound), [upper_float(value) for value in contributions]


def exact_centered_curvature_certificate(
    box: FaceBox,
    variables: list[int],
    center: Jet,
    hessian_enclosure: list[list],
    radii: list[Fraction],
) -> dict[str, Any] | None:
    """Certify strong concavity and the box maximum using exact rationals."""

    dimension = len(variables)
    interval_lower = [[ZERO for _ in range(dimension)] for _ in range(dimension)]
    interval_upper = [[ZERO for _ in range(dimension)] for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            lower_ij, upper_ij = outward_fraction_bounds(hessian_enclosure[i][j])
            lower_ji, upper_ji = outward_fraction_bounds(hessian_enclosure[j][i])
            interval_lower[i][j] = min(lower_ij, lower_ji)
            interval_upper[i][j] = max(upper_ij, upper_ji)
    midpoint = [
        [
            (interval_lower[i][j] + interval_upper[i][j]) / 2
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    error = [
        [
            (interval_upper[i][j] - interval_lower[i][j]) / 2
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    row_error = max(sum(row, ZERO) for row in error)

    curvature = ONE
    pivots = None
    for _ in range(128):
        negative_upper = [
            [
                -midpoint[i][j]
                - (row_error + curvature if i == j else ZERO)
                for j in range(dimension)
            ]
            for i in range(dimension)
        ]
        pivots = exact_ldl_pivots(negative_upper)
        if pivots is not None:
            break
        curvature /= 2
    if pivots is None:
        return None

    center_value_upper = outward_fraction_bounds(center.value)[1]
    gradient_upper = [
        outward_fraction_bounds(abs(value))[1] for value in center.gradient
    ]
    displacements = [
        min(radius, gradient / curvature)
        for radius, gradient in zip(radii, gradient_upper, strict=True)
    ]
    contributions = [
        gradient * displacement
        - curvature * displacement * displacement / 2
        for gradient, displacement in zip(
            gradient_upper, displacements, strict=True
        )
    ]
    total_upper = center_value_upper + sum(contributions, ZERO)
    return {
        "status": "exact rational spectral and quadratic replay over directed intervals",
        "variables": variables,
        "box_lower": [fraction_text(value) for value in box.lower],
        "box_upper": [fraction_text(value) for value in box.upper],
        "hessian_interval_lower": [
            [fraction_text(value) for value in row] for row in interval_lower
        ],
        "hessian_interval_upper": [
            [fraction_text(value) for value in row] for row in interval_upper
        ],
        "row_sum_error_bound": fraction_text(row_error),
        "certified_curvature": fraction_text(curvature),
        "ldl_pivots": [fraction_text(value) for value in pivots],
        "center_log_value_upper": fraction_text(center_value_upper),
        "gradient_absolute_upper": [
            fraction_text(value) for value in gradient_upper
        ],
        "radii": [fraction_text(value) for value in radii],
        "maximizing_displacements": [
            fraction_text(value) for value in displacements
        ],
        "quadratic_contributions": [
            fraction_text(value) for value in contributions
        ],
        "total_log_upper": fraction_text(total_upper),
        "closed": total_upper < ZERO,
    }


def fast_affine_hessian_curvature_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Floating proposal using the correlated affine Hessian enclosure directly.

    If H(epsilon) = C + sum_k A_k epsilon_k + R for epsilon in [-1,1]^d,
    convexity of lambda_max reduces the affine part to the 2^d vertices.  The
    residual matrix R is charged by its maximum absolute row sum.  This route
    deliberately remains proposal-only until the same data are emitted and
    replayed with directed intervals and exact rational LDL certificates.
    """

    import numpy as np

    enclosure = mixture_log_affine_jet(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
        FastAffine,
    )
    dimension = len(variables)
    center = np.zeros((dimension, dimension), dtype=float)
    affine_matrices = [
        np.zeros((dimension, dimension), dtype=float) for _ in range(dimension)
    ]
    residual = np.zeros((dimension, dimension), dtype=float)
    for i in range(dimension):
        for j in range(i, dimension):
            form = enclosure.hessian[i][j]
            if not isinstance(form, FastAffine):
                raise TypeError("fast affine Hessian lost its scalar form")
            center[i, j] = center[j, i] = form.center
            residual[i, j] = residual[j, i] = form.error
            for k in range(dimension):
                affine_matrices[k][i, j] = affine_matrices[k][j, i] = (
                    form.coefficients[k]
                )

    maximum_affine_eigenvalue = -math.inf
    for signs in itertools.product((-1.0, 1.0), repeat=dimension):
        candidate = center.copy()
        for sign, matrix in zip(signs, affine_matrices, strict=True):
            candidate += sign * matrix
        maximum_affine_eigenvalue = max(
            maximum_affine_eigenvalue,
            float(np.linalg.eigvalsh(candidate)[-1]),
        )
    residual_error = float(np.max(np.sum(np.abs(residual), axis=1)))
    lambda_upper = maximum_affine_eigenvalue + residual_error + 1e-12
    contributions = [
        float(np.max(np.sum(np.abs(matrix), axis=1)))
        for matrix in affine_matrices
    ]
    CENTERED_CURVATURE_STATS["evaluations"] += 1
    CENTERED_CURVATURE_STATS["minimum_lambda_upper"] = min(
        CENTERED_CURVATURE_STATS["minimum_lambda_upper"], lambda_upper
    )
    CENTERED_CURVATURE_STATS["maximum_lambda_upper"] = max(
        CENTERED_CURVATURE_STATS["maximum_lambda_upper"], lambda_upper
    )
    CENTERED_CURVATURE_STATS["maximum_affine_hessian_residual"] = max(
        CENTERED_CURVATURE_STATS["maximum_affine_hessian_residual"], residual_error
    )
    if lambda_upper < 0.0:
        CENTERED_CURVATURE_STATS["negative_curvature"] += 1
    else:
        CENTERED_CURVATURE_STATS["fallback"] += 1
    return lambda_upper, contributions


def fast_radial_taylor_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    target: list[Fraction],
) -> tuple[float, list[float], dict[str, float]]:
    """Floating centered Taylor proposal for (x-target) dot grad(log M)."""

    center_lower = list(box.lower)
    center_upper = list(box.upper)
    centers = []
    radii = []
    for coordinate in variables:
        center_value = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center_value
        center_upper[coordinate] = center_value
        centers.append(center_value)
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center = jet_log(
        mixture_jet(
            FaceBox(tuple(center_lower), tuple(center_upper)),
            variables,
            dependent,
            coefficients,
            coordinates,
            pairs,
            first_cuts,
            second_cuts,
        )
    )
    enclosure = mixture_log_symmetric_third(
        box,
        variables,
        dependent,
        coefficients,
        coordinates,
        pairs,
        first_cuts,
        second_cuts,
    )
    dimension = len(variables)
    delta_center = [
        iv.mpf(
            f"{(center_value - target[coordinate]).numerator}/"
            f"{(center_value - target[coordinate]).denominator}"
        )
        for center_value, coordinate in zip(centers, variables, strict=True)
    ]
    q_center = sum(
        (
            delta_center[k] * center.gradient[k]
            for k in range(dimension)
        ),
        iv.mpf(0),
    )
    q_gradient = [
        center.gradient[i]
        + sum(
            (
                delta_center[k] * center.hessian[k][i]
                for k in range(dimension)
            ),
            iv.mpf(0),
        )
        for i in range(dimension)
    ]
    displacement_forms = []
    for position, (center_value, radius, coordinate) in enumerate(
        zip(centers, radii, variables, strict=True)
    ):
        displacement_forms.append(
            FastAffine.variable(
                float(center_value - target[coordinate]),
                float(radius),
                position,
                dimension,
            )
        )
    q_hessian = []
    maximum_hessian_magnitude = 0.0
    for i in range(dimension):
        row = []
        for j in range(dimension):
            value = enclosure.hessian[i][j] * 2.0
            for k in range(dimension):
                value += displacement_forms[k] * third_component(
                    enclosure, k, i, j
                )
            magnitude = max(abs(value.lower()), abs(value.upper()))
            maximum_hessian_magnitude = max(
                maximum_hessian_magnitude, magnitude
            )
            row.append(magnitude)
        q_hessian.append(row)
    radius_floats = [float(value) for value in radii]
    contributions = []
    total = upper_float(q_center)
    for i in range(dimension):
        contribution = upper_float(abs(q_gradient[i])) * radius_floats[i]
        for j in range(dimension):
            contribution += (
                q_hessian[i][j]
                * radius_floats[i]
                * radius_floats[j]
                / 2.0
            )
        contributions.append(contribution)
        total += contribution
    return total, contributions, {
        "center_upper": upper_float(q_center),
        "linear_quadratic_total": sum(contributions),
        "maximum_radial_hessian_magnitude": maximum_hessian_magnitude,
    }


def centered_curvature_upper(
    box: FaceBox,
    variables: list[int],
    dependent: list[int],
    coefficients: list[list[Fraction]],
    coordinates: list[int],
    pairs: list[dict[str, int]],
    first_cuts: list[dict[str, Any]],
    second_cuts: list[dict[str, Any]],
    *,
    affine_third: bool = False,
    fast_affine_third: bool = False,
    curvature_only: bool = False,
) -> tuple[float, list[float]]:
    """Hybrid centered third-order curvature bound; spectral replay pending."""

    import numpy as np

    center_lower = list(box.lower)
    center_upper = list(box.upper)
    radii = []
    for coordinate in variables:
        center_value = (box.lower[coordinate] + box.upper[coordinate]) / 2
        center_lower[coordinate] = center_value
        center_upper[coordinate] = center_value
        radii.append((box.upper[coordinate] - box.lower[coordinate]) / 2)
    center_box = FaceBox(tuple(center_lower), tuple(center_upper))
    center = jet_log(
        mixture_jet(
            center_box, variables, dependent, coefficients, coordinates, pairs,
            first_cuts, second_cuts,
        )
    )
    if fast_affine_third:
        enclosure = mixture_log_symmetric_third(
            box, variables, dependent, coefficients, coordinates, pairs,
            first_cuts, second_cuts,
        )
    elif affine_third:
        enclosure = (
            mixture_log_affine_third(
                box, variables, dependent, coefficients, coordinates, pairs,
                first_cuts, second_cuts,
                AffineInterval,
            )
        )
    else:
        enclosure = third_log(
            mixture_third_jet(
                box, variables, dependent, coefficients, coordinates, pairs,
                first_cuts, second_cuts,
            )
        )
    dimension = len(variables)
    hessian_enclosure = []
    maximum_third = 0.0
    radius_intervals = [
        iv.mpf(f"{radius.numerator}/{radius.denominator}") for radius in radii
    ]
    coordinate_curvature_errors = [
        [[0.0 for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    affine_third_centers = [
        [[0.0 for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    affine_third_residuals = [
        [[0.0 for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for i in range(dimension):
        row = []
        for j in range(dimension):
            value = center.hessian[i][j]
            deviation = iv.mpf(0)
            for k in range(dimension):
                component = third_component(enclosure, i, j, k)
                component_interval = scalar_interval(component)
                magnitude = abs(component_interval)
                maximum_third = max(maximum_third, upper_float(magnitude))
                deviation += magnitude * radius_intervals[k]
                coordinate_curvature_errors[k][i][j] = (
                    upper_float(magnitude) * float(radii[k])
                )
                if isinstance(component, FastAffine):
                    affine_third_centers[k][i][j] = component.center
                    affine_third_residuals[k][i][j] = component.radius()
                elif isinstance(component, AffineInterval):
                    center_lower = lower_float(component.center)
                    center_upper = upper_float(component.center)
                    affine_third_centers[k][i][j] = (
                        center_lower + center_upper
                    ) / 2.0
                    affine_third_residuals[k][i][j] = (
                        (center_upper - center_lower) / 2.0
                        + upper_float(component.radius())
                    )
            row.append(
                value
                + iv.mpf(
                    [
                        str(-upper_float(deviation)),
                        str(upper_float(deviation)),
                    ]
                )
            )
        hessian_enclosure.append(row)

    midpoint = np.empty((dimension, dimension))
    error = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            lower = lower_float(hessian_enclosure[i][j])
            upper = upper_float(hessian_enclosure[i][j])
            midpoint[i, j] = (lower + upper) / 2.0
            error[i, j] = (upper - lower) / 2.0
    midpoint = (midpoint + midpoint.T) / 2.0
    error = np.maximum(error, error.T)
    if affine_third or fast_affine_third:
        center_midpoint = np.empty((dimension, dimension))
        center_error = np.empty((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                lower = lower_float(center.hessian[i][j])
                upper = upper_float(center.hessian[i][j])
                center_midpoint[i, j] = (lower + upper) / 2.0
                center_error[i, j] = (upper - lower) / 2.0
        center_midpoint = (center_midpoint + center_midpoint.T) / 2.0
        center_error = np.maximum(center_error, center_error.T)
        affine_matrices = [
            np.asarray(matrix, dtype=float) * float(radii[k])
            for k, matrix in enumerate(affine_third_centers)
        ]
        maximum_affine_eigenvalue = -math.inf
        for signs in itertools.product((-1.0, 1.0), repeat=dimension):
            candidate = center_midpoint.copy()
            for sign, matrix in zip(signs, affine_matrices, strict=True):
                candidate += sign * matrix
            candidate = (candidate + candidate.T) / 2.0
            maximum_affine_eigenvalue = max(
                maximum_affine_eigenvalue,
                float(np.linalg.eigvalsh(candidate)[-1]),
            )
        residual_error = center_error.copy()
        for k in range(dimension):
            residual_error += (
                np.asarray(affine_third_residuals[k], dtype=float)
                * float(radii[k])
            )
        residual_error = np.maximum(residual_error, residual_error.T)
        spectral_error = float(np.max(np.sum(residual_error, axis=1)))
        lambda_upper = maximum_affine_eigenvalue + spectral_error + 1e-12
    else:
        spectral_error = float(np.max(np.sum(error, axis=1)))
        lambda_upper = (
            float(np.linalg.eigvalsh(midpoint)[-1]) + spectral_error + 1e-12
        )
    CENTERED_CURVATURE_STATS["evaluations"] += 1
    CENTERED_CURVATURE_STATS["minimum_lambda_upper"] = min(
        CENTERED_CURVATURE_STATS["minimum_lambda_upper"], lambda_upper
    )
    CENTERED_CURVATURE_STATS["maximum_lambda_upper"] = max(
        CENTERED_CURVATURE_STATS["maximum_lambda_upper"], lambda_upper
    )
    CENTERED_CURVATURE_STATS["maximum_third_derivative_magnitude"] = max(
        CENTERED_CURVATURE_STATS["maximum_third_derivative_magnitude"],
        maximum_third,
    )
    curvature_contributions = [
        max(sum(row) for row in matrix)
        for matrix in coordinate_curvature_errors
    ]

    if curvature_only:
        if lambda_upper < 0.0:
            CENTERED_CURVATURE_STATS["negative_curvature"] += 1
            exact_certificate = (
                None
                if fast_affine_third
                else exact_centered_curvature_certificate(
                    box, variables, center, hessian_enclosure, radii
                )
            )
            if exact_certificate is not None:
                CENTERED_CURVATURE_CERTIFICATES.append(exact_certificate)
                return (
                    -float(Fraction(exact_certificate["certified_curvature"])),
                    curvature_contributions,
                )
            if not fast_affine_third:
                return max(lambda_upper, 0.0), curvature_contributions
        else:
            CENTERED_CURVATURE_STATS["fallback"] += 1
        return lambda_upper, curvature_contributions

    radius_floats = [float(radius) for radius in radii]
    if lambda_upper < 0.0:
        CENTERED_CURVATURE_STATS["negative_curvature"] += 1
        gradient_bounds = [upper_float(abs(value)) for value in center.gradient]
        contributions = []
        for gradient, radius in zip(gradient_bounds, radius_floats, strict=True):
            displacement = min(radius, gradient / (-lambda_upper))
            contributions.append(
                gradient * displacement
                + lambda_upper * displacement * displacement / 2.0
            )
        exact_certificate = (
            None
            if fast_affine_third
            else exact_centered_curvature_certificate(
                box, variables, center, hessian_enclosure, radii
            )
        )
        if exact_certificate is not None:
            CENTERED_CURVATURE_CERTIFICATES.append(exact_certificate)
            if exact_certificate["closed"]:
                exact_upper = float(Fraction(exact_certificate["total_log_upper"]))
                return exact_upper, contributions
        return upper_float(center.value) + sum(contributions) + 1e-14, contributions

    CENTERED_CURVATURE_STATS["fallback"] += 1
    bound = center.value
    contributions = [iv.mpf(0) for _ in variables]
    linear_terms = []
    quadratic_terms = [
        [iv.mpf(0) for _ in range(dimension)] for _ in range(dimension)
    ]
    for i in range(dimension):
        linear = abs(center.gradient[i]) * radius_intervals[i]
        bound += linear
        contributions[i] += linear
        linear_terms.append(linear)
        for j in range(dimension):
            quadratic = (
                abs(hessian_enclosure[i][j])
                * radius_intervals[i]
                * radius_intervals[j]
                / 2
            )
            bound += quadratic
            contributions[i] += quadratic
            quadratic_terms[i][j] = quadratic
    center_upper = outward_fraction_bounds(center.value)[1]
    linear_upper = [outward_fraction_bounds(value)[1] for value in linear_terms]
    quadratic_upper = [
        [outward_fraction_bounds(value)[1] for value in row]
        for row in quadratic_terms
    ]
    exact_total = (
        center_upper
        + sum(linear_upper, ZERO)
        + sum((sum(row, ZERO) for row in quadratic_upper), ZERO)
    )
    if exact_total < ZERO:
        DIRECTED_TAYLOR_CERTIFICATES.append(
            {
                "status": "exact rational replay of directed second-order Taylor bound",
                "variables": variables,
                "box_lower": [fraction_text(value) for value in box.lower],
                "box_upper": [fraction_text(value) for value in box.upper],
                "center_log_value_upper": fraction_text(center_upper),
                "linear_term_upper": [
                    fraction_text(value) for value in linear_upper
                ],
                "quadratic_term_upper": [
                    [fraction_text(value) for value in row]
                    for row in quadratic_upper
                ],
                "total_log_upper": fraction_text(exact_total),
                "closed": True,
            }
        )
        return float(exact_total), [upper_float(value) for value in contributions]
    return upper_float(bound), [upper_float(value) for value in contributions]


def apply_face(
    lower: list[Fraction], upper: list[Fraction], face: dict[str, Any]
) -> tuple[FaceBox, list[int], list[int]]:
    lower = list(lower)
    upper = list(upper)
    fixed_indices = set()
    for specification in face["fixed"]:
        dimension_text, side = specification.split(":")
        dimension = int(dimension_text)
        fixed_indices.add(dimension)
        if side == "lower":
            upper[dimension] = lower[dimension]
        elif side == "upper":
            lower[dimension] = upper[dimension]
        else:
            raise ValueError("fixed face side must be lower or upper")
    for dimension_text, value_text in face.get("fixed_values", {}).items():
        dimension = int(dimension_text)
        value = Fraction(value_text)
        if dimension in fixed_indices:
            raise ValueError("a coordinate cannot have both side and value fixes")
        if not lower[dimension] <= value <= upper[dimension]:
            raise ValueError("fixed face value lies outside the chamber box")
        fixed_indices.add(dimension)
        lower[dimension] = value
        upper[dimension] = value
    dependent = [int(value) for value in face["dependent_coordinates"]]
    if fixed_indices.intersection(dependent):
        raise ValueError("fixed and dependent coordinates overlap")
    variables = [index for index in range(15) if index not in fixed_indices and index not in dependent]
    tightened = tighten_simplex_box(FaceBox(tuple(lower), tuple(upper)), variables, dependent)
    if tightened is None:
        raise ValueError("candidate face is infeasible")
    return tightened, variables, dependent


def serialized_box(box: FaceBox) -> dict[str, list[str]]:
    return {
        "lower": [fraction_text(value) for value in box.lower],
        "upper": [fraction_text(value) for value in box.upper],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("--face", required=True)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--max-boxes", type=int, default=10000)
    parser.add_argument(
        "--center-from",
        type=Path,
        help="restrict to a local box centered at the point in this optimizer result",
    )
    parser.add_argument(
        "--local-radius-fraction",
        default="1",
        help="exact decimal radius as a fraction of each full face width",
    )
    parser.add_argument(
        "--bound-method",
        choices=(
            "natural",
            "mean",
            "taylor",
            "derivative-affine",
            "derivative-fast-affine",
            "derivative-log-affine",
            "derivative-log-fast-affine",
            "derivative-natural",
            "derivative-mean",
            "log-hessian-affine",
            "log-hessian-fast-affine",
            "log-concave",
            "centered-curvature",
            "affine-centered-curvature",
            "affine-curvature-cover",
            "fast-affine-centered-curvature",
            "fast-affine-curvature-cover",
            "fast-affine-hessian-cover",
        ),
        default="log-concave",
    )
    parser.add_argument(
        "--derivative-coordinate",
        type=int,
        help=(
            "free face coordinate differentiated by derivative-mean; the "
            "dependent coordinate in its simplex block decreases"
        ),
    )
    parser.add_argument(
        "--derivative-sign",
        choices=("positive", "negative"),
        help=(
            "strict selected log-derivative sign; defaults to positive for "
            "backward compatibility"
        ),
    )
    parser.add_argument(
        "--hessian-coordinate",
        type=int,
        help="second free coordinate for a log-Hessian sign bound",
    )
    parser.add_argument(
        "--hessian-sign",
        choices=("negative", "positive"),
        help="strict log-Hessian sign to certify",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume-subdivision",
        type=Path,
        help="resume the pending queue from an earlier result of the same method",
    )
    parser.add_argument(
        "--seed-boxes-from",
        type=Path,
        help=(
            "rigorously reevaluate the terminal rational boxes from another "
            "subdivision before refining any that remain open"
        ),
    )
    parser.add_argument(
        "--seed-box-offset",
        type=int,
        default=0,
        help="zero-based first terminal seed box to replay",
    )
    parser.add_argument(
        "--seed-box-count",
        type=int,
        help="number of terminal seed boxes to replay",
    )
    args = parser.parse_args()
    if args.resume_subdivision is not None and args.seed_boxes_from is not None:
        parser.error("--resume-subdivision and --seed-boxes-from are mutually exclusive")
    if args.seed_box_offset < 0 or (
        args.seed_box_count is not None and args.seed_box_count <= 0
    ):
        parser.error("seed box offset/count must be nonnegative/positive")
    if args.seed_boxes_from is None and (
        args.seed_box_offset != 0 or args.seed_box_count is not None
    ):
        parser.error("seed box slicing requires --seed-boxes-from")
    if args.precision < 40 or args.max_boxes < 0:
        parser.error("precision must be at least 40 and max-boxes must be nonnegative")
    derivative_methods = (
        "derivative-affine",
        "derivative-fast-affine",
        "derivative-log-affine",
        "derivative-log-fast-affine",
        "derivative-natural",
        "derivative-mean",
    )
    hessian_methods = ("log-hessian-affine", "log-hessian-fast-affine")
    if args.bound_method in derivative_methods + hessian_methods:
        if args.derivative_coordinate is None:
            parser.error("the selected method requires --derivative-coordinate")
    elif args.derivative_coordinate is not None:
        parser.error("--derivative-coordinate requires a derivative or Hessian method")
    if args.bound_method in hessian_methods:
        if args.hessian_coordinate is None or args.hessian_sign is None:
            parser.error(
                "a log-Hessian method requires --hessian-coordinate and --hessian-sign"
            )
    elif args.hessian_coordinate is not None or args.hessian_sign is not None:
        parser.error("Hessian options require a log-Hessian method")
    derivative_log_methods = (
        "derivative-log-affine",
        "derivative-log-fast-affine",
    )
    if args.derivative_sign is not None and args.bound_method not in derivative_log_methods:
        parser.error("--derivative-sign requires a derivative-log method")
    local_radius_fraction = rational(args.local_radius_fraction)
    if local_radius_fraction <= ZERO:
        parser.error("--local-radius-fraction must be positive")
    iv.dps = args.precision

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = (args.manifest.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    face_file = json.loads(args.faces.read_text(encoding="utf-8"))
    leaf_name = face_file["leaf"]
    leaf = leaves[leaf_name]
    matches = [face for face in face_file["faces"] if face["name"] == args.face]
    if len(matches) != 1:
        parser.error("face selection did not identify exactly one face")
    face = matches[0]
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(record for record in partitions["leaves"] if record["leaf"] == leaf_name)
    if not record.get("partition_identity_exact"):
        raise ValueError("rational partition identity has not been certified")
    if not record.get("nonnegative_on_product_box_exact"):
        raise ValueError("affine-weight nonnegativity has not been certified")
    if Fraction(record["minimum_exact_weight"]) < ZERO:
        raise ValueError("affine partition contains a negative certified weight")
    denominator = int(record["coefficient_denominator"])
    coefficients = [
        [Fraction(int(value), denominator) for value in row]
        for row in record["coefficient_numerators"]
    ]
    verify_affine_partition_identity(coefficients)
    checkpoint = json.loads((root / manifest["global_checkpoint"]).read_text(encoding="utf-8"))
    selected = sorted(checkpoint["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    base_lower = [rational(value) for value in selected["lower"]]
    base_upper = [rational(value) for value in selected["upper"]]
    gate_reference = face.get("gate", record.get("gate", leaf["gate"]))
    gate_path = root / str(gate_reference).replace("\\", "/")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lower, upper = chamber_bounds(base_lower, base_upper, gate["restrictions"])
    initial, variables, dependent = apply_face(lower, upper, face)
    if (
        args.bound_method in derivative_methods
        and args.bound_method not in (*hessian_methods, *derivative_log_methods)
        and args.derivative_coordinate not in variables
    ):
        parser.error("--derivative-coordinate must select a free face coordinate")
    if (
        args.bound_method in hessian_methods
        and (
            args.derivative_coordinate in dependent
            or not 0 <= int(args.derivative_coordinate) < 15
            or args.hessian_coordinate not in variables
        )
    ):
        parser.error(
            "the selected derivative must be explicit and the Hessian coordinate free"
        )
    if (
        args.bound_method in derivative_log_methods
        and (
            args.derivative_coordinate in dependent
            or not 0 <= int(args.derivative_coordinate) < 15
        )
    ):
        parser.error("the selected log derivative must use an explicit coordinate")
    if args.center_from is not None:
        center_payload = json.loads(args.center_from.read_text(encoding="utf-8"))
        center_point = [rational(value) for value in center_payload["point"]]
        local_lower = list(initial.lower)
        local_upper = list(initial.upper)
        for coordinate in variables:
            radius = (
                local_radius_fraction
                * (initial.upper[coordinate] - initial.lower[coordinate])
            )
            local_lower[coordinate] = max(
                local_lower[coordinate], center_point[coordinate] - radius
            )
            local_upper[coordinate] = min(
                local_upper[coordinate], center_point[coordinate] + radius
            )
        tightened = tighten_simplex_box(
            FaceBox(tuple(local_lower), tuple(local_upper)), variables, dependent
        )
        if tightened is None:
            raise ValueError("local optimizer neighborhood is infeasible")
        initial = tightened
    bank = json.loads((root / "results/f5-projected-kl-dual-bank-0.22.json").read_text(encoding="utf-8"))
    first_cuts = [exact_cut(raw) for raw in bank["first_cuts"]]
    second_cuts = [exact_cut(raw) for raw in bank["second_cuts"]]

    coordinates = [int(value) for value in record["feature_coordinates"]]

    def evaluate(box: FaceBox) -> tuple[float, list[float]]:
        if args.bound_method == "natural":
            return (
                mixture_upper(
                    box, variables, dependent, coefficients, coordinates,
                    record["expert_pairs"], first_cuts, second_cuts,
                ),
                [float(box.upper[index] - box.lower[index]) for index in variables],
            )
        if args.bound_method == "mean":
            return mean_value_upper(
                box, variables, dependent, coefficients, coordinates,
                record["expert_pairs"], first_cuts, second_cuts,
            )
        if args.bound_method == "taylor":
            return taylor_upper(
                box, variables, dependent, coefficients, coordinates,
                record["expert_pairs"], first_cuts, second_cuts,
            )
        if args.bound_method == "derivative-mean":
            return directional_derivative_negative_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
            )
        if args.bound_method == "derivative-natural":
            return directional_derivative_natural_negative_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
            )
        if args.bound_method == "derivative-affine":
            return directional_derivative_affine_negative_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
            )
        if args.bound_method == "derivative-fast-affine":
            return directional_derivative_fast_affine_negative_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
            )
        if args.bound_method in (
            "derivative-log-affine",
            "derivative-log-fast-affine",
        ):
            return directional_log_derivative_affine_negative_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
                (
                    FastAffine
                    if args.bound_method == "derivative-log-fast-affine"
                    else AffineInterval
                ),
                args.derivative_sign or "positive",
            )
        if args.bound_method in hessian_methods:
            return log_hessian_sign_violation_upper(
                box,
                variables,
                dependent,
                int(args.derivative_coordinate),
                int(args.hessian_coordinate),
                str(args.hessian_sign),
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
                (
                    FastAffine
                    if args.bound_method == "log-hessian-fast-affine"
                    else AffineInterval
                ),
            )
        if args.bound_method == "log-concave":
            return log_concavity_upper(
                box, variables, dependent, coefficients, coordinates,
                record["expert_pairs"], first_cuts, second_cuts,
            )
        if args.bound_method == "fast-affine-hessian-cover":
            return fast_affine_hessian_curvature_upper(
                box,
                variables,
                dependent,
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
            )
        return centered_curvature_upper(
            box, variables, dependent, coefficients, coordinates,
            record["expert_pairs"], first_cuts, second_cuts,
            affine_third=args.bound_method
            in ("affine-centered-curvature", "affine-curvature-cover"),
            fast_affine_third=(
                args.bound_method
                in (
                    "fast-affine-centered-curvature",
                    "fast-affine-curvature-cover",
                )
            ),
            curvature_only=args.bound_method
            in ("affine-curvature-cover", "fast-affine-curvature-cover"),
        )

    if args.resume_subdivision is None and args.seed_boxes_from is None:
        initial_upper, initial_contributions = evaluate(initial)
        queue: list[tuple[float, int, FaceBox, list[float]]] = [
            (-initial_upper, 0, initial, initial_contributions)
        ]
        identifier = 1
        closed = 0
        closed_records = []
        processed = 0
    elif args.resume_subdivision is not None:
        resumed = json.loads(args.resume_subdivision.read_text(encoding="utf-8"))
        if (
            resumed.get("leaf") != leaf_name
            or resumed.get("face") != args.face
            or resumed.get("bound_method") != args.bound_method
            or resumed.get("variables") != variables
            or resumed.get("derivative_coordinate") != args.derivative_coordinate
            or resumed.get("hessian_coordinate") != args.hessian_coordinate
            or resumed.get("hessian_sign") != args.hessian_sign
            or resumed.get("derivative_sign")
            != (
                args.derivative_sign or "positive"
                if args.bound_method in derivative_log_methods
                else None
            )
        ):
            raise ValueError("subdivision checkpoint does not match this face and method")
        initial_upper = float(resumed["initial_interval_upper"])
        initial_contributions = [
            float(value) for value in resumed["initial_mean_value_contributions"]
        ]
        queue = []
        identifier = 0
        for entry in resumed["pending_box_records"]:
            box = FaceBox(
                tuple(Fraction(value) for value in entry["lower"]),
                tuple(Fraction(value) for value in entry["upper"]),
            )
            heapq.heappush(
                queue,
                (
                    -float(entry["bound_upper"]),
                    identifier,
                    box,
                    [float(value) for value in entry["contributions"]],
                ),
            )
            identifier += 1
        closed_records = list(resumed.get("closed_box_records", []))
        closed = len(closed_records)
        processed = int(resumed["processed_boxes"])
    else:
        seeded = json.loads(args.seed_boxes_from.read_text(encoding="utf-8"))
        if (
            seeded.get("leaf") != leaf_name
            or seeded.get("face") != args.face
            or seeded.get("variables") != variables
            or seeded.get("derivative_coordinate") != args.derivative_coordinate
            or seeded.get("hessian_coordinate") != args.hessian_coordinate
            or seeded.get("hessian_sign") != args.hessian_sign
            or seeded.get("derivative_sign")
            != (
                args.derivative_sign or "positive"
                if args.bound_method in derivative_log_methods
                else None
            )
            or seeded.get("initial_box") != serialized_box(initial)
        ):
            raise ValueError("seed subdivision does not match this face and target")
        entries = [
            *seeded.get("closed_box_records", []),
            *seeded.get("pending_box_records", []),
        ]
        if not entries:
            raise ValueError("seed subdivision contains no terminal boxes")
        seed_total_entries = len(entries)
        seed_stop = (
            len(entries)
            if args.seed_box_count is None
            else args.seed_box_offset + args.seed_box_count
        )
        entries = entries[args.seed_box_offset:seed_stop]
        if not entries:
            raise ValueError("seed box slice is empty")
        queue = []
        identifier = 0
        initial_upper = None
        initial_contributions = []
        for entry in entries:
            box = FaceBox(
                tuple(Fraction(value) for value in entry["lower"]),
                tuple(Fraction(value) for value in entry["upper"]),
            )
            value, contributions = evaluate(box)
            heapq.heappush(queue, (-value, identifier, box, contributions))
            identifier += 1
        closed_records = []
        closed = 0
        processed = 0
    while queue and processed < args.max_boxes:
        neg_upper, _, box, contributions = heapq.heappop(queue)
        upper_value = -neg_upper
        comparison_target = (
            0.0
            if args.bound_method
            in (
                "log-concave",
                "derivative-affine",
                "derivative-fast-affine",
                "derivative-log-affine",
                "derivative-log-fast-affine",
                "derivative-natural",
                "derivative-mean",
                "log-hessian-affine",
                "log-hessian-fast-affine",
                "centered-curvature",
                "affine-centered-curvature",
                "affine-curvature-cover",
                "fast-affine-centered-curvature",
                "fast-affine-curvature-cover",
                "fast-affine-hessian-cover",
            )
            else 1.0
        )
        if upper_value < comparison_target:
            closed += 1
            closed_records.append(
                {
                    **serialized_box(box),
                    "bound_upper": upper_value,
                }
            )
            processed += 1
            continue
        dimension = variables[max(range(len(variables)), key=contributions.__getitem__)]
        threshold = (box.lower[dimension] + box.upper[dimension]) / 2
        for left in (True, False):
            child_lower = list(box.lower)
            child_upper = list(box.upper)
            if left:
                child_upper[dimension] = threshold
            else:
                child_lower[dimension] = threshold
            child = tighten_simplex_box(
                FaceBox(tuple(child_lower), tuple(child_upper)), variables, dependent
            )
            if child is None:
                continue
            child_upper_value, child_contributions = evaluate(child)
            heapq.heappush(
                queue,
                (-child_upper_value, identifier, child, child_contributions),
            )
            identifier += 1
        processed += 1
        if processed % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "pending": len(queue),
                        "closed": closed,
                        "worst_interval_upper": -queue[0][0] if queue else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    fast_proposal = args.bound_method in (
        "derivative-fast-affine",
        "derivative-log-fast-affine",
        "log-hessian-fast-affine",
        "fast-affine-centered-curvature",
        "fast-affine-curvature-cover",
        "fast-affine-hessian-cover",
    )
    payload = {
        "status": (
            "floating affine-form subdivision proposal closed"
            if fast_proposal and not queue
            else "floating affine-form subdivision proposal remains open"
            if fast_proposal
            else "directed interval face closed"
            if not queue
            else "directed interval face remains open"
        ),
        "rigorous": not fast_proposal,
        "scope": "one candidate face only; does not prove complete chamber coverage",
        "leaf": leaf_name,
        "face": args.face,
        "gate": str(gate_reference),
        "precision": args.precision,
        "center_source": str(args.center_from) if args.center_from is not None else None,
        "resume_source": (
            str(args.resume_subdivision)
            if args.resume_subdivision is not None
            else None
        ),
        "seed_boxes_source": (
            str(args.seed_boxes_from) if args.seed_boxes_from is not None else None
        ),
        "seed_box_offset": args.seed_box_offset,
        "seed_box_count": (
            len(entries) if args.seed_boxes_from is not None else None
        ),
        "seed_box_total": (
            seed_total_entries if args.seed_boxes_from is not None else None
        ),
        "local_radius_fraction": str(local_radius_fraction),
        "bound_method": args.bound_method,
        "derivative_coordinate": args.derivative_coordinate,
        "derivative_sign": (
            args.derivative_sign or "positive"
            if args.bound_method in derivative_log_methods
            else None
        ),
        "hessian_coordinate": args.hessian_coordinate,
        "hessian_sign": args.hessian_sign,
        "variables": variables,
        "dependent_coordinates": dependent,
        "initial_box": serialized_box(initial),
        "initial_interval_upper": initial_upper,
        "initial_mean_value_contributions": initial_contributions,
        "processed_boxes": processed,
        "closed_boxes": closed,
        "pending_boxes": len(queue),
        "worst_interval_upper": -queue[0][0] if queue else None,
        "closed_box_records": closed_records,
        "pending_box_records": [
            {
                **serialized_box(box),
                "bound_upper": -negative_upper,
                "contributions": contributions,
            }
            for negative_upper, _, box, contributions in sorted(queue)
        ],
    }
    if args.bound_method == "log-concave":
        payload["log_concavity_diagnostics"] = dict(LOG_CONCAVITY_STATS)
    if args.bound_method in (
        "centered-curvature",
        "affine-centered-curvature",
        "affine-curvature-cover",
        "fast-affine-centered-curvature",
        "fast-affine-curvature-cover",
        "fast-affine-hessian-cover",
    ):
        payload["centered_curvature_diagnostics"] = dict(
            CENTERED_CURVATURE_STATS
        )
        payload["exact_centered_curvature_certificates"] = list(
            CENTERED_CURVATURE_CERTIFICATES
        )
        payload["exact_directed_taylor_certificates"] = list(
            DIRECTED_TAYLOR_CERTIFICATES
        )
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
