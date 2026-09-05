#!/usr/bin/env python3
"""Test support-only bounds induced by multiplicatively symmetric Perron cuts.

A Perron cut fitted to compositions that are uniform on F_5^* has potentials
constant on F_5^*.  Its value therefore depends only on support weights after
using H(X) >= h(Pr[X != 0]).  The resulting fixed-cut objective is convex in
the three support weights, so its maximum is attained at a vertex of the
support polytope.  Computations here use floating spectral data and are not
certificates.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import explore_perron_bound as perron


def support_vertices(
    eta: float, delta: float, message_cutoff: float | None = None
) -> list[np.ndarray]:
    # Rows encode M (A,B,C) <= b.
    rows = []
    target = []

    def add(row: tuple[float, float, float], value: float) -> None:
        rows.append(np.array(row))
        target.append(value)

    add(
        (-1.0, 0.0, 0.0),
        -(eta if message_cutoff is None else max(eta, message_cutoff)),
    )
    add((1.0, 0.0, 0.0), 4.0 * delta)
    add((0.0, -1.0, 0.0), -eta / 2.0)
    add((0.0, 1.0, 0.0), 2.0 * delta)
    add((0.0, 0.0, -1.0), -eta / 4.0)
    add((0.0, 0.0, 1.0), delta)
    add((1.0, -2.0, 0.0), 0.0)
    add((0.0, 1.0, -2.0), 0.0)
    matrix = np.stack(rows)
    rhs = np.array(target)
    vertices = {}
    for active in itertools.combinations(range(len(rows)), 3):
        active_matrix = matrix[list(active)]
        if np.linalg.matrix_rank(active_matrix) < 3:
            continue
        point = np.linalg.solve(active_matrix, rhs[list(active)])
        if np.max(matrix @ point - rhs) <= 1e-10:
            vertices[tuple(np.round(point, 14))] = point
    return list(vertices.values())


def uniform_composition(support: float) -> np.ndarray:
    return np.array([1.0 - support, *(support / 4.0 for _ in range(4))])


def concentrated_composition(support: float) -> np.ndarray:
    return np.array([1.0 - support, support, 0.0, 0.0, 0.0])


def run(args: argparse.Namespace) -> dict[str, object]:
    vertices = support_vertices(
        args.eta, args.delta, args.message_cutoff
    )
    random = np.random.default_rng(args.seed)
    references = list(vertices)
    while len(references) < args.references:
        chosen = random.choice(len(vertices), size=min(6, len(vertices)), replace=True)
        weights = random.dirichlet(np.ones(len(chosen)))
        references.append(
            sum(
                weight * vertices[index]
                for weight, index in zip(weights, chosen, strict=True)
            )
        )
    references = references[: args.references]
    first = [
        perron.fit_first_cut(
            uniform_composition(point[0]),
            uniform_composition(point[1]),
            args.log_bound,
        )
        for point in references
    ]
    second = [
        perron.fit_first_cut(
            uniform_composition(point[1]),
            uniform_composition(point[2]),
            args.log_bound,
        )
        for point in references
    ]
    pairs = [(left, right) for left in first for right in second]
    values = np.array(
        [
            [
                perron.full_perron_upper(
                    concentrated_composition(point[0]),
                    concentrated_composition(point[1]),
                    concentrated_composition(point[2]),
                    left,
                    right,
                    3,
                )
                for left, right in pairs
            ]
            for point in vertices
        ]
    )
    upper, weights = perron.common_mixture_bound(values)
    vertex_values = values @ weights
    worst = int(np.argmax(vertex_values))
    return {
        "support_vertices": len(vertices),
        "references": len(references),
        "cut_pairs": len(pairs),
        "active_cut_pairs": int(np.count_nonzero(weights > 1e-9)),
        "global_support_only_upper": upper,
        "message_cutoff": args.message_cutoff,
        "worst_support_vertex": vertices[worst].tolist(),
        "worst_support_vertex_upper": float(vertex_values[worst]),
        "status": "floating symmetric-Perron diagnostic only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.002)
    parser.add_argument("--delta", type=float, default=0.1323)
    parser.add_argument("--references", type=int, default=32)
    parser.add_argument("--message-cutoff", type=float)
    parser.add_argument("--log-bound", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
