#!/usr/bin/env python3
"""Measure active-face dimension and curvature at rational-gate maximizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import null_space

import explore_projected_kl_box_cover as projected
from diagnose_affine_gate_concavity import mixture_hessian, mixture_log_derivatives
from diagnose_joint_cut_concavity import simplex_tangent_basis
from optimize_affine_exponential_gate import domain_matrices
from rationalize_affine_cover import collect_leaves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("--optimizer-prefix", default="results/f5-rational-fine-affine-gate-optimization-cell20766-")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--bank", type=Path, default=Path("results/f5-projected-kl-dual-bank-0.22.json")
    )
    parser.add_argument(
        "--frontier", type=Path,
        default=Path("results/f5-projected-kl-box-cover-0.22.checkpoint.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    records = {record["leaf"]: record for record in partitions["leaves"]}
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    bank_args = SimpleNamespace(
        eta=0.002, delta=0.22, bank_iterations=20, bank_restarts=12,
        seed=11, bank_cache=str(args.bank), rebuild_bank=False,
    )
    _, first_cuts, second_cuts = projected.load_or_generate_bank(bank_args)
    first_kernels = tuple(projected.envelope.kernel(cut) for cut in first_cuts)
    second_kernels = tuple(projected.envelope.kernel(cut) for cut in second_cuts)
    basis = simplex_tangent_basis()
    _, _, domain_upper, domain_target = domain_matrices()
    output = []
    for name, leaf in leaves.items():
        optimizer_path = Path(f"{args.optimizer_prefix}{name}.json")
        optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
        point = np.asarray(optimizer["point"], dtype=float)
        record = records[name]
        denominator = float(int(record["coefficient_denominator"]))
        coefficients = np.asarray(record["coefficient_numerators"], dtype=float) / denominator
        hessian = mixture_hessian(
            point,
            coefficients,
            [int(value) for value in record["feature_coordinates"]],
            record["expert_pairs"],
            first_cuts,
            second_cuts,
            first_kernels,
            second_kernels,
        )
        tangent_gradient, _ = mixture_log_derivatives(
            point,
            coefficients,
            [int(value) for value in record["feature_coordinates"]],
            record["expert_pairs"],
            first_cuts,
            second_cuts,
            first_kernels,
            second_kernels,
        )
        lower = np.asarray(selected["lower"], dtype=float)
        upper = np.asarray(selected["upper"], dtype=float)
        gate = json.loads(Path(leaf["gate"]).read_text(encoding="utf-8"))
        for restriction in gate["restrictions"]:
            dimension_text, side, threshold_text = restriction.split(":")
            dimension = int(dimension_text)
            threshold = float(threshold_text)
            if side == "le":
                upper[dimension] = min(upper[dimension], threshold)
            else:
                lower[dimension] = max(lower[dimension], threshold)
        active_rows = []
        active_labels = []
        for dimension in range(15):
            if abs(point[dimension] - lower[dimension]) <= args.tolerance:
                active_rows.append(basis[dimension])
                active_labels.append(f"d{dimension}=lower")
            elif abs(point[dimension] - upper[dimension]) <= args.tolerance:
                active_rows.append(basis[dimension])
                active_labels.append(f"d{dimension}=upper")
        domain_values = domain_upper @ point - domain_target
        for index, value in enumerate(domain_values):
            if abs(value) <= args.tolerance:
                active_rows.append(domain_upper[index] @ basis)
                active_labels.append(f"domain{index}=active")
        active = np.asarray(active_rows) if active_rows else np.zeros((0, 12))
        tangent = null_space(active)
        face_dimension = tangent.shape[1]
        maximum_eigenvalue = (
            float(np.linalg.eigvalsh(tangent.T @ hessian @ tangent)[-1])
            if face_dimension
            else float("-inf")
        )
        entry = {
            "leaf": name,
            "face_dimension": face_dimension,
            "active_rank": int(np.linalg.matrix_rank(active)),
            "active_constraints": active_labels,
            "maximum_face_hessian_eigenvalue": maximum_eigenvalue,
            "strict_local_maximum_on_face": maximum_eigenvalue < -1e-8,
            "maximum_log_mixture": optimizer["maximum_log_mixture"],
            "tangent_gradient": tangent_gradient.tolist(),
        }
        output.append(entry)
        print(json.dumps({key: value for key, value in entry.items() if key != "active_constraints"}, sort_keys=True))
    payload = {
        "status": "floating active-face Hessian diagnostic; not a global proof",
        "leaves": output,
        "strict_face_maxima": sum(entry["strict_local_maximum_on_face"] for entry in output),
        "maximum_face_dimension": max(entry["face_dimension"] for entry in output),
        "maximum_face_hessian_eigenvalue": max(entry["maximum_face_hessian_eigenvalue"] for entry in output),
    }
    if args.output is not None:
        projected.write_json_atomic(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "leaves"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
