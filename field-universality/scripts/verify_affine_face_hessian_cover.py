#!/usr/bin/env python3
"""Replay a chunked affine-face selected-derivative sign certificate.

The subdivision generator tightens every rectangular child against three
simplex equalities.  Its terminal boxes therefore cover the feasible slice,
not the whole ambient rectangle.  This verifier reconstructs that feasible
guillotine cover with exact rational arithmetic and independently recomputes
the directed affine-interval bound on every terminal box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from mpmath import iv

from explore_affine_face_interval import (
    AffineInterval,
    FaceBox,
    apply_face,
    exact_cut,
    outward_fraction_bounds,
    relative_log_mixture_affine_directional,
    serialized_box,
    tighten_simplex_box,
    verify_affine_partition_identity,
)
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


ZERO = Fraction(0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_box(record: dict[str, Any]) -> FaceBox:
    return FaceBox(
        tuple(Fraction(value) for value in record["lower"]),
        tuple(Fraction(value) for value in record["upper"]),
    )


def contains(container: FaceBox, contained: FaceBox) -> bool:
    return all(
        container.lower[index] <= contained.lower[index]
        and contained.upper[index] <= container.upper[index]
        for index in range(len(container.lower))
    )


def overlaps(left: FaceBox, right: FaceBox) -> bool:
    return all(
        left.lower[index] <= right.upper[index]
        and right.lower[index] <= left.upper[index]
        for index in range(len(left.lower))
    )


def verify_feasible_guillotine_cover(
    parent: FaceBox,
    leaves: list[FaceBox],
    variables: list[int],
    dependent: list[int],
) -> dict[str, int]:
    """Prove that ``leaves`` cover the feasible simplex slice of ``parent``.

    The proof recursively reconstructs axis-aligned splits.  Each node is
    tightened using exact simplex propagation before containment or further
    splitting is considered.  Thus ambient rectangular gaps that contain no
    feasible point are correctly ignored.
    """

    parent = tighten_simplex_box(parent, variables, dependent)
    if parent is None:
        raise ValueError("cover parent is infeasible")
    if not leaves:
        raise ValueError("cover contains no leaves")
    for leaf in leaves:
        if len(leaf.lower) != len(parent.lower) or len(leaf.upper) != len(parent.upper):
            raise ValueError("cover leaf has the wrong ambient dimension")
        if any(
            leaf.lower[index] > leaf.upper[index]
            for index in range(len(leaf.lower))
        ):
            raise ValueError("cover leaf has a reversed interval")
        if not contains(parent, leaf):
            raise ValueError("cover leaf lies outside its feasible parent hull")

    memo: dict[tuple[tuple[Fraction, ...], tuple[Fraction, ...]], bool] = {}
    calls = 0

    def covered(node: FaceBox) -> bool:
        nonlocal calls
        tightened = tighten_simplex_box(node, variables, dependent)
        if tightened is None:
            return True
        key = (tightened.lower, tightened.upper)
        if key in memo:
            return memo[key]
        calls += 1
        active = [leaf for leaf in leaves if overlaps(leaf, tightened)]
        if any(contains(leaf, tightened) for leaf in active):
            memo[key] = True
            return True

        for coordinate in variables:
            thresholds = sorted(
                {
                    endpoint
                    for leaf in active
                    for endpoint in (
                        leaf.lower[coordinate],
                        leaf.upper[coordinate],
                    )
                    if tightened.lower[coordinate]
                    < endpoint
                    < tightened.upper[coordinate]
                }
            )
            for threshold in thresholds:
                left_upper = list(tightened.upper)
                left_upper[coordinate] = threshold
                right_lower = list(tightened.lower)
                right_lower[coordinate] = threshold
                left = FaceBox(tightened.lower, tuple(left_upper))
                right = FaceBox(tuple(right_lower), tightened.upper)
                if covered(left) and covered(right):
                    memo[key] = True
                    return True
        memo[key] = False
        return False

    if not covered(parent):
        raise ValueError("terminal boxes do not cover the feasible simplex slice")
    return {"recursive_nodes": calls, "memoized_nodes": len(memo)}


def target_metadata(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("leaf"),
        payload.get("face"),
        payload.get("derivative_coordinate"),
        payload.get("derivative_sign"),
        payload.get("hessian_coordinate"),
        payload.get("hessian_sign"),
        tuple(payload.get("variables", [])),
        tuple(payload.get("dependent_coordinates", [])),
    )


def load_problem(
    manifest_path: Path,
    partitions_path: Path,
    faces_path: Path,
    face_name: str,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    root = (manifest_path.resolve().parent / manifest.get("artifact_root", "..")).resolve()
    leaves = {leaf["leaf"]: leaf for leaf in collect_leaves(manifest["tree"])}
    face_file = load_json(faces_path)
    matches = [face for face in face_file["faces"] if face["name"] == face_name]
    if len(matches) != 1:
        raise ValueError("face selection did not identify exactly one face")
    face = matches[0]
    leaf_name = face_file["leaf"]
    leaf = leaves[leaf_name]

    partitions = load_json(partitions_path)
    if Path(str(partitions["manifest"]).replace("\\", "/")).name != manifest_path.name:
        raise ValueError("partition file names a different manifest")
    records = [record for record in partitions["leaves"] if record["leaf"] == leaf_name]
    if len(records) != 1:
        raise ValueError("partition file does not identify exactly one leaf record")
    record = records[0]
    if not record.get("partition_identity_exact"):
        raise ValueError("partition identity is not exact")
    if not record.get("nonnegative_on_product_box_exact"):
        raise ValueError("partition nonnegativity is not exact")
    if Fraction(record["minimum_exact_weight"]) < ZERO:
        raise ValueError("partition has a negative exact weight")
    denominator = int(record["coefficient_denominator"])
    coefficients = [
        [Fraction(int(value), denominator) for value in row]
        for row in record["coefficient_numerators"]
    ]
    verify_affine_partition_identity(coefficients)

    checkpoint = load_json(root / manifest["global_checkpoint"])
    selected = sorted(checkpoint["queue"], key=lambda entry: entry["priority"])[0][
        "evaluated"
    ]
    base_lower = [rational(value) for value in selected["lower"]]
    base_upper = [rational(value) for value in selected["upper"]]
    gate_reference = face.get("gate", record.get("gate", leaf["gate"]))
    gate_path = root / str(gate_reference).replace("\\", "/")
    gate = load_json(gate_path)
    lower, upper = chamber_bounds(base_lower, base_upper, gate["restrictions"])
    initial, variables, dependent = apply_face(lower, upper, face)

    bank_path = root / "results/f5-projected-kl-dual-bank-0.22.json"
    bank = load_json(bank_path)
    first_cuts = [exact_cut(raw) for raw in bank["first_cuts"]]
    second_cuts = [exact_cut(raw) for raw in bank["second_cuts"]]
    return {
        "root": root,
        "leaf": leaf_name,
        "face": face,
        "record": record,
        "initial": initial,
        "variables": variables,
        "dependent": dependent,
        "coefficients": coefficients,
        "coordinates": [int(value) for value in record["feature_coordinates"]],
        "pairs": record["expert_pairs"],
        "first_cuts": first_cuts,
        "second_cuts": second_cuts,
        "gate_path": gate_path,
        "bank_path": bank_path,
    }


def recompute_sign_violation_upper(
    box: FaceBox,
    problem: dict[str, Any],
    derivative_coordinate: int,
    expected_sign: str,
    hessian_coordinate: int | None,
) -> Fraction:
    log_mixture = relative_log_mixture_affine_directional(
        box,
        problem["variables"],
        problem["dependent"],
        derivative_coordinate,
        problem["coefficients"],
        problem["coordinates"],
        problem["pairs"],
        problem["first_cuts"],
        problem["second_cuts"],
        AffineInterval,
    )
    if hessian_coordinate is None:
        value = log_mixture.directional.value.interval()
    else:
        position = problem["variables"].index(hessian_coordinate)
        value = log_mixture.directional.gradient[position].interval()
    lower, upper = outward_fraction_bounds(value)
    if expected_sign == "negative":
        return upper
    if expected_sign == "positive":
        return -lower
    raise ValueError("expected selected-derivative sign must be negative or positive")


def verify_cover(
    manifest_path: Path,
    partitions_path: Path,
    faces_path: Path,
    proposal_path: Path,
    chunk_paths: list[Path],
    precision: int,
) -> dict[str, Any]:
    if precision < 40:
        raise ValueError("replay precision must be at least 40 decimal digits")
    proposal = load_json(proposal_path)
    if proposal.get("rigorous") is not False:
        raise ValueError("seed proposal must be explicitly non-rigorous")
    proposal_method = proposal.get("bound_method")
    if proposal_method == "log-hessian-fast-affine":
        directed_method = "log-hessian-affine"
        expected_sign = str(proposal["hessian_sign"])
        hessian_coordinate = int(proposal["hessian_coordinate"])
        certificate_kind = "selected log-Hessian entry"
    elif proposal_method == "derivative-log-fast-affine":
        directed_method = "derivative-log-affine"
        expected_sign = str(proposal["derivative_sign"])
        hessian_coordinate = None
        certificate_kind = "selected log derivative"
    else:
        raise ValueError("seed proposal has the wrong method")
    if proposal.get("pending_boxes") != 0:
        raise ValueError("seed proposal is not a terminal cover")
    face_name = str(proposal["face"])
    problem = load_problem(manifest_path, partitions_path, faces_path, face_name)
    if proposal.get("leaf") != problem["leaf"]:
        raise ValueError("proposal leaf disagrees with the reconstructed problem")
    if proposal.get("initial_box") != serialized_box(problem["initial"]):
        raise ValueError("proposal initial box does not replay")
    if proposal.get("variables") != problem["variables"]:
        raise ValueError("proposal free coordinates do not replay")
    if proposal.get("dependent_coordinates") != problem["dependent"]:
        raise ValueError("proposal dependent coordinates do not replay")

    seed_records = [
        *proposal.get("closed_box_records", []),
        *proposal.get("pending_box_records", []),
    ]
    seeds = [parse_box(record) for record in seed_records]
    if len(set((box.lower, box.upper) for box in seeds)) != len(seeds):
        raise ValueError("seed proposal contains duplicate terminal boxes")
    seed_cover_stats = verify_feasible_guillotine_cover(
        problem["initial"], seeds, problem["variables"], problem["dependent"]
    )

    target = target_metadata(proposal)
    claimed_offsets: list[int] = []
    chunk_cover_stats = []
    replay_records = []
    all_terminal_boxes: list[FaceBox] = []
    iv.dps = precision
    for chunk_path in chunk_paths:
        chunk = load_json(chunk_path)
        if target_metadata(chunk) != target:
            raise ValueError(f"chunk target mismatch: {chunk_path}")
        if chunk.get("bound_method") != directed_method:
            raise ValueError(f"chunk is not a directed affine replay: {chunk_path}")
        if chunk.get("rigorous") is not True or chunk.get("pending_boxes") != 0:
            raise ValueError(f"chunk is not a closed rigorous result: {chunk_path}")
        if chunk.get("initial_box") != proposal.get("initial_box"):
            raise ValueError(f"chunk initial box mismatch: {chunk_path}")
        if int(chunk.get("precision", 0)) < 40:
            raise ValueError(f"chunk precision is below 40 digits: {chunk_path}")
        offset = int(chunk["seed_box_offset"])
        count = int(chunk["seed_box_count"])
        if int(chunk.get("seed_box_total", -1)) != len(seeds):
            raise ValueError(f"chunk seed total mismatch: {chunk_path}")
        if offset < 0 or count <= 0 or offset + count > len(seeds):
            raise ValueError(f"chunk seed slice is invalid: {chunk_path}")
        claimed_offsets.extend(range(offset, offset + count))
        assigned = seeds[offset : offset + count]
        records = chunk.get("closed_box_records", [])
        if not records:
            raise ValueError(f"chunk contains no closed terminal boxes: {chunk_path}")
        terminals = [parse_box(record) for record in records]
        if any(not any(contains(seed, terminal) for seed in assigned) for terminal in terminals):
            raise ValueError(f"chunk terminal escapes its assigned seed slice: {chunk_path}")
        parent_stats = []
        for seed in assigned:
            descendants = [terminal for terminal in terminals if contains(seed, terminal)]
            parent_stats.append(
                verify_feasible_guillotine_cover(
                    seed, descendants, problem["variables"], problem["dependent"]
                )
            )
        chunk_cover_stats.append(
            {
                "chunk": str(chunk_path),
                "offset": offset,
                "count": count,
                "terminal_boxes": len(terminals),
                "parent_cover_stats": parent_stats,
            }
        )
        for record, terminal in zip(records, terminals, strict=True):
            exact_upper = recompute_sign_violation_upper(
                terminal,
                problem,
                int(proposal["derivative_coordinate"]),
                expected_sign,
                hessian_coordinate,
            )
            if exact_upper >= ZERO:
                raise ValueError(
                    f"directed replay does not prove the requested sign on {chunk_path}"
                )
            replay_records.append(
                {
                    "chunk": str(chunk_path),
                    "lower": record["lower"],
                    "upper": record["upper"],
                    "sign_violation_upper": (
                        f"{exact_upper.numerator}/{exact_upper.denominator}"
                    ),
                }
            )
        all_terminal_boxes.extend(terminals)

    if sorted(claimed_offsets) != list(range(len(seeds))):
        raise ValueError("chunk seed slices do not partition the seed cover exactly once")
    final_cover_stats = verify_feasible_guillotine_cover(
        problem["initial"],
        all_terminal_boxes,
        problem["variables"],
        problem["dependent"],
    )
    worst = max(Fraction(record["sign_violation_upper"]) for record in replay_records)
    sources = [
        manifest_path,
        partitions_path,
        faces_path,
        proposal_path,
        problem["gate_path"],
        problem["bank_path"],
        *chunk_paths,
    ]
    return {
        "status": "verified feasible cover and strict directed selected-derivative sign",
        "scope": f"one affine-gate face and one {certificate_kind}",
        "leaf": problem["leaf"],
        "face": face_name,
        "derivative_coordinate": proposal["derivative_coordinate"],
        "derivative_sign": proposal.get("derivative_sign"),
        "hessian_coordinate": proposal["hessian_coordinate"],
        "hessian_sign": proposal["hessian_sign"],
        "precision": precision,
        "seed_boxes": len(seeds),
        "directed_terminal_boxes": len(all_terminal_boxes),
        "worst_exact_sign_violation_upper": f"{worst.numerator}/{worst.denominator}",
        "seed_cover_stats": seed_cover_stats,
        "chunk_cover_stats": chunk_cover_stats,
        "final_cover_stats": final_cover_stats,
        "source_sha256": {str(path): sha256(path) for path in sources},
        "box_replays": replay_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("proposal", type=Path)
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_cover(
        args.manifest,
        args.partitions,
        args.faces,
        args.proposal,
        args.chunks,
        args.precision,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
