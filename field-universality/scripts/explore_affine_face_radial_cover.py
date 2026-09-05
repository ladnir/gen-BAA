#!/usr/bin/env python3
"""Cover a face outside a local box by affine radial-derivative bounds."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from mpmath import iv

from explore_affine_face_interval import (
    AffineInterval,
    FaceBox,
    FastAffine,
    affine_radial_form,
    apply_face,
    exact_cut,
    fast_radial_taylor_upper,
    fraction_text,
    outward_fraction_bounds,
    serialized_box,
    tighten_simplex_box,
    verify_affine_partition_identity,
)
from rationalize_affine_cover import chamber_bounds, collect_leaves, rational


ZERO = Fraction(0)


def radial_certificate(form: AffineInterval) -> dict[str, Any]:
    center_lower, center_upper = outward_fraction_bounds(form.center)
    coefficient_bounds = [outward_fraction_bounds(value) for value in form.coefficients]
    error_upper = outward_fraction_bounds(form.error)[1]
    absolute_coefficients = [
        max(abs(lower), abs(upper)) for lower, upper in coefficient_bounds
    ]
    total_upper = center_upper + sum(absolute_coefficients, ZERO) + error_upper
    return {
        "center_lower": fraction_text(center_lower),
        "center_upper": fraction_text(center_upper),
        "coefficient_lower": [fraction_text(value[0]) for value in coefficient_bounds],
        "coefficient_upper": [fraction_text(value[1]) for value in coefficient_bounds],
        "absolute_coefficient_upper": [
            fraction_text(value) for value in absolute_coefficients
        ],
        "error_upper": fraction_text(error_upper),
        "total_upper": fraction_text(total_upper),
        "closed": total_upper < ZERO,
    }


def complement_slabs(
    outer: FaceBox,
    variables: list[int],
    dependent: list[int],
    target: list[Fraction],
    radius_fraction: Fraction,
    local_override: FaceBox | None = None,
) -> tuple[FaceBox, list[tuple[str, FaceBox]]]:
    if local_override is None:
        local_lower = list(outer.lower)
        local_upper = list(outer.upper)
        for coordinate in variables:
            radius = radius_fraction * (
                outer.upper[coordinate] - outer.lower[coordinate]
            )
            local_lower[coordinate] = max(
                local_lower[coordinate], target[coordinate] - radius
            )
            local_upper[coordinate] = min(
                local_upper[coordinate], target[coordinate] + radius
            )
        local = tighten_simplex_box(
            FaceBox(tuple(local_lower), tuple(local_upper)), variables, dependent
        )
    else:
        local = local_override
    if local is None:
        raise ValueError("local target box is infeasible")
    if any(
        outer.lower[index] > local.lower[index]
        or local.lower[index] > local.upper[index]
        or local.upper[index] > outer.upper[index]
        for index in range(15)
    ):
        raise ValueError("local override lies outside the face box")

    prefix_lower = list(outer.lower)
    prefix_upper = list(outer.upper)
    slabs: list[tuple[str, FaceBox]] = []
    for coordinate in variables:
        lower_upper = list(prefix_upper)
        lower_upper[coordinate] = local.lower[coordinate]
        lower_slab = tighten_simplex_box(
            FaceBox(tuple(prefix_lower), tuple(lower_upper)), variables, dependent
        )
        if lower_slab is not None:
            slabs.append((f"d{coordinate}-lower", lower_slab))

        upper_lower = list(prefix_lower)
        upper_lower[coordinate] = local.upper[coordinate]
        upper_slab = tighten_simplex_box(
            FaceBox(tuple(upper_lower), tuple(prefix_upper)), variables, dependent
        )
        if upper_slab is not None:
            slabs.append((f"d{coordinate}-upper", upper_slab))

        prefix_lower[coordinate] = local.lower[coordinate]
        prefix_upper[coordinate] = local.upper[coordinate]
    return local, slabs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("partitions", type=Path)
    parser.add_argument("faces", type=Path)
    parser.add_argument("--face", required=True)
    parser.add_argument("--center-from", type=Path, required=True)
    parser.add_argument(
        "--target-from", type=Path,
        help="optional diagnostic containing optimized_radial_target.point",
    )
    parser.add_argument(
        "--local-box-from", type=Path,
        help="optional closed face result whose initial_box is the certified region",
    )
    parser.add_argument("--local-radius-fraction", default="1/25")
    parser.add_argument("--form", choices=("fast", "directed"), default="fast")
    parser.add_argument(
        "--bound-method", choices=("affine", "taylor"), default="affine"
    )
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--max-boxes", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.precision < 40 or args.max_boxes < 0:
        parser.error("precision must be at least 40 and max-boxes nonnegative")
    if args.bound_method == "taylor" and args.form != "fast":
        parser.error("the Taylor radial path is currently proposal-only")
    radius_fraction = rational(args.local_radius_fraction)
    if not ZERO < radius_fraction < 1:
        parser.error("local radius fraction must lie in (0,1)")
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
    partitions = json.loads(args.partitions.read_text(encoding="utf-8"))
    record = next(record for record in partitions["leaves"] if record["leaf"] == leaf_name)
    if not record.get("partition_identity_exact") or not record.get(
        "nonnegative_on_product_box_exact"
    ):
        raise ValueError("rational affine partition has not been certified")
    denominator = int(record["coefficient_denominator"])
    coefficients = [
        [Fraction(int(value), denominator) for value in row]
        for row in record["coefficient_numerators"]
    ]
    verify_affine_partition_identity(coefficients)
    coordinates = [int(value) for value in record["feature_coordinates"]]

    frontier = json.loads((root / manifest["global_checkpoint"]).read_text(encoding="utf-8"))
    selected = sorted(frontier["queue"], key=lambda entry: entry["priority"])[0]["evaluated"]
    lower = [rational(value) for value in selected["lower"]]
    upper = [rational(value) for value in selected["upper"]]
    gate = json.loads((root / str(leaf["gate"]).replace("\\", "/")).read_text(encoding="utf-8"))
    lower, upper = chamber_bounds(lower, upper, gate["restrictions"])
    outer, variables, dependent = apply_face(lower, upper, matches[0])
    target_payload = json.loads(
        (args.target_from or args.center_from).read_text(encoding="utf-8")
    )
    target_values = (
        target_payload["optimized_radial_target"]["point"]
        if "optimized_radial_target" in target_payload
        else target_payload["point"]
    )
    target = [rational(value) for value in target_values]
    local_override = None
    if args.local_box_from is not None:
        local_payload = json.loads(args.local_box_from.read_text(encoding="utf-8"))
        initial_box = local_payload["initial_box"]
        local_override = FaceBox(
            tuple(Fraction(value) for value in initial_box["lower"]),
            tuple(Fraction(value) for value in initial_box["upper"]),
        )
    local, slabs = complement_slabs(
        outer,
        variables,
        dependent,
        target,
        radius_fraction,
        local_override,
    )

    bank = json.loads((root / "results/f5-projected-kl-dual-bank-0.22.json").read_text(encoding="utf-8"))
    first_cuts = [exact_cut(raw) for raw in bank["first_cuts"]]
    second_cuts = [exact_cut(raw) for raw in bank["second_cuts"]]
    form_type = FastAffine if args.form == "fast" else AffineInterval

    def evaluate(box: FaceBox):
        if args.bound_method == "taylor":
            upper_value, contributions, diagnostics = fast_radial_taylor_upper(
                box,
                variables,
                dependent,
                coefficients,
                coordinates,
                record["expert_pairs"],
                first_cuts,
                second_cuts,
                target,
            )
            return upper_value, contributions, None, diagnostics
        form = affine_radial_form(
            box,
            variables,
            dependent,
            coefficients,
            coordinates,
            record["expert_pairs"],
            first_cuts,
            second_cuts,
            target,
            form_type,
        )
        if isinstance(form, FastAffine):
            upper_value = form.upper()
            contributions = [abs(value) for value in form.coefficients]
            certificate = None
            residual = form.error
            diagnostics = {
                "center": form.center,
                "coefficient_absolute_sum": sum(
                    abs(value) for value in form.coefficients
                ),
                "nonlinear_error": form.error,
            }
        else:
            certificate = radial_certificate(form)
            upper_value = float(Fraction(certificate["total_upper"]))
            contributions = [
                float(Fraction(value))
                for value in certificate["absolute_coefficient_upper"]
            ]
            residual = float(Fraction(certificate["error_upper"]))
            diagnostics = {
                "center_upper": float(Fraction(certificate["center_upper"])),
                "coefficient_absolute_sum_upper": sum(contributions),
                "nonlinear_error_upper": residual,
            }
        residual_share = residual / max(len(contributions), 1)
        contributions = [value + residual_share for value in contributions]
        return upper_value, contributions, certificate, diagnostics

    initial_records = []
    for slab_name, slab in slabs:
        initial_records.append({"name": slab_name, **serialized_box(slab)})
    if args.resume is None:
        queue = []
        identifier = 0
        for slab_name, slab in slabs:
            upper_value, contributions, certificate, diagnostics = evaluate(slab)
            heapq.heappush(
                queue,
                (
                    -upper_value,
                    identifier,
                    slab_name,
                    slab,
                    contributions,
                    certificate,
                    diagnostics,
                ),
            )
            identifier += 1
        processed = 0
        closed_records = []
    else:
        resumed = json.loads(args.resume.read_text(encoding="utf-8"))
        if (
            resumed.get("leaf") != leaf_name
            or resumed.get("face") != args.face
            or resumed.get("form") != args.form
            or resumed.get("bound_method") != args.bound_method
            or resumed.get("local_radius_fraction") != fraction_text(radius_fraction)
            or resumed.get("variables") != variables
            or resumed.get("target") != [fraction_text(value) for value in target]
        ):
            raise ValueError("radial-cover checkpoint does not match this run")
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
                    -float(entry["radial_upper"]),
                    identifier,
                    entry["slab"],
                    box,
                    [float(value) for value in entry["contributions"]],
                    None,
                    entry.get("form_diagnostics"),
                ),
            )
            identifier += 1
        processed = int(resumed["processed_boxes"])
        closed_records = list(resumed.get("closed_box_records", []))
    while queue and processed < args.max_boxes:
        (
            neg_upper,
            _,
            slab_name,
            box,
            contributions,
            certificate,
            diagnostics,
        ) = heapq.heappop(queue)
        upper_value = -neg_upper
        if upper_value < 0.0:
            closed_records.append(
                {
                    "slab": slab_name,
                    **serialized_box(box),
                    "radial_upper": upper_value,
                    "certificate": certificate,
                    "form_diagnostics": diagnostics,
                }
            )
            processed += 1
            continue
        split_position = max(range(len(variables)), key=contributions.__getitem__)
        dimension = variables[split_position]
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
            (
                child_upper_value,
                child_contributions,
                child_certificate,
                child_diagnostics,
            ) = evaluate(child)
            heapq.heappush(
                queue,
                (
                    -child_upper_value,
                    identifier,
                    slab_name,
                    child,
                    child_contributions,
                    child_certificate,
                    child_diagnostics,
                ),
            )
            identifier += 1
        processed += 1
        if processed % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "closed": len(closed_records),
                        "pending": len(queue),
                        "worst_radial_upper": -queue[0][0] if queue else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    payload = {
        "status": (
            "floating radial-cover proposal closed"
            if args.form == "fast" and not queue
            else "floating radial-cover proposal remains open"
            if args.form == "fast"
            else "directed radial cover closed"
            if not queue
            else "directed radial cover remains open"
        ),
        "rigorous": args.form == "directed",
        "scope": "one candidate face outside its certified local box",
        "leaf": leaf_name,
        "face": args.face,
        "form": args.form,
        "bound_method": args.bound_method,
        "precision": args.precision,
        "resume_source": str(args.resume) if args.resume is not None else None,
        "center_source": str(args.center_from),
        "target_source": str(args.target_from or args.center_from),
        "local_box_source": (
            str(args.local_box_from) if args.local_box_from is not None else None
        ),
        "target": [fraction_text(value) for value in target],
        "local_radius_fraction": fraction_text(radius_fraction),
        "outer_box": serialized_box(outer),
        "local_box": serialized_box(local),
        "initial_slabs": initial_records,
        "variables": variables,
        "dependent_coordinates": dependent,
        "processed_boxes": processed,
        "closed_boxes": len(closed_records),
        "pending_boxes": len(queue),
        "worst_radial_upper": -queue[0][0] if queue else None,
        "closed_box_records": closed_records,
        "pending_box_records": [
            {
                "slab": slab_name,
                **serialized_box(box),
                "radial_upper": -negative_upper,
                "contributions": contributions,
                "form_diagnostics": diagnostics,
            }
            for (
                negative_upper,
                _,
                slab_name,
                box,
                contributions,
                _,
                diagnostics,
            ) in sorted(queue)
        ],
    }
    if args.output is not None:
        from explore_projected_kl_box_cover import write_json_atomic

        write_json_atomic(args.output, payload)
    if args.summary_only:
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in (
                        "status",
                        "rigorous",
                        "form",
                        "bound_method",
                        "local_radius_fraction",
                        "processed_boxes",
                        "closed_boxes",
                        "pending_boxes",
                        "worst_radial_upper",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
