#!/usr/bin/env python3
"""Resumably scan a saved frontier against a fixed stationary-dual bank.

This is a numerical audit, not a certificate: SLSQP supplies candidate local
maxima.  The script records every box and never upgrades local optimization to
a proof claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import diagnose_fixed_stationary_box as fixed
import explore_projected_kl_box_cover as projected


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(args: argparse.Namespace) -> dict[str, object]:
    frontier_path = Path(args.frontier)
    bank_path = Path(args.bank)
    output_path = Path(args.output)
    with frontier_path.open("r", encoding="utf-8") as source:
        frontier = json.load(source)
    queue = sorted(frontier["queue"], key=lambda entry: entry["priority"])
    stop = len(queue) if args.stop <= 0 else min(args.stop, len(queue))
    signature = {
        "version": 1,
        "frontier": str(frontier_path),
        "frontier_digest": file_digest(frontier_path),
        "bank": str(bank_path),
        "bank_digest": file_digest(bank_path),
        "start": args.start,
        "stop": stop,
        "restarts": args.restarts,
        "optimizer_maxiter": args.optimizer_maxiter,
        "seed": args.seed,
    }
    records: list[dict[str, object]] = []
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as source:
            saved = json.load(source)
        if saved.get("signature") != signature:
            raise RuntimeError("scan checkpoint signature does not match")
        records = list(saved["records"])

    def save() -> None:
        tested = [
            record
            for record in records
            if record["method"] == "fixed_stationary_bank"
        ]
        unreliable = [
            int(record["box_index"])
            for record in tested
            if int(record["successful_restarts"]) == 0
        ]
        positive = [
            int(record["box_index"])
            for record in records
            if float(record["upper_found"]) > args.target
        ]
        report = {
            "signature": signature,
            "records": records,
            "boxes_recorded": len(records),
            "boxes_requested": stop - args.start,
            "maximum_found": max(
                (float(record["upper_found"]) for record in records),
                default=None,
            ),
            "positive_boxes": positive,
            "unreliable_boxes": unreliable,
            "status": (
                "complete numerical scan; no positive local maximum found"
                if len(records) == stop - args.start
                and not positive
                and not unreliable
                else "numerical scan incomplete or requiring review"
            ),
            "scope": (
                "SLSQP local maximization of a mathematically valid fixed-dual "
                "upper envelope; not a directed-rounding or global-optimality certificate"
            ),
        }
        projected.write_json_atomic(output_path, report)

    next_index = args.start + len(records)
    for box_index in range(next_index, stop):
        projected_upper = float(queue[box_index]["evaluated"]["bound"])
        if args.skip_projected_closed and projected_upper <= args.target:
            record = {
                "box_index": box_index,
                "projected_upper": projected_upper,
                "upper_found": projected_upper,
                "method": "projected_lp",
                "successful_restarts": None,
            }
        else:
            diagnostic_args = argparse.Namespace(
                checkpoint=str(frontier_path),
                box_index=box_index,
                restarts=args.restarts,
                cut_iterations=1,
                target=-1e100,
                seed=args.seed + box_index,
                interior_mass=args.interior_mass,
                maximum_interior_mass=args.maximum_interior_mass,
                maximum_cut_residual=args.maximum_cut_residual,
                include_cuts=False,
                seed_bank=str(bank_path),
                output=None,
                optimizer_maxiter=args.optimizer_maxiter,
            )
            diagnostic = fixed.diagnose(diagnostic_args)
            record = {
                "box_index": box_index,
                "projected_upper": projected_upper,
                "upper_found": float(diagnostic["maximum_found"]),
                "method": "fixed_stationary_bank",
                "successful_restarts": int(
                    diagnostic["successful_restarts"]
                ),
            }
        records.append(record)
        if args.checkpoint_every and len(records) % args.checkpoint_every == 0:
            save()
        if args.progress_every and len(records) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "through_box": box_index,
                        "maximum_found": max(
                            float(value["upper_found"]) for value in records
                        ),
                        "positive": sum(
                            float(value["upper_found"]) > args.target
                            for value in records
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    save()
    with output_path.open("r", encoding="utf-8") as source:
        return json.load(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--optimizer-maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--interior-mass", type=float, default=1e-6)
    parser.add_argument("--maximum-interior-mass", type=float, default=0.5)
    parser.add_argument("--maximum-cut-residual", type=float, default=1e-6)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--skip-projected-closed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(scan(parse_args()), indent=2, sort_keys=True))
