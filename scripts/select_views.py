#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    cell_set,
    ensure_dir,
    read_jsonl,
    safe_div,
    universe_cells,
    valid_candidates,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic greedy CM-EVS view selection.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny"), help="Run output directory.")
    parser.add_argument("--budget", type=int, default=4, help="Maximum number of selected views.")
    parser.add_argument("--lambda-conflict", type=float, default=0.35, help="Conflict-prior penalty weight.")
    parser.add_argument("--min-gain", type=float, default=0.01, help="Stop when marginal coverage falls below this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    candidates = valid_candidates(rows)
    universe = universe_cells(candidates)
    universe_size = max(1, len(universe))

    covered: set[str] = set()
    selected: list[dict] = []
    log_rows: list[dict] = []
    used: set[str] = set()

    for step in range(args.budget):
        best = None
        best_tuple = None
        for candidate in candidates:
            cid = str(candidate["candidate_id"])
            if cid in used:
                continue
            cells = cell_set(candidate)
            new_cells = cells - covered
            marginal_gain = safe_div(len(new_cells), universe_size)
            probe = float(candidate.get("single_view_probe_coverage", 0.0))
            conflict = float(candidate.get("conflict_prior", 0.0))
            score = marginal_gain + 0.15 * probe - args.lambda_conflict * conflict
            score_tuple = (score, marginal_gain, -conflict, cid)
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best = (candidate, new_cells, score, marginal_gain)

        if best is None:
            break

        candidate, new_cells, score, marginal_gain = best
        if selected and marginal_gain < args.min_gain:
            break

        cid = str(candidate["candidate_id"])
        used.add(cid)
        covered.update(new_cells)
        rank = len(selected)
        selected.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "position": candidate.get("position", [0.0, 0.0, 0.0]),
                "yaw_deg": float(candidate.get("yaw_deg", 0.0)),
                "score": round(float(score), 6),
                "marginal_gain": round(float(marginal_gain), 6),
            }
        )
        log_rows.append(
            {
                "step": step,
                "candidate_id": cid,
                "score": round(float(score), 6),
                "marginal_gain": round(float(marginal_gain), 6),
                "coverage_after": round(safe_div(len(covered), universe_size), 6),
                "conflict_prior": float(candidate.get("conflict_prior", 0.0)),
            }
        )

    scene_id = str(candidates[0].get("scene_id", "unknown")) if candidates else "unknown"
    metadata_dir = ensure_dir(args.output_dir / "metadata")
    write_json(
        metadata_dir / "selected_viewpoints.json",
        {
            "scene_id": scene_id,
            "method": "cmevs_greedy_conflict_minimized",
            "selected_viewpoints": selected,
            "summary": {
                "budget": args.budget,
                "lambda_conflict": args.lambda_conflict,
                "min_gain": args.min_gain,
                "num_candidates": len(rows),
                "num_valid_candidates": len(candidates),
                "num_selected": len(selected),
                "coverage": round(safe_div(len(covered), universe_size), 6),
                "universe_cells": len(universe),
            },
        },
    )
    write_jsonl(metadata_dir / "per_step_log.jsonl", log_rows)
    print(f"Selected {len(selected)} viewpoints; final coverage={safe_div(len(covered), universe_size):.3f}")


if __name__ == "__main__":
    main()
