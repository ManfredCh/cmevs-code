#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import candidate_by_id, cell_set, read_json, read_jsonl, safe_div, selected_ids, universe_cells, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selected-view coverage.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--validity", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    selected_doc = read_json(args.selected)
    by_id = candidate_by_id(candidates)
    universe = universe_cells(candidates)

    covered: set[str] = set()
    missing_ids: list[str] = []
    for cid in selected_ids(selected_doc):
        if cid not in by_id:
            missing_ids.append(cid)
            continue
        covered.update(cell_set(by_id[cid]))

    invalid_rate = ""
    if args.validity and args.validity.exists():
        validity_rows = read_jsonl(args.validity)
        invalid_rate = round(
            safe_div(sum(1 for row in validity_rows if not bool(row.get("valid", False))), len(validity_rows)),
            6,
        )

    write_csv(
        args.output,
        [
            {
                "scene_id": selected_doc.get("scene_id", "unknown"),
                "method": selected_doc.get("method", "unknown"),
                "num_candidates": len(candidates),
                "num_selected": len(selected_doc.get("selected_viewpoints", [])),
                "universe_cells": len(universe),
                "covered_cells": len(covered),
                "coverage": round(safe_div(len(covered), max(1, len(universe))), 6),
                "invalid_rate": invalid_rate,
                "missing_selected_ids": ";".join(missing_ids),
            }
        ],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
