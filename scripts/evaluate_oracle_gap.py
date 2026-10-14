#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import candidate_by_id, pearson, read_json, read_jsonl, selected_ids, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected-view marginal gains against oracle annotations.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    selected_doc = read_json(args.selected)
    by_id = candidate_by_id(candidates)
    selected_by_id = {str(row["candidate_id"]): row for row in selected_doc.get("selected_viewpoints", [])}

    oracle: list[float] = []
    scores: list[float] = []
    marginal: list[float] = []
    gaps: list[float] = []
    for cid in selected_ids(selected_doc):
        candidate = by_id.get(cid)
        selected = selected_by_id.get(cid)
        if not candidate or not selected or "oracle_gain" not in candidate:
            continue
        o = float(candidate["oracle_gain"])
        s = float(selected.get("score", 0.0))
        m = float(selected.get("marginal_gain", 0.0))
        oracle.append(o)
        scores.append(s)
        marginal.append(m)
        gaps.append(abs(o - m))

    row = {
        "scene_id": selected_doc.get("scene_id", "unknown"),
        "num_pairs": len(oracle),
        "score_oracle_pearson": pearson(scores, oracle),
        "marginal_oracle_pearson": pearson(marginal, oracle),
        "mean_abs_oracle_gap": sum(gaps) / len(gaps) if gaps else math.nan,
        "max_abs_oracle_gap": max(gaps) if gaps else math.nan,
    }
    write_csv(args.output, [row])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
