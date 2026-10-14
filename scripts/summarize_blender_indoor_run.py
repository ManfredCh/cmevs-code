#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import read_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize native Blender-indoor pipeline outputs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/blender_indoor"))
    parser.add_argument("--output", type=Path, default=Path("outputs/blender_indoor/results/coverage_main.csv"))
    return parser.parse_args()


def find_selected_files(output_root: Path) -> list[Path]:
    patterns = [
        "*/input/frame_selection/selected_frames.json",
        "*/frame_selection/selected_frames.json",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(output_root.glob(pattern))
    return sorted(set(files))


def summarize_one(path: Path, output_root: Path) -> dict:
    doc = read_json(path)
    frames = doc.get("frames", [])
    pred_gains = [float(row.get("gain", 0.0)) for row in frames]
    actual_gains = [float(row.get("actual_gain", 0.0)) for row in frames]
    scores = [float(row.get("score", 0.0)) for row in frames]
    deltas = [float(row.get("delta_ratio", 0.0)) for row in frames]
    scene_dir = path.parents[2] if path.parent.name == "frame_selection" and path.parent.parent.name == "input" else path.parents[1]
    scene_id = scene_dir.name
    total_actual_gain = min(1.0, sum(actual_gains))
    return {
        "scene_id": scene_id,
        "scene_file": doc.get("scene", ""),
        "scene_format": doc.get("scene_format", ""),
        "selected_frames": int(doc.get("total_frames", len(frames))),
        "candidates_count": int(doc.get("candidates_count", 0)),
        "coverage_proxy_from_actual_gain": round(total_actual_gain, 6),
        "mean_predicted_gain": round(sum(pred_gains) / len(pred_gains), 6) if pred_gains else 0.0,
        "mean_actual_gain": round(sum(actual_gains) / len(actual_gains), 6) if actual_gains else 0.0,
        "last_actual_gain": round(actual_gains[-1], 6) if actual_gains else 0.0,
        "last_delta_ratio": round(deltas[-1], 6) if deltas else 0.0,
        "last_score": round(scores[-1], 6) if scores else 0.0,
        "selected_json": str(path.relative_to(output_root)),
    }


def main() -> None:
    args = parse_args()
    selected_files = find_selected_files(args.output_root)
    if not selected_files:
        raise SystemExit(f"No selected_frames.json files found under {args.output_root}")
    rows = [summarize_one(path, args.output_root) for path in selected_files]
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} scene summaries to {args.output}")


if __name__ == "__main__":
    main()

