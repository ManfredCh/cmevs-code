#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _common import safe_div, write_csv, write_latex_table, write_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frame-level quality audit results.")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.audit_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    passed = sum(1 for row in rows if str(row.get("passed", "")).lower() == "true")
    rgb_ok = sum(1 for row in rows if str(row.get("rgb_png_magic", "")).lower() == "true")
    depth_ok = sum(1 for row in rows if str(row.get("depth_npy_magic", "")).lower() == "true")
    pose_ok = sum(1 for row in rows if str(row.get("pose_json_valid", "")).lower() == "true")
    summary = [
        {
            "audited_frames": total,
            "passed_frames": passed,
            "pass_rate": round(safe_div(passed, total), 6),
            "rgb_valid_rate": round(safe_div(rgb_ok, total), 6),
            "depth_valid_rate": round(safe_div(depth_ok, total), 6),
            "pose_valid_rate": round(safe_div(pose_ok, total), 6),
        }
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "audit_summary.csv", summary)
    write_markdown_table(args.output_dir / "audit_summary.md", summary)
    write_latex_table(
        args.output_dir / "audit_summary.tex",
        summary,
        caption="Render and metadata quality audit.",
        label="tab:quality_audit",
    )
    print(f"Wrote quality-audit summary to {args.output_dir}")


if __name__ == "__main__":
    main()

