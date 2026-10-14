#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import read_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated CM-EVS render artifacts.")
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def has_magic(path: Path, magic: bytes) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as f:
        return f.read(len(magic)) == magic


def main() -> None:
    args = parse_args()
    selected_doc = read_json(args.metadata)
    rows = []
    for item in selected_doc.get("selected_viewpoints", [])[: args.limit]:
        cid = str(item["candidate_id"])
        rank = int(item["rank"])
        stem = f"{rank:03d}_{cid}"
        rgb = args.render_dir / f"{stem}_rgb.png"
        depth = args.render_dir / f"{stem}_depth.npy"
        pose = args.render_dir / f"{stem}_pose.json"
        pose_ok = False
        if pose.exists():
            try:
                read_json(pose)
                pose_ok = True
            except Exception:
                pose_ok = False
        rows.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "rgb_exists": rgb.exists(),
                "rgb_png_magic": has_magic(rgb, b"\x89PNG\r\n\x1a\n"),
                "depth_exists": depth.exists(),
                "depth_npy_magic": has_magic(depth, b"\x93NUMPY"),
                "pose_exists": pose.exists(),
                "pose_json_valid": pose_ok,
                "passed": rgb.exists() and depth.exists() and pose_ok,
            }
        )
    write_csv(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
