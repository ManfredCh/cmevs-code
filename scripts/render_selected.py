#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _common import ensure_dir, read_json, write_json, write_jsonl, write_npy_f4, write_solid_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize selected-view render artifacts.")
    parser.add_argument("--selected", type=Path, required=True, help="selected_viewpoints.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny"), help="Run output directory.")
    parser.add_argument("--tiny", action="store_true", help="Write deterministic placeholder render artifacts.")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    return parser.parse_args()


def color_from_id(candidate_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    return tuple(48 + int(digest[i]) % 176 for i in range(3))


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if not args.tiny:
        raise SystemExit("Real rendering is handled by pipelines/run_full_pipeline.py. Use --tiny for smoke-test artifacts.")

    selected_doc = read_json(args.selected)
    render_dir = ensure_dir(args.output_dir / "renders")
    rows = []
    for item in selected_doc.get("selected_viewpoints", []):
        cid = str(item["candidate_id"])
        stem = f"{int(item['rank']):03d}_{cid}"
        rgb_path = render_dir / f"{stem}_rgb.png"
        depth_path = render_dir / f"{stem}_depth.npy"
        pose_path = render_dir / f"{stem}_pose.json"
        write_solid_png(rgb_path, args.width, args.height, color_from_id(cid))
        write_npy_f4(depth_path, args.height, args.width, 1.0 + 0.05 * int(item["rank"]))
        write_json(
            pose_path,
            {
                "candidate_id": cid,
                "position": item.get("position", [0.0, 0.0, 0.0]),
                "yaw_deg": item.get("yaw_deg", 0.0),
                "coordinate_frame": "right-handed, x-right, y-up, z-forward",
            },
        )
        rows.append(
            {
                "candidate_id": cid,
                "rgb": rel(rgb_path, args.output_dir),
                "depth": rel(depth_path, args.output_dir),
                "pose": rel(pose_path, args.output_dir),
                "valid": True,
                "invalid_reason": "",
            }
        )
    write_jsonl(render_dir / "validity.jsonl", rows)
    print(f"Wrote {len(rows)} tiny render records to {render_dir}")


if __name__ == "__main__":
    main()
