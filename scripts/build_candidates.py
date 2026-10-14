#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, copy_file, ensure_dir, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or copy CM-EVS candidate metadata.")
    parser.add_argument("--tiny", action="store_true", help="Use the bundled tiny metadata example.")
    parser.add_argument("--input", type=Path, default=None, help="Existing candidate JSONL to normalize/copy.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny"), help="Run output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_dir = ensure_dir(args.output_dir / "metadata")
    out_path = metadata_dir / "candidates.jsonl"

    if args.tiny:
        src = REPO_ROOT / "examples" / "metadata" / "candidates.jsonl"
        mode = "tiny"
    elif args.input:
        src = args.input
        mode = "copy"
    else:
        raise SystemExit("Provide --tiny or --input. Full generation is handled by pipelines/run_full_pipeline.py.")

    rows = read_jsonl(src)
    copy_file(src, out_path)
    write_json(
        metadata_dir / "source_manifest.json",
        {
            "mode": mode,
            "candidate_file": str(out_path.relative_to(args.output_dir)),
            "num_candidates": len(rows),
            "num_valid_candidates": sum(1 for row in rows if bool(row.get("valid", True))),
            "note": "Tiny mode validates the metadata contract and does not reproduce paper-scale results.",
        },
    )
    print(f"Wrote {out_path} with {len(rows)} candidates")


if __name__ == "__main__":
    main()
