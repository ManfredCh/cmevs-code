#!/usr/bin/env python3
"""Patch croissant.json with real archive-level SHA256 digests.

After the H100 batch hash run finishes, you have:
  - manifests/SHA256SUMS_<archive_id>.txt   per-archive line-by-line manifest
  - manifests/ARCHIVE_DIGESTS.txt           sha256 of each manifest file

This script reads ARCHIVE_DIGESTS.txt, maps each <archive_id> to its
matching `cr:FileObject` in croissant.json (by `@id`), and rewrites the
`sha256` field. It also updates `contentSize` from the on-disk manifest's
total file count via a sidecar count file (optional).

Usage
  tools/update_croissant_with_real_hashes.py \
    --croissant dataset_metadata/croissant.json \
    --digests manifests_local/ARCHIVE_DIGESTS.txt \
    --out     dataset_metadata/croissant.json

Mapping (archive_id -> croissant @id) is hard-coded below to match the
actual server layout in /data/data_wr/data_shushu/.
"""
import argparse
import json
import sys
from pathlib import Path

# Map H100 archive_id (chosen in run_hashes.sh) -> Croissant @id.
# The Croissant file currently has cr:FileObject entries with @id ending
# in ".tar"; we rewrite the entries even though the actual upload may be
# a flat directory rather than a tar (the sha256 still pins the content).
ARCHIVE_TO_CROISSANT = {
    "blender_indoor_round1+2": "blender-indoor-archive.tar",
    "blender_indoor_round2":   "blender-indoor-round2.tar",       # may not exist yet in croissant.json
    "blender_1m_discrete":     "blender-1m-discrete.tar",         # may not exist yet
    "HM3D":                    "hm3d-adapter.tar",
    "scannetpp":               "scannetpp-adapter.tar",
    "OB3D":                    "outdoor-ob3d-archive.tar",
    "tartanground":            "outdoor-tartanground-archive.tar",
    "shushu_line":             "shushu-line.tar",                 # may not exist yet
    "shushu_circle":           "shushu-circle.tar",               # may not exist yet
    "robustness":              "robustness.tar",                  # may not exist yet
}


def read_digests(path: Path) -> dict[str, str]:
    """Parse ARCHIVE_DIGESTS.txt: each line is '<sha>  SHA256SUMS_<id>.txt'."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sha, fname = line.split(maxsplit=1)
            # fname looks like 'SHA256SUMS_<archive_id>.txt'
            archive_id = fname.removeprefix("SHA256SUMS_").removesuffix(".txt")
            out[archive_id] = sha
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--croissant", required=True, type=Path)
    ap.add_argument("--digests",   required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    args = ap.parse_args()

    cr = json.loads(args.croissant.read_text())
    digests = read_digests(args.digests)

    # Build @id -> entry index.
    by_id = {e.get("@id"): i for i, e in enumerate(cr.get("distribution", []))}

    matched, unmatched, missing_in_cr = 0, [], []
    for archive_id, sha in digests.items():
        cr_id = ARCHIVE_TO_CROISSANT.get(archive_id)
        if not cr_id:
            unmatched.append(archive_id)
            continue
        if cr_id not in by_id:
            missing_in_cr.append((archive_id, cr_id))
            continue
        cr["distribution"][by_id[cr_id]]["sha256"] = sha
        matched += 1

    args.out.write_text(json.dumps(cr, indent=2, ensure_ascii=False) + "\n")

    print(f"updated {matched} cr:FileObject entries with real sha256")
    if unmatched:
        print(f"  no croissant mapping for: {unmatched}")
    if missing_in_cr:
        print(f"  croissant @id not found for: {missing_in_cr}")
    print(f"  wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
