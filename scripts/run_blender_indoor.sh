#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BLENDER="${BLENDER:-}"
INPUT_DIR="${INPUT_DIR:-$ROOT/data/blender_indoor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/blender_indoor}"
NUM_FRAMES="${NUM_FRAMES:-30}"
RESOLUTION="${RESOLUTION:-2048,1024}"
GRID_SPACING="${GRID_SPACING:-0.5}"
MIN_FRAMES="${MIN_FRAMES:-5}"
STOP_GAIN="${STOP_GAIN:-0.08}"
STOP_SCORE="${STOP_SCORE:--0.3}"
STOP_DELTA="${STOP_DELTA:-0.08}"
ROTATION_TYPE="${ROTATION_TYPE:-random_yaw}"

if [[ -z "$BLENDER" ]]; then
  echo "Set BLENDER=/path/to/blender before running the Blender-indoor pipeline." >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "INPUT_DIR does not exist: $INPUT_DIR" >&2
  echo "Place .blend scenes under data/blender_indoor or set INPUT_DIR=/path/to/blend_scenes." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

export PYTHONPATH="$ROOT:$ROOT/pipelines:${PYTHONPATH:-}"

"$PYTHON_BIN" "$ROOT/pipelines/run_full_pipeline.py" \
  --blender "$BLENDER" \
  --input-dir "$INPUT_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --num-frames "$NUM_FRAMES" \
  --resolution "$RESOLUTION" \
  --grid-spacing "$GRID_SPACING" \
  --min-frames "$MIN_FRAMES" \
  --stop-gain "$STOP_GAIN" \
  --stop-score "$STOP_SCORE" \
  --stop-delta "$STOP_DELTA" \
  --rotation-type "$ROTATION_TYPE" \
  "${EXTRA_ARGS[@]}"

