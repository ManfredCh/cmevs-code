#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/outputs/tiny}"

export PYTHONPATH="$ROOT:$ROOT/pipelines:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" "$ROOT/scripts/build_candidates.py" \
  --tiny \
  --output-dir "$OUT"

"$PYTHON_BIN" "$ROOT/scripts/select_views.py" \
  --candidates "$OUT/metadata/candidates.jsonl" \
  --output-dir "$OUT" \
  --budget 4 \
  --lambda-conflict 0.35 \
  --min-gain 0.01

"$PYTHON_BIN" "$ROOT/scripts/render_selected.py" \
  --tiny \
  --selected "$OUT/metadata/selected_viewpoints.json" \
  --output-dir "$OUT"

"$PYTHON_BIN" "$ROOT/scripts/evaluate_coverage.py" \
  --candidates "$OUT/metadata/candidates.jsonl" \
  --selected "$OUT/metadata/selected_viewpoints.json" \
  --validity "$OUT/renders/validity.jsonl" \
  --output "$OUT/results/coverage_main.csv"

"$PYTHON_BIN" "$ROOT/scripts/evaluate_oracle_gap.py" \
  --candidates "$OUT/metadata/candidates.jsonl" \
  --selected "$OUT/metadata/selected_viewpoints.json" \
  --output "$OUT/results/oracle_validation.csv"

"$PYTHON_BIN" "$ROOT/scripts/audit_quality.py" \
  --render-dir "$OUT/renders" \
  --metadata "$OUT/metadata/selected_viewpoints.json" \
  --output "$OUT/results/audit_50_frames.csv"

bash "$ROOT/scripts/check_anonymity.sh"

echo "Tiny CM-EVS smoke test completed: $OUT"
