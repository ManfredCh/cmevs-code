#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Author-identifying strings and absolute path prefixes that must not appear
# anywhere in the repository (excluding this script itself and ignored dirs).
PATTERN='/Users/|/home/|/root/|plob|chasedream|autodl|Developer|Jiale|jiale|刘|李佳'

# Directories / files that legitimately contain those substrings or are
# generated artefacts that we do not commit.
EXCLUDES=(
  --exclude-dir=.git
  --exclude-dir=outputs
  --exclude-dir=__pycache__
  --exclude-dir=.venv
  --exclude-dir=venv
  --exclude=check_anonymity.sh
  --exclude=*.zip
  --exclude=*.tar.gz
)

echo "[1/3] Scanning for local paths and author-identifying strings..."
if grep -rnE "${EXCLUDES[@]}" "$PATTERN" "$ROOT" 2>/dev/null; then
  echo "Anonymity check failed: remove or rewrite the matched strings above." >&2
  exit 1
fi

echo "[2/3] Checking for cache and platform files..."
matches=$(find "$ROOT" \( -name '__pycache__' -o -name '.DS_Store' \) -not -path '*/.git/*' 2>/dev/null)
if [ -n "$matches" ]; then
  echo "Anonymity check failed: remove cache/platform files below." >&2
  echo "$matches" >&2
  exit 1
fi

echo "[3/3] Checking required top-level files..."
test -f "$ROOT/README.md"
test -f "$ROOT/LICENSE"

echo "Anonymity check passed."
