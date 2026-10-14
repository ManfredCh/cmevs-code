#!/usr/bin/env bash
# tools/make_sha256sums.sh
#
# Generate / verify the SHA256SUMS manifest for the CM-EVS release.
#
# Why this exists:
#   The dataset release is split across multiple roots — code, sample data,
#   regen scripts for HM3D / ScanNet++, and the public Blender frames. The
#   reviewer (or any downloader) needs a single text file that lists the
#   sha256 of every distributable file, so they can verify integrity with a
#   one-liner: `shasum -a 256 -c SHA256SUMS`.
#
# Usage:
#   tools/make_sha256sums.sh generate [ROOT [OUT_FILE]]
#   tools/make_sha256sums.sh verify   [ROOT [SUMS_FILE]]
#
# Defaults:
#   ROOT      = current directory
#   OUT_FILE  = SHA256SUMS (in ROOT)
#   SUMS_FILE = SHA256SUMS (in ROOT)
#
# Examples:
#   # 1) refresh manifest for the code release we are sitting in
#   tools/make_sha256sums.sh generate
#
#   # 2) make a manifest for a freshly populated data drop somewhere else
#   tools/make_sha256sums.sh generate /Volumes/CMEVS/cm-evs-data /Volumes/CMEVS/cm-evs-data/SHA256SUMS
#
#   # 3) verify a downloaded data drop
#   tools/make_sha256sums.sh verify /Volumes/CMEVS/cm-evs-data
#
# Format:
#   <sha256-hex>  <relative-path>
#   one per line, sorted by path. Paths are POSIX-style relative paths from
#   ROOT, so the manifest is portable between machines.
#
# What is excluded by default:
#   - .git/, .DS_Store, *.pyc, __pycache__/, .ipynb_checkpoints/, the
#     SHA256SUMS file itself, and any tar/zip that ROOT itself owns
#     (those are checksummed separately by their containing release root).
#
# Tooling:
#   uses `shasum -a 256` (macOS / BSD / Linux coreutils all ship it).
#   `find` + `xargs -0` handle filenames with spaces.

set -euo pipefail

readonly CMD="${1:-generate}"
readonly ROOT="${2:-$PWD}"
readonly DEFAULT_SUMS="$ROOT/SHA256SUMS"
readonly SUMS="${3:-$DEFAULT_SUMS}"

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: ROOT directory does not exist: $ROOT" >&2
    exit 2
fi

# A POSIX-portable shasum invocation.
hasher() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$@"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$@"
    else
        echo "ERROR: neither shasum nor sha256sum is on PATH" >&2
        exit 3
    fi
}

generate() {
    local out="$SUMS"
    local tmp; tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT

    echo "Hashing files under: $ROOT"
    echo "Writing manifest to: $out"

    cd "$ROOT"

    # Collect every regular file, exclude common junk, sort by path so the
    # manifest is deterministic across machines.
    # Exclusions:
    #   - .git/, __pycache__/, .ipynb_checkpoints/, .DS_Store, *.pyc
    #   - SHA256SUMS itself (chicken/egg)
    #   - dataset_metadata/manifests_h100/  — these are large data-snapshot
    #     manifests (per-frame sha256 of the H100 data drop, ~71 MB total);
    #     they have their own ARCHIVE_DIGESTS.txt for self-verification and
    #     are not part of the redistributable code+metadata package.
    find . -type f \
        -not -path './.git/*' \
        -not -path '*/__pycache__/*' \
        -not -path '*/.ipynb_checkpoints/*' \
        -not -path './dataset_metadata/manifests_h100/*' \
        -not -name '.DS_Store' \
        -not -name '*.pyc' \
        -not -name 'SHA256SUMS' \
        -print0 | LC_ALL=C sort -z | xargs -0 -n 50 shasum -a 256 \
        | sed 's|\./||' \
        > "$tmp"

    mv "$tmp" "$out"
    trap - EXIT

    local n; n="$(wc -l < "$out" | tr -d ' ')"
    echo "OK: hashed $n files."
    echo "Verify with: tools/make_sha256sums.sh verify"
}

verify() {
    local sums="$SUMS"
    if [[ ! -f "$sums" ]]; then
        echo "ERROR: manifest not found: $sums" >&2
        exit 4
    fi

    echo "Verifying files under: $ROOT"
    echo "Against manifest:      $sums"

    cd "$ROOT"
    if shasum -a 256 -c "$sums"; then
        echo ""
        echo "PASS: all checksums match."
    else
        echo ""
        echo "FAIL: at least one file does not match (see above)."
        exit 5
    fi
}

case "$CMD" in
    generate|gen|g)
        generate
        ;;
    verify|check|c)
        verify
        ;;
    -h|--help|help)
        sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
        exit 0
        ;;
    *)
        echo "Unknown command: $CMD" >&2
        echo "Try: $0 --help" >&2
        exit 1
        ;;
esac
