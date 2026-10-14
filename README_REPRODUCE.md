# Reproducibility Guide

This guide maps the code release to the experiments in the paper. The primary reproducibility path is Blender indoor; other sources are retained as secondary adapters for robustness checks.

## Primary Asset Requirement

| Source | Expected Input | Primary Command |
| --- | --- | --- |
| Blender indoor | `.blend` scenes | `scripts/run_blender_indoor.sh` |

The repository does not redistribute scene assets. Reviewers can run the no-data smoke test immediately, and can run the full path after mounting local `.blend` scenes.

## Secondary Assets

| Source | Expected Input | Config |
| --- | --- | --- |
| Blender outdoor / generic meshes | `.glb` or `.gltf` | `configs/blender_outdoor.yaml` |
| HM3D | `.glb` or `.gltf` plus optional semantic/navmesh files | `configs/hm3d.yaml` |
| ScanNet++ | `.ply` | `configs/scannetpp.yaml` |

## Building Blocks Available in This Release

| Module | Purpose | Entry Point |
| --- | --- | --- |
| Candidate generation | Phase 1 of §3 — produce \(\mathcal{P}_\varphi\) | `scripts/build_candidates.py` |
| Conflict-aware selection | Phase 2 of §3 — greedy with \(s_t = G_t - \lambda L_t + \beta B_t\) | `scripts/select_views.py` |
| Selected-view rendering | Phase 3 — final ERP render from chosen candidates | `scripts/render_selected.py` |
| Coverage metric | §6.1 high-resolution oracle coverage | `scripts/evaluate_coverage.py` |
| Oracle-gain validation | §6.2 warping vs. pre-render-all comparison | `scripts/evaluate_oracle_gap.py` |
| Quality audit | Appendix F.2 50-frame audit | `scripts/audit_quality.py` |
| Run summarization | Aggregate per-scene `selected_frames.json` into a CSV | `scripts/summarize_blender_indoor_run.py` |
| Audit summarization | Aggregate per-frame audit results into a CSV | `scripts/summarize_quality_audit.py` |

The §6 driver scripts that orchestrate these building blocks across an entire baseline sweep (e.g., the `K\!\in\!\{8,16,24,32\}` table of §6.1, the \(\lambda\) sweep of §6.5, and the four-source benchmark of §6.6) are scheduled to be released alongside the camera-ready paper.

## Minimal Review Run

```bash
bash scripts/run_tiny.sh
```

This validates the Blender-indoor-style metadata format, greedy selection loop, render-output contract, coverage metric, oracle-gap script, and quality audit script — end-to-end without any private scene assets.

## Blender-Indoor Full Run

```bash
DRY_RUN=1 \
BLENDER=/path/to/blender \
INPUT_DIR=/path/to/blend_scenes \
OUTPUT_ROOT=outputs/blender_indoor \
bash scripts/run_blender_indoor.sh
```

After confirming the detected scene list, remove `DRY_RUN=1`:

```bash
BLENDER=/path/to/blender \
INPUT_DIR=/path/to/blend_scenes \
OUTPUT_ROOT=outputs/blender_indoor \
NUM_FRAMES=30 \
RESOLUTION=2048,1024 \
GRID_SPACING=0.5 \
bash scripts/run_blender_indoor.sh
```

## Metric Scripts

The native Blender-indoor pipeline emits `selected_frames.json` under each scene output directory. Summarize a completed run with:

```bash
python3 scripts/summarize_blender_indoor_run.py \
  --output-root outputs/blender_indoor \
  --output outputs/blender_indoor/results/coverage_main.csv
```

If you have consolidated candidate and selection metadata into the normalized JSONL/JSON contract used by the smoke test, use:

```bash
python3 scripts/evaluate_coverage.py \
  --candidates outputs/blender_indoor/metadata/candidates.jsonl \
  --selected outputs/blender_indoor/metadata/selected_viewpoints.json \
  --output outputs/blender_indoor/results/coverage_main.csv

python3 scripts/evaluate_oracle_gap.py \
  --candidates outputs/blender_indoor/metadata/candidates.jsonl \
  --selected outputs/blender_indoor/metadata/selected_viewpoints.json \
  --output outputs/blender_indoor/results/oracle_validation.csv

python3 scripts/audit_quality.py \
  --render-dir outputs/blender_indoor/renders \
  --metadata outputs/blender_indoor/metadata/selected_viewpoints.json \
  --output outputs/blender_indoor/results/audit_50_frames.csv
```

The exact dataset paths should be adapted to the local machine. Do not commit generated data, logs, checkpoints, third-party repositories, or scene assets to the anonymous code repository.
