# Code Submission Checklist

Use this checklist before pasting the Code URL into OpenReview.

| Requirement | Status | Where to Check |
| --- | --- | --- |
| Anonymous repository content | Ready | `bash scripts/check_anonymity.sh` |
| Executable smoke test | Ready | `bash scripts/run_tiny.sh` |
| Primary Blender-indoor entry point | Ready | `scripts/run_blender_indoor.sh` |
| Native Blender-indoor result summarizer | Ready | `scripts/summarize_blender_indoor_run.py` |
| Per-stage evaluation building blocks | Ready | `scripts/evaluate_coverage.py`, `scripts/evaluate_oracle_gap.py`, `scripts/audit_quality.py` |
| Clear README | Ready | `README.md` |
| Reproducibility guide | Ready | `README_REPRODUCE.md` |
| Environment specification | Ready | `environment.yml`, `requirements.txt` |
| Dataset paths excluded | Ready | `.gitignore`, `data/README.md` |
| Third-party checkpoints excluded | Ready | `.gitignore`, `third_party/README.md` |
| License included | Ready | `LICENSE` |
| Metadata schemas included | Ready | `metadata_examples/` |
| Generated outputs excluded before upload | Check manually | `find . -maxdepth 2 -type d -name outputs` |

Recommended final commands:

```bash
bash scripts/run_tiny.sh
rm -rf outputs
bash scripts/check_anonymity.sh
```

Then upload the repository or zip archive.
