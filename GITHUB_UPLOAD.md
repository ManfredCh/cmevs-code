# Anonymous GitHub Upload Guide

The NeurIPS review policy requires the code URL to be accessible and anonymized at submission time. Use an anonymous repository service or a repository that does not reveal author identity.

## Final Local Check

Run from the repository root:

```bash
bash scripts/run_tiny.sh
rm -rf outputs
bash scripts/check_anonymity.sh
```

## Initialize and Push

```bash
git init
git add .
git commit -m "Anonymous CM-EVS code release"
git branch -M main
git remote add origin <anonymous-repository-url>
git push -u origin main
```

Before pushing, verify that `git status --short` does not include local scene data, rendered outputs, checkpoints, or third-party dataset assets.

## OpenReview Field

Paste the anonymous repository URL into the **Code URL** field. If the repository is mirrored through an anonymous hosting service, use the anonymized URL rather than a personal GitHub URL.

