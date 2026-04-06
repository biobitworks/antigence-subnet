# Public GitHub Publish Checklist

Use this checklist for the curated public artifact at `dist/public-mirror`.
Do not make the private working repository public as-is.

## Artifact To Publish

- Source of truth: private working repo
- Public artifact: `dist/public-mirror`
- Public-safety approval record:
  - `.planning/phases/101-public-release-dry-run-launch-approval/101-dry-run-report.md`
  - `.planning/phases/101-public-release-dry-run-launch-approval/101-launch-approval.json`

## Pre-Publish Checks

1. Rebuild the mirror:

```bash
python scripts/build_public_mirror.py \
  --manifest .planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json \
  --output-dir dist/public-mirror \
  --clean
```

2. Confirm excluded surfaces stay out:

```bash
rg -n "docs/pitch|PUBLIC_RELEASE|\\.planning/ROADMAP|CLAUDE\\.md|continue-here" dist/public-mirror
```

Expected: no matches.

3. Spot-check the top-level artifact:

```bash
find dist/public-mirror -maxdepth 2 -type f | sort | sed -n '1,80p'
```

Confirm at minimum:
- `README.md`
- `LICENSE`
- `.env.example`
- `.env.phase94.example`
- `pyproject.toml`
- `docs/manuscript.md`

4. Review the generated artifact diff or tree before pushing:

```bash
cd dist/public-mirror
git init
git add .
git status --short
```

## Publish Options

### Option A: New Public Repository

Recommended.

```bash
cd dist/public-mirror
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <new-public-repo-url>
git push -u origin main
```

### Option B: Public Release Branch

Only use this if you intentionally want the current private repo history and
governance around the mirror build to stay tied to the same GitHub repository.

Push only from `dist/public-mirror` contents, not from the private repo root.

## Post-Publish Follow-Through

These do not block publishing the curated mirror, but they are still open:

- Observe remote GitHub Actions on the intended release branch or tag
- Publish the Zenodo deposit and backfill the real DOI
- Submit the Computers & Security package and record the submission ID

## Do Not Do

- Do not flip the private working repo public directly without a separate review
- Do not publish `.planning/`, `docs/pitch/`, `PUBLIC_RELEASE.md`, or workflow-only files
- Do not treat the current dirty repo root as the public artifact
