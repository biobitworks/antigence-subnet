# Public Release Notes

This file ships in the public mirror of the Antigence Subnet. It documents
what the published artifact is and is not, so external readers can audit the
project's release boundary without having to dig through the planning history.

## What this repository is

This is the **curated public mirror** of the Antigence Subnet codebase
(`https://github.com/biobitworks/antigence-subnet`). It is generated from a
private working repository by `scripts/build_public_mirror.py`, driven by
`.planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json`.

The mirror is a bounded artifact, not a full clone. Some directories that
exist in the private working tree are intentionally not published.

## What is published

- The runtime code for the subnet: `antigence_subnet/`, `neurons/`, `scripts/`,
  `tests/`, `configs/`, `data/`.
- Build, lint, and CI configuration: `pyproject.toml`, `Makefile`,
  `Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `.github/`,
  `osv-scanner.toml`.
- Citation and release metadata: `CITATION.cff`, `.zenodo.json`, `LICENSE`,
  `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `README.md`,
  `min_compute.yml`.
- Operator templates: `antigence_subnet.toml.example`, `.env.example`,
  `.env.phase94.example`.
- Public-safe planning artifacts: only the governance and dry-run records that
  the curation manifest explicitly allows under
  `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/`.

## What is intentionally not in the mirror

- Internal handoff state (`.planning/HANDOFF.json`) and per-run logs
  (`.planning/runs/`).
- Local agent state directories (`.antigence/`, `.claude/`).
- Internal pitch and application drafts (`docs/pitch/`).
- Private working notes, cache directories, and historical scratch files.
- Tests that depend on internal preflight, submission, or release-package
  fixtures (`tests/test_phase9{3,4,5,6,7}_*`).

## Release status

- License: MIT (see `LICENSE`).
- Current package version: see `CHANGELOG.md` and `pyproject.toml`.
- Citation: see `CITATION.cff` (a Zenodo DOI for this subnet release will be
  added once minted).
- Deployment evidence: simulation-only, as recorded in the v10 release notes
  under `docs/release/`. There is no funded-wallet live-network deployment
  bundled with this release.

## Reporting issues

- Security issues: see `SECURITY.md`.
- Functional issues, questions, and contributions: open a GitHub issue or
  pull request against
  `https://github.com/biobitworks/antigence-subnet`.
