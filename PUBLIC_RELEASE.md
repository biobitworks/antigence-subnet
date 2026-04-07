# Public Release Checklist

This repository is close to public-ready, but a few checks should be completed
before changing GitHub visibility from private to public.

## Immediate Sharing Options

- For one colleague, prefer adding them as a GitHub collaborator to the private repo.
- For broader sharing, create a public-safe release branch or flip the repo to public after the checks below.

## Publication Checks

Canonical audit artifacts for the new public-safe release track:

- `.planning/phases/98-public-safety-audit-content-inventory/98-public-surface-inventory.json`
- `.planning/phases/98-public-safety-audit-content-inventory/98-public-surface-inventory.md`
- `.planning/phases/98-public-safety-audit-content-inventory/98-release-strategy.md`
- `.planning/phases/98-public-safety-audit-content-inventory/98-public-release-action-map.json`
- `.planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json`
- `.planning/phases/99-public-mirror-curation-history-sanitization/99-history-strategy.md`
- `.planning/phases/99-public-mirror-curation-history-sanitization/99-public-tree-verification.json`
- `.planning/phases/101-public-release-dry-run-launch-approval/101-dry-run-report.md`
- `.planning/phases/101-public-release-dry-run-launch-approval/101-release-checks.json`
- `.planning/phases/101-public-release-dry-run-launch-approval/101-launch-approval.json`

1. Confirm repo visibility strategy.
   - Option A: keep this repo private and create a separate public mirror.
   - Option B: make `biobitworks/antigence-bittensor` public directly.

2. Review historical pitch/application docs.
   - The curated public mirror now excludes `docs/pitch/` and `PUBLIC_RELEASE.md`.
   - If you ever flip this working repo public directly instead of publishing the mirror, review `docs/pitch/` separately first.

3. Confirm sample data policy.
   - Evaluation and test fixtures include intentionally synthetic credential-like strings for security-detection examples.
   - Keep them only if you are comfortable with public readers seeing them as test fixtures.
   - Final owner approval for those optics remains part of the Phase 101 launch gate.

4. Confirm external metadata.
   - README no longer contains a placeholder DOI badge.
   - Add a real Zenodo/DOI badge later if you mint one.

5. Final verification before publish.
   - Run `ruff check antigence_subnet`
   - Run the relevant test suite
   - Review `git diff` for any private notes or local-only changes
   - Launch remains blocked pending explicit human approval recorded in the Phase 101 approval artifact

## Current Status

- MIT license present
- README present
- `.env.example` is template-only
- No `.env`, `.pem`, `.key`, `.p12`, `.pfx`, or `.crt` files detected in the repo root scan
- Current GitHub repo visibility: private
