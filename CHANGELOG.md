# Changelog

All notable changes to this repository are documented in this file.

## [10.1.0] - 2026-04-28

### Added

- `osv-scanner.toml` with explicit, dated `IgnoredVulns` entries for the
  transitive `py==1.11.0` and `setuptools<78` advisories. Each entry carries
  a verified OSV ID, an `ignoreUntil` deadline, and a rationale.
- Bandit and OSV scanner reports now upload as workflow artifacts so the
  full picture is auditable even when the gating step passes.
- `ollama>=0.6.0,<1.0` added to the `[dev]` extras in `pyproject.toml` so
  the Ollama-harness contract tests no longer disappear on minimal CI.

### Changed

- Security workflow (`.github/workflows/security.yml`) now fails on any
  Bandit medium-or-higher finding — the wholesale `--skip B104,B310,B608,B614`
  exemption is replaced by per-line `# nosec` annotations across 13 sites,
  each with a rationale comment. OSV scanning no longer carries
  `continue-on-error: true`.
- `scripts/ollama_test_harness.py` defers `import ollama` to the functions
  that actually call it, so artifact-shape and variance helpers in
  Phase 81/83 tests collect on environments without the package.
- Cross-domain dataset balance check (`tests/test_data_expansion.py`) now
  enforces a ratio-based `>=45%` per-class threshold instead of the
  absolute floor that no longer scaled with the 220-sample dataset.

### Fixed

- Restored the collusion forward-pass integration test (`tests/test_collusion.py`).
  The detector's penalty path is wired into `validator/forward.py` Stage 6b;
  the prior class-level skip masked a core anti-cheating contract.
- Restored `test_forward_queries_miners_and_updates_scores`. The mock
  dendrite now produces ground-truth-aligned scores so honeypot zero-out
  and collusion zero-out cannot mask a real scoring regression.
- Scrubbed absolute developer paths (`/Users/byron/...`) from
  `data/audit/*.json` and `data/benchmarks/phase{81,83}-*.json` so public
  readers do not see the original author's machine layout.

### Notes

- Three integration tests for `cold_start`, `validator_agreement`, and
  `dendritic_cell` weight-manager adaptation remain marked
  `@pytest.mark.skip` with documented "not yet wired into Validator"
  reasons. The unit tests for these modules pass; the validator-level
  wiring is deferred to a future milestone.
- The performance test
  `test_performance_256_miners_under_10ms` is hardware-flake-prone on
  consumer laptops and uses a median-of-5 guard. CI runners typically
  meet the threshold.

## [10.0.0] - 2026-04-05

### Added

- Statistical and semantic validator scoring modes for LLM non-determinism hardening.
- Continuous benchmark canon and operator-facing `operator_multiband` policy artifacts.
- Public-safe release track artifacts covering mirror curation, dry-run validation, and explicit launch approval.
- Release notes and a Zenodo handoff checklist for the v10 release package.

### Changed

- Finalized the v10 stack around the weighted ensemble + operator_multiband runtime contract.
- Updated package, citation, and Zenodo-prep metadata to the v10.0.0 release line.
- Shifted release/publication framing to the simulation-only deployment evidence accepted in Phases 94-95.

### Fixed

- Repaired deployment wording drift so downstream release and publication work cites the simulation-only contract honestly.

### Notes

- Phase 84 swarm NO-GO remained in force; the shipped outcome is non-determinism hardening plus weighted-ensemble optimization rather than swarm deployment.
- The release package records simulation-only deployment evidence and public-safe launch approval, not a real funded-wallet live deployment.
- Zenodo publication for this subnet release is still pending human action; do not record a fake DOI.
