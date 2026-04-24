# ADR 0001: `.npz` state persistence coexists with the SHA-linked audit chain

**Status:** Accepted; promoted to production in v13.1.1 (Phase 1103)
**Date:** 2026-04-24
**Milestone:** v13.1 Production Integration (Migration)
**Phase:** 1100 — audit-chain bridge (experiment)
**Deciders:** executor-agent on behalf of maintainers
**Supersedes:** none

## Context

v13.0 shipped an additive `deterministic_scoring` package with an
immutable, SHA-256 linked audit chain (`AuditChainWriter`,
`FrozenRoundRecord`, `verify_chain`) that is fully replay-verifiable and
tamper-evident. It does not, however, replace the existing
`BaseValidatorNeuron.save_state()` path, which writes per-validator
`.npz` snapshots (scores, hotkeys, step) via `numpy.savez` for fast
restart on crash or upgrade.

Phase 1100 adds an experimental bridge
(`experiments/v13.1-migration/audit_bridge.py`) that converts the
production `get_rewards()` numpy output into `FrozenRoundScore` +
`FrozenRoundRecord` and appends it to `chain.jsonl`. That immediately
raises a persistence-policy question: **what is the authoritative
restart state once the audit chain flows — the `.npz` snapshot, the
chain, or both?**

Three options were considered:

| # | Option | Restart cost | Replay audit | Operator churn |
|---|--------|--------------|--------------|----------------|
| A | `.npz`-only (status quo, no chain write) | O(1) read | None | None |
| B | Chain-only (remove `.npz`) | O(N rounds) walk | Native | Large — every operator must redeploy |
| C | Coexist (`.npz` fast restart + chain for replay/audit) | O(1) read | Native | None |

## Decision

**Option C — Coexist.** For the duration of v13.1 (and expected onward):

1. `.npz` remains the **authoritative fast-restart snapshot**. The
   existing `BaseValidatorNeuron.save_state()` / `load_state()` path is
   not modified in v13.1. Operators upgrading from v13.0 do not need to
   migrate anything.
2. The audit chain is the **authoritative replay / tamper-detection
   artifact**. It is written in parallel per evaluation round via
   `RewardToAuditAdapter.record_round` and is the sole input to the
   convergence and syndrome detectors that Phases 1101–1102 will wire.
3. Chain continuity is verified at validator startup via
   `resume_chain_prev_hash(path)`, which calls `verify_chain(path)` and
   raises `ChainIntegrityError` on any hash or structural break.
4. A future **`validator.audit.enabled` toggle** (default `true`)
   landing in Phase 1102 lets operators opt out of chain writes. When
   disabled, `.npz` behavior is exactly the v13.0 status quo.

The chain is per-round (one `FrozenRoundRecord` bundling every
`FrozenRoundScore` for the round), not per-miner. This matches the
existing `FrozenRoundRecord.scores: Tuple[FrozenRoundScore, ...]`
shape and gives naturally contiguous `round_index` linkage.

## Consequences

**Positive**

- Zero operator migration required to adopt v13.1 (Option C preserves
  `.npz` semantics).
- Tamper detection and replay are available for every completed round
  without slowing down restart.
- The audit chain supplies the deterministic input surface needed by
  Phase 1001 convergence detectors and Phase 1002 syndrome classifier
  without re-running production scoring.

**Negative**

- Small amount of duplicate persistence. Disk cost is approximately
  `O(rounds × canonical_json bytes per record)`; at typical subnet
  cadence (one round every few minutes, few-KB canonical JSON per
  record), this is well under 1 GB/year per validator.
- Two writes per round instead of one. Benchmarks in Phase 1101 will
  confirm no material latency regression; preliminary numbers on the
  adapter path are sub-millisecond per round.

**Neutral**

- Chain-only replay is possible if an operator chooses; `.npz` is not
  required for correctness, only for restart speed.

## Alternatives considered

- **Option A (`.npz`-only):** Keeps status quo but forfeits the audit
  chain entirely — defeats the whole purpose of v13.0. Rejected.
- **Option B (chain-only):** Forces every restart to walk the entire
  chain, scaling poorly as rounds accumulate, and creates a migration
  burden on every operator. Rejected.
- **Per-miner chain records (one JSONL record per miner per round):**
  Considered and rejected. `FrozenRoundRecord` already bundles all
  per-round scores in a single immutable record; splitting to one
  record per miner would multiply chain length N× (where N = miner
  count), bloat `prev_hash` walks, and break the natural
  round-contiguity assumption that `verify_chain` relies on. A
  per-round record keeps the chain `O(rounds)` while still preserving
  every miner's individual score.

## References

- `experiments/v13.1-migration/README.md` — isolation contract
- `antigence_subnet/validator/deterministic_scoring/chain.py` — chain writer
- `antigence_subnet/validator/deterministic_scoring/state.py` — FrozenRoundRecord shape
- `.planning/REQUIREMENTS.md` — MIGRATE-01..04 (v13.1 migration)
- `.planning/ROADMAP.md` — Phase 1100 success criteria

## 2026-04-23 — Phase 1102 implementation note

STATEPOL-01 / STATEPOL-02 / STATEPOL-03 are now implemented on the
experiment copies in `experiments/v13.1-migration/`:

- `audit_config.py` — frozen `AuditConfig` + `ConvergenceConfig`
  dataclasses with `enabled: bool = False` default. Opt-in by design
  (STATEPOL-02).
- `audit_state.py` — parallel `save_audit_state` / `load_audit_state`
  hooks that run AFTER `.npz` save/load. They never read or write the
  `.npz` file (STATEPOL-01) and are pure no-ops when
  `config.audit.enabled` is False.
- `production_copy/base_validator.py` — modified COPY calls the audit
  hooks after the existing `.npz` path. `.npz` path byte-identical to
  v13.0.
- `production_copy/forward.py` — Stage 7.5 audit write + Stage 8.5
  convergence hook both gated on `audit.enabled AND audit_chain_path`.
- STATEPOL-03 ("clean start, no history replay") confirmed by
  `test_mid_session_audit_enable` — flipping `enabled=True` mid-session
  starts the chain from `GENESIS_PREV_HASH` with exactly 2 records
  after 2 enabled rounds (no replay of the 3 prior disabled rounds).

The default flipped from "`enabled=true`" (as speculated in the 2026-04-24
decision §4 above) to "`enabled=false`". Rationale: STATEPOL-02 explicitly
specifies off-by-default so operators who do not want parallel logging
get the v13.0 status quo on upgrade, which is the stronger backward-
compat contract.

Promotion of these experiment copies into production code remains
deferred to Phase 1103.
