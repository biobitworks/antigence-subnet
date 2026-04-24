# ADR 0002: `[validator.audit]` + `[validator.convergence]` config surface

**Status:** Accepted; promoted to production in v13.1.1 (Phase 1103)
**Date:** 2026-04-23
**Milestone:** v13.1 Production Integration (Migration)
**Phase:** 1102 — state-persistence policy (experiment)
**Deciders:** executor-agent on behalf of maintainers
**Supersedes:** none (extends ADR 0001)

## Context

Phase 1101 wired the convergence detectors (`detect_oscillation`,
`detect_metastability`, `detect_convergence_failure`) into the COPY of
the validator forward loop through `run_convergence_checks`, with the
six detector thresholds exposed only as a module-level
`DEFAULT_CONVERGENCE_CONFIG` dict and a `config=` kwarg. WIRE-03
required operator-facing configurability through both TOML and CLI
flags; that surface was deferred to this phase (which also touches
`production_copy/validate_config.py`).

Phase 1102 also needs a new top-level toggle: `validator.audit.enabled`
(STATEPOL-02), which controls whether the audit chain is written at all
and whether the convergence hook fires.

## Decision

Introduce two TOML sections and matching CLI flag families with uniform
precedence:

### TOML

```toml
[validator.audit]
enabled = false          # default: OFF (STATEPOL-02 backward compat)
chain_path = ""          # default: "" -> <neuron.full_path>/chain.jsonl

[validator.convergence]
window_size = 20
sign_change_threshold = 4
variance_bound = 1e-4
top_quantile_cut = 0.5
min_consecutive_rounds = 10
epsilon = 0.05
```

### CLI

```
--audit.enabled          --validator.audit.enabled
--audit.chain_path       --validator.audit.chain_path

--convergence.window_size             --validator.convergence.window_size
--convergence.sign_change_threshold   --validator.convergence.sign_change_threshold
--convergence.variance_bound          --validator.convergence.variance_bound
--convergence.top_quantile_cut        --validator.convergence.top_quantile_cut
--convergence.min_consecutive_rounds  --validator.convergence.min_consecutive_rounds
--convergence.epsilon                 --validator.convergence.epsilon
```

Both short and long forms match the pre-existing `--scoring.*` /
`--validator.scoring.*` duality.

### Precedence

```
CLI flags  >  TOML section  >  module defaults
```

Implemented as a two-step composition:

1. `audit_config.audit_config_from_toml(toml_dict) -> AuditConfig` —
   builds a frozen `AuditConfig` from the TOML layer (absent sections
   fall back to defaults; backward compat with v13.0 configs).
2. `audit_config.apply_audit_cli_overrides(cfg, cli_kv) -> AuditConfig`
   — takes a flat mapping `{"audit.enabled": True, "convergence.epsilon":
   0.1}` and returns a new frozen instance with overrides applied.

Both `AuditConfig` and `ConvergenceConfig` are `@dataclass(frozen=True)`
— overrides return a new instance rather than mutating, preventing
silent in-place configuration drift.

## Consequences

**Positive**

- WIRE-03 is fully closed: operators tune thresholds through the same
  mechanisms they use for scoring/policy.
- STATEPOL-02 is a single-flag toggle (`--audit.enabled=false`) —
  operators upgrading from v13.0 keep the status quo.
- Frozen dataclasses make the config immutable at runtime; no rogue
  mid-session mutation.
- `validate_config.py` emits "Unknown key" warnings for typos under
  `[validator.audit]` / `[validator.convergence]`, catching operator
  errors at config-load time rather than at first invocation.

**Negative**

- Two more TOML sections and six more CLI flags to document. Operators
  needing non-default convergence thresholds now have first-class
  support, so this is a net win.

## Alternatives considered

- **Single aggregated section `[validator.qec]`** with nested tables:
  rejected because it hides the boundary between "write the chain"
  (audit) and "read the chain for detection" (convergence). Two sections
  make the toggle-vs-tuning distinction explicit.
- **Command-line-only (no TOML):** rejected — most operator workflows
  use TOML as the source of truth; CLI-only would force everyone to
  maintain shell wrappers.
- **Global default `enabled=true`:** rejected per ADR 0001's 2026-04-23
  update — the stronger backward-compat contract is off-by-default.

## References

- `experiments/v13.1-migration/audit_config.py` — dataclasses + loader
- `experiments/v13.1-migration/production_copy/validate_config.py` —
  TOML schema + range validation
- `experiments/v13.1-migration/production_copy/base_validator.py` —
  `argparse` CLI flag registration
- ADR 0001 — `.npz` ↔ audit-chain coexistence
- `.planning/REQUIREMENTS.md` — WIRE-03, STATEPOL-01..03
- `tests/experiments/v13_1_migration/test_audit_config.py::test_config_precedence_cli_gt_toml`
