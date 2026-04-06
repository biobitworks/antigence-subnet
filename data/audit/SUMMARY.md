# Feature Audit Summary

Generated: 2026-04-01 07:24:49 UTC

This report characterizes all 10 dendritic features across all evaluation domains. It informs Phase 32 NK Cell z-score thresholds and Phase 33 DCA signal mapping.

## Per-Domain Feature Statistics

### hallucination (n=60)

| Feature | Type | Mean | Std | Min | Max |
|---------|------|------|-----|-----|-----|
| claim_density | continuous | 0.5067 | 0.0249 | 0.5000 | 0.6000 |
| citation_count | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hedging_ratio | binary | 0.0667 | 0.2494 | 0.0000 | 1.0000 |
| specificity | continuous | 0.7406 | 0.2474 | 0.0000 | 1.0000 |
| numeric_density | binary | 0.2333 | 0.4230 | 0.0000 | 1.0000 |
| pamp_score | continuous | 0.0056 | 0.0427 | 0.0000 | 0.3333 |
| exaggeration | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| certainty | binary | 0.0333 | 0.1795 | 0.0000 | 1.0000 |
| controversy | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| danger_signal | continuous | 0.0033 | 0.0256 | 0.0000 | 0.2000 |

### code_security (n=60)

| Feature | Type | Mean | Std | Min | Max |
|---------|------|------|-----|-----|-----|
| claim_density | continuous | 0.5033 | 0.0256 | 0.5000 | 0.7000 |
| citation_count | binary | 0.0167 | 0.1280 | 0.0000 | 1.0000 |
| hedging_ratio | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| specificity | continuous | 0.2980 | 0.3370 | 0.0000 | 1.0000 |
| numeric_density | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pamp_score | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exaggeration | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| certainty | binary | 0.0167 | 0.1280 | 0.0000 | 1.0000 |
| controversy | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| danger_signal | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### reasoning (n=60)

| Feature | Type | Mean | Std | Min | Max |
|---------|------|------|-----|-----|-----|
| claim_density | continuous | 0.5117 | 0.0321 | 0.5000 | 0.6000 |
| citation_count | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hedging_ratio | binary | 0.1167 | 0.3210 | 0.0000 | 1.0000 |
| specificity | continuous | 0.8663 | 0.1730 | 0.0025 | 1.0000 |
| numeric_density | binary | 0.2167 | 0.4120 | 0.0000 | 1.0000 |
| pamp_score | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exaggeration | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| certainty | binary | 0.1000 | 0.3000 | 0.0000 | 1.0000 |
| controversy | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| danger_signal | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### bio (n=60)

| Feature | Type | Mean | Std | Min | Max |
|---------|------|------|-----|-----|-----|
| claim_density | continuous | 0.5017 | 0.0128 | 0.5000 | 0.6000 |
| citation_count | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hedging_ratio | binary | 0.0167 | 0.1280 | 0.0000 | 1.0000 |
| specificity | continuous | 0.9061 | 0.1696 | 0.2083 | 1.0000 |
| numeric_density | binary | 0.4667 | 0.4989 | 0.0000 | 1.0000 |
| pamp_score | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exaggeration | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| certainty | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| controversy | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| danger_signal | constant | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Feature Classification (Binary vs Continuous)

| Feature | hallucination | code_security | reasoning | bio |
|---------|------|------|------|------|
| claim_density | continuous | continuous | continuous | continuous |
| citation_count | constant | binary | constant | constant |
| hedging_ratio | binary | constant | binary | binary |
| specificity | continuous | continuous | continuous | continuous |
| numeric_density | binary | constant | binary | binary |
| pamp_score | continuous | constant | constant | constant |
| exaggeration | constant | constant | constant | constant |
| certainty | binary | binary | binary | constant |
| controversy | constant | constant | constant | constant |
| danger_signal | continuous | constant | constant | constant |

## High Correlations (|r| > 0.7)

### hallucination

- **claim_density / hedging_ratio**: r = 1.0000
  - claim_density includes +0.1 for hedging_ratio presence. r=1.0000. Partial algebraic dependency.
- **pamp_score / certainty**: r = 0.7011
  - Moderate positive correlation (r=0.7011).
- **pamp_score / danger_signal**: r = 1.0000
  - Algebraic dependency: both derived from same danger pattern matches (pamp=matches/3, danger=matches/5). r=1.0000. DCA must assign to same signal category to avoid double-counting.
- **certainty / danger_signal**: r = 0.7011
  - Moderate positive correlation (r=0.7011).

### code_security

- **claim_density / citation_count**: r = 1.0000
  - claim_density includes +0.2 when citation_count present. r=1.0000. Partial algebraic dependency.

### reasoning

- **claim_density / hedging_ratio**: r = 1.0000
  - claim_density includes +0.1 for hedging_ratio presence. r=1.0000. Partial algebraic dependency.

### bio

- **claim_density / hedging_ratio**: r = 1.0000
  - claim_density includes +0.1 for hedging_ratio presence. r=1.0000. Partial algebraic dependency.

## Key Warnings

### pamp_score / danger_signal Algebraic Relationship

Both `pamp_score` and `danger_signal` are derived from the same underlying danger pattern match count:

- `pamp_score = min(danger_matches / 3.0, 1.0)`
- `danger_signal = min(danger_matches / 5.0, 1.0)`

This creates a perfect or near-perfect linear correlation (r=1.0 when neither saturates). Phase 33 DCA **must** assign these to the same signal category to avoid double-counting.

### claim_density Dependency Chain

`claim_density` (source credibility) is computed from 4 other features:

```
credibility = 0.5
  + 0.2 * (citation_count > 0)
  + 0.1 * (hedging_ratio > 0)
  - 0.2 * (exaggeration > 0.5)
  - 0.3 * (pamp_score > 0.5)
```

This means claim_density is algebraically dependent on citation_count, hedging_ratio, exaggeration, and pamp_score. Phase 33 DCA should group claim_density with its input features or treat it as a derived signal.

### Constant Feature Warnings

- **hallucination**: 3 constant feature(s): citation_count, exaggeration, controversy
- **code_security**: 6 constant feature(s): hedging_ratio, numeric_density, pamp_score, exaggeration, controversy, danger_signal
- **reasoning**: 5 constant feature(s): citation_count, pamp_score, exaggeration, controversy, danger_signal
- **bio**: 6 constant feature(s): citation_count, pamp_score, exaggeration, certainty, controversy, danger_signal

## Downstream Consumption

### Phase 32 (NK Cell)

Use `feature_stats.{name}.mean` and `feature_stats.{name}.std` from per-domain JSON files to compute z-score thresholds. Skip features with `std == 0.0` (constant features produce undefined z-scores).

### Phase 33 (DCA)

Use `feature_stats.{name}.is_binary` for signal mapping. Use `high_correlations` to identify features that must be in the same signal category (especially pamp_score/danger_signal).
