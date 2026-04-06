# Orchestrator vs Flat Ensemble Benchmark

**Generated:** 2026-04-01T09:23:17.647587+00:00

## Performance Comparison

| Domain | Pipeline | F1 | Precision | Recall | Mean Normal | Mean Anomalous |
|--------|----------|----|-----------|--------|-------------|----------------|
| hallucination | Flat Ensemble | 0.9677 | 1.0 | 0.9375 | 0.2527 | 0.4696 |
| hallucination | Orchestrator | 0.9677 | 1.0 | 0.9375 | 0.2527 | 0.4844 |
| code_security | Flat Ensemble | 0.9841 | 1.0 | 0.9688 | 0.2637 | 0.4852 |
| code_security | Orchestrator | 0.9688 | 0.9688 | 0.9688 | 0.2848 | 0.4852 |
| reasoning | Flat Ensemble | 1.0 | 1.0 | 1.0 | 0.1613 | 0.5753 |
| reasoning | Orchestrator | 1.0 | 1.0 | 1.0 | 0.1613 | 0.591 |
| bio | Flat Ensemble | 1.0 | 1.0 | 1.0 | 0.1596 | 0.6029 |
| bio | Orchestrator | 1.0 | 1.0 | 1.0 | 0.1596 | 0.6152 |

## Score Distribution Divergence

| Domain | KL-Divergence | F1 Delta | Status |
|--------|---------------|----------|--------|
| hallucination | 0.016951 | 0.0 | PASS |
| code_security | 0.020273 | -0.0153 | PASS |
| reasoning | 0.002107 | 0.0 | PASS |
| bio | 0.301141 | 0.0 | FAIL |

## Orchestrator Metadata

| Domain | NK Trigger Rate | Immature | Semi-Mature | Mature |
|--------|-----------------|----------|-------------|--------|
| hallucination | 0.0167 | 37 | 0 | 1 |
| code_security | 0.0167 | 7 | 0 | 0 |
| reasoning | 0.0167 | 45 | 0 | 0 |
| bio | 0.0333 | 57 | 0 | 0 |

## Validation Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Hallucination F1 >= flat baseline | >= 0.9477 (flat=0.9677, tolerance=0.02) | 0.9677 | PASS |
| code_security F1 regression <= 0.02 | delta <= 0.02 (flat=0.9841) | delta=0.0153 | PASS |
| reasoning F1 regression <= 0.02 | delta <= 0.02 (flat=1.0) | delta=0.0 | PASS |
| bio F1 regression <= 0.02 | delta <= 0.02 (flat=1.0) | delta=0.0 | PASS |
| hallucination KL-div < 0.5 | < 0.5 | 0.016951 | PASS |
| code_security KL-div < 0.5 | < 0.5 | 0.020273 | PASS |
| reasoning KL-div < 0.5 | < 0.5 | 0.002107 | PASS |
| bio KL-div < 0.5 | < 0.5 | 0.301141 | PASS |
| hallucination mean normal in [0, 0.3] | [0.0, 0.3] | 0.2527 | PASS |
| code_security mean normal in [0, 0.3] | [0.0, 0.3] | 0.2848 | PASS |
| reasoning mean normal in [0, 0.3] | [0.0, 0.3] | 0.1613 | PASS |
| bio mean normal in [0, 0.3] | [0.0, 0.3] | 0.1596 | PASS |
| hallucination mean anomalous >= flat baseline | >= 0.4696 (flat baseline) | 0.4844 | PASS |
| code_security mean anomalous >= flat baseline | >= 0.4852 (flat baseline) | 0.4852 | PASS |
| reasoning mean anomalous >= flat baseline | >= 0.5753 (flat baseline) | 0.591 | PASS |
| bio mean anomalous >= flat baseline | >= 0.6029 (flat baseline) | 0.6152 | PASS |
