# Research Note: QEC Foundations Applicable to Antigence Subnet

**Date:** 2026-04-07
**Sources:**
- [QSOLKCB/QEC](https://github.com/QSOLKCB/QEC) — Deterministic QEC architecture (Trent Slade, QSOL-IMC)
- [QSOLKCB/QSOLAI](https://github.com/QSOLKCB/QSOLAI) — Quantum-sourced optimization logic AI kernel
- [multimodalas/fusion-qec](https://github.com/multimodalas/fusion-qec) — Fusion QEC simulator (CC-BY-4.0)
- Author ORCID: 0009-0002-4515-9237

**Red-team status:** All three repos verified as legitimate research projects. Safe to reference. fusion-qec is CC-BY-4.0 (attribution required if adapting code).

## Abstract

Quantum Error Correction (QEC) and Artificial Immune Systems (AIS) share a deep structural isomorphism: both detect deviations from expected patterns and apply corrective responses. This note maps specific QEC concepts from Trent Slade's architecture onto Antigence's immune-inspired anomaly detection subnet, identifying three tiers of applicability.

## Tier 1: Directly Applicable

### 1.1 Syndrome Detection ↔ Anomaly Detection

QEC detects error syndromes by measuring parity checks on quantum states. Antigence detects anomalous patterns in AI outputs via dendritic cell feature extraction + negative selection.

**Isomorphism:**
- QEC codeword = expected AI output pattern (self)
- QEC syndrome = deviation from expected pattern (non-self signal)
- QEC decoder = Antigence detector ensemble (B-cell, NK-cell, dendritic cell)

**Concrete application:** Frame each miner's anomaly score vector as a "codeword." The validator's evaluation dataset defines the "codebook" (expected responses). Deviations are "syndromes" that map to specific anomaly types — this gives a formal error-theoretic foundation to the scoring system.

### 1.2 Deterministic Replay ↔ Validator Consensus

QEC's core invariant: "same input = same bytes." This is exactly what Antigence validators need — deterministic scoring so that independent validators produce identical weights for the same miner responses.

**Current gap:** Antigence validators use floating-point EMA scoring which can accumulate numerical differences across validators. QEC's canonical serialization (frozen dataclasses + SHA-256 chains) provides a pattern for making validator state byte-identical.

**Concrete application:**
- Adopt frozen dataclasses for score state (not mutable numpy arrays)
- Canonical JSON serialization for all validator state checkpoints
- SHA-256 chain for weight-setting audit trail (complements Bittensor's ExtrinsicResponse)

### 1.3 Belief Propagation Trajectories ↔ Score Convergence Monitoring

QEC uses BP trajectory analysis to detect decoder pathologies — oscillations, metastability, failure to converge. This maps directly to monitoring miner score trajectories.

**Concrete application:**
- Track per-miner score trajectories over evaluation rounds
- Detect oscillation (gaming — miner alternates between strategies)
- Detect metastability (miner scores plateau at mediocre level)
- Detect convergence failure (scores diverge across validators — consensus breakdown)

## Tier 2: Requires Adaptation

### 2.1 Quantized Symbolic Lattices ↔ Miner Behavior States

QEC discretizes continuous decoder states into symbolic lattice points for governed control. Applied to Antigence: discretize continuous miner behavior into a finite state alphabet.

**Potential states:** {honest, gaming, colluding, sybil, dormant, adapting}

**Benefit:** Symbolic states enable formal reasoning about miner populations — finite state machines, Markov chains, equilibrium analysis.

### 2.2 Attractor/Basin Detection ↔ Gaming Equilibria

QEC detects when decoder dynamics settle into attractors (stable states) or basins (convergence regions). In Antigence: detect when the miner population converges to a gaming equilibrium.

**Application:** If all miners converge to the same scoring strategy (a basin of attraction), that's a collusion signal even without explicit coordination.

### 2.3 Governed Control Stack ↔ Adaptive Immune Response

QEC's supervised control: phase-space steering + escalation paths + timeout control. Maps to Antigence's danger signal cascade: dendritic feature extraction → danger modulation → immune orchestrator response.

**Enhancement:** QEC's "proactive escalation" pattern — detect approaching a failure boundary before crossing it — could improve the danger modulator's sensitivity to emerging threats.

## Tier 3: Future Research Directions

### 3.1 LDPC Codes for Byzantine-Tolerant Consensus

LDPC/QLDPC error-correcting codes tolerate a known fraction of errors. In theory, this maps to tolerating N byzantine validators in the consensus. Major theoretical lift — worth tracking as a research question, not an engineering task.

### 3.2 Spectral Anomaly Detection (from QSOLAI)

QSOLAI's spectral algebraics — frequency-domain decomposition of signals — could enable a novel anomaly detection approach: decompose AI output feature vectors into spectral components and detect anomalies as unexpected frequency patterns.

### 3.3 Topology-Aware Detector Selection

QEC uses graph kernel topology to select the right decoder family for a given error structure. Antigence could use a similar approach: characterize the "topology" of an anomaly domain (e.g., hallucination vs. code security) and select the optimal detector ensemble for that topology.

## References

1. Slade, T. (2026). QEC: Deterministic Quantum Error Correction Architecture. GitHub: QSOLKCB/QEC. v137.4.3.
2. Slade, T. (2026). QSOLAI: Quantum-Sourced Optimization Logic AI Kernel. GitHub: QSOLKCB/QSOLAI.
3. Slade, T. (2026). fusion-qec: Deterministic Governed QEC Simulator. GitHub: multimodalas/fusion-qec. CC-BY-4.0.
