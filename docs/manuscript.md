# Antigence Subnet v10.0: Non-Determinism Hardening, Weighted-Ensemble Policy, and a Simulation-Bounded Release Path for Decentralized AI Verification

**Authors:** BiobitWorks

**Version:** 10.0.0 | **Date:** 2026-04-05

**Submission target:** Computers & Security (`NO-GO` branch)

---

## Abstract

Decentralized AI networks need a verification layer that can detect anomalous
outputs without assuming one trusted operator or one deterministic model stack.
Antigence Subnet frames that problem as immune-inspired anomaly detection on
top of Bittensor: miners act as antibodies, validators apply selection
pressure, and the network emits an operator-consumable trust signal. The v10.0
milestone asked two concrete questions. First, does large-language-model
non-determinism materially destabilize validator scoring enough to justify new
scoring controls? Second, does an intra-miner swarm improve anomaly detection
enough to justify a new framework? The measured answers were asymmetric. Across
the four evaluation domains already shipped by the subnet, the exact,
statistical, and semantic scoring benchmark in Phase 83 observed `0.00%`
variance on the local evidence stack, so the expected instability story did not
materialize in the benchmark artifact. The scoring work still paid off because
it separated score generation from policy interpretation and made the operator
surface explicit. The swarm hypothesis failed more decisively. A thin
three-detector spike underperformed the existing baseline on every domain,
increased latency by `4.95x` to `8.39x`, and remained highly sensitive to one
extreme detector vote, leading to a recorded `NO-GO` in Phase 84. The shipped
outcome therefore became a weighted-ensemble plus `operator_multiband` policy
stack, not a swarm framework. Phase 92 preserved the post-`NO-GO` benchmark
canon as continuous score surfaces and policy overlays, while Phase 93 selected
`operator_multiband` as the default operator policy with thresholds
`high_threshold = 0.5`, `low_threshold = 0.493536`, and `min_confidence = 0.6`.
Under the saved score surface, every approved overlay recorded
`precision = recall = f1 = balanced_accuracy = 1.000`, and
`operator_multiband` also recorded `review_rate = 0.000` and
`auto_decision_coverage = 1.000`. Phases 94 and 95 then bounded deployment
claims to simulation / localhost-backed evidence and recorded an explicit
public-safe deferral of real funded-wallet execution. The resulting paper is
therefore a security and verification systems result, not a swarm-intelligence
result: evidence favored hardening, calibration, and operator policy clarity
over adding another consensus layer.

## 1. Introduction

AI systems increasingly emit outputs that matter operationally: code patches,
security guidance, reasoning traces, scientific analyses, and user-facing text.
In decentralized AI networks, those outputs are produced by many independent
actors under incentive pressure. The verification problem is therefore not just
"is this output correct?" but also "who verifies it, under what incentives, and
how does the verifier avoid becoming a new central point of failure?"

Antigence Subnet addresses that problem inside the Bittensor ecosystem. Miners
run anomaly detectors over candidate AI outputs. Validators score those miner
responses against hidden evaluation datasets and allocate reward according to
precision-first accuracy. The network therefore acts as a verification market:
participants who better distinguish benign from anomalous outputs accumulate
weight and reward.

Earlier milestones established the base detector stack, the immune-inspired
negative-selection detector, and the initial public/documentation surfaces. The
v10.0 milestone added a stricter scientific question: should the project move
toward swarm-style intra-miner consensus, or should it invest in a narrower
hardening path around scoring stability, ensemble calibration, and operator
decision policy? That question matters for security work because complexity can
either improve resilience or simply create new attack surfaces.

This manuscript records the final v10.0 answer. The swarm hypothesis was tested
and rejected under the declared gate. The shipped outcome is a
non-determinism-hardening and weighted-ensemble path with explicit operator
policy semantics and an honest simulation-only deployment boundary.

## 2. Research Questions And Milestone Contract

v10.0 was governed by two linked questions.

1. **Non-determinism question.** Do repeated validator-scoring passes over the
   same evidence surface show enough instability to justify new scoring modes
   and operator controls?
2. **Swarm question.** Does a minimal swarm-style aggregation layer outperform
   the existing weighted-ensemble baseline enough to justify framework buildout?

The milestone also carried a hard gate: Phase 84 would authorize later swarm
phases only if a thin three-detector spike achieved more than `0.03` absolute
F1 gain on at least two domains, stayed under `<2x` latency, showed a
plausible mechanism beyond "add another detector," and revealed no unmitigable
adversarial issue.

That governance matters for interpretation. A `NO-GO` here is not a project
failure. It is a falsified branch in a milestone that explicitly allowed
rejection of the more complex path.

## 3. System Overview

Antigence Subnet keeps the basic architecture established in earlier releases.

- **Validators** own hidden evaluation datasets and score miners against known
  labels.
- **Miners** run detector implementations over prompts, outputs, code, and
  domain metadata.
- **The response contract** carries `score` and `confidence` values that can be
  consumed by policy layers without changing detector internals.
- **The trust signal** can be exposed to external consumers as a verification
  primitive.

The immune-system analogy remains useful at the systems level. Miners are
antibodies, validators provide selection pressure, and false positives are
treated as autoimmune errors. What changed in v10.0 is not the metaphor but the
interpretation layer: the milestone separated detector evidence from operator
policy decisions more cleanly than earlier releases.

## 4. Methods

### 4.1 Phase 81-83: Non-Determinism Measurement And Scoring Controls

Phase 81 measured repeated-run behavior over the committed v9.2 baseline path
and separated observed variance from inferred variance sources. Phase 82 then
implemented exact, statistical, and semantic scoring modes. Phase 83 benchmarked
those modes on the same four domains used throughout the subnet:

- hallucination detection
- code security analysis
- agent reasoning audit
- bio pipeline verification

The important design constraint was backwards compatibility. The validator could
request additional scoring structure, but the system could not assume miners
would become deterministic or expose new protocol guarantees.

### 4.2 Phase 84: Thin Swarm Falsification Spike

The swarm experiment was intentionally narrow. Rather than introducing persona
simulation, peer messaging, or a new multi-agent framework, Phase 84 tested a
minimal three-detector aggregate (`IsolationForest + OCSVM + NegSel`) against
the already measured baseline. This made the causal claim tractable:
improvement would have to come from the extra detector and the aggregation
scheme, not from a broad framework rewrite.

Two simple aggregators were measured:

- `mean3`
- `median3`

An adversarial probe also replaced one detector contribution with an extreme
score to test whether one outlier vote could distort the aggregate.

### 4.3 Phase 92-93: Continuous Score Canon And Operator Policy

After the `NO-GO`, the project preserved the post-swarm path by fixing the
benchmark canon to weighted two-detector surfaces and explicit policy overlays.
Phase 92 kept score-quality metrics separate from policy metrics. Phase 93 then
selected `operator_multiband` as the operator-facing default while preserving
`global_threshold` and `domain_thresholds` as supported alternatives.

The selected default was:

```toml
[validator.policy]
mode = "operator_multiband"
high_threshold = 0.5
low_threshold = 0.493536
min_confidence = 0.6
```

This matters because it formalizes a distinction that security systems often
blur:

- the detector layer produces evidence (`score`, `confidence`)
- the operator layer decides whether to allow, review, or block

### 4.4 Phase 94-95: Deployment Evidence Boundary

Phases 94 and 95 deliberately stopped short of real funded-wallet deployment.
Instead, they verified the runtime and artifact contract through simulation /
localhost-backed resources, fail-closed governance templates, and a
deployment-deferral report. This was an explicit public-safe choice, not an
accidental omission.

## 5. Results

### 5.1 Scoring Variance Was Not Observed On The Local Benchmark Surface

Phase 83 recorded `0.00%` variance for exact, statistical, and semantic scoring
across all four measured domains.

| Domain | Exact variance_pct | Statistical variance_pct | Semantic variance_pct |
| --- | ---: | ---: | ---: |
| hallucination | 0.00% | 0.00% | 0.00% |
| code_security | 0.00% | 0.00% | 0.00% |
| reasoning | 0.00% | 0.00% | 0.00% |
| bio | 0.00% | 0.00% | 0.00% |

Interpretation must stay narrow. These observations do **not** prove that all
future miner or validator stacks are deterministic. They do show that the
expected instability narrative was not supported on the measured local surface.
The exact-mode roadmap expectation (`>15%`) failed; the semantic-mode target
(`<5%`) passed.

### 5.2 The Swarm Hypothesis Failed Under The Declared Gate

Phase 84 compared the thin spike to the existing baseline and to the current
two-detector control. Every measured domain regressed.

| Domain | Baseline F1 | `mean3` F1 | `median3` F1 | Best latency multiplier |
| --- | ---: | ---: | ---: | ---: |
| hallucination | 0.9533 | 0.6854 | 0.5700 | 4.9477x |
| code_security | 0.9725 | 0.1854 | 0.1928 | 8.3941x |
| reasoning | 0.9333 | 0.3947 | 0.3448 | 5.2987x |
| bio | 1.0000 | 0.5333 | 0.4932 | 5.2279x |

The adversarial sensitivity result was also severe:

- `mean3` max flip rate: `0.8136` with `179` prediction flips
- `median3` max flip rate: `0.7455` with `164` prediction flips

Under the declared gate, this is an unambiguous `NO-GO`. The spike improved on
zero domains, exceeded the latency budget everywhere, showed no credible
mechanism beyond adding another detector, and remained vulnerable to a single
extreme detector contribution.

### 5.3 The Post-NO-GO Control Surface Performed Cleanly

Phase 92 fixed the post-`NO-GO` benchmark canon around weighted variants of the
two-detector control and explicit policy overlays. All approved surfaces
reported perfect discrimination metrics on the saved score artifact, while
calibration differences remained small and measurable.

| Surface | AP | ROC AUC | Brier | ECE | Avg latency (ms) | Peak memory (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_equal` | 1.000 | 1.000 | 0.068 | 0.237 | 5.0 | 33.0 |
| `weighted_ocsvm_0.65_negsel_0.35` | 1.000 | 1.000 | 0.062 | 0.234 | 5.0 | 33.0 |
| `weighted_ocsvm_0.80_negsel_0.20` | 1.000 | 1.000 | 0.056 | 0.230 | 5.0 | 33.0 |
| `confidence_modulated_static` | 1.000 | 1.000 | 0.062 | 0.233 | 5.0 | 33.0 |

Policy overlays then preserved the same measured outcome on the saved surface.

| Policy | Precision | Recall | F1 | Balanced Accuracy | Policy Reward | Review Rate | Auto Decision Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_threshold` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a | n/a |
| `domain_thresholds` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a | n/a |
| `operator_multiband` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

Because the overlays tied on the locked artifact, Phase 93 chose
`operator_multiband` on usability grounds: it preserved measured parity while
adding explicit `allow`, `review`, and `block` semantics.

### 5.4 Deployment Evidence Stayed Simulation-Bounded By Design

Phase 94 verified the winning runtime stack through simulation and
localhost-backed resources, not a real funded-wallet chain run. Phase 95 then
recorded that decision as an explicit deployment-deferral contract.

Observed evidence includes:

- runtime and metrics checks under simulation / localhost-backed resources
- template-only governance artifacts that fail closed before any live action
- a deployment-deferral report separating observed, inferred, and intentionally
  unknown claims

Intentionally unknown evidence includes:

- funded-wallet registration proof on a real subnet
- live `set_weights` proof
- 24h or 72h real-network soak artifacts

That boundary is important for a security paper. It avoids overstating
production-readiness and treats missing live evidence as a first-class fact
rather than a footnote.

## 6. Security Interpretation

The strongest v10.0 result is not a new detector. It is the negative result on
complexity. The milestone asked whether adding a swarm-style consensus layer
would improve security-relevant anomaly detection. Measured evidence said no.

That negative result has practical value:

- It avoids new attack surfaces around vote manipulation, correlated agents, and
  latency inflation.
- It preserves the simpler evidence path where detector outputs and operator
  policy are separable and auditable.
- It keeps the security story focused on verification quality, false-positive
  control, and deployment honesty rather than on an unsupported multi-agent
  narrative.

The `operator_multiband` default also matters operationally. Many security
systems need a review lane, not just binary allow/block output. The policy layer
therefore translates continuous detector evidence into operator action bands
without changing the detector contract itself.

## 7. Threats To Validity

Several limitations remain.

1. **Observed zero variance does not prove universal determinism.** It proves
   only that the local Phase 83 benchmark surface did not expose instability.
2. **The evaluation domains are fixed.** The paper does not claim that the same
   exact metrics transfer to arbitrary future domains.
3. **Deployment claims are simulation-bounded.** Real network behavior, wallet
   lifecycle, and long-run chain interaction remain deferred and intentionally
   unknown.
4. **The `NO-GO` is scoped to the tested swarm formulation.** It rejects the
   measured v10 swarm path; it does not claim all imaginable multi-agent
   architectures are impossible.

## 8. Reproducibility And Artifact Map

The final manuscript is grounded in repo-local artifacts rather than summary
slides or unrecoverable notebooks.

- Phase 83 variance report:
  `.planning/phases/83-determinism-controls-scoring-benchmark/83-benchmark-report.md`
- Phase 84 `NO-GO` decision:
  `.planning/phases/84-swarm-literature-review-adversarial-pre-analysis-fast-fail-spike/84-go-no-go-adr.md`
- Phase 92 benchmark canon:
  `.planning/phases/92-continuous-antibody-benchmark-canon/92-benchmark-report.md`
- Phase 93 ADR:
  `.planning/phases/93-decision-policy-adr-operator-migration/93-ADR.md`
- Phase 94 verification:
  `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/94-VERIFICATION.md`
- Phase 95 deployment-deferral report:
  `.planning/phases/95-simulation-performance-deployment-deferral-report/95-report.md`

The active submission package for the `NO-GO` branch is prepared for
Computers & Security, and the portal upload remains pending human action.

## 9. Conclusion

v10.0 answered the right question by permitting a negative result. The swarm
path was tested and rejected under measured criteria. The evidence-supported
ship target became a weighted-ensemble plus `operator_multiband` policy stack
with explicit simulation-bounded deployment claims. For decentralized AI
verification, that is a useful outcome: the project avoided adding a new
consensus layer that degraded accuracy, increased latency, and weakened
robustness, and instead shipped a cleaner security posture grounded in explicit
evidence and honest operational boundaries.

## References

1. Umair, M., Rashid, N., Khan, U.S., Hamza, A., Zeb, A., Nawaz, T.H., &
   Ansari, A.R. (2025). Negative selection-based artificial immune system
   (NegSl-AIS): A hybrid multimodal emotional effect classification model.
   *Results in Engineering*, 27, 106601.
   DOI: [10.1016/j.rineng.2025.106601](https://doi.org/10.1016/j.rineng.2025.106601)

2. Ji, Z., & Dasgupta, D. (2007). Revisiting negative selection algorithms.
   *Evolutionary Computation*, 15(2), 223-251.
   DOI: [10.1162/evco.2007.15.2.223](https://doi.org/10.1162/evco.2007.15.2.223)

3. Blum, C., Merkle, D., Kappel, C., Oswald, P., & Timmis, J. (2015). An
   artificial bioindicator system for network intrusion detection.
   *Artificial Life*, 21(2), 93-118.
   DOI: [10.1162/ARTL_a_00162](https://doi.org/10.1162/ARTL_a_00162)

4. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey.
   *ACM Computing Surveys*, 41(3), 1-58.
   DOI: [10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
