# Publication Strategy

**Manuscript:** Antigence Subnet v10.0: Non-Determinism Hardening, Weighted-Ensemble Policy, and a Simulation-Bounded Release Path for Decentralized AI Verification
**Authors:** BiobitWorks
**Version:** 2.0 | **Date:** 2026-04-05

---

## Active Submission Path

**Current active target:** **Computers & Security**

Reason: the v10 milestone closed the swarm branch with a recorded `NO-GO`, so
the strongest contribution is the security-facing verification story:
non-determinism hardening, explicit operator policy, honest simulation-only
deployment boundaries, and a negative result on complexity inflation.

This is the active NO-GO branch publication path. Other venues remain useful
fallbacks, but the prepared package should now be treated as a
Computers & Security submission first.

---

## Preprint Status

| Platform | Status | ID/DOI | Date |
|----------|--------|--------|------|
| bioRxiv  | Pending | -- | -- |
| arXiv    | Pending | -- | -- |

---

## Journal Targets

### Top-tier (Reach, Deferred)

**IEEE Transactions on Dependable and Secure Computing (TDSC)**

- **Publisher:** IEEE
- **Impact Factor:** ~7.0
- **Scope alignment:** TDSC covers dependable and secure computing systems, including anomaly detection, AI safety, and distributed trust mechanisms. Our paper's anti-cheating mechanisms, trust score API, and adversarial resilience framing fit squarely within scope.
- **Open access:** Hybrid (optional OA with APC ~$2,500)
- **Review timeline:** Typically 3-6 months for first decision
- **Audience:** Security researchers, systems engineers, AI safety practitioners
- **Why this over Nature Machine Intelligence:** NMI is broader (requires groundbreaking ML contributions); TDSC is a better scope fit for a systems paper with security emphasis. The decentralized verification angle and anti-cheating mechanisms are core TDSC themes.

### Mid-tier (Realistic Primary, Active NO-GO Branch Target)

**Computers & Security (Elsevier)**

- **Publisher:** Elsevier
- **Impact Factor:** ~5.0
- **Scope alignment:** Directly covers AI security, anomaly detection, intrusion detection, and immune-inspired computing. The journal has published multiple AIS (Artificial Immune System) papers. Our combination of NegSel adaptation with decentralized incentive mechanisms is well within scope.
- **Open access:** Hybrid (optional OA with APC ~$3,000) or traditional subscription
- **Review timeline:** Typically 3-4 months for first decision
- **Audience:** Cybersecurity researchers, AI safety community, anomaly detection practitioners
- **Why primary mid-tier:** Highest probability of acceptance given direct scope overlap with immune-inspired security. Previous AIS publications in this venue create a receptive reviewer pool.

### Mid-tier (Realistic Fallback)

**Applied Soft Computing (Elsevier)**

- **Publisher:** Elsevier
- **Impact Factor:** ~8.0
- **Scope alignment:** Covers bio-inspired algorithms, soft computing methods, and their real-world applications. NegSel-AIS, ensemble optimization, and the immune system paradigm are core topics. Higher IF than Computers & Security but broader scope means more competition.
- **Open access:** Hybrid (optional OA) or traditional subscription
- **Review timeline:** Typically 3-5 months for first decision
- **Audience:** Bio-inspired computing researchers, applied ML practitioners, optimization community
- **Why fallback:** Broader scope means the paper competes against a wider range of soft computing contributions. Best reserved as fallback if the security-focused framing does not resonate with Computers & Security reviewers.

---

## Submission Timeline

**Total submission window: 6 months (April -- September 2026)**

| Month | Action | Target | Notes |
|-------|--------|--------|-------|
| 0 (April 2026) | Refresh manuscript and package for v10 | Computers & Security | Use the NO-GO security framing and simulation-bounded deployment evidence |
| 0-1 (April-May 2026) | Submit primary journal package | Computers & Security | Cover letter and supplementary package prepared in-repo |
| 1-4 (May-August 2026) | Await review or revise | Computers & Security | Incorporate reviewer feedback if requested |
| 4 (August 2026) | Decision gate | -- | If rejected or stalled, cascade to next venue |
| 4-5 (August-September 2026) | Reformat and submit | IEEE TDSC | Reframe toward broader dependable/distributed verification if useful |
| 6 (September 2026) | Fallback submission | Applied Soft Computing | Re-emphasize AIS and negative-result complexity findings |

**Cascade logic:** Each rejection triggers reformatting for the next target. Reviewer feedback from higher-tier journals should be incorporated before resubmission to lower tiers.

---

## Framing Guidance

### For AI / ML Venues (IEEE TDSC, arXiv cs.AI)

- **Lead with:** Decentralized verification as an unsolved problem in distributed AI networks
- **Emphasize:** Trust Score API as cross-subnet verification primitive, ensemble optimization (OCSVM+NegSel outperforming 7-detector ensemble), anti-cheating mechanisms (honeypot injection, perturbation stability)
- **Downplay:** Biological metaphor details; present immune inspiration as design motivation, not the contribution itself
- **Key result:** F1 0.968-1.000 across 4 domains with targeted 2-detector ensemble

### For Security Venues (Computers & Security)

- **Lead with:** AI output verification as a security challenge where complexity must earn its keep under evidence.
- **Emphasize:** The measured `NO-GO` on the swarm branch, non-determinism hardening, explicit operator decision policy, and simulation-bounded deployment honesty.
- **Highlight:** The project rejected a more complex multi-agent path because it was slower, less accurate, and less robust under the declared gate.
- **Key result:** The weighted-ensemble plus `operator_multiband` path preserved perfect policy metrics on the locked benchmark surface while the swarm spike regressed on every domain.

### For Bio-inspired / Soft Computing Venues (Applied Soft Computing)

- **Lead with:** Novel adaptation of NegSel-AIS to text anomaly detection via rule-based dendritic features
- **Emphasize:** Immune system paradigm mapping (antibodies as miners, selection pressure as validation, autoimmunity as false positives), adaptive r_self from 95th percentile NN distance, 10-dimensional dendritic feature space without model dependency
- **Highlight:** Mathematical foundations from Umair et al. (2025) and three key adaptations for text domain
- **Key result:** NegSel as zero-FP complement to OCSVM in 2-detector ensemble, outperforming 7-detector configuration

---

## Deferred Items

The following are deferred for preparation closer to each journal submission:

- **LaTeX journal templates:** Each target journal has specific formatting requirements (IEEE double-column, Elsevier single-column). Templates will be applied when preparing each submission.
- **Cover letters:** Tailored per journal, incorporating preprint DOIs and any initial citation metrics.
- **Reviewer suggestions:** Require deeper literature search of AIS, anomaly detection, and Bittensor communities to identify appropriate reviewers.
- **Supplementary materials expansion:** Additional experimental details may be requested during peer review.
