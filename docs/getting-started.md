```
                         Fab           Fab
                     ___/ \___     ___/ \___
                    |   VH    |   |   VH    |
                    |___|  |__|   |__|  |___|
                    |   VL    |   |   VL    |
                    |_________|   |_________|
                         \    S-S    /
                          \  S-S   /
                           | Fc  |
                           |     |
                           |_____|
                           |     |
                           |_____|

      A N T I G E N C E   S U B N E T
   Immune-Inspired Verification for Bittensor
```

# Getting Started Guide

This guide walks you through cloning, installing, and running Antigence Subnet from scratch. Tested on macOS (Apple Silicon) and Ubuntu 24.04 (GitHub Actions CI).

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.10, < 3.15 | 3.12 recommended |
| pip | >= 23.0 | Comes with Python |
| Rust/Cargo | Latest | **Linux only** — needed to compile `bittensor-wallet` from source. macOS/Windows get pre-built wheels. Install: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Git | Any recent | For cloning |

## Step 1: Clone

```bash
git clone https://github.com/biobitworks/antigence-subnet.git
cd antigence-subnet
```

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

## Step 3: Install

**Basic install** (run neurons):
```bash
pip install -e .
```

**Dev install** (run tests):
```bash
pip install -e '.[dev]'
```

**Full install** (dev + sentence-transformers for hallucination domain):
```bash
pip install -e '.[dev,sbert]'
```

**With CLI tools** (wallet management, subnet registration):
```bash
pip install -e '.[cli]'
```

### What gets installed

| Package | Purpose |
|---------|---------|
| bittensor 10.2.0 | Subnet SDK (Axon, Dendrite, Synapse, Metagraph) |
| scikit-learn | Classical anomaly detection (Isolation Forest, LOF) |
| numpy, scipy | Numerical operations, reward computation |
| fastapi, uvicorn | Miner HTTP server (via Bittensor Axon) |
| aiohttp | Validator HTTP client (via Bittensor Dendrite) |

## Step 4: Verify Installation

```bash
# Check version
python -c "import antigence_subnet; print(antigence_subnet.__version__)"
# Expected: 10.0.0

# Check validator loads
python neurons/validator.py --mock --help
# Expected: usage message with --logging options

# Check miner loads
python neurons/miner.py --mock --help
# Expected: usage message with --logging options
```

## Step 5: Run Tests

```bash
pytest tests/ -x -q
```

Expected output (approximate):
```
1545 passed, 17 skipped, 84 warnings
```

### Why are tests skipped?

Skipped tests are expected and do not indicate failures. They fall into three categories:

**Optional dependencies not installed:**

| Tests | Dependency | Install with |
|-------|-----------|--------------|
| `test_ollama_harness`, `test_phase81_*`, `test_phase83_*` | `ollama` | Local install: [ollama.ai](https://ollama.ai) |
| `test_sbert_*`, `test_model_manager`, `test_hallucination_pack` (partial) | `sentence-transformers` | `pip install -e '.[sbert]'` |
| `test_metrics_api` | `prometheus_client` | `pip install prometheus-client` |
| `test_autoencoder_parity` | `torch` | `pip install torch` |

**Features under development (wired in future milestones):**

| Test | Feature | Status |
|------|---------|--------|
| `test_cold_start::TestValidatorIntegration` | Cold-start warmup protocol | Planned |
| `test_validator_agreement::TestValidatorIntegration` | Cross-validator agreement tracking | Planned |
| `test_collusion::TestCollusionForwardIntegration` | Collusion zeroing in reward path | Planned |
| `test_forward_integration::test_forward_queries_*` | End-to-end scoring pipeline | Planned |
| `test_dendritic_cell::test_classify_with_weight_manager` | Adaptive weight manager integration | Planned |

**Environment-conditional:**

| Test | Condition |
|------|-----------|
| `test_validate_testnet::TestPhase94StrictLive` | Requires testnet deployment artifacts |
| `test_domain_packs_integration` (partial) | Requires all domain data loaded |

## Step 6: Run in Mock Mode

Mock mode simulates the Bittensor network locally. No wallet, no registration, no TAO required.

**Terminal 1 — Validator:**
```bash
python neurons/validator.py --mock --netuid 1
```

**Terminal 2 — Miner:**
```bash
python neurons/miner.py --mock --netuid 1
```

The validator will query the miner with evaluation samples. The miner routes to domain-specific detectors and returns anomaly scores. Watch the logs to see the immune-inspired detection pipeline in action.

## Step 7: Run Lint (Contributors)

```bash
pip install ruff
ruff check .
# Expected: All checks passed!
```

## Project Structure

```
antigence-subnet/
├── antigence_subnet/          # Core package
│   ├── miner/                 # Miner-side code
│   │   ├── detectors/         # Domain-specific anomaly detectors
│   │   ├── orchestrator/      # Immune orchestrator (ensemble, weighting)
│   │   └── data.py            # Training data loading
│   ├── validator/             # Validator-side code
│   │   ├── evaluation.py      # Evaluation dataset management
│   │   ├── scoring.py         # Reward computation
│   │   ├── forward.py         # Forward pass (query miners)
│   │   ├── collusion.py       # Collusion detection
│   │   ├── microglia.py       # Network health surveillance
│   │   └── agreement.py       # Cross-validator agreement
│   └── protocol.py            # Synapse definitions (API contract)
├── neurons/
│   ├── validator.py           # Validator entry point
│   └── miner.py               # Miner entry point
├── data/
│   └── evaluation/            # Evaluation datasets (4 domains)
│       ├── hallucination/     # LLM hallucination detection
│       ├── code_security/     # Code security analysis
│       ├── reasoning/         # Agent reasoning audit
│       └── bio/               # Bio pipeline verification
├── scripts/                   # Benchmark and utility scripts
├── tests/                     # Test suite (1500+ tests)
├── docs/                      # Documentation
│   ├── manuscript.md          # Research manuscript
│   ├── testnet-setup.md       # Testnet operator guide
│   └── research/              # Research notes
├── configs/                   # Configuration profiles
├── SECURITY.md                # Security policy + typosquatting warning
├── CONTRIBUTING.md            # Contributor guide
└── pyproject.toml             # Package metadata
```

## Domain Packs

Each domain pack contains evaluation samples with known ground truth:

| Domain | Detects | Samples | Honeypots |
|--------|---------|---------|-----------|
| Hallucination | Fabricated facts, invented citations, false statistics in LLM output | 220 | 20 |
| Code Security | SQL injection, XSS, path traversal, insecure patterns | 220 | 15 |
| Reasoning | Logical fallacies, circular arguments, unsupported conclusions | 220 | 11 |
| Bio | Anomalous measurements, impossible values, data integrity issues | 220 | 21 |

All evaluation samples are synthetic (`"source": "synthetic"` in metadata). They are designed to test detector accuracy, not to represent real-world data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    BITTENSOR NETWORK                     │
│                                                          │
│  ┌──────────────┐          ┌──────────────────────────┐ │
│  │  VALIDATOR    │          │  MINER (Antibody)        │ │
│  │              │  Dendrite │                           │ │
│  │  Evaluation  │ ──────── │  Axon (FastAPI)          │ │
│  │  Dataset     │  Synapse  │    │                     │ │
│  │              │ ◄──────── │  Protocol Router         │ │
│  │  Scoring     │          │    ├── Hallucination Det. │ │
│  │  (precision  │          │    ├── Code Security Det. │ │
│  │   first)     │          │    ├── Reasoning Det.     │ │
│  │              │          │    └── Bio Det.           │ │
│  │  Collusion   │          │                           │ │
│  │  Detection   │          │  Immune Orchestrator      │ │
│  │              │          │    ├── Danger Modulator   │ │
│  │  Microglia   │          │    ├── NK Cell (fast)     │ │
│  │  Surveillance│          │    └── B Cell (adaptive)  │ │
│  └──────────────┘          └──────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  TRUST SCORE API (Verification-as-a-Service)      │   │
│  │  External consumers query consensus anomaly scores │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Immune System Analogy

| Biological | Antigence | Role |
|------------|-----------|------|
| Antibody | Miner detector | Recognizes specific anomaly patterns |
| Antigen | Evaluation sample | Test case with known ground truth |
| Self | Normal AI output | Legitimate, non-anomalous |
| Non-self | Anomalous AI output | Hallucinated, insecure, flawed |
| Dendritic Cell | Feature extractor | Extracts signals from raw output |
| NK Cell | Fast detector | Quick statistical anomaly check |
| B Cell | Adaptive detector | Learns from training data |
| Danger Signal | Anomaly score | How suspicious the output is |
| Thymic Selection | Negative selection | Training on self to detect non-self |
| Microglia | Network monitor | Detects coordinated attacks, sybils |
| Autoimmune | False positive | Flagging legitimate output as anomalous |

## Troubleshooting

### `ModuleNotFoundError: No module named 'bittensor_wallet'` (Linux)
You need Rust/Cargo installed. See Prerequisites.

### Tests skip with "ollama not available in CI"
Expected. These tests require a local Ollama server with GPU inference. They run in development environments with `ollama` installed.

### `pip install` fails with "Building wheel for bittensor-wallet failed"
Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
Then restart your shell and retry.

### Validator shows "No evaluation data" in mock mode
The `data/evaluation/` directory must be present. If you cloned with `--depth 1`, some files may be missing. Clone without shallow depth.

## Next Steps

- Read the [research manuscript](docs/manuscript.md) for the scientific foundation
- Review [testnet-setup.md](docs/testnet-setup.md) for operator deployment
- Explore [CONTRIBUTING.md](CONTRIBUTING.md) to contribute detectors
- Check [SECURITY.md](SECURITY.md) for the security policy

## Links

- **Repository:** https://github.com/biobitworks/antigence-subnet
- **Bittensor:** https://bittensor.com
- **Bittensor SDK Docs:** https://docs.learnbittensor.org/
