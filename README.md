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

[![CI](https://github.com/biobitworks/antigence-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/biobitworks/antigence-subnet/actions/workflows/ci.yml)
[![Security](https://github.com/biobitworks/antigence-subnet/actions/workflows/security.yml/badge.svg)](https://github.com/biobitworks/antigence-subnet/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Bittensor](https://img.shields.io/badge/bittensor-10.2.0-green.svg)](https://bittensor.com)

A decentralized verification subnet for [Bittensor](https://bittensor.com) where miners run anomaly detectors and validators apply selection pressure through hidden evaluation datasets. The network produces a shared trust score for AI outputs.

Current tracked release: `v10.0` with the final weighted-ensemble plus
`operator_multiband` policy stack. See
[docs/release/v10.0-release-notes.md](docs/release/v10.0-release-notes.md) for
the release summary and evidence trail.

**Immune-inspired design:** Miners are antibodies competing to detect anomalies (non-self) without false-positiving on legitimate outputs (self). Validators apply selection pressure. The best detectors survive and earn TAO rewards.

## Project Overview

Antigence Subnet applies the biological immune system's self/non-self discrimination paradigm to AI output verification. Miners act as antibodies: they learn what normal (self) AI outputs look like and flag anomalous (non-self) outputs -- hallucinated facts, insecure code, flawed reasoning, or corrupted data. Validators apply selection pressure by testing miners against hidden evaluation datasets with known ground truth, rewarding accuracy and penalizing false positives (autoimmune responses).

The economic model is straightforward: miners earn TAO by accurately detecting anomalies. The reward function is precision-first (70/30 precision/recall weighting) with additional bonuses for calibration, robustness, and diversity. Honeypot samples with known labels are injected each round -- miners that fail honeypots receive zero reward for that round. This creates competitive pressure where only effective detectors survive.

Four domain packs ship with the subnet: LLM hallucination detection, code security analysis, agent reasoning audit, and bio pipeline verification. External consumers can query the Trust Score API for consensus anomaly scores on any AI output, making verification available as a service to the broader Bittensor ecosystem.

## Features

- **4 domain packs**: Hallucination detection, code security, agent reasoning, bio pipelines
- **Precision-first reward**: 70% base + 10% calibration + 10% robustness + 10% diversity
- **Anti-cheating**: Per-miner unique challenges, perturbation stability, sybil detection, weight audit
- **Microglia surveillance**: Network-level health monitoring with coordinated attack detection
- **Verification-as-a-Service**: Trust Score API for external consumers
- **Pluggable detectors**: Miners bring their own models via the `BaseDetector` interface

## Prerequisites

- **Python** >= 3.10, < 3.15 (3.12 recommended)
- **Rust and Cargo** (Linux only) — required to compile `bittensor-wallet` and `bittensor-drand` from source. macOS and Windows users get pre-built wheels. Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **pip** >= 23.0

## Quick Start

Start in local mock mode unless you are actively operating on Bittensor
testnet. Most contributors can evaluate the project, run the neurons, and work
on the codebase without a funded wallet or subnet registration.

### Local Evaluation (Recommended)

No testnet wallet or registration required. Run locally with mock
infrastructure:

```bash
pip install -e .

# Run validator and miner in separate terminals
python neurons/validator.py --mock --netuid 1
python neurons/miner.py --mock --netuid 1

# Run tests
pip install -e '.[dev]'
pytest tests/ -x -q
```

### Testnet Deployment (Operator Path)

Use this path only if you are validating real subnet behavior on Bittensor
testnet. It requires funded wallets, subnet registration, and the operator
steps documented in [docs/testnet-setup.md](docs/testnet-setup.md).

```bash
pip install -e '.[cli]'

# Create and fund wallets (see docs/testnet-setup.md for faucet details)
btcli wallet create --wallet.name miner --wallet.hotkey default
btcli wallet create --wallet.name validator --wallet.hotkey default

# Register on testnet
btcli subnet register --wallet.name miner --wallet.hotkey default \
  --subtensor.network test --netuid <NETUID>
btcli subnet register --wallet.name validator --wallet.hotkey default \
  --subtensor.network test --netuid <NETUID>

# Run with runner scripts
./scripts/run_miner.sh
./scripts/run_validator.sh
```

For complete operator instructions, Docker deployment, and governed live-gate
details, see [docs/testnet-setup.md](docs/testnet-setup.md).

## Architecture

Validators query miners with evaluation samples. Miners route to domain-specific detectors. Rewards are precision-first (70/30 precision/recall) with honeypot, calibration, robustness, and diversity checks.

```
Validator                          Miner
   |                                  |
   |-- Sample eval dataset            |
   |-- Per-miner challenge selection  |
   |-- Query via Dendrite ----------- |-- Receive via Axon
   |                                  |-- Route to detector (domain)
   |                                  |-- Return anomaly scores
   |-- Validate response <----------- |
   |-- Compute composite reward       |
   |-- EMA update + set_weights       |
   +-- Microglia surveillance         |
```

## Deployment Methods

Three deployment methods are supported. If you are only evaluating the project,
start with mock mode above; these deployment paths are primarily for operators
running on testnet. See [docs/testnet-setup.md](docs/testnet-setup.md) for full
details.

**Runner scripts** -- Shell wrappers with sensible defaults and env var overrides:

```bash
./scripts/run_miner.sh                         # testnet defaults
NETUID=42 WALLET_NAME=my_miner ./scripts/run_miner.sh  # custom settings
```

**Docker Compose** -- Profile-based, all services optional:

```bash
docker compose --profile miner up -d           # miner only
docker compose --profile validator --profile miner --profile api up -d  # all
```

**TOML config** -- Place `antigence_subnet.toml` in the working directory or `~/.antigence/config.toml`. CLI args override TOML values:

```bash
cp antigence_subnet.toml.example antigence_subnet.toml
# Edit settings, then run normally -- config is loaded automatically
./scripts/run_miner.sh
```

## Custom Detectors

Miners can bring their own detection models by implementing the `BaseDetector` interface:

```python
from typing import Optional
from antigence_subnet.miner.detector import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    domain = "hallucination"

    def fit(self, samples: list[dict]) -> None:
        self._is_fitted = True

    async def detect(self, prompt: str, output: str,
                     code: Optional[str] = None,
                     context: Optional[str] = None) -> DetectionResult:
        return DetectionResult(score=0.5, confidence=0.8,
                               anomaly_type="normal")

    def get_info(self) -> dict:
        return {"name": "MyDetector", "domain": self.domain,
                "version": "1.0.0", "backend": "custom",
                "is_fitted": getattr(self, "_is_fitted", False)}
```

```bash
python neurons/miner.py --detector my_package.MyDetector ...
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Requirements

- Python >=3.10, <3.15 (3.12 recommended)
- bittensor SDK 10.2.0
- scikit-learn (CPU detectors)
- PyTorch (optional, GPU detectors): `pip install -e '.[torch]'`
- Trust Score API: `pip install -e '.[api]'`

## Documentation

| Guide | Description |
|-------|-------------|
| [Testnet Setup Guide](docs/testnet-setup.md) | Wallet creation, registration, deployment |
| [Contributing](CONTRIBUTING.md) | Build custom detectors, submit domain packs |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and solutions |
| [Performance Tuning](docs/performance-tuning.md) | Detector and validator optimization |
| [Training Data](docs/miner-training-data.md) | Dataset format and evaluation data |
| [v10.0 Release Notes](docs/release/v10.0-release-notes.md) | Release summary, evidence chain, and handoff notes |

## Public Data Notes

Some evaluation fixtures and security examples intentionally contain synthetic
credential-like strings such as fake API keys, tokens, or passwords. They are
included as adversarial examples for anomaly detection and should not be read as
real leaked secrets.

## License

MIT
