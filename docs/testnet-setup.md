# Testnet Setup Guide

This guide is for operators validating the subnet on Bittensor testnet.
If you are evaluating the codebase, contributing detectors, or reviewing the
public repository, start with mock mode from [README.md](../README.md) first
and only use this document when you need real testnet operation.

## Prerequisites

- Python 3.12+ (3.10 minimum, 3.12 recommended)
- Bittensor SDK installed: `pip install 'bittensor>=10.0'`
- Bittensor CLI installed: `pip install -e '.[cli]'`
- Antigence subnet package installed: `pip install -e .`
- Docker and Docker Compose (for Docker deployment method only)

## 1. Create Wallets

Create separate wallets for your miner and validator:

```bash
# Create miner wallet
btcli wallet create --wallet.name miner --wallet.hotkey default

# Create validator wallet
btcli wallet create --wallet.name validator --wallet.hotkey default
```

Store the mnemonic phrases securely. Never share them or commit them to version control.

## 2. Fund Wallets

Fund your wallets with testnet TAO for registration fees. Visit the Bittensor testnet faucet or check the official documentation at [docs.bittensor.com](https://docs.bittensor.com) for current faucet availability.

```bash
# Check wallet balance
btcli wallet balance --wallet.name miner --subtensor.network test
btcli wallet balance --wallet.name validator --subtensor.network test
```

## 3. Register on Testnet

Register both wallets on the Antigence subnet. Replace `<NETUID>` with the subnet's network UID (assigned at subnet creation).

```bash
# Register miner
btcli subnet register \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network test \
  --netuid <NETUID>

# Register validator
btcli subnet register \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network test \
  --netuid <NETUID>
```

## 4. Deploy

Three deployment methods are available. Choose the one that fits your setup.

### Method A: Bare Python

Run neurons directly with Python:

```bash
# Start miner with reference IsolationForest detector
python neurons/miner.py \
  --subtensor.network test \
  --netuid <NETUID> \
  --wallet.name miner \
  --wallet.hotkey default \
  --detector antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector \
  --neuron.training_data_dir data/evaluation \
  --neuron.training_domain hallucination

# Start validator
python neurons/validator.py \
  --subtensor.network test \
  --netuid <NETUID> \
  --wallet.name validator \
  --wallet.hotkey default \
  --neuron.eval_data_dir data/evaluation \
  --neuron.eval_domain hallucination \
  --neuron.samples_per_round 10 \
  --neuron.n_honeypots 2 \
  --neuron.set_weights_interval 100

# Start API server (optional)
python neurons/api_server.py \
  --subtensor.network test --netuid <NETUID> \
  --port 8080
```

Custom detectors can be specified via the `--detector` flag:

```bash
python neurons/miner.py \
  --subtensor.network test \
  --netuid <NETUID> \
  --wallet.name miner \
  --wallet.hotkey default \
  --detector my_package.my_module.MyDetector
```

The detector must implement `antigence_subnet.miner.detector.BaseDetector`.

### Method B: Runner Scripts

Runner scripts provide sensible testnet defaults and environment variable overrides:

```bash
# Default: testnet, netuid 1, wallet "default/default"
./scripts/run_miner.sh
./scripts/run_validator.sh
./scripts/run_api.sh

# Override via environment variables
NETUID=42 WALLET_NAME=my_miner ./scripts/run_miner.sh

# Pass-through CLI args
./scripts/run_miner.sh --neuron.training_domain code_security

# With TOML config
CONFIG_FILE=antigence_subnet.toml ./scripts/run_miner.sh
```

**Environment variable reference:**

| Variable | Default | Description |
|----------|---------|-------------|
| SUBTENSOR_NETWORK | test | Network (test, finney, local) |
| NETUID | 1 | Subnet UID |
| WALLET_NAME | default | Wallet name |
| WALLET_HOTKEY | default | Hotkey name |
| BT_WALLET_PATH | ~/.bittensor/wallets | Wallet directory |
| CONFIG_FILE | (none) | Path to TOML config file |
| API_PORT | 8080 | API server port (run_api.sh only) |

### Phase 94 Live Validation Flow (Operator-Only / Internal Process)

Phase 94 uses a governed pre-action gate. The canonical execution mode is
`same-host-private`, the copied live config and registration proof artifacts are
`internal-only`, and the operator must approve the exact deployment candidate
before registration, launch, `test_set_weights_testnet`, or 24h collection starts.

Public readers do not need this section to evaluate the codebase. It is kept
here as operator-only process guidance for governed testnet validation.

```bash
mkdir -p .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config
cp antigence_subnet.toml.example \
  .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml
```

Before touching the chain, create the no-start checklist and candidate manifest:

- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/no-start-checklist.json`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/deployment-candidate.json`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/operator-approval.json`

The repository now includes checked-in pending stubs for those three files plus
a helper that regenerates them from current HEAD, the copied config, and env
presence. Run it after you finish any local Phase 94 prep and before requesting
human approval:

```bash
cp .env.phase94.example .env.phase94
# fill .env.phase94 with the real Phase 94 values first

python scripts/generate_phase94_governance_artifacts.py \
  --artifact-root .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts \
  --config-file .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml
```

The generator auto-loads `.env.phase94` when present, or you can pass
`--env-file /path/to/file`. The helper intentionally leaves
`operator-approval.json` in a non-live state with `"approved": false`. A human
must review the regenerated candidate and checklist, then set the approval
record manually only after verifying the exact candidate hash is still current.
The checked-in JSON files stay as public-safe templates and should not be
treated as real local operator state until you regenerate them.

Before asking for approval, run the local preflight validator. It fail-closes
on stale candidate hashes, missing env presence, missing wallet confirmations,
or incomplete approval fields:

```bash
python scripts/phase94_preflight.py \
  --artifact-root .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts
```

After the human approver updates `operator-approval.json`, rerun the same check
with `--require-approval` before starting any governed live action:

```bash
python scripts/phase94_preflight.py \
  --artifact-root .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts \
  --require-approval
```

The checklist must record:

- `execution_mode = "same-host-private"`
- approved `PHASE94_MINER_METRICS_PORT` and `PHASE94_VALIDATOR_METRICS_PORT`
- funded-wallet confirmations for the miner and validator hotkeys
- required env-var presence booleans including `BT_WALLET_PATH` as redacted or presence-only
- `btcli` availability/version
- redaction rules that forbid mnemonic phrases, seed values, private keys, and raw coldkey material in env vars, copied configs, JSON artifacts, or shared logs

Confirm the copied config still preserves the winning Phase 93 policy tuple:

```toml
[validator.policy]
mode = "operator_multiband"
high_threshold = 0.5
low_threshold = 0.493536
min_confidence = 0.6
```

Archive both registration proofs after metagraph confirmation:

- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/registration/miner-metagraph-proof.json`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/registration/validator-metagraph-proof.json`

Each proof file should include at least:
`subtensor_network`, `netuid`, `wallet_name`, `wallet_hotkey`, `hotkey_ss58`,
`uid`, `is_registered`, and the metagraph block height or hash used for the proof.
Keep them `classification = "internal-only"` with `public_release = false`.

Launch the real miner and validator with the copied config plus the Phase 94
observability env vars:

```bash
CONFIG_FILE=.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml \
NETUID="$PHASE94_NETUID" WALLET_NAME="$PHASE94_MINER_WALLET_NAME" WALLET_HOTKEY="$PHASE94_MINER_WALLET_HOTKEY" \
PHASE94_MINER_METRICS_PORT="${PHASE94_MINER_METRICS_PORT}" \
PHASE94_MINER_TELEMETRY_EXPORT_DIR=.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/stability-24h/miner \
PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS=300 \
./scripts/run_miner.sh

CONFIG_FILE=.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml \
NETUID="$PHASE94_NETUID" WALLET_NAME="$PHASE94_VALIDATOR_WALLET_NAME" WALLET_HOTKEY="$PHASE94_VALIDATOR_WALLET_HOTKEY" \
PHASE94_VALIDATOR_METRICS_PORT="${PHASE94_VALIDATOR_METRICS_PORT}" \
PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR=.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/stability-24h/validator \
PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS=300 \
./scripts/run_validator.sh
```

Run the strict-live 100+ round validation and the concurrent metrics collector:

```bash
.venv/bin/pytest tests/test_network_integration.py -q -k test_set_weights_testnet

python scripts/validate_testnet.py \
  --strict-live \
  --config-file .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml \
  --artifact-root .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts \
  --output-dir .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/validation \
  --subtensor.network test \
  --netuid "$PHASE94_NETUID" \
  --wallet.name "$PHASE94_VALIDATOR_WALLET_NAME" \
  --wallet.hotkey "$PHASE94_VALIDATOR_WALLET_HOTKEY" \
  --rounds 100 \
  --min-rounds 100 \
  --duration-hours 24

python scripts/phase94_collect_metrics.py \
  --miner-url "http://127.0.0.1:${PHASE94_MINER_METRICS_PORT}/metrics" \
  --validator-url "http://127.0.0.1:${PHASE94_VALIDATOR_METRICS_PORT}/metrics" \
  --interval-seconds 60 \
  --duration-hours 24 \
  --collector-mode validation-24h \
  --output-dir .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/stability-24h \
  --config-file .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml \
  --candidate-manifest .planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/deployment-candidate.json
```

The passing Phase 94 artifact set is:

- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/validation/validation-manifest.json`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/validation/testnet-validation-report.json`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/stability-24h/prometheus-scrapes.jsonl`
- `.planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/stability-24h/run-summary.json`

Do not record mnemonic phrases, seed phrases, private keys, or raw coldkey material
in the checklist, candidate manifest, copied config, registration proof, or logs.

### Method C: Docker Compose

Build and run with Docker Compose using profiles for selective service startup.

```bash
# Build images
docker compose build

# Copy and edit environment file
cp .env.example .env
# Edit .env with your wallet name, netuid, etc.

# Start validator only
docker compose --profile validator up -d

# Start miner only
docker compose --profile miner up -d

# Start API server only
docker compose --profile api up -d

# Start all services
docker compose --profile validator --profile miner --profile api up -d

# View logs
docker compose logs -f miner

# Stop
docker compose --profile miner down
```

Running `docker compose up` without `--profile` starts nothing because all services use profiles for selective startup.

**Volume mounts:**

| Host Path | Container Path | Mode | Purpose |
|-----------|---------------|------|---------|
| ~/.bittensor/wallets | /home/antigence/.bittensor/wallets | read-only | Wallet keys |
| (named volume) validator-state | /home/antigence/.bittensor/neurons | read-write | Validator state |
| (named volume) miner-state | /home/antigence/.bittensor/neurons | read-write | Miner state |
| ./data | /app/data | read-only | Evaluation data |

**Important**: The container runs as user `antigence` (UID 1000), not root. If wallet files are not readable by this user, either:

1. `chmod 644` your wallet key files, or
2. Add `user: "$(id -u):$(id -g)"` to the service in docker-compose.yml

See `.env.example` for all Docker environment variables and the wallet UID mismatch warning.

## 5. Configuration

### TOML Config File

```bash
# Copy example config
cp antigence_subnet.toml.example antigence_subnet.toml
# Edit settings as needed
```

**Search order** (from `config_file.py`):

1. Explicit `--config-file <path>` argument (highest priority)
2. `antigence_subnet.toml` in current working directory
3. `~/.antigence/config.toml`

CLI args always override TOML values. Example:

```bash
# TOML sets device = "cpu", but CLI overrides to cuda
python neurons/miner.py --neuron.device cuda --config-file antigence_subnet.toml
```

### Key Settings

```toml
[neuron]
device = "cpu"                  # "cpu" or "cuda"
shutdown_timeout = 30           # Seconds to wait for graceful shutdown

[neuron.logging]
level = "info"                  # debug, info, warning, error

[validator]
samples_per_round = 10          # Evaluation samples per forward pass
n_honeypots = 2                 # Honeypot samples per round
set_weights_interval = 100      # Steps between weight updates

[miner]
training_data_dir = "data/evaluation"

[miner.detectors]
hallucination = "antigence_subnet.miner.detectors.domain_packs.hallucination.detector.HallucinationDetector"
# Add more domains for multi-domain miner:
# code_security = "antigence_subnet.miner.detectors.domain_packs.code_security.detector.CodeSecurityDetector"

[api]
port = 8080
```

### Environment Variables (Docker)

Reference the `.env.example` file for Docker environment variables. The same variables listed in the runner scripts table above (SUBTENSOR_NETWORK, NETUID, WALLET_NAME, WALLET_HOTKEY, BT_WALLET_PATH, API_PORT) are used by Docker Compose.

### TOML Config with Docker

To use a TOML config file with Docker Compose, uncomment the config volume mount in `docker-compose.yml`:

```yaml
volumes:
  # Uncomment this line:
  - ./antigence_subnet.toml:/app/antigence_subnet.toml:ro
```

The config file is auto-discovered in `/app` (the container's working directory). No `--config-file` argument is needed.

## 6. Verify Operation

Check the subnet metagraph to confirm neurons are registered and weights are being set:

```bash
# View metagraph
btcli subnet metagraph --netuid <NETUID> --subtensor.network test
```

Check the API health endpoint (if running):

```bash
curl http://localhost:8080/health
```

Expected log output:

```
Forward pass complete | Step N | Queried X miners | Y samples
set_weights on chain successfully!
```

## 7. State Persistence

### What Gets Saved

- **Miner**: Fitted detector model state (e.g., `iforest_state.joblib` for IsolationForest)
- **Validator**: EMA scores, hotkey tracking, step counter (numpy `.npz` format)

### Where

Default state directory: `~/.bittensor/neurons/<netuid>/<hotkey>/`

Inside Docker: named volumes `miner-state` and `validator-state` mounted to `/home/antigence/.bittensor/neurons/`.

### Verify Restore-on-Restart

```bash
# Stop and restart miner
# Check logs for "Loaded detector state" message (not "Fitting detector")
# A restored miner skips the fit() step and resumes scoring immediately
```

## 8. Mock Mode (Development)

For offline development and testing without a live testnet:

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run validator in mock mode
python neurons/validator.py --mock --netuid 1

# Run miner in mock mode
python neurons/miner.py --mock --netuid 1
```

Mock mode uses `MockSubtensor`, `MockMetagraph`, and `MockDendrite` for fully offline operation. No wallet funding or testnet registration is required.

### Running Tests

```bash
# Full test suite (uses mock infrastructure)
pytest tests/ -x -q

# E2E cycle test only
pytest tests/test_e2e_cycle.py -v
```

## Notes

- Always default to `--subtensor.network test` for safety. Never use `finney` (mainnet) without explicit intent.
- Testnet operations and faucet availability may change. Refer to the [official Bittensor documentation](https://docs.bittensor.com) for the latest instructions.
- Do not embed real wallet keys or mnemonics in code or configuration files.
- The validator's evaluation dataset should not be shared with miners -- this is the hidden ground truth that drives selection pressure.
