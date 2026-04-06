# Troubleshooting Guide

Common errors when running Antigence subnet neurons, with symptoms, causes, and
solutions.

This guide is written for public readers as well as operators. If you are
exploring the repository or contributing code, use mock mode first and treat
the testnet-specific sections below as operator guidance rather than a required
setup path.

## Registration Failure

**Symptom:** `Hotkey <address> not registered on subnet <netuid>` error on startup. Neuron exits immediately.

**Cause:** The wallet hotkey is not registered on the target subnet via btcli.

**Solution:**
```bash
btcli subnet register \
  --wallet.name <name> --wallet.hotkey <hotkey> \
  --subtensor.network test --netuid <NETUID>
```

Verify registration:
```bash
btcli wallet overview --wallet.name <name> --subtensor.network test
```

## Port Binding / Axon Startup Failure

**Symptom:** `Address already in use` or `OSError: [Errno 48]` when starting a miner.

**Cause:** Another process (or a previous miner instance) is using the Axon port (default 8091).

**Solution:**
```bash
# Find process using the port
lsof -i :8091

# Kill if it's a stale miner
kill <PID>

# Or use an alternate port
python neurons/miner.py --axon.port 8092 ...
```

## Metagraph Sync Failures

**Symptom:** `Syncing metagraph...` hangs indefinitely, or connection errors to subtensor.

**Cause:** Network connectivity to the subtensor endpoint is disrupted, or the endpoint is down.

**Solution:**
```bash
# Verify endpoint reachability
curl -s https://test.finney.opentensor.ai:443

# Try specifying the endpoint explicitly
python neurons/validator.py \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 ...

# For local development, use mock mode
python neurons/validator.py --mock --netuid 1
```

## Weight Setting Failures

**Symptom:** `set_weights failed after N retries` in validator logs.

**Cause:** Subtensor connection drops during weight commit, insufficient stake for validator permit, or chain rate limiting.

**Solution:**
1. Check validator permit status:
   ```bash
   btcli subnet metagraph --netuid <NETUID> --subtensor.network test
   ```
   Look for `validator_permit: True` on your UID.

2. Ensure sufficient stake (validators need minimum stake to set weights).

3. Increase retry count:
   ```bash
   python neurons/validator.py --neuron.set_weights_retries 5 ...
   ```

4. Check subtensor connectivity — the validator retries automatically with 2-second backoff.

## Evaluation Dataset Not Found

**Symptom:** `No evaluation dataset loaded. Cannot run forward pass.` warning. Validator runs but never scores miners.

**Cause:** The evaluation data directory is missing or doesn't contain the expected files.

**Solution:**
```bash
# Verify data directory structure
ls data/evaluation/hallucination/
# Expected: samples.json, manifest.json

# Specify a different directory
python neurons/validator.py \
  --neuron.eval_data_dir /path/to/eval/data \
  --neuron.eval_domain hallucination ...
```

Each domain needs a `<domain>/samples.json` (evaluation samples) and `<domain>/manifest.json` (ground truth labels).

## Miner Detector Unfitted

**Symptom:** `Detector is unfitted -- scores will be meaningless` warning on miner startup.

**Cause:** Training data directory is missing or contains no samples matching the configured domain.

**Solution:**
```bash
# Check training data exists
ls data/evaluation/hallucination/samples.json

# Specify training data location
python neurons/miner.py \
  --neuron.training_data_dir data/evaluation \
  --neuron.training_domain hallucination ...
```

The detector trains on startup using normal (non-anomalous) samples filtered from the training data.

## Mock Mode Issues

**Symptom:** Unexpected behavior when running with `--mock` flag.

**Notes:**
- Mock mode uses `MockSubtensor` and `MockMetagraph` — no chain connection
- Wallets are still real (coldkey/hotkey) but registration is auto-handled
- `set_weights` in mock mode always succeeds (returns mocked `ExtrinsicResponse`)
- Use mock mode for development and testing only — not for testnet validation

```bash
# Run both neurons in mock mode
python neurons/validator.py --mock --netuid 1
python neurons/miner.py --mock --netuid 1
```
