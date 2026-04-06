# Performance Tuning Guide

Reference for all tunable CLI parameters and guidance on optimizing Antigence subnet performance.

## Validator Parameters

| CLI Flag | Default | Type | Description | Tuning Notes |
|----------|---------|------|-------------|--------------|
| `--neuron.sample_size` | 16 | int | Miners queried per forward pass | Scale with subnet size. For <32 miners use 16 (all). For >64 consider 32. |
| `--neuron.timeout` | 12.0 | float | Dendrite query timeout (seconds) | Lower (6.0) = faster rounds, risks dropping slow miners. Higher (30.0) = accommodates heavy models. |
| `--neuron.moving_average_alpha` | 0.1 | float | EMA smoothing factor for scores | Higher (0.3) = faster reaction, more volatile. Lower (0.05) = smoother, slower adaptation. |
| `--neuron.samples_per_round` | 10 | int | Evaluation samples per miner per round | More = better evaluation precision, longer rounds. |
| `--neuron.n_honeypots` | 2 | int | Honeypot samples per round | More = stronger cheater detection, fewer real samples. 2/10 (20%) recommended. |
| `--neuron.set_weights_interval` | 100 | int | Steps between weight commits | Lower (50) = faster on-chain updates, more extrinsics. Higher (300) = saves fees. |
| `--neuron.set_weights_retries` | 3 | int | Retry attempts for failed set_weights | Increase to 5 on unstable connections. Each retry has 2s backoff. |
| `--neuron.eval_data_dir` | data/evaluation | str | Path to evaluation dataset | Must contain `<domain>/samples.json` and `<domain>/manifest.json`. |
| `--neuron.eval_domain` | hallucination | str | Active evaluation domain | Options: hallucination, code_security, reasoning, bio |

## Microglia Parameters

| CLI Flag | Default | Type | Description | Tuning Notes |
|----------|---------|------|-------------|--------------|
| `--microglia.enabled` | True | flag | Enable/disable surveillance | Disable for minimal deployments. |
| `--microglia.interval` | 100 | int | Steps between surveillance runs | Lower = more frequent alerts, higher CPU. 100 is fine for most deployments. |
| `--microglia.inactive_threshold` | 10 | int | Rounds without response before flagging | Lower = stricter activity requirements. |
| `--microglia.stale_threshold` | 5 | int | Consecutive identical scores before flagging | Catches miners returning hardcoded values. |
| `--microglia.deregistration_threshold` | 50 | int | Rounds inactive before deregistration candidate | Conservative default — production may lower to 30. |
| `--microglia.webhook_url` | None | str | Webhook URL for alert notifications | Set to Slack/Discord webhook for operator alerts. |

## Miner Parameters

| CLI Flag | Default | Type | Description | Tuning Notes |
|----------|---------|------|-------------|--------------|
| `--detector` | (built-in) | str | Detector class path | e.g. `antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector` |
| `--neuron.training_data_dir` | data/evaluation | str | Path to training data | Must contain domain-specific samples. |
| `--neuron.training_domain` | hallucination | str | Training domain | Must match available data. |

## Common Parameters

| CLI Flag | Default | Type | Description |
|----------|---------|------|-------------|
| `--netuid` | 1 | int | Subnet network UID |
| `--mock` | False | flag | Run in mock mode (offline testing) |
| `--neuron.full_path` | ~/.bittensor/neurons | str | State persistence directory |
| `--neuron.device` | cpu | str | Device for neural detectors (cpu/cuda) |

## Tuning Guidance

### Timeout

Controls how long the validator waits for each miner response.

- **6.0s** — Fast rounds. Good for lightweight detectors (Isolation Forest). Risks timing out miners with heavier models.
- **12.0s** (default) — Balanced. Recommended for testnet with mixed miner hardware.
- **30.0s** — Generous. Use when miners run large neural models (Autoencoder on CPU).

Timed-out miners receive a zero score for the round and are logged as failures by the microglia monitor.

### EMA Alpha (moving_average_alpha)

Controls how quickly scores respond to new performance data.

- **0.3** — High reactivity. Scores change rapidly. Good for catching sudden quality drops but volatile.
- **0.1** (default) — Balanced. Smooths noise while adapting within ~10 rounds.
- **0.05** — Conservative. Very smooth scores. Takes ~20 rounds to reflect changes. Good for stable networks.

Formula: `new_score = alpha * round_reward + (1 - alpha) * old_score`

### Set Weights Interval

How often the validator commits weights to the blockchain.

- **50** — Frequent updates. Better for volatile networks. More extrinsic fees.
- **100** (default) — Good balance for testnet.
- **300** — Infrequent. Saves fees on mainnet but weights lag behind performance.

### Honeypot Ratio

Honeypots are known-answer samples that catch miners guessing or memorizing.

- **1/10** (10%) — Minimal anti-cheating. More real evaluation.
- **2/10** (20%, default) — Recommended balance.
- **4/10** (40%) — Aggressive anti-cheating. Fewer real samples for scoring.

Any honeypot failure zeroes the entire round's reward for that miner.

## Reward Formula Reference

The composite reward combines four components:

```
composite = 0.70 * base + 0.10 * calibration + 0.10 * robustness + 0.10 * diversity
```

Where:
- **Base** (70%): `0.7 * precision + 0.3 * recall` — precision-first to penalize false positives
- **Calibration** (10%): ECE-based bonus — rewards miners whose confidence matches accuracy
- **Robustness** (10%): Perturbation stability — rewards consistent scores across input variations
- **Diversity** (10%): `1 - max_cosine_similarity` — rewards unique detection approaches

Honeypot rule: Any honeypot failure zeroes the entire round reward (all components).
