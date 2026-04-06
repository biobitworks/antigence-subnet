#!/usr/bin/env bash
# Run Antigence miner with sensible testnet defaults.
# All settings can be overridden via environment variables.
#
# Usage:
#   ./scripts/run_miner.sh                         # testnet defaults
#   NETUID=42 ./scripts/run_miner.sh               # custom netuid
#   ./scripts/run_miner.sh --neuron.training_domain code_anomaly  # pass-through
#
# Phase 94 observability env vars pass through to neurons/miner.py:
#   PHASE94_MINER_METRICS_PORT
#   PHASE94_MINER_TELEMETRY_EXPORT_DIR
#   PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS
set -euo pipefail

NETWORK="${SUBTENSOR_NETWORK:-test}"
NETUID="${NETUID:-1}"
WALLET_NAME="${WALLET_NAME:-default}"
WALLET_HOTKEY="${WALLET_HOTKEY:-default}"
WALLET_PATH="${BT_WALLET_PATH:-$HOME/.bittensor/wallets}"
CONFIG_FILE="${CONFIG_FILE:-}"

ARGS=(
    "--subtensor.network" "$NETWORK"
    "--netuid" "$NETUID"
    "--wallet.name" "$WALLET_NAME"
    "--wallet.hotkey" "$WALLET_HOTKEY"
    "--wallet.path" "$WALLET_PATH"
)

if [[ -n "$CONFIG_FILE" ]]; then
    ARGS+=("--config-file" "$CONFIG_FILE")
fi

# Pass through any additional arguments
ARGS+=("$@")

exec python neurons/miner.py "${ARGS[@]}"
