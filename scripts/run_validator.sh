#!/usr/bin/env bash
# Run Antigence validator with sensible testnet defaults.
# All settings can be overridden via environment variables.
#
# Usage:
#   ./scripts/run_validator.sh                    # testnet defaults
#   NETUID=42 ./scripts/run_validator.sh          # custom netuid
#   ./scripts/run_validator.sh --logging.level debug  # pass-through args
#
# Phase 94 observability env vars pass through to neurons/validator.py:
#   PHASE94_VALIDATOR_METRICS_PORT
#   PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR
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

exec python neurons/validator.py "${ARGS[@]}"
