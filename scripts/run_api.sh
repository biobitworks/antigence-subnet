#!/usr/bin/env bash
# Run Antigence Trust Score API server with sensible testnet defaults.
# All settings can be overridden via environment variables.
#
# Usage:
#   ./scripts/run_api.sh                     # testnet defaults, port 8080
#   API_PORT=9090 ./scripts/run_api.sh       # custom port
#   ./scripts/run_api.sh --netuid 42         # pass-through args
set -euo pipefail

NETWORK="${SUBTENSOR_NETWORK:-test}"
NETUID="${NETUID:-1}"
PORT="${API_PORT:-8080}"
CONFIG_FILE="${CONFIG_FILE:-}"

ARGS=(
    "--subtensor.network" "$NETWORK"
    "--netuid" "$NETUID"
    "--port" "$PORT"
)

if [[ -n "$CONFIG_FILE" ]]; then
    ARGS+=("--config-file" "$CONFIG_FILE")
fi

# Pass through any additional arguments
ARGS+=("$@")

exec python neurons/api_server.py "${ARGS[@]}"
