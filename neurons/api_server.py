"""
Trust Score API server entry point.

Runs a standalone FastAPI server that connects to a validator instance
and exposes the POST /verify endpoint for verification-as-a-service.

Supports degraded mode: the server starts immediately with validator=None
(returning 503 on /verify and /health), then initializes the validator in
a background thread with exponential-backoff retry.  Once initialization
succeeds, the API transitions to healthy mode automatically.

Usage:
    python neurons/api_server.py --subtensor.network test --netuid 1 --port 8080
"""

import argparse
import sys
import threading
import time
from collections.abc import Callable

import bittensor as bt
import uvicorn
from fastapi import FastAPI

from antigence_subnet.api.health import health_router, set_health_validator
from antigence_subnet.api.metrics import metrics_router
from antigence_subnet.api.trust_score import router, set_validator
from antigence_subnet.utils.config_file import apply_toml_defaults


def create_app(validator=None) -> FastAPI:
    """Create FastAPI app with trust score, health, and metrics routers.

    If validator is None, API starts in degraded mode (503 on /verify
    and /health).  Call set_validator() / set_health_validator() later
    to transition to healthy mode.
    """
    app = FastAPI(
        title="Antigence Trust Score API",
        description="Verification-as-a-service for Bittensor subnet",
        version="1.0.0",
    )
    if validator is not None:
        set_validator(validator)
        set_health_validator(validator)
    app.include_router(router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    return app


def init_validator_with_retry(
    validator_factory: Callable | None = None,
    max_retries: int = 5,
    backoff: float = 5.0,
) -> None:
    """Initialize validator with retry logic.  On success, sets module-level refs.

    Parameters
    ----------
    validator_factory:
        Callable that returns a validator instance.  Defaults to
        ``neurons.validator.Validator`` (imported lazily to avoid
        circular imports).
    max_retries:
        Maximum number of initialization attempts.
    backoff:
        Base backoff in seconds; actual wait is ``backoff * attempt``.
    """
    if validator_factory is None:
        from neurons.validator import Validator
        validator_factory = Validator

    for attempt in range(1, max_retries + 1):
        try:
            validator = validator_factory()
            set_validator(validator)
            set_health_validator(validator)
            bt.logging.info(
                f"Validator initialized successfully (attempt {attempt}/{max_retries})"
            )
            return
        except Exception as e:
            bt.logging.warning(
                f"Validator init attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                time.sleep(backoff * attempt)

    bt.logging.error(
        f"Validator initialization failed after {max_retries} retries -- "
        "API running in degraded mode (503 on /verify and /health)"
    )


def main():
    parser = argparse.ArgumentParser(description="Antigence Trust Score API")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument(
        "--subtensor.network",
        type=str,
        default="test",
        dest="network",
        help="Subtensor network",
    )
    parser.add_argument("--netuid", type=int, default=1, help="Subnet UID")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to TOML config file (default: auto-discover antigence_subnet.toml)",
    )

    # Inject TOML config defaults before parsing CLI args
    pre_args, _ = parser.parse_known_args()
    config_file_path = getattr(pre_args, "config_file", None)
    apply_toml_defaults(parser, config_path=config_file_path)

    args = parser.parse_args()

    bt.logging.info(f"Starting Trust Score API on port {args.port}")
    bt.logging.info(f"Network: {args.network}, Netuid: {args.netuid}")

    # Start API immediately in degraded mode (validator=None -> 503 on endpoints)
    app = create_app(validator=None)

    # Launch validator init in background thread with retry
    init_thread = threading.Thread(
        target=init_validator_with_retry,
        kwargs={"max_retries": 5, "backoff": 5.0},
        daemon=True,
    )
    init_thread.start()

    try:
        # The API server intentionally binds 0.0.0.0 so the validator can
        # be reached by miners and operators on the host network. This is
        # the documented deployment model; if you need a tighter bind run
        # behind a reverse proxy or firewall the port. nosec: B104
        uvicorn.run(app, host="0.0.0.0", port=args.port)  # nosec B104
    except KeyboardInterrupt:
        bt.logging.info("API server shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
