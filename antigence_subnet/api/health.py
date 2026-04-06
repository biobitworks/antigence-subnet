"""
Health check endpoint for validator monitoring (MON-01).

Provides GET /health endpoint that returns:
- status: "healthy" | "degraded" | "unavailable"
- uptime_seconds: time since module load
- last_forward_timestamp: epoch seconds of last completed forward pass
- connected_miners_count: miners with non-zero scores
- step: current validator step

Follows the same pattern as trust_score.py: module-level _validator_ref,
set_health_validator() for dependency injection, health_router for mounting.
"""

import time

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_validator_ref = None
_start_time: float = time.monotonic()
_last_forward_timestamp: float | None = None

# Threshold in seconds: if last forward was more than this long ago,
# report "degraded" status.
_DEGRADED_THRESHOLD_SECONDS = 300.0


def set_health_validator(validator) -> None:
    """Store a reference to the running validator for health checks."""
    global _validator_ref
    _validator_ref = validator


def record_forward_complete() -> None:
    """Record the timestamp of the most recent completed forward pass."""
    global _last_forward_timestamp
    _last_forward_timestamp = time.time()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

health_router = APIRouter()


@health_router.get("/health")
async def health_endpoint():
    """Return validator health status.

    Returns 200 with JSON health payload when validator is available.
    Returns 503 with {"status": "unavailable"} when validator is not set.
    """
    validator = _validator_ref
    if validator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable"},
        )

    # Compute uptime
    uptime_seconds = time.monotonic() - _start_time

    # Determine status
    status = "healthy"
    if _last_forward_timestamp is not None:
        elapsed = time.time() - _last_forward_timestamp
        if elapsed > _DEGRADED_THRESHOLD_SECONDS:
            status = "degraded"

    # Count connected miners (non-zero scores)
    try:
        scores = validator.scores
        connected_miners_count = int(np.count_nonzero(scores))
    except Exception:
        connected_miners_count = 0

    # Get step
    try:
        step = int(validator.step)
    except Exception:
        step = 0

    return {
        "status": status,
        "uptime_seconds": round(uptime_seconds, 2),
        "last_forward_timestamp": _last_forward_timestamp,
        "connected_miners_count": connected_miners_count,
        "step": step,
    }
