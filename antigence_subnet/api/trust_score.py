"""
Trust Score API endpoint for verification-as-a-service (NET-03).

Provides POST /verify endpoint that:
- Authenticates callers via X-Bittensor-Hotkey header
- Rate limits to 60 requests/min per caller
- Queries top-K miners by EMA score via dendrite
- Returns weighted-average trust score from miner responses
"""

import time
from typing import Protocol, runtime_checkable

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from antigence_subnet.protocol import VerificationSynapse

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TrustScoreRequest(BaseModel):
    """Verification request accepted by POST /verify."""

    prompt: str
    output: str
    domain: str
    code: str | None = None
    context: str | None = None


class TrustScoreResponse(BaseModel):
    """Verification response returned by POST /verify."""

    trust_score: float
    confidence: float
    anomaly_types: list[str]
    contributing_miners: int


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter per caller ID.

    Tracks request timestamps per caller and prunes expired entries
    on each check.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        purge_interval: float = 300.0,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.purge_interval = purge_interval
        self._requests: dict[str, list[float]] = {}
        self._last_purge = time.monotonic()

    def check(self, caller_id: str) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        now = time.monotonic()

        # Periodic purge of all expired entries
        if now - self._last_purge > self.purge_interval:
            self.purge_expired(now)

        timestamps = self._requests.get(caller_id, [])

        # Prune expired entries
        cutoff = now - self.window_seconds
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self.max_requests:
            self._requests[caller_id] = timestamps
            return False

        timestamps.append(now)
        self._requests[caller_id] = timestamps
        return True

    def purge_expired(self, now: float | None = None) -> int:
        """Remove callers with no unexpired timestamps. Returns count removed."""
        now = now or time.monotonic()
        cutoff = now - self.window_seconds
        expired_keys = [
            k for k, timestamps in self._requests.items()
            if not any(t > cutoff for t in timestamps)
        ]
        for k in expired_keys:
            del self._requests[k]
        self._last_purge = now
        return len(expired_keys)


# ---------------------------------------------------------------------------
# Validator state protocol (avoids circular imports)
# ---------------------------------------------------------------------------


@runtime_checkable
class ValidatorState(Protocol):
    """Typing protocol for the validator reference.

    The router receives a reference at mount time via set_validator().
    This avoids importing the actual validator class.
    """

    scores: np.ndarray

    @property
    def metagraph(self): ...

    @property
    def dendrite(self): ...


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_validator_ref: ValidatorState | None = None
_rate_limiter = RateLimiter()


def set_validator(validator: ValidatorState) -> None:
    """Store a reference to the running validator for API use."""
    global _validator_ref
    _validator_ref = validator


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/verify", response_model=TrustScoreResponse)
async def verify_endpoint(
    request: TrustScoreRequest,
    x_bittensor_hotkey: str | None = Header(None),
):
    """Verify an AI output and return an aggregated trust score.

    Authentication: X-Bittensor-Hotkey header must contain a hotkey
    registered in the metagraph.

    Rate limiting: 60 requests per minute per hotkey.

    Scoring: Queries top-K miners by EMA score, computes weighted average
    of anomaly_scores using EMA scores as weights.
    """
    # --- Auth: require hotkey header ---
    if x_bittensor_hotkey is None:
        raise HTTPException(status_code=401, detail="Missing X-Bittensor-Hotkey header")

    validator = _validator_ref
    if validator is None:
        raise HTTPException(status_code=503, detail="Validator not initialized")

    # --- Auth: hotkey must be in metagraph ---
    if x_bittensor_hotkey not in validator.metagraph.hotkeys:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered in metagraph",
        )

    # --- Rate limiting ---
    if not _rate_limiter.check(x_bittensor_hotkey):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (60/min)")

    # --- Select top-K miners by EMA score ---
    k = min(5, len(validator.scores))
    top_uids = np.argsort(validator.scores)[-k:]

    # --- Query miners ---
    synapse = VerificationSynapse(
        prompt=request.prompt,
        output=request.output,
        domain=request.domain,
        code=request.code,
        context=request.context,
    )

    axons = [validator.metagraph.axons[uid] for uid in top_uids]
    responses = await validator.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=12.0,
        deserialize=False,
    )

    # --- Aggregate responses ---
    valid_scores = []
    valid_confidences = []
    valid_weights = []
    anomaly_types_set: set[str] = set()

    for i, resp in enumerate(responses):
        if resp.anomaly_score is not None:
            uid = top_uids[i]
            valid_scores.append(resp.anomaly_score)
            valid_confidences.append(
                resp.confidence if resp.confidence is not None else 0.5
            )
            valid_weights.append(float(validator.scores[uid]))
            if resp.anomaly_type is not None:
                anomaly_types_set.add(resp.anomaly_type)

    if not valid_scores:
        return TrustScoreResponse(
            trust_score=0.5,
            confidence=0.0,
            anomaly_types=[],
            contributing_miners=0,
        )

    # Normalize weights to sum to 1
    weights = np.array(valid_weights, dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(valid_scores)) / len(valid_scores)

    scores_arr = np.array(valid_scores, dtype=np.float64)
    conf_arr = np.array(valid_confidences, dtype=np.float64)

    trust_score = float(np.dot(weights, scores_arr))
    confidence = float(np.dot(weights, conf_arr))

    return TrustScoreResponse(
        trust_score=round(trust_score, 6),
        confidence=round(confidence, 6),
        anomaly_types=sorted(anomaly_types_set),
        contributing_miners=len(valid_scores),
    )
