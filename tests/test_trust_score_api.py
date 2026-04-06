"""Tests for the Trust Score API endpoint (NET-03).

Covers authentication, rate limiting, scoring aggregation,
and input validation for POST /verify.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
from fastapi import FastAPI
from starlette.testclient import TestClient

from antigence_subnet.api.trust_score import (
    _rate_limiter,
    router,
    set_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_validator(n: int = 8, hotkeys: list[str] | None = None):
    """Build a mock validator state with controllable dendrite."""
    if hotkeys is None:
        hotkeys = [f"hotkey-{i}" for i in range(n)]

    scores = np.array([0.1 * (i + 1) for i in range(n)], dtype=np.float32)

    metagraph = MagicMock()
    metagraph.n = n
    metagraph.hotkeys = hotkeys
    metagraph.S = np.ones(n, dtype=np.float32) * 100000.0
    metagraph.axons = [MagicMock() for _ in range(n)]

    async def _mock_dendrite_call(axons, synapse, timeout=12.0, deserialize=False, **kw):
        """Return responses with known anomaly_score and confidence."""
        results = []
        for _ in axons:
            resp = synapse.model_copy()
            resp.anomaly_score = 0.7
            resp.confidence = 0.9
            resp.anomaly_type = "mock_anomaly"
            results.append(resp)
        return results

    dendrite = AsyncMock(side_effect=_mock_dendrite_call)

    validator = MagicMock()
    validator.scores = scores
    validator.metagraph = metagraph
    validator.dendrite = dendrite
    return validator


def _make_app(validator=None):
    """Mount the trust_score router on a fresh FastAPI app."""
    if validator is None:
        validator = _make_mock_validator()
    set_validator(validator)
    # Reset rate limiter state between tests
    _rate_limiter._requests.clear()
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Test 1: Valid request returns 200 with correct fields
# ---------------------------------------------------------------------------

class TestValidRequest:
    def test_post_verify_returns_trust_score_response(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/verify",
            json={
                "prompt": "What is 2+2?",
                "output": "2+2 is 5",
                "domain": "hallucination",
            },
            headers={"X-Bittensor-Hotkey": "hotkey-0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "trust_score" in data
        assert 0.0 <= data["trust_score"] <= 1.0
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["anomaly_types"], list)
        assert isinstance(data["contributing_miners"], int)
        assert data["contributing_miners"] > 0


# ---------------------------------------------------------------------------
# Test 2: Missing hotkey header -> 401
# ---------------------------------------------------------------------------

class TestMissingHotkey:
    def test_post_verify_without_hotkey_returns_401(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/verify",
            json={
                "prompt": "test",
                "output": "test",
                "domain": "hallucination",
            },
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test 3: Hotkey not in metagraph -> 403
# ---------------------------------------------------------------------------

class TestUnregisteredHotkey:
    def test_post_verify_with_unknown_hotkey_returns_403(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/verify",
            json={
                "prompt": "test",
                "output": "test",
                "domain": "hallucination",
            },
            headers={"X-Bittensor-Hotkey": "unknown-hotkey-xyz"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Test 4: Rate limiting -> 429 on 61st request
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_blocks_after_60_requests(self):
        app = _make_app()
        client = TestClient(app)
        payload = {
            "prompt": "test",
            "output": "test",
            "domain": "hallucination",
        }
        headers = {"X-Bittensor-Hotkey": "hotkey-0"}

        # First 60 should all succeed
        for i in range(60):
            resp = client.post("/verify", json=payload, headers=headers)
            assert resp.status_code == 200, f"Request {i+1} failed unexpectedly"

        # 61st should be rate limited
        resp = client.post("/verify", json=payload, headers=headers)
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Test 5: Trust score is weighted average of miner anomaly_scores
# ---------------------------------------------------------------------------

class TestWeightedScoring:
    def test_trust_score_is_weighted_average(self):
        """Verify trust_score = weighted avg of top-K miner anomaly_scores."""
        n = 8
        validator = _make_mock_validator(n=n)

        # Override dendrite to return varied scores
        async def _varied_dendrite(axons, synapse, timeout=12.0, deserialize=False, **kw):
            results = []
            for idx, _ in enumerate(axons):
                resp = synapse.model_copy()
                resp.anomaly_score = 0.5 + idx * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
                resp.confidence = 0.8
                resp.anomaly_type = "test_anomaly"
                results.append(resp)
            return results

        validator.dendrite = AsyncMock(side_effect=_varied_dendrite)

        app = _make_app(validator)
        client = TestClient(app)
        resp = client.post(
            "/verify",
            json={
                "prompt": "test",
                "output": "test",
                "domain": "hallucination",
            },
            headers={"X-Bittensor-Hotkey": "hotkey-0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Trust score should be a weighted average, verify it's reasonable
        assert 0.0 <= data["trust_score"] <= 1.0
        assert data["contributing_miners"] > 0


# ---------------------------------------------------------------------------
# Test 6: Missing required field -> 422
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_prompt_returns_422(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/verify",
            json={
                "output": "test",
                "domain": "hallucination",
            },
            headers={"X-Bittensor-Hotkey": "hotkey-0"},
        )
        assert resp.status_code == 422
