"""Tests for the health check API endpoint (MON-01).

Covers:
- GET /health returns structured JSON with status, uptime, last_forward_timestamp,
  connected_miners_count, step
- 503 when validator is not set
- connected_miners_count reflects non-zero-score miners
- last_forward_timestamp updates via record_forward_complete()
"""

from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from starlette.testclient import TestClient

from antigence_subnet.api.health import (
    health_router,
    record_forward_complete,
    set_health_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_validator(n: int = 8):
    """Build a mock validator with controllable scores and metagraph."""
    scores = np.array([0.0, 0.1, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7], dtype=np.float32)[:n]
    metagraph = MagicMock()
    metagraph.n = n
    validator = MagicMock()
    validator.scores = scores
    validator.metagraph = metagraph
    validator.step = 42
    return validator


def _make_app(validator=None):
    """Mount health_router on a fresh FastAPI app."""
    # Reset module state
    import antigence_subnet.api.health as health_mod
    health_mod._validator_ref = None
    health_mod._last_forward_timestamp = None

    if validator is not None:
        set_health_validator(validator)
    app = FastAPI()
    app.include_router(health_router)
    return app


# ---------------------------------------------------------------------------
# Test: GET /health returns 200 with correct fields when validator is set
# ---------------------------------------------------------------------------

class TestHealthEndpointFields:
    def test_health_returns_required_fields(self):
        validator = _make_mock_validator()
        app = _make_app(validator)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "last_forward_timestamp" in data
        assert "connected_miners_count" in data


# ---------------------------------------------------------------------------
# Test: GET /health returns status="healthy" when validator is set
# ---------------------------------------------------------------------------

class TestHealthStatusHealthy:
    def test_health_returns_healthy_status(self):
        validator = _make_mock_validator()
        app = _make_app(validator)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Test: GET /health returns 503 when validator is not set
# ---------------------------------------------------------------------------

class TestHealthNoValidator:
    def test_health_returns_503_when_no_validator(self):
        app = _make_app(validator=None)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Test: connected_miners_count equals number of non-zero-score miners
# ---------------------------------------------------------------------------

class TestConnectedMinersCount:
    def test_connected_miners_count_reflects_nonzero_scores(self):
        validator = _make_mock_validator(n=8)
        # scores = [0.0, 0.1, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7] -> 4 non-zero
        app = _make_app(validator)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected_miners_count"] == 4


# ---------------------------------------------------------------------------
# Test: last_forward_timestamp is null before first forward, numeric after
# ---------------------------------------------------------------------------

class TestLastForwardTimestamp:
    def test_last_forward_timestamp_null_initially(self):
        validator = _make_mock_validator()
        app = _make_app(validator)
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["last_forward_timestamp"] is None

    def test_last_forward_timestamp_updates_after_record(self):
        validator = _make_mock_validator()
        app = _make_app(validator)
        record_forward_complete()
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["last_forward_timestamp"] is not None
        assert isinstance(data["last_forward_timestamp"], int | float)
        assert data["last_forward_timestamp"] > 0


# ---------------------------------------------------------------------------
# Test: step field from validator
# ---------------------------------------------------------------------------

class TestHealthStep:
    def test_health_returns_validator_step(self):
        validator = _make_mock_validator()
        validator.step = 99
        app = _make_app(validator)
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["step"] == 99
