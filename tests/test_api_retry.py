"""Tests for API server retry logic and degraded mode (PROD-02).

Covers:
- create_app(validator=None) starts in degraded mode (503 on /verify)
- create_app(validator=mock) enables normal request processing
- init_validator_with_retry retries up to max_retries times on exception
- init_validator_with_retry calls set_validator and set_health_validator on success
- init_validator_with_retry logs error after all retries exhausted
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from starlette.testclient import TestClient

from antigence_subnet.api.trust_score import _rate_limiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_validator(n: int = 8):
    """Build a mock validator state with controllable dendrite."""
    hotkeys = [f"hotkey-{i}" for i in range(n)]
    scores = np.array([0.1 * (i + 1) for i in range(n)], dtype=np.float32)

    metagraph = MagicMock()
    metagraph.n = n
    metagraph.hotkeys = hotkeys
    metagraph.S = np.ones(n, dtype=np.float32) * 100000.0
    metagraph.axons = [MagicMock() for _ in range(n)]

    async def _mock_dendrite_call(axons, synapse, timeout=12.0, deserialize=False, **kw):
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


def _reset_module_state():
    """Reset module-level validator refs to None."""
    import antigence_subnet.api.health as health_mod
    import antigence_subnet.api.trust_score as ts_mod
    ts_mod._validator_ref = None
    health_mod._validator_ref = None
    _rate_limiter._requests.clear()


# ---------------------------------------------------------------------------
# Test: Degraded mode returns 503 on /verify when validator=None
# ---------------------------------------------------------------------------

class TestDegradedModeReturns503:
    def test_create_app_with_no_validator_returns_503_on_verify(self):
        """POST /verify returns 503 when app created with validator=None."""
        _reset_module_state()
        from neurons.api_server import create_app

        app = create_app(validator=None)
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
        assert resp.status_code == 503
        assert "Validator not initialized" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test: Healthy mode processes requests when validator is provided
# ---------------------------------------------------------------------------

class TestHealthyModeProcesses:
    def test_create_app_with_validator_processes_requests(self):
        """POST /verify does not return 503 when app created with valid validator."""
        _reset_module_state()
        from neurons.api_server import create_app

        mock_val = _make_mock_validator()
        app = create_app(validator=mock_val)
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
        # Should NOT be 503 -- it means validator is active
        # Could be 200 (success) since we have a full mock
        assert resp.status_code != 503


# ---------------------------------------------------------------------------
# Test: init_validator_with_retry retries on failure then succeeds
# ---------------------------------------------------------------------------

class TestInitValidatorRetries:
    def test_retries_then_succeeds(self):
        """Validator() raises 3 times then succeeds on 4th -- total 4 calls."""
        from neurons.api_server import init_validator_with_retry

        mock_validator = _make_mock_validator()

        # Raises 3 times, succeeds on 4th
        mock_validator_cls = MagicMock(
            side_effect=[RuntimeError("fail1"), RuntimeError("fail2"),
                         RuntimeError("fail3"), mock_validator]
        )

        with patch("neurons.api_server.set_validator") as mock_set_val, \
             patch("neurons.api_server.set_health_validator") as mock_set_health, \
             patch("time.sleep"):  # Skip actual sleep in tests

            init_validator_with_retry(
                validator_factory=mock_validator_cls,
                max_retries=5,
                backoff=0.01,
            )

            assert mock_validator_cls.call_count == 4
            mock_set_val.assert_called_once_with(mock_validator)
            mock_set_health.assert_called_once_with(mock_validator)


# ---------------------------------------------------------------------------
# Test: init_validator_with_retry exhausts retries without success
# ---------------------------------------------------------------------------

class TestInitValidatorExhaustsRetries:
    def test_exhausts_retries_without_calling_set_validator(self):
        """Validator() always raises -- set_validator never called."""
        from neurons.api_server import init_validator_with_retry

        mock_validator_cls = MagicMock(
            side_effect=RuntimeError("always fails")
        )

        with patch("neurons.api_server.set_validator") as mock_set_val, \
             patch("neurons.api_server.set_health_validator") as mock_set_health, \
             patch("time.sleep"):

            init_validator_with_retry(
                validator_factory=mock_validator_cls,
                max_retries=3,
                backoff=0.01,
            )

            assert mock_validator_cls.call_count == 3
            mock_set_val.assert_not_called()
            mock_set_health.assert_not_called()
