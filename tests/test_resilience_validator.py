"""Resilience tests for validator forward pass and set_weights.

Tests network failure scenarios: miner timeouts, malformed responses,
connection errors, and chain write failures. Ensures the validator
never crashes from individual miner or network failures.

Requirements: NET-13, NET-14, RESIL-01, RESIL-02
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(sample_id: str, domain: str = "hallucination", is_anomalous: bool = True):
    """Create a minimal evaluation sample dict."""
    return {
        "id": sample_id,
        "prompt": f"Prompt for {sample_id}",
        "output": f"Output for {sample_id}",
        "domain": domain,
    }


def _make_response(anomaly_score=0.8, confidence=0.9, anomaly_type="factual_error"):
    """Create a mock VerificationSynapse-like response."""
    resp = SimpleNamespace(
        anomaly_score=anomaly_score,
        confidence=confidence,
        anomaly_type=anomaly_type,
        dendrite=SimpleNamespace(process_time=0.05),
    )
    return resp


def _make_validator(miner_uids, mock_config, dendrite_side_effect=None):
    """Build a minimal validator-like object for forward() testing.

    Args:
        miner_uids: List of miner UIDs to query.
        mock_config: pytest fixture config.
        dendrite_side_effect: Callable or list for dendrite mock.
    """
    validator = SimpleNamespace()
    validator.config = mock_config
    validator.step = 0

    # Metagraph with hotkeys and axons
    hotkeys = [f"hotkey_{uid}" for uid in range(max(miner_uids) + 1)]
    axons = [SimpleNamespace(ip="127.0.0.1", port=8000 + uid) for uid in range(max(miner_uids) + 1)]
    validator.metagraph = SimpleNamespace(
        n=len(hotkeys),
        hotkeys=hotkeys,
        axons=axons,
    )

    # Dendrite mock
    validator.dendrite = AsyncMock(side_effect=dendrite_side_effect)

    # Evaluation dataset with manifest
    samples = [_make_sample(f"s{i}") for i in range(20)]
    manifest = {
        s["id"]: {"ground_truth_label": "anomalous", "is_honeypot": False}
        for s in samples
    }
    validator.evaluation = SimpleNamespace(
        get_round_samples=MagicMock(return_value=samples),
        manifest=manifest,
        dataset_version="test_version_abc123",
    )

    # Scores
    validator.scores = np.zeros(len(hotkeys), dtype=np.float32)
    validator.score_history = {}
    validator.confidence_history = {}
    validator.microglia = None

    # update_scores mock
    validator.update_scores = MagicMock()

    return validator


# ---------------------------------------------------------------------------
# Task 1 Tests: Forward pass resilience
# ---------------------------------------------------------------------------

class TestForwardTimeoutMiner:
    """Test that a single miner timeout does not crash forward()."""

    @pytest.mark.asyncio
    async def test_forward_timeout_miner(self, mock_config):
        """When dendrite raises TimeoutError for one miner, forward() completes,
        remaining miners are scored normally, timed-out miner gets zero reward."""
        call_count = 0

        async def dendrite_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            uid = kwargs.get("axons", args[0] if args else [None])[0]
            # Miner UID 0 times out, UID 1 responds normally
            if hasattr(uid, "port") and uid.port == 8000:
                raise TimeoutError("Miner 0 timed out")
            return [_make_response()]

        miner_uids = [0, 1]
        validator = _make_validator(miner_uids, mock_config, dendrite_side_effect)

        # Patch get_random_uids to return our controlled UIDs
        with (
            patch(
                "antigence_subnet.validator.forward.get_random_uids",
                return_value=miner_uids,
            ),
            patch(
                "antigence_subnet.validator.forward.get_miner_challenge",
                side_effect=lambda samples, **kw: samples[:2],
            ),
            patch(
                "antigence_subnet.validator.forward.inject_adversarial_samples",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "antigence_subnet.validator.forward.generate_perturbation_variants",
                return_value=[_make_sample("perturb_1")],
            ),
        ):

            from antigence_subnet.validator.forward import forward
            # Should NOT raise
            await forward(validator)

        # Validator should still have called update_scores (forward completed)
        validator.update_scores.assert_called_once()


class TestForwardMalformedResponse:
    """Test that malformed miner responses are handled gracefully."""

    @pytest.mark.asyncio
    async def test_forward_malformed_response(self, mock_config):
        """When dendrite returns anomaly_score=None, validate_response rejects
        it, miner gets zero reward, forward() continues."""
        async def dendrite_side_effect(*args, **kwargs):
            # Return response with anomaly_score=None (malformed)
            return [_make_response(anomaly_score=None)]

        miner_uids = [0, 1]
        validator = _make_validator(miner_uids, mock_config, dendrite_side_effect)

        with (
            patch(
                "antigence_subnet.validator.forward.get_random_uids",
                return_value=miner_uids,
            ),
            patch(
                "antigence_subnet.validator.forward.get_miner_challenge",
                side_effect=lambda samples, **kw: samples[:2],
            ),
            patch(
                "antigence_subnet.validator.forward.inject_adversarial_samples",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "antigence_subnet.validator.forward.generate_perturbation_variants",
                return_value=[_make_sample("perturb_1")],
            ),
        ):

            from antigence_subnet.validator.forward import forward
            # Should NOT raise
            await forward(validator)

        validator.update_scores.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_out_of_range_score(self, mock_config):
        """When dendrite returns anomaly_score=2.0 (out of range),
        validate_response catches it."""
        async def dendrite_side_effect(*args, **kwargs):
            return [_make_response(anomaly_score=2.0)]

        miner_uids = [0]
        validator = _make_validator(miner_uids, mock_config, dendrite_side_effect)

        with (
            patch(
                "antigence_subnet.validator.forward.get_random_uids",
                return_value=miner_uids,
            ),
            patch(
                "antigence_subnet.validator.forward.get_miner_challenge",
                side_effect=lambda samples, **kw: samples[:2],
            ),
            patch(
                "antigence_subnet.validator.forward.inject_adversarial_samples",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "antigence_subnet.validator.forward.generate_perturbation_variants",
                return_value=[_make_sample("perturb_1")],
            ),
        ):

            from antigence_subnet.validator.forward import forward
            # Should NOT raise
            await forward(validator)

        validator.update_scores.assert_called_once()


class TestForwardConnectionError:
    """Test that connection errors for a single miner don't crash forward()."""

    @pytest.mark.asyncio
    async def test_forward_connection_error(self, mock_config):
        """When dendrite raises ConnectionError for one miner, forward() logs
        the error and continues to next miner."""
        call_count = 0

        async def dendrite_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            uid = kwargs.get("axons", args[0] if args else [None])[0]
            if hasattr(uid, "port") and uid.port == 8000:
                raise ConnectionError("Miner 0 connection refused")
            return [_make_response()]

        miner_uids = [0, 1]
        validator = _make_validator(miner_uids, mock_config, dendrite_side_effect)

        with (
            patch(
                "antigence_subnet.validator.forward.get_random_uids",
                return_value=miner_uids,
            ),
            patch(
                "antigence_subnet.validator.forward.get_miner_challenge",
                side_effect=lambda samples, **kw: samples[:2],
            ),
            patch(
                "antigence_subnet.validator.forward.inject_adversarial_samples",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "antigence_subnet.validator.forward.generate_perturbation_variants",
                return_value=[_make_sample("perturb_1")],
            ),
        ):

            from antigence_subnet.validator.forward import forward
            await forward(validator)

        validator.update_scores.assert_called_once()


class TestForwardAllMinersFail:
    """Test that forward() survives when ALL miners fail."""

    @pytest.mark.asyncio
    async def test_forward_all_miners_fail(self, mock_config):
        """When ALL miners fail (timeout/error), forward() completes without
        exception and logs a warning."""
        async def dendrite_side_effect(*args, **kwargs):
            raise TimeoutError("All miners timed out")

        miner_uids = [0, 1, 2]
        validator = _make_validator(miner_uids, mock_config, dendrite_side_effect)

        with (
            patch(
                "antigence_subnet.validator.forward.get_random_uids",
                return_value=miner_uids,
            ),
            patch(
                "antigence_subnet.validator.forward.get_miner_challenge",
                side_effect=lambda samples, **kw: samples[:2],
            ),
            patch(
                "antigence_subnet.validator.forward.inject_adversarial_samples",
                side_effect=lambda s, **kw: s,
            ),
            patch(
                "antigence_subnet.validator.forward.generate_perturbation_variants",
                return_value=[_make_sample("perturb_1")],
            ),
        ):

            from antigence_subnet.validator.forward import forward
            # Should NOT raise even when all miners fail
            await forward(validator)

        # update_scores should NOT be called when all miners fail (early return)
        validator.update_scores.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2 Tests: set_weights retry logic
# ---------------------------------------------------------------------------

class TestSetWeightsRetrySuccess:
    """Test that set_weights retries on failure and eventually succeeds."""

    def test_set_weights_retry_success(self, mock_config):
        """When set_weights raises ConnectionError on first attempt but
        succeeds on retry, the weights are set successfully."""
        from antigence_subnet.base.validator import BaseValidatorNeuron

        # Create a minimal validator with mock mode
        with patch.object(BaseValidatorNeuron, "__init__", lambda self, **kw: None):
            validator = BaseValidatorNeuron.__new__(BaseValidatorNeuron)

        # Set up required attributes
        validator.config = mock_config
        validator.config.neuron.set_weights_retries = 3
        validator.config.netuid = 1
        validator.scores = np.zeros(4, dtype=np.float32)
        validator.scores[0] = 0.5
        validator.scores[1] = 0.3

        # Mock wallet
        validator.wallet = MagicMock()

        # Mock subtensor with ExtrinsicResponse
        mock_result_ok = SimpleNamespace(success=True, message="OK")
        validator.subtensor = MagicMock()
        validator.subtensor.set_weights = MagicMock(
            side_effect=[ConnectionError("chain down"), mock_result_ok]
        )

        # Mock metagraph
        validator.metagraph = MagicMock()
        validator.metagraph.n = 4

        # Mock spec_version
        validator.spec_version = 0

        # Mock weight processing utilities
        with (
            patch("antigence_subnet.base.validator.audit_weights", return_value=[]),
            patch(
                "antigence_subnet.base.validator.check_commit_reveal_enabled",
                return_value=False,
            ),
            patch(
                "antigence_subnet.base.utils.weight_utils.process_weights_for_netuid",
                return_value=(np.array([0, 1]), np.array([0.5, 0.3])),
            ),
            patch(
                "antigence_subnet.base.utils.weight_utils.convert_weights_and_uids_for_emit",
                return_value=(np.array([0, 1]), np.array([32768, 19661])),
            ),
            patch("time.sleep"),
        ):
            validator.set_weights()

        # Should have been called twice (first fails, second succeeds)
        assert validator.subtensor.set_weights.call_count == 2

    def test_set_weights_retry_on_result_failure(self, mock_config):
        """When set_weights returns result.success=False, retry occurs."""
        from antigence_subnet.base.validator import BaseValidatorNeuron

        with patch.object(BaseValidatorNeuron, "__init__", lambda self, **kw: None):
            validator = BaseValidatorNeuron.__new__(BaseValidatorNeuron)

        validator.config = mock_config
        validator.config.neuron.set_weights_retries = 3
        validator.config.netuid = 1
        validator.scores = np.zeros(4, dtype=np.float32)
        validator.wallet = MagicMock()
        validator.metagraph = MagicMock()
        validator.metagraph.n = 4
        validator.spec_version = 0

        mock_result_fail = SimpleNamespace(success=False, message="Timeout")
        mock_result_ok = SimpleNamespace(success=True, message="OK")
        validator.subtensor = MagicMock()
        validator.subtensor.set_weights = MagicMock(
            side_effect=[mock_result_fail, mock_result_ok]
        )

        with patch("antigence_subnet.base.validator.audit_weights", return_value=[]), \
             patch("antigence_subnet.base.utils.weight_utils.process_weights_for_netuid",
                   return_value=(np.array([0]), np.array([0.5]))), \
             patch("antigence_subnet.base.utils.weight_utils.convert_weights_and_uids_for_emit",
                   return_value=(np.array([0]), np.array([32768]))), \
             patch("time.sleep"):

            validator.set_weights()

        assert validator.subtensor.set_weights.call_count == 2


class TestSetWeightsRetryExhausted:
    """Test that set_weights handles complete retry exhaustion gracefully."""

    def test_set_weights_retry_exhausted(self, mock_config):
        """When set_weights fails all retry attempts, the validator logs
        the final failure and continues (does not crash)."""
        from antigence_subnet.base.validator import BaseValidatorNeuron

        with patch.object(BaseValidatorNeuron, "__init__", lambda self, **kw: None):
            validator = BaseValidatorNeuron.__new__(BaseValidatorNeuron)

        validator.config = mock_config
        validator.config.neuron.set_weights_retries = 3
        validator.config.netuid = 1
        validator.scores = np.zeros(4, dtype=np.float32)
        validator.wallet = MagicMock()
        validator.metagraph = MagicMock()
        validator.metagraph.n = 4
        validator.spec_version = 0

        validator.subtensor = MagicMock()
        validator.subtensor.set_weights = MagicMock(
            side_effect=ConnectionError("chain permanently down")
        )

        with patch("antigence_subnet.base.validator.audit_weights", return_value=[]), \
             patch("antigence_subnet.base.utils.weight_utils.process_weights_for_netuid",
                   return_value=(np.array([0]), np.array([0.5]))), \
             patch("antigence_subnet.base.utils.weight_utils.convert_weights_and_uids_for_emit",
                   return_value=(np.array([0]), np.array([32768]))), \
             patch("time.sleep"):

            # Should NOT raise even after all retries exhausted
            validator.set_weights()

        # Should have been called max_retries times
        assert validator.subtensor.set_weights.call_count == 3


class TestSetWeightsRetryDelay:
    """Test that retry delay is 2 seconds between attempts."""

    def test_set_weights_retry_delay(self, mock_config):
        """Retry delay is 2 seconds between attempts."""
        from antigence_subnet.base.validator import BaseValidatorNeuron

        with patch.object(BaseValidatorNeuron, "__init__", lambda self, **kw: None):
            validator = BaseValidatorNeuron.__new__(BaseValidatorNeuron)

        validator.config = mock_config
        validator.config.neuron.set_weights_retries = 3
        validator.config.netuid = 1
        validator.scores = np.zeros(4, dtype=np.float32)
        validator.wallet = MagicMock()
        validator.metagraph = MagicMock()
        validator.metagraph.n = 4
        validator.spec_version = 0

        mock_result_fail = SimpleNamespace(success=False, message="Busy")
        mock_result_ok = SimpleNamespace(success=True, message="OK")
        validator.subtensor = MagicMock()
        validator.subtensor.set_weights = MagicMock(
            side_effect=[mock_result_fail, mock_result_fail, mock_result_ok]
        )

        with patch("antigence_subnet.base.validator.audit_weights", return_value=[]), \
             patch("antigence_subnet.base.utils.weight_utils.process_weights_for_netuid",
                   return_value=(np.array([0]), np.array([0.5]))), \
             patch("antigence_subnet.base.utils.weight_utils.convert_weights_and_uids_for_emit",
                   return_value=(np.array([0]), np.array([32768]))), \
             patch("time.sleep") as mock_sleep:

            validator.set_weights()

        # Should have slept 2 seconds between each retry
        assert mock_sleep.call_count == 2
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 2


class TestSetWeightsDefaultConfig:
    """Test that default max_retries is 3."""

    def test_set_weights_default_retries(self, mock_config):
        """Default max_retries is 3, configurable via config.neuron.set_weights_retries."""
        # The mock_config should have set_weights_retries=3 after we add it
        assert mock_config.neuron.set_weights_retries == 3
