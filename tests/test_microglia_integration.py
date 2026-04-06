"""Integration tests for microglia surveillance in the validator lifecycle.

Tests:
- Validator init creates MicrogliaMonitor instance with correct config
- CLI args propagate to microglia thresholds
- Surveillance cycle produces valid SubnetHealthMetrics
- Microglia disabled flag prevents surveillance
- Surveillance runs at correct interval in run loop
"""

import argparse
from unittest.mock import patch

import bittensor as bt
import numpy as np

from antigence_subnet.base.validator import BaseValidatorNeuron
from antigence_subnet.validator.microglia import MicrogliaMonitor, SubnetHealthMetrics

# ---------------------------------------------------------------------------
# Test validator subclass for integration tests
# ---------------------------------------------------------------------------


class IntegrationValidator(BaseValidatorNeuron):
    """Minimal validator subclass for microglia integration testing."""

    def __init__(self, config=None):
        super().__init__(config=config)

    async def forward(self):
        """No-op forward pass for integration tests."""
        pass


# ---------------------------------------------------------------------------
# Test: Validator init creates MicrogliaMonitor
# ---------------------------------------------------------------------------


class TestMicrogliaValidatorInit:
    """Tests for microglia initialization in BaseValidatorNeuron.__init__."""

    def test_validator_has_microglia_attribute(self, mock_config):
        """Validator in mock mode has self.microglia (MicrogliaMonitor instance)."""
        validator = IntegrationValidator(config=mock_config)
        assert hasattr(validator, "microglia")
        assert isinstance(validator.microglia, MicrogliaMonitor)

    def test_default_microglia_interval(self, mock_config):
        """Default microglia_interval is 100."""
        validator = IntegrationValidator(config=mock_config)
        assert validator.microglia_interval == 100

    def test_microglia_enabled_by_default(self, mock_config):
        """Microglia is enabled by default."""
        validator = IntegrationValidator(config=mock_config)
        assert validator.microglia_enabled is True

    def test_default_inactive_threshold(self, mock_config):
        """Default inactive_threshold propagates to MicrogliaMonitor."""
        validator = IntegrationValidator(config=mock_config)
        assert validator.microglia.inactive_threshold == 10

    def test_default_stale_threshold(self, mock_config):
        """Default stale_threshold propagates to MicrogliaMonitor."""
        validator = IntegrationValidator(config=mock_config)
        assert validator.microglia.stale_threshold == 5

    def test_default_deregistration_threshold(self, mock_config):
        """Default deregistration_threshold propagates to MicrogliaMonitor."""
        validator = IntegrationValidator(config=mock_config)
        assert validator.microglia.deregistration_threshold == 50


# ---------------------------------------------------------------------------
# Test: CLI args control microglia configuration
# ---------------------------------------------------------------------------


class TestMicrogliaCLIArgs:
    """Tests for microglia CLI arg propagation."""

    def test_custom_inactive_threshold(self, mock_wallet, tmp_path):
        """Custom --microglia.inactive_threshold=20 propagates."""
        parser = argparse.ArgumentParser()
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.Axon.add_args(parser)
        parser.add_argument("--netuid", type=int, default=1)
        parser.add_argument("--mock", action="store_true", default=False)
        parser.add_argument("--neuron.full_path", type=str, default=str(tmp_path))
        parser.add_argument("--neuron.device", type=str, default="cpu")
        parser.add_argument("--neuron.sample_size", type=int, default=16)
        parser.add_argument("--neuron.timeout", type=float, default=12.0)
        parser.add_argument("--neuron.moving_average_alpha", type=float, default=0.1)
        parser.add_argument("--neuron.eval_data_dir", type=str, default="data/evaluation")
        parser.add_argument("--neuron.eval_domain", type=str, default="hallucination")
        parser.add_argument("--neuron.samples_per_round", type=int, default=10)
        parser.add_argument("--neuron.n_honeypots", type=int, default=2)
        parser.add_argument("--neuron.set_weights_interval", type=int, default=100)
        parser.add_argument(
            "--detector", type=str,
            default="antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector",
        )
        parser.add_argument("--neuron.training_data_dir", type=str, default="data/evaluation")
        parser.add_argument("--neuron.training_domain", type=str, default="hallucination")
        parser.add_argument("--microglia.interval", type=int, default=100)
        parser.add_argument("--microglia.webhook_url", type=str, default=None)
        parser.add_argument("--microglia.inactive_threshold", type=int, default=10)
        parser.add_argument("--microglia.stale_threshold", type=int, default=5)
        parser.add_argument("--microglia.deregistration_threshold", type=int, default=50)
        parser.add_argument("--microglia.enabled", action="store_true", default=True)
        parser.add_argument("--microglia.alert_cooldown", type=int, default=10)
        parser.add_argument("--reward.base_weight", type=float, default=0.70)
        parser.add_argument("--reward.calibration_weight", type=float, default=0.10)
        parser.add_argument("--reward.robustness_weight", type=float, default=0.10)
        parser.add_argument("--reward.diversity_weight", type=float, default=0.10)
        parser.add_argument("--reward.decision_threshold", type=float, default=0.5)

        config = bt.Config(
            parser,
            args=[
                "--mock",
                "--netuid", "1",
                "--wallet.name", mock_wallet.name,
                "--wallet.hotkey", mock_wallet.hotkey_str,
                "--wallet.path", mock_wallet.path,
                "--neuron.full_path", str(tmp_path),
                "--no_prompt",
                "--microglia.inactive_threshold", "20",
                "--microglia.interval", "50",
            ],
        )

        validator = IntegrationValidator(config=config)
        assert validator.microglia.inactive_threshold == 20
        assert validator.microglia_interval == 50


# ---------------------------------------------------------------------------
# Test: Surveillance cycle from validator context
# ---------------------------------------------------------------------------


class TestMicrogliaSurveillanceCycle:
    """Tests for running surveillance cycle through the validator's microglia."""

    def test_surveillance_cycle_returns_metrics(self, mock_config):
        """Manually calling run_surveillance_cycle returns SubnetHealthMetrics."""
        validator = IntegrationValidator(config=mock_config)

        # Set up some scores
        validator.scores = np.array(
            [0.5] * validator.metagraph.n, dtype=np.float32
        )

        metrics = validator.microglia.run_surveillance_cycle(
            scores=validator.scores,
            score_history=validator.score_history,
            hotkeys=list(validator.metagraph.hotkeys),
            n_total=validator.metagraph.n,
            current_step=validator.step,
        )

        assert isinstance(metrics, SubnetHealthMetrics)
        assert 0.0 <= metrics.inflammation_score <= 1.0
        assert metrics.threat_level in ("low", "medium", "high", "critical")

    def test_surveillance_runs_in_run_loop(self, mock_config):
        """Surveillance cycle fires at the correct interval in run()."""
        validator = IntegrationValidator(config=mock_config)
        validator.microglia_interval = 1  # fire every step
        validator.config.neuron.set_weights_interval = 9999

        call_count = 0
        surveillance_called = False

        def sleep_side_effect(_duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                validator.should_exit = True

        original_run = validator.microglia.run_surveillance_cycle

        def tracking_run(*args, **kwargs):
            nonlocal surveillance_called
            surveillance_called = True
            return original_run(*args, **kwargs)

        with (
            patch("time.sleep", side_effect=sleep_side_effect),
            patch.object(validator, "sync"),
            patch.object(validator, "save_state"),
            patch.object(validator, "load_state"),
            patch.object(
                validator.microglia,
                "run_surveillance_cycle",
                side_effect=tracking_run,
            ),
        ):
            validator.run()

        assert surveillance_called, "Surveillance cycle was not called in run loop"


# ---------------------------------------------------------------------------
# Test: Microglia disabled
# ---------------------------------------------------------------------------


class TestMicrogliaDisabled:
    """Tests for microglia disabled mode."""

    def test_disabled_no_surveillance_in_run(self, mock_config):
        """With microglia_enabled=False, surveillance does not run."""
        validator = IntegrationValidator(config=mock_config)
        validator.microglia_enabled = False
        validator.microglia_interval = 1  # would fire every step if enabled
        validator.config.neuron.set_weights_interval = 9999

        call_count = 0

        def sleep_side_effect(_duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                validator.should_exit = True

        with (
            patch("time.sleep", side_effect=sleep_side_effect),
            patch.object(validator, "sync"),
            patch.object(validator, "save_state"),
            patch.object(validator, "load_state"),
            patch.object(
                validator.microglia,
                "run_surveillance_cycle",
            ) as mock_surveillance,
        ):
            validator.run()

        mock_surveillance.assert_not_called()
