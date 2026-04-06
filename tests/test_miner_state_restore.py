"""Tests for miner detector state restore-before-fit logic (PROD-05).

Verifies that:
- When saved detector state exists and is valid, fit() is NOT called
- When saved detector state exists, load_state() IS called
- When no saved state exists, fit() IS called with training samples
- When load_state() raises an exception, fit() IS called as fallback
- When state exists but detector reports is_fitted=False after load, fit() IS called
"""

import os
from unittest.mock import MagicMock, patch

from antigence_subnet.miner.detector import BaseDetector


def _make_mock_detector(is_fitted_after_load=True, load_raises=None):
    """Create a mock detector with controllable behavior."""
    detector = MagicMock(spec=BaseDetector)
    detector.domain = "hallucination"
    detector.get_info.return_value = {
        "name": "MockDetector",
        "domain": "hallucination",
        "is_fitted": False,
    }

    def _load_state(path):
        if load_raises:
            raise load_raises
        # After successful load, update is_fitted
        detector.get_info.return_value = {
            "name": "MockDetector",
            "domain": "hallucination",
            "is_fitted": is_fitted_after_load,
        }

    detector.load_state.side_effect = _load_state
    return detector


class TestMinerStateRestore:
    """Test restore-before-fit logic in Miner.__init__."""

    @patch("neurons.miner.load_training_samples")
    @patch("neurons.miner.load_detector")
    def test_restore_skips_fit(self, mock_load_detector, mock_load_samples, mock_config):
        """When valid saved state exists, fit() is NOT called."""
        detector = _make_mock_detector(is_fitted_after_load=True)
        mock_load_detector.return_value = detector
        mock_load_samples.return_value = [{"prompt": "test", "output": "test"}]

        # Create state dir so os.path.exists returns True
        state_dir = os.path.join(str(mock_config.neuron.full_path), "detector_state")
        os.makedirs(state_dir, exist_ok=True)

        from neurons.miner import Miner
        _miner = Miner(config=mock_config)

        detector.load_state.assert_called_once_with(state_dir)
        detector.fit.assert_not_called()

    @patch("neurons.miner.load_training_samples")
    @patch("neurons.miner.load_detector")
    def test_no_state_does_fit(self, mock_load_detector, mock_load_samples, mock_config):
        """When no saved state exists, fit() IS called."""
        detector = _make_mock_detector()
        mock_load_detector.return_value = detector
        training_samples = [{"prompt": "test", "output": "test"}]
        mock_load_samples.return_value = training_samples

        # Ensure state dir does NOT exist
        state_dir = os.path.join(str(mock_config.neuron.full_path), "detector_state")
        if os.path.exists(state_dir):
            os.rmdir(state_dir)

        from neurons.miner import Miner
        _miner = Miner(config=mock_config)

        detector.load_state.assert_not_called()
        detector.fit.assert_called_once_with(training_samples)

    @patch("neurons.miner.load_training_samples")
    @patch("neurons.miner.load_detector")
    def test_load_failure_does_fit(self, mock_load_detector, mock_load_samples, mock_config):
        """When load_state() raises, fit() IS called as fallback."""
        detector = _make_mock_detector(load_raises=RuntimeError("corrupt state"))
        mock_load_detector.return_value = detector
        training_samples = [{"prompt": "test", "output": "test"}]
        mock_load_samples.return_value = training_samples

        # Create state dir so restore is attempted
        state_dir = os.path.join(str(mock_config.neuron.full_path), "detector_state")
        os.makedirs(state_dir, exist_ok=True)

        from neurons.miner import Miner
        _miner = Miner(config=mock_config)

        detector.load_state.assert_called_once()
        detector.fit.assert_called_once_with(training_samples)

    @patch("neurons.miner.load_training_samples")
    @patch("neurons.miner.load_detector")
    def test_load_not_fitted_does_fit(self, mock_load_detector, mock_load_samples, mock_config):
        """When state exists but detector is not fitted after load, fit() IS called."""
        detector = _make_mock_detector(is_fitted_after_load=False)
        mock_load_detector.return_value = detector
        training_samples = [{"prompt": "test", "output": "test"}]
        mock_load_samples.return_value = training_samples

        # Create state dir
        state_dir = os.path.join(str(mock_config.neuron.full_path), "detector_state")
        os.makedirs(state_dir, exist_ok=True)

        from neurons.miner import Miner
        _miner = Miner(config=mock_config)

        detector.load_state.assert_called_once()
        detector.fit.assert_called_once_with(training_samples)
