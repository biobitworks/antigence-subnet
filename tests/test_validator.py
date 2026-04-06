"""
Tests for validator neuron (PROTO-02, PROTO-05).

Verifies validator instantiation, dendrite setup, score initialization,
forward signature, neuron type, CLI args, and EMA update.
"""

import inspect

import numpy as np

from neurons.validator import Validator


def test_validator_instantiation_mock(mock_config):
    """Test Validator creates successfully with mock config."""
    validator = Validator(config=mock_config)
    assert validator is not None
    assert validator.config.mock is True


def test_validator_has_dendrite(mock_config):
    """Test validator has a dendrite."""
    validator = Validator(config=mock_config)
    assert validator.dendrite is not None


def test_validator_has_scores(mock_config):
    """Test validator scores is numpy array of zeros with correct shape."""
    validator = Validator(config=mock_config)
    assert isinstance(validator.scores, np.ndarray)
    assert validator.scores.dtype == np.float32
    assert validator.scores.shape == (validator.metagraph.n,)
    assert np.all(validator.scores == 0.0)


def test_validator_forward_signature(mock_config):
    """Test Validator.forward takes no args (self only) -- Pitfall 3."""
    validator = Validator(config=mock_config)
    sig = inspect.signature(validator.forward)
    params = list(sig.parameters.keys())
    # forward() should have no parameters (self is implicit)
    assert params == []


def test_validator_neuron_type(mock_config):
    """Test validator neuron type is 'ValidatorNeuron'."""
    validator = Validator(config=mock_config)
    assert validator.neuron_type == "ValidatorNeuron"


def test_validator_cli_args():
    """Test Validator.add_args() produces config with expected attributes."""
    config = Validator.add_args()
    # Config should have all expected attributes from parsed args
    assert hasattr(config, "netuid")
    assert hasattr(config, "mock")
    assert hasattr(config, "neuron")
    assert hasattr(config.neuron, "sample_size")
    assert hasattr(config.neuron, "timeout")
    assert hasattr(config.neuron, "moving_average_alpha")


def test_validator_update_scores(mock_config):
    """Test EMA update with known inputs produces expected outputs."""
    validator = Validator(config=mock_config)
    validator.config.neuron.moving_average_alpha = 0.2

    # Set up known scores
    validator.scores = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    rewards = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    uids = [0, 1, 2]
    validator.update_scores(rewards, uids)

    # EMA: alpha * reward + (1 - alpha) * old_score
    # UID 0: 0.2 * 0.0 + 0.8 * 1.0 = 0.8
    # UID 1: 0.2 * 1.0 + 0.8 * 0.0 = 0.2
    # UID 2: 0.2 * 0.5 + 0.8 * 0.5 = 0.5
    np.testing.assert_almost_equal(validator.scores[0], 0.8, decimal=4)
    np.testing.assert_almost_equal(validator.scores[1], 0.2, decimal=4)
    np.testing.assert_almost_equal(validator.scores[2], 0.5, decimal=4)
