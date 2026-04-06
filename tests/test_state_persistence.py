"""
Tests for state persistence (PROTO-06).

Verifies validator save/load state, corruption recovery, missing state
handling, and EMA score update correctness.
"""

import os

import numpy as np
import pytest

from neurons.miner import Miner
from neurons.validator import Validator


@pytest.mark.asyncio
async def test_validator_save_load_state(mock_config, tmp_path):
    """Test validator saves state and loads it correctly on restart."""
    # Create validator and set known scores
    validator = Validator(config=mock_config)
    validator.scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    validator.step = 42
    validator.hotkeys = ["hk1", "hk2", "hk3", "hk4", "hk5"]

    validator.save_state()

    # Create a new validator and load the saved state
    validator2 = Validator(config=mock_config)
    validator2.load_state()

    assert validator2.step == 42
    np.testing.assert_array_almost_equal(
        validator2.scores, np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    )
    assert validator2.hotkeys == ["hk1", "hk2", "hk3", "hk4", "hk5"]


@pytest.mark.asyncio
async def test_validator_save_contains_expected_keys(mock_config, tmp_path):
    """Test that saved .npz file contains step, scores, hotkeys keys."""
    validator = Validator(config=mock_config)
    validator.step = 10
    validator.save_state()

    state_file = os.path.join(str(tmp_path), "state.npz")
    state = np.load(state_file, allow_pickle=True)

    assert "step" in state.files
    assert "scores" in state.files
    assert "hotkeys" in state.files


@pytest.mark.asyncio
async def test_validator_corrupted_state_recovery(mock_config, tmp_path):
    """Test corrupted state files are handled gracefully (D-12)."""
    validator = Validator(config=mock_config)

    # Write garbage bytes to state file
    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(state_file, "wb") as f:
        f.write(b"garbage corrupted data bytes")

    # load_state should not raise; reinitializes defaults
    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)
    assert all(s == 0.0 for s in validator.scores)


@pytest.mark.asyncio
async def test_validator_missing_state_recovery(mock_config, tmp_path):
    """Test missing state file is handled gracefully."""
    validator = Validator(config=mock_config)

    # Ensure no state file exists
    state_file = os.path.join(str(tmp_path), "state.npz")
    if os.path.exists(state_file):
        os.remove(state_file)

    # load_state should not raise; uses defaults
    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)


@pytest.mark.asyncio
async def test_miner_save_load_state(mock_config):
    """Test miner save_state/load_state are callable (placeholder)."""
    miner = Miner(config=mock_config)

    # Should not raise -- placeholder methods
    miner.save_state()
    miner.load_state()


@pytest.mark.asyncio
async def test_ema_score_update(mock_config):
    """Test EMA score update with known inputs."""
    validator = Validator(config=mock_config)
    validator.config.neuron.moving_average_alpha = 0.1

    # Initialize scores for 2 UIDs
    validator.scores = np.array([0.5, 0.5], dtype=np.float32)

    rewards = np.array([1.0, 0.0], dtype=np.float32)
    uids = [0, 1]
    validator.update_scores(rewards, uids)

    # EMA: 0.1 * reward + 0.9 * old_score
    # UID 0: 0.1 * 1.0 + 0.9 * 0.5 = 0.55
    # UID 1: 0.1 * 0.0 + 0.9 * 0.5 = 0.45
    np.testing.assert_almost_equal(validator.scores[0], 0.55, decimal=4)
    np.testing.assert_almost_equal(validator.scores[1], 0.45, decimal=4)
