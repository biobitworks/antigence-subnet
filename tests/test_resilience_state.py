"""
Tests for validator state persistence resilience (RESIL-04).

Verifies atomic save_state via write-to-temp-then-rename pattern,
and enhanced load_state corruption detection with recovery to defaults.
"""

import os

import numpy as np
import pytest

from neurons.validator import Validator


@pytest.mark.asyncio
async def test_save_state_atomic_write(mock_config, tmp_path):
    """save_state writes to temp file first, then atomically renames via os.replace."""
    validator = Validator(config=mock_config)
    validator.scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    validator.step = 10

    validator.save_state()

    # Verify the final state.npz exists and is valid
    state_file = os.path.join(str(tmp_path), "state.npz")
    assert os.path.exists(state_file), "state.npz should exist after save_state"

    state = np.load(state_file, allow_pickle=True)
    assert "step" in state.files
    assert "scores" in state.files
    assert "hotkeys" in state.files
    assert int(state["step"]) == 10

    # Verify no temp files left behind
    tmp_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp.npz")]
    assert len(tmp_files) == 0, "No temp files should remain after successful save"


@pytest.mark.asyncio
async def test_save_state_uses_os_replace(mock_config, tmp_path, monkeypatch):
    """Verify save_state calls os.replace for atomic rename."""
    validator = Validator(config=mock_config)
    validator.step = 5

    replace_calls = []
    original_replace = os.replace

    def tracking_replace(src, dst):
        replace_calls.append((src, dst))
        return original_replace(src, dst)

    monkeypatch.setattr("os.replace", tracking_replace)

    validator.save_state()

    assert len(replace_calls) == 1, "os.replace should be called exactly once"
    src, dst = replace_calls[0]
    assert dst.endswith("state.npz"), "Destination should be state.npz"
    assert ".tmp" in src or "tmp" in src.lower(), "Source should be a temp file"


@pytest.mark.asyncio
async def test_save_state_uses_mkstemp(mock_config, tmp_path, monkeypatch):
    """Verify save_state uses tempfile.mkstemp for temp file creation."""
    validator = Validator(config=mock_config)
    validator.step = 5

    mkstemp_calls = []
    import tempfile
    original_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(**kwargs):
        result = original_mkstemp(**kwargs)
        mkstemp_calls.append(kwargs)
        return result

    monkeypatch.setattr("tempfile.mkstemp", tracking_mkstemp)

    validator.save_state()

    assert len(mkstemp_calls) >= 1, "tempfile.mkstemp should be called"


@pytest.mark.asyncio
async def test_load_state_corrupted_bytes(mock_config, tmp_path):
    """When state.npz contains random garbage bytes, load_state recovers to defaults."""
    validator = Validator(config=mock_config)

    # Write random garbage to state file
    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(state_file, "wb") as f:
        f.write(os.urandom(256))

    # load_state should not raise
    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)
    assert all(s == 0.0 for s in validator.scores)


@pytest.mark.asyncio
async def test_load_state_empty_file(mock_config, tmp_path):
    """When state.npz is zero bytes (empty file from interrupted write), load_state recovers."""
    validator = Validator(config=mock_config)

    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(state_file, "wb"):
        pass  # Write nothing -- zero bytes

    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)
    assert all(s == 0.0 for s in validator.scores)


@pytest.mark.asyncio
async def test_load_state_missing_step_key(mock_config, tmp_path):
    """When state.npz is valid numpy but missing 'step' key, load_state recovers."""
    validator = Validator(config=mock_config)

    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    # Save a valid npz but without the required 'step' key
    np.savez(state_file, scores=np.array([0.5]), hotkeys=np.array(["hk1"]))

    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)
    assert all(s == 0.0 for s in validator.scores)


@pytest.mark.asyncio
async def test_load_state_missing_scores_key(mock_config, tmp_path):
    """When state.npz is valid numpy but missing 'scores' key, load_state recovers."""
    validator = Validator(config=mock_config)

    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    # Save a valid npz but without 'scores'
    np.savez(state_file, step=10, hotkeys=np.array(["hk1"]))

    validator.load_state()

    assert validator.step == 0
    assert isinstance(validator.scores, np.ndarray)
    assert all(s == 0.0 for s in validator.scores)


@pytest.mark.asyncio
async def test_load_state_recovery_resets_histories(mock_config, tmp_path):
    """On corruption recovery, score_history and confidence_history are reset to empty dicts."""
    validator = Validator(config=mock_config)

    # Set non-empty histories
    validator.score_history = {0: [0.5, 0.6], 1: [0.3]}
    validator.confidence_history = {0: [([0.8], [1])]}

    # Write garbage to state file
    state_file = os.path.join(str(tmp_path), "state.npz")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(state_file, "wb") as f:
        f.write(b"corrupted data that is not a valid npz file")

    validator.load_state()

    assert validator.step == 0
    assert validator.score_history == {}
    assert validator.confidence_history == {}


@pytest.mark.asyncio
async def test_save_state_roundtrip_all_keys(mock_config, tmp_path):
    """After save_state, the resulting state.npz is loadable with all expected keys."""
    validator = Validator(config=mock_config)
    validator.step = 99
    validator.scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    validator.hotkeys = ["hk_a", "hk_b", "hk_c"]
    validator.score_history = {0: [0.1, 0.2], 1: [0.9]}
    validator.confidence_history = {0: [([0.8, 0.7], [1, 0])]}

    validator.save_state()

    # Load with a fresh validator
    validator2 = Validator(config=mock_config)
    validator2.load_state()

    assert validator2.step == 99
    np.testing.assert_array_almost_equal(
        validator2.scores, np.array([0.1, 0.9, 0.5], dtype=np.float32)
    )
    assert validator2.hotkeys == ["hk_a", "hk_b", "hk_c"]
    assert 0 in validator2.score_history
    assert validator2.score_history[0] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_save_state_cleanup_on_failure(mock_config, tmp_path, monkeypatch):
    """If np.savez fails, the temp file is cleaned up and no partial state.npz is created."""
    validator = Validator(config=mock_config)
    validator.step = 5

    # Make np.savez raise an error
    def failing_savez(*args, **kwargs):
        raise OSError("Simulated write failure")

    monkeypatch.setattr("numpy.savez", failing_savez)

    with pytest.raises(IOError, match="Simulated write failure"):
        validator.save_state()

    # No temp files should remain
    tmp_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp.npz")]
    assert len(tmp_files) == 0, "Temp file should be cleaned up on failure"
