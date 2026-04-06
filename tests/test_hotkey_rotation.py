"""Tests for hotkey rotation detection in validator sync() (CORR-02).

Verifies that:
- When hotkeys change between syncs, scores for rotated UIDs are zeroed
- When hotkeys change, score_history entries for rotated UIDs are removed
- When hotkeys change, confidence_history entries for rotated UIDs are removed
- When metagraph size increases, scores array is resized with zero-padding
- When metagraph size decreases, scores array is truncated
- When no hotkeys change, scores remain unchanged
- Hotkey rotation is logged with old and new hotkey prefixes
"""

from unittest.mock import patch

import pytest

from antigence_subnet.base.validator import BaseValidatorNeuron


def _make_validator(mock_config):
    """Create a concrete validator subclass for testing."""

    class _TestValidator(BaseValidatorNeuron):
        async def forward(self):
            pass

    return _TestValidator(config=mock_config)


class TestHotkeyRotationDetection:
    """Test hotkey rotation detection in sync()."""

    def test_rotated_uid_scores_zeroed(self, mock_config):
        """When a hotkey changes at a UID, that UID's score is zeroed."""
        validator = _make_validator(mock_config)
        original_hotkeys = list(validator.metagraph.hotkeys)

        # Set non-zero score at UID 2
        validator.scores[2] = 0.8

        # Simulate hotkey rotation at UID 2
        new_hotkeys = list(original_hotkeys)
        new_hotkeys[2] = "rotated-new-hotkey-uid2"

        with patch.object(
            type(validator.metagraph), "hotkeys",
            new_callable=lambda: property(lambda self: new_hotkeys),
        ):
            validator.sync()

        assert validator.scores[2] == 0.0

    def test_rotated_uid_score_history_removed(self, mock_config):
        """When a hotkey changes, score_history for that UID is removed."""
        validator = _make_validator(mock_config)
        original_hotkeys = list(validator.metagraph.hotkeys)

        # Set up history for UID 2
        validator.score_history[2] = [0.5, 0.6, 0.7]

        new_hotkeys = list(original_hotkeys)
        new_hotkeys[2] = "rotated-new-hotkey-uid2"

        with patch.object(
            type(validator.metagraph), "hotkeys",
            new_callable=lambda: property(lambda self: new_hotkeys),
        ):
            validator.sync()

        assert 2 not in validator.score_history

    def test_rotated_uid_confidence_history_removed(self, mock_config):
        """When a hotkey changes, confidence_history for that UID is removed."""
        validator = _make_validator(mock_config)
        original_hotkeys = list(validator.metagraph.hotkeys)

        # Set up confidence history for UID 2
        validator.confidence_history[2] = [([0.9], [1])]

        new_hotkeys = list(original_hotkeys)
        new_hotkeys[2] = "rotated-new-hotkey-uid2"

        with patch.object(
            type(validator.metagraph), "hotkeys",
            new_callable=lambda: property(lambda self: new_hotkeys),
        ):
            validator.sync()

        assert 2 not in validator.confidence_history

    def test_metagraph_size_increase_zero_pads(self, mock_config):
        """When metagraph grows, scores array is resized with zero-padding."""
        validator = _make_validator(mock_config)
        original_n = validator.metagraph.n
        original_hotkeys = list(validator.metagraph.hotkeys)

        # Set some scores
        validator.scores[0] = 0.5
        validator.scores[1] = 0.7

        # Simulate metagraph growth: add 4 more UIDs
        new_n = original_n + 4
        new_hotkeys = list(original_hotkeys) + [f"new-miner-{i}" for i in range(4)]

        with (
            patch.object(
                type(validator.metagraph), "hotkeys",
                new_callable=lambda: property(lambda self: new_hotkeys),
            ),
            patch.object(
                type(validator.metagraph), "n",
                new_callable=lambda: property(lambda self: new_n),
            ),
        ):
            validator.sync()

        assert len(validator.scores) == new_n
        assert validator.scores[0] == pytest.approx(0.5)
        assert validator.scores[1] == pytest.approx(0.7)
        # New UIDs should be zero
        for uid in range(original_n, new_n):
            assert validator.scores[uid] == 0.0

    def test_metagraph_size_decrease_truncates(self, mock_config):
        """When metagraph shrinks, scores array is truncated."""
        validator = _make_validator(mock_config)
        original_n = validator.metagraph.n

        # Set some scores
        validator.scores[0] = 0.5
        validator.scores[1] = 0.7

        # Simulate metagraph shrinkage
        new_n = original_n - 4
        original_hotkeys = list(validator.metagraph.hotkeys)
        new_hotkeys = original_hotkeys[:new_n]

        with (
            patch.object(
                type(validator.metagraph), "hotkeys",
                new_callable=lambda: property(lambda self: new_hotkeys),
            ),
            patch.object(
                type(validator.metagraph), "n",
                new_callable=lambda: property(lambda self: new_n),
            ),
        ):
            validator.sync()

        assert len(validator.scores) == new_n
        assert validator.scores[0] == pytest.approx(0.5)
        assert validator.scores[1] == pytest.approx(0.7)

    def test_no_hotkey_change_scores_unchanged(self, mock_config):
        """When no hotkeys change, scores remain as they were."""
        validator = _make_validator(mock_config)

        # Set non-zero scores
        validator.scores[0] = 0.5
        validator.scores[1] = 0.7
        validator.scores[2] = 0.8

        # Sync without any hotkey changes
        validator.sync()

        assert validator.scores[0] == pytest.approx(0.5)
        assert validator.scores[1] == pytest.approx(0.7)
        assert validator.scores[2] == pytest.approx(0.8)

    def test_hotkey_rotation_logged(self, mock_config):
        """Hotkey rotation is logged with old and new hotkey prefixes."""
        validator = _make_validator(mock_config)
        original_hotkeys = list(validator.metagraph.hotkeys)

        new_hotkeys = list(original_hotkeys)
        new_hotkeys[3] = "brand-new-hotkey-at-uid3"

        with (
            patch.object(
                type(validator.metagraph), "hotkeys",
                new_callable=lambda: property(lambda self: new_hotkeys),
            ),
            patch("bittensor.logging.info") as mock_log,
        ):
            validator.sync()

            # Find the rotation log call
            rotation_logged = False
            for call in mock_log.call_args_list:
                msg = str(call)
                if "Hotkey rotation detected at UID 3" in msg:
                    rotation_logged = True
                    break

            assert rotation_logged, "Hotkey rotation was not logged"
