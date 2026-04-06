"""
End-to-end cycle tests for the validator run loop.

Tests:
- Validator survives forward() errors (per-cycle exception handling)
- Full register -> query -> score -> set_weights cycle in mock mode
"""

from unittest.mock import MagicMock, patch

import bittensor as bt
import numpy as np
import pytest

from antigence_subnet.base.validator import BaseValidatorNeuron
from neurons.validator import Validator

# ---------------------------------------------------------------------------
# Task 1: Per-cycle exception handling tests
# ---------------------------------------------------------------------------


class SurvivalValidator(BaseValidatorNeuron):
    """Validator subclass that controls forward() behavior for testing."""

    def __init__(self, config=None, fail_on_calls=None):
        super().__init__(config=config)
        self.forward_call_count = 0
        self._fail_on_calls = fail_on_calls or set()

    async def forward(self):
        self.forward_call_count += 1
        if self.forward_call_count in self._fail_on_calls:
            raise RuntimeError(
                f"Simulated failure on call {self.forward_call_count}"
            )


def test_cycle_survives_error(mock_config):
    """Validator run loop continues after forward() raises on first call.

    Both cycles should run (step incremented to 2) despite the first
    forward() raising an exception.
    """
    validator = SurvivalValidator(
        config=mock_config, fail_on_calls={1}
    )
    # Override set_weights_interval to avoid triggering set_weights
    validator.config.neuron.set_weights_interval = 9999

    call_count = 0

    def sleep_side_effect(_duration):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            validator.should_exit = True

    with (
        patch("time.sleep", side_effect=sleep_side_effect),
        patch.object(validator, "sync"),
        patch.object(validator, "save_state"),
        patch.object(validator, "load_state"),
    ):
        validator.run()

    assert validator.step == 2, (
        f"Expected step=2 (both cycles ran), got step={validator.step}"
    )
    assert validator.forward_call_count == 2, (
        f"Expected 2 forward calls, got {validator.forward_call_count}"
    )


def test_cycle_logs_error(mock_config, caplog):
    """When forward() raises, the error is logged with 'Forward pass error'."""
    validator = SurvivalValidator(
        config=mock_config, fail_on_calls={1}
    )
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
        patch.object(bt.logging, "error") as mock_log_error,
    ):
        validator.run()

    # Check that the error was logged via bt.logging.error
    error_calls = [str(c) for c in mock_log_error.call_args_list]
    assert any("Forward pass error" in s for s in error_calls), (
        f"Expected 'Forward pass error' in logged errors, got: {error_calls}"
    )


# ---------------------------------------------------------------------------
# Task 2: E2E mock integration test -- full cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_cycle_mock(mock_config):
    """Full cycle: register -> query -> score -> set_weights.

    Proves:
    1. Validator creates with mock infrastructure (registered on metagraph)
    2. Forward pass queries miners via MockDendrite and updates scores
    3. set_weights is called with valid uid/weight args (non-empty, non-zero)
    """
    # 1. Create validator -- mock config wires MockSubtensor + MockMetagraph + MockDendrite
    validator = Validator(config=mock_config)

    # 2. Run one forward pass
    await validator.forward()

    # 3. Verify scores were updated (at least one non-zero score)
    assert np.any(validator.scores > 0), (
        f"Expected at least one non-zero score after forward, got all zeros: "
        f"{validator.scores}"
    )

    # 4. Wrap subtensor.set_weights to capture call args
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.message = ""

    with patch.object(
        validator.subtensor, "set_weights", return_value=mock_result
    ) as mock_set_weights:
        validator.set_weights()

        # 5. Verify set_weights was called
        mock_set_weights.assert_called_once()

        # 6. Extract call args and verify uids/weights are valid
        call_kwargs = mock_set_weights.call_args
        # set_weights is called with keyword args
        kw = call_kwargs.kwargs if call_kwargs.kwargs else {}
        if not kw:
            # Fallback: may be positional
            kw = dict(
                zip(
                    ["wallet", "netuid", "uids", "weights", "version_key",
                     "wait_for_inclusion", "wait_for_finalization"],
                    call_kwargs.args, strict=False,
                )
            )

        assert "uids" in kw, f"set_weights missing 'uids' arg: {kw.keys()}"
        assert "weights" in kw, f"set_weights missing 'weights' arg: {kw.keys()}"

        uids = kw["uids"]
        weights = kw["weights"]

        assert len(uids) > 0, "set_weights called with empty uids"
        assert len(weights) > 0, "set_weights called with empty weights"
        # Weights should be uint16 values (non-zero after conversion)
        assert any(w > 0 for w in weights), (
            f"Expected non-zero uint16 weights, got: {weights}"
        )


@pytest.mark.asyncio
async def test_full_cycle_scores_updated(mock_config):
    """After forward pass, validator.scores has non-zero entries for queried miners."""
    validator = Validator(config=mock_config)

    # Confirm scores start at zero
    assert np.all(validator.scores == 0), "Scores should start at zero"

    # Run forward pass
    await validator.forward()

    # At least one score should be non-zero
    non_zero_count = np.count_nonzero(validator.scores)
    assert non_zero_count > 0, (
        f"Expected non-zero scores after forward pass, got {non_zero_count} non-zero"
    )
