"""Tests for SIGTERM/SIGINT graceful shutdown (PROD-01).

Verifies that:
- should_exit flag is initialized to False
- _handle_shutdown_signal sets should_exit to True
- SIGTERM and SIGINT handlers are registered
- Signal registration is graceful in non-main threads (ValueError)
- --neuron.shutdown_timeout defaults to 30
"""

import signal
from unittest.mock import MagicMock, patch

from antigence_subnet.base.neuron import BaseNeuron


class TestShouldExitFlag:
    """Test should_exit flag initialization and behavior."""

    def test_should_exit_initialized_false(self, mock_config):
        """BaseNeuron.should_exit is False after initialization."""
        neuron = BaseNeuron(config=mock_config)
        assert neuron.should_exit is False

    def test_handle_signal_sets_should_exit(self, mock_config):
        """_handle_shutdown_signal sets should_exit to True."""
        neuron = BaseNeuron(config=mock_config)
        assert neuron.should_exit is False
        neuron._handle_shutdown_signal(signal.SIGTERM, None)
        assert neuron.should_exit is True

    def test_handle_signal_sets_should_exit_sigint(self, mock_config):
        """_handle_shutdown_signal works for SIGINT as well."""
        neuron = BaseNeuron(config=mock_config)
        neuron._handle_shutdown_signal(signal.SIGINT, None)
        assert neuron.should_exit is True


class TestSignalRegistration:
    """Test that signal handlers are registered correctly."""

    def test_signal_handlers_registered(self, mock_config):
        """SIGTERM and SIGINT handlers are registered to _handle_shutdown_signal."""
        with patch("antigence_subnet.base.neuron.signal_module") as mock_signal:
            # Provide the signal constants for the handler to use
            mock_signal.SIGTERM = signal.SIGTERM
            mock_signal.SIGINT = signal.SIGINT
            mock_signal.Signals = signal.Signals
            mock_signal.signal = MagicMock()

            neuron = BaseNeuron(config=mock_config)

            # Verify both signals were registered
            calls = mock_signal.signal.call_args_list
            sigterm_calls = [c for c in calls if c[0][0] == signal.SIGTERM]
            sigint_calls = [c for c in calls if c[0][0] == signal.SIGINT]

            assert len(sigterm_calls) == 1, "SIGTERM handler not registered"
            assert len(sigint_calls) == 1, "SIGINT handler not registered"

            # Verify the handler is the neuron's method
            assert sigterm_calls[0][0][1] == neuron._handle_shutdown_signal
            assert sigint_calls[0][0][1] == neuron._handle_shutdown_signal

    def test_signal_registration_graceful_in_non_main_thread(self, mock_config):
        """Signal registration failure (ValueError) does not raise."""
        with patch("antigence_subnet.base.neuron.signal_module") as mock_signal:
            mock_signal.SIGTERM = signal.SIGTERM
            mock_signal.SIGINT = signal.SIGINT
            mock_signal.Signals = signal.Signals
            # Raise ValueError to simulate non-main thread
            mock_signal.signal = MagicMock(side_effect=ValueError("not main thread"))

            # Should not raise
            neuron = BaseNeuron(config=mock_config)
            assert neuron.should_exit is False


class TestShutdownTimeout:
    """Test shutdown timeout argument."""

    def test_shutdown_timeout_default(self, mock_config):
        """--neuron.shutdown_timeout defaults to 30."""
        neuron = BaseNeuron(config=mock_config)
        assert neuron._shutdown_timeout == 30

    def test_shutdown_timeout_from_config(self, mock_config):
        """Shutdown timeout uses config value when set."""
        mock_config.neuron.shutdown_timeout = 60
        neuron = BaseNeuron(config=mock_config)
        assert neuron._shutdown_timeout == 60
