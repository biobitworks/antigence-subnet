"""Tests for configurable log level via --logging.level (PROD-06).

Verifies that BaseNeuron accepts --logging.level arg with choices
[debug, info, warning, error] and applies the selected level during
__init__.
"""

import argparse
from unittest.mock import patch

import bittensor as bt

from antigence_subnet.base.neuron import BaseNeuron


class TestLogLevelArg:
    """Verify --logging.level arg is registered in the parser."""

    def test_logging_level_arg_exists(self):
        """--logging.level is accepted by the parser."""
        parser = argparse.ArgumentParser()
        BaseNeuron._add_args_to_parser(parser)

        # Should parse without error
        args = parser.parse_args(["--logging.level", "debug"])
        assert getattr(args, "logging.level") == "debug"

    def test_logging_level_choices(self):
        """--logging.level only accepts debug/info/warning/error."""
        parser = argparse.ArgumentParser()
        BaseNeuron._add_args_to_parser(parser)

        # Valid choices should work
        for level in ("debug", "info", "warning", "error"):
            args = parser.parse_args(["--logging.level", level])
            assert getattr(args, "logging.level") == level

    def test_logging_level_default_is_info(self):
        """Default --logging.level is 'info'."""
        parser = argparse.ArgumentParser()
        BaseNeuron._add_args_to_parser(parser)

        args = parser.parse_args([])
        assert getattr(args, "logging.level") == "info"


class TestLogLevelApplication:
    """Verify that BaseNeuron.__init__ applies the log level setting."""

    def test_debug_level_calls_set_debug(self, mock_config):
        """When logging.level='debug', bt.logging.set_debug(True) is called."""
        # Set logging.level on the config
        if not hasattr(mock_config, "logging"):
            mock_config.logging = bt.Config()
        mock_config.logging.level = "debug"

        with patch.object(bt.logging, "set_debug") as mock_set_debug:
            BaseNeuron(config=mock_config)
            mock_set_debug.assert_called_once_with(True)

    def test_info_level_does_not_call_set_debug(self, mock_config):
        """When logging.level='info', neither set_debug nor set_trace is called."""
        if not hasattr(mock_config, "logging"):
            mock_config.logging = bt.Config()
        mock_config.logging.level = "info"

        with patch.object(bt.logging, "set_debug") as mock_set_debug, \
             patch.object(bt.logging, "set_trace") as mock_set_trace:
            BaseNeuron(config=mock_config)
            mock_set_debug.assert_not_called()
            mock_set_trace.assert_not_called()
