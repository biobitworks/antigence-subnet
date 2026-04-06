"""Tests for structured JSON logging utility."""

import json
from datetime import datetime
from unittest.mock import patch

from antigence_subnet.utils.structured_logging import StructuredLogger, get_logger


class TestStructuredLoggerSeverityLevels:
    """Test that each severity level outputs correct JSON with severity field."""

    def test_info_outputs_valid_json_with_severity_timestamp_message(self, capsys):
        """StructuredLogger.info() outputs valid JSON with keys: severity, timestamp, message."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("test message")

        captured = capsys.readouterr()
        line = captured.err.strip()
        data = json.loads(line)
        assert data["severity"] == "INFO"
        assert "timestamp" in data
        assert data["message"] == "test message"

    def test_warning_sets_severity_to_warning(self, capsys):
        """StructuredLogger.warning() sets severity to 'WARNING'."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.warning("warn message")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["severity"] == "WARNING"
        assert data["message"] == "warn message"

    def test_error_sets_severity_to_error(self, capsys):
        """StructuredLogger.error() sets severity to 'ERROR'."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.error("error message")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["severity"] == "ERROR"
        assert data["message"] == "error message"

    def test_debug_sets_severity_to_debug(self, capsys):
        """StructuredLogger.debug() sets severity to 'DEBUG'."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.debug("debug message")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["severity"] == "DEBUG"
        assert data["message"] == "debug message"


class TestStructuredLoggerOptionalFields:
    """Test optional step, uid, and extra fields."""

    def test_step_included_in_json_output(self, capsys):
        """StructuredLogger with step=5 includes 'step': 5 in JSON output."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("step test", step=5)

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["step"] == 5

    def test_uid_included_in_json_output(self, capsys):
        """StructuredLogger with uid=3 includes 'uid': 3 in JSON output."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("uid test", uid=3)

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["uid"] == 3

    def test_extra_fields_included_in_json_output(self, capsys):
        """StructuredLogger with extra={'key': 'val'} includes extra fields."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("extra test", key="val")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["key"] == "val"

    def test_step_omitted_when_none(self, capsys):
        """Step field should be omitted from JSON when not provided."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("no step")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert "step" not in data

    def test_uid_omitted_when_none(self, capsys):
        """UID field should be omitted from JSON when not provided."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("no uid")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert "uid" not in data


class TestStructuredLoggerDelegation:
    """Test that StructuredLogger delegates to bt.logging methods."""

    def test_info_delegates_to_bt_logging_info(self):
        """StructuredLogger.info() delegates to bt.logging.info."""
        with patch("antigence_subnet.utils.structured_logging.bt") as mock_bt:
            logger = StructuredLogger()
            logger.info("delegate test")
            mock_bt.logging.info.assert_called_once_with("delegate test")

    def test_warning_delegates_to_bt_logging_warning(self):
        """StructuredLogger.warning() delegates to bt.logging.warning."""
        with patch("antigence_subnet.utils.structured_logging.bt") as mock_bt:
            logger = StructuredLogger()
            logger.warning("warn delegate")
            mock_bt.logging.warning.assert_called_once_with("warn delegate")

    def test_error_delegates_to_bt_logging_error(self):
        """StructuredLogger.error() delegates to bt.logging.error."""
        with patch("antigence_subnet.utils.structured_logging.bt") as mock_bt:
            logger = StructuredLogger()
            logger.error("error delegate")
            mock_bt.logging.error.assert_called_once_with("error delegate")

    def test_debug_delegates_to_bt_logging_debug(self):
        """StructuredLogger.debug() delegates to bt.logging.debug."""
        with patch("antigence_subnet.utils.structured_logging.bt") as mock_bt:
            logger = StructuredLogger()
            logger.debug("debug delegate")
            mock_bt.logging.debug.assert_called_once_with("debug delegate")


class TestGetLoggerFactory:
    """Test get_logger() factory function."""

    def test_get_logger_returns_structured_logger(self):
        """get_logger('component') returns StructuredLogger with component field set."""
        with patch("antigence_subnet.utils.structured_logging.bt"):
            logger = get_logger("my_component")
            assert isinstance(logger, StructuredLogger)

    def test_get_logger_sets_component_in_output(self, capsys):
        """JSON output includes 'component' field when logger created with component name."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = get_logger("forward")
            logger.info("component test")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["component"] == "forward"


class TestTimestampFormat:
    """Test that timestamp is ISO 8601 format."""

    def test_timestamp_is_iso_8601_format(self, capsys):
        """Timestamp field is a valid ISO 8601 format string."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("timestamp test")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        ts = data["timestamp"]
        # Should parse without error -- ISO 8601 format
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None
        # Should contain timezone info (UTC)
        assert parsed.tzinfo is not None or "Z" in ts or "+" in ts

    def test_default_component_is_validator(self, capsys):
        """Default component name is 'validator' when not specified."""
        with patch("antigence_subnet.utils.structured_logging.bt") as _mock_bt:
            logger = StructuredLogger()
            logger.info("default component")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert data["component"] == "validator"
