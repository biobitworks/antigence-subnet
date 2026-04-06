"""Structured JSON logging utility wrapping bt.logging.

Outputs machine-parseable JSON lines to stderr for log aggregation
tools (ELK, Loki, CloudWatch) while preserving Bittensor's built-in
logging infrastructure via delegation to bt.logging methods.

Usage:
    from antigence_subnet.utils.structured_logging import get_logger

    log = get_logger("forward")
    log.info("Miner query complete", step=42, uid=3, latency=0.12)

    # Output on stderr (one JSON line):
    # {"severity":"INFO","timestamp":"2026-03-29T12:00:00+00:00",
    #  "component":"forward","step":42,"uid":3,"latency":0.12,
    #  "message":"Miner query complete"}
"""

import json
import sys
from datetime import datetime, timezone

import bittensor as bt


class StructuredLogger:
    """JSON-line logger that wraps bt.logging for structured log output.

    Each log call writes a single JSON line to stderr and delegates
    the human-readable message to the corresponding bt.logging method.

    Args:
        component: Logical component name (e.g. "forward", "reward").
                   Defaults to "validator".
        step: Optional default step number included in every log entry.
    """

    def __init__(self, component: str = "validator", step: int | None = None):
        self.component = component
        self._default_step = step

    def _emit(
        self,
        severity: str,
        msg: str,
        *,
        step: int | None = None,
        uid: int | None = None,
        **extra,
    ) -> None:
        """Build JSON dict, write to stderr, delegate to bt.logging.

        Args:
            severity: Uppercase severity string (INFO, WARNING, ERROR, DEBUG).
            msg: Human-readable log message.
            step: Optional step number (overrides default).
            uid: Optional miner UID.
            **extra: Additional key-value pairs merged into the JSON output.
        """
        record: dict = {
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": self.component,
        }

        # Resolve step: explicit > default > omit
        resolved_step = step if step is not None else self._default_step
        if resolved_step is not None:
            record["step"] = resolved_step

        if uid is not None:
            record["uid"] = uid

        # Merge extra fields before message (message last for readability)
        record.update(extra)
        record["message"] = msg

        # Serialize with default=str for safety (handles numpy types, etc.)
        json_line = json.dumps(record, default=str)
        print(json_line, file=sys.stderr)

        # Delegate to bt.logging
        bt_method = getattr(bt.logging, severity.lower(), bt.logging.info)
        bt_method(msg)

    def info(self, msg: str, *, step: int | None = None, uid: int | None = None, **extra) -> None:
        """Log at INFO level."""
        self._emit("INFO", msg, step=step, uid=uid, **extra)

    def warning(
        self, msg: str, *, step: int | None = None, uid: int | None = None, **extra,
    ) -> None:
        """Log at WARNING level."""
        self._emit("WARNING", msg, step=step, uid=uid, **extra)

    def error(self, msg: str, *, step: int | None = None, uid: int | None = None, **extra) -> None:
        """Log at ERROR level."""
        self._emit("ERROR", msg, step=step, uid=uid, **extra)

    def debug(self, msg: str, *, step: int | None = None, uid: int | None = None, **extra) -> None:
        """Log at DEBUG level."""
        self._emit("DEBUG", msg, step=step, uid=uid, **extra)


def get_logger(component: str) -> StructuredLogger:
    """Factory function returning a StructuredLogger for a named component.

    Args:
        component: Logical component name (e.g. "forward", "reward").

    Returns:
        Configured StructuredLogger instance.
    """
    return StructuredLogger(component=component)
