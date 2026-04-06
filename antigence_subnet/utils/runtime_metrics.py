"""Shared Phase 94 runtime metrics helpers for live miner and validator processes."""

from __future__ import annotations

import json
import os
import resource
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase94RuntimeConfig:
    """Runtime metrics/export settings for a live Phase 94 process."""

    role: str
    metrics_port: int | None
    export_dir: Path | None
    export_interval_seconds: int


def get_process_rss_bytes() -> int:
    """Return current process RSS in bytes using stdlib-only APIs."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if os.uname().sysname == "Darwin":
        return rss
    return rss * 1024


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so long-running evidence files are never partial."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        Path(tmp_path).replace(output_path)
    finally:
        with suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)


def load_phase94_runtime_config(role: str) -> Phase94RuntimeConfig:
    """Load Phase 94 runtime observability config from the environment."""
    import os

    if role == "miner":
        metrics_port_env = "PHASE94_MINER_METRICS_PORT"
        export_dir_env = "PHASE94_MINER_TELEMETRY_EXPORT_DIR"
    elif role == "validator":
        metrics_port_env = "PHASE94_VALIDATOR_METRICS_PORT"
        export_dir_env = "PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR"
    else:
        raise ValueError(f"Unsupported Phase 94 runtime role: {role}")

    metrics_port = os.getenv(metrics_port_env)
    export_dir = os.getenv(export_dir_env)
    export_interval = os.getenv("PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS", "300")

    return Phase94RuntimeConfig(
        role=role,
        metrics_port=int(metrics_port) if metrics_port else None,
        export_dir=Path(export_dir) if export_dir else None,
        export_interval_seconds=int(export_interval),
    )


def start_metrics_http_server(port: int | None):
    """Start the Prometheus HTTP exporter when a port is configured."""
    if port is None:
        return None

    from prometheus_client import start_http_server as prom_start_http_server

    return prom_start_http_server(port)


def bootstrap_phase94_prometheus_exporter(
    port: int | None,
    collector_factory: Callable[[], Any] | None = None,
):
    """Initialize shared collectors before starting the Prometheus exporter."""
    if collector_factory is not None:
        collector_factory()
    return start_metrics_http_server(port)


def build_runtime_snapshot(
    *,
    role: str,
    metrics_port: int | None,
    started_at_utc: str,
    baseline_rss_bytes: int,
    extra_fields: dict[str, Any] | None = None,
    anomaly_count: int = 0,
    process_restarts_total: int = 0,
    unexpected_exit_count: int = 0,
    chain_submission_failures: int = 0,
) -> dict[str, Any]:
    """Build a machine-readable Phase 94 runtime export payload."""
    now = datetime.now(timezone.utc).isoformat()
    current_rss_bytes = get_process_rss_bytes()
    baseline = max(baseline_rss_bytes, 1)
    payload: dict[str, Any] = {
        "role": role,
        "metrics_port": metrics_port,
        "started_at_utc": started_at_utc,
        "last_export_timestamp_utc": now,
        "rss_bytes": current_rss_bytes,
        "baseline_rss_bytes": baseline_rss_bytes,
        "max_memory_growth_pct": round(
            max(0.0, ((current_rss_bytes - baseline) / baseline) * 100.0),
            6,
        ),
        "process_restarts_total": int(process_restarts_total),
        "unexpected_exit_count": int(unexpected_exit_count),
        "chain_submission_failures": int(chain_submission_failures),
        "anomaly_count": int(anomaly_count),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def start_periodic_json_export(
    export_path: str | Path | None,
    interval_seconds: int,
    build_payload: Callable[[], dict[str, Any]],
) -> tuple[threading.Event | None, threading.Thread | None]:
    """Start a daemon thread that periodically writes JSON snapshots."""
    if export_path is None:
        return None, None

    output_path = Path(export_path)
    stop_event = threading.Event()

    def _export_loop() -> None:
        while not stop_event.is_set():
            payload = build_payload()
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            atomic_write_json(output_path, payload)
            stop_event.wait(interval_seconds)

    thread = threading.Thread(
        target=_export_loop,
        name=f"phase94-json-export-{output_path.stem}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread
