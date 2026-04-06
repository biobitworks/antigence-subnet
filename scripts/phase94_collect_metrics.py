#!/usr/bin/env python3
"""Collect Phase 94 Prometheus scrapes from the live miner and validator."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from antigence_subnet.utils.runtime_metrics import atomic_write_json

PHASE94_MINER_METRICS_PORT = "PHASE94_MINER_METRICS_PORT"
PHASE94_VALIDATOR_METRICS_PORT = "PHASE94_VALIDATOR_METRICS_PORT"
PHASE94_MINER_TELEMETRY_EXPORT_DIR = "PHASE94_MINER_TELEMETRY_EXPORT_DIR"
PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR = "PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR"
PROVENANCE_FIELDS = (
    "git_commit_sha",
    "config_sha256",
    "netuid",
    "subtensor_endpoint",
    "execution_mode",
    "policy_mode",
    "high_threshold",
    "low_threshold",
    "min_confidence",
    "start_time_utc",
    "end_time_utc",
)
THRESHOLD_PROFILES = {
    "validation-24h": {
        "summary_dir": "stability-24h",
        "summary_file": "run-summary.json",
        "max_memory_growth_pct": 15.0,
        "scrape_failure_budget": 3,
        "stale_exporter_seconds_max": 600.0,
    },
    "soak-72h": {
        "summary_dir": "soak-72h",
        "summary_file": "soak-summary.json",
        "max_memory_growth_pct": 25.0,
        "scrape_failure_budget": 9,
        "stale_exporter_seconds_max": 600.0,
    },
}


def _fetch_metrics(url: str) -> str:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - operator-provided local URLs
        return response.read().decode("utf-8")


def _select_threshold_profile(duration_hours: int, collector_mode: str | None) -> str:
    if collector_mode in THRESHOLD_PROFILES:
        return collector_mode
    if duration_hours >= 72:
        return "soak-72h"
    return "validation-24h"


def _resolve_metrics_url(explicit_url: str | None, env_var: str) -> str:
    if explicit_url:
        return explicit_url
    port = os.getenv(env_var)
    if not port:
        raise ValueError(
            f"Provide --miner-url/--validator-url or set {env_var} for the Phase 94 collector."
        )
    return f"http://127.0.0.1:{port}/metrics"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_export_payload(env_var: str) -> dict:
    export_dir = os.getenv(env_var)
    if not export_dir:
        return {}
    return _load_json(Path(export_dir) / "runtime.json")


def _default_provenance(config_file: str, end_time_utc: str) -> dict:
    start_time_utc = datetime.now(timezone.utc).isoformat()
    return {
        "git_commit_sha": os.getenv("PHASE94_GIT_COMMIT_SHA", "unknown"),
        "config_sha256": os.getenv("PHASE94_CONFIG_SHA256", "unknown"),
        "netuid": int(os.getenv("PHASE94_NETUID", "0")),
        "subtensor_endpoint": os.getenv("PHASE94_SUBTENSOR_ENDPOINT", "unknown"),
        "execution_mode": "same-host-private",
        "policy_mode": os.getenv("PHASE94_POLICY_MODE", "operator_multiband"),
        "high_threshold": float(os.getenv("PHASE94_HIGH_THRESHOLD", "0.5")),
        "low_threshold": float(os.getenv("PHASE94_LOW_THRESHOLD", "0.493536")),
        "min_confidence": float(os.getenv("PHASE94_MIN_CONFIDENCE", "0.6")),
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "config_file": config_file,
    }


def _load_candidate_provenance(candidate_manifest: str | None, fallback: dict) -> dict:
    if not candidate_manifest:
        return fallback
    payload = _load_json(Path(candidate_manifest))
    if not payload:
        return fallback
    merged = fallback.copy()
    for field in PROVENANCE_FIELDS:
        if field in payload:
            merged[field] = payload[field]
    return merged


def collect_scrape_window(
    *,
    miner_url: str,
    validator_url: str,
    output_dir: str | Path,
    config_file: str,
    interval_seconds: int,
    duration_hours: int,
    collector_mode: str | None = None,
    provenance: dict | None = None,
    fetch_metrics: Callable[[str], str] | None = None,
    iterations: int | None = None,
) -> dict:
    """Scrape both live metrics endpoints and persist phase-local artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scrape_log = output_path / "prometheus-scrapes.jsonl"
    fetch = fetch_metrics or _fetch_metrics
    threshold_profile = _select_threshold_profile(duration_hours, collector_mode)
    summary_relpath = Path(THRESHOLD_PROFILES[threshold_profile]["summary_dir"]) / (
        THRESHOLD_PROFILES[threshold_profile]["summary_file"]
    )
    summary_path = output_path / summary_relpath

    total_iterations = iterations
    if total_iterations is None:
        total_iterations = max(1, int((duration_hours * 3600) / interval_seconds))

    end_time_utc = datetime.now(timezone.utc).isoformat()
    provenance_core = provenance or _default_provenance(config_file, end_time_utc)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": config_file,
        "duration_hours": duration_hours,
        "interval_seconds": interval_seconds,
        "threshold_profile": threshold_profile,
        "scrape_count": 0,
        "miner_scrape_failures": 0,
        "validator_scrape_failures": 0,
        "stale_exporter_intervals": 0,
        "stale_exporter_seconds_max": 0.0,
        "process_restarts_total": 0,
        "unexpected_exit_count": 0,
        "chain_submission_failures": 0,
        "anomaly_count": 0,
        "max_memory_growth_pct": 0.0,
        "artifact_files": [
            "prometheus-scrapes.jsonl",
            summary_relpath.as_posix(),
        ],
    }
    for field in PROVENANCE_FIELDS:
        if field in provenance_core:
            summary[field] = provenance_core[field]
    summary.setdefault("start_time_utc", datetime.now(timezone.utc).isoformat())
    summary.setdefault("end_time_utc", end_time_utc)

    with scrape_log.open("a", encoding="utf-8") as handle:
        for iteration in range(total_iterations):
            for endpoint, url in (("miner", miner_url), ("validator", validator_url)):
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration": iteration,
                    "endpoint": endpoint,
                    "url": url,
                    "success": True,
                }
                try:
                    metrics_text = fetch(url)
                    record["sample_count"] = len(metrics_text.splitlines())
                except Exception as exc:  # noqa: BLE001 - collector must persist failures
                    record["success"] = False
                    record["error"] = str(exc)
                    if endpoint == "miner":
                        summary["miner_scrape_failures"] += 1
                    else:
                        summary["validator_scrape_failures"] += 1
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                summary["scrape_count"] += 1

            if iteration + 1 < total_iterations:
                time.sleep(interval_seconds)

    for env_var in (
        PHASE94_MINER_TELEMETRY_EXPORT_DIR,
        PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR,
    ):
        runtime_payload = _runtime_export_payload(env_var)
        if not runtime_payload:
            continue
        summary["process_restarts_total"] += int(runtime_payload.get("process_restarts_total", 0))
        summary["unexpected_exit_count"] += int(runtime_payload.get("unexpected_exit_count", 0))
        summary["chain_submission_failures"] += int(runtime_payload.get("chain_submission_failures", 0))
        summary["anomaly_count"] += int(runtime_payload.get("anomaly_count", 0))
        summary["max_memory_growth_pct"] = max(
            float(summary["max_memory_growth_pct"]),
            float(runtime_payload.get("max_memory_growth_pct", 0.0)),
        )
        last_export = runtime_payload.get("last_export_timestamp_utc")
        if last_export:
            export_dt = datetime.fromisoformat(last_export)
            end_dt = datetime.fromisoformat(summary["end_time_utc"])
            stale_seconds = max(0.0, (end_dt - export_dt).total_seconds())
            summary["stale_exporter_seconds_max"] = max(
                float(summary["stale_exporter_seconds_max"]),
                stale_seconds,
            )
            if stale_seconds >= THRESHOLD_PROFILES[threshold_profile]["stale_exporter_seconds_max"]:
                summary["stale_exporter_intervals"] += 1

    atomic_write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Phase 94 Prometheus metrics")
    parser.add_argument(
        "--miner-url",
        default=None,
        help=f"Miner Prometheus URL. Defaults to http://127.0.0.1:${{{PHASE94_MINER_METRICS_PORT}}}/metrics",
    )
    parser.add_argument(
        "--validator-url",
        default=None,
        help=f"Validator Prometheus URL. Defaults to http://127.0.0.1:${{{PHASE94_VALIDATOR_METRICS_PORT}}}/metrics",
    )
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--duration-hours", type=int, default=24)
    parser.add_argument(
        "--collector-mode",
        choices=sorted(THRESHOLD_PROFILES),
        default=None,
        help="Optional explicit threshold profile. Overrides duration-based selection.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument(
        "--candidate-manifest",
        default=None,
        help="Optional deployment-candidate manifest supplying immutable provenance fields.",
    )
    args = parser.parse_args()

    miner_url = _resolve_metrics_url(args.miner_url, PHASE94_MINER_METRICS_PORT)
    validator_url = _resolve_metrics_url(
        args.validator_url,
        PHASE94_VALIDATOR_METRICS_PORT,
    )
    end_time_utc = datetime.now(timezone.utc).isoformat()
    provenance = _load_candidate_provenance(
        args.candidate_manifest,
        _default_provenance(args.config_file, end_time_utc),
    )
    summary = collect_scrape_window(
        miner_url=miner_url,
        validator_url=validator_url,
        interval_seconds=args.interval_seconds,
        duration_hours=args.duration_hours,
        collector_mode=args.collector_mode,
        output_dir=args.output_dir,
        config_file=args.config_file,
        provenance=provenance,
    )
    threshold_profile = summary["threshold_profile"]
    scrape_budget = THRESHOLD_PROFILES[threshold_profile]["scrape_failure_budget"]
    stale_budget = THRESHOLD_PROFILES[threshold_profile]["stale_exporter_seconds_max"]
    return 0 if (
        summary["miner_scrape_failures"] <= scrape_budget
        and summary["validator_scrape_failures"] <= scrape_budget
        and summary["stale_exporter_intervals"] == 0
        and float(summary["stale_exporter_seconds_max"]) < stale_budget
        and summary["process_restarts_total"] == 0
        and summary["unexpected_exit_count"] == 0
        and summary["chain_submission_failures"] == 0
        and summary["anomaly_count"] == 0
        and float(summary["max_memory_growth_pct"])
        <= THRESHOLD_PROFILES[threshold_profile]["max_memory_growth_pct"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
