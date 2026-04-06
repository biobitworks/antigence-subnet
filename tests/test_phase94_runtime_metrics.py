"""Tests for Phase 94 runtime metrics and collector artifact contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.utils.runtime_metrics import (  # noqa: E402
    load_phase94_runtime_config,
)
from scripts.phase94_collect_metrics import collect_scrape_window  # noqa: E402


class TestPhase94RuntimeConfig:
    def test_validator_runtime_config_uses_phase94_env_contract(self, monkeypatch, tmp_path):
        runtime_dir = tmp_path / "validator"
        monkeypatch.setenv("PHASE94_VALIDATOR_METRICS_PORT", "9101")
        monkeypatch.setenv("PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR", str(runtime_dir))
        monkeypatch.setenv("PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS", "300")

        config = load_phase94_runtime_config("validator")

        assert config.metrics_port == 9101
        assert config.export_dir == runtime_dir
        assert config.export_interval_seconds == 300

    def test_miner_runtime_config_uses_phase94_env_contract(self, monkeypatch, tmp_path):
        telemetry_dir = tmp_path / "miner"
        monkeypatch.setenv("PHASE94_MINER_METRICS_PORT", "9100")
        monkeypatch.setenv("PHASE94_MINER_TELEMETRY_EXPORT_DIR", str(telemetry_dir))
        monkeypatch.setenv("PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS", "300")

        config = load_phase94_runtime_config("miner")

        assert config.metrics_port == 9100
        assert config.export_dir == telemetry_dir
        assert config.export_interval_seconds == 300


class TestPhase94Collector:
    @pytest.fixture
    def phase94_provenance(self) -> dict:
        return {
            "git_commit_sha": "abc1234",
            "config_sha256": "deadbeef" * 8,
            "netuid": 94,
            "subtensor_endpoint": "wss://test.finney.opentensor.ai:443",
            "execution_mode": "same-host-private",
            "policy_mode": "operator_multiband",
            "high_threshold": 0.5,
            "low_threshold": 0.493536,
            "min_confidence": 0.6,
            "start_time_utc": "2026-04-04T00:00:00+00:00",
            "end_time_utc": "2026-04-05T00:00:00+00:00",
        }

    def test_collect_scrape_window_writes_24h_summary_with_threshold_profile_and_provenance(
        self, tmp_path, phase94_provenance
    ):
        summary = collect_scrape_window(
            miner_url="http://127.0.0.1:7777/metrics",
            validator_url="http://127.0.0.1:8888/metrics",
            output_dir=tmp_path,
            config_file=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml",
            interval_seconds=60,
            duration_hours=24,
            provenance=phase94_provenance,
            fetch_metrics=lambda url: f"# HELP from {url}\nvalue 1\n",
            iterations=2,
        )

        scrape_log = tmp_path / "prometheus-scrapes.jsonl"
        run_summary = tmp_path / "stability-24h" / "run-summary.json"

        assert scrape_log.exists()
        assert run_summary.exists()
        assert summary["scrape_count"] == 4
        assert summary["miner_scrape_failures"] == 0
        assert summary["validator_scrape_failures"] == 0
        assert summary["duration_hours"] == 24
        assert summary["config_file"].endswith("artifacts/config/live.toml")
        assert summary["threshold_profile"] == "validation-24h"
        assert summary["stale_exporter_intervals"] == 0
        assert summary["stale_exporter_seconds_max"] < 600
        assert summary["process_restarts_total"] == 0
        assert summary["unexpected_exit_count"] == 0
        assert summary["chain_submission_failures"] == 0
        assert summary["anomaly_count"] == 0
        assert summary["max_memory_growth_pct"] <= 15
        assert summary["artifact_files"] == [
            "prometheus-scrapes.jsonl",
            "stability-24h/run-summary.json",
        ]
        for key, value in phase94_provenance.items():
            assert summary[key] == value

        lines = scrape_log.read_text().strip().splitlines()
        assert len(lines) == 4
        for line in lines:
            data = json.loads(line)
            assert data["endpoint"] in {"miner", "validator"}
            assert "metrics" in data["url"]
            assert "9100" not in data["url"]
            assert "9101" not in data["url"]

        persisted_summary = json.loads(run_summary.read_text())
        assert persisted_summary["scrape_count"] == 4
        assert persisted_summary["threshold_profile"] == "validation-24h"
        assert persisted_summary["artifact_files"] == [
            "prometheus-scrapes.jsonl",
            "stability-24h/run-summary.json",
        ]
        for key, value in phase94_provenance.items():
            assert persisted_summary[key] == value

    def test_collect_scrape_window_supports_soak_mode_and_soak_summary_name(
        self, tmp_path, phase94_provenance
    ):
        summary = collect_scrape_window(
            miner_url="http://127.0.0.1:7777/metrics",
            validator_url="http://127.0.0.1:8888/metrics",
            output_dir=tmp_path,
            config_file=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml",
            interval_seconds=60,
            duration_hours=24,
            collector_mode="soak-72h",
            provenance=phase94_provenance,
            fetch_metrics=lambda _url: "# HELP phase94_metric\na 1\n",
            iterations=1,
        )

        soak_summary = tmp_path / "soak-72h" / "soak-summary.json"
        assert soak_summary.exists()
        assert summary["threshold_profile"] == "soak-72h"
        assert json.loads(soak_summary.read_text())["threshold_profile"] == "soak-72h"

    def test_phase94_cli_contract_uses_env_port_names_not_literal_phase94_ports(self):
        script_path = Path("scripts/phase94_collect_metrics.py")
        script_text = script_path.read_text()

        assert "PHASE94_MINER_METRICS_PORT" in script_text
        assert "PHASE94_VALIDATOR_METRICS_PORT" in script_text
        assert "--collector-mode" in script_text
        assert "soak-summary.json" in script_text
        assert "threshold_profile" in script_text
        assert "http://127.0.0.1:9100/metrics" not in script_text
        assert "http://127.0.0.1:9101/metrics" not in script_text
