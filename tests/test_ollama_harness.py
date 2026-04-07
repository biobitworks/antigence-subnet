"""
Tests for the Ollama test harness — single-round simulation pipeline.

Defines the contract for scripts/ollama_test_harness.py:
- check_ollama_available(): Pre-flight check for Ollama server and model
- run_single_round(): Async single-round eval pipeline producing metrics dict

Mocked tests run without Ollama. Live tests require @pytest.mark.ollama.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_chat():
    """Patch ollama.chat to return a fake ChatResponse-like object."""
    fake_response = SimpleNamespace(
        message=SimpleNamespace(content="Q: What year did Rome fall?\nA: Rome fell in 1066."),
        total_duration=500_000_000,  # 500ms in nanoseconds
        eval_duration=400_000_000,  # 400ms
        prompt_eval_duration=80_000_000,  # 80ms
        load_duration=20_000_000,  # 20ms
        eval_count=32,
    )
    with patch("ollama.chat", return_value=fake_response) as mock_chat:
        yield mock_chat


@pytest.fixture
def mock_ollama_list():
    """Patch ollama.list to return a model list including qwen2.5:1.5b."""
    fake_model = SimpleNamespace(model="qwen2.5:1.5b")
    fake_response = SimpleNamespace(models=[fake_model])
    with patch("ollama.list", return_value=fake_response) as mock_list:
        yield mock_list


@pytest.fixture
def mock_ollama_list_empty():
    """Patch ollama.list to return an empty model list."""
    fake_response = SimpleNamespace(models=[])
    with patch("ollama.list", return_value=fake_response) as mock_list:
        yield mock_list


@pytest.fixture
def mock_ollama_list_connection_error():
    """Patch ollama.list to raise ConnectionError (server down)."""
    with patch("ollama.list", side_effect=ConnectionError("Connection refused")) as mock_list:
        yield mock_list


# ---------------------------------------------------------------------------
# Pre-flight check tests
# ---------------------------------------------------------------------------


class TestPreflightChecks:
    """Tests for check_ollama_available() pre-flight function."""

    def test_preflight_ollama_unavailable(self, mock_ollama_list_connection_error):
        """When Ollama server is down, check_ollama_available returns False."""
        from scripts.ollama_test_harness import check_ollama_available

        result = check_ollama_available("qwen2.5:1.5b")
        assert result is False

    def test_preflight_model_missing(self, mock_ollama_list_empty, capsys):
        """When model is not pulled, returns False with 'ollama pull' hint."""
        from scripts.ollama_test_harness import check_ollama_available

        result = check_ollama_available("qwen2.5:1.5b")
        assert result is False
        captured = capsys.readouterr()
        assert "ollama pull" in captured.out.lower() or "ollama pull" in captured.err.lower()

    def test_preflight_model_available(self, mock_ollama_list):
        """When model is present in Ollama, returns True."""
        from scripts.ollama_test_harness import check_ollama_available

        result = check_ollama_available("qwen2.5:1.5b")
        assert result is True


# ---------------------------------------------------------------------------
# Single-round pipeline tests
# ---------------------------------------------------------------------------


class TestSingleRound:
    """Tests for run_single_round() async pipeline."""

    def test_single_round_produces_metrics(self, mock_ollama_chat):
        """run_single_round returns dict with metrics, latency, samples_evaluated."""
        from scripts.ollama_test_harness import run_single_round

        result = asyncio.run(
            run_single_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                samples_per_round=10,
                warmup=False,
                seed=42,
            )
        )

        # Top-level keys
        assert "metrics" in result
        assert "latency" in result
        assert "samples_evaluated" in result
        assert result["samples_evaluated"] > 0

        # Metrics sub-keys
        metrics = result["metrics"]
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "accuracy" in metrics

        # Latency sub-keys
        latency = result["latency"]
        assert "ollama_generate_ms" in latency
        assert "detection_avg_ms" in latency
        assert "round_total_ms" in latency

    def test_single_round_metrics_valid_range(self, mock_ollama_chat):
        """All metric values in [0.0, 1.0], all latency values >= 0.0."""
        from scripts.ollama_test_harness import run_single_round

        result = asyncio.run(
            run_single_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                samples_per_round=10,
                warmup=False,
                seed=42,
            )
        )

        for key, val in result["metrics"].items():
            assert 0.0 <= val <= 1.0, f"metrics.{key}={val} out of [0,1]"

        for key, val in result["latency"].items():
            assert val >= 0.0, f"latency.{key}={val} is negative"

    def test_json_output_schema(self, mock_ollama_chat):
        """Result dict has all required top-level keys for JSON output."""
        from scripts.ollama_test_harness import run_single_round

        result = asyncio.run(
            run_single_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                samples_per_round=10,
                warmup=False,
                seed=42,
            )
        )

        required_keys = {"round", "domain", "metrics", "latency", "samples_evaluated"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_ollama_warmup_called(self, mock_ollama_chat):
        """When warmup=True, ollama.chat is called at least once extra (warmup)."""
        from scripts.ollama_test_harness import run_single_round

        _result = asyncio.run(
            run_single_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                samples_per_round=10,
                warmup=True,
                seed=42,
            )
        )

        # warmup call + generate call = at least 2 calls
        assert mock_ollama_chat.call_count >= 2, (
            f"Expected at least 2 ollama.chat calls (warmup + generate), got {mock_ollama_chat.call_count}"  # noqa: E501
        )


# ---------------------------------------------------------------------------
# Live test (requires running Ollama)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
class TestLiveSingleRound:
    """Live integration test -- requires Ollama running with qwen2.5:1.5b."""

    def test_live_single_round(self):
        """Actual Ollama call + detector + scoring produces valid metrics dict."""
        from scripts.ollama_test_harness import run_single_round

        result = asyncio.run(
            run_single_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                samples_per_round=5,
                warmup=True,
                seed=42,
            )
        )

        assert result["samples_evaluated"] > 0
        assert "metrics" in result
        assert "latency" in result
        for key, val in result["metrics"].items():
            assert 0.0 <= val <= 1.0, f"metrics.{key}={val} out of [0,1]"
        for key, val in result["latency"].items():
            assert val >= 0.0, f"latency.{key}={val} is negative"


# ---------------------------------------------------------------------------
# Plan 02: Multi-round runner, JSON reporting, summary statistics
# ---------------------------------------------------------------------------


class TestMultiRound:
    """Tests for run_multi_round() multi-round runner."""

    def test_multi_round_returns_all_rounds(self, mock_ollama_chat):
        """run_multi_round returns dict with 'rounds' list of correct length."""
        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=3,
                samples_per_round=10,
                warmup=True,
                seed=42,
            )
        )

        assert "rounds" in result
        assert len(result["rounds"]) == 3
        # Each round should have metrics and latency
        for rnd in result["rounds"]:
            assert "metrics" in rnd
            assert "latency" in rnd

    def test_multi_round_summary_statistics(self, mock_ollama_chat):
        """Summary dict contains all required aggregate statistics."""
        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=3,
                samples_per_round=10,
                seed=42,
            )
        )

        summary = result["summary"]

        # Float aggregates
        for key in [
            "avg_f1",
            "avg_precision",
            "avg_recall",
            "std_f1",
            "avg_ollama_latency_ms",
            "avg_detection_latency_ms",
        ]:
            assert key in summary, f"Missing summary key: {key}"
            assert isinstance(summary[key], float), f"summary.{key} should be float"

        # Integer counts
        for key in ["total_rounds", "total_samples"]:
            assert key in summary, f"Missing summary key: {key}"
            assert isinstance(summary[key], int), f"summary.{key} should be int"

        # Total time
        assert "total_time_s" in summary
        assert isinstance(summary["total_time_s"], float)

    def test_multi_round_summary_values_correct(self, mock_ollama_chat):
        """For 3 rounds with known metrics, avg/std values are correct."""
        import numpy as np

        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=3,
                samples_per_round=10,
                seed=42,
            )
        )

        f1_values = [r["metrics"]["f1"] for r in result["rounds"]]
        expected_avg = float(np.mean(f1_values))
        expected_std = float(np.std(f1_values))

        assert abs(result["summary"]["avg_f1"] - expected_avg) < 1e-6
        assert abs(result["summary"]["std_f1"] - expected_std) < 1e-6

    def test_multi_round_increments_seed(self, mock_ollama_chat):
        """Each round gets seed + round_index to vary sample selection."""
        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=3,
                samples_per_round=10,
                seed=100,
            )
        )

        seeds = [r["seed"] for r in result["rounds"]]
        assert seeds == [100, 101, 102], f"Expected [100, 101, 102], got {seeds}"

    def test_model_config_passed_through(self, mock_ollama_chat):
        """run_multi_round passes model name into config and each round."""
        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5-coder:7b",
                rounds=2,
                samples_per_round=10,
                seed=42,
            )
        )

        assert result["config"]["model"] == "qwen2.5-coder:7b"
        for rnd in result["rounds"]:
            assert rnd["model"] == "qwen2.5-coder:7b"


class TestJsonReport:
    """Tests for JSON report writing (write_report) and full schema."""

    def test_json_report_full_schema(self, mock_ollama_chat):
        """Full report has harness_version, timestamp, config, rounds, summary."""
        from scripts.ollama_test_harness import run_multi_round

        result = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=2,
                samples_per_round=10,
                seed=42,
            )
        )

        assert result["harness_version"] == "1.0"
        assert "timestamp" in result
        # Timestamp should be ISO-8601 format
        assert "T" in result["timestamp"]
        assert result["config"]["model"] == "qwen2.5:1.5b"
        assert result["config"]["rounds"] == 2
        assert result["config"]["domain"] == "hallucination"
        assert result["config"]["detector"] == "IsolationForest"
        assert result["config"]["samples_per_round"] == 10
        assert isinstance(result["rounds"], list)
        assert isinstance(result["summary"], dict)

    def test_json_report_writes_to_file(self, mock_ollama_chat, tmp_path):
        """write_report writes valid JSON with all required keys."""
        import json as json_mod

        from scripts.ollama_test_harness import run_multi_round, write_report

        report = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=2,
                samples_per_round=10,
                seed=42,
            )
        )

        output_path = str(tmp_path / "test_report.json")
        returned_path = write_report(report, output_path)

        assert returned_path == output_path
        with open(output_path) as f:
            loaded = json_mod.load(f)
        assert "harness_version" in loaded
        assert "rounds" in loaded
        assert "summary" in loaded
        assert "config" in loaded

    def test_json_report_auto_filename(self, mock_ollama_chat, tmp_path, monkeypatch):
        """When output_path is None, auto-generates timestamped filename."""
        from scripts.ollama_test_harness import run_multi_round, write_report

        report = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=2,
                samples_per_round=10,
                seed=42,
            )
        )

        # Monkeypatch the default output directory to tmp_path
        import scripts.ollama_test_harness as harness_mod

        original_benchmarks_dir = getattr(harness_mod, "BENCHMARKS_DIR", None)
        monkeypatch.setattr(harness_mod, "BENCHMARKS_DIR", tmp_path)

        returned_path = write_report(report, None)

        assert returned_path is not None
        assert str(tmp_path) in returned_path
        assert "ollama_harness_" in returned_path
        assert returned_path.endswith(".json")

        # Verify file is valid JSON
        import json as json_mod

        with open(returned_path) as f:
            loaded = json_mod.load(f)
        assert "harness_version" in loaded

        # Restore if needed
        if original_benchmarks_dir is not None:
            monkeypatch.setattr(harness_mod, "BENCHMARKS_DIR", original_benchmarks_dir)


class TestHumanReadableSummary:
    """Tests for print_summary() human-readable output."""

    def test_human_readable_summary(self, mock_ollama_chat, capsys):
        """print_summary produces stdout with formatted metrics."""
        from scripts.ollama_test_harness import print_summary, run_multi_round

        report = asyncio.run(
            run_multi_round(
                domain="hallucination",
                detector_name="IsolationForest",
                model="qwen2.5:1.5b",
                rounds=2,
                samples_per_round=10,
                seed=42,
            )
        )

        print_summary(report)

        captured = capsys.readouterr()
        # Should contain key metric labels
        assert "F1" in captured.out or "f1" in captured.out.lower()
        assert "Precision" in captured.out or "precision" in captured.out.lower()
        assert "Recall" in captured.out or "recall" in captured.out.lower()
        # Should contain formatted numbers (3 decimal places)
        assert "0." in captured.out  # At least one decimal value
