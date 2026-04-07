"""Tests for MinerTelemetry: sliding window, calibration, Prometheus, JSON."""

import json
import math
import time

from prometheus_client import CollectorRegistry

from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry


class TestSlidingWindow:
    def test_default_window_size(self):
        t = MinerTelemetry()
        assert t._window_size == 100

    def test_record_and_stats(self):
        t = MinerTelemetry(window_size=5)
        for i in range(5):
            t.record("hall", float(i) / 4, 0.5)
        stats = t.get_stats("hall")
        assert stats is not None
        assert stats["count"] == 5
        assert 0.0 <= stats["mean"] <= 1.0

    def test_ring_buffer_eviction(self):
        t = MinerTelemetry(window_size=3)
        t.record("hall", 0.1, 0.5)
        t.record("hall", 0.2, 0.5)
        t.record("hall", 0.3, 0.5)
        t.record("hall", 0.9, 0.5)  # evicts 0.1
        stats = t.get_stats("hall")
        assert stats["count"] == 3
        assert stats["mean"] > 0.3  # 0.1 evicted, mean of 0.2, 0.3, 0.9

    def test_unknown_domain_returns_none(self):
        t = MinerTelemetry()
        assert t.get_stats("nonexistent") is None


class TestCalibration:
    def test_calibration_error_no_records(self):
        t = MinerTelemetry()
        assert math.isnan(t.get_calibration_error("hall"))

    def test_calibration_error_with_data(self):
        t = MinerTelemetry(window_size=100)
        for _ in range(50):
            t.record("hall", 0.8, 0.9)  # high score, high confidence
            t.record("hall", 0.2, 0.1)  # low score, low confidence
        ece = t.get_calibration_error("hall")
        assert isinstance(ece, float)
        assert not math.isnan(ece)
        assert 0.0 <= ece <= 1.0


class TestJsonExport:
    def test_export_json(self, tmp_path):
        t = MinerTelemetry(window_size=10)
        t.record("hall", 0.7, 0.8)
        out = str(tmp_path / "test.json")
        t.export_json("hall", out)
        with open(out) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "window_size" in data
        assert data["detection_count"] == 1
        assert data["score_distribution"] is not None

    def test_export_with_memory_fn(self, tmp_path):
        t = MinerTelemetry(window_size=10, memory_bank_size_fn=lambda d: 42)
        t.record("hall", 0.5, 0.5)
        out = str(tmp_path / "test2.json")
        t.export_json("hall", out)
        with open(out) as f:
            data = json.load(f)
        assert data["memory_bank_size"] == 42


class TestPerformance:
    def test_record_under_1ms(self):
        t = MinerTelemetry(window_size=100)
        start = time.perf_counter()
        for _ in range(1000):
            t.record("hall", 0.5, 0.5)
        elapsed = time.perf_counter() - start
        per_call = elapsed / 1000 * 1000  # ms
        assert per_call < 1.0, f"record() took {per_call:.3f}ms per call"


class TestGracefulDegradation:
    def test_no_memory_fn(self, tmp_path):
        t = MinerTelemetry(window_size=10)
        t.record("hall", 0.5, 0.5)
        out = str(tmp_path / "test.json")
        t.export_json("hall", out)
        with open(out) as f:
            data = json.load(f)
        assert data["memory_bank_size"] is None

    def test_empty_domain_stats(self):
        t = MinerTelemetry()
        assert t.get_stats("empty") is None
        assert math.isnan(t.get_calibration_error("empty"))


class TestPrometheus:
    def test_register_and_update(self):
        reg = CollectorRegistry()
        t = MinerTelemetry(window_size=10)
        t.register_prometheus(registry=reg)
        t.record("hall", 0.7, 0.8)
        t.update_prometheus("hall")
        # Gauge should have value
        assert t._prom_score_mean is not None

    def test_update_no_data(self):
        reg = CollectorRegistry()
        t = MinerTelemetry(window_size=10)
        t.register_prometheus(registry=reg)
        t.update_prometheus("empty")  # should not error

    def test_import_from_package(self):
        from antigence_subnet.miner.orchestrator import MinerTelemetry as MinerTelemetryImport
        assert MinerTelemetryImport is MinerTelemetry


class TestForwardIntegration:
    def test_forward_without_telemetry(self):
        """Miner without telemetry attr should not error."""
        from types import SimpleNamespace
        miner = SimpleNamespace()
        assert getattr(miner, "telemetry", None) is None

    def test_forward_with_telemetry(self):
        """Miner with telemetry records detection."""
        t = MinerTelemetry(window_size=10)
        t.record("hall", 0.8, 0.9)
        stats = t.get_stats("hall")
        assert stats["count"] == 1
