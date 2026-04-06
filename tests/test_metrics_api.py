"""Tests for the Prometheus metrics API endpoint (MON-03).

Covers:
- GET /metrics returns 200 with Prometheus text format
- Metric names: antigence_forward_pass_latency_seconds,
  antigence_miner_response_seconds, antigence_reward_distribution
- MetricsCollector records forward pass latency, miner response times, rewards
"""

import pytest

pytest.importorskip("prometheus_client", reason="prometheus_client required for metrics tests")

from fastapi import FastAPI  # noqa: E402
from prometheus_client import CollectorRegistry  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from antigence_subnet.api.metrics import (
    MetricsCollector,
    get_collector,
    metrics_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(registry: CollectorRegistry | None = None):
    """Mount metrics_router on a fresh FastAPI app."""
    app = FastAPI()
    app.include_router(metrics_router)
    return app


def _fresh_collector() -> MetricsCollector:
    """Create a MetricsCollector with an isolated registry to avoid conflicts."""
    registry = CollectorRegistry()
    return MetricsCollector(registry=registry)


# ---------------------------------------------------------------------------
# Test: GET /metrics returns 200 with Content-Type text/plain
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Test: GET /metrics contains forward_pass_latency metric
# ---------------------------------------------------------------------------

class TestMetricsContainsForwardPassLatency:
    def test_metrics_contains_forward_pass_latency(self):
        # Record a value so the metric appears
        collector = get_collector()
        collector.record_forward_pass(1.5)
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert "antigence_forward_pass_latency_seconds" in resp.text


# ---------------------------------------------------------------------------
# Test: GET /metrics contains reward_distribution metric
# ---------------------------------------------------------------------------

class TestMetricsContainsRewardDistribution:
    def test_metrics_contains_reward_distribution(self):
        collector = get_collector()
        collector.record_reward(0, 0.8)
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert "antigence_reward_distribution" in resp.text


# ---------------------------------------------------------------------------
# Test: GET /metrics contains miner_response_seconds metric
# ---------------------------------------------------------------------------

class TestMetricsContainsMinerResponse:
    def test_metrics_contains_miner_response(self):
        collector = get_collector()
        collector.record_miner_response(0, 0.5)
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert "antigence_miner_response_seconds" in resp.text


# ---------------------------------------------------------------------------
# Test: MetricsCollector.record_forward_pass updates histogram
# ---------------------------------------------------------------------------

class TestCollectorRecordForwardPass:
    def test_record_forward_pass_increments_histogram(self):
        collector = _fresh_collector()
        collector.record_forward_pass(2.0)
        collector.record_forward_pass(3.0)
        # Access the underlying metric data via collect()
        data = list(collector.forward_pass_latency.collect())
        assert len(data) > 0
        # Find _count sample
        count_found = False
        for metric_family in data:
            for sample in metric_family.samples:
                if sample.name.endswith("_count"):
                    assert sample.value == 2.0
                    count_found = True
        assert count_found, "Forward pass latency histogram count not found"


# ---------------------------------------------------------------------------
# Test: MetricsCollector.record_miner_response updates histogram
# ---------------------------------------------------------------------------

class TestCollectorRecordMinerResponse:
    def test_record_miner_response_increments_histogram(self):
        collector = _fresh_collector()
        collector.record_miner_response(1, 0.5)
        collector.record_miner_response(2, 0.3)
        data = list(collector.miner_response_time.collect())
        assert len(data) > 0
        # Check that samples exist (labeled histogram)
        total_count = 0
        for metric_family in data:
            for sample in metric_family.samples:
                if sample.name.endswith("_count"):
                    total_count += sample.value
        assert total_count == 2.0, f"Expected 2 miner response observations, got {total_count}"


# ---------------------------------------------------------------------------
# Test: MetricsCollector.record_reward updates summary
# ---------------------------------------------------------------------------

class TestCollectorRecordReward:
    def test_record_reward_increments_summary(self):
        collector = _fresh_collector()
        collector.record_reward(0, 0.8)
        collector.record_reward(0, 0.6)
        data = list(collector.reward_distribution.collect())
        assert len(data) > 0
        count_found = False
        for metric_family in data:
            for sample in metric_family.samples:
                if sample.name.endswith("_count"):
                    count_found = True
                    assert sample.value == 2.0
        assert count_found, "Reward distribution summary count not found"
