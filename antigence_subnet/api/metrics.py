"""
Prometheus metrics collector and /metrics endpoint (MON-03).

Provides:
- MetricsCollector: singleton that records forward pass latency,
  miner response times, and reward distribution.
- GET /metrics: returns Prometheus text format via generate_latest().
- metrics_router: FastAPI APIRouter for mounting.

Metric names are prefixed with "antigence_" to avoid collisions with
other Prometheus exporters.
"""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Singleton-style Prometheus metrics collector for the Antigence subnet.

    Accepts an optional CollectorRegistry for test isolation. When no
    registry is provided, uses the prometheus_client global REGISTRY.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self._registry = registry or REGISTRY

        self.forward_pass_latency = Histogram(
            "antigence_forward_pass_latency_seconds",
            "Duration of a complete validator forward pass in seconds",
            registry=self._registry,
        )

        self.miner_response_time = Histogram(
            "antigence_miner_response_seconds",
            "Duration of individual miner responses in seconds",
            labelnames=["uid"],
            registry=self._registry,
        )

        self.reward_distribution = Summary(
            "antigence_reward_distribution",
            "Distribution of rewards assigned to miners",
            labelnames=["uid"],
            registry=self._registry,
        )

        # Validator agreement gauge (VHARD-04)
        self.validator_agreement = Gauge(
            "antigence_validator_agreement",
            "Mean pairwise Spearman rank correlation between validators",
            registry=self._registry,
        )

    def record_forward_pass(self, duration_seconds: float) -> None:
        """Record the duration of a forward pass."""
        self.forward_pass_latency.observe(duration_seconds)

    def record_miner_response(self, uid: int, duration_seconds: float) -> None:
        """Record a miner's response time."""
        self.miner_response_time.labels(uid=str(uid)).observe(duration_seconds)

    def record_reward(self, uid: int, reward: float) -> None:
        """Record a reward value for a miner."""
        self.reward_distribution.labels(uid=str(uid)).observe(reward)

    def record_agreement(self, mean_correlation: float) -> None:
        """Record the mean validator agreement (Spearman rho)."""
        self.validator_agreement.set(mean_correlation)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Return the module-level MetricsCollector singleton.

    Creates it on first access. Uses the global prometheus_client REGISTRY
    so that GET /metrics picks up all recorded observations.
    """
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics_endpoint():
    """Return Prometheus metrics in text exposition format.

    Uses generate_latest() on the global REGISTRY so all MetricsCollector
    observations are included.
    """
    # Ensure collector is initialized (registers metrics on first access)
    get_collector()
    data = generate_latest(REGISTRY)
    return Response(
        content=data,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
