"""Miner-local detection telemetry with sliding window stats and calibration.

Tracks per-domain score distributions and confidence calibration over a
configurable sliding window. Exports Prometheus gauges and JSON snapshots.

Performance: < 1ms per record() call using numpy ring buffers.
Graceful degradation: works without B Cell memory or ground truth.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class _DomainTelemetry:
    """Per-domain ring buffer state."""

    scores: np.ndarray
    confidences: np.ndarray
    write_idx: int = 0
    count: int = 0
    window_size: int = 100


class MinerTelemetry:
    """Miner-local detection telemetry with sliding window and calibration.

    Tracks per-domain score distributions and confidence calibration error
    over a sliding window. Exports Prometheus gauges and JSON snapshots.

    Args:
        window_size: Number of detections to track per domain.
        memory_bank_size_fn: Optional callback returning B Cell memory size per domain.
    """

    def __init__(
        self,
        window_size: int = 100,
        memory_bank_size_fn: Callable[[str], int] | None = None,
    ) -> None:
        self._window_size = window_size
        self._memory_bank_size_fn = memory_bank_size_fn
        self._domains: dict[str, _DomainTelemetry] = {}
        # Prometheus metrics (set after register_prometheus())
        self._prom_accuracy = None
        self._prom_score_mean = None
        self._prom_score_std = None
        self._prom_memory_size = None
        self._prom_detections = None

    def _get_or_create(self, domain: str) -> _DomainTelemetry:
        if domain not in self._domains:
            self._domains[domain] = _DomainTelemetry(
                scores=np.zeros(self._window_size, dtype=np.float64),
                confidences=np.zeros(self._window_size, dtype=np.float64),
                window_size=self._window_size,
            )
        return self._domains[domain]

    def record(self, domain: str, score: float, confidence: float) -> None:
        """Record a detection result into the per-domain sliding window."""
        dt = self._get_or_create(domain)
        dt.scores[dt.write_idx] = score
        dt.confidences[dt.write_idx] = confidence
        dt.write_idx = (dt.write_idx + 1) % dt.window_size
        dt.count = min(dt.count + 1, dt.window_size)

    def get_stats(self, domain: str) -> dict[str, float] | None:
        """Get score distribution stats for a domain. None if untracked."""
        if domain not in self._domains:
            return None
        dt = self._domains[domain]
        if dt.count == 0:
            return None
        scores = dt.scores[:dt.count]
        return {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p95": round(float(np.percentile(scores, 95)), 4),
            "count": dt.count,
        }

    def get_calibration_error(self, domain: str) -> float:
        """Compute Expected Calibration Error (ECE) for a domain.

        Returns nan if no records exist.
        """
        if domain not in self._domains:
            return float("nan")
        dt = self._domains[domain]
        if dt.count == 0:
            return float("nan")

        scores = dt.scores[:dt.count]
        confs = dt.confidences[:dt.count]
        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = 0

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (
                (confs >= lo) & (confs < hi)
                if i < n_bins - 1
                else (confs >= lo) & (confs <= hi)
            )
            count = int(mask.sum())
            if count == 0:
                continue
            avg_conf = float(confs[mask].mean())
            frac_anomaly = float((scores[mask] > 0.5).mean())
            ece += count * abs(avg_conf - frac_anomaly)
            total += count

        return ece / total if total > 0 else float("nan")

    def export_json(self, domain: str, path: str | None = None) -> None:
        """Export telemetry snapshot as JSON.

        Args:
            domain: Domain to export.
            path: Output file path. Defaults to ~/.bittensor/neurons/telemetry/{domain}.json.
        """
        if path is None:
            path = str(
                Path.home() / ".bittensor" / "neurons" / "telemetry" / f"{domain}.json"
            )

        stats = self.get_stats(domain)
        ece = self.get_calibration_error(domain)
        accuracy = None if math.isnan(ece) else round(1.0 - ece, 4)

        mem_size = None
        if self._memory_bank_size_fn is not None:
            try:
                mem_size = self._memory_bank_size_fn(domain)
            except Exception:
                mem_size = None

        dt = self._domains.get(domain)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_size": self._window_size,
            "accuracy_estimate": accuracy,
            "score_distribution": stats,
            "detection_count": dt.count if dt else 0,
            "memory_bank_size": mem_size,
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(out_path))
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def register_prometheus(self, registry=None) -> None:
        """Register Prometheus gauges and counters.

        Handles duplicate registration gracefully (e.g., multiple Miner
        instances in tests sharing the global REGISTRY).

        Args:
            registry: Optional CollectorRegistry for test isolation.
        """
        try:
            import prometheus_client as prom

            reg = registry or prom.REGISTRY

            def _get_existing(name: str):
                """Retrieve an already-registered collector by metric name."""
                return reg._names_to_collectors.get(name)

            def _gauge(name: str, desc: str):
                try:
                    return prom.Gauge(name, desc, labelnames=["domain"], registry=reg)
                except ValueError:
                    existing = _get_existing(name)
                    if existing is not None:
                        return existing
                    raise

            def _counter(name: str, desc: str):
                try:
                    return prom.Counter(name, desc, labelnames=["domain"], registry=reg)
                except ValueError:
                    # Counter registers under "{name}_total" as well
                    existing = _get_existing(name + "_total")
                    if existing is not None:
                        return existing
                    existing = _get_existing(name)
                    if existing is not None:
                        return existing
                    raise

            self._prom_accuracy = _gauge(
                "antigence_miner_accuracy_estimate",
                "Estimated detection accuracy (1 - ECE)",
            )
            self._prom_score_mean = _gauge(
                "antigence_miner_score_mean",
                "Mean detection score over sliding window",
            )
            self._prom_score_std = _gauge(
                "antigence_miner_score_std",
                "Std deviation of detection scores",
            )
            self._prom_memory_size = _gauge(
                "antigence_miner_memory_bank_size",
                "B Cell memory bank size",
            )
            self._prom_detections = _counter(
                "antigence_miner_detections_total",
                "Total detections processed",
            )
        except ImportError:
            pass  # prometheus_client not installed -- graceful degradation

    def update_prometheus(self, domain: str) -> None:
        """Update Prometheus metrics for a domain."""
        if self._prom_accuracy is None:
            return

        stats = self.get_stats(domain)
        if stats is not None:
            self._prom_score_mean.labels(domain=domain).set(stats["mean"])
            self._prom_score_std.labels(domain=domain).set(stats["std"])

        ece = self.get_calibration_error(domain)
        if not math.isnan(ece):
            self._prom_accuracy.labels(domain=domain).set(1.0 - ece)
        else:
            self._prom_accuracy.labels(domain=domain).set(float("nan"))

        if self._memory_bank_size_fn is not None:
            try:
                size = self._memory_bank_size_fn(domain)
                self._prom_memory_size.labels(domain=domain).set(size)
            except Exception:
                pass

        self._prom_detections.labels(domain=domain).inc()
