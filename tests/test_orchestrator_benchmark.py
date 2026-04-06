"""Smoke tests for orchestrator benchmark functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_orchestrator import compute_metrics, compute_kl_divergence, load_eval_data


class TestComputeMetrics:
    def test_perfect_scores(self):
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = ["anomalous", "anomalous", "normal", "normal"]
        m = compute_metrics(scores, labels)
        assert m["f1"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_all_false_negatives(self):
        scores = [0.1, 0.2]
        labels = ["anomalous", "anomalous"]
        m = compute_metrics(scores, labels)
        assert m["f1"] == 0.0
        assert m["recall"] == 0.0

    def test_mean_scores(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = ["normal", "normal", "anomalous", "anomalous"]
        m = compute_metrics(scores, labels)
        assert m["mean_normal"] == 0.15
        assert m["mean_anomalous"] == 0.85


class TestKLDivergence:
    def test_identical_distributions(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
        kl = compute_kl_divergence(scores, scores)
        assert kl < 0.01

    def test_different_distributions(self):
        a = [0.1, 0.2, 0.15, 0.25, 0.1] * 10
        b = [0.8, 0.9, 0.85, 0.95, 0.7] * 10
        kl = compute_kl_divergence(a, b)
        assert kl > 0.0


class TestLoadEvalData:
    def test_hallucination_data_exists(self):
        samples, manifest = load_eval_data("hallucination")
        assert len(samples) > 0
        assert len(manifest) > 0
        assert "prompt" in samples[0] or "output" in samples[0]
