"""Contract tests for the thin Phase 84 swarm spike.

These tests intentionally encode D-03 and D-05:
- exactly three existing detectors
- mean3 and median3 only
- no SwarmAgent / SwarmPool / peer messaging / memory / weighting framework
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "phase84_swarm_spike.py"
)
FORBIDDEN_TOKENS = (
    "SwarmAgent",
    "SwarmPool",
    "peer",
    "memory",
    "weighting",
    "weighted_vote",
    "bayesian",
    "orchestrator_mode",
)


def load_module():
    """Load the phase84 spike script directly from disk."""
    spec = importlib.util.spec_from_file_location("phase84_swarm_spike", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not create import spec for {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDetector:
    """Tiny fake detector that records fit inputs."""

    def __init__(self, *args, **kwargs):
        self.fit_samples = None
        self.domain = None

    def fit(self, samples):
        self.fit_samples = samples


def test_create_phase84_detectors_is_exactly_three_existing_detectors(monkeypatch):
    """The spike hard-codes IF + OCSVM + NegSel and rejects framework drift."""
    module = load_module()

    monkeypatch.setattr(module, "IsolationForestDetector", _FakeDetector)
    monkeypatch.setattr(module, "OCSVMDetector", _FakeDetector)
    monkeypatch.setattr(module, "NegSelAISDetector", _FakeDetector)

    detectors = module.create_phase84_detectors(
        domain="hallucination",
        training_samples=[{"prompt": "p", "output": "o"}],
    )

    assert len(detectors) == 3
    assert [det.__class__.__name__ for det in detectors] == [
        "_FakeDetector",
        "_FakeDetector",
        "_FakeDetector",
    ]
    assert all(det.fit_samples == [{"prompt": "p", "output": "o"}] for det in detectors)
    assert all(det.domain == "hallucination" for det in detectors)

    source = inspect.getsource(module)
    assert "IsolationForestDetector" in source
    assert "OCSVMDetector" in source
    assert "NegSelAISDetector" in source
    assert "LOFDetector" not in source
    assert "AutoencoderDetector" not in source
    for token in FORBIDDEN_TOKENS:
        assert token not in source


def test_aggregate_scores_exposes_mean3_and_median3_with_adversarial_control():
    """A single extreme score should move mean3 more than median3."""
    module = load_module()

    aggregates = module.aggregate_scores([0.15, 0.18, 0.99])

    assert set(aggregates) == {"mean3", "median3"}
    assert aggregates["mean3"] == pytest.approx((0.15 + 0.18 + 0.99) / 3)
    assert aggregates["median3"] == pytest.approx(0.18)
    baseline = 0.18
    assert abs(aggregates["mean3"] - baseline) > abs(aggregates["median3"] - baseline)


def test_build_phase84_artifact_records_per_detector_scores_and_latency_fields():
    """Wave 3 needs raw detector outputs plus aggregate and timing fields."""
    module = load_module()

    sample_result = {
        "sample_id": "s1",
        "label": "anomalous",
        "prompt": "prompt",
        "output": "output",
        "per_detector": {
            "IsolationForest": {"score": 0.21, "confidence": 0.42, "latency_ms": 1.2},
            "OCSVM": {"score": 0.56, "confidence": 0.81, "latency_ms": 2.3},
            "NegSel": {"score": 0.92, "confidence": 0.97, "latency_ms": 3.4},
        },
        "aggregate_scores": {"mean3": 0.5633, "median3": 0.56},
        "latency_ms": {"detectors_total": 6.9, "aggregate_overhead": 0.1, "end_to_end": 7.0},
    }

    artifact = module.build_phase84_artifact(
        domain="hallucination",
        sample_results=[sample_result],
        spike_metrics={
            "mean3": {"f1": 0.91, "precision": 0.9, "recall": 0.92},
            "median3": {"f1": 0.89, "precision": 0.88, "recall": 0.9},
        },
        baseline_metrics_reference={"f1": 0.85, "precision": 0.84, "recall": 0.86, "accuracy": 0.85},
        baseline_detector_latencies_reference={"OCSVM": 1.1, "NegSel": 1.2, "IsolationForest": 1.3},
        baseline_control_metrics={"f1": 0.87, "precision": 0.86, "recall": 0.88, "accuracy": 0.87},
        average_control_latency_ms=3.5,
        adversarial_probe={
            "probe": "one_extreme_agent_perturbation",
            "description": "test",
            "perturbed_detector": "IsolationForest",
            "aggregators": {
                "mean3": {"prediction_flips": 1, "flip_rate": 1.0, "mean_score_shift": 0.2, "max_score_shift": 0.3},
                "median3": {"prediction_flips": 0, "flip_rate": 0.0, "mean_score_shift": 0.0, "max_score_shift": 0.0},
            },
        },
        mechanism_notes=["detector diversity only"],
    )

    assert artifact["phase"] == "84"
    assert artifact["aggregators"] == ["mean3", "median3"]
    assert artifact["domain"] == "hallucination"
    assert artifact["baseline_reference"] == "data/benchmarks/v9.2-baseline.json"
    assert artifact["sample_results"][0]["label"] == "anomalous"
    assert artifact["sample_results"][0]["per_detector"]["IsolationForest"]["score"] == 0.21
    assert artifact["sample_results"][0]["per_detector"]["OCSVM"]["latency_ms"] == 2.3
    assert artifact["sample_results"][0]["aggregate_scores"]["median3"] == 0.56
    assert artifact["sample_results"][0]["latency_ms"]["end_to_end"] == 7.0
    assert artifact["baseline"]["current_two_detector_control"]["avg_latency_ms"] == 3.5
    assert artifact["spike"]["aggregators"]["mean3"]["metrics"]["f1"] == 0.91
    assert artifact["adversarial_probe"]["aggregators"]["mean3"]["prediction_flips"] == 1
    assert artifact["mechanism_notes"] == ["detector diversity only"]


def test_validate_phase84_artifacts_succeeds_without_live_benchmark(tmp_path):
    """Validation mode checks staged structure without rerunning detectors."""
    module = load_module()

    valid_artifact = {
        "phase": "84",
        "baseline_reference": "data/benchmarks/v9.2-baseline.json",
        "phase83_stability_reference": {
            "artifact": ".planning/phases/83-determinism-controls-scoring-benchmark/83-benchmark-report.md",
            "exact_variance_pct": 0.0,
            "semantic_variance_pct": 0.0,
        },
        "decision_bar": {
            "absolute_f1_gain_gt": 0.03,
            "domains_required": 2,
            "latency_multiplier_lt": 2.0,
            "require_plausible_mechanism": True,
            "require_no_unmitigable_attack": True,
        },
        "domains": {
            "reasoning": {
                "phase": "84",
                "domain": "reasoning",
                "baseline_reference": "data/benchmarks/v9.2-baseline.json",
                "detectors": ["IsolationForest", "OCSVM", "NegSel"],
                "aggregators": ["mean3", "median3"],
                "baseline": {
                    "reference_metrics": {"f1": 0.8, "precision": 0.78, "recall": 0.82, "accuracy": 0.81},
                    "reference_detector_latencies_ms": {"OCSVM": 1.1, "NegSel": 1.2, "IsolationForest": 1.3},
                    "current_two_detector_control": {
                        "detectors": ["OCSVM", "NegSel"],
                        "metrics": {"f1": 0.79, "precision": 0.77, "recall": 0.81, "accuracy": 0.8},
                        "avg_latency_ms": 3.3,
                    },
                },
                "spike": {
                    "aggregators": {
                        "mean3": {
                            "metrics": {"f1": 0.8, "precision": 0.78, "recall": 0.82},
                            "f1_delta_vs_baseline": 0.0,
                            "f1_delta_vs_control": 0.01,
                            "avg_latency_ms": 3.4,
                            "latency_multiplier_vs_control": 1.03,
                        },
                        "median3": {
                            "metrics": {"f1": 0.79, "precision": 0.77, "recall": 0.81},
                            "f1_delta_vs_baseline": -0.01,
                            "f1_delta_vs_control": 0.0,
                            "avg_latency_ms": 3.4,
                            "latency_multiplier_vs_control": 1.03,
                        },
                    }
                },
                "adversarial_probe": {
                    "probe": "one_extreme_agent_perturbation",
                    "description": "test",
                    "perturbed_detector": "IsolationForest",
                    "aggregators": {
                        "mean3": {"prediction_flips": 1, "flip_rate": 1.0, "mean_score_shift": 0.2, "max_score_shift": 0.2},
                        "median3": {"prediction_flips": 0, "flip_rate": 0.0, "mean_score_shift": 0.0, "max_score_shift": 0.0},
                    },
                },
                "mechanism_notes": ["test note"],
                "sample_results": [
                    {
                        "sample_id": "r1",
                        "label": "normal",
                        "per_detector": {
                            "IsolationForest": {"score": 0.1, "confidence": 0.2, "latency_ms": 1.0},
                            "OCSVM": {"score": 0.2, "confidence": 0.3, "latency_ms": 1.1},
                            "NegSel": {"score": 0.3, "confidence": 0.4, "latency_ms": 1.2},
                        },
                        "aggregate_scores": {"mean3": 0.2, "median3": 0.2},
                        "latency_ms": {
                            "detectors_total": 3.3,
                            "aggregate_overhead": 0.1,
                            "end_to_end": 3.4,
                        },
                    }
                ],
            }
        },
        "decision_summary": {
            "aggregators": {
                "mean3": {
                    "domains_with_gain_gt_0_03": [],
                    "domain_gain_count": 0,
                    "max_latency_multiplier_vs_control": 1.03,
                    "plausible_mechanism": True,
                    "unmitigable_attack": True,
                    "meets_gate": False,
                },
                "median3": {
                    "domains_with_gain_gt_0_03": [],
                    "domain_gain_count": 0,
                    "max_latency_multiplier_vs_control": 1.03,
                    "plausible_mechanism": False,
                    "unmitigable_attack": False,
                    "meets_gate": False,
                },
            },
            "recommended_outcome": "NO-GO",
        },
    }

    artifact_path = tmp_path / "phase84-artifact.json"
    artifact_path.write_text(__import__("json").dumps(valid_artifact), encoding="utf-8")

    validated = module.validate_phase84_artifacts([artifact_path])

    assert validated["ok"] is True
    assert validated["validated_files"] == [str(artifact_path)]
    assert validated["artifact_count"] == 1
    assert validated["aggregators"] == ["mean3", "median3"]
