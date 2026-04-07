"""Contract tests for Phase 92 policy overlays and reporting boundaries."""

from __future__ import annotations

import pytest


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-1",
            "domain": "prompt_injection",
            "label": 1,
            "latency_ms": 5.0,
            "memory_mb": 32.0,
            "per_detector": {
                "OCSVM": {"score": 0.92, "confidence": 0.88},
                "NegSel": {"score": 0.78, "confidence": 0.74},
            },
        },
        {
            "sample_id": "sample-2",
            "domain": "prompt_injection",
            "label": 0,
            "latency_ms": 4.5,
            "memory_mb": 31.0,
            "per_detector": {
                "OCSVM": {"score": 0.34, "confidence": 0.82},
                "NegSel": {"score": 0.26, "confidence": 0.77},
            },
        },
        {
            "sample_id": "sample-3",
            "domain": "hallucination",
            "label": 1,
            "latency_ms": 5.8,
            "memory_mb": 33.0,
            "per_detector": {
                "OCSVM": {"score": 0.74, "confidence": 0.69},
                "NegSel": {"score": 0.68, "confidence": 0.72},
            },
        },
        {
            "sample_id": "sample-4",
            "domain": "hallucination",
            "label": 0,
            "latency_ms": 4.7,
            "memory_mb": 30.5,
            "per_detector": {
                "OCSVM": {"score": 0.22, "confidence": 0.83},
                "NegSel": {"score": 0.31, "confidence": 0.85},
            },
        },
    ]


def test_phase92_policy_overlays_use_saved_score_surfaces_and_pin_minimum_policy_set():
    from scripts import phase92_continuous_benchmark as benchmark

    artifact = benchmark.build_default_artifact()
    overlays = benchmark.apply_policy_overlays(artifact)
    expected_surfaces = list(artifact["score_quality"]["surfaces"])

    assert list(overlays) == [
        "global_threshold",
        "domain_thresholds",
        "operator_multiband",
    ]
    for _policy_name, policy_payload in overlays.items():
        assert policy_payload["source_surface"] == "saved_score_quality_surfaces"
        assert list(policy_payload["surface_results"]) == expected_surfaces
        assert policy_payload["uses_saved_scores"] is True
        assert policy_payload["detector_reruns"] == 0
        assert "control_equal" in policy_payload["deltas_vs_control"]
        assert policy_payload["deltas_vs_control"]["control_equal"]["f1_delta"] == pytest.approx(
            0.0
        )
        for surface_payload in policy_payload["surface_results"].values():
            assert set(surface_payload["metrics"]) >= {
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "policy_reward",
            }


def test_phase92_policy_outputs_stay_separate_from_score_quality_in_artifact_and_report(tmp_path):
    from scripts import phase92_continuous_benchmark as benchmark

    artifact = benchmark.build_default_artifact()
    benchmark.attach_policy_overlays(artifact)
    artifact_path = tmp_path / "phase92.json"
    report_path = tmp_path / "phase92.md"

    benchmark.write_phase92_artifact(artifact_path, artifact)
    benchmark.write_phase92_report(report_path, artifact)
    validation = benchmark.validate_phase92_artifacts(
        artifact_path=artifact_path,
        report_path=report_path,
    )
    report_text = report_path.read_text()

    assert "score_quality" in artifact
    assert "policy_overlays" in artifact
    assert "policy_overlays" not in artifact["score_quality"]
    assert "## Score Quality" in report_text
    assert "## Policy Overlays" in report_text
    assert validation["boundary_sections"]["score_quality"] is True
    assert validation["boundary_sections"]["policy_overlays"] is True
    assert validation["report_sections"]["score_quality"] is True
    assert validation["report_sections"]["policy_overlays"] is True


def test_phase92_operator_multiband_reports_allow_review_block_and_coverage_metrics():
    from scripts import phase92_continuous_benchmark as benchmark

    artifact = benchmark.build_default_artifact()
    overlays = benchmark.apply_policy_overlays(artifact)
    multiband = overlays["operator_multiband"]["surface_results"]["control_equal"]

    assert multiband["bands"] == ["allow", "review", "block"]
    assert set(multiband["decision_counts"]) == {"allow", "review", "block"}
    assert set(multiband["metrics"]) >= {
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "policy_reward",
        "review_rate",
        "auto_decision_coverage",
    }
    assert 0.0 <= multiband["metrics"]["review_rate"] <= 1.0
    assert 0.0 <= multiband["metrics"]["auto_decision_coverage"] <= 1.0
    assert (
        pytest.approx(1.0 - multiband["metrics"]["review_rate"], rel=0.0, abs=1e-6)
        == multiband["metrics"]["auto_decision_coverage"]
    )


def test_phase92_validation_rejects_scope_drift_for_swarm_terms_and_detector_expansion(tmp_path):
    from scripts import phase92_continuous_benchmark as benchmark

    artifact = benchmark.build_default_artifact()
    benchmark.attach_policy_overlays(artifact)
    artifact["policy_overlays"]["mean3_shadow"] = {"source_surface": "control_equal"}
    artifact["score_quality"]["surfaces"]["control_equal"]["weights"]["ExtraDetector"] = 0.1
    artifact["score_quality"]["surface_names"].append("swarm_shadow")
    artifact_path = tmp_path / "phase92-invalid.json"
    report_path = tmp_path / "phase92-invalid.md"

    benchmark.write_phase92_artifact(artifact_path, artifact)
    report_path.write_text("# Invalid\n")

    with pytest.raises(ValueError, match="forbidden|detector-count expansion"):
        benchmark.validate_phase92_artifacts(artifact_path=artifact_path, report_path=report_path)


def test_phase92_report_validation_rejects_missing_boundary_sections(tmp_path):
    from scripts import phase92_continuous_benchmark as benchmark

    artifact = benchmark.build_default_artifact()
    benchmark.attach_policy_overlays(artifact)
    artifact_path = tmp_path / "phase92.json"
    report_path = tmp_path / "phase92.md"

    benchmark.write_phase92_artifact(artifact_path, artifact)
    report_path.write_text("# Phase 92\n\n## Score Quality\n")

    with pytest.raises(ValueError, match="report must separate Score Quality and Policy Overlays"):
        benchmark.validate_phase92_artifacts(artifact_path=artifact_path, report_path=report_path)
