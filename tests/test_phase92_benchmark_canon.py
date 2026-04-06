"""Contract tests for the Phase 92 continuous benchmark canon."""

from __future__ import annotations

from pathlib import Path

import pytest


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-1",
            "label": 1,
            "latency_ms": 5.0,
            "memory_mb": 32.0,
            "per_detector": {
                "OCSVM": {"score": 0.92, "confidence": 0.81},
                "NegSel": {"score": 0.78, "confidence": 0.73},
            },
        },
        {
            "sample_id": "sample-2",
            "label": 0,
            "latency_ms": 4.5,
            "memory_mb": 31.0,
            "per_detector": {
                "OCSVM": {"score": 0.18, "confidence": 0.84},
                "NegSel": {"score": 0.28, "confidence": 0.76},
            },
        },
        {
            "sample_id": "sample-3",
            "label": 1,
            "latency_ms": 5.8,
            "memory_mb": 33.0,
            "per_detector": {
                "OCSVM": {"score": 0.71, "confidence": 0.66},
                "NegSel": {"score": 0.65, "confidence": 0.69},
            },
        },
        {
            "sample_id": "sample-4",
            "label": 0,
            "latency_ms": 4.7,
            "memory_mb": 30.5,
            "per_detector": {
                "OCSVM": {"score": 0.27, "confidence": 0.79},
                "NegSel": {"score": 0.35, "confidence": 0.82},
            },
        },
    ]


def test_phase92_comparison_surfaces_are_fixed_to_two_detector_control():
    from scripts import phase92_continuous_benchmark as benchmark

    surfaces = benchmark.build_fixed_comparison_surfaces()
    surface_names = [surface["name"] for surface in surfaces]

    assert surface_names[0] == "control_equal"
    assert "confidence_modulated_static" in surface_names
    assert len(surfaces) == 7
    assert surface_names == [
        "control_equal",
        "weighted_ocsvm_0.20_negsel_0.80",
        "weighted_ocsvm_0.35_negsel_0.65",
        "weighted_ocsvm_0.50_negsel_0.50",
        "weighted_ocsvm_0.65_negsel_0.35",
        "weighted_ocsvm_0.80_negsel_0.20",
        "confidence_modulated_static",
    ]
    forbidden_terms = ("mean3", "median3", "swarm", "agent", "vote", "iforest")
    assert all(not any(term in name.lower() for term in forbidden_terms) for name in surface_names)
    assert all(set(surface["weights"]) == {"OCSVM", "NegSel"} for surface in surfaces)


def test_phase92_bootstrap_rounds_are_paired_across_candidates():
    from scripts import phase92_continuous_benchmark as benchmark

    rows = _sample_rows()
    candidates = benchmark.build_candidate_records(rows)
    paired_rounds = benchmark.generate_paired_bootstrap_rounds(
        sample_ids=[row["sample_id"] for row in rows],
        rounds=4,
        sample_size=3,
        seed=11,
        candidates=tuple(candidates),
    )

    assert len(paired_rounds) == 4
    assert paired_rounds[0]["candidate_names"] == list(candidates)
    assert all(sorted(round_payload["sample_ids"]) == sorted(round_payload["samples_by_candidate"]["control_equal"]) for round_payload in paired_rounds)
    assert all(
        round_payload["samples_by_candidate"]["control_equal"]
        == round_payload["samples_by_candidate"]["confidence_modulated_static"]
        for round_payload in paired_rounds
    )
    assert len({tuple(round_payload["sample_ids"]) for round_payload in paired_rounds}) > 1


def test_phase92_artifact_schema_separates_score_quality_from_policy_overlays(tmp_path):
    from scripts import phase92_continuous_benchmark as benchmark

    rows = _sample_rows()
    artifact = benchmark.build_phase92_artifact(
        samples=benchmark.build_sample_records(rows),
        surfaces=benchmark.evaluate_score_surfaces(rows),
        paired_rounds=benchmark.generate_paired_bootstrap_rounds(
            sample_ids=[row["sample_id"] for row in rows],
            rounds=3,
            sample_size=3,
            seed=7,
            candidates=("control_equal", "weighted_ocsvm_0.65_negsel_0.35"),
        ),
    )
    artifact_path = tmp_path / "phase92-continuous-benchmark.json"

    benchmark.write_phase92_artifact(artifact_path, artifact)
    validation = benchmark.validate_phase92_artifacts(artifact_path=artifact_path)

    assert set(artifact) >= {
        "baseline_reference",
        "score_quality",
        "policy_overlays",
        "paired_bootstrap",
        "samples",
    }
    assert "surfaces" in artifact["score_quality"]
    assert artifact["policy_overlays"] == {}
    assert validation["ok"] is True
    assert validation["boundary_sections"]["score_quality"] is True
    assert validation["boundary_sections"]["policy_overlays"] is True


def test_phase92_score_surface_records_preserve_detectionresult_score_and_confidence():
    from scripts import phase92_continuous_benchmark as benchmark

    rows = _sample_rows()
    samples = benchmark.build_sample_records(rows)
    surfaces = benchmark.evaluate_score_surfaces(rows)

    assert samples[0]["per_detector"]["OCSVM"] == {
        "score": pytest.approx(0.92),
        "confidence": pytest.approx(0.81),
    }
    assert "score" in surfaces["control_equal"]["per_sample"][0]
    assert "confidence" in surfaces["control_equal"]["per_sample"][0]
    assert "score" in surfaces["confidence_modulated_static"]["per_sample"][0]
    assert "confidence" in surfaces["confidence_modulated_static"]["per_sample"][0]
    assert Path("data/benchmarks/v9.2-baseline.json").as_posix() == benchmark.BASELINE_REFERENCE
