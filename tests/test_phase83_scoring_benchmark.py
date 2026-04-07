"""Contract tests for the Phase 83 scorer benchmark runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _mode_spread(mean: float, cv: float, flip_count: int) -> dict[str, float | int]:
    variance_pct = cv * 100.0
    return {
        "mean": mean,
        "std": mean * cv,
        "min": mean - 0.05,
        "max": mean + 0.05,
        "cv": cv,
        "variance_pct": variance_pct,
        "flip_count": flip_count,
    }


def _mode_summary(mean: float, cv: float, flip_count: int = 0) -> dict[str, object]:
    return {
        "reward_spread": _mode_spread(mean=mean, cv=cv, flip_count=flip_count),
        "repeats": 3,
        "samples_per_round": 2,
    }


def _full_results() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "hallucination": {
            "exact": _mode_summary(mean=0.60, cv=0.20, flip_count=2),
            "statistical": _mode_summary(mean=0.58, cv=0.08, flip_count=1),
            "semantic": _mode_summary(mean=0.62, cv=0.03, flip_count=0),
        },
        "code_security": {
            "exact": _mode_summary(mean=0.52, cv=0.18, flip_count=2),
            "statistical": _mode_summary(mean=0.55, cv=0.07, flip_count=1),
            "semantic": _mode_summary(mean=0.57, cv=0.02, flip_count=0),
        },
        "reasoning": {
            "exact": _mode_summary(mean=0.48, cv=0.17, flip_count=1),
            "statistical": _mode_summary(mean=0.51, cv=0.06, flip_count=1),
            "semantic": _mode_summary(mean=0.54, cv=0.04, flip_count=0),
        },
        "bio": {
            "exact": _mode_summary(mean=0.50, cv=0.16, flip_count=1),
            "statistical": _mode_summary(mean=0.49, cv=0.05, flip_count=1),
            "semantic": _mode_summary(mean=0.53, cv=0.01, flip_count=0),
        },
    }


def test_phase83_pilot_uses_build_validator_scorer_and_identical_fixture_inputs(monkeypatch):
    from scripts import phase83_scoring_benchmark as benchmark

    validator = SimpleNamespace(name="validator")
    miner_uids = [10, 11]
    responses_by_sample = {
        "sample-1": [SimpleNamespace(anomaly_score=0.8), SimpleNamespace(anomaly_score=0.2)]
    }
    manifest = {
        "sample-1": {
            "domain": "reasoning",
            "prompt": "Explain the bug",
            "output": "The invariant was violated.",
            "ground_truth_label": "anomalous",
            "is_honeypot": False,
        }
    }
    fixture_calls: list[tuple[str, int]] = []
    builder_calls: list[str] = []
    score_calls: list[tuple[str, object, object, object, object]] = []

    def fixture_factory(domain: str, repeat_index: int):
        fixture_calls.append((domain, repeat_index))
        return validator, miner_uids, responses_by_sample, manifest

    class StubScorer:
        def __init__(self, mode: str):
            self.mode = mode

        def score_round(self, *, validator, miner_uids, responses_by_sample, manifest):
            score_calls.append((self.mode, validator, miner_uids, responses_by_sample, manifest))
            return benchmark.ScoreResultLike(
                rewards=[0.6, 0.4],
                means=[0.6, 0.4],
                samples=[[0.6, 0.4]],
                repeats=1 if self.mode != "statistical" else 3,
            )

    def stub_builder(mode: str, *, repeats: int = 3, confidence_level: float = 0.95):
        del repeats, confidence_level
        builder_calls.append(mode)
        return StubScorer(mode)

    monkeypatch.setattr(benchmark, "build_validator_scorer", stub_builder)

    pilot = benchmark.run_pilot_benchmark(
        domain="reasoning",
        rounds=2,
        fixture_factory=fixture_factory,
    )

    assert fixture_calls == [
        ("reasoning", 0),
        ("reasoning", 1),
    ]
    assert builder_calls == ["exact", "statistical", "semantic"]
    assert pilot["domain"] == "reasoning"
    assert "pass" not in pilot["threshold_evaluation"]["exact"]["status"].lower()
    assert all(call[1] is validator for call in score_calls)
    assert all(call[2] is miner_uids for call in score_calls)
    assert all(call[3] is responses_by_sample for call in score_calls)
    assert all(call[4] is manifest for call in score_calls)


def test_phase83_pilot_records_measured_variance_pct_without_hardcoded_success():
    from scripts import phase83_scoring_benchmark as benchmark

    pilot = benchmark.build_phase83_artifact(
        pilot_result={
            "domain": "reasoning",
            "modes": {
                "exact": _mode_summary(mean=0.5, cv=0.02),
                "statistical": _mode_summary(mean=0.5, cv=0.01),
                "semantic": _mode_summary(mean=0.5, cv=0.00),
            },
        },
        domain_results={},
        rounds=2,
        model="stub-model",
    )["pilot"]

    assert pilot["modes"]["exact"]["reward_spread"]["variance_pct"] == pytest.approx(2.0)
    assert pilot["threshold_evaluation"]["exact"]["observed"] == pytest.approx(2.0)
    assert pilot["threshold_evaluation"]["exact"]["passed"] is False
    assert pilot["threshold_evaluation"]["exact"]["status"].lower() == "failed"


def test_phase83_artifact_schema_contains_pilot_domains_modes_spread_and_threshold_outcome():
    from scripts import phase83_scoring_benchmark as benchmark

    artifact = benchmark.build_phase83_artifact(
        pilot_result={
            "domain": "reasoning",
            "modes": {
                "exact": _mode_summary(mean=0.5, cv=0.17),
                "statistical": _mode_summary(mean=0.5, cv=0.06),
                "semantic": _mode_summary(mean=0.5, cv=0.04),
            },
        },
        domain_results=_full_results(),
        rounds=10,
        model="qwen2.5:1.5b",
    )

    assert artifact["pilot"]["domain"] == "reasoning"
    assert sorted(artifact["modes"]) == ["exact", "semantic", "statistical"]
    assert set(artifact["domains"]) == {
        "hallucination",
        "code_security",
        "reasoning",
        "bio",
    }
    assert artifact["domains"]["hallucination"]["exact"]["reward_spread"][
        "variance_pct"
    ] == pytest.approx(20.0)
    assert artifact["threshold_evaluation"]["semantic"]["target"] == "<5.0%"
    assert artifact["threshold_evaluation"]["semantic"]["passed"] is True
    assert artifact["threshold_evaluation"]["exact"]["target"] == ">15.0%"
    assert artifact["threshold_evaluation"]["exact"]["passed"] is True


def test_phase83_report_and_validator_preserve_best_effort_seed_boundary_and_swarm_scope(tmp_path):
    from scripts import phase83_scoring_benchmark as benchmark

    artifact = benchmark.build_phase83_artifact(
        pilot_result={
            "domain": "reasoning",
            "modes": {
                "exact": _mode_summary(mean=0.5, cv=0.02),
                "statistical": _mode_summary(mean=0.5, cv=0.01),
                "semantic": _mode_summary(mean=0.5, cv=0.00),
            },
        },
        domain_results=_full_results(),
        rounds=10,
        model="qwen2.5:1.5b",
    )
    artifact_path = tmp_path / "phase83-scoring-variance.json"
    report_path = tmp_path / "83-benchmark-report.md"

    benchmark._write_json(artifact_path, artifact)
    benchmark.write_phase83_report(artifact=artifact, destination=report_path)

    report_text = report_path.read_text()
    assert "best-effort" in report_text
    assert "swarm work remains out of scope" in report_text
    assert "threshold" in report_text
    assert "phase83-scoring-variance.json" in report_text

    validation = benchmark.validate_phase83_artifacts(
        artifact_path=artifact_path,
        report_path=report_path,
    )

    assert validation["ok"] is True
    assert validation["artifact"] == str(artifact_path)
    assert validation["report"] == str(report_path)
    assert validation["threshold_status"]["exact"] == "failed"
    assert validation["threshold_status"]["semantic"] == "passed"
    assert validation["boundary_language"]["best_effort_seed"] is True
    assert validation["boundary_language"]["swarm_out_of_scope"] is True
