"""Regression tests for Phase 81 non-determinism measurement artifacts.

These tests cover pure helpers (artifact schemas, variance summarisation,
overwatch payload shape) and never call into the live `ollama` package.
The harness module imports `ollama` lazily inside the functions that need
it, so this file collects and runs cleanly on minimal CI where the
`ollama` Python package is not installed.
"""

from scripts.ollama_test_harness import DECISION_THRESHOLD


def test_fixed_input_repeats_hold_seed_and_samples():
    from scripts.phase81_nondeterminism import build_fixed_input_repeats

    selected_samples = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]
    repeats = build_fixed_input_repeats(
        domain="hallucination",
        selected_samples=selected_samples,
        seed=42,
        repeats=10,
    )

    assert len(repeats) == 10
    expected_ids = ["s1", "s2", "s3"]
    for repeat_index, repeat in enumerate(repeats, start=1):
        assert repeat["repeat"] == repeat_index
        assert repeat["seed"] == 42
        assert repeat["sample_ids"] == expected_ids
        assert "response" in repeat
        assert "score_outputs" in repeat


def test_variance_summary_counts_threshold_flips():
    from scripts.phase81_nondeterminism import summarize_domain_variance

    summary = summarize_domain_variance(
        domain="hallucination",
        response_repeats=[
            {"response": {"text": "a"}, "score_outputs": [0.49]},
            {"response": {"text": "b"}, "score_outputs": [0.51]},
            {"response": {"text": "c"}, "score_outputs": [0.48]},
            {"response": {"text": "d"}, "score_outputs": [0.52]},
        ],
        round_runs=[
            {"metrics": {"reward": 0.40, "f1": 0.20}},
            {"metrics": {"reward": 0.60, "f1": 0.30}},
            {"metrics": {"reward": 0.45, "f1": 0.40}},
            {"metrics": {"reward": 0.55, "f1": 0.50}},
        ],
    )

    spread = summary["response_variance"]["spread"]
    assert summary["decision_threshold"] == DECISION_THRESHOLD
    assert set(spread) == {"mean", "std", "min", "max", "cv", "flip_count"}
    assert spread["flip_count"] == 3
    assert spread["min"] == 0.48
    assert spread["max"] == 0.52
    assert summary["round_variance"]["spread"]["flip_count"] == 3


def test_phase81_artifact_schema():
    from scripts.phase81_nondeterminism import validate_phase81_artifacts

    artifact = {
        "environment": {"python": "python3.11", "model": "qwen2.5:1.5b"},
        "baseline_reference": "data/benchmarks/v9.2-baseline.json",
        "domains": {
            "hallucination": {
                "response_variance": {"spread": {"mean": 0.5}},
                "round_variance": {"spread": {"mean": 0.5}},
                "source_decomposition": {
                    "gpu_kernel": {},
                    "batching_scheduling": {},
                    "floating_point_kernel": {},
                    "harness_level_randomness": {},
                },
            }
        },
        "response_variance": {"global_summary": {}},
        "round_variance": {"global_summary": {}},
    }
    markdown = """# Phase 81

miners remain adversarial. determinism is best-effort only. validators cannot force
exact miner inference behavior. controls must remain backward-compatible and
commodity-hardware feasible.

## Observed variance
Observed variance.

## Inferred variance sources
Inferred variance sources.

## Recommended mitigations
Recommended mitigations.

source_decomposition
"""
    overwatch = {
        "claims": [
            {
                "text": (
                    "Observed variance stays separate from inferred sources; determinism is best-effort, "  # noqa: E501
                    "validators cannot force exact miner inference behavior, and controls must remain "  # noqa: E501
                    "backward-compatible on commodity-hardware."
                ),
                "observed_variance": ["Measured reward spread"],
                "inferred_sources": ["Likely batching effects under best-effort serving controls."],
                "recommended_mitigations": [
                    "Use repeated scoring windows with backward-compatible commodity-hardware controls."  # noqa: E501
                ],
                "source_decomposition": {
                    "gpu_kernel": {"evidence_refs": ["local:artifact"]},
                    "batching_scheduling": {"evidence_refs": ["local:artifact"]},
                    "floating_point_kernel": {"evidence_refs": ["local:artifact"]},
                    "harness_level_randomness": {"evidence_refs": ["local:artifact"]},
                },
            }
        ]
    }
    report = {"run_id": "phase81-demo", "counts": {"claims": 0}, "errors": []}

    validate_phase81_artifacts(
        artifact_data=artifact,
        markdown_text=markdown,
        overwatch_payload=overwatch,
        writeback_report=report,
    )


def test_overwatch_payload_separates_observed_and_inferred():
    from scripts.phase81_nondeterminism import build_overwatch_claim_payload

    payload = build_overwatch_claim_payload(
        experiment_data={
            "run_id": "phase81-demo",
            "source_decomposition": {
                "gpu_kernel": {
                    "observed_signal": "Observed reward spread under fixed inputs.",
                    "evidence_refs": ["pub:2408.04667", "local:artifact#gpu_kernel"],
                    "inference": "This may reflect runtime-level kernel variance.",
                    "mitigation": "Use repeated runs before changing thresholds.",
                },
                "batching_scheduling": {
                    "observed_signal": "Observed response drift across repeats.",
                    "evidence_refs": ["pub:vllm", "local:artifact#batching_scheduling"],
                    "inference": "This may reflect scheduling order sensitivity.",
                    "mitigation": "Measure on stable load before threshold tuning.",
                },
                "floating_point_kernel": {
                    "observed_signal": "Observed near-threshold score movement.",
                    "evidence_refs": ["pub:mbrenndoerfer", "local:artifact#floating_point_kernel"],
                    "inference": "This may reflect floating-point or kernel differences.",
                    "mitigation": "Prefer margin-aware scoring rather than exact matches.",
                },
                "harness_level_randomness": {
                    "observed_signal": "Observed selection-path variability in baseline mode.",
                    "evidence_refs": ["local:artifact#harness_level_randomness"],
                    "inference": "This reflects controllable harness randomness, not miner behavior.",  # noqa: E501
                    "mitigation": "Freeze sample IDs in research measurements.",
                },
            },
        }
    )

    claim = payload["claims"][0]
    assert "observed_variance" in claim
    assert "inferred_sources" in claim
    assert "recommended_mitigations" in claim
    assert claim["observed_variance"] != claim["inferred_sources"]
    assert "may" in claim["inferred_sources"][0].lower()


def test_writeback_report_schema():
    from scripts.phase81_nondeterminism import build_writeback_report

    report = build_writeback_report(
        run_id="phase81-demo",
        counts={"claims": 2, "derived_from": 2, "cites": 3},
        errors=[],
    )

    assert report["run_id"] == "phase81-demo"
    assert report["counts"]["claims"] == 2
    assert report["errors"] == []
