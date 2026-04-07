#!/usr/bin/env python3
"""Minimal Phase 84 three-detector spike runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from antigence_subnet.miner.data import load_training_samples  # noqa: E402
from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector  # noqa: E402
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector  # noqa: E402
from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector  # noqa: E402
from scripts.benchmark_orchestrator import (  # noqa: E402
    DECISION_THRESHOLD,
    compute_metrics,
    load_eval_data,
)

DATA_DIR = ROOT_DIR / "data" / "evaluation"
BASELINE_REFERENCE = "data/benchmarks/v9.2-baseline.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "benchmarks" / "phase84-swarm-spike.json"
DETECTOR_ORDER = ("IsolationForest", "OCSVM", "NegSel")
AGGREGATORS = ("mean3", "median3")
BASELINE_PAIR = ("OCSVM", "NegSel")


def create_phase84_detectors(
    domain: str,
    training_samples: list[dict] | None = None,
) -> list:
    """Build and fit the exact three detectors used by the spike."""
    samples = training_samples
    if samples is None:
        samples = load_training_samples(str(DATA_DIR), domain)

    detectors = [
        IsolationForestDetector(random_state=42),
        OCSVMDetector(random_state=42),
        NegSelAISDetector(random_state=42),
    ]
    for detector in detectors:
        detector.domain = domain
        detector.fit(samples)
    return detectors


def aggregate_scores(scores: list[float]) -> dict[str, float]:
    """Return the only two aggregations allowed in Phase 84."""
    if len(scores) != 3:
        raise ValueError("Phase 84 spike requires exactly three detector scores")
    return {
        "mean3": float(np.mean(scores)),
        "median3": float(np.median(scores)),
    }


def _load_baseline_reference() -> dict:
    with (ROOT_DIR / BASELINE_REFERENCE).open(encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_domain_metrics(baseline_reference: dict, domain: str) -> dict:
    return baseline_reference["sections"]["orchestrator"]["data"]["domains"][domain][
        "flat_ensemble"
    ]


def _baseline_detector_latency_map(baseline_reference: dict, domain: str) -> dict[str, float]:
    latencies = {}
    for entry in baseline_reference["sections"]["detectors"]["data"]:
        if entry["domain"] == domain and entry["detector"] in DETECTOR_ORDER:
            latencies[entry["detector"]] = float(entry["avg_latency_ms"])
    return latencies


def _score_mean(scores: list[float]) -> float:
    return float(np.mean(scores))


def _one_extreme_agent_probe(sample_results: list[dict]) -> dict:
    aggregators = {}
    for aggregator in AGGREGATORS:
        score_shifts: list[float] = []
        prediction_flips = 0
        for sample in sample_results:
            original_score = float(sample["aggregate_scores"][aggregator])
            original_prediction = original_score >= DECISION_THRESHOLD
            perturbed_scores = [
                1.0,
                float(sample["per_detector"]["OCSVM"]["score"]),
                float(sample["per_detector"]["NegSel"]["score"]),
            ]
            perturbed_score = float(aggregate_scores(perturbed_scores)[aggregator])
            perturbed_prediction = perturbed_score >= DECISION_THRESHOLD
            score_shifts.append(perturbed_score - original_score)
            if perturbed_prediction != original_prediction:
                prediction_flips += 1
        aggregators[aggregator] = {
            "prediction_flips": prediction_flips,
            "flip_rate": round(prediction_flips / len(sample_results), 4),
            "mean_score_shift": round(float(np.mean(score_shifts)), 4),
            "max_score_shift": round(float(np.max(score_shifts)), 4),
        }
    return {
        "probe": "one_extreme_agent_perturbation",
        "description": "Replace the IsolationForest score with 1.0 to simulate one extreme detector vote.",  # noqa: E501
        "perturbed_detector": "IsolationForest",
        "aggregators": aggregators,
    }


def _mechanism_notes(
    *,
    baseline_f1: float,
    control_metrics: dict,
    spike_metrics: dict[str, dict],
    adversarial_probe: dict,
) -> list[str]:
    notes = []
    mean3_delta_vs_control = round(spike_metrics["mean3"]["f1"] - control_metrics["f1"], 4)
    median3_delta_vs_control = round(spike_metrics["median3"]["f1"] - control_metrics["f1"], 4)
    if mean3_delta_vs_control > 0 or median3_delta_vs_control > 0:
        notes.append(
            "Any uplift over the two-detector control comes from adding IsolationForest diversity; no swarm messaging, vote-tuning, or persistent state exists in this spike."  # noqa: E501
        )
    else:
        notes.append(
            "The added IsolationForest detector did not improve on the current OCSVM+NegSel control, so the spike shows no detector-diversification benefit."  # noqa: E501
        )
    if spike_metrics["mean3"]["f1"] > spike_metrics["median3"]["f1"]:
        notes.append(
            "Mean3 scores higher than median3 here, so any apparent gain depends on aggregation choice rather than a stable multi-agent mechanism."  # noqa: E501
        )
    elif spike_metrics["median3"]["f1"] > spike_metrics["mean3"]["f1"]:
        notes.append(
            "Median3 outperforms mean3, which suggests robustness matters more than raw averaging."
        )
    else:
        notes.append(
            "Mean3 and median3 tie on F1, so aggregation choice provides no differentiating mechanism."  # noqa: E501
        )
    mean3_flips = adversarial_probe["aggregators"]["mean3"]["prediction_flips"]
    median3_flips = adversarial_probe["aggregators"]["median3"]["prediction_flips"]
    notes.append(
        f"Against the one-extreme-agent probe, mean3 flips {mean3_flips} predictions and median3 flips {median3_flips}; this measures whether the gain survives a single extreme detector vote."  # noqa: E501
    )
    notes.append(
        f"The immutable v9.2 flat baseline F1 for this domain is {baseline_f1:.4f}; gate deltas are computed against that measured reference, not a recomputed story."  # noqa: E501
    )
    return notes


async def run_domain_spike(domain: str, *, baseline_reference: dict) -> dict:
    """Run the thin spike for one domain and return a benchmark artifact."""
    samples, manifest = load_eval_data(domain)
    detectors = create_phase84_detectors(domain)
    baseline_metrics_reference = _baseline_domain_metrics(baseline_reference, domain)
    baseline_detector_latencies_reference = _baseline_detector_latency_map(
        baseline_reference, domain
    )

    sample_results = []
    aggregate_score_sets = {name: [] for name in AGGREGATORS}
    baseline_control_scores = []
    labels = []

    for sample in samples:
        per_detector = {}
        detector_scores = []
        domain_start = time.perf_counter()

        for detector_name, detector in zip(DETECTOR_ORDER, detectors, strict=True):
            start = time.perf_counter()
            result = await detector.detect(
                prompt=sample.get("prompt", ""),
                output=sample.get("output", ""),
                code=sample.get("code"),
                context=sample.get("context"),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            detector_scores.append(result.score)
            per_detector[detector_name] = {
                "score": round(float(result.score), 6),
                "confidence": round(float(result.confidence), 6),
                "latency_ms": round(latency_ms, 3),
            }

        aggregate_start = time.perf_counter()
        aggregate_scores_result = aggregate_scores(detector_scores)
        aggregate_overhead_ms = (time.perf_counter() - aggregate_start) * 1000
        end_to_end_ms = (time.perf_counter() - domain_start) * 1000

        label = manifest.get(sample["id"], {}).get("ground_truth_label", "normal")
        labels.append(label)
        for aggregator, score in aggregate_scores_result.items():
            aggregate_score_sets[aggregator].append(score)
        baseline_control_scores.append(
            _score_mean(
                [
                    float(per_detector["OCSVM"]["score"]),
                    float(per_detector["NegSel"]["score"]),
                ]
            )
        )

        sample_results.append(
            {
                "sample_id": sample["id"],
                "label": label,
                "prompt": sample.get("prompt", ""),
                "output": sample.get("output", ""),
                "per_detector": per_detector,
                "aggregate_scores": {
                    name: round(float(score), 6) for name, score in aggregate_scores_result.items()
                },
                "latency_ms": {
                    "detectors_total": round(
                        sum(item["latency_ms"] for item in per_detector.values()),
                        3,
                    ),
                    "aggregate_overhead": round(aggregate_overhead_ms, 3),
                    "end_to_end": round(end_to_end_ms, 3),
                },
            }
        )

    baseline_control_metrics = compute_metrics(
        scores=baseline_control_scores,
        labels=labels,
        threshold=DECISION_THRESHOLD,
    )
    metrics = {
        aggregator: compute_metrics(
            scores=score_set,
            labels=labels,
            threshold=DECISION_THRESHOLD,
        )
        for aggregator, score_set in aggregate_score_sets.items()
    }
    average_control_latency_ms = round(
        float(
            np.mean(
                [
                    sample["per_detector"]["OCSVM"]["latency_ms"]
                    + sample["per_detector"]["NegSel"]["latency_ms"]
                    for sample in sample_results
                ]
            )
        ),
        4,
    )
    adversarial_probe = _one_extreme_agent_probe(sample_results)
    mechanism_notes = _mechanism_notes(
        baseline_f1=float(baseline_metrics_reference["f1"]),
        control_metrics=baseline_control_metrics,
        spike_metrics=metrics,
        adversarial_probe=adversarial_probe,
    )
    return build_phase84_artifact(
        domain=domain,
        sample_results=sample_results,
        spike_metrics=metrics,
        baseline_metrics_reference=baseline_metrics_reference,
        baseline_detector_latencies_reference=baseline_detector_latencies_reference,
        baseline_control_metrics=baseline_control_metrics,
        average_control_latency_ms=average_control_latency_ms,
        adversarial_probe=adversarial_probe,
        mechanism_notes=mechanism_notes,
    )


def build_phase84_artifact(
    domain: str,
    sample_results: list[dict],
    spike_metrics: dict[str, dict],
    baseline_metrics_reference: dict,
    baseline_detector_latencies_reference: dict[str, float],
    baseline_control_metrics: dict,
    average_control_latency_ms: float,
    adversarial_probe: dict,
    mechanism_notes: list[str],
) -> dict:
    """Create the domain artifact consumed by Wave 3."""
    spike_aggregators = {}
    for aggregator in AGGREGATORS:
        avg_latency_ms = round(
            float(np.mean([sample["latency_ms"]["end_to_end"] for sample in sample_results])),
            4,
        )
        spike_aggregators[aggregator] = {
            "metrics": spike_metrics[aggregator],
            "f1_delta_vs_baseline": round(
                float(spike_metrics[aggregator]["f1"]) - float(baseline_metrics_reference["f1"]),
                4,
            ),
            "f1_delta_vs_control": round(
                float(spike_metrics[aggregator]["f1"]) - float(baseline_control_metrics["f1"]),
                4,
            ),
            "avg_latency_ms": avg_latency_ms,
            "latency_multiplier_vs_control": round(avg_latency_ms / average_control_latency_ms, 4),
        }
    return {
        "phase": "84",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "baseline_reference": BASELINE_REFERENCE,
        "detectors": list(DETECTOR_ORDER),
        "aggregators": list(AGGREGATORS),
        "baseline": {
            "reference_metrics": baseline_metrics_reference,
            "reference_detector_latencies_ms": baseline_detector_latencies_reference,
            "current_two_detector_control": {
                "detectors": list(BASELINE_PAIR),
                "metrics": baseline_control_metrics,
                "avg_latency_ms": average_control_latency_ms,
            },
        },
        "spike": {
            "aggregators": spike_aggregators,
        },
        "adversarial_probe": adversarial_probe,
        "mechanism_notes": mechanism_notes,
        "sample_results": sample_results,
    }


def build_phase84_benchmark_artifact(domain_artifacts: list[dict]) -> dict:
    summary = {}
    overall_outcome = "NO-GO"
    for aggregator in AGGREGATORS:
        domains_with_gain = [
            artifact["domain"]
            for artifact in domain_artifacts
            if artifact["spike"]["aggregators"][aggregator]["f1_delta_vs_baseline"] > 0.03
        ]
        max_latency_multiplier = max(
            artifact["spike"]["aggregators"][aggregator]["latency_multiplier_vs_control"]
            for artifact in domain_artifacts
        )
        max_prediction_flips = max(
            artifact["adversarial_probe"]["aggregators"][aggregator]["prediction_flips"]
            for artifact in domain_artifacts
        )
        max_flip_rate = max(
            artifact["adversarial_probe"]["aggregators"][aggregator]["flip_rate"]
            for artifact in domain_artifacts
        )
        plausible_mechanism = any(
            artifact["spike"]["aggregators"][aggregator]["f1_delta_vs_control"] > 0
            for artifact in domain_artifacts
        )
        unmitigable_attack = max_flip_rate > 0.1
        meets_gate = (
            len(domains_with_gain) >= 2
            and max_latency_multiplier < 2.0
            and plausible_mechanism
            and not unmitigable_attack
        )
        summary[aggregator] = {
            "domains_with_gain_gt_0_03": domains_with_gain,
            "domain_gain_count": len(domains_with_gain),
            "max_latency_multiplier_vs_control": round(max_latency_multiplier, 4),
            "max_adversarial_flip_rate": round(max_flip_rate, 4),
            "max_adversarial_prediction_flips": max_prediction_flips,
            "plausible_mechanism": plausible_mechanism,
            "unmitigable_attack": unmitigable_attack,
            "meets_gate": meets_gate,
        }
        if meets_gate:
            overall_outcome = "GO"

    return {
        "phase": "84",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_reference": BASELINE_REFERENCE,
        "phase83_stability_reference": {
            "artifact": ".planning/phases/83-determinism-controls-scoring-benchmark/83-benchmark-report.md",  # noqa: E501
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
        "domains": {artifact["domain"]: artifact for artifact in domain_artifacts},
        "decision_summary": {
            "aggregators": summary,
            "recommended_outcome": overall_outcome,
        },
    }


def _staged_validation_artifact() -> dict:
    """Small staged artifact used by --validate-artifacts."""
    return build_phase84_artifact(
        domain="staged",
        sample_results=[
            {
                "sample_id": "staged-1",
                "label": "normal",
                "prompt": "prompt",
                "output": "output",
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
        spike_metrics={
            "mean3": {"f1": 0.8, "precision": 0.78, "recall": 0.82},
            "median3": {"f1": 0.79, "precision": 0.77, "recall": 0.81},
        },
        baseline_metrics_reference={"f1": 0.78, "precision": 0.76, "recall": 0.8, "accuracy": 0.79},
        baseline_detector_latencies_reference={"OCSVM": 1.1, "NegSel": 1.2, "IsolationForest": 1.3},
        baseline_control_metrics={"f1": 0.77, "precision": 0.75, "recall": 0.79, "accuracy": 0.78},
        average_control_latency_ms=2.3,
        adversarial_probe={
            "probe": "one_extreme_agent_perturbation",
            "description": "Replace the IsolationForest score with 1.0 to simulate one extreme detector vote.",  # noqa: E501
            "perturbed_detector": "IsolationForest",
            "aggregators": {
                "mean3": {
                    "prediction_flips": 1,
                    "flip_rate": 1.0,
                    "mean_score_shift": 0.4,
                    "max_score_shift": 0.4,
                },
                "median3": {
                    "prediction_flips": 0,
                    "flip_rate": 0.0,
                    "mean_score_shift": 0.0,
                    "max_score_shift": 0.0,
                },
            },
        },
        mechanism_notes=["staged validation artifact"],
    )


def _validate_artifact_dict(artifact: dict) -> None:
    """Raise ValueError if the artifact shape drifts from the Phase 84 contract."""
    if artifact.get("phase") != "84":
        raise ValueError("artifact phase must be '84'")
    if artifact.get("aggregators") != list(AGGREGATORS):
        raise ValueError("artifact aggregators must be mean3 and median3")
    if artifact.get("detectors") not in (None, list(DETECTOR_ORDER)):
        raise ValueError("artifact detectors must match the three-detector spike")
    if artifact.get("baseline_reference") != BASELINE_REFERENCE:
        raise ValueError(
            "artifact baseline_reference must anchor to data/benchmarks/v9.2-baseline.json"
        )

    baseline = artifact.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("artifact baseline must be a dict")
    if "reference_metrics" not in baseline or "current_two_detector_control" not in baseline:
        raise ValueError(
            "artifact baseline must include reference metrics and two-detector control"
        )

    spike = artifact.get("spike")
    if not isinstance(spike, dict) or "aggregators" not in spike:
        raise ValueError("artifact spike must include aggregators")
    for aggregator in AGGREGATORS:
        if aggregator not in spike["aggregators"]:
            raise ValueError(f"artifact spike missing aggregator {aggregator}")
        entry = spike["aggregators"][aggregator]
        for field in (
            "metrics",
            "f1_delta_vs_baseline",
            "f1_delta_vs_control",
            "avg_latency_ms",
            "latency_multiplier_vs_control",
        ):
            if field not in entry:
                raise ValueError(f"artifact spike aggregator {aggregator} missing {field}")

    adversarial_probe = artifact.get("adversarial_probe")
    if not isinstance(adversarial_probe, dict):
        raise ValueError("artifact must include adversarial_probe")
    for aggregator in AGGREGATORS:
        if aggregator not in adversarial_probe.get("aggregators", {}):
            raise ValueError(f"adversarial probe missing {aggregator}")

    sample_results = artifact.get("sample_results")
    if not isinstance(sample_results, list) or not sample_results:
        raise ValueError("artifact sample_results must be a non-empty list")

    for sample in sample_results:
        if "sample_id" not in sample or "label" not in sample:
            raise ValueError("sample result missing sample_id or label")
        per_detector = sample.get("per_detector")
        if list(per_detector.keys()) != list(DETECTOR_ORDER):
            raise ValueError("per_detector must include the exact detector set")
        for detector_name in DETECTOR_ORDER:
            detector_result = per_detector[detector_name]
            for field in ("score", "confidence", "latency_ms"):
                if field not in detector_result:
                    raise ValueError(f"{detector_name} missing {field}")
        if sample.get("aggregate_scores") is None:
            raise ValueError("sample result missing aggregate_scores")
        if list(sample["aggregate_scores"].keys()) != list(AGGREGATORS):
            raise ValueError("aggregate_scores must include mean3 and median3")
        latency = sample.get("latency_ms")
        if latency is None:
            raise ValueError("sample result missing latency_ms")
        for field in ("detectors_total", "aggregate_overhead", "end_to_end"):
            if field not in latency:
                raise ValueError(f"latency_ms missing {field}")


def _validate_benchmark_dict(artifact: dict) -> None:
    if artifact.get("phase") != "84":
        raise ValueError("benchmark phase must be '84'")
    if artifact.get("baseline_reference") != BASELINE_REFERENCE:
        raise ValueError(
            "benchmark baseline_reference must anchor to data/benchmarks/v9.2-baseline.json"
        )
    domains = artifact.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("benchmark domains must be a non-empty dict")
    for domain_artifact in domains.values():
        _validate_artifact_dict(domain_artifact)
    decision_summary = artifact.get("decision_summary")
    if not isinstance(decision_summary, dict):
        raise ValueError("benchmark missing decision_summary")
    for aggregator in AGGREGATORS:
        if aggregator not in decision_summary.get("aggregators", {}):
            raise ValueError(f"decision_summary missing {aggregator}")


def validate_phase84_artifacts(
    artifact_paths: list[str | Path] | None = None,
) -> dict:
    """Validate staged artifacts without running the benchmark."""
    validated_files: list[str] = []
    artifacts = []

    if artifact_paths:
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            with path.open(encoding="utf-8") as handle:
                artifact = json.load(handle)
            if "domains" in artifact and "decision_summary" in artifact:
                _validate_benchmark_dict(artifact)
            else:
                _validate_artifact_dict(artifact)
            artifacts.append(artifact)
            validated_files.append(str(path))
    else:
        staged = build_phase84_benchmark_artifact([_staged_validation_artifact()])
        _validate_benchmark_dict(staged)
        artifacts.append(staged)

    return {
        "ok": True,
        "artifact_count": len(artifacts),
        "validated_files": validated_files,
        "aggregators": list(AGGREGATORS),
    }


def _write_artifact(output_path: Path, artifact: dict) -> Path:
    """Persist the decision-grade benchmark artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path = output_path
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    return path


async def _run_cli(args: argparse.Namespace) -> int:
    if args.validate_artifacts:
        result = validate_phase84_artifacts(args.artifact_paths)
        print(json.dumps(result, indent=2))
        return 0

    baseline_reference = _load_baseline_reference()
    domain_artifacts = []
    for domain in args.domains:
        artifact = await run_domain_spike(domain, baseline_reference=baseline_reference)
        domain_artifacts.append(artifact)
    benchmark_artifact = build_phase84_benchmark_artifact(domain_artifacts)
    written_path = str(_write_artifact(args.output_path, benchmark_artifact))

    print(
        json.dumps(
            {
                "ok": True,
                "domains": args.domains,
                "written_path": written_path,
                "aggregators": list(AGGREGATORS),
                "recommended_outcome": benchmark_artifact["decision_summary"][
                    "recommended_outcome"
                ],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thin Phase 84 spike")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["hallucination", "code_security", "reasoning", "bio"],
        help="Domains to benchmark",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the benchmark artifact JSON",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Validate staged artifacts without running detectors",
    )
    parser.add_argument(
        "--artifact-paths",
        nargs="*",
        default=None,
        help="Optional artifact JSON paths for validation",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run_cli(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
