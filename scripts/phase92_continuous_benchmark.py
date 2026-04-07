#!/usr/bin/env python3
"""Phase 92 continuous-score benchmark canon."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.validator.calibration import compute_ece

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_REFERENCE = "data/benchmarks/v9.2-baseline.json"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "data/benchmarks/phase92-continuous-benchmark.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / ".planning/phases/92-continuous-antibody-benchmark-canon/92-benchmark-report.md"
)
ALLOWED_DETECTORS = ("OCSVM", "NegSel")
MINIMUM_POLICY_SET = ("global_threshold", "domain_thresholds", "operator_multiband")
STATIC_WEIGHT_SURFACES = (
    ("weighted_ocsvm_0.20_negsel_0.80", {"OCSVM": 0.20, "NegSel": 0.80}),
    ("weighted_ocsvm_0.35_negsel_0.65", {"OCSVM": 0.35, "NegSel": 0.65}),
    ("weighted_ocsvm_0.50_negsel_0.50", {"OCSVM": 0.50, "NegSel": 0.50}),
    ("weighted_ocsvm_0.65_negsel_0.35", {"OCSVM": 0.65, "NegSel": 0.35}),
    ("weighted_ocsvm_0.80_negsel_0.20", {"OCSVM": 0.80, "NegSel": 0.20}),
)
FORBIDDEN_SURFACE_TERMS = ("mean3", "median3", "swarm", "agent", "vote", "iforest")
FORBIDDEN_POLICY_TERMS = FORBIDDEN_SURFACE_TERMS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: float) -> float:
    return float(round(value, 6))


def _detection_result(payload: dict[str, float]) -> DetectionResult:
    return DetectionResult(
        score=float(payload["score"]),
        confidence=float(payload["confidence"]),
        anomaly_type="phase92_surface",
    )


def build_fixed_comparison_surfaces() -> list[dict[str, Any]]:
    surfaces = [
        {
            "name": "control_equal",
            "weights": {"OCSVM": 0.50, "NegSel": 0.50},
            "confidence_mode": "mean",
        }
    ]
    for name, weights in STATIC_WEIGHT_SURFACES:
        surfaces.append({"name": name, "weights": dict(weights), "confidence_mode": "mean"})
    surfaces.append(
        {
            "name": "confidence_modulated_static",
            "weights": {"OCSVM": 0.65, "NegSel": 0.35},
            "confidence_mode": "weighted_by_confidence",
        }
    )
    for surface in surfaces:
        lowered = surface["name"].lower()
        if any(term in lowered for term in FORBIDDEN_SURFACE_TERMS):
            raise ValueError(f"forbidden comparison surface: {surface['name']}")
        if set(surface["weights"]) != set(ALLOWED_DETECTORS):
            raise ValueError("comparison surfaces must stay on the fixed OCSVM + NegSel set")
    return surfaces


def build_candidate_records(rows: list[dict[str, Any]]) -> list[str]:
    del rows
    return [surface["name"] for surface in build_fixed_comparison_surfaces()]


def build_sample_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        detectors = row["per_detector"]
        if set(detectors) != set(ALLOWED_DETECTORS):
            raise ValueError("sample rows must preserve the fixed detector set")
        sample_record = {
            "sample_id": row["sample_id"],
            "domain": row.get("domain", "unknown"),
            "label": int(row["label"]),
            "latency_ms": _float(float(row["latency_ms"])),
            "memory_mb": _float(float(row["memory_mb"])),
            "per_detector": {},
        }
        for detector_name in ALLOWED_DETECTORS:
            result = _detection_result(detectors[detector_name])
            sample_record["per_detector"][detector_name] = {
                "score": _float(result.score),
                "confidence": _float(result.confidence),
            }
        samples.append(sample_record)
    return samples


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def _surface_score(
    detector_results: dict[str, dict[str, float]],
    weights: dict[str, float],
    *,
    confidence_mode: str,
) -> tuple[float, float]:
    ordered = [weights[detector] for detector in ALLOWED_DETECTORS]
    total_weight = sum(ordered)
    normalized = {detector: weights[detector] / total_weight for detector in ALLOWED_DETECTORS}

    if confidence_mode == "weighted_by_confidence":
        effective_weights = {}
        for detector in ALLOWED_DETECTORS:
            effective_weights[detector] = normalized[detector] * float(
                detector_results[detector]["confidence"]
            )
        effective_total = sum(effective_weights.values())
        if effective_total == 0.0:
            effective_weights = normalized
            effective_total = 1.0
        score = sum(
            (effective_weights[detector] / effective_total)
            * float(detector_results[detector]["score"])
            for detector in ALLOWED_DETECTORS
        )
        confidence = _bounded_ratio(
            sum(
                effective_weights[detector] * float(detector_results[detector]["confidence"])
                for detector in ALLOWED_DETECTORS
            ),
            effective_total,
        )
        return score, confidence

    score = sum(
        normalized[detector] * float(detector_results[detector]["score"])
        for detector in ALLOWED_DETECTORS
    )
    confidence = sum(
        normalized[detector] * float(detector_results[detector]["confidence"])
        for detector in ALLOWED_DETECTORS
    )
    return score, confidence


def _safe_roc_auc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _reliability_bins(
    confidences: list[float], accuracies: list[int], n_bins: int = 5
) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_arr = np.asarray(confidences, dtype=np.float64)
    acc_arr = np.asarray(accuracies, dtype=np.float64)
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index < n_bins - 1:
            mask = (conf_arr >= lower) & (conf_arr < upper)
        else:
            mask = (conf_arr >= lower) & (conf_arr <= upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bins.append(
            {
                "lower": _float(lower),
                "upper": _float(upper),
                "count": count,
                "avg_confidence": _float(float(np.mean(conf_arr[mask]))),
                "avg_accuracy": _float(float(np.mean(acc_arr[mask]))),
            }
        )
    return bins


def _score_quality_metrics(
    *,
    labels: list[int],
    scores: list[float],
    confidences: list[float],
    latencies: list[float],
    memories: list[float],
) -> dict[str, Any]:
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    accuracies = [
        int(prediction == label) for prediction, label in zip(predictions, labels, strict=True)
    ]
    anomalous_scores = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    normal_scores = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    separation = float(np.mean(anomalous_scores) - np.mean(normal_scores))
    return {
        "average_precision": _float(float(average_precision_score(labels, scores))),
        "roc_auc": _float(_safe_roc_auc(labels, scores)),
        "separation": _float(separation),
        "avg_latency_ms": _float(float(np.mean(latencies))),
        "peak_memory_mb": _float(float(np.max(memories))),
        "brier_loss": _float(float(brier_score_loss(labels, scores))),
        "ece": _float(float(compute_ece(confidences, accuracies, n_bins=10))),
        "reliability_bins": _reliability_bins(confidences, accuracies),
    }


def evaluate_score_surfaces(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    samples = build_sample_records(rows)
    labels = [sample["label"] for sample in samples]
    latencies = [float(sample["latency_ms"]) for sample in samples]
    memories = [float(sample["memory_mb"]) for sample in samples]
    surfaces: dict[str, dict[str, Any]] = {}
    for surface in build_fixed_comparison_surfaces():
        per_sample: list[dict[str, Any]] = []
        scores: list[float] = []
        confidences: list[float] = []
        for sample in samples:
            score, confidence = _surface_score(
                sample["per_detector"],
                surface["weights"],
                confidence_mode=surface["confidence_mode"],
            )
            per_sample_record = {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "label": sample["label"],
                "score": _float(score),
                "confidence": _float(confidence),
            }
            per_sample.append(per_sample_record)
            scores.append(float(per_sample_record["score"]))
            confidences.append(float(per_sample_record["confidence"]))
        surfaces[surface["name"]] = {
            "weights": dict(surface["weights"]),
            "confidence_mode": surface["confidence_mode"],
            "per_sample": per_sample,
            "metrics": _score_quality_metrics(
                labels=labels,
                scores=scores,
                confidences=confidences,
                latencies=latencies,
                memories=memories,
            ),
        }
    return surfaces


def generate_paired_bootstrap_rounds(
    *,
    sample_ids: list[str],
    rounds: int,
    sample_size: int,
    seed: int,
    candidates: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not sample_ids:
        raise ValueError("sample_ids must not be empty")
    candidate_names = list(candidates)
    rng = Random(seed)
    paired_rounds: list[dict[str, Any]] = []
    for round_index in range(rounds):
        sampled_ids = [rng.choice(sample_ids) for _ in range(sample_size)]
        paired_rounds.append(
            {
                "round_index": round_index,
                "sample_ids": sampled_ids,
                "candidate_names": candidate_names,
                "samples_by_candidate": {
                    candidate: list(sampled_ids) for candidate in candidate_names
                },
            }
        )
    return paired_rounds


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    return {
        "precision": _float(precision),
        "recall": _float(recall),
        "f1": _float(float(f1_score(labels, predictions, zero_division=0))),
        "balanced_accuracy": _float(float(balanced_accuracy_score(labels, predictions))),
        "policy_reward": _float(0.7 * precision + 0.3 * recall),
    }


def _threshold_candidates(scores: list[float]) -> list[float]:
    unique_scores = sorted({float(score) for score in scores})
    if not unique_scores:
        return [0.5]
    candidates = {0.5}
    candidates.update(unique_scores)
    for lower, upper in zip(unique_scores, unique_scores[1:], strict=False):
        candidates.add((lower + upper) / 2.0)
    return sorted(candidates)


def _choose_global_threshold(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    scores = [float(record["score"]) for record in records]
    best_threshold = 0.5
    best_predictions = [1 if score >= best_threshold else 0 for score in scores]
    best_metrics = _binary_metrics(labels, best_predictions)
    best_reward = float(best_metrics["policy_reward"])
    for threshold in _threshold_candidates(scores):
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = _binary_metrics(labels, predictions)
        reward = float(metrics["policy_reward"])
        if reward > best_reward:
            best_threshold = threshold
            best_predictions = predictions
            best_metrics = metrics
            best_reward = reward
    return {
        "selection": {"threshold": _float(best_threshold)},
        "predictions": best_predictions,
        "metrics": best_metrics,
    }


def _choose_domain_thresholds(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(str(record.get("domain", "unknown")), []).append(record)
    thresholds: dict[str, float] = {}
    predictions_by_sample: dict[str, int] = {}
    for domain, domain_records in by_domain.items():
        choice = _choose_global_threshold(domain_records)
        thresholds[domain] = float(choice["selection"]["threshold"])
        for record, prediction in zip(domain_records, choice["predictions"], strict=True):
            predictions_by_sample[str(record["sample_id"])] = int(prediction)
    ordered_predictions = [predictions_by_sample[str(record["sample_id"])] for record in records]
    labels = [int(record["label"]) for record in records]
    return {
        "selection": {
            "thresholds_by_domain": {domain: _float(value) for domain, value in thresholds.items()}
        },
        "predictions": ordered_predictions,
        "metrics": _binary_metrics(labels, ordered_predictions),
    }


def _choose_operator_multiband(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    score_values = [float(record["score"]) for record in records]
    confidence_values = [float(record["confidence"]) for record in records]
    low_candidates = _threshold_candidates(score_values)
    high_candidates = _threshold_candidates(score_values)
    confidence_candidates = sorted({0.6, 0.7, 0.8, *confidence_values})

    best_payload: dict[str, Any] | None = None
    best_objective = -math.inf
    for low_threshold, high_threshold, min_confidence in product(
        low_candidates,
        high_candidates,
        confidence_candidates,
    ):
        if low_threshold >= high_threshold:
            continue
        decisions: list[str] = []
        binary_predictions: list[int] = []
        for record in records:
            score = float(record["score"])
            confidence = float(record["confidence"])
            if score >= high_threshold and confidence >= min_confidence:
                decision = "block"
                prediction = 1
            elif score < low_threshold and confidence >= min_confidence:
                decision = "allow"
                prediction = 0
            else:
                decision = "review"
                prediction = 1
            decisions.append(decision)
            binary_predictions.append(prediction)
        metrics = _binary_metrics(labels, binary_predictions)
        review_rate = _safe_divide(decisions.count("review"), len(decisions))
        auto_coverage = 1.0 - review_rate
        objective = float(metrics["policy_reward"]) + 0.05 * auto_coverage
        if objective > best_objective:
            best_objective = objective
            best_payload = {
                "selection": {
                    "low_threshold": _float(low_threshold),
                    "high_threshold": _float(high_threshold),
                    "min_confidence": _float(min_confidence),
                },
                "decisions": decisions,
                "binary_predictions": binary_predictions,
                "metrics": {
                    **metrics,
                    "review_rate": _float(review_rate),
                    "auto_decision_coverage": _float(auto_coverage),
                },
                "decision_counts": {
                    "allow": decisions.count("allow"),
                    "review": decisions.count("review"),
                    "block": decisions.count("block"),
                },
                "bands": ["allow", "review", "block"],
            }
    if best_payload is None:
        raise ValueError("operator multiband selection requires score records")
    return best_payload


def _policy_candidates(policy_name: str, surface_records: list[dict[str, Any]]) -> dict[str, Any]:
    if policy_name == "global_threshold":
        result = _choose_global_threshold(surface_records)
    elif policy_name == "domain_thresholds":
        result = _choose_domain_thresholds(surface_records)
    elif policy_name == "operator_multiband":
        result = _choose_operator_multiband(surface_records)
    else:
        raise ValueError(f"unsupported policy overlay: {policy_name}")
    result["sample_count"] = len(surface_records)
    return result


def apply_policy_overlays(artifact: dict[str, Any]) -> dict[str, Any]:
    surfaces = artifact.get("score_quality", {}).get("surfaces", {})
    overlays: dict[str, Any] = {}
    control_name = "control_equal"
    for policy_name in MINIMUM_POLICY_SET:
        surface_results: dict[str, Any] = {}
        for surface_name, surface_payload in surfaces.items():
            surface_results[surface_name] = _policy_candidates(
                policy_name,
                list(surface_payload.get("per_sample", [])),
            )
        control_metrics = surface_results[control_name]["metrics"]
        deltas_vs_control = {}
        for surface_name, surface_payload in surface_results.items():
            surface_metrics = surface_payload["metrics"]
            deltas_vs_control[surface_name] = {
                "precision_delta": _float(
                    float(surface_metrics["precision"]) - float(control_metrics["precision"])
                ),
                "recall_delta": _float(
                    float(surface_metrics["recall"]) - float(control_metrics["recall"])
                ),
                "f1_delta": _float(float(surface_metrics["f1"]) - float(control_metrics["f1"])),
                "balanced_accuracy_delta": _float(
                    float(surface_metrics["balanced_accuracy"])
                    - float(control_metrics["balanced_accuracy"])
                ),
                "policy_reward_delta": _float(
                    float(surface_metrics["policy_reward"])
                    - float(control_metrics["policy_reward"])
                ),
            }
            if policy_name == "operator_multiband":
                deltas_vs_control[surface_name]["review_rate_delta"] = _float(
                    float(surface_metrics["review_rate"]) - float(control_metrics["review_rate"])
                )
                deltas_vs_control[surface_name]["auto_decision_coverage_delta"] = _float(
                    float(surface_metrics["auto_decision_coverage"])
                    - float(control_metrics["auto_decision_coverage"])
                )
        overlays[policy_name] = {
            "source_surface": "saved_score_quality_surfaces",
            "source_control_surface": control_name,
            "uses_saved_scores": True,
            "detector_reruns": 0,
            "surface_results": surface_results,
            "deltas_vs_control": deltas_vs_control,
        }
    return overlays


def attach_policy_overlays(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["policy_overlays"] = apply_policy_overlays(artifact)
    artifact["plan"] = "02"
    artifact["generated_at"] = _now_iso()
    return artifact


def build_phase92_artifact(
    *,
    samples: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    paired_rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": "92",
        "plan": "01",
        "generated_at": _now_iso(),
        "baseline_reference": BASELINE_REFERENCE,
        "score_quality": {
            "detector_set": list(ALLOWED_DETECTORS),
            "surface_names": list(surfaces),
            "surfaces": surfaces,
        },
        "policy_overlays": {},
        "paired_bootstrap": {
            "design": "shared_sample_ids_per_round",
            "round_count": len(paired_rounds),
            "rounds": paired_rounds,
        },
        "samples": samples,
    }


def write_phase92_artifact(destination: Path, artifact: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def _validation_error(message: str) -> None:
    raise ValueError(message)


def _default_rows_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): dict(row) for row in _default_rows()}


def _enrich_saved_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    defaults_by_id = _default_rows_by_id()
    for sample in artifact.get("samples", []):
        default_row = defaults_by_id.get(str(sample.get("sample_id")))
        if default_row and "domain" not in sample:
            sample["domain"] = default_row.get("domain", "unknown")
    for surface_payload in artifact.get("score_quality", {}).get("surfaces", {}).values():
        metrics = surface_payload.get("metrics", {})
        if "brier_loss" not in metrics and "brier" in metrics:
            metrics["brier_loss"] = metrics.pop("brier")
        for record in surface_payload.get("per_sample", []):
            default_row = defaults_by_id.get(str(record.get("sample_id")))
            if default_row and "domain" not in record:
                record["domain"] = default_row.get("domain", "unknown")
    return artifact


def load_saved_score_surface(*, artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    if not artifact_path.exists():
        return build_default_artifact()
    artifact = json.loads(artifact_path.read_text())
    return _enrich_saved_artifact(artifact)


def _format_metric_line(label: str, metrics: dict[str, Any], keys: tuple[str, ...]) -> str:
    rendered = " | ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
    return f"- {label}: {rendered}"


def build_phase92_report(artifact: dict[str, Any]) -> str:
    score_quality = artifact["score_quality"]
    policy_overlays = artifact["policy_overlays"]
    lines = [
        "# Phase 92 Benchmark Report",
        "",
        f"- Generated at: {artifact['generated_at']}",
        f"- Baseline reference: `{artifact['baseline_reference']}`",
        "- Source surface: saved continuous score artifact from Plan 92-01",
        "- Scope guard: Phase 92 preserves the Phase 84 swarm NO-GO and does not add detector families or swarm terms.",  # noqa: E501
        "",
        "## Score Quality",
        "",
        "Continuous score-quality and calibration metrics are reported separately from policy overlays.",  # noqa: E501
        "",
    ]
    for surface_name in score_quality["surface_names"]:
        surface_payload = score_quality["surfaces"][surface_name]
        metrics = surface_payload["metrics"]
        lines.extend(
            [
                f"### {surface_name}",
                f"- weights: {surface_payload['weights']}",
                f"- confidence_mode: {surface_payload['confidence_mode']}",
                _format_metric_line(
                    "headline",
                    metrics,
                    (
                        "average_precision",
                        "roc_auc",
                        "brier_loss",
                        "ece",
                        "avg_latency_ms",
                        "peak_memory_mb",
                    ),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Policy Overlays",
            "",
            "Decision policies are post-hoc overlays on the saved score surfaces above. They do not rerun detectors.",  # noqa: E501
            "",
        ]
    )
    for policy_name in MINIMUM_POLICY_SET:
        policy_payload = policy_overlays[policy_name]
        lines.extend([f"### {policy_name}", ""])
        for surface_name, surface_payload in policy_payload["surface_results"].items():
            lines.append(f"#### {surface_name}")
            selection = surface_payload["selection"]
            lines.append(f"- selection: {json.dumps(selection, sort_keys=True)}")
            lines.append(
                _format_metric_line(
                    "metrics",
                    surface_payload["metrics"],
                    (
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "policy_reward",
                        "review_rate",
                        "auto_decision_coverage",
                    ),
                )
            )
            delta = policy_payload["deltas_vs_control"][surface_name]
            lines.append(f"- paired_delta_vs_control_equal: {json.dumps(delta, sort_keys=True)}")
            if policy_name == "operator_multiband":
                lines.append(
                    f"- decision_counts: {json.dumps(surface_payload['decision_counts'], sort_keys=True)}"  # noqa: E501
                )
                lines.append(f"- bands: {', '.join(surface_payload['bands'])}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_phase92_report(destination: Path, artifact: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_phase92_report(artifact))


def validate_phase92_artifacts(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    report_path: Path | None = None,
    require_policy_overlays: bool = False,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("baseline_reference") != BASELINE_REFERENCE:
        _validation_error(
            "artifact baseline_reference must anchor to data/benchmarks/v9.2-baseline.json"
        )
    if "score_quality" not in artifact or "policy_overlays" not in artifact:
        _validation_error("artifact must separate score_quality and policy_overlays")
    score_quality = artifact["score_quality"]
    if score_quality.get("detector_set") != list(ALLOWED_DETECTORS):
        _validation_error("score_quality.detector_set must preserve the fixed two-detector control")
    surface_names = score_quality.get("surface_names", [])
    if any(any(term in name.lower() for term in FORBIDDEN_SURFACE_TERMS) for name in surface_names):
        _validation_error("forbidden swarm or detector-expansion term found in score surfaces")
    if set(score_quality.get("surfaces", {})) != set(surface_names):
        _validation_error("score_quality.surfaces keys must match surface_names")
    for surface_name, surface in score_quality.get("surfaces", {}).items():
        if set(surface.get("weights", {})) != set(ALLOWED_DETECTORS):
            _validation_error(f"{surface_name} introduced detector-count expansion")
        for record in surface.get("per_sample", []):
            if not {"sample_id", "domain", "label", "score", "confidence"} <= set(record):
                _validation_error(
                    f"{surface_name} per-sample records must preserve score and confidence"
                )
        metrics = surface.get("metrics", {})
        required_score_metrics = {
            "average_precision",
            "roc_auc",
            "brier_loss",
            "ece",
            "avg_latency_ms",
            "peak_memory_mb",
        }
        if not required_score_metrics <= set(metrics):
            _validation_error(f"{surface_name} is missing required score-quality metrics")
    policy_overlays = artifact["policy_overlays"]
    if not isinstance(policy_overlays, dict):
        _validation_error("policy_overlays must be a mapping")
    if require_policy_overlays and set(policy_overlays) != set(MINIMUM_POLICY_SET):
        _validation_error("policy_overlays must contain exactly the minimum policy set")
    for policy_name, policy_payload in policy_overlays.items():
        lowered = policy_name.lower()
        if any(term in lowered for term in FORBIDDEN_POLICY_TERMS):
            _validation_error("forbidden policy overlay scope drift detected")
        if (
            policy_payload.get("uses_saved_scores") is not True
            or policy_payload.get("detector_reruns") != 0
        ):
            _validation_error(f"{policy_name} must consume saved scores without detector reruns")
        surface_results = policy_payload.get("surface_results", {})
        if set(surface_results) != set(surface_names):
            _validation_error(f"{policy_name} must evaluate every saved score surface")
        for surface_name, surface_payload in surface_results.items():
            metrics = surface_payload.get("metrics", {})
            required_policy_metrics = {
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "policy_reward",
            }
            if not required_policy_metrics <= set(metrics):
                _validation_error(
                    f"{policy_name}/{surface_name} is missing required policy metrics"
                )
            if policy_name == "operator_multiband":
                multiband_required = {"review_rate", "auto_decision_coverage"}
                if not multiband_required <= set(metrics):
                    _validation_error(
                        "operator_multiband must report review_rate and auto_decision_coverage"
                    )
                if surface_payload.get("bands") != ["allow", "review", "block"]:
                    _validation_error(
                        "operator_multiband must preserve allow/review/block semantics"
                    )
    paired_bootstrap = artifact.get("paired_bootstrap", {})
    rounds = paired_bootstrap.get("rounds", [])
    for round_payload in rounds:
        base_ids = round_payload.get("sample_ids", [])
        for candidate_name, candidate_ids in round_payload.get("samples_by_candidate", {}).items():
            if list(candidate_ids) != list(base_ids):
                _validation_error(f"paired bootstrap mismatch for candidate {candidate_name}")
    report_sections = {"score_quality": False, "policy_overlays": False}
    if report_path is not None:
        report_text = report_path.read_text()
        report_sections = {
            "score_quality": "## Score Quality" in report_text,
            "policy_overlays": "## Policy Overlays" in report_text,
        }
        if not all(report_sections.values()):
            _validation_error("report must separate Score Quality and Policy Overlays")
    return {
        "ok": True,
        "artifact": str(artifact_path),
        "report": str(report_path) if report_path is not None else None,
        "boundary_sections": {
            "score_quality": "score_quality" in artifact,
            "policy_overlays": "policy_overlays" in artifact,
        },
        "report_sections": report_sections,
        "surface_names": surface_names,
        "policy_overlay_count": len(policy_overlays),
    }


def _default_rows() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "sample-1",
            "domain": "prompt_injection",
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
            "domain": "prompt_injection",
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
            "domain": "hallucination",
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
            "domain": "hallucination",
            "label": 0,
            "latency_ms": 4.7,
            "memory_mb": 30.5,
            "per_detector": {
                "OCSVM": {"score": 0.27, "confidence": 0.79},
                "NegSel": {"score": 0.35, "confidence": 0.82},
            },
        },
    ]


def build_default_artifact() -> dict[str, Any]:
    rows = _default_rows()
    samples = build_sample_records(rows)
    surfaces = evaluate_score_surfaces(rows)
    paired_rounds = generate_paired_bootstrap_rounds(
        sample_ids=[row["sample_id"] for row in rows],
        rounds=5,
        sample_size=3,
        seed=92,
        candidates=tuple(surfaces),
    )
    return build_phase92_artifact(samples=samples, surfaces=surfaces, paired_rounds=paired_rounds)


def build_final_artifact(*, artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    artifact = load_saved_score_surface(artifact_path=artifact_path)
    return attach_policy_overlays(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to the Phase 92 score-layer benchmark artifact.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the Phase 92 markdown report.",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Validate the staged Phase 92 artifact without rerunning a live benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_artifacts:
        artifact = build_final_artifact(artifact_path=args.artifact_path)
        write_phase92_artifact(args.artifact_path, artifact)
        write_phase92_report(args.report_path, artifact)
        validation = validate_phase92_artifacts(
            artifact_path=args.artifact_path,
            report_path=args.report_path,
            require_policy_overlays=True,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    artifact = build_final_artifact(artifact_path=args.artifact_path)
    write_phase92_artifact(args.artifact_path, artifact)
    write_phase92_report(args.report_path, artifact)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact_path),
                "report": str(args.report_path),
                "surface_names": list(artifact["score_quality"]["surfaces"]),
                "policy_overlays": list(artifact["policy_overlays"]),
                "paired_rounds": artifact["paired_bootstrap"]["round_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
