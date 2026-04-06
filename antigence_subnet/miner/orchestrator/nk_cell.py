"""NK Cell fast-path anomaly gate for the immune orchestrator pipeline.

NK (Natural Killer) Cells provide innate immune fast-path detection:
if any continuous feature exceeds its z-score threshold, the sample is
flagged immediately with DetectionResult(score=1.0, anomaly_type="nk_fast_path")
without running the full ensemble pipeline.

Feature-type-aware thresholds:
- Continuous features: z-score = |value - mean| / std, trigger if z > threshold
- Binary features (non-constant): skipped (normal variation)
- Constant features (std=0): skipped (z-score undefined)
- danger_signal: skipped by default (correlated with pamp_score, r=1.0)

Confidence scales per D-03: min(1.0, z_score / (2.0 * z_threshold)).

Loads per-domain statistics from Phase 31 audit JSON files via from_audit_json().
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from antigence_subnet.miner.detector import DetectionResult

# Per CONTEXT.md: pamp_score and danger_signal are perfectly correlated (r=1.0).
# Only threshold on pamp_score; skip danger_signal to avoid double-counting.
DEFAULT_SKIP_FEATURES: set[str] = {"danger_signal"}


@dataclass(frozen=True)
class FeatureStatistics:
    """Per-feature statistics from Phase 31 audit data.

    Attributes:
        name: Feature name (matches DendriticFeatureExtractor.FEATURE_NAMES).
        index: Position in the 10-dim feature vector.
        mean: Mean value across training samples.
        std: Standard deviation across training samples.
        is_binary: Whether feature takes only {0.0, 1.0} values.
        is_constant: Whether feature has zero variance (all samples same value).
    """

    name: str
    index: int
    mean: float
    std: float
    is_binary: bool
    is_constant: bool


class NKCell:
    """NK Cell fast-path anomaly gate.

    Evaluates 10-dim dendritic features against per-feature z-score
    thresholds. Returns DetectionResult immediately when any continuous
    feature exceeds its threshold, bypassing the ensemble pipeline.

    Satisfies ImmuneCellType Protocol via duck typing (structural subtyping).
    """

    def __init__(
        self,
        feature_stats: list[FeatureStatistics],
        z_threshold: float = 3.0,
        skip_features: set[str] | None = None,
    ) -> None:
        """Initialize NK Cell with per-feature statistics.

        Args:
            feature_stats: List of FeatureStatistics for each feature.
            z_threshold: Z-score threshold for triggering detection.
                Features with z > z_threshold trigger immediate detection.
            skip_features: Feature names to skip during evaluation.
                Defaults to DEFAULT_SKIP_FEATURES ({"danger_signal"}).
        """
        self._stats = feature_stats
        self._z_threshold = z_threshold
        self._skip_features = skip_features if skip_features is not None else DEFAULT_SKIP_FEATURES

    @classmethod
    def from_audit_json(
        cls,
        audit_path: str | Path,
        z_threshold: float = 3.0,
        skip_features: set[str] | None = None,
    ) -> NKCell:
        """Create NKCell from a Phase 31 audit JSON file.

        Args:
            audit_path: Path to audit JSON (e.g., data/audit/hallucination.json).
            z_threshold: Z-score threshold for triggering detection.
            skip_features: Feature names to skip. Defaults to DEFAULT_SKIP_FEATURES.

        Returns:
            NKCell configured with per-feature statistics from the audit.
        """
        with open(audit_path) as f:
            data = json.load(f)

        stats: list[FeatureStatistics] = []
        for name in data["feature_names"]:
            fs = data["feature_stats"][name]
            stats.append(
                FeatureStatistics(
                    name=name,
                    index=fs["index"],
                    mean=fs["mean"],
                    std=fs["std"],
                    is_binary=fs["is_binary"],
                    is_constant=fs["is_constant"],
                )
            )

        return cls(feature_stats=stats, z_threshold=z_threshold, skip_features=skip_features)

    def process(
        self,
        features: np.ndarray,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
        *,
        z_threshold: float | None = None,
    ) -> DetectionResult | None:
        """Evaluate features against per-feature z-score thresholds.

        Returns DetectionResult immediately if any continuous feature exceeds
        the z-score threshold. Returns None if all features are within normal
        range, deferring to the ensemble pipeline.

        Args:
            features: Feature vector (shape (10,) from DendriticFeatureExtractor).
            prompt: Original prompt text.
            output: AI-generated output text.
            code: Optional code content.
            context: Optional JSON-serialized metadata.
            z_threshold: Optional per-request z-score threshold override.
                When not None, used instead of self._z_threshold for this
                invocation (per-domain config support, Phase 36).

        Returns:
            DetectionResult with score=1.0 if fast-path triggered, None otherwise.
        """
        effective_threshold = z_threshold if z_threshold is not None else self._z_threshold

        for stat in self._stats:
            # Skip features in the skip set (e.g., danger_signal)
            if stat.name in self._skip_features:
                continue

            # Skip constant features (std=0, z-score undefined)
            if stat.is_constant:
                continue

            value = float(features[stat.index])

            # Skip binary non-constant features (normal variation)
            if stat.is_binary:
                continue

            # Continuous feature: compute z-score
            z_score = abs(value - stat.mean) / stat.std

            if z_score > effective_threshold:
                confidence = min(1.0, z_score / (2.0 * effective_threshold))
                return DetectionResult(
                    score=1.0,
                    confidence=confidence,
                    anomaly_type="nk_fast_path",
                    feature_attribution={stat.name: round(z_score, 4)},
                )

        return None
