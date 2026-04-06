"""Danger Theory two-signal costimulation modulator for ensemble anomaly scores.

Adjusts ensemble detection scores using costimulation derived from dendritic
features. The two-signal model requires both an anomaly signal (raw ensemble
score) AND a costimulation signal (max of pamp_score and danger_signal) for
full activation -- mirroring biological T cell activation.

Formula (per D-02):
    modulated_score = raw_score + (1 - raw_score) * costimulation * alpha

Properties (per DANGER-01/02/03):
- Costimulation = max(pamp_score[5], danger_signal[9]) from dendritic features
- Scores stay in [0.0, 1.0]: raw + (1-raw)*c*a <= raw + (1-raw) = 1
- Monotonic: if raw_A > raw_B, then modulated_A >= modulated_B (same costim/alpha)
- Only adds to raw score, never subtracts (high-confidence preserved)
- 0.5 decision threshold unchanged (modulation shifts distribution, not threshold)
- When disabled, returns raw score unchanged (backward compatibility)
"""

from __future__ import annotations

import numpy as np

from antigence_subnet.miner.detector import DetectionResult

# Feature indices in the 10-dim dendritic feature vector
_PAMP_SCORE_IDX = 5
_DANGER_SIGNAL_IDX = 9


class DangerTheoryModulator:
    """Two-signal costimulation modulator for ensemble anomaly scores.

    Per D-02: modulated_score = raw_score + (1 - raw_score) * costimulation * alpha

    Properties (per DANGER-01/02/03):
    - Costimulation = max(pamp_score, danger_signal) from dendritic features
    - Scores stay in [0.0, 1.0] (formula guarantees: raw + (1-raw)*c*a <= raw + (1-raw) = 1)
    - Monotonic: if raw_A > raw_B, then modulated_A >= modulated_B (same costim/alpha)
    - Only adds to raw score, never subtracts (high-confidence preserved)
    - 0.5 decision threshold unchanged (modulation shifts distribution, not threshold)
    """

    def __init__(self, alpha: float = 0.3, enabled: bool = True) -> None:
        """Initialize the danger theory modulator.

        Args:
            alpha: Modulation strength parameter. Controls how much
                costimulation boosts the raw score. Default 0.3.
                alpha=0.0 means no modulation (passthrough).
            enabled: Whether modulation is active. When False, modulate()
                returns the raw score unchanged.
        """
        self._alpha = alpha
        self._enabled = enabled

    @classmethod
    def from_config(cls, danger_config: dict) -> DangerTheoryModulator:
        """Create DangerTheoryModulator from a danger_config dict.

        Args:
            danger_config: Configuration dict with optional keys:
                - alpha (float): Modulation strength. Default 0.3.
                - enabled (bool): Whether modulation is active. Default True.

        Returns:
            DangerTheoryModulator configured from the provided dict.
        """
        return cls(
            alpha=danger_config.get("alpha", 0.3),
            enabled=danger_config.get("enabled", True),
        )

    def costimulation(self, features: np.ndarray) -> float:
        """Extract costimulation signal: max(pamp_score[5], danger_signal[9]).

        Args:
            features: Feature vector of shape (10,) from DendriticFeatureExtractor.

        Returns:
            Costimulation value in [0.0, 1.0].
        """
        return float(max(features[_PAMP_SCORE_IDX], features[_DANGER_SIGNAL_IDX]))

    def modulate(
        self,
        raw_score: float,
        costimulation: float,
        *,
        alpha: float | None = None,
        enabled: bool | None = None,
    ) -> float:
        """Apply danger theory modulation to a single score.

        If disabled, returns raw_score unchanged.
        Formula: raw_score + (1 - raw_score) * costimulation * alpha
        Clamped to [0.0, 1.0] for safety.

        Args:
            raw_score: Raw ensemble anomaly score in [0.0, 1.0].
            costimulation: Costimulation signal in [0.0, 1.0].
            alpha: Optional per-request alpha override. When not None,
                used instead of self._alpha (per-domain config, Phase 36).
            enabled: Optional per-request enabled override. When not None,
                used instead of self._enabled (per-domain config, Phase 36).

        Returns:
            Modulated score in [0.0, 1.0].
        """
        effective_enabled = enabled if enabled is not None else self._enabled
        if not effective_enabled:
            return raw_score
        effective_alpha = alpha if alpha is not None else self._alpha
        modulated = raw_score + (1.0 - raw_score) * costimulation * effective_alpha
        return max(0.0, min(1.0, modulated))

    def modulate_result(
        self,
        result: DetectionResult,
        features: np.ndarray,
        *,
        alpha: float | None = None,
        enabled: bool | None = None,
    ) -> DetectionResult:
        """Apply modulation to a DetectionResult, preserving all other fields.

        Args:
            result: Original DetectionResult from ensemble_detect.
            features: Feature vector of shape (10,) from DendriticFeatureExtractor.
            alpha: Optional per-request alpha override (per-domain config, Phase 36).
            enabled: Optional per-request enabled override (per-domain config, Phase 36).

        Returns:
            New DetectionResult with modulated score, same confidence/anomaly_type/attribution.
        """
        costim = self.costimulation(features)
        new_score = self.modulate(result.score, costim, alpha=alpha, enabled=enabled)
        return DetectionResult(
            score=new_score,
            confidence=result.confidence,
            anomaly_type=result.anomaly_type,
            feature_attribution=result.feature_attribution,
        )
