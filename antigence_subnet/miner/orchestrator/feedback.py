"""Validator feedback loop for miner self-improvement.

Monitors miner's metagraph weight over time. Weight increases indicate
rewarded detection patterns; decreases indicate penalized patterns.
Feeds inferred signals into B Cell memory and DCA weight adaptation.

Per FEEDBACK-01: Track own weight, correlate with detection patterns.
Per FEEDBACK-02: Feed feedback into B Cell reinforcement and DCA adaptation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from antigence_subnet.miner.orchestrator.b_cell import _N_FEATURES, _OUTCOME_COL

logger = logging.getLogger(__name__)


@dataclass
class RoundRecord:
    """Record of a single evaluation round for feedback correlation."""

    round_num: int
    weight_before: float
    weight_after: float
    avg_score: float
    detection_count: int
    domain: str


@dataclass
class DetectionRecord:
    """Record of a single detection for per-signature weight correlation.

    Stores features, anomaly score, domain, and timestamp so weight changes
    can be correlated with specific detection patterns.
    """

    round_num: int
    features: np.ndarray  # 10-dim dendritic features
    anomaly_score: float
    domain: str
    timestamp: float  # time.time()


class ValidatorFeedbackTracker:
    """Tracks miner weight changes and correlates with detection patterns.

    Args:
        lookback_rounds: Number of rounds to consider for correlation.
        enabled: Whether feedback tracking is active.
    """

    def __init__(
        self,
        lookback_rounds: int = 5,
        enabled: bool = False,
    ) -> None:
        self._lookback = lookback_rounds
        self._enabled = enabled
        self._history: deque[RoundRecord] = deque(maxlen=lookback_rounds * 2)
        self._detections: deque[DetectionRecord] = deque(
            maxlen=lookback_rounds * 50,
        )
        self._round_num = 0
        self._last_weight: float | None = None

    @classmethod
    def from_config(cls, feedback_config: dict) -> ValidatorFeedbackTracker:
        """Create from TOML config dict."""
        return cls(
            lookback_rounds=feedback_config.get("lookback_rounds", 5),
            enabled=feedback_config.get("enabled", False),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_round(
        self,
        current_weight: float,
        avg_score: float,
        detection_count: int,
        domain: str = "all",
    ) -> float:
        """Record a round and return the feedback signal.

        Args:
            current_weight: Miner's current weight in metagraph.
            avg_score: Average detection score this round.
            detection_count: Number of detections this round.
            domain: Domain for this round.

        Returns:
            Feedback signal in [-1.0, 1.0]. Positive = rewarded, negative = penalized.
            Returns 0.0 on first round (no prior weight to compare).
        """
        if not self._enabled:
            return 0.0

        weight_before = self._last_weight if self._last_weight is not None else current_weight
        self._last_weight = current_weight

        record = RoundRecord(
            round_num=self._round_num,
            weight_before=weight_before,
            weight_after=current_weight,
            avg_score=avg_score,
            detection_count=detection_count,
            domain=domain,
        )
        self._history.append(record)
        self._round_num += 1

        # Compute feedback signal from weight delta
        if weight_before == 0.0:
            return 0.0
        delta = (current_weight - weight_before) / max(abs(weight_before), 1e-8)
        # Clamp to [-1, 1]
        signal = max(-1.0, min(1.0, delta * 10.0))  # amplify small weight changes

        logger.debug(
            f"Feedback round {self._round_num}: weight {weight_before:.6f} -> {current_weight:.6f} "
            f"delta={delta:.6f} signal={signal:.4f}"
        )

        return signal

    def get_recent_signal(self) -> float:
        """Get aggregate feedback signal over lookback window.

        Returns:
            Mean feedback signal over recent rounds, or 0.0 if insufficient data.
        """
        if not self._enabled or len(self._history) < 2:
            return 0.0

        recent = list(self._history)[-self._lookback:]
        signals = []
        for i in range(1, len(recent)):
            prev_w = recent[i - 1].weight_after
            curr_w = recent[i].weight_after
            if prev_w > 0:
                delta = (curr_w - prev_w) / max(abs(prev_w), 1e-8)
                signals.append(max(-1.0, min(1.0, delta * 10.0)))

        return float(np.mean(signals)) if signals else 0.0

    def record_detection(
        self,
        features: np.ndarray,
        anomaly_score: float,
        domain: str,
    ) -> None:
        """Store a detection record for later per-signature correlation.

        Args:
            features: 10-dim dendritic feature vector.
            anomaly_score: Detection anomaly score (0.0-1.0).
            domain: Domain for this detection.
        """
        if not self._enabled:
            return

        record = DetectionRecord(
            round_num=self._round_num,
            features=features.copy(),
            anomaly_score=anomaly_score,
            domain=domain,
            timestamp=time.time(),
        )
        self._detections.append(record)

    def get_recent_detections(self, n_rounds: int | None = None) -> list[DetectionRecord]:
        """Return detection records within the lookback window.

        Args:
            n_rounds: Number of rounds to look back. Defaults to lookback_rounds.

        Returns:
            List of DetectionRecord within the window.
        """
        if n_rounds is None:
            n_rounds = self._lookback

        cutoff_round = self._round_num - n_rounds
        return [d for d in self._detections if d.round_num >= cutoff_round]

    def apply_to_bcell(self, b_cell, signal: float) -> None:
        """Apply feedback signal to B Cell memory (all signatures).

        Positive signal: reinforce all signatures (boost outcomes).
        Negative signal: accelerate decay of all signatures.

        BCell._memory is np.ndarray of shape (N, 12) where column 11
        is the outcome column (_OUTCOME_COL).

        Args:
            b_cell: BCell instance with numpy memory bank.
            signal: Feedback signal in [-1.0, 1.0].
        """
        if not self._enabled or b_cell is None or signal == 0.0:
            return

        if not hasattr(b_cell, '_memory') or b_cell._memory is None:
            return

        if len(b_cell._memory) == 0:
            return

        if signal > 0:
            # Reinforce: boost outcome scores slightly
            boost = signal * 0.1
            b_cell._memory[:, _OUTCOME_COL] = np.clip(
                b_cell._memory[:, _OUTCOME_COL] + boost, 0.0, 1.0
            )
        else:
            # Penalize: decay outcome scores
            decay = 1.0 + signal * 0.1  # signal is negative, so decay < 1.0
            b_cell._memory[:, _OUTCOME_COL] = (
                b_cell._memory[:, _OUTCOME_COL] * max(decay, 0.5)
            )

        logger.debug(
            f"BCell feedback applied: signal={signal:.4f}, "
            f"affected {len(b_cell._memory)} signatures"
        )

    def apply_feedback_to_bcell_correlated(
        self,
        b_cell,
        signal: float,
        top_k: int = 5,
    ) -> None:
        """Apply feedback to BCell signatures correlated with recent detections.

        Uses cosine similarity between recent detection features and BCell
        stored signatures (feature dims 0-9) to find the top-k most similar
        signatures per detection, then applies signal-proportional outcome
        adjustment only to those matched signatures.

        This is the refined path (vs apply_to_bcell which adjusts all).

        Args:
            b_cell: BCell instance with numpy memory bank.
            signal: Feedback signal in [-1.0, 1.0].
            top_k: Number of most-similar signatures to adjust per detection.
        """
        if not self._enabled or b_cell is None or signal == 0.0:
            return

        if not hasattr(b_cell, '_memory') or b_cell._memory is None:
            return

        if len(b_cell._memory) == 0:
            return

        recent = self.get_recent_detections()
        if not recent:
            return

        # Track which signature indices have been updated
        updated_indices: set[int] = set()
        n_sigs = len(b_cell._memory)
        stored_features = b_cell._memory[:, :_N_FEATURES]

        for det in recent:
            query = det.features[:_N_FEATURES]
            # Cosine similarity between query and all stored feature vectors
            query_norm = np.linalg.norm(query)
            stored_norms = np.linalg.norm(stored_features, axis=1)
            denominators = query_norm * stored_norms + 1e-8
            similarities = (stored_features @ query) / denominators

            # Top-k most similar
            k = min(top_k, n_sigs)
            top_indices = (
                np.argpartition(-similarities, k)[:k]
                if k < n_sigs
                else np.arange(n_sigs)
            )

            for idx in top_indices:
                updated_indices.add(int(idx))

        if not updated_indices:
            return

        idx_array = np.array(sorted(updated_indices))

        if signal > 0:
            boost = signal * 0.1
            b_cell._memory[idx_array, _OUTCOME_COL] = np.clip(
                b_cell._memory[idx_array, _OUTCOME_COL] + boost, 0.0, 1.0
            )
        else:
            decay = 1.0 + signal * 0.1
            b_cell._memory[idx_array, _OUTCOME_COL] = (
                b_cell._memory[idx_array, _OUTCOME_COL] * max(decay, 0.5)
            )

        logger.debug(
            f"BCell correlated feedback: signal={signal:.4f}, "
            f"updated {len(updated_indices)}/{n_sigs} signatures from "
            f"{len(recent)} detections"
        )

    def apply_to_dca(
        self,
        adaptive_weights,
        signal: float,
        features: np.ndarray | None = None,
    ) -> None:
        """Apply feedback signal as additional DCA weight importance.

        Args:
            adaptive_weights: AdaptiveWeightManager instance.
            signal: Feedback signal in [-1.0, 1.0].
            features: Optional feature vector for feature-specific feedback.
        """
        if not self._enabled or adaptive_weights is None or signal == 0.0:
            return

        if features is not None and hasattr(adaptive_weights, 'adapt'):
            # Use signal as a pseudo-outcome for weight adaptation
            adaptive_weights.adapt(features, signal)
            logger.debug(f"DCA feedback applied: signal={signal:.4f}")
