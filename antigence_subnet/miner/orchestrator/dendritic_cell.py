"""Deterministic Dendritic Cell Algorithm (dDCA) for signal classification and tier routing.

Classifies 10 dendritic features into PAMP/Danger/Safe signal categories,
computes a maturation state (immature/semi-mature/mature), and recommends a
detector tier for the ImmuneOrchestrator.

The DCA is the second tier of the immune pipeline -- after NK Cell fast-path.
It determines how suspicious an input is and routes it to the appropriate
detector set (lightweight vs full ensemble), reducing unnecessary computation
for normal-looking inputs.

Feature-to-category mapping (per D-01):
- PAMP signals:   pamp_score [5] -- pathogen-associated molecular pattern
- Danger signals: exaggeration [6], controversy [8], claim_density [0]
- Safe signals:   citation_count [1], hedging_ratio [2], numeric_density [4],
                  certainty [7], specificity [3]
- Excluded:       danger_signal [9] -- r=1.0 with pamp_score, would double-count

Maturation logic (per D-03):
- immature:    safe > pamp + danger (normal-looking)
- mature:      pamp >= pamp_threshold (clear pathogen pattern, default 0.3)
- semi-mature: everything else (suspicious but not alarming)

Tier routing (per D-04):
- immature    -> ["ocsvm", "negsel"] (standard 2-detector baseline for normal inputs)
- semi-mature -> ["ocsvm", "negsel"] (two-detector confirmation)
- mature      -> [] (empty = full ensemble, all registered detectors)

Fully deterministic (DCA-03): no random elements, same input always produces
the same DCAResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from antigence_subnet.miner.detector import DetectionResult

if TYPE_CHECKING:
    from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager

# ---------------------------------------------------------------------------
# Signal category constants (D-01)
# Maps feature_name -> (index_in_10dim_vector, weight)
# ---------------------------------------------------------------------------

PAMP_FEATURES: dict[str, tuple[int, float]] = {
    "pamp_score": (5, 1.0),
}

DANGER_FEATURES: dict[str, tuple[int, float]] = {
    "exaggeration": (6, 1.0 / 3.0),
    "controversy": (8, 1.0 / 3.0),
    "claim_density": (0, 1.0 / 3.0),
}

SAFE_FEATURES: dict[str, tuple[int, float]] = {
    "citation_count": (1, 0.2),
    "hedging_ratio": (2, 0.2),
    "numeric_density": (4, 0.2),
    "certainty": (7, 0.2),
    "specificity": (3, 0.2),
}

EXCLUDED_FEATURES: set[str] = {"danger_signal"}

# ---------------------------------------------------------------------------
# Tier routing map (D-04)
# ---------------------------------------------------------------------------

TIER_MAP: dict[str, list[str]] = {
    "immature": ["ocsvm", "negsel"],
    "semi-mature": ["ocsvm", "negsel"],
    "mature": [],
}


# ---------------------------------------------------------------------------
# DCAResult (D-06)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCAResult:
    """Result of DCA signal classification and tier routing.

    Attributes:
        maturation_state: One of "immature", "semi-mature", "mature".
        signal_scores: Dict with keys "pamp", "danger", "safe", each in [0.0, 1.0].
        recommended_tier: List of detector keys for the selected tier.
            Empty list means full ensemble (all registered detectors).
    """

    maturation_state: str
    signal_scores: dict[str, float]
    recommended_tier: list[str]


# ---------------------------------------------------------------------------
# DendriticCell
# ---------------------------------------------------------------------------

class DendriticCell:
    """Deterministic Dendritic Cell Algorithm (dDCA) for signal classification.

    Classifies 10-dim dendritic features into PAMP/Danger/Safe signal scores,
    determines maturation state, and recommends a detector tier.

    Satisfies ImmuneCellType Protocol via duck typing (process() method).
    process() always returns None because DCA is a router, not a detector.
    Use classify() to get the DCAResult routing recommendation.
    """

    def __init__(
        self,
        pamp_threshold: float = 0.3,
        signal_weights: dict[str, dict[str, tuple[int, float]]] | None = None,
        tier_map: dict[str, list[str]] | None = None,
        weight_manager: AdaptiveWeightManager | None = None,
    ) -> None:
        """Initialize DendriticCell with classification parameters.

        Args:
            pamp_threshold: Threshold for mature maturation state.
                Features with pamp_score >= pamp_threshold are classified
                as mature (clear pathogen pattern). Default 0.3.
            signal_weights: Optional override dict with keys "pamp", "danger",
                "safe", each mapping feature_name -> (index, weight).
                If None, uses module-level defaults.
            tier_map: Optional override dict mapping maturation_state ->
                list[str] of detector keys. If None, uses TIER_MAP default.
            weight_manager: Optional AdaptiveWeightManager for dynamic weight
                adaptation (Phase 44). When set, classify() refreshes weights
                from the manager before each classification.
        """
        self._pamp_threshold = pamp_threshold

        if signal_weights is not None:
            self._pamp_weights = signal_weights.get("pamp", PAMP_FEATURES)
            self._danger_weights = signal_weights.get("danger", DANGER_FEATURES)
            self._safe_weights = signal_weights.get("safe", SAFE_FEATURES)
        else:
            self._pamp_weights = PAMP_FEATURES
            self._danger_weights = DANGER_FEATURES
            self._safe_weights = SAFE_FEATURES

        self._tier_map = tier_map if tier_map is not None else TIER_MAP
        self._weight_manager = weight_manager

    @classmethod
    def from_config(
        cls,
        dca_config: dict[str, Any],
        weight_manager: AdaptiveWeightManager | None = None,
    ) -> DendriticCell:
        """Create DendriticCell from a dca_config dict (e.g., from OrchestratorConfig).

        Args:
            dca_config: Configuration dict with optional keys:
                - pamp_threshold (float): Threshold for mature state. Default 0.3.
                - signal_weights (dict): Override signal category weights.
                - tier_map (dict): Override tier routing map.
            weight_manager: Optional AdaptiveWeightManager for dynamic weight
                adaptation (Phase 44). Passed through to __init__.

        Returns:
            DendriticCell configured from the provided dict.
        """
        pamp_threshold = dca_config.get("pamp_threshold")
        if pamp_threshold is None:
            pamp_threshold = 0.3

        signal_weights = dca_config.get("signal_weights")
        if signal_weights is not None:
            # Ensure it's None if explicitly set to None in config
            signal_weights = signal_weights

        tier_map = dca_config.get("tier_map")
        if tier_map is not None:
            tier_map = tier_map

        return cls(
            pamp_threshold=pamp_threshold,
            signal_weights=signal_weights,
            tier_map=tier_map,
            weight_manager=weight_manager,
        )

    def refresh_weights(self) -> None:
        """Update signal weights from the AdaptiveWeightManager if attached.

        Called automatically by classify() before each classification so that
        the DendriticCell always uses the latest adapted weights. No-op when
        weight_manager is None.
        """
        if self._weight_manager is None:
            return
        weights = self._weight_manager.get_weights()
        self._pamp_weights = weights.get("pamp", self._pamp_weights)
        self._danger_weights = weights.get("danger", self._danger_weights)
        self._safe_weights = weights.get("safe", self._safe_weights)

    def classify_signals(self, features: np.ndarray) -> dict[str, float]:
        """Compute PAMP, Danger, and Safe signal scores from features.

        Each score is the weighted sum of its category's features, clamped
        to [0.0, 1.0]. danger_signal [9] is excluded from all categories.

        Args:
            features: Feature vector of shape (10,) from DendriticFeatureExtractor.

        Returns:
            Dict with keys "pamp", "danger", "safe", each in [0.0, 1.0].
        """
        pamp = self._weighted_sum(features, self._pamp_weights)
        danger = self._weighted_sum(features, self._danger_weights)
        safe = self._weighted_sum(features, self._safe_weights)

        return {
            "pamp": max(0.0, min(1.0, pamp)),
            "danger": max(0.0, min(1.0, danger)),
            "safe": max(0.0, min(1.0, safe)),
        }

    def determine_maturation(
        self,
        signal_scores: dict[str, float],
        *,
        pamp_threshold: float | None = None,
    ) -> str:
        """Determine maturation state from signal scores (D-03 logic).

        Priority order:
        1. mature: pamp >= pamp_threshold (clear pathogen)
        2. immature: safe > pamp + danger (normal-looking)
        3. semi-mature: everything else

        Args:
            signal_scores: Dict with keys "pamp", "danger", "safe".
            pamp_threshold: Optional per-request threshold override.
                When not None, used instead of self._pamp_threshold
                (per-domain config support, Phase 36).

        Returns:
            One of "immature", "semi-mature", "mature".
        """
        effective_threshold = pamp_threshold if pamp_threshold is not None else self._pamp_threshold

        pamp = signal_scores["pamp"]
        danger = signal_scores["danger"]
        safe = signal_scores["safe"]

        if pamp >= effective_threshold:
            return "mature"
        if safe > pamp + danger:
            return "immature"
        return "semi-mature"

    def classify(
        self,
        features: np.ndarray,
        *,
        pamp_threshold: float | None = None,
    ) -> DCAResult:
        """Classify features into signal scores, maturation state, and tier.

        This is the primary DCA method. Returns a DCAResult with routing
        recommendation for the ImmuneOrchestrator.

        Args:
            features: Feature vector of shape (10,) from DendriticFeatureExtractor.
            pamp_threshold: Optional per-request threshold override.
                When not None, passed to determine_maturation() instead of
                using self._pamp_threshold (per-domain config support, Phase 36).

        Returns:
            DCAResult with maturation_state, signal_scores, and recommended_tier.
        """
        self.refresh_weights()
        signal_scores = self.classify_signals(features)
        maturation_state = self.determine_maturation(
            signal_scores, pamp_threshold=pamp_threshold,
        )
        recommended_tier = list(self._tier_map.get(maturation_state, []))

        return DCAResult(
            maturation_state=maturation_state,
            signal_scores=signal_scores,
            recommended_tier=recommended_tier,
        )

    def process(
        self,
        features: np.ndarray,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult | None:
        """ImmuneCellType Protocol method. Always returns None.

        DCA is a signal router, not a detector. It does not produce
        DetectionResults. The ImmuneOrchestrator (Phase 34) calls
        classify() directly to get the DCAResult routing recommendation.

        Args:
            features: Feature vector (shape (10,) from DendriticFeatureExtractor).
            prompt: Original prompt text (unused by DCA).
            output: AI-generated output text (unused by DCA).
            code: Optional code content (unused by DCA).
            context: Optional JSON-serialized metadata (unused by DCA).

        Returns:
            None always. Use classify() for the DCAResult.
        """
        return None

    @staticmethod
    def _weighted_sum(
        features: np.ndarray,
        weight_map: dict[str, tuple[int, float]],
    ) -> float:
        """Compute weighted sum of features for a signal category.

        Args:
            features: Full 10-dim feature vector.
            weight_map: Dict mapping feature_name -> (index, weight).

        Returns:
            Weighted sum (not yet clamped).
        """
        total = 0.0
        for _name, (idx, weight) in weight_map.items():
            total += float(features[idx]) * weight
        return total
