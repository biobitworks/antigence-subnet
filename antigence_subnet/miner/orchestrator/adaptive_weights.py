"""Adaptive DCA weight manager with EMA-based learning and bounded constraints.

Implements per-feature importance tracking via exponential moving average (EMA).
After each evaluation round with a known outcome, computes per-feature importance
as |feature_value * outcome_signal| and updates weights via EMA blending.

Weight bounds enforced: [min_weight, max_weight] per feature. After clamping,
each category (PAMP, Danger, Safe) is re-normalized to preserve the original
category sum proportions.

Persistence: adapted weights save to JSON and reload across miner restarts.
Cold start: new miners start with default static weights from DendriticCell.

Design decisions:
- D-01: EMA importance = |feature * outcome|, alpha blending
- D-02: Clamp [0.05, 0.5], renormalize within category
- D-03: Persist to ~/.bittensor/neurons/dca_weights/<domain>.json
- D-05: Cold start = default weights, adapt after first round
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

import numpy as np

from antigence_subnet.miner.orchestrator.dendritic_cell import (
    DANGER_FEATURES,
    PAMP_FEATURES,
    SAFE_FEATURES,
)

logger = logging.getLogger(__name__)

# Default persistence path (D-03)
_DEFAULT_BASE_PATH = os.path.join(
    os.path.expanduser("~"), ".bittensor", "neurons", "dca_weights"
)


class AdaptiveWeightManager:
    """EMA-based adaptive weight manager for DCA signal features.

    Updates weights based on detection outcomes so features that are most
    predictive for each domain get emphasized automatically.

    Attributes:
        _alpha: EMA blending rate. Higher = faster adaptation.
        _min_weight: Minimum allowed weight per feature (prevents degeneration).
        _max_weight: Maximum allowed weight per feature (prevents dominance).
        _round_count: Number of adapt() calls since creation or last load.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_weight: float = 0.05,
        max_weight: float = 0.5,
    ) -> None:
        """Initialize with default weights from DendriticCell constants.

        Args:
            alpha: EMA blending rate in (0.0, 1.0]. Default 0.1.
            min_weight: Minimum weight per feature. Default 0.05.
            max_weight: Maximum weight per feature. Default 0.5.
        """
        self._alpha = alpha
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._round_count = 0

        # Deep copy so mutations don't affect module-level constants
        self._pamp: dict[str, tuple[int, float]] = copy.deepcopy(PAMP_FEATURES)
        self._danger: dict[str, tuple[int, float]] = copy.deepcopy(DANGER_FEATURES)
        self._safe: dict[str, tuple[int, float]] = copy.deepcopy(SAFE_FEATURES)

        # Cache original category sums for re-normalization
        self._pamp_sum = sum(w for _, w in PAMP_FEATURES.values())
        self._danger_sum = sum(w for _, w in DANGER_FEATURES.values())
        self._safe_sum = sum(w for _, w in SAFE_FEATURES.values())

    def adapt(self, features: np.ndarray, outcome: float) -> None:
        """Update weights based on feature importance from detection outcome.

        Per D-01: For each feature, importance = |features[idx] * outcome|.
        Per D-01: EMA update: new_weight = alpha * importance + (1 - alpha) * old_weight.
        Per D-02: Clamp to [min_weight, max_weight], then re-normalize within
            each category to preserve original category sum.

        Args:
            features: Feature vector of shape (10,) from DendriticFeatureExtractor.
            outcome: Detection outcome signal. Positive = true positive confirmed,
                negative = false positive or missed detection.
        """
        self._update_category(self._pamp, features, outcome, self._pamp_sum)
        self._update_category(self._danger, features, outcome, self._danger_sum)
        self._update_category(self._safe, features, outcome, self._safe_sum)
        self._round_count += 1

    def _update_category(
        self,
        category: dict[str, tuple[int, float]],
        features: np.ndarray,
        outcome: float,
        target_sum: float,
    ) -> None:
        """Update weights for one signal category (PAMP/Danger/Safe).

        Steps:
        1. Compute importance for each feature
        2. EMA blend with current weight
        3. Clamp to [min_weight, max_weight]
        4. Re-normalize so category weights sum to target_sum

        Args:
            category: Mutable dict mapping name -> (idx, weight).
            features: Full 10-dim feature vector.
            outcome: Detection outcome signal.
            target_sum: Original sum for this category (for re-normalization).
        """
        # Step 1-2: EMA update
        updated: dict[str, tuple[int, float]] = {}
        for name, (idx, old_weight) in category.items():
            importance = abs(float(features[idx]) * outcome)
            new_weight = self._alpha * importance + (1.0 - self._alpha) * old_weight
            updated[name] = (idx, new_weight)

        # Single-feature categories: weight must equal target_sum (no redistribution possible)
        if len(updated) == 1:
            name, (idx, _) = next(iter(updated.items()))
            updated[name] = (idx, target_sum)
            category.update(updated)
            return

        # Step 3: Clamp (multi-feature categories only)
        for name, (idx, w) in updated.items():
            clamped = max(self._min_weight, min(self._max_weight, w))
            updated[name] = (idx, clamped)

        # Step 4: Re-normalize to preserve category sum, iterating clamp+normalize
        # until convergence (typically 2-3 iterations)
        for _ in range(5):
            current_sum = sum(w for _, w in updated.values())
            if current_sum <= 0 or abs(current_sum - target_sum) < 1e-12:
                break
            scale = target_sum / current_sum
            for name, (idx, w) in updated.items():
                new_w = max(self._min_weight, min(self._max_weight, w * scale))
                updated[name] = (idx, new_w)
        else:
            # All weights at zero (shouldn't happen with min_weight > 0)
            # Distribute equally
            equal_w = target_sum / len(updated) if updated else 0.0
            for name, (idx, _) in updated.items():
                updated[name] = (idx, equal_w)

        # Write back
        category.update(updated)

    def get_weights(self) -> dict[str, dict[str, tuple[int, float]]]:
        """Return current weights as deep copy (same shape DendriticCell expects).

        Returns:
            Dict with keys "pamp", "danger", "safe", each mapping
            feature_name -> (index, weight).
        """
        return {
            "pamp": copy.deepcopy(self._pamp),
            "danger": copy.deepcopy(self._danger),
            "safe": copy.deepcopy(self._safe),
        }

    def get_round_count(self) -> int:
        """Return number of adapt() calls since creation or last load.

        Returns:
            Count of adaptation rounds.
        """
        return self._round_count

    def save(self, domain: str, base_path: str | None = None) -> None:
        """Persist adapted weights to JSON file.

        Per D-03: Default path = ~/.bittensor/neurons/dca_weights/<domain>.json.
        Uses atomic write (write to .tmp, os.replace) to prevent corruption.

        Args:
            domain: Domain name (e.g., 'hallucination', 'code_security').
            base_path: Override base directory. If None, uses default.
        """
        if base_path is None:
            base_path = _DEFAULT_BASE_PATH

        os.makedirs(base_path, exist_ok=True)
        fpath = os.path.join(base_path, f"{domain}.json")
        tmp_path = fpath + ".tmp"

        data: dict[str, Any] = {
            "pamp": {name: [idx, w] for name, (idx, w) in self._pamp.items()},
            "danger": {name: [idx, w] for name, (idx, w) in self._danger.items()},
            "safe": {name: [idx, w] for name, (idx, w) in self._safe.items()},
            "round_count": self._round_count,
            "alpha": self._alpha,
        }

        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)

        os.replace(tmp_path, fpath)
        logger.info("Saved adaptive weights for domain '%s' to %s", domain, fpath)

    def load(self, domain: str, base_path: str | None = None) -> bool:
        """Load adapted weights from JSON file.

        Per D-05: If file doesn't exist, keep defaults, return False.

        Args:
            domain: Domain name to load weights for.
            base_path: Override base directory. If None, uses default.

        Returns:
            True if weights were loaded, False if file not found (cold start).
        """
        if base_path is None:
            base_path = _DEFAULT_BASE_PATH

        fpath = os.path.join(base_path, f"{domain}.json")

        if not os.path.exists(fpath):
            logger.info(
                "No adaptive weights for domain '%s' at %s -- using defaults",
                domain,
                fpath,
            )
            return False

        with open(fpath) as f:
            data = json.load(f)

        # Rebuild weight dicts from JSON (lists -> tuples)
        for name, val in data.get("pamp", {}).items():
            if name in self._pamp:
                self._pamp[name] = (int(val[0]), float(val[1]))

        for name, val in data.get("danger", {}).items():
            if name in self._danger:
                self._danger[name] = (int(val[0]), float(val[1]))

        for name, val in data.get("safe", {}).items():
            if name in self._safe:
                self._safe[name] = (int(val[0]), float(val[1]))

        self._round_count = data.get("round_count", 0)

        logger.info(
            "Loaded adaptive weights for domain '%s' from %s (round %d)",
            domain,
            fpath,
            self._round_count,
        )
        return True
