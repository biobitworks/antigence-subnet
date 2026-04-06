"""
Multi-validator agreement module for miner ranking consensus.

Computes Spearman rank correlation between validators' miner rankings
to verify consensus on miner quality assessments and detect validator-
side scoring anomalies.

Requirements: VHARD-04 (Multi-validator agreement and outlier detection)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class AgreementConfig:
    """Configuration for validator agreement tracking.

    Attributes:
        min_validators: Minimum validators needed for agreement computation.
        correlation_threshold: Below this correlation, a validator is
            considered an outlier.
        max_snapshots: Maximum ranking snapshots stored per validator.
    """

    min_validators: int = 2
    correlation_threshold: float = 0.5
    max_snapshots: int = 100


@dataclass
class RankingSnapshot:
    """A point-in-time snapshot of a validator's miner rankings.

    Attributes:
        validator_hotkey: The validator's hotkey (ss58 address).
        step: The evaluation step when this ranking was produced.
        rankings: Dict mapping miner UID to normalized score [0, 1].
        timestamp: Unix timestamp when the snapshot was created.
    """

    validator_hotkey: str
    step: int
    rankings: dict[int, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgreementResult:
    """Result of pairwise agreement computation between two validators.

    Attributes:
        correlation: Spearman rank correlation coefficient (-1 to 1).
        p_value: Statistical significance of the correlation.
        n_common_uids: Number of UIDs common to both rankings.
        is_significant: Whether p_value < 0.05.
    """

    correlation: float
    p_value: float
    n_common_uids: int
    is_significant: bool


class ValidatorAgreement:
    """Tracks and computes agreement between multiple validators' rankings.

    Stores ranking snapshots per validator and computes pairwise Spearman
    rank correlations to assess consensus on miner quality.

    Attributes:
        config: AgreementConfig with thresholds and limits.
    """

    def __init__(self, config: AgreementConfig | None = None) -> None:
        self.config = config or AgreementConfig()
        self._snapshots: dict[str, list[RankingSnapshot]] = defaultdict(list)

    def record_ranking(self, snapshot: RankingSnapshot) -> None:
        """Store a ranking snapshot from a validator.

        Evicts oldest snapshots when max_snapshots is reached.

        Args:
            snapshot: RankingSnapshot with validator_hotkey and rankings.
        """
        hotkey = snapshot.validator_hotkey
        self._snapshots[hotkey].append(snapshot)
        # Evict oldest if over limit
        if len(self._snapshots[hotkey]) > self.config.max_snapshots:
            self._snapshots[hotkey] = self._snapshots[hotkey][
                -self.config.max_snapshots :
            ]

    def get_latest_ranking(self, validator_hotkey: str) -> RankingSnapshot | None:
        """Return the most recent snapshot for a validator.

        Args:
            validator_hotkey: The validator's hotkey.

        Returns:
            Latest RankingSnapshot or None if no snapshots exist.
        """
        snapshots = self._snapshots.get(validator_hotkey, [])
        return snapshots[-1] if snapshots else None

    @property
    def validator_count(self) -> int:
        """Number of distinct validators with recorded snapshots."""
        return len(self._snapshots)

    @property
    def validator_hotkeys(self) -> list[str]:
        """List of all validator hotkeys with recorded snapshots."""
        return list(self._snapshots.keys())

    def compute_agreement(
        self, validator_a: str, validator_b: str
    ) -> AgreementResult:
        """Compute Spearman rank correlation between two validators.

        Uses the latest ranking snapshot from each validator. Only UIDs
        present in both rankings are compared.

        Args:
            validator_a: Hotkey of the first validator.
            validator_b: Hotkey of the second validator.

        Returns:
            AgreementResult with correlation, p_value, and significance.

        Raises:
            ValueError: If either validator has no recorded snapshots
                or there are fewer than 3 common UIDs.
        """
        snap_a = self.get_latest_ranking(validator_a)
        snap_b = self.get_latest_ranking(validator_b)

        if snap_a is None:
            raise ValueError(f"No rankings recorded for validator {validator_a}")
        if snap_b is None:
            raise ValueError(f"No rankings recorded for validator {validator_b}")

        # Find common UIDs
        common_uids = sorted(
            set(snap_a.rankings.keys()) & set(snap_b.rankings.keys())
        )

        if len(common_uids) < 2:
            return AgreementResult(
                correlation=0.0,
                p_value=1.0,
                n_common_uids=len(common_uids),
                is_significant=False,
            )

        scores_a = np.array([snap_a.rankings[uid] for uid in common_uids])
        scores_b = np.array([snap_b.rankings[uid] for uid in common_uids])

        # Handle constant arrays (all same value) -- Spearman undefined
        if np.all(scores_a == scores_a[0]) or np.all(scores_b == scores_b[0]):
            return AgreementResult(
                correlation=0.0,
                p_value=1.0,
                n_common_uids=len(common_uids),
                is_significant=False,
            )

        rho, p_value = stats.spearmanr(scores_a, scores_b)

        # Handle NaN from scipy (can happen with ties)
        if np.isnan(rho):
            rho = 0.0
        if np.isnan(p_value):
            p_value = 1.0

        return AgreementResult(
            correlation=float(rho),
            p_value=float(p_value),
            n_common_uids=len(common_uids),
            is_significant=bool(p_value < 0.05),
        )

    def get_network_agreement(self) -> float:
        """Compute mean pairwise agreement across all validators.

        Returns:
            Mean Spearman correlation across all validator pairs.
            Returns 0.0 if fewer than 2 validators have data.
        """
        hotkeys = self.validator_hotkeys
        if len(hotkeys) < 2:
            return 0.0

        correlations: list[float] = []
        for i in range(len(hotkeys)):
            for j in range(i + 1, len(hotkeys)):
                try:
                    result = self.compute_agreement(hotkeys[i], hotkeys[j])
                    correlations.append(result.correlation)
                except ValueError:
                    continue

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def detect_outlier_validator(
        self, threshold: float | None = None
    ) -> list[str]:
        """Identify validators whose rankings disagree with the majority.

        A validator is an outlier if its mean pairwise correlation with
        all other validators is below the threshold.

        Args:
            threshold: Correlation threshold (default: config.correlation_threshold).

        Returns:
            List of hotkeys for outlier validators.
        """
        if threshold is None:
            threshold = self.config.correlation_threshold

        hotkeys = self.validator_hotkeys
        if len(hotkeys) < 2:
            return []

        outliers: list[str] = []
        for hotkey in hotkeys:
            others = [h for h in hotkeys if h != hotkey]
            if not others:
                continue

            correlations: list[float] = []
            for other in others:
                try:
                    result = self.compute_agreement(hotkey, other)
                    correlations.append(result.correlation)
                except ValueError:
                    continue

            if correlations:
                mean_corr = float(np.mean(correlations))
                if mean_corr < threshold:
                    outliers.append(hotkey)

        return outliers


def parse_agreement_config(toml_dict: dict) -> AgreementConfig:
    """Parse AgreementConfig from a TOML config dict.

    Expects keys under ``validator.agreement``.

    Args:
        toml_dict: Raw TOML dict (top-level keys).

    Returns:
        AgreementConfig populated from TOML or defaults.
    """
    section = toml_dict.get("validator", {}).get("agreement", {})
    kwargs: dict = {}
    if "min_validators" in section:
        kwargs["min_validators"] = int(section["min_validators"])
    if "correlation_threshold" in section:
        kwargs["correlation_threshold"] = float(section["correlation_threshold"])
    if "max_snapshots" in section:
        kwargs["max_snapshots"] = int(section["max_snapshots"])
    return AgreementConfig(**kwargs)
