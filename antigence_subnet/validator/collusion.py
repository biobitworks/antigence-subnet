"""Validator-side collusion detection for miner response coordination.

Detects groups of miners returning near-identical response patterns by
computing pairwise cosine similarity on shared-sample anomaly scores.
Uses clique-based grouping (all pairwise similarities must exceed threshold).

Per VHARD-05: Detect 3+ miners with cosine similarity > 0.99.
Per VHARD-06: Configurable penalty (default: zero scores), structured alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import bittensor as bt
import numpy as np


@dataclass
class CollusionConfig:
    """Configuration for collusion detection.

    Matches TOML section [validator.collusion]:
        similarity_threshold = 0.99
        min_group_size = 3
        penalty = "zero"
        enabled = true
    """

    similarity_threshold: float = 0.99
    min_group_size: int = 3
    penalty: str = "zero"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> CollusionConfig:
        """Create config from a dict matching TOML structure."""
        return cls(
            similarity_threshold=data.get("similarity_threshold", 0.99),
            min_group_size=data.get("min_group_size", 3),
            penalty=data.get("penalty", "zero"),
            enabled=data.get("enabled", True),
        )


@dataclass
class CollusionAlert:
    """Alert for a detected collusion group.

    Attributes:
        round_num: Evaluation round number.
        colluding_uids: Sorted list of colluding miner UIDs.
        similarity_values: Pairwise cosine similarities within the group,
            keyed by (uid_a, uid_b) tuples where uid_a < uid_b.
        penalty_applied: Penalty type applied ("zero").
    """

    round_num: int
    colluding_uids: list[int]
    similarity_values: dict[tuple[int, int], float]
    penalty_applied: str


class CollusionDetector:
    """Detects miner collusion via pairwise cosine similarity on shared samples.

    Uses clique-based grouping: a collusion group requires ALL pairwise
    similarities to exceed the configured threshold (not just connected
    components). This prevents false positives from transitive similarity.
    """

    def __init__(self, config: CollusionConfig | None = None) -> None:
        self._config = config or CollusionConfig()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero norm (avoids division by zero).
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def detect(
        self,
        round_num: int,
        miner_uids: list[int],
        miner_sample_scores: dict[int, dict[str, float]],
    ) -> list[CollusionAlert]:
        """Detect collusion groups from shared-sample score comparison.

        For each miner pair, finds shared sample IDs, builds score vectors
        from shared samples only, and computes cosine similarity. Pairs with
        fewer than 3 shared samples are skipped.

        Groups are clique-based: ALL pairwise similarities in the group must
        exceed the threshold (red-team fix #17).

        Uses a vectorized approach when all miners share the same samples
        (common case), falling back to pairwise comparison otherwise.

        Args:
            round_num: Current evaluation round.
            miner_uids: List of miner UIDs in this round.
            miner_sample_scores: uid -> {sample_id: anomaly_score}.

        Returns:
            List of CollusionAlert for each detected group.
        """
        if not self._config.enabled:
            return []
        if len(miner_uids) < self._config.min_group_size:
            return []

        # Filter to UIDs that have score data
        uid_list = [uid for uid in miner_uids if uid in miner_sample_scores]
        n = len(uid_list)
        if n < self._config.min_group_size:
            return []

        # Check if all miners share the same sample IDs (fast path)
        first_keys = miner_sample_scores[uid_list[0]].keys()
        first_key_count = len(first_keys)
        # Quick length check first (fast rejection), then set comparison
        all_same = all(
            len(miner_sample_scores[uid]) == first_key_count
            and miner_sample_scores[uid].keys() == first_keys
            for uid in uid_list[1:]
        )

        if all_same and first_key_count >= 3:
            sorted_keys = sorted(first_keys)
            # Fast path: vectorized matrix approach with threshold filtering
            above_threshold = self._detect_vectorized(
                uid_list, miner_sample_scores, sorted_keys,
                self._config.similarity_threshold,
            )
        else:
            # Slow path: pairwise with shared sample intersection
            sim_matrix = self._detect_pairwise(uid_list, miner_sample_scores)
            above_threshold = {
                pair: sim
                for pair, sim in sim_matrix.items()
                if sim >= self._config.similarity_threshold
            }

        if not above_threshold:
            return []

        return self._find_cliques(round_num, above_threshold)

    def _detect_vectorized(
        self,
        uid_list: list[int],
        miner_sample_scores: dict[int, dict[str, float]],
        sorted_keys: list[str],
        threshold: float,
    ) -> dict[tuple[int, int], float]:
        """Vectorized pairwise cosine similarity when all miners share samples.

        Returns only pairs above threshold (avoids building full dict).
        """
        # Build (n_uids, n_samples) matrix -- use list comprehension for speed
        rows = []
        for uid in uid_list:
            scores = miner_sample_scores[uid]
            rows.append([scores[k] for k in sorted_keys])
        matrix = np.array(rows, dtype=np.float64)

        # Compute norms and cosine similarity matrix
        norms = np.linalg.norm(matrix, axis=1)
        dot_mat = matrix @ matrix.T
        norm_outer = np.outer(norms, norms)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim_mat = np.where(norm_outer > 0, dot_mat / norm_outer, 0.0)

        # Extract only upper-triangle pairs above threshold
        ri, ci = np.where(np.triu(sim_mat >= threshold, k=1))

        result: dict[tuple[int, int], float] = {}
        for idx in range(len(ri)):
            i, j = int(ri[idx]), int(ci[idx])
            uid_i, uid_j = uid_list[i], uid_list[j]
            pair_key = (min(uid_i, uid_j), max(uid_i, uid_j))
            result[pair_key] = float(sim_mat[i, j])
        return result

    def _detect_pairwise(
        self,
        uid_list: list[int],
        miner_sample_scores: dict[int, dict[str, float]],
    ) -> dict[tuple[int, int], float]:
        """Pairwise cosine similarity with per-pair shared sample intersection."""
        n = len(uid_list)
        result: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                uid_i, uid_j = uid_list[i], uid_list[j]
                scores_i = miner_sample_scores[uid_i]
                scores_j = miner_sample_scores[uid_j]
                shared = sorted(set(scores_i.keys()) & set(scores_j.keys()))
                if len(shared) < 3:
                    continue
                vec_i = np.array([scores_i[k] for k in shared])
                vec_j = np.array([scores_j[k] for k in shared])
                sim = self._cosine_similarity(vec_i, vec_j)
                pair_key = (min(uid_i, uid_j), max(uid_i, uid_j))
                result[pair_key] = sim
        return result

    def _find_cliques(
        self,
        round_num: int,
        above_threshold: dict[tuple[int, int], float],
    ) -> list[CollusionAlert]:
        """Find clique-based collusion groups from above-threshold pairs."""
        # Build adjacency from above-threshold pairs
        neighbors: dict[int, set[int]] = {}
        for a, b in above_threshold:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

        # Greedy clique expansion from each candidate
        alerts: list[CollusionAlert] = []
        used: set[int] = set()
        candidate_uids = sorted(neighbors.keys())

        for seed in candidate_uids:
            if seed in used:
                continue
            # Start with seed, greedily add neighbors that form a clique
            clique = {seed}
            for candidate in candidate_uids:
                if candidate in used or candidate in clique:
                    continue
                # Check candidate has above-threshold similarity with ALL clique members
                fits = True
                for member in clique:
                    pair = (min(candidate, member), max(candidate, member))
                    if pair not in above_threshold:
                        fits = False
                        break
                if fits:
                    clique.add(candidate)

            if len(clique) >= self._config.min_group_size:
                # Collect pairwise similarities within the clique
                clique_sims: dict[tuple[int, int], float] = {}
                for a, b in combinations(sorted(clique), 2):
                    pair = (min(a, b), max(a, b))
                    if pair in above_threshold:
                        clique_sims[pair] = round(above_threshold[pair], 4)

                alerts.append(
                    CollusionAlert(
                        round_num=round_num,
                        colluding_uids=sorted(clique),
                        similarity_values=clique_sims,
                        penalty_applied=self._config.penalty,
                    )
                )
                used |= clique

        return alerts

    def apply_penalty(
        self,
        rewards: np.ndarray,
        miner_uids: list[int],
        alerts: list[CollusionAlert],
    ) -> np.ndarray:
        """Zero rewards for colluding UIDs (in-place modification).

        Args:
            rewards: Reward array to modify.
            miner_uids: Ordered list of miner UIDs matching rewards indices.
            alerts: Collusion alerts from detect().

        Returns:
            The same rewards array (modified in-place).
        """
        uid_to_idx = {uid: i for i, uid in enumerate(miner_uids)}
        for alert in alerts:
            for uid in alert.colluding_uids:
                if uid in uid_to_idx:
                    rewards[uid_to_idx[uid]] = 0.0
        return rewards

    def log_alerts(self, alerts: list[CollusionAlert]) -> None:
        """Log structured warnings for each collusion alert via bt.logging."""
        for alert in alerts:
            max_sim = max(alert.similarity_values.values()) if alert.similarity_values else 0.0
            bt.logging.warning(
                f"Collusion detected | round={alert.round_num} | "
                f"uids={alert.colluding_uids} | "
                f"max_similarity={max_sim:.4f} | "
                f"penalty={alert.penalty_applied}"
            )
