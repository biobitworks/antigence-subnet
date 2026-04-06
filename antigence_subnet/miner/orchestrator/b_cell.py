"""B Cell adaptive memory for the immune orchestrator pipeline.

B Cells maintain a memory bank of detection signatures (10-dim dendritic
features + anomaly_score + outcome). They implement clonal selection:
true-positive signatures are amplified (cloned with jitter), false-positive
signatures decay over time. Memory provides a prior score via k-nearest
neighbor weighted average that combines with ensemble output.

Phase 43 adds dual-mode operation:
- Feature-only mode (Phase 37): kNN with euclidean distance on 10-dim features
- Embedding mode (Phase 43): kNN with cosine similarity on 384-dim SLM embeddings

Design decisions:
- D-01: Memory bank is numpy array shape (N, 12): 10 features + anomaly_score + outcome
- D-02: Embedding mode uses cosine similarity on 384-dim vectors for kNN
- D-03: Missing persistence file = cold start (empty memory)
- D-04: TP cloned 2x with gaussian jitter; FP decayed by half_life
- D-05: Eviction below threshold; max_memory enforced by lowest-outcome
- D-06: kNN prior with weight = 1/distance (brute force, N <= 1000)
- D-07: Cold start: bcell_weight forced to 0.0 (no influence)
- D-08: Satisfies ImmuneCellType Protocol
- D-09: process() returns None always; influence() adjusts ensemble score
- D-10: Embedding sigma (0.01) smaller than feature jitter sigma (0.05)
- D-11: Graceful fallback to feature-only when model_manager unavailable
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from antigence_subnet.miner.detector import DetectionResult

logger = logging.getLogger(__name__)

# Memory array column layout
_N_FEATURES = 10
_ANOMALY_SCORE_COL = 10
_OUTCOME_COL = 11
_N_COLS = 12

# Default embedding dimensionality (all-MiniLM-L6-v2)
_EMBEDDING_DIM = 384

# Small constant to avoid division by zero in kNN weighting
_EPSILON = 1e-8


def _cosine_similarities(query: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and each row of stored.

    Args:
        query: 1-D array of shape (D,).
        stored: 2-D array of shape (N, D).

    Returns:
        1-D array of shape (N,) with cosine similarities in [-1, 1].
    """
    query_norm = np.linalg.norm(query)
    stored_norms = np.linalg.norm(stored, axis=1)
    dot_products = stored @ query
    denominators = query_norm * stored_norms + _EPSILON
    return dot_products / denominators


class BCell:
    """B Cell adaptive memory implementing ImmuneCellType Protocol.

    Maintains a numpy memory bank of detection signatures. Provides
    kNN-based prior scoring and clonal selection for memory refinement.

    Supports dual-mode operation:
    - embedding_mode=False (default): Phase 37 behavior, euclidean kNN on features
    - embedding_mode=True: Phase 43 cosine kNN on 384-dim SLM embeddings

    Satisfies ImmuneCellType Protocol via duck typing: process() accepts
    (features, prompt, output, code, context) and returns DetectionResult | None.
    """

    def __init__(
        self,
        max_memory: int = 1000,
        k: int = 5,
        bcell_weight: float = 0.2,
        half_life: float = 0.9,
        eviction_threshold: float = 0.1,
        jitter_sigma: float = 0.05,
        embedding_mode: bool = False,
        model_manager: object | None = None,
        embedding_sigma: float = 0.01,
    ) -> None:
        """Initialize BCell with configurable parameters.

        Args:
            max_memory: Maximum number of signatures to retain.
            k: Number of nearest neighbors for prior scoring.
            bcell_weight: Weight for combining prior with ensemble score.
                Forced to 0.0 when memory is empty (cold start).
            half_life: Decay factor applied to FP outcomes each round.
            eviction_threshold: Minimum outcome to avoid eviction.
            jitter_sigma: Standard deviation for gaussian jitter on feature clones.
            embedding_mode: When True, stores/queries 384-dim embeddings
                alongside features. Requires model_manager.
            model_manager: Optional ModelManager (typed as object to avoid
                hard import dependency; only needs embed() method).
            embedding_sigma: Standard deviation for gaussian noise on
                embedding dims during clonal selection (per D-10).
        """
        self._max_memory = max_memory
        self._k = k
        self._bcell_weight = bcell_weight
        self._half_life = half_life
        self._eviction_threshold = eviction_threshold
        self._jitter_sigma = jitter_sigma
        self._embedding_sigma = embedding_sigma
        self._memory: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

        # Resolve embedding mode with graceful fallback (per D-11)
        if embedding_mode and model_manager is None:
            logger.warning(
                "BCell embedding_mode=True but model_manager is None; "
                "falling back to feature-only mode"
            )
            self._embedding_mode = False
            self._model_manager = None
        else:
            self._embedding_mode = embedding_mode
            self._model_manager = model_manager

    def process(
        self,
        features: np.ndarray,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult | None:
        """Satisfy ImmuneCellType Protocol. Always returns None.

        BCell does not gate detection. Its influence is applied separately
        via influence() after the ensemble pipeline (per D-09).

        Args:
            features: 10-dim dendritic feature vector.
            prompt: Original prompt text.
            output: AI-generated output text.
            code: Optional code content.
            context: Optional JSON-serialized metadata.

        Returns:
            None (BCell never gates; influence applied separately).
        """
        return None

    def store_signature(
        self,
        features: np.ndarray,
        anomaly_score: float,
        ground_truth: float,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Store a detection signature in the memory bank.

        Appends a row [features(10) | anomaly_score | outcome] to memory.
        If at max_memory, evicts the lowest-outcome row first.

        In embedding mode, also stores the 384-dim embedding alongside.
        If embedding is None in embedding mode, stores a zero-vector
        placeholder to keep arrays aligned.

        Args:
            features: 10-dim dendritic feature vector.
            anomaly_score: Detection anomaly score (0.0-1.0).
            ground_truth: Outcome label (1.0=TP, 0.0=FP, or continuous).
            embedding: Optional 384-dim embedding vector (Phase 43).
        """
        row = np.zeros((_N_COLS,), dtype=np.float64)
        row[:_N_FEATURES] = features[:_N_FEATURES]
        row[_ANOMALY_SCORE_COL] = anomaly_score
        row[_OUTCOME_COL] = ground_truth

        # Resolve embedding for this store
        emb_row: np.ndarray | None = None
        if self._embedding_mode:
            if embedding is not None:
                emb_row = np.asarray(embedding, dtype=np.float32).ravel()
            else:
                emb_row = np.zeros((_EMBEDDING_DIM,), dtype=np.float32)

        if self._memory is None:
            self._memory = row.reshape(1, _N_COLS)
            if self._embedding_mode and emb_row is not None:
                self._embeddings = emb_row.reshape(1, _EMBEDDING_DIM)
        else:
            # Evict lowest-outcome if at capacity
            if len(self._memory) >= self._max_memory:
                worst_idx = int(np.argmin(self._memory[:, _OUTCOME_COL]))
                self._memory = np.delete(self._memory, worst_idx, axis=0)
                if self._embedding_mode and self._embeddings is not None:
                    self._embeddings = np.delete(self._embeddings, worst_idx, axis=0)
            self._memory = np.vstack([self._memory, row])
            if self._embedding_mode and emb_row is not None:
                if self._embeddings is None:
                    self._embeddings = emb_row.reshape(1, _EMBEDDING_DIM)
                else:
                    self._embeddings = np.vstack([self._embeddings, emb_row])

    def prior_score(
        self,
        features: np.ndarray,
        embedding: np.ndarray | None = None,
    ) -> float:
        """Compute prior score via kNN weighted average from stored signatures.

        In embedding mode with valid embedding: uses cosine similarity on
        384-dim embeddings (higher similarity = closer neighbor).
        In feature-only mode: uses euclidean distance on feature dims (cols 0-9).
        Returns 0.5 (neutral) if memory is empty.

        Args:
            features: 10-dim query feature vector.
            embedding: Optional 384-dim query embedding (Phase 43).

        Returns:
            Prior anomaly score in [0.0, 1.0]. 0.5 if memory empty.
        """
        if self._memory is None or len(self._memory) == 0:
            return 0.5

        use_cosine = (
            self._embedding_mode
            and embedding is not None
            and self._embeddings is not None
            and len(self._embeddings) > 0
        )

        if use_cosine:
            return self._prior_score_cosine(embedding)
        else:
            return self._prior_score_euclidean(features)

    def _prior_score_euclidean(self, features: np.ndarray) -> float:
        """kNN prior using euclidean distance on feature dims (Phase 37)."""
        stored_features = self._memory[:, :_N_FEATURES]
        diffs = stored_features - features[:_N_FEATURES]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        k = min(self._k, len(distances))
        nearest_indices = (
            np.argpartition(distances, k)[:k]
            if k < len(distances)
            else np.arange(len(distances))
        )

        nearest_distances = distances[nearest_indices]
        nearest_scores = self._memory[nearest_indices, _ANOMALY_SCORE_COL]

        weights = 1.0 / (nearest_distances + _EPSILON)
        weighted_sum = np.sum(weights * nearest_scores)
        total_weight = np.sum(weights)

        return float(weighted_sum / total_weight)

    def _prior_score_cosine(self, embedding: np.ndarray) -> float:
        """kNN prior using cosine similarity on embeddings (Phase 43).

        Higher cosine similarity = closer neighbor. Uses top-k by similarity.
        Weight = max(cosine_similarity, 0) + epsilon to avoid negative weights.
        """
        query = np.asarray(embedding, dtype=np.float32).ravel()
        similarities = _cosine_similarities(query, self._embeddings)

        k = min(self._k, len(similarities))
        # Top-k by highest similarity (argpartition for largest)
        if k < len(similarities):
            nearest_indices = np.argpartition(-similarities, k)[:k]
        else:
            nearest_indices = np.arange(len(similarities))

        nearest_sims = similarities[nearest_indices]
        nearest_scores = self._memory[nearest_indices, _ANOMALY_SCORE_COL]

        # Weight = max(sim, 0) + epsilon to avoid negative/zero weights
        weights = np.maximum(nearest_sims, 0.0) + _EPSILON
        weighted_sum = np.sum(weights * nearest_scores)
        total_weight = np.sum(weights)

        return float(weighted_sum / total_weight)

    def influence(
        self,
        features: np.ndarray,
        ensemble_result: DetectionResult,
        embedding: np.ndarray | None = None,
    ) -> DetectionResult:
        """Combine ensemble score with B Cell prior via weighted average.

        Per D-07: new_score = (1 - bcell_weight) * ensemble_score + bcell_weight * prior
        Cold start (empty memory): returns ensemble_result unchanged
        (bcell_weight forced to 0.0).

        Args:
            features: 10-dim dendritic feature vector.
            ensemble_result: DetectionResult from the ensemble pipeline.
            embedding: Optional 384-dim embedding (Phase 43).

        Returns:
            New DetectionResult with adjusted score, same confidence/anomaly_type/attribution.
        """
        if self._memory is None or len(self._memory) == 0:
            # Cold start: no influence
            return DetectionResult(
                score=ensemble_result.score,
                confidence=ensemble_result.confidence,
                anomaly_type=ensemble_result.anomaly_type,
                feature_attribution=ensemble_result.feature_attribution,
            )

        prior = self.prior_score(features, embedding=embedding)
        new_score = (1.0 - self._bcell_weight) * ensemble_result.score + self._bcell_weight * prior

        return DetectionResult(
            score=new_score,
            confidence=ensemble_result.confidence,
            anomaly_type=ensemble_result.anomaly_type,
            feature_attribution=ensemble_result.feature_attribution,
        )

    def clonal_selection(self) -> None:
        """Apply clonal selection to the memory bank.

        Per D-04/D-05:
        - TP signatures (outcome >= 0.5): cloned 2x with gaussian jitter
          on feature dims only, clamped to [0, 1].
        - FP signatures (outcome < 0.5): outcome decayed by half_life.
        - Signatures with outcome < eviction_threshold are removed.
        - After cloning, enforce max_memory by evicting lowest-outcome.

        In embedding mode (Phase 43):
        - TP clone embeddings also get gaussian noise with embedding_sigma (0.01).
        - Eviction and max_memory enforcement also remove corresponding embeddings.
        """
        if self._memory is None or len(self._memory) == 0:
            return

        # Step 1: Decay FP outcomes
        fp_mask = self._memory[:, _OUTCOME_COL] < 0.5
        self._memory[fp_mask, _OUTCOME_COL] *= self._half_life

        # Step 2: Evict below threshold
        keep_mask = self._memory[:, _OUTCOME_COL] >= self._eviction_threshold
        self._memory = self._memory[keep_mask]
        if self._embedding_mode and self._embeddings is not None:
            self._embeddings = self._embeddings[keep_mask]

        if len(self._memory) == 0:
            self._memory = None
            if self._embedding_mode:
                self._embeddings = None
            return

        # Step 3: Clone TP signatures 2x with jitter
        tp_mask = self._memory[:, _OUTCOME_COL] >= 0.5
        tp_sigs = self._memory[tp_mask]
        tp_embeddings = None
        if self._embedding_mode and self._embeddings is not None:
            tp_embeddings = self._embeddings[tp_mask]

        clones = []
        clone_embeddings = []
        for _ in range(2):
            for i, sig in enumerate(tp_sigs):
                clone = sig.copy()
                # Apply gaussian jitter to feature dims only (cols 0-9)
                jitter = np.random.normal(0, self._jitter_sigma, _N_FEATURES)
                clone[:_N_FEATURES] = np.clip(clone[:_N_FEATURES] + jitter, 0.0, 1.0)
                clones.append(clone)

                # Embedding jitter with embedding_sigma
                if self._embedding_mode and tp_embeddings is not None:
                    emb_clone = tp_embeddings[i].copy()
                    emb_jitter = np.random.normal(
                        0, self._embedding_sigma, emb_clone.shape[0]
                    ).astype(np.float32)
                    emb_clone = emb_clone + emb_jitter
                    clone_embeddings.append(emb_clone)

        if clones:
            clone_array = np.array(clones)
            self._memory = np.vstack([self._memory, clone_array])
            if self._embedding_mode and clone_embeddings:
                emb_clone_array = np.array(clone_embeddings)
                if self._embeddings is not None:
                    self._embeddings = np.vstack([self._embeddings, emb_clone_array])
                else:
                    self._embeddings = emb_clone_array

        # Step 4: Enforce max_memory by evicting lowest-outcome
        if len(self._memory) > self._max_memory:
            # Keep top max_memory by outcome
            sorted_indices = np.argsort(self._memory[:, _OUTCOME_COL])
            keep_indices = sorted_indices[-self._max_memory:]
            self._memory = self._memory[keep_indices]
            if self._embedding_mode and self._embeddings is not None:
                self._embeddings = self._embeddings[keep_indices]

    def save_memory(self, path: str) -> None:
        """Save memory bank to .npz file.

        No-op if memory is empty. In embedding mode, includes
        embeddings array in the .npz.

        Args:
            path: File path for the .npz file.
        """
        if self._memory is None:
            return
        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        save_kwargs: dict[str, np.ndarray] = {"memory": self._memory}
        if self._embeddings is not None:
            save_kwargs["embeddings"] = self._embeddings

        np.savez(path, **save_kwargs)

    def load_memory(self, path: str) -> None:
        """Load memory bank from .npz file.

        If file doesn't exist, memory stays empty (cold start per D-03).
        Validates that loaded array has correct shape (N, 12).
        In embedding mode, also restores embeddings if present.

        Args:
            path: File path for the .npz file.
        """
        if not os.path.exists(path):
            return
        try:
            data = np.load(path)
            memory = data["memory"]
            if memory.ndim == 2 and memory.shape[1] == _N_COLS:
                self._memory = memory
            else:
                logger.warning(
                    "B Cell memory file %s has invalid shape %s (expected (N, %d)), ignoring",
                    path,
                    memory.shape,
                    _N_COLS,
                )
                return

            # Restore embeddings if present and in embedding mode
            if self._embedding_mode and "embeddings" in data.files:
                self._embeddings = data["embeddings"]
        except Exception:
            logger.warning("Failed to load B Cell memory from %s", path, exc_info=True)

    @classmethod
    def from_config(cls, bcell_config: dict[str, Any]) -> BCell:
        """Create BCell from a bcell_config dict (from TOML).

        Same pattern as DangerTheoryModulator.from_config().
        Note: model_manager is NOT passed via from_config -- it's wired
        externally by the orchestrator (Plan 02).

        Args:
            bcell_config: Configuration dict with optional keys:
                max_memory, k, bcell_weight, half_life,
                eviction_threshold, jitter_sigma,
                embedding_mode, embedding_sigma.

        Returns:
            BCell configured from the provided dict.
        """
        return cls(
            max_memory=bcell_config.get("max_memory", 1000),
            k=bcell_config.get("k", 5),
            bcell_weight=bcell_config.get("bcell_weight", 0.2),
            half_life=bcell_config.get("half_life", 0.9),
            eviction_threshold=bcell_config.get("eviction_threshold", 0.1),
            jitter_sigma=bcell_config.get("jitter_sigma", 0.05),
            embedding_mode=bcell_config.get("embedding_mode", False),
            embedding_sigma=bcell_config.get("embedding_sigma", 0.01),
        )

    @property
    def memory_size(self) -> int:
        """Return number of signatures in memory bank.

        Returns:
            0 if empty, else number of stored signatures.
        """
        if self._memory is None:
            return 0
        return len(self._memory)
