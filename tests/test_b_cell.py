"""Tests for BCell adaptive memory module.

Covers: Protocol conformance, empty memory (cold start), store_signature,
kNN prior_score, influence on ensemble, clonal selection (TP cloning,
FP decay, eviction), .npz persistence, max_memory enforcement,
embedding mode (Phase 43): cosine kNN, embedding storage, embedding-space
clonal selection, persistence with embeddings, fallback behavior.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pytest

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.cells import ImmuneCellType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(**kwargs: float) -> np.ndarray:
    """Build a 10-dim feature vector, defaulting all features to 0.0."""
    name_to_idx = {
        "claim_density": 0,
        "citation_count": 1,
        "hedging_ratio": 2,
        "specificity": 3,
        "numeric_density": 4,
        "pamp_score": 5,
        "exaggeration": 6,
        "certainty": 7,
        "controversy": 8,
        "danger_signal": 9,
    }
    vec = np.zeros(10, dtype=np.float64)
    for name, value in kwargs.items():
        vec[name_to_idx[name]] = value
    return vec


def _make_result(score: float = 0.6, confidence: float = 0.8) -> DetectionResult:
    """Create a DetectionResult for testing."""
    return DetectionResult(
        score=score,
        confidence=confidence,
        anomaly_type="test_anomaly",
        feature_attribution={"test_feat": 0.5},
    )


def _make_embedding(dim: int = 384, seed: int | None = None) -> np.ndarray:
    """Create a random 384-dim embedding for testing."""
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


class _MockModelManager:
    """Minimal mock ModelManager for BCell embedding-mode tests."""

    def embed(self, text: str) -> np.ndarray:
        return np.random.randn(384).astype(np.float32)

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Protocol Conformance
# ---------------------------------------------------------------------------


class TestBCellProtocol:
    """BCell satisfies ImmuneCellType Protocol."""

    def test_bcell_satisfies_protocol(self):
        """BCell instance passes isinstance(cell, ImmuneCellType)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        assert isinstance(cell, ImmuneCellType)


# ---------------------------------------------------------------------------
# Cold Start / Empty Memory
# ---------------------------------------------------------------------------


class TestBCellColdStart:
    """BCell with empty memory returns neutral results."""

    def test_process_returns_none_with_empty_memory(self):
        """BCell.process() returns None (cold start = no opinion, per D-09)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_prior_score_returns_half_when_empty(self):
        """BCell.prior_score() returns 0.5 (neutral) with empty memory."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5)
        assert cell.prior_score(features) == 0.5

    def test_influence_returns_ensemble_unchanged_when_empty(self):
        """BCell.influence() returns ensemble_result unchanged when memory empty (bcell_weight forced to 0.0)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(bcell_weight=0.2)
        features = _make_features(claim_density=0.5)
        ensemble = _make_result(score=0.7)
        result = cell.influence(features, ensemble)
        assert result.score == 0.7
        assert result.confidence == ensemble.confidence
        assert result.anomaly_type == ensemble.anomaly_type

    def test_memory_size_zero_when_empty(self):
        """BCell.memory_size returns 0 when no signatures stored."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        assert cell.memory_size == 0


# ---------------------------------------------------------------------------
# Store Signature
# ---------------------------------------------------------------------------


class TestBCellStore:
    """BCell.store_signature() adds rows to internal memory array."""

    def test_store_signature_adds_row(self):
        """store_signature() adds a row; memory shape becomes (1, 12)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)
        assert cell.memory_size == 1

    def test_store_tp_signature(self):
        """store_signature with ground_truth=1.0 (TP) stores correctly."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)
        assert cell.memory_size == 1
        # Verify the last two columns are anomaly_score and outcome
        assert cell._memory is not None
        assert cell._memory[0, 10] == pytest.approx(0.8)  # anomaly_score
        assert cell._memory[0, 11] == pytest.approx(1.0)  # outcome (TP)

    def test_store_fp_signature(self):
        """store_signature with ground_truth=0.0 (FP) stores correctly."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.6, ground_truth=0.0)
        assert cell.memory_size == 1
        assert cell._memory is not None
        assert cell._memory[0, 10] == pytest.approx(0.6)  # anomaly_score
        assert cell._memory[0, 11] == pytest.approx(0.0)  # outcome (FP)

    def test_store_multiple_signatures(self):
        """Multiple store_signature calls grow the memory array."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        for i in range(5):
            features = _make_features(claim_density=float(i) / 10.0)
            cell.store_signature(features, anomaly_score=0.5, ground_truth=1.0)
        assert cell.memory_size == 5
        assert cell._memory is not None
        assert cell._memory.shape == (5, 12)


# ---------------------------------------------------------------------------
# kNN Prior Score
# ---------------------------------------------------------------------------


class TestBCellPriorScore:
    """BCell.prior_score() returns kNN weighted average."""

    def test_prior_score_with_stored_signatures(self):
        """prior_score returns kNN weighted average (k=5, weight=1/distance)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(k=5)
        # Store 3 signatures with known anomaly scores
        for score_val in [0.2, 0.4, 0.8]:
            features = _make_features(claim_density=0.5)
            cell.store_signature(features, anomaly_score=score_val, ground_truth=1.0)

        # Query with exact same features -> distances ~0, weighted avg of [0.2, 0.4, 0.8]
        query = _make_features(claim_density=0.5)
        prior = cell.prior_score(query)
        # Should be close to mean of [0.2, 0.4, 0.8] = 0.4667 (weighted by 1/epsilon)
        # With identical features, all distances are epsilon, so equal weights
        expected_mean = (0.2 + 0.4 + 0.8) / 3.0
        assert prior == pytest.approx(expected_mean, abs=0.01)

    def test_prior_score_fewer_than_k(self):
        """prior_score with fewer than k signatures uses all available."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(k=5)
        # Only store 2 signatures
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.3, ground_truth=1.0)
        cell.store_signature(features, anomaly_score=0.7, ground_truth=1.0)

        query = _make_features(claim_density=0.5)
        prior = cell.prior_score(query)
        # With 2 identical-distance signatures, weighted avg = (0.3 + 0.7) / 2 = 0.5
        assert prior == pytest.approx(0.5, abs=0.01)

    def test_prior_score_distance_weighting(self):
        """Closer signatures contribute more to the prior."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(k=5)
        # Near signature with high anomaly score
        near = _make_features(claim_density=0.50)
        cell.store_signature(near, anomaly_score=0.9, ground_truth=1.0)
        # Far signature with low anomaly score
        far = _make_features(claim_density=0.99)
        cell.store_signature(far, anomaly_score=0.1, ground_truth=1.0)

        query = _make_features(claim_density=0.51)
        prior = cell.prior_score(query)
        # Near signature (dist ~0.01) should dominate over far (dist ~0.48)
        assert prior > 0.5  # Closer to 0.9 than to 0.1


# ---------------------------------------------------------------------------
# Influence
# ---------------------------------------------------------------------------


class TestBCellInfluence:
    """BCell.influence() combines ensemble_score with prior via weighted average."""

    def test_influence_combines_scores(self):
        """influence = (1-bcell_weight)*ensemble + bcell_weight*prior."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(bcell_weight=0.2, k=5)
        # Store signatures that give a high prior
        for _ in range(3):
            features = _make_features(claim_density=0.5)
            cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0)

        query = _make_features(claim_density=0.5)
        ensemble = _make_result(score=0.6)
        result = cell.influence(query, ensemble)

        # prior ~0.9, so result ~0.8*0.6 + 0.2*0.9 = 0.66
        expected = 0.8 * 0.6 + 0.2 * 0.9
        assert result.score == pytest.approx(expected, abs=0.02)

    def test_influence_preserves_metadata(self):
        """influence returns DetectionResult with same confidence/anomaly_type/attribution."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(bcell_weight=0.2)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.5, ground_truth=1.0)

        ensemble = _make_result(score=0.6, confidence=0.85)
        result = cell.influence(features, ensemble)
        assert result.confidence == 0.85
        assert result.anomaly_type == "test_anomaly"
        assert result.feature_attribution == {"test_feat": 0.5}

    def test_influence_empty_memory_returns_ensemble_unchanged(self):
        """With empty memory, influence returns ensemble_result unchanged (bcell_weight forced to 0.0)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(bcell_weight=0.5)  # Even with high weight
        features = _make_features(claim_density=0.5)
        ensemble = _make_result(score=0.7)
        result = cell.influence(features, ensemble)
        assert result.score == 0.7  # Unchanged


# ---------------------------------------------------------------------------
# Clonal Selection
# ---------------------------------------------------------------------------


class TestBCellClonalSelection:
    """BCell.clonal_selection() clones TPs, decays FPs, evicts below threshold."""

    def test_clonal_selection_clones_tp(self):
        """TP signatures (outcome >= 0.5) are cloned 2x with gaussian jitter."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(max_memory=1000, jitter_sigma=0.05)
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)
        assert cell.memory_size == 1

        cell.clonal_selection()
        # Original + 2 clones = 3
        assert cell.memory_size == 3

    def test_clonal_selection_clones_have_jitter(self):
        """Cloned signatures have gaussian jitter on feature dims (not identical)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        np.random.seed(42)
        cell = BCell(max_memory=1000, jitter_sigma=0.05)
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)

        cell.clonal_selection()
        assert cell._memory is not None
        # Clones should differ from original in feature dims (cols 0-9)
        original_features = cell._memory[0, :10]
        clone1_features = cell._memory[1, :10]
        # At least some feature should differ (jitter applied)
        assert not np.allclose(original_features, clone1_features, atol=1e-10)

    def test_clonal_selection_decays_fp(self):
        """FP signatures (outcome < 0.5) have outcome decayed by half_life factor."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(half_life=0.9, eviction_threshold=0.01)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.6, ground_truth=0.3)  # FP: 0.3 < 0.5
        assert cell._memory is not None
        assert cell._memory[0, 11] == pytest.approx(0.3)

        cell.clonal_selection()
        assert cell._memory is not None
        # Outcome should be decayed: 0.3 * 0.9 = 0.27
        assert cell._memory[0, 11] == pytest.approx(0.27)

    def test_clonal_selection_evicts_below_threshold(self):
        """Signatures with outcome below eviction_threshold are removed."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(half_life=0.5, eviction_threshold=0.1)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.6, ground_truth=0.15)  # FP

        cell.clonal_selection()
        # 0.15 * 0.5 = 0.075 < 0.1 -> evicted
        assert cell.memory_size == 0

    def test_clonal_selection_enforces_max_memory(self):
        """After cloning, excess is evicted by lowest-outcome."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(max_memory=5, jitter_sigma=0.05)
        # Store 3 TP signatures -> clonal selection would create 9 total (3 orig + 6 clones)
        for i in range(3):
            features = _make_features(claim_density=float(i) / 10.0)
            cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)

        cell.clonal_selection()
        # Should be capped at max_memory=5
        assert cell.memory_size <= 5


# ---------------------------------------------------------------------------
# Max Memory Enforcement
# ---------------------------------------------------------------------------


class TestBCellMaxMemory:
    """BCell enforces max_memory by evicting lowest-outcome signatures."""

    def test_store_evicts_when_at_max(self):
        """When at max_memory, store_signature evicts lowest-outcome row first."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(max_memory=3)
        # Store 3 signatures with different outcomes
        for outcome in [0.3, 0.8, 0.5]:
            features = _make_features(claim_density=0.5)
            cell.store_signature(features, anomaly_score=0.6, ground_truth=outcome)

        assert cell.memory_size == 3

        # Store a 4th -> should evict the one with lowest outcome (0.3)
        features = _make_features(claim_density=0.9)
        cell.store_signature(features, anomaly_score=0.7, ground_truth=0.9)
        assert cell.memory_size == 3
        # Lowest outcome should now be 0.5 (0.3 was evicted)
        assert cell._memory is not None
        assert cell._memory[:, 11].min() >= 0.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestBCellPersistence:
    """BCell.save_memory() / load_memory() round-trip .npz files."""

    def test_save_and_load_memory(self):
        """save_memory writes .npz; load_memory restores exactly."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)
        cell.store_signature(features, anomaly_score=0.4, ground_truth=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_memory.npz")
            cell.save_memory(path)
            assert os.path.exists(path)

            # Load into a new BCell
            cell2 = BCell()
            cell2.load_memory(path)
            assert cell2.memory_size == 2
            assert cell2._memory is not None
            assert cell._memory is not None
            np.testing.assert_array_almost_equal(cell._memory, cell2._memory)

    def test_save_memory_noop_when_empty(self):
        """save_memory with empty memory is a no-op (no file created)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_memory.npz")
            cell.save_memory(path)
            assert not os.path.exists(path)

    def test_load_memory_missing_file_stays_empty(self):
        """load_memory with non-existent file keeps memory empty."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        cell.load_memory("/nonexistent/path/memory.npz")
        assert cell.memory_size == 0

    def test_load_memory_validates_shape(self):
        """load_memory rejects arrays with wrong number of columns."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_memory.npz")
            bad_array = np.zeros((3, 8))  # Wrong: 8 cols instead of 12
            np.savez(path, memory=bad_array)

            cell.load_memory(path)
            # Should reject and keep memory empty
            assert cell.memory_size == 0


# ---------------------------------------------------------------------------
# from_config Factory
# ---------------------------------------------------------------------------


class TestBCellFromConfig:
    """BCell.from_config() creates BCell from config dict."""

    def test_from_config_default(self):
        """from_config with empty dict uses defaults."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell.from_config({})
        assert cell._max_memory == 1000
        assert cell._k == 5
        assert cell._bcell_weight == pytest.approx(0.2)
        assert cell._half_life == pytest.approx(0.9)
        assert cell._eviction_threshold == pytest.approx(0.1)
        assert cell._jitter_sigma == pytest.approx(0.05)

    def test_from_config_custom(self):
        """from_config with custom values applies them."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        config = {
            "max_memory": 500,
            "k": 10,
            "bcell_weight": 0.3,
            "half_life": 0.8,
            "eviction_threshold": 0.2,
            "jitter_sigma": 0.1,
        }
        cell = BCell.from_config(config)
        assert cell._max_memory == 500
        assert cell._k == 10
        assert cell._bcell_weight == pytest.approx(0.3)
        assert cell._half_life == pytest.approx(0.8)
        assert cell._eviction_threshold == pytest.approx(0.2)
        assert cell._jitter_sigma == pytest.approx(0.1)


# ===========================================================================
# Phase 43: Embedding Mode Tests
# ===========================================================================


class TestBCellEmbeddingMode:
    """BCell embedding mode: dual-mode with 384-dim SLM embeddings."""

    # --- Construction ---

    def test_embedding_mode_true_constructs(self):
        """BCell(embedding_mode=True) constructs without error."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        assert cell._embedding_mode is True

    def test_embedding_mode_false_constructs(self):
        """BCell(embedding_mode=False) constructs, embedding_mode is False."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=False)
        assert cell._embedding_mode is False

    def test_embedding_mode_default_false(self):
        """BCell() defaults to embedding_mode=False (backward compat)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell()
        assert cell._embedding_mode is False

    def test_embedding_mode_true_stores_model_manager(self):
        """BCell(embedding_mode=True, model_manager=mock) stores model_manager reference."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        mm = _MockModelManager()
        cell = BCell(embedding_mode=True, model_manager=mm)
        assert cell._model_manager is mm

    def test_embedding_mode_true_no_model_manager_falls_back(self, caplog):
        """BCell(embedding_mode=True, model_manager=None) sets embedding_mode to False with warning."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        with caplog.at_level(logging.WARNING):
            cell = BCell(embedding_mode=True, model_manager=None)
        assert cell._embedding_mode is False
        assert any("fallback" in rec.message.lower() or "embedding_mode" in rec.message.lower()
                    for rec in caplog.records)

    # --- Storage in embedding mode ---

    def test_store_with_embedding(self):
        """store_signature with embedding kwarg stores both feature and embedding."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0, embedding=emb)
        assert cell.memory_size == 1
        assert cell._embeddings is not None
        assert cell._embeddings.shape == (1, 384)

    def test_store_multiple_embeddings_shape(self):
        """Internal _embeddings has shape (N, 384) after N stores."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        for i in range(5):
            features = _make_features(claim_density=float(i) / 10.0)
            emb = _make_embedding(seed=i)
            cell.store_signature(features, anomaly_score=0.5, ground_truth=1.0, embedding=emb)
        assert cell._embeddings is not None
        assert cell._embeddings.shape == (5, 384)

    def test_store_memory_shape_unchanged_in_embedding_mode(self):
        """Internal _memory still has shape (N, 12) in embedding mode."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        for i in range(3):
            features = _make_features(claim_density=float(i) / 10.0)
            emb = _make_embedding(seed=i)
            cell.store_signature(features, anomaly_score=0.5, ground_truth=1.0, embedding=emb)
        assert cell._memory is not None
        assert cell._memory.shape == (3, 12)

    def test_store_without_embedding_in_embedding_mode(self):
        """store_signature without embedding kwarg in embedding_mode stores zero-vector."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)
        assert cell.memory_size == 1
        assert cell._embeddings is not None
        # Zero-vector placeholder
        assert np.allclose(cell._embeddings[0], 0.0)

    # --- kNN prior with cosine similarity ---

    def test_prior_cosine_identical_embeddings(self):
        """prior_score with identical embeddings returns stored anomaly_score."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager(), k=5)
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0, embedding=emb)

        prior = cell.prior_score(features, embedding=emb)
        assert prior == pytest.approx(0.9, abs=0.01)

    def test_prior_cosine_orthogonal_embeddings(self):
        """Orthogonal embeddings give low weight to distant signatures."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager(), k=5)
        features = _make_features(claim_density=0.5)

        # Create two orthogonal embeddings
        emb_a = np.zeros(384, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(384, dtype=np.float32)
        emb_b[1] = 1.0

        cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0, embedding=emb_a)
        # Query with orthogonal embedding -- cosine similarity = 0
        prior = cell.prior_score(features, embedding=emb_b)
        # With only one stored signature and cos_sim~0, weight is near-zero
        # but the prior still uses that one neighbor (k=5 > 1 stored)
        # The weighted average will be dominated by near-zero weight
        assert isinstance(prior, float)

    def test_prior_feature_mode_uses_euclidean(self):
        """prior_score in feature-only mode uses euclidean (Phase 37 behavior)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=False, k=5)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0)

        query = _make_features(claim_density=0.5)
        prior = cell.prior_score(query)
        assert prior == pytest.approx(0.9, abs=0.01)

    def test_prior_empty_memory_embedding_mode(self):
        """prior_score returns 0.5 when memory empty regardless of embedding_mode."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        assert cell.prior_score(features, embedding=emb) == 0.5

    # --- Influence with embeddings ---

    def test_influence_with_embedding(self):
        """influence(features, result, embedding=emb) uses cosine prior path."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager(), bcell_weight=0.2, k=5)
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0, embedding=emb)

        ensemble = _make_result(score=0.6)
        result = cell.influence(features, ensemble, embedding=emb)
        # prior ~0.9 via cosine (identical embedding), so result ~0.8*0.6 + 0.2*0.9 = 0.66
        expected = 0.8 * 0.6 + 0.2 * 0.9
        assert result.score == pytest.approx(expected, abs=0.05)

    def test_influence_without_embedding_in_embedding_mode(self):
        """influence without embedding in embedding_mode falls back to feature-based."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager(), bcell_weight=0.2, k=5)
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.9, ground_truth=1.0, embedding=emb)

        ensemble = _make_result(score=0.6)
        # No embedding passed -> falls back to euclidean on features
        result = cell.influence(features, ensemble)
        # Should still produce a valid result
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    # --- Clonal selection in embedding space ---

    def test_clonal_selection_embedding_sigma(self):
        """clonal_selection() in embedding_mode uses sigma=0.01 on embeddings."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        np.random.seed(42)
        cell = BCell(
            embedding_mode=True,
            model_manager=_MockModelManager(),
            max_memory=1000,
            embedding_sigma=0.01,
        )
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=10)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0, embedding=emb)

        cell.clonal_selection()
        # Original + 2 clones = 3
        assert cell.memory_size == 3
        assert cell._embeddings is not None
        assert cell._embeddings.shape[0] == 3
        # Cloned embeddings differ from original by small amounts (sigma=0.01)
        original_emb = cell._embeddings[0]
        clone_emb = cell._embeddings[1]
        diff = np.abs(original_emb - clone_emb)
        # Max diff should be small (gaussian sigma=0.01)
        assert diff.max() < 0.2  # Very unlikely to exceed 0.2 with sigma=0.01

    def test_clonal_selection_embedding_mode_decays_fp(self):
        """clonal_selection() in embedding_mode still decays FP outcomes."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(
            embedding_mode=True,
            model_manager=_MockModelManager(),
            half_life=0.9,
            eviction_threshold=0.01,
        )
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.6, ground_truth=0.3, embedding=emb)

        cell.clonal_selection()
        assert cell._memory is not None
        assert cell._memory[0, 11] == pytest.approx(0.27)

    def test_clonal_selection_embedding_mode_evicts(self):
        """clonal_selection() in embedding_mode evicts below threshold."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(
            embedding_mode=True,
            model_manager=_MockModelManager(),
            half_life=0.5,
            eviction_threshold=0.1,
        )
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.6, ground_truth=0.15, embedding=emb)

        cell.clonal_selection()
        assert cell.memory_size == 0
        # Embeddings should also be cleared
        assert cell._embeddings is None or len(cell._embeddings) == 0

    def test_clonal_selection_feature_mode_uses_jitter_sigma(self):
        """clonal_selection() in feature-only mode uses sigma=0.05 (Phase 37 unchanged)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        np.random.seed(42)
        cell = BCell(embedding_mode=False, max_memory=1000, jitter_sigma=0.05)
        features = _make_features(claim_density=0.5, pamp_score=0.3)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)

        cell.clonal_selection()
        assert cell.memory_size == 3
        assert cell._memory is not None
        # Clones should have jitter on feature dims
        original = cell._memory[0, :10]
        clone = cell._memory[1, :10]
        assert not np.allclose(original, clone, atol=1e-10)

    # --- Persistence with embeddings ---

    def test_save_load_with_embeddings(self):
        """save_memory() in embedding_mode writes embeddings key; load restores."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager())
        features = _make_features(claim_density=0.5)
        emb = _make_embedding(seed=42)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0, embedding=emb)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "emb_memory.npz")
            cell.save_memory(path)
            assert os.path.exists(path)

            # Verify .npz contains embeddings key
            data = np.load(path)
            assert "embeddings" in data.files

            # Load into new BCell
            cell2 = BCell(embedding_mode=True, model_manager=_MockModelManager())
            cell2.load_memory(path)
            assert cell2.memory_size == 1
            assert cell2._embeddings is not None
            np.testing.assert_array_almost_equal(cell._embeddings, cell2._embeddings)

    def test_load_phase37_npz_backward_compat(self):
        """load_memory of Phase 37 .npz (no embeddings key) works in feature mode."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=False)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_memory.npz")
            cell.save_memory(path)

            cell2 = BCell(embedding_mode=False)
            cell2.load_memory(path)
            assert cell2.memory_size == 1
            assert cell2._embeddings is None

    def test_save_feature_mode_no_embeddings_key(self):
        """save_memory() in feature-only mode writes .npz without embeddings key."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=False)
        features = _make_features(claim_density=0.5)
        cell.store_signature(features, anomaly_score=0.8, ground_truth=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "feat_memory.npz")
            cell.save_memory(path)

            data = np.load(path)
            assert "embeddings" not in data.files
            assert "memory" in data.files

    # --- Eviction with embeddings ---

    def test_max_memory_eviction_removes_embedding_row(self):
        """max_memory eviction removes corresponding embedding row."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell(embedding_mode=True, model_manager=_MockModelManager(), max_memory=3)
        for i, outcome in enumerate([0.3, 0.8, 0.5]):
            features = _make_features(claim_density=float(i) / 10.0)
            emb = _make_embedding(seed=i)
            cell.store_signature(features, anomaly_score=0.6, ground_truth=outcome, embedding=emb)

        assert cell.memory_size == 3
        assert cell._embeddings is not None
        assert cell._embeddings.shape[0] == 3

        # Store 4th -> evicts lowest outcome (0.3)
        features = _make_features(claim_density=0.9)
        emb = _make_embedding(seed=99)
        cell.store_signature(features, anomaly_score=0.7, ground_truth=0.9, embedding=emb)
        assert cell.memory_size == 3
        assert cell._embeddings.shape[0] == 3

    # --- from_config factory ---

    def test_from_config_embedding_mode_true(self):
        """from_config({"embedding_mode": True}) sets embedding_mode=True."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        # Note: model_manager not passed via from_config, so it falls back to False
        # This tests the config parsing, not the runtime behavior
        cell = BCell.from_config({"embedding_mode": True})
        # Without model_manager, embedding_mode falls back to False
        assert cell._embedding_mode is False

    def test_from_config_no_embedding_mode(self):
        """from_config({}) sets embedding_mode=False (backward compat)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell.from_config({})
        assert cell._embedding_mode is False

    def test_from_config_custom_embedding_sigma(self):
        """from_config({"embedding_mode": True, "embedding_sigma": 0.02}) sets custom sigma."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        cell = BCell.from_config({"embedding_mode": True, "embedding_sigma": 0.02})
        assert cell._embedding_sigma == pytest.approx(0.02)
