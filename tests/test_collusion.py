"""Tests for validator-side collusion detection.

Covers VHARD-05 (collusion detection) and VHARD-06 (zero-score penalty).
Includes forward pass integration test for collusion wiring.
"""

import time
import unittest.mock as mock
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from antigence_subnet.validator.collusion import (
    CollusionAlert,
    CollusionConfig,
    CollusionDetector,
)


class TestCollusionConfig:
    """Tests for CollusionConfig dataclass."""

    def test_defaults_match_spec(self):
        """Default config matches D-02: threshold=0.99, min_group_size=3, penalty='zero'."""
        cfg = CollusionConfig()
        assert cfg.similarity_threshold == 0.99
        assert cfg.min_group_size == 3
        assert cfg.penalty == "zero"
        assert cfg.enabled is True

    def test_from_dict_full(self):
        """CollusionConfig.from_dict loads all fields from a dict matching TOML structure."""
        data = {
            "similarity_threshold": 0.95,
            "min_group_size": 2,
            "penalty": "zero",
            "enabled": False,
        }
        cfg = CollusionConfig.from_dict(data)
        assert cfg.similarity_threshold == 0.95
        assert cfg.min_group_size == 2
        assert cfg.penalty == "zero"
        assert cfg.enabled is False

    def test_from_dict_partial_uses_defaults(self):
        """from_dict with partial dict fills missing fields with defaults."""
        data = {"similarity_threshold": 0.95}
        cfg = CollusionConfig.from_dict(data)
        assert cfg.similarity_threshold == 0.95
        assert cfg.min_group_size == 3  # default
        assert cfg.penalty == "zero"  # default
        assert cfg.enabled is True  # default

    def test_from_dict_empty_uses_all_defaults(self):
        """from_dict with empty dict returns all defaults."""
        cfg = CollusionConfig.from_dict({})
        assert cfg.similarity_threshold == 0.99
        assert cfg.min_group_size == 3


class TestCollusionDetectorDetect:
    """Tests for CollusionDetector.detect() method."""

    def test_identical_vectors_detected(self):
        """3 miners with identical score vectors (similarity=1.0) are detected."""
        detector = CollusionDetector()
        scores = {"s1": 0.8, "s2": 0.3, "s3": 0.6, "s4": 0.9}
        miner_sample_scores = {0: scores, 1: scores, 2: scores}
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 1
        assert set(alerts[0].colluding_uids) == {0, 1, 2}

    def test_slightly_varied_vectors_detected(self):
        """3 miners with similarity > 0.99 threshold are detected."""
        detector = CollusionDetector()
        base = [0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.5, 0.1, 0.95]
        miner_sample_scores = {
            0: {f"s{i}": v for i, v in enumerate(base)},
            1: {f"s{i}": v + 0.001 * (i % 2) for i, v in enumerate(base)},
            2: {f"s{i}": v - 0.001 * (i % 2) for i, v in enumerate(base)},
        }
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) >= 1
        all_uids_flagged = set()
        for alert in alerts:
            all_uids_flagged.update(alert.colluding_uids)
        assert {0, 1, 2}.issubset(all_uids_flagged)

    def test_different_vectors_not_flagged(self):
        """3 miners with different vectors (low similarity) are NOT flagged."""
        detector = CollusionDetector()
        miner_sample_scores = {
            0: {"s1": 0.9, "s2": 0.1, "s3": 0.9, "s4": 0.1},
            1: {"s1": 0.1, "s2": 0.9, "s3": 0.1, "s4": 0.9},
            2: {"s1": 0.5, "s2": 0.5, "s3": 0.2, "s4": 0.8},
        }
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 0

    def test_two_miners_identical_not_flagged(self):
        """2 miners with identical vectors NOT flagged (min_group_size=3 per D-06)."""
        detector = CollusionDetector()
        scores = {"s1": 0.8, "s2": 0.3, "s3": 0.6}
        miner_sample_scores = {0: scores, 1: scores}
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 0

    def test_zero_miners_no_error(self):
        """0 total miners return empty results (edge case per D-06)."""
        detector = CollusionDetector()
        alerts = detector.detect(round_num=1, miner_uids=[], miner_sample_scores={})
        assert alerts == []

    def test_one_miner_no_error(self):
        """1 total miner returns empty results (edge case per D-06)."""
        detector = CollusionDetector()
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0],
            miner_sample_scores={0: {"s1": 0.5}},
        )
        assert alerts == []

    def test_two_miners_no_error(self):
        """2 total miners return empty results with default min_group_size=3."""
        detector = CollusionDetector()
        scores = {"s1": 0.5, "s2": 0.3, "s3": 0.7}
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1],
            miner_sample_scores={0: scores, 1: scores},
        )
        assert alerts == []

    def test_custom_threshold_catches_more(self):
        """Custom threshold (0.95) catches groups that default (0.99) would miss."""
        base = [0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.5, 0.1, 0.95]
        varied = [v + 0.05 * ((i % 3) - 1) for i, v in enumerate(base)]

        miner_sample_scores = {
            0: {f"s{i}": v for i, v in enumerate(base)},
            1: {f"s{i}": v for i, v in enumerate(varied)},
            2: {f"s{i}": v for i, v in enumerate(base)},
        }

        # Default threshold (0.99) should NOT catch this
        detector_default = CollusionDetector()
        alerts_default = detector_default.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )

        # Custom threshold (0.95) should catch this
        cfg = CollusionConfig(similarity_threshold=0.95)
        detector_custom = CollusionDetector(cfg)
        alerts_custom = detector_custom.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )

        assert len(alerts_custom) >= len(alerts_default)

    def test_custom_min_group_size_catches_pairs(self):
        """Custom min_group_size=2 catches pairs that default (3) would miss."""
        cfg = CollusionConfig(min_group_size=2)
        detector = CollusionDetector(cfg)
        scores = {"s1": 0.8, "s2": 0.3, "s3": 0.6}
        miner_sample_scores = {0: scores, 1: scores}
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 1
        assert set(alerts[0].colluding_uids) == {0, 1}

    def test_alert_contains_required_fields(self):
        """CollusionAlert contains round_num, colluding_uids, similarity_values, penalty (per D-04)."""  # noqa: E501
        detector = CollusionDetector()
        scores = {"s1": 0.8, "s2": 0.3, "s3": 0.6}
        miner_sample_scores = {0: scores, 1: scores, 2: scores}
        alerts = detector.detect(
            round_num=42,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.round_num == 42
        assert set(alert.colluding_uids) == {0, 1, 2}
        assert isinstance(alert.similarity_values, dict)
        assert len(alert.similarity_values) > 0
        # Each key should be a (uid_a, uid_b) tuple
        for key, val in alert.similarity_values.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, float)
        assert alert.penalty_applied == "zero"

    def test_all_zero_vectors_no_division_error(self):
        """All-zero score vectors do not cause division-by-zero errors."""
        detector = CollusionDetector()
        miner_sample_scores = {
            0: {"s1": 0.0, "s2": 0.0, "s3": 0.0},
            1: {"s1": 0.0, "s2": 0.0, "s3": 0.0},
            2: {"s1": 0.0, "s2": 0.0, "s3": 0.0},
        }
        # Should not raise; zero vectors have cosine sim = 0.0 -> not flagged
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 0

    def test_single_element_vectors_skipped(self):
        """Single-element score vectors are skipped (< 3 shared samples)."""
        detector = CollusionDetector()
        miner_sample_scores = {
            0: {"s1": 0.8},
            1: {"s1": 0.8},
            2: {"s1": 0.8},
        }
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 0

    def test_shared_samples_only(self):
        """Collusion detection only compares scores on shared sample IDs."""
        detector = CollusionDetector()
        # Miners share s1-s3, each has one unique sample
        miner_sample_scores = {
            0: {"s1": 0.8, "s2": 0.3, "s3": 0.9, "s4": 0.1},
            1: {"s1": 0.8, "s2": 0.3, "s3": 0.9, "s5": 0.2},
            2: {"s1": 0.8, "s2": 0.3, "s3": 0.9, "s6": 0.3},
        }
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 1

    def test_insufficient_shared_samples(self):
        """Pairs with < 3 shared samples are skipped."""
        detector = CollusionDetector()
        miner_sample_scores = {
            0: {"s1": 0.8, "s2": 0.3},
            1: {"s1": 0.8, "s3": 0.3},
            2: {"s1": 0.8, "s4": 0.3},
        }
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert alerts == []

    def test_clique_based_grouping(self):
        """Groups are cliques: ALL pairwise similarities must exceed threshold."""
        cfg = CollusionConfig(similarity_threshold=0.95, min_group_size=3)
        detector = CollusionDetector(cfg)

        base_a = [0.8, 0.3, 0.6, 0.9, 0.2]
        base_b = [0.1, 0.7, 0.4, 0.2, 0.9]

        miner_sample_scores = {
            0: {f"s{i}": v for i, v in enumerate(base_a)},
            1: {f"s{i}": v + 0.001 for i, v in enumerate(base_a)},
            2: {f"s{i}": v for i, v in enumerate(base_b)},
        }

        # 0-1 similar, 0-2 and 1-2 not similar -> no clique of 3
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores=miner_sample_scores,
        )
        assert len(alerts) == 0

    def test_disabled_returns_empty(self):
        """Disabled config returns empty alerts even with colluding miners."""
        detector = CollusionDetector(CollusionConfig(enabled=False))
        scores = {"s1": 0.8, "s2": 0.3, "s3": 0.9}
        alerts = detector.detect(
            round_num=1,
            miner_uids=[0, 1, 2],
            miner_sample_scores={0: scores, 1: scores, 2: scores},
        )
        assert alerts == []

    def test_performance_256_miners_under_10ms(self):
        """Detection on 256 miners with 10-element vectors completes in < 10ms (per D-05).

        Uses median of multiple runs to avoid GC spike false failures.
        """
        detector = CollusionDetector()
        rng = np.random.default_rng(42)
        sample_ids = [f"s{i}" for i in range(10)]
        miner_sample_scores = {}
        for uid in range(256):
            scores = rng.random(10)
            miner_sample_scores[uid] = {sample_ids[i]: float(scores[i]) for i in range(10)}

        miner_uids = list(range(256))

        # Warm up
        detector.detect(
            round_num=0,
            miner_uids=miner_uids,
            miner_sample_scores=miner_sample_scores,
        )

        # Take median of 5 runs (robust against GC spikes)
        timings = []
        for run in range(5):
            start = time.perf_counter()
            detector.detect(
                round_num=run + 1,
                miner_uids=miner_uids,
                miner_sample_scores=miner_sample_scores,
            )
            timings.append((time.perf_counter() - start) * 1000)

        median_ms = sorted(timings)[len(timings) // 2]
        assert median_ms < 10.0, (
            f"Detection median {median_ms:.2f}ms (runs: "
            f"{', '.join(f'{t:.1f}' for t in timings)}ms), must be < 10ms"
        )


class TestCollusionDetectorApplyPenalty:
    """Tests for CollusionDetector.apply_penalty() method."""

    def test_zeros_scores_for_colluding_uids(self):
        """apply_penalty zeros scores for all UIDs in collusion group (per D-03)."""
        detector = CollusionDetector()
        rewards = np.array([0.8, 0.7, 0.6, 0.9], dtype=np.float32)
        miner_uids = [10, 20, 30, 40]
        alerts = [
            CollusionAlert(
                round_num=1,
                colluding_uids=[10, 20, 30],
                similarity_values={(10, 20): 1.0, (10, 30): 1.0, (20, 30): 1.0},
                penalty_applied="zero",
            )
        ]
        result = detector.apply_penalty(rewards, miner_uids, alerts)
        assert result[0] == 0.0  # uid 10
        assert result[1] == 0.0  # uid 20
        assert result[2] == 0.0  # uid 30
        assert result[3] == 0.9  # uid 40 unchanged

    def test_does_not_modify_non_colluding(self):
        """apply_penalty does not modify scores for non-colluding miners."""
        detector = CollusionDetector()
        rewards = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        miner_uids = [0, 1, 2, 3, 4]
        alerts = [
            CollusionAlert(
                round_num=1,
                colluding_uids=[0, 1, 2],
                similarity_values={(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0},
                penalty_applied="zero",
            )
        ]
        result = detector.apply_penalty(rewards, miner_uids, alerts)
        assert result[3] == 0.8
        assert result[4] == 0.9

    def test_empty_alerts_no_change(self):
        """Empty alerts list does not modify rewards."""
        detector = CollusionDetector()
        rewards = np.array([0.5, 0.6], dtype=np.float32)
        original = rewards.copy()
        result = detector.apply_penalty(rewards, [0, 1], [])
        np.testing.assert_array_equal(result, original)

    def test_returns_same_array(self):
        """apply_penalty modifies in-place and returns the same array."""
        detector = CollusionDetector()
        rewards = np.array([0.5, 0.6], dtype=np.float32)
        result = detector.apply_penalty(rewards, [0, 1], [])
        assert result is rewards


class TestCollusionDetectorLogAlerts:
    """Tests for CollusionDetector.log_alerts() method."""

    def test_log_alerts_calls_bt_logging(self):
        """log_alerts emits structured bt.logging.warning per alert."""
        detector = CollusionDetector()
        alerts = [
            CollusionAlert(
                round_num=5,
                colluding_uids=[1, 2, 3],
                similarity_values={(1, 2): 0.995, (1, 3): 0.993, (2, 3): 0.997},
                penalty_applied="zero",
            )
        ]
        with mock.patch("antigence_subnet.validator.collusion.bt") as mock_bt:
            detector.log_alerts(alerts)
            mock_bt.logging.warning.assert_called_once()
            call_args = mock_bt.logging.warning.call_args[0][0]
            assert "Collusion detected" in call_args
            assert "round=5" in call_args
            assert "penalty=zero" in call_args
            assert "max_similarity=0.9970" in call_args


class TestCollusionForwardIntegration:
    """Integration test verifying collusion detection fires during forward pass."""

    @pytest.mark.asyncio
    async def test_colluding_miners_get_zero_rewards(self, tmp_path):
        """3 miners returning identical scores get zeroed by collusion detection in forward."""
        from antigence_subnet.validator.forward import forward

        # Build a minimal mock validator
        config = SimpleNamespace()
        config.netuid = 1
        config.neuron = SimpleNamespace(
            sample_size=3,
            timeout=12.0,
            moving_average_alpha=0.1,
            samples_per_round=10,
            n_honeypots=2,
            set_weights_interval=100,
            eval_data_dir="data/evaluation",
            eval_domain="hallucination",
            full_path=str(tmp_path),
        )
        config.mock = True
        # Enable collusion detection via validator.collusion config
        config.validator = SimpleNamespace(
            collusion=SimpleNamespace(
                similarity_threshold=0.99,
                min_group_size=3,
                penalty="zero",
                enabled=True,
            )
        )

        n_miners = 3
        total = n_miners + 1
        metagraph = SimpleNamespace(
            n=total,
            axons=[SimpleNamespace(ip="127.0.0.1", port=8091 + i) for i in range(total)],
            hotkeys=[f"hotkey-{i}" for i in range(total)],
        )

        validator = SimpleNamespace(
            config=config,
            metagraph=metagraph,
            uid=0,
            wallet=MagicMock(),
            subtensor=MagicMock(),
            step=1,
            scores=np.zeros(total, dtype=np.float32),
            hotkeys=list(metagraph.hotkeys),
            score_history={},
            confidence_history={},
        )

        # Try to load real evaluation data; skip if unavailable
        try:
            from pathlib import Path

            from antigence_subnet.validator.evaluation import EvaluationDataset

            eval_path = Path("data/evaluation")
            if eval_path.exists() and (eval_path / "hallucination").exists():
                validator.evaluation = EvaluationDataset(data_dir=eval_path, domain="hallucination")
            else:
                pytest.skip("Seed data not available")
        except Exception:
            pytest.skip("Evaluation dataset not loadable")

        # All 3 miners return IDENTICAL anomaly_scores -> triggers collusion
        async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
            resp = MagicMock()
            resp.anomaly_score = 0.75  # All same score
            resp.anomaly_type = "factual_error"
            resp.confidence = 0.9
            resp.feature_attribution = {"mock": 0.5}
            return [resp for _ in axons]

        validator.dendrite = mock_dendrite

        update_calls = []

        def mock_update_scores(rewards, uids):
            update_calls.append((rewards.copy(), list(uids)))
            alpha = validator.config.neuron.moving_average_alpha
            for i, uid in enumerate(uids):
                if 0 <= uid < len(validator.scores):
                    validator.scores[uid] = alpha * rewards[i] + (1 - alpha) * validator.scores[uid]

        validator.update_scores = mock_update_scores

        await forward(validator)

        # update_scores should have been called
        assert len(update_calls) == 1
        rewards_passed, uids_passed = update_calls[0]

        # All 3 miners returned identical scores -> detected as colluding -> all zeroed
        for i in range(len(uids_passed)):
            assert rewards_passed[i] == 0.0, (
                f"Miner uid={uids_passed[i]} should have been zeroed by collusion "
                f"detection but got reward={rewards_passed[i]}"
            )
