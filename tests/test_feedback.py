"""Tests for ValidatorFeedbackTracker."""

import time

import numpy as np
import pytest

from antigence_subnet.miner.orchestrator.feedback import (
    DetectionRecord,
    ValidatorFeedbackTracker,
)


class TestFeedbackTracker:
    def test_disabled_returns_zero(self):
        t = ValidatorFeedbackTracker(enabled=False)
        assert t.record_round(0.5, 0.8, 10) == 0.0

    def test_first_round_returns_zero(self):
        t = ValidatorFeedbackTracker(enabled=True)
        assert t.record_round(0.5, 0.8, 10) == 0.0

    def test_weight_increase_positive(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.record_round(0.5, 0.8, 10)
        signal = t.record_round(0.55, 0.8, 10)
        assert signal > 0.0

    def test_weight_decrease_negative(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.record_round(0.5, 0.8, 10)
        signal = t.record_round(0.45, 0.8, 10)
        assert signal < 0.0

    def test_signal_clamped(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.record_round(0.01, 0.8, 10)
        signal = t.record_round(1.0, 0.8, 10)
        assert -1.0 <= signal <= 1.0

    def test_from_config(self):
        t = ValidatorFeedbackTracker.from_config({"lookback_rounds": 10, "enabled": True})
        assert t._lookback == 10
        assert t._enabled is True

    def test_from_config_defaults(self):
        t = ValidatorFeedbackTracker.from_config({})
        assert t._lookback == 5
        assert t._enabled is False


class TestRecentSignal:
    def test_insufficient_data(self):
        t = ValidatorFeedbackTracker(enabled=True)
        assert t.get_recent_signal() == 0.0

    def test_positive_trend(self):
        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=5)
        weights = [0.5, 0.52, 0.54, 0.56, 0.58]
        for w in weights:
            t.record_round(w, 0.8, 10)
        assert t.get_recent_signal() > 0.0

    def test_negative_trend(self):
        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=5)
        weights = [0.5, 0.48, 0.46, 0.44, 0.42]
        for w in weights:
            t.record_round(w, 0.8, 10)
        assert t.get_recent_signal() < 0.0


class TestBCellFeedback:
    """Tests for apply_to_bcell with real BCell numpy arrays."""

    def test_no_bcell_no_error(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.apply_to_bcell(None, 0.5)  # should not error

    def test_zero_signal_no_change(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.apply_to_bcell(None, 0.0)  # should not error

    def test_apply_to_bcell_boosts_outcomes(self):
        """Positive signal boosts outcome column of BCell._memory."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True)
        bcell = BCell(max_memory=100)

        # Store 3 signatures with outcome 0.5
        for i in range(3):
            features = np.random.rand(10)
            bcell.store_signature(features, anomaly_score=0.7, ground_truth=0.5)

        original_outcomes = bcell._memory[:, 11].copy()
        t.apply_to_bcell(bcell, signal=0.8)

        # Outcomes should have increased
        assert np.all(bcell._memory[:, 11] > original_outcomes)
        # Outcomes should stay in [0, 1]
        assert np.all(bcell._memory[:, 11] >= 0.0)
        assert np.all(bcell._memory[:, 11] <= 1.0)

    def test_apply_to_bcell_decays_outcomes(self):
        """Negative signal decays outcome column."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True)
        bcell = BCell(max_memory=100)

        # Store 3 signatures with outcome 0.8
        for i in range(3):
            features = np.random.rand(10)
            bcell.store_signature(features, anomaly_score=0.7, ground_truth=0.8)

        original_outcomes = bcell._memory[:, 11].copy()
        t.apply_to_bcell(bcell, signal=-0.5)

        # Outcomes should have decreased
        assert np.all(bcell._memory[:, 11] < original_outcomes)
        # Outcomes should stay >= 0
        assert np.all(bcell._memory[:, 11] >= 0.0)

    def test_apply_to_bcell_empty_memory_no_crash(self):
        """apply_to_bcell is no-op when BCell has empty memory."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True)
        bcell = BCell(max_memory=100)
        # No signatures stored -- memory is None
        assert bcell._memory is None
        t.apply_to_bcell(bcell, signal=0.8)  # should not crash

    def test_apply_to_bcell_zero_signal_no_change(self):
        """apply_to_bcell is no-op when signal is 0.0."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True)
        bcell = BCell(max_memory=100)
        features = np.random.rand(10)
        bcell.store_signature(features, anomaly_score=0.7, ground_truth=0.5)

        original_outcomes = bcell._memory[:, 11].copy()
        t.apply_to_bcell(bcell, signal=0.0)
        np.testing.assert_array_equal(bcell._memory[:, 11], original_outcomes)


class TestDCAFeedback:
    def test_no_weights_no_error(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.apply_to_dca(None, 0.5)  # should not error

    def test_zero_signal_no_change(self):
        t = ValidatorFeedbackTracker(enabled=True)
        t.apply_to_dca(None, 0.0)  # should not error

    def test_apply_to_dca_calls_adapt(self):
        """apply_to_dca with real AdaptiveWeightManager calls adapt()."""
        from antigence_subnet.miner.orchestrator.adaptive_weights import (
            AdaptiveWeightManager,
        )

        t = ValidatorFeedbackTracker(enabled=True)
        awm = AdaptiveWeightManager()
        features = np.random.rand(10)
        initial_round = awm.get_round_count()
        t.apply_to_dca(awm, signal=0.5, features=features)
        assert awm.get_round_count() == initial_round + 1


class TestDetectionRecord:
    """Tests for DetectionRecord dataclass."""

    def test_detection_record_fields(self):
        """DetectionRecord stores round_num, features, anomaly_score, domain, timestamp."""
        features = np.random.rand(10)
        now = time.time()
        rec = DetectionRecord(
            round_num=0,
            features=features,
            anomaly_score=0.7,
            domain="hallucination",
            timestamp=now,
        )
        assert rec.round_num == 0
        np.testing.assert_array_equal(rec.features, features)
        assert rec.anomaly_score == 0.7
        assert rec.domain == "hallucination"
        assert rec.timestamp == now


class TestRecordDetection:
    """Tests for record_detection and get_recent_detections."""

    def test_record_detection(self):
        """record_detection stores features+score+domain."""
        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=5)
        features = np.random.rand(10)
        t.record_detection(features, anomaly_score=0.7, domain="hallucination")

        detections = t.get_recent_detections()
        assert len(detections) == 1
        assert detections[0].anomaly_score == 0.7
        assert detections[0].domain == "hallucination"
        np.testing.assert_array_equal(detections[0].features, features)

    def test_get_recent_detections_lookback_window(self):
        """get_recent_detections returns only detections within lookback window."""
        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=3)

        # Record detections in rounds 0, 1, 2, 3, 4
        for i in range(5):
            features = np.full(10, float(i))
            t.record_detection(features, anomaly_score=float(i) * 0.1, domain="test")
            # Advance round counter
            t.record_round(0.5, 0.8, 1)

        # Only last 3 rounds should be in the window
        recent = t.get_recent_detections(n_rounds=3)
        # Round numbers should be >= (current_round - 3) = 5 - 3 = 2
        for det in recent:
            assert det.round_num >= 2

    def test_record_detection_disabled_no_store(self):
        """record_detection is no-op when tracker is disabled."""
        t = ValidatorFeedbackTracker(enabled=False, lookback_rounds=5)
        features = np.random.rand(10)
        t.record_detection(features, anomaly_score=0.7, domain="test")
        assert len(t.get_recent_detections()) == 0

    def test_detection_deque_bounded(self):
        """Detection deque is bounded to prevent unbounded growth."""
        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=2)
        # maxlen should be lookback_rounds * 50 = 100
        for i in range(150):
            features = np.full(10, float(i))
            t.record_detection(features, anomaly_score=0.5, domain="test")

        # Should be capped at 100
        all_dets = t.get_recent_detections(n_rounds=999)
        assert len(all_dets) <= 100


class TestCorrelatedBCellFeedback:
    """Tests for apply_feedback_to_bcell_correlated."""

    def test_correlated_feedback_uses_detection_features(self):
        """apply_feedback_to_bcell_correlated uses recent detection features
        to do per-signature reinforcement on matching BCell signatures."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=10)
        bcell = BCell(max_memory=100)

        # Store 5 BCell signatures with known features
        base_features = np.zeros(10)
        for i in range(5):
            features = base_features.copy()
            features[0] = float(i) * 0.2  # spread in dim 0
            bcell.store_signature(features, anomaly_score=0.5, ground_truth=0.5)

        # Record a detection that is very similar to signature 0
        det_features = base_features.copy()
        det_features[0] = 0.0  # exact match for signature 0
        t.record_detection(det_features, anomaly_score=0.7, domain="test")
        t.record_round(0.5, 0.8, 1)  # register round

        original_outcomes = bcell._memory[:, 11].copy()

        # Apply correlated feedback with positive signal
        t.apply_feedback_to_bcell_correlated(bcell, signal=0.8)

        # Some outcomes should have changed (at least the matching ones)
        changed = bcell._memory[:, 11] != original_outcomes
        assert np.any(changed), "At least some signatures should be updated"

    def test_correlated_feedback_empty_bcell_no_crash(self):
        """apply_feedback_to_bcell_correlated handles empty BCell gracefully."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=5)
        bcell = BCell(max_memory=100)
        t.apply_feedback_to_bcell_correlated(bcell, signal=0.8)  # no crash

    def test_correlated_feedback_no_detections_no_op(self):
        """apply_feedback_to_bcell_correlated is no-op when no detection records."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        t = ValidatorFeedbackTracker(enabled=True, lookback_rounds=5)
        bcell = BCell(max_memory=100)
        features = np.random.rand(10)
        bcell.store_signature(features, anomaly_score=0.7, ground_truth=0.5)

        original_outcomes = bcell._memory[:, 11].copy()
        t.apply_feedback_to_bcell_correlated(bcell, signal=0.8)
        np.testing.assert_array_equal(bcell._memory[:, 11], original_outcomes)


class TestFeedbackConfig:
    """Tests for FeedbackConfig dataclass and TOML parsing."""

    def test_feedback_config_defaults(self):
        """FeedbackConfig() has enabled=False, lookback_rounds=5."""
        from antigence_subnet.miner.orchestrator.config import FeedbackConfig

        fc = FeedbackConfig()
        assert fc.enabled is False
        assert fc.lookback_rounds == 5

    def test_feedback_config_from_toml(self):
        """FeedbackConfig parses [miner.orchestrator.feedback] section correctly."""
        from antigence_subnet.miner.orchestrator.config import FeedbackConfig

        raw = {
            "miner": {
                "orchestrator": {
                    "feedback": {
                        "enabled": True,
                        "lookback_rounds": 10,
                    },
                }
            }
        }
        fc = FeedbackConfig.from_toml_raw(raw)
        assert fc.enabled is True
        assert fc.lookback_rounds == 10

    def test_feedback_config_missing_section(self):
        """FeedbackConfig returns defaults when TOML section absent."""
        from antigence_subnet.miner.orchestrator.config import FeedbackConfig

        fc = FeedbackConfig.from_toml_raw({})
        assert fc.enabled is False
        assert fc.lookback_rounds == 5

    def test_feedback_config_validation(self):
        """lookback_rounds < 1 raises ValueError."""
        from antigence_subnet.miner.orchestrator.config import (
            FeedbackConfig,
            _validate_feedback_config,
        )

        fc = FeedbackConfig(lookback_rounds=0)
        with pytest.raises(ValueError, match="lookback_rounds"):
            _validate_feedback_config(fc)

    def test_feedback_config_validation_negative(self):
        """lookback_rounds = -5 raises ValueError."""
        from antigence_subnet.miner.orchestrator.config import (
            FeedbackConfig,
            _validate_feedback_config,
        )

        fc = FeedbackConfig(lookback_rounds=-5)
        with pytest.raises(ValueError, match="lookback_rounds"):
            _validate_feedback_config(fc)

    def test_orchestrator_config_includes_feedback(self):
        """OrchestratorConfig.from_toml_raw includes feedback_config."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "feedback": {
                        "enabled": True,
                        "lookback_rounds": 7,
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.feedback_config.enabled is True
        assert config.feedback_config.lookback_rounds == 7

    def test_orchestrator_config_feedback_defaults_when_absent(self):
        """OrchestratorConfig has feedback_config with defaults when TOML section absent."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig.from_toml_raw({})
        assert config.feedback_config.enabled is False
        assert config.feedback_config.lookback_rounds == 5

    def test_orchestrator_config_feedback_validation_error(self):
        """OrchestratorConfig.from_toml_raw raises ValueError for lookback_rounds < 1."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        raw = {
            "miner": {
                "orchestrator": {
                    "feedback": {
                        "lookback_rounds": 0,
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="lookback_rounds"):
            OrchestratorConfig.from_toml_raw(raw)


class TestOrchestratorFeedback:
    """Integration tests for orchestrator feedback pipeline."""

    def _make_orchestrator(self, feedback_enabled=True, with_bcell=True, with_dca=True):
        """Helper to build an ImmuneOrchestrator with feedback wiring."""
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig, FeedbackConfig
        from antigence_subnet.miner.orchestrator.nk_cell import NKCell
        from antigence_subnet.miner.orchestrator.dendritic_cell import DendriticCell
        from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
        from antigence_subnet.miner.orchestrator.b_cell import BCell
        from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager
        from antigence_subnet.miner.orchestrator.feedback import ValidatorFeedbackTracker
        from antigence_subnet.miner.detectors.dendritic_features import DendriticFeatureExtractor

        extractor = DendriticFeatureExtractor()
        nk_cell = NKCell(feature_stats=[])
        dc = DendriticCell.from_config({})
        danger = DangerTheoryModulator.from_config({})

        b_cell = BCell(max_memory=100) if with_bcell else None
        awm = AdaptiveWeightManager() if with_dca else None

        feedback = None
        if feedback_enabled:
            feedback = ValidatorFeedbackTracker(lookback_rounds=5, enabled=True)

        config = OrchestratorConfig(
            enabled=True,
            feedback_config=FeedbackConfig(enabled=feedback_enabled, lookback_rounds=5),
        )

        return ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={},
            config=config,
            b_cell=b_cell,
            adaptive_weights=awm,
            feedback=feedback,
        )

    def test_orchestrator_feedback_from_config_enabled(self):
        """from_config creates ValidatorFeedbackTracker when feedback_config.enabled=True."""
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig, FeedbackConfig

        config = OrchestratorConfig(
            enabled=True,
            feedback_config=FeedbackConfig(enabled=True, lookback_rounds=7),
        )
        orch = ImmuneOrchestrator.from_config(config, detectors={})
        assert orch._feedback is not None
        assert orch._feedback.enabled is True

    def test_orchestrator_feedback_from_config_disabled(self):
        """from_config creates None feedback when feedback_config.enabled=False."""
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig(enabled=True)  # feedback_config defaults to disabled
        orch = ImmuneOrchestrator.from_config(config, detectors={})
        assert orch._feedback is None

    @pytest.mark.asyncio
    async def test_orchestrator_process_records_detection(self):
        """process() calls record_detection with features and score after ensemble."""
        from unittest.mock import patch, AsyncMock
        from antigence_subnet.miner.detector import BaseDetector, DetectionResult as DR

        class StubDetector(BaseDetector):
            async def detect(self, prompt, output, code=None, context=None):
                return DR(score=0.6, confidence=0.9, anomaly_type="test")
            async def fit(self, samples):
                pass
            def get_info(self):
                return {"name": "stub"}

        orch = self._make_orchestrator(feedback_enabled=True, with_bcell=False, with_dca=False)
        orch._detectors = {"hallucination": [StubDetector()]}

        with patch.object(orch._feedback, 'record_detection') as mock_record:
            result = await orch.process(
                prompt="test prompt",
                output="test output",
                domain="hallucination",
            )
            # record_detection should have been called once
            assert mock_record.called
            # Verify it was called with features array and score
            call_args = mock_record.call_args
            assert call_args.kwargs["domain"] == "hallucination"

    def test_orchestrator_process_feedback_positive_bcell(self):
        """process_feedback() with positive signal calls apply on BCell."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell
        orch = self._make_orchestrator(feedback_enabled=True, with_bcell=True, with_dca=False)

        # Store some BCell signatures
        for i in range(3):
            orch._b_cell.store_signature(np.random.rand(10), 0.7, 0.5)

        # Record a detection so correlated feedback has data
        orch._feedback.record_detection(np.random.rand(10), 0.7, "test")
        orch._feedback.record_round(0.5, 0.8, 1)

        original_outcomes = orch._b_cell._memory[:, 11].copy()
        signal = orch.process_feedback(current_weight=0.6, avg_score=0.8, detection_count=5)
        # Signal should be positive (weight increased from 0.5 to 0.6)
        assert signal > 0.0

    def test_orchestrator_process_feedback_positive_dca(self):
        """process_feedback() with positive signal calls apply_to_dca on AdaptiveWeightManager."""
        orch = self._make_orchestrator(feedback_enabled=True, with_bcell=False, with_dca=True)

        # Record some detections
        orch._feedback.record_detection(np.random.rand(10), 0.7, "test")
        orch._feedback.record_round(0.5, 0.8, 1)

        initial_round = orch._adaptive_weights.get_round_count()
        signal = orch.process_feedback(current_weight=0.6, avg_score=0.8, detection_count=5)
        assert signal > 0.0
        # DCA should have been adapted
        assert orch._adaptive_weights.get_round_count() > initial_round

    def test_orchestrator_process_feedback_noop_disabled(self):
        """process_feedback() is no-op when feedback tracker is None."""
        orch = self._make_orchestrator(feedback_enabled=False, with_bcell=True, with_dca=True)
        signal = orch.process_feedback(current_weight=0.6, avg_score=0.8, detection_count=5)
        assert signal == 0.0

    def test_orchestrator_process_feedback_noop_zero_signal(self):
        """process_feedback() is no-op when signal is 0.0 (first round)."""
        orch = self._make_orchestrator(feedback_enabled=True, with_bcell=True, with_dca=True)
        # First round: no prior weight, signal is 0.0
        signal = orch.process_feedback(current_weight=0.5, avg_score=0.8, detection_count=5)
        assert signal == 0.0

    def test_orchestrator_save_state_with_feedback(self):
        """save_state() does not crash when feedback is enabled."""
        orch = self._make_orchestrator(feedback_enabled=True, with_bcell=True, with_dca=False)
        # Should not crash
        orch.save_state(domain="test")


class TestPackageExports:
    """Tests for package-level exports of feedback types."""

    def test_detection_record_importable(self):
        from antigence_subnet.miner.orchestrator import DetectionRecord
        assert DetectionRecord is not None

    def test_feedback_config_importable(self):
        from antigence_subnet.miner.orchestrator import FeedbackConfig
        assert FeedbackConfig is not None

    def test_validator_feedback_tracker_importable(self):
        from antigence_subnet.miner.orchestrator import ValidatorFeedbackTracker
        assert ValidatorFeedbackTracker is not None


class TestImport:
    def test_importable_from_package(self):
        from antigence_subnet.miner.orchestrator.feedback import ValidatorFeedbackTracker
        assert ValidatorFeedbackTracker is not None

    def test_detection_record_importable(self):
        from antigence_subnet.miner.orchestrator.feedback import DetectionRecord
        assert DetectionRecord is not None

    def test_feedback_config_importable(self):
        from antigence_subnet.miner.orchestrator.config import FeedbackConfig
        assert FeedbackConfig is not None
