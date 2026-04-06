"""Tests for MicrogliaMonitor -- per-miner health tracking, detection, alerting, health metrics.

Covers GLIA-01, GLIA-02, GLIA-03, GLIA-04: inactive/stale/deregistration detection,
coordinated attack detection, registration surge detection, webhook alerting,
alert generation with deduplication, and subnet health metric aggregation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from antigence_subnet.validator.microglia import (
    AlertType,
    MicrogliaMonitor,
    MinerHealthState,
    SubnetHealthMetrics,
)


class TestMinerHealthState:
    """Tests for MinerHealthState dataclass defaults."""

    def test_default_values(self):
        """MinerHealthState initializes with sensible defaults."""
        state = MinerHealthState()
        assert state.last_response_step == 0
        assert state.response_count == 0
        assert state.consecutive_failures == 0
        assert state.avg_latency == 0.0
        assert state.last_anomaly_scores == []

    def test_custom_values(self):
        """MinerHealthState accepts custom values."""
        state = MinerHealthState(
            last_response_step=42,
            response_count=10,
            consecutive_failures=2,
            avg_latency=0.5,
            last_anomaly_scores=[0.1, 0.2, 0.3],
        )
        assert state.last_response_step == 42
        assert state.response_count == 10
        assert state.consecutive_failures == 2
        assert state.avg_latency == pytest.approx(0.5)
        assert state.last_anomaly_scores == [0.1, 0.2, 0.3]


class TestSubnetHealthMetrics:
    """Tests for SubnetHealthMetrics dataclass."""

    def test_stores_all_fields(self):
        """SubnetHealthMetrics stores inflammation, threat, diversity, counts."""
        metrics = SubnetHealthMetrics(
            inflammation_score=0.25,
            threat_level="medium",
            population_diversity_index=0.6,
            active_miners=12,
            inactive_miners=3,
            stale_miners=1,
            deregistration_candidates=0,
        )
        assert metrics.inflammation_score == pytest.approx(0.25)
        assert metrics.threat_level == "medium"
        assert metrics.population_diversity_index == pytest.approx(0.6)
        assert metrics.active_miners == 12
        assert metrics.inactive_miners == 3
        assert metrics.stale_miners == 1
        assert metrics.deregistration_candidates == 0


class TestAlertType:
    """Tests for AlertType enum values."""

    def test_miner_inactive(self):
        assert AlertType.MINER_INACTIVE == "MINER_INACTIVE"

    def test_miner_stale(self):
        assert AlertType.MINER_STALE == "MINER_STALE"

    def test_deregistration_candidate(self):
        assert AlertType.DEREGISTRATION_CANDIDATE == "DEREGISTRATION_CANDIDATE"

    def test_coordinated_attack(self):
        assert AlertType.COORDINATED_ATTACK == "COORDINATED_ATTACK"

    def test_registration_surge(self):
        assert AlertType.REGISTRATION_SURGE == "REGISTRATION_SURGE"

    def test_metagraph_anomaly_alert_type_exists(self):
        """METAGRAPH_ANOMALY alert type exists for metagraph anomaly detection (NET-05)."""
        assert AlertType.METAGRAPH_ANOMALY == "METAGRAPH_ANOMALY"


class TestMicrogliaMonitorInit:
    """Tests for MicrogliaMonitor initialization."""

    def test_default_thresholds(self):
        """Init with default thresholds."""
        monitor = MicrogliaMonitor()
        assert monitor.inactive_threshold == 10
        assert monitor.stale_threshold == 5
        assert monitor.deregistration_threshold == 50
        assert monitor.alert_cooldown == 10

    def test_custom_thresholds(self):
        """Init with custom thresholds."""
        monitor = MicrogliaMonitor(
            inactive_threshold=20,
            stale_threshold=8,
            deregistration_threshold=100,
            alert_cooldown=5,
        )
        assert monitor.inactive_threshold == 20
        assert monitor.stale_threshold == 8
        assert monitor.deregistration_threshold == 100
        assert monitor.alert_cooldown == 5


class TestMicrogliaMonitorRecording:
    """Tests for record_response and record_failure."""

    def test_record_response_creates_health_state(self):
        """record_response creates MinerHealthState for new UID."""
        monitor = MicrogliaMonitor()
        monitor.record_response(uid=5, anomaly_score=0.7, latency=0.1, current_step=10)
        state = monitor._miner_health[5]
        assert state.response_count == 1
        assert state.last_response_step == 10
        assert state.consecutive_failures == 0
        assert state.avg_latency == pytest.approx(0.1)
        assert state.last_anomaly_scores == [0.7]

    def test_record_response_increments_count(self):
        """Multiple record_response calls increment response_count."""
        monitor = MicrogliaMonitor()
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        monitor.record_response(uid=1, anomaly_score=0.6, latency=0.2, current_step=2)
        monitor.record_response(uid=1, anomaly_score=0.7, latency=0.3, current_step=3)
        state = monitor._miner_health[1]
        assert state.response_count == 3
        assert state.last_response_step == 3

    def test_record_response_resets_consecutive_failures(self):
        """record_response resets consecutive_failures to 0."""
        monitor = MicrogliaMonitor()
        monitor.record_failure(uid=2)
        monitor.record_failure(uid=2)
        assert monitor._miner_health[2].consecutive_failures == 2
        monitor.record_response(uid=2, anomaly_score=0.5, latency=0.1, current_step=5)
        assert monitor._miner_health[2].consecutive_failures == 0

    def test_record_response_updates_running_avg_latency(self):
        """avg_latency is a running average across responses."""
        monitor = MicrogliaMonitor()
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        assert monitor._miner_health[1].avg_latency == pytest.approx(0.1)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.3, current_step=2)
        assert monitor._miner_health[1].avg_latency == pytest.approx(0.2)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.5, current_step=3)
        assert monitor._miner_health[1].avg_latency == pytest.approx(0.3)

    def test_record_response_caps_anomaly_scores(self):
        """last_anomaly_scores is capped at max_score_window."""
        monitor = MicrogliaMonitor(max_score_window=3)
        for i in range(5):
            monitor.record_response(uid=1, anomaly_score=float(i), latency=0.1, current_step=i)
        scores = monitor._miner_health[1].last_anomaly_scores
        assert len(scores) == 3
        assert scores == [2.0, 3.0, 4.0]

    def test_record_failure_creates_health_state(self):
        """record_failure creates MinerHealthState for new UID."""
        monitor = MicrogliaMonitor()
        monitor.record_failure(uid=7)
        assert monitor._miner_health[7].consecutive_failures == 1

    def test_record_failure_increments_consecutive(self):
        """record_failure increments consecutive_failures."""
        monitor = MicrogliaMonitor()
        monitor.record_failure(uid=3)
        monitor.record_failure(uid=3)
        monitor.record_failure(uid=3)
        assert monitor._miner_health[3].consecutive_failures == 3


class TestMicrogliaMonitorDetection:
    """Tests for detect_inactive, detect_stale, detect_deregistration_candidates."""

    def test_detect_inactive_finds_old_miners(self):
        """Miners with (current_step - last_response_step) > threshold are inactive."""
        monitor = MicrogliaMonitor(inactive_threshold=10)
        # Miner responded at step 5
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=5)
        # At step 20, miner 1 has been inactive for 15 steps (> 10)
        inactive = monitor.detect_inactive(current_step=20)
        assert 1 in inactive

    def test_detect_inactive_excludes_recent_miners(self):
        """Miners who responded recently are NOT inactive."""
        monitor = MicrogliaMonitor(inactive_threshold=10)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=15)
        inactive = monitor.detect_inactive(current_step=20)
        assert 1 not in inactive

    def test_detect_inactive_never_responded_miners(self):
        """Miners with response_count=0 become inactive after threshold steps."""
        monitor = MicrogliaMonitor(inactive_threshold=10)
        # Create a miner entry that never responded (e.g., from record_failure)
        monitor.record_failure(uid=99)
        inactive = monitor.detect_inactive(current_step=15)
        assert 99 in inactive

    def test_detect_stale_identical_scores(self):
        """Miner with identical anomaly_scores (within tolerance) is stale."""
        monitor = MicrogliaMonitor(stale_threshold=5)
        for i in range(5):
            monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=i)
        assert monitor.detect_stale(uid=1) is True

    def test_detect_stale_varying_scores(self):
        """Miner with varying anomaly_scores is NOT stale."""
        monitor = MicrogliaMonitor(stale_threshold=5)
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, score in enumerate(scores):
            monitor.record_response(uid=1, anomaly_score=score, latency=0.1, current_step=i)
        assert monitor.detect_stale(uid=1) is False

    def test_detect_stale_insufficient_history(self):
        """Miner with fewer scores than stale_threshold is NOT stale."""
        monitor = MicrogliaMonitor(stale_threshold=5)
        for i in range(3):
            monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=i)
        assert monitor.detect_stale(uid=1) is False

    def test_detect_stale_within_tolerance(self):
        """Scores within 1e-6 tolerance are considered identical (stale)."""
        monitor = MicrogliaMonitor(stale_threshold=5)
        scores = [0.5, 0.5 + 1e-7, 0.5 - 1e-7, 0.5 + 5e-7, 0.5]
        for i, score in enumerate(scores):
            monitor.record_response(uid=1, anomaly_score=score, latency=0.1, current_step=i)
        assert monitor.detect_stale(uid=1) is True

    def test_detect_stale_unknown_uid(self):
        """Unknown UID returns False (not stale)."""
        monitor = MicrogliaMonitor()
        assert monitor.detect_stale(uid=999) is False

    def test_detect_deregistration_candidates(self):
        """Miners inactive > deregistration_threshold are candidates."""
        monitor = MicrogliaMonitor(deregistration_threshold=50)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        candidates = monitor.detect_deregistration_candidates(current_step=60)
        assert 1 in candidates

    def test_detect_deregistration_excludes_recent(self):
        """Miners active within deregistration_threshold are NOT candidates."""
        monitor = MicrogliaMonitor(deregistration_threshold=50)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=40)
        candidates = monitor.detect_deregistration_candidates(current_step=60)
        assert 1 not in candidates


class TestAlertGeneration:
    """Tests for generate_alerts."""

    def test_inactive_miner_alert(self):
        """generate_alerts produces MINER_INACTIVE alert for inactive miner."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        alerts = monitor.generate_alerts(current_step=10)
        inactive_alerts = [a for a in alerts if a["type"] == AlertType.MINER_INACTIVE.value]
        assert len(inactive_alerts) >= 1
        assert inactive_alerts[0]["uid"] == 1
        assert inactive_alerts[0]["step"] == 10
        assert "message" in inactive_alerts[0]

    def test_stale_miner_alert(self):
        """generate_alerts produces MINER_STALE alert for stale miner."""
        monitor = MicrogliaMonitor(stale_threshold=5, inactive_threshold=100)
        for i in range(5):
            monitor.record_response(uid=2, anomaly_score=0.5, latency=0.1, current_step=i + 1)
        alerts = monitor.generate_alerts(current_step=5)
        stale_alerts = [a for a in alerts if a["type"] == AlertType.MINER_STALE.value]
        assert len(stale_alerts) >= 1
        assert stale_alerts[0]["uid"] == 2

    def test_deregistration_candidate_alert(self):
        """generate_alerts produces DEREGISTRATION_CANDIDATE for long-inactive miner."""
        monitor = MicrogliaMonitor(deregistration_threshold=50, inactive_threshold=5)
        monitor.record_response(uid=3, anomaly_score=0.5, latency=0.1, current_step=1)
        alerts = monitor.generate_alerts(current_step=55)
        dereg_alerts = [a for a in alerts if a["type"] == AlertType.DEREGISTRATION_CANDIDATE.value]
        assert len(dereg_alerts) >= 1
        assert dereg_alerts[0]["uid"] == 3

    def test_alert_dict_structure(self):
        """Alert dicts contain type, uid, step, message keys."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        alerts = monitor.generate_alerts(current_step=10)
        assert len(alerts) > 0
        alert = alerts[0]
        assert "type" in alert
        assert "uid" in alert
        assert "step" in alert
        assert "message" in alert


class TestAlertDeduplication:
    """Tests for alert deduplication within cooldown period."""

    def test_duplicate_alert_suppressed_within_cooldown(self):
        """Same alert type+uid within cooldown period is suppressed."""
        monitor = MicrogliaMonitor(inactive_threshold=5, alert_cooldown=10)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)

        # First alert fires
        alerts1 = monitor.generate_alerts(current_step=10)
        inactive1 = [
            a for a in alerts1
            if a["type"] == AlertType.MINER_INACTIVE.value and a["uid"] == 1
        ]
        assert len(inactive1) == 1

        # Same alert within cooldown: suppressed
        alerts2 = monitor.generate_alerts(current_step=15)
        inactive2 = [
            a for a in alerts2
            if a["type"] == AlertType.MINER_INACTIVE.value and a["uid"] == 1
        ]
        assert len(inactive2) == 0

    def test_duplicate_alert_fires_after_cooldown(self):
        """Same alert type+uid AFTER cooldown period fires again."""
        monitor = MicrogliaMonitor(inactive_threshold=5, alert_cooldown=10)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)

        # First alert fires at step 10
        alerts1 = monitor.generate_alerts(current_step=10)
        inactive1 = [
            a for a in alerts1
            if a["type"] == AlertType.MINER_INACTIVE.value and a["uid"] == 1
        ]
        assert len(inactive1) == 1

        # After cooldown (step 10 + cooldown 10 = step 20), fires again at step 25
        alerts3 = monitor.generate_alerts(current_step=25)
        inactive3 = [
            a for a in alerts3
            if a["type"] == AlertType.MINER_INACTIVE.value and a["uid"] == 1
        ]
        assert len(inactive3) == 1

    def test_different_uids_not_deduplicated(self):
        """Different UIDs with same alert type are NOT deduplicated."""
        monitor = MicrogliaMonitor(inactive_threshold=5, alert_cooldown=10)
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        monitor.record_response(uid=2, anomaly_score=0.5, latency=0.1, current_step=1)

        alerts = monitor.generate_alerts(current_step=10)
        inactive_uids = [a["uid"] for a in alerts if a["type"] == AlertType.MINER_INACTIVE.value]
        assert 1 in inactive_uids
        assert 2 in inactive_uids


class TestHealthMetrics:
    """Tests for get_health_metrics."""

    def test_basic_health_metrics(self):
        """get_health_metrics returns SubnetHealthMetrics with computed values."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # 2 miners responded recently, 1 is inactive
        monitor.record_response(uid=0, anomaly_score=0.5, latency=0.1, current_step=9)
        monitor.record_response(uid=1, anomaly_score=0.6, latency=0.2, current_step=9)
        monitor.record_response(uid=2, anomaly_score=0.5, latency=0.1, current_step=1)  # inactive

        metrics = monitor.get_health_metrics(
            n_total=3, score_history={}, current_step=10
        )
        assert isinstance(metrics, SubnetHealthMetrics)
        assert metrics.inactive_miners >= 1
        assert metrics.active_miners >= 1

    def test_inflammation_score_calculation(self):
        """inflammation_score = (inactive + stale) / total, clamped 0-1."""
        monitor = MicrogliaMonitor(inactive_threshold=5, stale_threshold=3)
        # 1 active miner
        monitor.record_response(uid=0, anomaly_score=0.5, latency=0.1, current_step=9)
        # 1 inactive miner
        monitor.record_response(uid=1, anomaly_score=0.5, latency=0.1, current_step=1)
        # 1 stale miner (identical scores)
        for i in range(3):
            monitor.record_response(uid=2, anomaly_score=0.5, latency=0.1, current_step=i + 8)

        metrics = monitor.get_health_metrics(
            n_total=3, score_history={}, current_step=10
        )
        # inactive: 1, stale: 1 (uid 2 is stale but active)
        # inflammation = (inactive + stale) / total
        assert 0.0 <= metrics.inflammation_score <= 1.0

    def test_threat_level_low(self):
        """inflammation < 0.1 -> threat_level 'low'."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # All miners active
        for uid in range(10):
            monitor.record_response(
                uid=uid, anomaly_score=float(uid) * 0.1,
                latency=0.1, current_step=9,
            )

        metrics = monitor.get_health_metrics(
            n_total=10, score_history={}, current_step=10
        )
        assert metrics.threat_level == "low"

    def test_threat_level_medium(self):
        """inflammation 0.1-0.3 -> threat_level 'medium'."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # 8 active, 2 inactive -> inflammation = 2/10 = 0.2
        for uid in range(8):
            monitor.record_response(
                uid=uid, anomaly_score=float(uid) * 0.1,
                latency=0.1, current_step=9,
            )
        for uid in range(8, 10):
            monitor.record_response(uid=uid, anomaly_score=0.5, latency=0.1, current_step=1)

        metrics = monitor.get_health_metrics(
            n_total=10, score_history={}, current_step=10
        )
        assert metrics.threat_level == "medium"

    def test_threat_level_high(self):
        """inflammation 0.3-0.6 -> threat_level 'high'."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # 5 active, 5 inactive -> inflammation = 5/10 = 0.5
        for uid in range(5):
            monitor.record_response(
                uid=uid, anomaly_score=float(uid) * 0.1,
                latency=0.1, current_step=9,
            )
        for uid in range(5, 10):
            monitor.record_response(uid=uid, anomaly_score=0.5, latency=0.1, current_step=1)

        metrics = monitor.get_health_metrics(
            n_total=10, score_history={}, current_step=10
        )
        assert metrics.threat_level == "high"

    def test_threat_level_critical(self):
        """inflammation >= 0.6 -> threat_level 'critical'."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # 2 active, 8 inactive -> inflammation = 8/10 = 0.8
        for uid in range(2):
            monitor.record_response(
                uid=uid, anomaly_score=float(uid) * 0.1,
                latency=0.1, current_step=9,
            )
        for uid in range(2, 10):
            monitor.record_response(uid=uid, anomaly_score=0.5, latency=0.1, current_step=1)

        metrics = monitor.get_health_metrics(
            n_total=10, score_history={}, current_step=10
        )
        assert metrics.threat_level == "critical"

    def test_population_diversity_index_default(self):
        """population_diversity_index = 0.5 when no score_history."""
        monitor = MicrogliaMonitor()
        metrics = monitor.get_health_metrics(
            n_total=10, score_history={}, current_step=10
        )
        assert metrics.population_diversity_index == pytest.approx(0.5)

    def test_population_diversity_index_with_history(self):
        """population_diversity_index computed from score_history std dev."""
        monitor = MicrogliaMonitor()
        # Create diverse score histories
        score_history = {
            0: [0.1, 0.2, 0.3, 0.4, 0.5],
            1: [0.9, 0.8, 0.7, 0.6, 0.5],
        }
        metrics = monitor.get_health_metrics(
            n_total=2, score_history=score_history, current_step=10
        )
        # Should be > 0 since there is variance
        assert 0.0 <= metrics.population_diversity_index <= 1.0
        assert metrics.population_diversity_index != pytest.approx(0.5)

    def test_population_diversity_index_uniform(self):
        """Uniform scores across miners -> low diversity index."""
        monitor = MicrogliaMonitor()
        # All miners have identical score histories -> zero std dev
        score_history = {
            0: [0.5, 0.5, 0.5, 0.5, 0.5],
            1: [0.5, 0.5, 0.5, 0.5, 0.5],
        }
        metrics = monitor.get_health_metrics(
            n_total=2, score_history=score_history, current_step=10
        )
        assert metrics.population_diversity_index == pytest.approx(0.0, abs=0.01)

    def test_active_miners_count(self):
        """active_miners excludes inactive miners."""
        monitor = MicrogliaMonitor(inactive_threshold=5)
        # 3 active
        for uid in range(3):
            monitor.record_response(uid=uid, anomaly_score=0.5, latency=0.1, current_step=9)
        # 2 inactive
        for uid in range(3, 5):
            monitor.record_response(uid=uid, anomaly_score=0.5, latency=0.1, current_step=1)

        metrics = monitor.get_health_metrics(
            n_total=5, score_history={}, current_step=10
        )
        assert metrics.active_miners == 3
        assert metrics.inactive_miners == 2


class TestCoordinatedAttackDetection:
    """Tests for detect_coordinated_attack (GLIA-03)."""

    def test_first_call_returns_false(self):
        """First call initializes baseline and returns False."""
        monitor = MicrogliaMonitor()
        scores = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        assert monitor.detect_coordinated_attack(scores, n_total=4) is False

    def test_no_drop_returns_false(self):
        """Stable scores return False."""
        monitor = MicrogliaMonitor()
        scores1 = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        monitor.detect_coordinated_attack(scores1, n_total=4)  # baseline
        scores2 = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        assert monitor.detect_coordinated_attack(scores2, n_total=4) is False

    def test_small_drops_return_false(self):
        """Small drops (<50% for <30% miners) return False."""
        monitor = MicrogliaMonitor()
        scores1 = np.array([0.8, 0.7, 0.9, 0.6, 0.5, 0.8, 0.7, 0.9, 0.6, 0.5], dtype=np.float32)
        monitor.detect_coordinated_attack(scores1, n_total=10)  # baseline
        # Drop 1 miner (10%) by 60% -- below 30% threshold
        scores2 = scores1.copy()
        scores2[0] = 0.3
        assert monitor.detect_coordinated_attack(scores2, n_total=10) is False

    def test_coordinated_drop_triggers_true(self):
        """Score drops >50% for >30% of miners triggers True."""
        monitor = MicrogliaMonitor(attack_drop_ratio=0.5, attack_miner_ratio=0.3)
        scores1 = np.array([0.8, 0.7, 0.9, 0.6, 0.5, 0.8, 0.7, 0.9, 0.6, 0.5], dtype=np.float32)
        monitor.detect_coordinated_attack(scores1, n_total=10)  # baseline
        # Drop 4 miners (40% > 30%) by >50%
        scores2 = scores1.copy()
        scores2[0] = 0.1  # 87.5% drop
        scores2[1] = 0.1  # 85.7% drop
        scores2[2] = 0.1  # 88.9% drop
        scores2[3] = 0.1  # 83.3% drop
        assert monitor.detect_coordinated_attack(scores2, n_total=10) is True

    def test_zero_previous_scores_no_false_trigger(self):
        """Miners with zero previous scores do not trigger false positives."""
        monitor = MicrogliaMonitor()
        scores1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        monitor.detect_coordinated_attack(scores1, n_total=4)  # baseline
        scores2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        # Scores went up from zero -- no drop
        assert monitor.detect_coordinated_attack(scores2, n_total=4) is False


class TestRegistrationSurge:
    """Tests for detect_registration_surge (GLIA-03)."""

    def test_first_call_no_surge(self):
        """First call sets known hotkeys, does not trigger if below threshold."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        hotkeys = ["hk1", "hk2"]
        assert monitor.detect_registration_surge(hotkeys, current_step=1) is False

    def test_first_call_with_surge(self):
        """First call with many hotkeys triggers surge (all are new)."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        hotkeys = ["hk1", "hk2", "hk3", "hk4"]
        assert monitor.detect_registration_surge(hotkeys, current_step=1) is True

    def test_same_hotkeys_no_surge(self):
        """Same hotkeys repeated do not trigger surge."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        hotkeys = ["hk1", "hk2", "hk3"]
        monitor.detect_registration_surge(hotkeys, current_step=1)
        assert monitor.detect_registration_surge(hotkeys, current_step=2) is False

    def test_one_new_hotkey_no_surge(self):
        """One new hotkey does not trigger surge (below threshold 3)."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        monitor.detect_registration_surge(["hk1", "hk2"], current_step=1)
        assert monitor.detect_registration_surge(["hk1", "hk2", "hk3"], current_step=2) is False

    def test_three_new_hotkeys_triggers_surge(self):
        """Three new hotkeys triggers surge (meets threshold 3)."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        monitor.detect_registration_surge(["hk1", "hk2"], current_step=1)
        new_hotkeys = ["hk1", "hk2", "hk3", "hk4", "hk5"]
        assert monitor.detect_registration_surge(new_hotkeys, current_step=2) is True

    def test_hotkeys_replaced_count_as_new(self):
        """When old hotkeys are replaced with new ones, new ones count."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        monitor.detect_registration_surge(["hk1", "hk2", "hk3"], current_step=1)
        # Replace all 3 with new ones
        assert monitor.detect_registration_surge(["hk4", "hk5", "hk6"], current_step=2) is True


class TestWebhook:
    """Tests for send_webhook."""

    @pytest.mark.asyncio
    async def test_webhook_none_url_returns_immediately(self):
        """send_webhook with no URL returns without error."""
        monitor = MicrogliaMonitor(webhook_url=None)
        # Should not raise
        await monitor.send_webhook([{"type": "test", "message": "test"}])

    @pytest.mark.asyncio
    async def test_webhook_posts_payload(self):
        """send_webhook sends correct JSON payload to URL."""
        monitor = MicrogliaMonitor(webhook_url="http://example.com/hook")
        alerts = [{"type": "test", "uid": 1, "step": 10, "message": "test"}]

        with patch("antigence_subnet.validator.microglia.urllib.request.urlopen") as mock_urlopen:
            await monitor.send_webhook(alerts)
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://example.com/hook"
            assert req.get_header("Content-type") == "application/json"

    @pytest.mark.asyncio
    async def test_webhook_error_logged_not_raised(self):
        """send_webhook logs error on failure but does not raise."""
        monitor = MicrogliaMonitor(webhook_url="http://example.com/hook")
        alerts = [{"type": "test"}]

        with (
            patch(
                "antigence_subnet.validator.microglia.urllib.request.urlopen",
                side_effect=Exception("Connection refused"),
            ),
            patch("bittensor.logging.warning") as mock_warn,
        ):
            await monitor.send_webhook(alerts)
            mock_warn.assert_called_once()
            assert "Webhook send failed" in str(mock_warn.call_args)


class TestRunSurveillanceCycle:
    """Tests for run_surveillance_cycle."""

    def test_returns_health_metrics(self):
        """run_surveillance_cycle returns SubnetHealthMetrics."""
        monitor = MicrogliaMonitor()
        scores = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        metrics = monitor.run_surveillance_cycle(
            scores=scores,
            score_history={},
            hotkeys=["hk1", "hk2", "hk3", "hk4"],
            n_total=4,
            current_step=10,
        )
        assert isinstance(metrics, SubnetHealthMetrics)

    def test_detects_coordinated_attack_in_cycle(self):
        """Surveillance cycle detects coordinated attack."""
        monitor = MicrogliaMonitor(attack_drop_ratio=0.5, attack_miner_ratio=0.3)
        scores1 = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        # First cycle sets baseline
        monitor.run_surveillance_cycle(
            scores=scores1, score_history={}, hotkeys=["hk1", "hk2", "hk3", "hk4"],
            n_total=4, current_step=10,
        )
        # Second cycle with coordinated drop (all 4 = 100% > 30%)
        scores2 = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        # Patch bt.logging.warning to capture alerts
        with patch("bittensor.logging.warning") as mock_warn:
            monitor.run_surveillance_cycle(
                scores=scores2, score_history={}, hotkeys=["hk1", "hk2", "hk3", "hk4"],
                n_total=4, current_step=20,
            )
            # Should have logged coordinated attack
            warn_messages = [str(c) for c in mock_warn.call_args_list]
            assert any("COORDINATED_ATTACK" in m for m in warn_messages)

    def test_detects_registration_surge_in_cycle(self):
        """Surveillance cycle detects registration surge."""
        monitor = MicrogliaMonitor(surge_threshold=3)
        scores = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        # First cycle sets known hotkeys
        monitor.run_surveillance_cycle(
            scores=scores, score_history={}, hotkeys=["hk1"],
            n_total=4, current_step=10,
        )
        # Second cycle with 3+ new hotkeys
        with patch("bittensor.logging.warning") as mock_warn:
            monitor.run_surveillance_cycle(
                scores=scores, score_history={}, hotkeys=["hk1", "hk2", "hk3", "hk4"],
                n_total=4, current_step=20,
            )
            warn_messages = [str(c) for c in mock_warn.call_args_list]
            assert any("REGISTRATION_SURGE" in m for m in warn_messages)

    def test_combines_all_alert_types(self):
        """Surveillance cycle combines miner-level and network-level alerts."""
        monitor = MicrogliaMonitor(
            inactive_threshold=5,
            surge_threshold=2,
            attack_drop_ratio=0.5,
            attack_miner_ratio=0.3,
        )
        # Set up an inactive miner
        monitor.record_response(uid=0, anomaly_score=0.5, latency=0.1, current_step=1)

        scores = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)
        # First cycle baseline
        monitor.run_surveillance_cycle(
            scores=scores, score_history={}, hotkeys=["hk1"],
            n_total=4, current_step=10,
        )

        # Second cycle: attack + surge + inactive miner at step 20
        dropped_scores = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        with patch("bittensor.logging.warning"):
            metrics = monitor.run_surveillance_cycle(
                scores=dropped_scores,
                score_history={},
                hotkeys=["hk1", "hk2", "hk3"],
                n_total=4,
                current_step=20,
            )
        # Should detect inactive miner (uid 0 last responded at step 1)
        assert metrics.inactive_miners >= 1

    def test_health_metrics_valid_fields(self):
        """Surveillance cycle returns metrics with valid field ranges."""
        monitor = MicrogliaMonitor()
        scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        metrics = monitor.run_surveillance_cycle(
            scores=scores,
            score_history={0: [0.5, 0.6, 0.7, 0.8, 0.9], 1: [0.1, 0.2, 0.3, 0.4, 0.5]},
            hotkeys=["hk1", "hk2", "hk3", "hk4"],
            n_total=4,
            current_step=10,
        )
        assert 0.0 <= metrics.inflammation_score <= 1.0
        assert metrics.threat_level in ("low", "medium", "high", "critical")
        assert 0.0 <= metrics.population_diversity_index <= 1.0
        assert metrics.active_miners >= 0
        assert metrics.inactive_miners >= 0
