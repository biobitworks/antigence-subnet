"""Tests for MetagraphMonitor -- metagraph snapshot comparison and anomaly detection (NET-05).

Covers mass registration detection, mass deregistration detection,
sudden stake shift detection, snapshot baseline behavior, simultaneous
anomaly firing, and configurable thresholds.
"""

import numpy as np
import pytest

from antigence_subnet.validator.metagraph_monitor import (
    MetagraphAnomaly,
    MetagraphMonitor,
    MetagraphSnapshot,
)


class TestMetagraphSnapshotBaseline:
    """Test 1: First call records snapshot and returns empty anomaly list."""

    def test_first_call_returns_empty(self):
        """First check_anomalies call records baseline snapshot and returns []."""
        monitor = MetagraphMonitor()
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes = np.ones(10, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes, n=10, step=1)
        assert anomalies == []

    def test_first_call_stores_previous_snapshot(self):
        """First call stores snapshot for future comparisons."""
        monitor = MetagraphMonitor()
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes, n=10, step=1)
        assert monitor._previous_snapshot is not None
        assert isinstance(monitor._previous_snapshot, MetagraphSnapshot)


class TestMassRegistrationDetection:
    """Test 2: Adding 4 new hotkeys between snapshots triggers MASS_REGISTRATION."""

    def test_four_new_hotkeys_triggers_mass_registration(self):
        """4 new hotkeys (>3 threshold) triggers MASS_REGISTRATION anomaly."""
        monitor = MetagraphMonitor()
        # Baseline: 10 hotkeys
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes_v1, n=10, step=1)

        # Add 4 new hotkeys (total 14)
        hotkeys_v2 = hotkeys_v1 + [f"new-hk-{i}" for i in range(4)]
        stakes_v2 = np.ones(14, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=14, step=2)

        mass_reg = [a for a in anomalies if a.anomaly_type == "MASS_REGISTRATION"]
        assert len(mass_reg) == 1
        assert mass_reg[0].severity == "high"
        assert "4" in mass_reg[0].details
        assert mass_reg[0].step == 2


class TestMassDeregistrationDetection:
    """Test 3: Removing 4 hotkeys between snapshots triggers MASS_DEREGISTRATION."""

    def test_four_removed_hotkeys_triggers_mass_deregistration(self):
        """4 removed hotkeys (>3 threshold) triggers MASS_DEREGISTRATION anomaly."""
        monitor = MetagraphMonitor()
        # Baseline: 10 hotkeys
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes_v1, n=10, step=1)

        # Remove 4 hotkeys (keep first 6)
        hotkeys_v2 = hotkeys_v1[:6]
        stakes_v2 = np.ones(6, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=6, step=2)

        mass_dereg = [a for a in anomalies if a.anomaly_type == "MASS_DEREGISTRATION"]
        assert len(mass_dereg) == 1
        assert mass_dereg[0].severity == "high"
        assert "4" in mass_dereg[0].details
        assert mass_dereg[0].step == 2


class TestStakeShiftDetection:
    """Test 4: Moving >10% total stake between snapshots triggers STAKE_SHIFT."""

    def test_twenty_percent_stake_shift_triggers(self):
        """20% total stake change triggers STAKE_SHIFT anomaly."""
        monitor = MetagraphMonitor()
        # Baseline: total stake = 10 * 100000 = 1,000,000
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v1, n=10, step=1)

        # Increase total stake to 1,200,000 (20% shift)
        stakes_v2 = np.ones(10, dtype=np.float32) * 120000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v2, n=10, step=2)

        stake_shift = [a for a in anomalies if a.anomaly_type == "STAKE_SHIFT"]
        assert len(stake_shift) == 1
        assert stake_shift[0].severity == "medium"
        assert "20.0%" in stake_shift[0].details
        assert stake_shift[0].step == 2


class TestNormalChanges:
    """Test 5: Normal changes (1 new hotkey, <10% stake movement) return empty list."""

    def test_one_new_hotkey_no_anomaly(self):
        """1 new hotkey (<=3 threshold) does not trigger anomaly."""
        monitor = MetagraphMonitor()
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes_v1, n=10, step=1)

        hotkeys_v2 = hotkeys_v1 + ["new-hk-0"]
        stakes_v2 = np.ones(11, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=11, step=2)
        assert anomalies == []

    def test_small_stake_movement_no_anomaly(self):
        """5% stake movement (<10% threshold) does not trigger anomaly."""
        monitor = MetagraphMonitor()
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v1, n=10, step=1)

        stakes_v2 = np.ones(10, dtype=np.float32) * 105000.0  # 5% increase
        anomalies = monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v2, n=10, step=2)
        assert anomalies == []

    def test_no_changes_no_anomaly(self):
        """Identical metagraph produces no anomalies."""
        monitor = MetagraphMonitor()
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes, n=10, step=1)

        anomalies = monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes.copy(), n=10, step=2)
        assert anomalies == []


class TestSimultaneousAnomalies:
    """Test 6: Multiple anomalies can fire simultaneously."""

    def test_registration_and_stake_shift_together(self):
        """Mass registrations + stake shift fire simultaneously."""
        monitor = MetagraphMonitor()
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes_v1, n=10, step=1)

        # Add 4 new hotkeys AND increase stake by 20%
        hotkeys_v2 = hotkeys_v1 + [f"new-hk-{i}" for i in range(4)]
        stakes_v2 = np.ones(14, dtype=np.float32) * 120000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=14, step=2)

        types = [a.anomaly_type for a in anomalies]
        assert "MASS_REGISTRATION" in types
        assert "STAKE_SHIFT" in types
        assert len(anomalies) >= 2

    def test_all_three_anomalies_together(self):
        """Mass registrations, deregistrations, and stake shift simultaneously."""
        monitor = MetagraphMonitor()
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes_v1, n=10, step=1)

        # Remove 4 old, add 5 new, change stake by 25%
        hotkeys_v2 = [f"hk-{i}" for i in range(6)] + [f"brand-new-{i}" for i in range(5)]
        stakes_v2 = np.ones(11, dtype=np.float32) * 125000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=11, step=2)

        types = [a.anomaly_type for a in anomalies]
        assert "MASS_REGISTRATION" in types
        assert "MASS_DEREGISTRATION" in types
        assert "STAKE_SHIFT" in types
        assert len(anomalies) == 3


class TestConfigurableThresholds:
    """Test 7: Thresholds are configurable via constructor args."""

    def test_custom_registration_threshold(self):
        """Custom registration_threshold=5 requires 6 new hotkeys."""
        monitor = MetagraphMonitor(registration_threshold=5)
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes, n=10, step=1)

        # 4 new: should NOT trigger (<=5)
        hotkeys_v2 = hotkeys_v1 + [f"new-{i}" for i in range(4)]
        stakes_v2 = np.ones(14, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=14, step=2)
        assert all(a.anomaly_type != "MASS_REGISTRATION" for a in anomalies)

        # 6 new: should trigger (>5)
        hotkeys_v3 = hotkeys_v2 + [f"extra-{i}" for i in range(6)]
        stakes_v3 = np.ones(20, dtype=np.float32) * 100000.0
        anomalies2 = monitor.check_anomalies(hotkeys=hotkeys_v3, stakes=stakes_v3, n=20, step=3)
        assert any(a.anomaly_type == "MASS_REGISTRATION" for a in anomalies2)

    def test_custom_deregistration_threshold(self):
        """Custom deregistration_threshold=1 triggers on 2 removals."""
        monitor = MetagraphMonitor(deregistration_threshold=1)
        hotkeys_v1 = [f"hk-{i}" for i in range(10)]
        stakes = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys_v1, stakes=stakes, n=10, step=1)

        hotkeys_v2 = hotkeys_v1[:8]  # Remove 2 (>1)
        stakes_v2 = np.ones(8, dtype=np.float32) * 100000.0
        anomalies = monitor.check_anomalies(hotkeys=hotkeys_v2, stakes=stakes_v2, n=8, step=2)
        assert any(a.anomaly_type == "MASS_DEREGISTRATION" for a in anomalies)

    def test_custom_stake_shift_threshold(self):
        """Custom stake_shift_pct=0.05 triggers on 6% movement."""
        monitor = MetagraphMonitor(stake_shift_pct=0.05)
        hotkeys = [f"hk-{i}" for i in range(10)]
        stakes_v1 = np.ones(10, dtype=np.float32) * 100000.0
        monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v1, n=10, step=1)

        stakes_v2 = np.ones(10, dtype=np.float32) * 106000.0  # 6% increase
        anomalies = monitor.check_anomalies(hotkeys=hotkeys, stakes=stakes_v2, n=10, step=2)
        assert any(a.anomaly_type == "STAKE_SHIFT" for a in anomalies)

    def test_default_thresholds(self):
        """Default thresholds: registration=3, deregistration=3, stake_shift=0.10."""
        monitor = MetagraphMonitor()
        assert monitor.registration_threshold == 3
        assert monitor.deregistration_threshold == 3
        assert monitor.stake_shift_pct == pytest.approx(0.10)


class TestMetagraphSnapshot:
    """Tests for MetagraphSnapshot dataclass."""

    def test_snapshot_creation(self):
        """MetagraphSnapshot stores hotkeys, stakes, total_stake, n, step."""
        snapshot = MetagraphSnapshot(
            hotkeys=frozenset(["hk1", "hk2"]),
            stakes={"hk1": 100.0, "hk2": 200.0},
            total_stake=300.0,
            n=2,
            step=5,
        )
        assert snapshot.hotkeys == frozenset(["hk1", "hk2"])
        assert snapshot.stakes == {"hk1": 100.0, "hk2": 200.0}
        assert snapshot.total_stake == 300.0
        assert snapshot.n == 2
        assert snapshot.step == 5


class TestMetagraphAnomaly:
    """Tests for MetagraphAnomaly dataclass."""

    def test_anomaly_creation(self):
        """MetagraphAnomaly stores anomaly_type, details, severity, step."""
        anomaly = MetagraphAnomaly(
            anomaly_type="MASS_REGISTRATION",
            details="4 new registrations in one sync",
            severity="high",
            step=10,
        )
        assert anomaly.anomaly_type == "MASS_REGISTRATION"
        assert anomaly.details == "4 new registrations in one sync"
        assert anomaly.severity == "high"
        assert anomaly.step == 10


class TestTakeSnapshot:
    """Tests for take_snapshot method."""

    def test_take_snapshot_creates_correct_snapshot(self):
        """take_snapshot builds snapshot from hotkeys and stakes array."""
        monitor = MetagraphMonitor()
        hotkeys = ["hk1", "hk2", "hk3"]
        stakes = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        snapshot = monitor.take_snapshot(hotkeys=hotkeys, stakes=stakes, n=3, step=5)

        assert snapshot.hotkeys == frozenset(["hk1", "hk2", "hk3"])
        assert snapshot.stakes["hk1"] == pytest.approx(100.0)
        assert snapshot.stakes["hk2"] == pytest.approx(200.0)
        assert snapshot.stakes["hk3"] == pytest.approx(300.0)
        assert snapshot.total_stake == pytest.approx(600.0)
        assert snapshot.n == 3
        assert snapshot.step == 5

    def test_take_snapshot_handles_mismatched_lengths(self):
        """take_snapshot handles hotkeys longer than stakes array."""
        monitor = MetagraphMonitor()
        hotkeys = ["hk1", "hk2", "hk3", "hk4"]
        stakes = np.array([100.0, 200.0], dtype=np.float32)
        snapshot = monitor.take_snapshot(hotkeys=hotkeys, stakes=stakes, n=4, step=1)

        # Should only map first 2 (min of len(hotkeys), len(stakes))
        assert len(snapshot.stakes) == 2
        assert snapshot.total_stake == pytest.approx(300.0)
