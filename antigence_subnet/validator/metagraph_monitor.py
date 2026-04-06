"""
Metagraph anomaly detection -- snapshot comparison across metagraph syncs.

Monitors network-level attack patterns by comparing metagraph state between
sync cycles: mass registrations, coordinated deregistrations, and sudden
stake movements. Reports anomalies for logging and webhook alerting through
the existing MicrogliaMonitor infrastructure.

NET-05: Metagraph anomaly detection

Usage:
    monitor = MetagraphMonitor()
    anomalies = monitor.check_anomalies(
        hotkeys=list(metagraph.hotkeys),
        stakes=metagraph.S,
        n=metagraph.n,
        step=validator.step,
    )
    for anomaly in anomalies:
        bt.logging.warning(f"[MetagraphMonitor] {anomaly.anomaly_type}: {anomaly.details}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetagraphSnapshot:
    """Immutable snapshot of metagraph state at a particular sync step.

    Attributes:
        hotkeys: Set of hotkey addresses currently registered.
        stakes: Mapping of hotkey -> stake value.
        total_stake: Sum of all stakes.
        n: Number of neurons in the metagraph.
        step: Validator step when snapshot was taken.
    """

    hotkeys: frozenset[str]
    stakes: dict[str, float]
    total_stake: float
    n: int
    step: int


@dataclass(frozen=True)
class MetagraphAnomaly:
    """A detected metagraph anomaly.

    Attributes:
        anomaly_type: One of MASS_REGISTRATION, MASS_DEREGISTRATION, STAKE_SHIFT.
        details: Human-readable description of the anomaly.
        severity: "medium" or "high".
        step: Validator step when anomaly was detected.
    """

    anomaly_type: str
    details: str
    severity: str
    step: int


class MetagraphMonitor:
    """Compares metagraph snapshots across syncs to detect network-level anomalies.

    Detects three anomaly types:
    - MASS_REGISTRATION: More than registration_threshold new hotkeys in one sync.
    - MASS_DEREGISTRATION: More than deregistration_threshold hotkeys removed in one sync.
    - STAKE_SHIFT: Total stake changed by more than stake_shift_pct in one sync.

    Args:
        registration_threshold: Max new hotkeys before triggering (default 3).
        deregistration_threshold: Max removed hotkeys before triggering (default 3).
        stake_shift_pct: Max fractional stake change before triggering (default 0.10).
    """

    def __init__(
        self,
        registration_threshold: int = 3,
        deregistration_threshold: int = 3,
        stake_shift_pct: float = 0.10,
    ):
        self.registration_threshold = registration_threshold
        self.deregistration_threshold = deregistration_threshold
        self.stake_shift_pct = stake_shift_pct
        self._previous_snapshot: MetagraphSnapshot | None = None

    def take_snapshot(
        self,
        hotkeys: list[str],
        stakes: np.ndarray,
        n: int,
        step: int,
    ) -> MetagraphSnapshot:
        """Create a MetagraphSnapshot from current metagraph state.

        Args:
            hotkeys: List of hotkey ss58 addresses.
            stakes: Numpy array of stake values (metagraph.S).
            n: Number of neurons.
            step: Current validator step.

        Returns:
            MetagraphSnapshot capturing the current state.
        """
        min_len = min(len(hotkeys), len(stakes))
        stakes_map = {
            hotkeys[i]: float(stakes[i]) for i in range(min_len)
        }
        total_stake = float(np.sum(stakes[:min_len]))

        return MetagraphSnapshot(
            hotkeys=frozenset(hotkeys),
            stakes=stakes_map,
            total_stake=total_stake,
            n=n,
            step=step,
        )

    def check_anomalies(
        self,
        hotkeys: list[str],
        stakes: np.ndarray,
        n: int,
        step: int,
    ) -> list[MetagraphAnomaly]:
        """Compare current metagraph state against previous snapshot for anomalies.

        On the first call, records the baseline snapshot and returns an empty list.
        On subsequent calls, compares current state to previous and detects:
        - Mass registrations (new hotkeys > registration_threshold)
        - Mass deregistrations (removed hotkeys > deregistration_threshold)
        - Stake shifts (total stake change > stake_shift_pct)

        Args:
            hotkeys: Current metagraph hotkeys.
            stakes: Current metagraph stake array (S).
            n: Number of neurons.
            step: Current validator step.

        Returns:
            List of MetagraphAnomaly instances (empty if no anomalies).
        """
        current = self.take_snapshot(hotkeys=hotkeys, stakes=stakes, n=n, step=step)

        # First call: store baseline, return empty
        if self._previous_snapshot is None:
            self._previous_snapshot = current
            return []

        anomalies: list[MetagraphAnomaly] = []
        previous = self._previous_snapshot

        # Detect mass registrations
        new_hotkeys = current.hotkeys - previous.hotkeys
        if len(new_hotkeys) > self.registration_threshold:
            anomalies.append(
                MetagraphAnomaly(
                    anomaly_type="MASS_REGISTRATION",
                    details=f"{len(new_hotkeys)} new registrations in one sync",
                    severity="high",
                    step=step,
                )
            )

        # Detect mass deregistrations
        removed_hotkeys = previous.hotkeys - current.hotkeys
        if len(removed_hotkeys) > self.deregistration_threshold:
            anomalies.append(
                MetagraphAnomaly(
                    anomaly_type="MASS_DEREGISTRATION",
                    details=f"{len(removed_hotkeys)} deregistrations in one sync",
                    severity="high",
                    step=step,
                )
            )

        # Detect stake shifts
        stake_delta = abs(current.total_stake - previous.total_stake)
        stake_shift_ratio = stake_delta / max(previous.total_stake, 1e-6)
        if stake_shift_ratio > self.stake_shift_pct:
            anomalies.append(
                MetagraphAnomaly(
                    anomaly_type="STAKE_SHIFT",
                    details=f"{stake_shift_ratio * 100:.1f}% total stake shifted",
                    severity="medium",
                    step=step,
                )
            )

        # Update previous snapshot for next comparison
        self._previous_snapshot = current
        return anomalies
