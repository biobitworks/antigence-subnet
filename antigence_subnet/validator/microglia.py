"""
Microglia monitor -- per-miner health tracking, detection, alerting, and health metrics.

Implements the biological microglia concept: surveillance cells that monitor
neuron health, detect inactive/stale/deregistering miners, generate alerts
with deduplication, and compute subnet-wide health metrics.

GLIA-01: Per-miner health state tracking
GLIA-02: Inactive/stale/deregistration detection
GLIA-03: Coordinated attack and registration surge detection
GLIA-04: Subnet health metric aggregation

Usage:
    monitor = MicrogliaMonitor()
    monitor.record_response(uid=1, anomaly_score=0.7, latency=0.1, current_step=10)
    alerts = monitor.generate_alerts(current_step=20)
    metrics = monitor.get_health_metrics(n_total=16, score_history={}, current_step=20)

    # Full surveillance cycle (called from validator daemon)
    health = monitor.run_surveillance_cycle(
        scores=validator.scores,
        score_history=validator.score_history,
        hotkeys=list(metagraph.hotkeys),
        n_total=metagraph.n,
        current_step=validator.step,
    )
"""

import json
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum

import bittensor as bt
import numpy as np


class AlertType(str, Enum):
    """Types of alerts the microglia monitor can generate."""

    MINER_INACTIVE = "MINER_INACTIVE"
    MINER_STALE = "MINER_STALE"
    DEREGISTRATION_CANDIDATE = "DEREGISTRATION_CANDIDATE"
    COORDINATED_ATTACK = "COORDINATED_ATTACK"
    REGISTRATION_SURGE = "REGISTRATION_SURGE"
    METAGRAPH_ANOMALY = "METAGRAPH_ANOMALY"


@dataclass
class MinerHealthState:
    """Per-miner health state tracked by the microglia monitor.

    Attributes:
        last_response_step: Step number of the last successful response.
        response_count: Total number of successful responses.
        consecutive_failures: Number of consecutive failures since last success.
        avg_latency: Running average response latency in seconds.
        last_anomaly_scores: Rolling window of recent anomaly scores for stale detection.
    """

    last_response_step: int = 0
    response_count: int = 0
    consecutive_failures: int = 0
    avg_latency: float = 0.0
    last_anomaly_scores: list[float] = field(default_factory=list)


@dataclass
class SubnetHealthMetrics:
    """Aggregated subnet health metrics computed from miner states.

    Attributes:
        inflammation_score: Ratio of unhealthy miners (0-1).
        threat_level: Categorical threat assessment (low/medium/high/critical).
        population_diversity_index: Score diversity across miners (0-1).
        active_miners: Count of currently active miners.
        inactive_miners: Count of inactive miners.
        stale_miners: Count of stale miners.
        deregistration_candidates: Count of miners eligible for deregistration.
    """

    inflammation_score: float
    threat_level: str
    population_diversity_index: float
    active_miners: int
    inactive_miners: int
    stale_miners: int
    deregistration_candidates: int


class MicrogliaMonitor:
    """Surveillance engine for per-miner health tracking and subnet health assessment.

    Tracks individual miner health states, detects inactive/stale/deregistering
    miners, detects coordinated attacks and registration surges, generates
    deduplicated alerts with optional webhook dispatch, and computes
    subnet-wide health metrics.

    Args:
        inactive_threshold: Steps without response before a miner is considered inactive.
        stale_threshold: Number of identical consecutive scores to flag as stale.
        deregistration_threshold: Steps without response before recommending deregistration.
        alert_cooldown: Minimum steps between repeated alerts for the same (type, uid).
        max_score_window: Maximum anomaly scores kept per miner for stale detection.
        attack_drop_ratio: Minimum relative score drop to count a miner in attack detection.
        attack_miner_ratio: Fraction of miners that must drop to trigger coordinated attack alert.
        surge_threshold: Number of new hotkeys in one cycle to trigger registration surge alert.
        webhook_url: Optional URL for HTTP POST alert dispatch (Slack, Discord, etc.).
    """

    def __init__(
        self,
        inactive_threshold: int = 10,
        stale_threshold: int = 5,
        deregistration_threshold: int = 50,
        alert_cooldown: int = 10,
        max_score_window: int = 10,
        attack_drop_ratio: float = 0.5,
        attack_miner_ratio: float = 0.3,
        surge_threshold: int = 3,
        webhook_url: str | None = None,
    ):
        self.inactive_threshold = inactive_threshold
        self.stale_threshold = stale_threshold
        self.deregistration_threshold = deregistration_threshold
        self.alert_cooldown = alert_cooldown
        self.max_score_window = max_score_window

        # Coordinated attack detection (GLIA-03)
        self.attack_drop_ratio = attack_drop_ratio
        self.attack_miner_ratio = attack_miner_ratio

        # Registration surge detection (GLIA-03)
        self.surge_threshold = surge_threshold

        # Webhook alerting (GLIA-04)
        self.webhook_url = webhook_url

        # Per-miner health state: uid -> MinerHealthState
        self._miner_health: dict[int, MinerHealthState] = {}

        # Alert deduplication: (AlertType, uid) -> last_fired_step
        self._alert_history: dict[tuple[AlertType, int], int] = {}

        # Previous scores for coordinated attack drop detection
        self._previous_scores: dict[int, float] = {}

        # Known hotkeys for registration surge detection
        self._known_hotkeys: set[str] = set()

    def _get_or_create_state(self, uid: int) -> MinerHealthState:
        """Get existing MinerHealthState or create a new one for the UID."""
        if uid not in self._miner_health:
            self._miner_health[uid] = MinerHealthState()
        return self._miner_health[uid]

    def record_response(
        self,
        uid: int,
        anomaly_score: float,
        latency: float,
        current_step: int,
    ) -> None:
        """Record a successful miner response.

        Updates last_response_step, increments response_count, resets
        consecutive_failures, updates running avg_latency, and appends
        anomaly_score to the rolling window.

        Args:
            uid: Miner UID.
            anomaly_score: The anomaly score returned by the miner.
            latency: Response latency in seconds.
            current_step: Current validator step number.
        """
        state = self._get_or_create_state(uid)

        state.last_response_step = current_step
        state.response_count += 1
        state.consecutive_failures = 0

        # Running average latency
        if state.response_count == 1:
            state.avg_latency = latency
        else:
            # Incremental mean: new_avg = old_avg + (new - old_avg) / count
            state.avg_latency += (latency - state.avg_latency) / state.response_count

        # Append anomaly score to rolling window, cap at max_score_window
        state.last_anomaly_scores.append(anomaly_score)
        if len(state.last_anomaly_scores) > self.max_score_window:
            state.last_anomaly_scores = state.last_anomaly_scores[
                -self.max_score_window :
            ]

    def record_failure(self, uid: int) -> None:
        """Record a miner failure (timeout, error, no response).

        Increments consecutive_failures for the miner.

        Args:
            uid: Miner UID.
        """
        state = self._get_or_create_state(uid)
        state.consecutive_failures += 1

    def detect_inactive(self, current_step: int) -> list[int]:
        """Detect miners that have not responded within the inactive threshold.

        Returns UIDs where:
        - (current_step - last_response_step) > inactive_threshold AND response_count > 0
        - OR response_count == 0 AND current_step > inactive_threshold (never responded)

        Args:
            current_step: Current validator step number.

        Returns:
            List of inactive miner UIDs.
        """
        inactive: list[int] = []
        for uid, state in self._miner_health.items():
            if state.response_count > 0:
                if (current_step - state.last_response_step) > self.inactive_threshold:
                    inactive.append(uid)
            else:
                # Never responded -- inactive if we've waited long enough
                if current_step > self.inactive_threshold:
                    inactive.append(uid)
        return inactive

    def detect_stale(self, uid: int) -> bool:
        """Detect if a miner is producing stale (unchanging) anomaly scores.

        A miner is stale if the last stale_threshold scores are all within
        1e-6 tolerance of each other.

        Args:
            uid: Miner UID to check.

        Returns:
            True if the miner is stale, False otherwise.
        """
        if uid not in self._miner_health:
            return False

        scores = self._miner_health[uid].last_anomaly_scores
        if len(scores) < self.stale_threshold:
            return False

        # Check the last stale_threshold scores
        window = scores[-self.stale_threshold :]
        reference = window[0]
        return all(abs(s - reference) < 1e-6 for s in window)

    def detect_deregistration_candidates(self, current_step: int) -> list[int]:
        """Detect miners inactive long enough to recommend deregistration.

        Returns UIDs where (current_step - last_response_step) > deregistration_threshold.

        Args:
            current_step: Current validator step number.

        Returns:
            List of deregistration candidate UIDs.
        """
        candidates: list[int] = []
        for uid, state in self._miner_health.items():
            if (current_step - state.last_response_step) > self.deregistration_threshold:
                candidates.append(uid)
        return candidates

    def generate_alerts(self, current_step: int) -> list[dict]:
        """Generate alerts from all detection methods with deduplication.

        Runs inactive, stale, and deregistration detection. For each finding,
        checks if the same (alert_type, uid) was fired within alert_cooldown
        steps. If so, the alert is suppressed. Otherwise, creates an alert dict
        and updates the deduplication history.

        Args:
            current_step: Current validator step number.

        Returns:
            List of alert dicts, each with keys: type, uid, step, message.
        """
        alerts: list[dict] = []

        # Detect inactive miners
        inactive_uids = self.detect_inactive(current_step)
        for uid in inactive_uids:
            self._maybe_emit_alert(
                alerts=alerts,
                alert_type=AlertType.MINER_INACTIVE,
                uid=uid,
                current_step=current_step,
                message=f"Miner {uid} inactive for "
                f"{current_step - self._miner_health[uid].last_response_step} steps",
            )

        # Detect stale miners
        for uid in self._miner_health:
            if self.detect_stale(uid):
                self._maybe_emit_alert(
                    alerts=alerts,
                    alert_type=AlertType.MINER_STALE,
                    uid=uid,
                    current_step=current_step,
                    message=f"Miner {uid} producing stale anomaly scores "
                    f"(identical for {self.stale_threshold} rounds)",
                )

        # Detect deregistration candidates
        dereg_uids = self.detect_deregistration_candidates(current_step)
        for uid in dereg_uids:
            self._maybe_emit_alert(
                alerts=alerts,
                alert_type=AlertType.DEREGISTRATION_CANDIDATE,
                uid=uid,
                current_step=current_step,
                message=f"Miner {uid} recommended for deregistration "
                f"(inactive for {current_step - self._miner_health[uid].last_response_step} steps)",
            )

        return alerts

    def detect_coordinated_attack(
        self,
        scores: np.ndarray,
        n_total: int,
    ) -> bool:
        """Detect coordinated score drops across multiple miners.

        Compares current scores against previously recorded scores. If more
        than attack_miner_ratio of miners show a relative score drop exceeding
        attack_drop_ratio, returns True.

        The first call (no previous scores) always returns False and initializes
        the baseline.

        Args:
            scores: Current score array (length >= n_total).
            n_total: Number of miners to evaluate.

        Returns:
            True if a coordinated attack is detected, False otherwise.
        """
        if not self._previous_scores:
            # First call -- initialize baseline, no detection possible
            for uid in range(min(n_total, len(scores))):
                self._previous_scores[uid] = float(scores[uid])
            return False

        drop_count = 0
        evaluated = 0
        for uid in range(min(n_total, len(scores))):
            prev = self._previous_scores.get(uid, 0.0)
            current = float(scores[uid])
            if prev > 1e-6:
                relative_drop = (prev - current) / max(prev, 1e-6)
                if relative_drop > self.attack_drop_ratio:
                    drop_count += 1
            evaluated += 1

        # Update previous scores for next cycle
        for uid in range(min(n_total, len(scores))):
            self._previous_scores[uid] = float(scores[uid])

        if evaluated == 0:
            return False

        return (drop_count / max(1, n_total)) > self.attack_miner_ratio

    def detect_registration_surge(
        self,
        current_hotkeys: list[str],
        current_step: int,
    ) -> bool:
        """Detect registration surges (multiple new hotkeys in short window).

        A surge is detected when the number of new hotkeys (not previously
        known) meets or exceeds surge_threshold.

        Args:
            current_hotkeys: List of current metagraph hotkeys.
            current_step: Current validator step (for logging context).

        Returns:
            True if a registration surge is detected, False otherwise.
        """
        current_set = set(current_hotkeys)
        new_hotkeys = current_set - self._known_hotkeys

        # Update known hotkeys
        self._known_hotkeys = current_set

        return len(new_hotkeys) >= self.surge_threshold

    async def send_webhook(self, alerts: list[dict]) -> None:
        """Send alerts to configured webhook URL via HTTP POST.

        Fire-and-forget: logs errors but does not raise. Uses urllib
        to avoid extra dependencies.

        Args:
            alerts: List of alert dicts to send.
        """
        if self.webhook_url is None:
            return

        payload = json.dumps({
            "alerts": alerts,
            "timestamp": time.time(),
        }).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # webhook_url is operator-configured (validator config), not
            # user-controlled input, so file:/ / non-http schemes are not
            # exposed to attackers. Bandit B310 flags any urlopen by default;
            # justified by the operator-controlled trust boundary.
            urllib.request.urlopen(req, timeout=5)  # nosec B310
        except Exception as e:
            bt.logging.warning(f"[Microglia] Webhook send failed: {e}")

    def run_surveillance_cycle(
        self,
        scores: np.ndarray,
        score_history: dict[int, list[float]],
        hotkeys: list[str],
        n_total: int,
        current_step: int,
    ) -> SubnetHealthMetrics:
        """Run a complete surveillance cycle -- the main entry point for the validator daemon.

        Performs coordinated attack detection, registration surge detection,
        per-miner alert generation, optional webhook dispatch, and health
        metric computation.

        Args:
            scores: Current EMA score array.
            score_history: Per-miner rolling score history.
            hotkeys: Current metagraph hotkeys.
            n_total: Total miners in metagraph.
            current_step: Current validator step.

        Returns:
            SubnetHealthMetrics for this cycle.
        """
        all_alerts: list[dict] = []

        # Coordinated attack detection (GLIA-03)
        if self.detect_coordinated_attack(scores, n_total):
            alert = {
                "type": AlertType.COORDINATED_ATTACK.value,
                "uid": -1,
                "step": current_step,
                "message": (
                    f"Coordinated attack detected: >{self.attack_miner_ratio * 100:.0f}% of miners "
                    f"dropped >{self.attack_drop_ratio * 100:.0f}% score in step {current_step}"
                ),
            }
            all_alerts.append(alert)
            bt.logging.warning(f"[Microglia] {alert['type']}: {alert['message']}")

        # Registration surge detection (GLIA-03)
        if self.detect_registration_surge(hotkeys, current_step):
            alert = {
                "type": AlertType.REGISTRATION_SURGE.value,
                "uid": -1,
                "step": current_step,
                "message": (
                    f"Registration surge: >={self.surge_threshold} new hotkeys "
                    f"detected at step {current_step}"
                ),
            }
            all_alerts.append(alert)
            bt.logging.warning(f"[Microglia] {alert['type']}: {alert['message']}")

        # Per-miner alerts (inactive, stale, deregistration)
        miner_alerts = self.generate_alerts(current_step)
        all_alerts.extend(miner_alerts)

        # Webhook dispatch (fire-and-forget via sync wrapper)
        if all_alerts and self.webhook_url:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.send_webhook(all_alerts))
                else:
                    loop.run_until_complete(self.send_webhook(all_alerts))
            except RuntimeError:
                # No event loop available -- skip webhook
                bt.logging.debug("[Microglia] No event loop for webhook dispatch")

        # Compute and return health metrics
        return self.get_health_metrics(n_total, score_history, current_step)

    def _maybe_emit_alert(
        self,
        alerts: list[dict],
        alert_type: AlertType,
        uid: int,
        current_step: int,
        message: str,
    ) -> None:
        """Emit an alert if not deduplicated by cooldown.

        Args:
            alerts: List to append alert dict to.
            alert_type: Type of alert.
            uid: Miner UID.
            current_step: Current step.
            message: Human-readable alert message.
        """
        key = (alert_type, uid)
        last_fired = self._alert_history.get(key)

        if last_fired is not None and (current_step - last_fired) <= self.alert_cooldown:
            # Within cooldown -- suppress
            return

        # Emit alert
        alert = {
            "type": alert_type.value,
            "uid": uid,
            "step": current_step,
            "message": message,
        }
        alerts.append(alert)
        self._alert_history[key] = current_step

        bt.logging.warning(f"[Microglia] {alert_type.value}: {message}")

    def get_health_metrics(
        self,
        n_total: int,
        score_history: dict[int, list[float]],
        current_step: int,
    ) -> SubnetHealthMetrics:
        """Compute subnet-wide health metrics from miner states and score history.

        Args:
            n_total: Total number of miners in the metagraph.
            score_history: Per-miner rolling score history (uid -> list of scores).
            current_step: Current validator step number.

        Returns:
            SubnetHealthMetrics with computed values.
        """
        # Run detection
        inactive_list = self.detect_inactive(current_step)
        dereg_list = self.detect_deregistration_candidates(current_step)

        # Detect stale miners
        stale_list = [
            uid for uid in self._miner_health if self.detect_stale(uid)
        ]

        # Inflammation score: ratio of unhealthy miners
        n_unhealthy = len(set(inactive_list) | set(stale_list))
        inflammation_score = min(1.0, n_unhealthy / max(1, n_total))

        # Threat level from inflammation thresholds
        if inflammation_score < 0.1:
            threat_level = "low"
        elif inflammation_score < 0.3:
            threat_level = "medium"
        elif inflammation_score < 0.6:
            threat_level = "high"
        else:
            threat_level = "critical"

        # Population diversity index from score_history
        population_diversity_index = self._compute_diversity_index(score_history)

        # Active miners = total tracked - inactive
        active_miners = max(0, n_total - len(inactive_list))

        return SubnetHealthMetrics(
            inflammation_score=inflammation_score,
            threat_level=threat_level,
            population_diversity_index=population_diversity_index,
            active_miners=active_miners,
            inactive_miners=len(inactive_list),
            stale_miners=len(stale_list),
            deregistration_candidates=len(dereg_list),
        )

    @staticmethod
    def _compute_diversity_index(
        score_history: dict[int, list[float]],
        min_entries: int = 5,
    ) -> float:
        """Compute population diversity index from per-miner score histories.

        Uses the mean standard deviation of per-miner score arrays, normalized
        to the 0-1 range (std / 0.5, clamped). Higher std = more diverse behavior.

        Returns 0.5 (neutral) when insufficient data.

        Args:
            score_history: Per-miner score history (uid -> list of scores).
            min_entries: Minimum entries per miner to include in computation.

        Returns:
            Float in [0, 1] representing population diversity.
        """
        # Filter miners with sufficient history
        eligible_stds: list[float] = []
        for _uid, scores in score_history.items():
            if len(scores) >= min_entries:
                eligible_stds.append(float(np.std(scores)))

        # Need at least 2 miners with sufficient data
        if len(eligible_stds) < 2:
            return 0.5

        # Mean std dev normalized to 0-1 range
        mean_std = float(np.mean(eligible_stds))
        # Normalize: std of 0.5 -> diversity 1.0 (maximum possible std for [0,1] scores)
        diversity = min(1.0, mean_std / 0.5)
        return diversity
