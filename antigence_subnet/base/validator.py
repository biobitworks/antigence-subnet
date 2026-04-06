"""
Base validator neuron module.

Provides BaseValidatorNeuron with dendrite setup, EMA score tracking,
evaluation dataset management, set_weights integration, and state
persistence. Concrete validators subclass this.
"""

import asyncio
import copy
import os
import time
from abc import abstractmethod
from pathlib import Path

import bittensor as bt
import numpy as np

from antigence_subnet.base.neuron import _DEFAULT_EVAL_DATA_DIR, BaseNeuron
from antigence_subnet.mock import MockDendrite
from antigence_subnet.validator.evaluation import EvaluationDataset
from antigence_subnet.validator.metagraph_monitor import MetagraphMonitor
from antigence_subnet.validator.microglia import AlertType
from antigence_subnet.validator.weight_audit import (
    audit_weights,
    check_commit_reveal_enabled,
)

VALIDATOR_POLICY_GLOBAL_THRESHOLD = "global_threshold"
VALIDATOR_POLICY_DOMAIN_THRESHOLDS = "domain_thresholds"
VALIDATOR_POLICY_OPERATOR_MULTIBAND = "operator_multiband"
VALIDATOR_POLICY_MODES = {
    VALIDATOR_POLICY_GLOBAL_THRESHOLD,
    VALIDATOR_POLICY_DOMAIN_THRESHOLDS,
    VALIDATOR_POLICY_OPERATOR_MULTIBAND,
}

DEFAULT_POLICY_MODE = VALIDATOR_POLICY_OPERATOR_MULTIBAND
DEFAULT_POLICY_HIGH_THRESHOLD = 0.5
DEFAULT_POLICY_LOW_THRESHOLD = 0.493536
DEFAULT_POLICY_MIN_CONFIDENCE = 0.6


def resolve_validator_policy_config(config) -> bt.Config:
    """Canonicalize validator policy config with a narrow legacy bridge."""
    if not hasattr(config, "policy") or config.policy is None:
        config.policy = bt.Config()

    validator_section = getattr(config, "validator", None)
    validator_policy = (
        getattr(validator_section, "policy", None)
        if validator_section is not None
        else None
    )

    mode = getattr(config.policy, "mode", None)
    if mode is None and validator_policy is not None:
        mode = getattr(validator_policy, "mode", None)

    high_threshold = getattr(config.policy, "high_threshold", None)
    if high_threshold is None and validator_policy is not None:
        high_threshold = getattr(validator_policy, "high_threshold", None)

    low_threshold = getattr(config.policy, "low_threshold", None)
    if low_threshold is None and validator_policy is not None:
        low_threshold = getattr(validator_policy, "low_threshold", None)

    min_confidence = getattr(config.policy, "min_confidence", None)
    if min_confidence is None and validator_policy is not None:
        min_confidence = getattr(validator_policy, "min_confidence", None)

    explicit_policy = any(
        value is not None
        for value in (mode, high_threshold, low_threshold, min_confidence)
    )

    legacy_threshold = getattr(getattr(config, "reward", None), "decision_threshold", None)
    if not explicit_policy and legacy_threshold is not None:
        mode = VALIDATOR_POLICY_GLOBAL_THRESHOLD
        high_threshold = float(legacy_threshold)
        low_threshold = float(legacy_threshold)
        min_confidence = 0.0
    else:
        mode = (mode or DEFAULT_POLICY_MODE).lower()
        if mode not in VALIDATOR_POLICY_MODES:
            bt.logging.warning(
                f"Unknown validator policy mode '{mode}'; "
                f"falling back to {DEFAULT_POLICY_MODE}"
            )
            mode = DEFAULT_POLICY_MODE

        if high_threshold is None:
            high_threshold = DEFAULT_POLICY_HIGH_THRESHOLD
        if low_threshold is None:
            low_threshold = (
                DEFAULT_POLICY_LOW_THRESHOLD
                if mode == VALIDATOR_POLICY_OPERATOR_MULTIBAND
                else high_threshold
            )
        if min_confidence is None:
            min_confidence = (
                DEFAULT_POLICY_MIN_CONFIDENCE
                if mode == VALIDATOR_POLICY_OPERATOR_MULTIBAND
                else 0.0
            )

    config.policy.mode = mode
    config.policy.high_threshold = float(high_threshold)
    config.policy.low_threshold = float(low_threshold)
    config.policy.min_confidence = float(min_confidence)
    return config.policy


class BaseValidatorNeuron(BaseNeuron):
    """Base class for validator neurons.

    Sets up dendrite for querying miners, EMA score tracking,
    and state persistence to .npz files.
    """

    neuron_type: str = "ValidatorNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        # Set up dendrite (mock or real)
        if getattr(self.config, "mock", False):
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.Dendrite(wallet=self.wallet)

        # Initialize EMA score tracking
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.hotkeys = list(self.metagraph.hotkeys)
        self.step = 0

        # Set defaults for neuron config
        if not hasattr(self.config, "neuron"):
            self.config.neuron = bt.Config()
        self.config.neuron.sample_size = getattr(
            self.config.neuron, "sample_size", 16
        )
        self.config.neuron.timeout = getattr(
            self.config.neuron, "timeout", 12.0
        )
        self.config.neuron.moving_average_alpha = getattr(
            self.config.neuron, "moving_average_alpha", 0.1
        )

        # Evaluation dataset config defaults (use 'or' to handle None from argparse)
        self.config.neuron.eval_data_dir = (
            getattr(self.config.neuron, "eval_data_dir", None)
            or _DEFAULT_EVAL_DATA_DIR
        )
        self.config.neuron.eval_domain = (
            getattr(self.config.neuron, "eval_domain", None)
            or "hallucination"
        )
        self.config.neuron.samples_per_round = (
            getattr(self.config.neuron, "samples_per_round", None)
            or 10
        )
        self.config.neuron.n_honeypots = (
            getattr(self.config.neuron, "n_honeypots", None)
            or 2
        )
        self.config.neuron.set_weights_interval = (
            getattr(self.config.neuron, "set_weights_interval", None)
            or 100
        )
        self.config.neuron.set_weights_retries = (
            getattr(self.config.neuron, "set_weights_retries", None)
            or 3
        )

        # Initialize evaluation dataset
        eval_path = Path(self.config.neuron.eval_data_dir)
        if eval_path.exists():
            self.evaluation = EvaluationDataset(
                data_dir=eval_path,
                domain=self.config.neuron.eval_domain,
            )
        else:
            bt.logging.warning(
                f"Evaluation data directory not found: {eval_path}. "
                f"Using empty evaluation dataset."
            )
            self.evaluation = None

        # Score history for diversity tracking (rolling window per miner)
        self.score_history: dict[int, list[float]] = {}

        # Confidence history for ECE/calibration tracking (sliding window per miner)
        # Each entry: (confidences_list, accuracies_list) per round
        self.confidence_history: dict[int, list[tuple[list[float], list[int]]]] = {}

        # Microglia surveillance (GLIA-01)
        if not hasattr(self.config, "microglia"):
            self.config.microglia = bt.Config()
        microglia_interval = (
            getattr(self.config.microglia, "interval", None) or 100
        )
        microglia_webhook = getattr(
            self.config.microglia, "webhook_url", None
        )
        microglia_inactive = (
            getattr(self.config.microglia, "inactive_threshold", None) or 10
        )
        microglia_stale = (
            getattr(self.config.microglia, "stale_threshold", None) or 5
        )
        microglia_dereg = (
            getattr(self.config.microglia, "deregistration_threshold", None)
            or 50
        )
        microglia_cooldown = (
            getattr(self.config.microglia, "alert_cooldown", None) or 10
        )
        self.microglia_interval = microglia_interval
        self.microglia_enabled = getattr(
            self.config.microglia, "enabled", True
        )

        from antigence_subnet.validator.microglia import MicrogliaMonitor

        self.microglia = MicrogliaMonitor(
            inactive_threshold=microglia_inactive,
            stale_threshold=microglia_stale,
            deregistration_threshold=microglia_dereg,
            alert_cooldown=microglia_cooldown,
            webhook_url=microglia_webhook,
        )

        # Reward weight configuration (Phase 26 - MAIN-06)
        if not hasattr(self.config, "reward"):
            self.config.reward = bt.Config()
        self.config.reward.base_weight = float(
            getattr(self.config.reward, "base_weight", None) or 0.70
        )
        self.config.reward.calibration_weight = float(
            getattr(self.config.reward, "calibration_weight", None) or 0.10
        )
        self.config.reward.robustness_weight = float(
            getattr(self.config.reward, "robustness_weight", None) or 0.10
        )
        self.config.reward.diversity_weight = float(
            getattr(self.config.reward, "diversity_weight", None) or 0.10
        )
        self.config.reward.decision_threshold = float(
            getattr(self.config.reward, "decision_threshold", None) or 0.5
        )
        resolve_validator_policy_config(self.config)

        # Validator scoring mode configuration (Phase 82 - NONDET-02)
        if not hasattr(self.config, "scoring") or self.config.scoring is None:
            self.config.scoring = bt.Config()
        legacy_validator_section = getattr(self.config, "validator", None)
        legacy_scoring = (
            getattr(legacy_validator_section, "scoring", None)
            if legacy_validator_section is not None
            else None
        )
        scoring_mode = getattr(self.config.scoring, "mode", None)
        if scoring_mode is None and legacy_scoring is not None:
            scoring_mode = getattr(legacy_scoring, "mode", None)
        scoring_repeats = getattr(self.config.scoring, "repeats", None)
        if scoring_repeats is None and legacy_scoring is not None:
            scoring_repeats = getattr(legacy_scoring, "repeats", None)
        scoring_ci_level = getattr(self.config.scoring, "ci_level", None)
        if scoring_ci_level is None and legacy_scoring is not None:
            scoring_ci_level = getattr(legacy_scoring, "ci_level", None)

        self.config.scoring.mode = (scoring_mode or "exact").lower()
        self.config.scoring.repeats = int(scoring_repeats or 3)
        self.config.scoring.ci_level = float(scoring_ci_level or 0.95)

        # Validate reward weights sum to 1.0 (tolerance 0.01)
        weight_sum = (
            self.config.reward.base_weight
            + self.config.reward.calibration_weight
            + self.config.reward.robustness_weight
            + self.config.reward.diversity_weight
        )
        if abs(weight_sum - 1.0) > 0.01:
            bt.logging.warning(
                f"Reward weights sum to {weight_sum:.4f}, not 1.0. "
                "Falling back to defaults (0.70/0.10/0.10/0.10)."
            )
            self.config.reward.base_weight = 0.70
            self.config.reward.calibration_weight = 0.10
            self.config.reward.robustness_weight = 0.10
            self.config.reward.diversity_weight = 0.10

        # Metagraph anomaly detection (NET-05)
        self.metagraph_monitor = MetagraphMonitor()
        bt.logging.info("MetagraphMonitor initialized for metagraph anomaly detection")

        # Check commit-reveal status (informational)
        try:
            cr_enabled = check_commit_reveal_enabled(
                self.subtensor, self.config.netuid
            )
            bt.logging.info(f"Commit-reveal weights enabled: {cr_enabled}")
        except Exception:
            bt.logging.debug(
                "Could not check commit-reveal status (expected in mock mode)"
            )

        bt.logging.info(
            f"Validator initialized | Scores shape: {self.scores.shape} | "
            f"Sample size: {self.config.neuron.sample_size}"
        )

    @classmethod
    def _add_args_to_parser(cls, parser) -> None:
        """Add validator-specific args on top of base neuron args."""
        super()._add_args_to_parser(parser)
        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            default=16,
            help="Number of miners to query per forward pass",
        )
        parser.add_argument(
            "--neuron.timeout",
            type=float,
            default=12.0,
            help="Dendrite query timeout in seconds",
        )
        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            default=0.1,
            help="EMA alpha for score updates",
        )
        parser.add_argument(
            "--neuron.eval_data_dir",
            type=str,
            default=None,
            help="Path to evaluation dataset directory (default: <project_root>/data/evaluation)",
        )
        parser.add_argument(
            "--neuron.eval_domain",
            type=str,
            default="hallucination",
            help="Evaluation domain to use",
        )
        parser.add_argument(
            "--neuron.samples_per_round",
            type=int,
            default=10,
            help="Number of samples per evaluation round",
        )
        parser.add_argument(
            "--neuron.n_honeypots",
            type=int,
            default=2,
            help="Number of honeypot samples per round",
        )
        parser.add_argument(
            "--neuron.set_weights_interval",
            type=int,
            default=100,
            help="Set weights on chain every N steps",
        )
        parser.add_argument(
            "--neuron.set_weights_retries",
            type=int,
            default=3,
            help="Max retry attempts for set_weights on chain failure",
        )
        # Microglia surveillance CLI args (GLIA-01)
        parser.add_argument(
            "--microglia.interval",
            type=int,
            default=100,
            help="Microglia surveillance interval in steps",
        )
        parser.add_argument(
            "--microglia.webhook_url",
            type=str,
            default=None,
            help="Webhook URL for microglia alerts",
        )
        parser.add_argument(
            "--microglia.inactive_threshold",
            type=int,
            default=10,
            help="Steps without response before inactive flag",
        )
        parser.add_argument(
            "--microglia.stale_threshold",
            type=int,
            default=5,
            help="Consecutive identical scores before stale flag",
        )
        parser.add_argument(
            "--microglia.deregistration_threshold",
            type=int,
            default=50,
            help="Steps inactive before deregistration candidate",
        )
        parser.add_argument(
            "--microglia.alert_cooldown",
            type=int,
            default=10,
            help="Min steps between repeated alerts for same (type, uid) (default: 10)",
        )
        parser.add_argument(
            "--microglia.enabled",
            action="store_true",
            default=True,
            help="Enable microglia surveillance",
        )

        # Reward weight configuration (Phase 26 - MAIN-06)
        parser.add_argument(
            "--reward.base_weight",
            type=float,
            default=0.70,
            help="Weight for base precision-first reward component (default: 0.70)",
        )
        parser.add_argument(
            "--reward.calibration_weight",
            type=float,
            default=0.10,
            help="Weight for calibration bonus component (default: 0.10)",
        )
        parser.add_argument(
            "--reward.robustness_weight",
            type=float,
            default=0.10,
            help="Weight for robustness/stability bonus component (default: 0.10)",
        )
        parser.add_argument(
            "--reward.diversity_weight",
            type=float,
            default=0.10,
            help="Weight for diversity bonus component (default: 0.10)",
        )
        parser.add_argument(
            "--reward.decision_threshold",
            type=float,
            default=0.5,
            help="Anomaly score threshold for binary classification (default: 0.5)",
        )
        parser.add_argument(
            "--policy.mode",
            "--validator.policy.mode",
            type=str,
            default=DEFAULT_POLICY_MODE,
            help=(
                "Validator decision policy: global_threshold, domain_thresholds, "
                f"or operator_multiband (default: {DEFAULT_POLICY_MODE})"
            ),
        )
        parser.add_argument(
            "--policy.high_threshold",
            "--validator.policy.high_threshold",
            type=float,
            default=DEFAULT_POLICY_HIGH_THRESHOLD,
            help=(
                "High threshold for validator policy decisions "
                f"(default: {DEFAULT_POLICY_HIGH_THRESHOLD})"
            ),
        )
        parser.add_argument(
            "--policy.low_threshold",
            "--validator.policy.low_threshold",
            type=float,
            default=DEFAULT_POLICY_LOW_THRESHOLD,
            help=(
                "Low threshold for operator_multiband review/allow cutoff "
                f"(default: {DEFAULT_POLICY_LOW_THRESHOLD})"
            ),
        )
        parser.add_argument(
            "--policy.min_confidence",
            "--validator.policy.min_confidence",
            type=float,
            default=DEFAULT_POLICY_MIN_CONFIDENCE,
            help=(
                "Minimum confidence gate for policy decisions "
                f"(default: {DEFAULT_POLICY_MIN_CONFIDENCE})"
            ),
        )
        parser.add_argument(
            "--scoring.mode",
            "--validator.scoring.mode",
            type=str,
            default="exact",
            help="Validator scoring mode: exact, statistical, or semantic (default: exact)",
        )
        parser.add_argument(
            "--scoring.repeats",
            "--validator.scoring.repeats",
            type=int,
            default=3,
            help=(
                "Repeat count for statistical scoring; multiplies "
                "validator scorer work by repeats (default: 3)"
            ),
        )
        parser.add_argument(
            "--scoring.ci_level",
            "--validator.scoring.ci_level",
            type=float,
            default=0.95,
            help="Confidence level for statistical scoring intervals (default: 0.95)",
        )

        # Challenge rotation CLI args (VHARD-01)
        parser.add_argument(
            "--rotation.enabled",
            action="store_true",
            default=True,
            help="Enable round-based challenge rotation (default: True)",
        )
        parser.add_argument(
            "--rotation.window",
            type=int,
            default=10,
            help="Number of recent rounds to exclude samples from per miner (default: 10)",
        )

    @abstractmethod
    async def forward(self) -> None:
        """Run a single forward pass.

        Must be implemented by subclasses. The validator creates
        evaluation synapses internally and sends them to miners via
        dendrite. Does NOT take a synapse argument (Pitfall 3).
        """
        ...

    def sync(self) -> None:
        """Re-sync metagraph with hotkey rotation detection (CORR-02) and
        metagraph anomaly detection (NET-05).

        Overrides BaseNeuron.sync() to detect hotkey rotations (zeroing stale
        scores), resize scores on metagraph size changes, and run
        MetagraphMonitor checks after the metagraph re-sync completes.
        """
        # Save previous hotkeys before metagraph resync
        previous_hotkeys = list(self.hotkeys)

        super().sync()

        # Detect hotkey rotations (CORR-02)
        new_hotkeys = list(self.metagraph.hotkeys)
        overlap = min(len(previous_hotkeys), len(new_hotkeys))
        for uid in range(overlap):
            if previous_hotkeys[uid] != new_hotkeys[uid]:
                self.scores[uid] = 0.0
                self.score_history.pop(uid, None)
                self.confidence_history.pop(uid, None)
                bt.logging.info(
                    f"Hotkey rotation detected at UID {uid}: "
                    f"{previous_hotkeys[uid][:8]}... -> {new_hotkeys[uid][:8]}..."
                )

        # Resize scores if metagraph size changed
        if len(self.scores) != self.metagraph.n:
            new_scores = np.zeros(self.metagraph.n, dtype=np.float32)
            copy_len = min(len(self.scores), self.metagraph.n)
            new_scores[:copy_len] = self.scores[:copy_len]
            self.scores = new_scores
            bt.logging.info(
                f"Scores array resized: {len(previous_hotkeys)} -> {self.metagraph.n}"
            )

        # Update local hotkey cache
        self.hotkeys = list(new_hotkeys)

        # Metagraph anomaly detection after sync
        try:
            anomalies = self.metagraph_monitor.check_anomalies(
                hotkeys=list(self.metagraph.hotkeys),
                stakes=self.metagraph.S,
                n=self.metagraph.n,
                step=self.step,
            )
            for anomaly in anomalies:
                bt.logging.warning(
                    f"[MetagraphMonitor] {anomaly.anomaly_type}: {anomaly.details}"
                )
                # Dispatch through microglia webhook if configured
                if self.microglia_enabled and self.microglia.webhook_url:
                    alert = {
                        "type": AlertType.METAGRAPH_ANOMALY.value,
                        "uid": -1,
                        "step": self.step,
                        "message": f"{anomaly.anomaly_type}: {anomaly.details}",
                    }
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self.microglia.send_webhook([alert])
                            )
                        else:
                            loop.run_until_complete(
                                self.microglia.send_webhook([alert])
                            )
                    except RuntimeError:
                        bt.logging.debug(
                            "[MetagraphMonitor] No event loop for webhook dispatch"
                        )
        except Exception as e:
            bt.logging.error(f"MetagraphMonitor error: {e}")

    def update_scores(
        self, rewards: np.ndarray, uids: list[int]
    ) -> None:
        """Update EMA scores for given UIDs.

        Uses exponential moving average:
            scores[uid] = alpha * reward + (1 - alpha) * scores[uid]

        NaN rewards are sanitized to 0.0.

        Args:
            rewards: Array of reward values for queried miners.
            uids: List of UIDs corresponding to rewards.
        """
        alpha = self.config.neuron.moving_average_alpha

        # Sanitize NaN rewards to 0.0
        rewards = np.nan_to_num(rewards, nan=0.0)

        # Update scores using EMA
        for i, uid in enumerate(uids):
            if 0 <= uid < len(self.scores):
                self.scores[uid] = (
                    alpha * rewards[i] + (1 - alpha) * self.scores[uid]
                )

        bt.logging.debug(
            f"Updated scores for {len(uids)} UIDs | "
            f"Mean score: {np.mean(self.scores):.4f}"
        )

    def set_weights(self) -> None:
        """Set weights on chain from current EMA scores.

        Uses weight_utils to normalize and convert to uint16.
        Calls subtensor.set_weights() with ExtrinsicResponse API (v10).
        Retries up to config.neuron.set_weights_retries times on failure
        with 2-second backoff between attempts.
        """
        from antigence_subnet.base.utils.weight_utils import (
            convert_weights_and_uids_for_emit,
            process_weights_for_netuid,
        )

        uids = np.arange(len(self.scores))
        processed_uids, processed_weights = process_weights_for_netuid(
            uids=uids,
            weights=self.scores,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids,
            weights=processed_weights,
        )

        # Pre-commit weight audit (NET-06)
        warnings = audit_weights(processed_weights)
        for w in warnings:
            bt.logging.warning(f"Weight audit: {w}")

        max_retries = getattr(
            self.config.neuron, "set_weights_retries", 3
        ) or 3

        for attempt in range(1, max_retries + 1):
            try:
                result = self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                    uids=uint_uids,
                    weights=uint_weights,
                    version_key=getattr(self, "spec_version", 0),
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                )
                # v10 ExtrinsicResponse API -- use .success not tuple unpacking
                if result.success:
                    bt.logging.info(
                        f"set_weights on chain successfully! "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    return
                else:
                    bt.logging.warning(
                        f"set_weights attempt {attempt}/{max_retries} failed: "
                        f"{result.message}"
                    )
            except Exception as e:
                bt.logging.warning(
                    f"set_weights attempt {attempt}/{max_retries} error: "
                    f"{type(e).__name__}: {e}"
                )

            # Backoff before next retry (skip after last attempt)
            if attempt < max_retries:
                time.sleep(2)

        bt.logging.error(
            f"set_weights failed after {max_retries} retries"
        )

    def save_state(self) -> None:
        """Save validator state to .npz file (step, scores, hotkeys, eval metadata).

        Uses atomic write pattern: write to temp file, then os.replace to
        final path. This ensures the previous valid state is preserved if
        the process is killed mid-write (RESIL-04).
        """
        import json as _json
        import tempfile

        bt.logging.info("Saving validator state.")
        state_path = os.path.expanduser(self.config.neuron.full_path)
        os.makedirs(state_path, exist_ok=True)

        final_path = os.path.join(state_path, "state.npz")

        # Serialize score_history as JSON (dict of lists is not numpy-native)
        score_history_json = _json.dumps(
            {str(k): v for k, v in getattr(self, "score_history", {}).items()}
        )

        # Serialize confidence_history as JSON
        # Format: {str(uid): [[[confidences], [accuracies]], ...]}
        confidence_history_json = _json.dumps(
            {
                str(k): [list(pair) for pair in v]
                for k, v in getattr(self, "confidence_history", {}).items()
            }
        )

        # Serialize challenge rotation state as JSON (VHARD-01)
        rotation_obj = getattr(self, "challenge_rotation", None)
        rotation_json = _json.dumps(
            rotation_obj.to_dict() if rotation_obj is not None else None
        )

        # Atomic write: write to temp file in same directory, then rename.
        # os.replace is atomic on POSIX when src and dst are on same filesystem.
        # Use .npz suffix so np.savez doesn't append an extra .npz extension.
        fd, tmp_path = tempfile.mkstemp(dir=state_path, suffix=".tmp.npz")
        os.close(fd)
        try:
            np.savez(
                tmp_path,
                step=self.step,
                scores=self.scores,
                hotkeys=self.hotkeys,
                eval_round=self.step,
                dataset_version=getattr(
                    self.evaluation, "dataset_version", ""
                ) if self.evaluation else "",
                score_history=score_history_json,
                confidence_history=confidence_history_json,
                rotation_state=rotation_json,
            )
            os.replace(tmp_path, final_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load_state(self) -> None:
        """Load validator state from .npz file with corruption protection.

        Validates required keys (step, scores, hotkeys) exist before
        accessing them. On any failure (missing file, corrupted data,
        missing keys): logs a warning with error type and reinitializes
        all state from defaults including score_history and
        confidence_history (RESIL-04).
        """
        import json as _json

        bt.logging.info("Loading validator state.")
        state_path = os.path.expanduser(self.config.neuron.full_path)
        state_file = os.path.join(state_path, "state.npz")
        try:
            state = np.load(state_file, allow_pickle=True)
            # Validate required keys exist before accessing
            for key in ("step", "scores", "hotkeys"):
                if key not in state.files:
                    raise KeyError(f"Missing required key: {key}")
            self.step = int(state["step"])
            self.scores = state["scores"].astype(np.float32)
            self.hotkeys = list(state["hotkeys"])
            # Restore evaluation metadata if present
            if "eval_round" in state:
                eval_round = int(state["eval_round"])
                bt.logging.debug(f"Restored eval_round: {eval_round}")
            if "dataset_version" in state:
                loaded_version = str(state["dataset_version"])
                bt.logging.debug(
                    f"Restored dataset_version: {loaded_version}"
                )
            # Restore score_history if present
            if "score_history" in state:
                raw = str(state["score_history"])
                self.score_history = {
                    int(k): v for k, v in _json.loads(raw).items()
                }
            # Restore confidence_history if present (backward compat: default to empty)
            if "confidence_history" in state:
                raw_conf = str(state["confidence_history"])
                parsed = _json.loads(raw_conf)
                self.confidence_history = {
                    int(k): [tuple(pair) for pair in v]
                    for k, v in parsed.items()
                }
            else:
                self.confidence_history = {}
            # Restore challenge rotation state (VHARD-01, backward compat)
            if "rotation_state" in state:
                from antigence_subnet.validator.rotation import ChallengeRotation

                raw_rot = str(state["rotation_state"])
                rot_data = _json.loads(raw_rot)
                if rot_data is not None:
                    self.challenge_rotation = ChallengeRotation.from_dict(rot_data)
                    bt.logging.debug(
                        f"Restored challenge rotation state: "
                        f"window={self.challenge_rotation.rotation_window}"
                    )
            bt.logging.info(
                f"State loaded | Step: {self.step} | "
                f"Scores: {len(self.scores)} entries"
            )
        except Exception as e:
            bt.logging.warning(
                f"Could not load state ({type(e).__name__}: {e}), "
                "reinitializing from defaults."
            )
            self.step = 0
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            self.score_history = {}
            self.confidence_history = {}

    def run(self) -> None:
        """Start the validator: load state, forward loop, save state."""
        bt.logging.info(f"Starting {self.neuron_type}...")
        self.load_state()

        try:
            while not self.should_exit:
                try:
                    bt.logging.info(f"Forward step {self.step}")
                    asyncio.get_event_loop().run_until_complete(self.forward())
                except Exception as e:
                    bt.logging.error(f"Forward pass error at step {self.step}: {e}")
                    # Continue to next cycle -- don't crash

                self.step += 1
                self.sync()
                self.save_state()

                # Microglia surveillance cycle (GLIA-01)
                if (
                    self.microglia_enabled
                    and self.step > 0
                    and self.step % self.microglia_interval == 0
                ):
                    try:
                        health = self.microglia.run_surveillance_cycle(
                            scores=self.scores,
                            score_history=self.score_history,
                            hotkeys=list(self.metagraph.hotkeys),
                            n_total=self.metagraph.n,
                            current_step=self.step,
                        )
                        bt.logging.info(
                            f"Microglia health: inflammation={health.inflammation_score:.2f} "
                            f"threat={health.threat_level} "
                            f"diversity={health.population_diversity_index:.2f} "
                            f"active={health.active_miners} "
                            f"inactive={health.inactive_miners}"
                        )
                    except Exception as e:
                        bt.logging.error(f"Microglia surveillance error: {e}")

                # Set weights on chain periodically
                if (
                    self.step % self.config.neuron.set_weights_interval == 0
                    and self.step > 0
                ):
                    try:
                        self.set_weights()
                    except Exception as e:
                        bt.logging.error(f"set_weights error: {e}")

                if not self.should_exit:
                    time.sleep(12)  # Sync every block (~12s)
        finally:
            self.save_state()
            bt.logging.info("Validator graceful shutdown complete")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit -- save state."""
        self.save_state()
        bt.logging.info(f"Shutting down {self.neuron_type}.")
        return False
