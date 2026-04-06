"""
Validator cold-start protocol for bootstrap readiness and warmup.

Tracks validator readiness (metagraph synced, evaluation dataset loaded,
dendrite connected) and manages a warmup period with reduced sample sizes
so early rounds don't over-penalize miners with sparse data.

Requirements: VHARD-03 (Validator cold-start and state persistence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ColdStartConfig:
    """Configuration for validator cold-start behavior.

    Parsed from TOML [validator.cold_start] section.

    Attributes:
        max_startup_seconds: Maximum time to achieve readiness before
            proceeding with a warning.
        min_miners_required: Minimum miner count before starting evaluation.
        warmup_rounds: Number of initial rounds with reduced sample_size.
    """

    max_startup_seconds: float = 60.0
    min_miners_required: int = 1
    warmup_rounds: int = 5


def parse_cold_start_config(toml_dict: dict) -> ColdStartConfig:
    """Parse ColdStartConfig from a TOML config dict.

    Expects keys under ``validator.cold_start``:
      - max_startup_seconds (float)
      - min_miners_required (int)
      - warmup_rounds (int)

    Missing keys fall back to dataclass defaults.

    Args:
        toml_dict: Raw TOML dict (top-level keys).

    Returns:
        ColdStartConfig populated from TOML or defaults.
    """
    section = (
        toml_dict.get("validator", {}).get("cold_start", {})
    )
    kwargs: dict = {}
    if "max_startup_seconds" in section:
        kwargs["max_startup_seconds"] = float(section["max_startup_seconds"])
    if "min_miners_required" in section:
        kwargs["min_miners_required"] = int(section["min_miners_required"])
    if "warmup_rounds" in section:
        kwargs["warmup_rounds"] = int(section["warmup_rounds"])
    return ColdStartConfig(**kwargs)


class ColdStartManager:
    """Manages validator cold-start readiness and warmup lifecycle.

    Tracks prerequisite checks (metagraph synced, evaluation dataset loaded,
    dendrite connected) and a warmup period where sample sizes are reduced.

    Attributes:
        config: ColdStartConfig with startup parameters.
        bootstrap_step: Step at which cold-start completed (None if ongoing).
        warmup_complete: Whether warmup rounds are finished.
    """

    def __init__(self, config: ColdStartConfig | None = None) -> None:
        self.config = config or ColdStartConfig()
        self._init_time = time.monotonic()
        self.bootstrap_step: int | None = None
        self.warmup_complete: bool = False
        self._warmup_rounds_done: int = 0
        self._original_sample_size: int | None = None

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def readiness_checks(
        self,
        metagraph_synced: bool = False,
        eval_dataset_loaded: bool = False,
        dendrite_connected: bool = False,
    ) -> dict[str, bool]:
        """Return a dict of prerequisite checks and their status.

        Args:
            metagraph_synced: Whether the metagraph has been synced.
            eval_dataset_loaded: Whether the evaluation dataset is loaded.
            dendrite_connected: Whether the dendrite is connected.

        Returns:
            Dict mapping check name to boolean status.
        """
        return {
            "metagraph_synced": metagraph_synced,
            "eval_dataset_loaded": eval_dataset_loaded,
            "dendrite_connected": dendrite_connected,
        }

    def is_ready(
        self,
        metagraph_synced: bool = False,
        eval_dataset_loaded: bool = False,
        dendrite_connected: bool = False,
        n_miners: int = 0,
    ) -> bool:
        """Check whether the validator is ready to begin evaluation.

        Ready when all prerequisite checks pass AND the minimum number
        of miners is available in the metagraph.

        Args:
            metagraph_synced: Whether the metagraph has been synced.
            eval_dataset_loaded: Whether the evaluation dataset is loaded.
            dendrite_connected: Whether the dendrite is connected.
            n_miners: Number of miners currently visible.

        Returns:
            True if all checks pass and enough miners are available.
        """
        checks = self.readiness_checks(
            metagraph_synced=metagraph_synced,
            eval_dataset_loaded=eval_dataset_loaded,
            dendrite_connected=dendrite_connected,
        )
        all_checks = all(checks.values())
        enough_miners = n_miners >= self.config.min_miners_required
        return all_checks and enough_miners

    def startup_time_seconds(self) -> float:
        """Return elapsed seconds since ColdStartManager was created."""
        return time.monotonic() - self._init_time

    def has_exceeded_startup_timeout(self) -> bool:
        """Return True if startup has exceeded max_startup_seconds."""
        return self.startup_time_seconds() > self.config.max_startup_seconds

    # ------------------------------------------------------------------
    # Warmup management
    # ------------------------------------------------------------------

    def begin_warmup(self, original_sample_size: int) -> int:
        """Begin warmup period and return the reduced sample size.

        Args:
            original_sample_size: The configured sample_size to reduce
                during warmup.

        Returns:
            Reduced sample size for warmup rounds.
        """
        self._original_sample_size = original_sample_size
        self._warmup_rounds_done = 0
        self.warmup_complete = False
        return min(4, original_sample_size)

    def record_warmup_round(self) -> bool:
        """Record completion of a warmup round.

        Returns:
            True if warmup is now complete (all rounds done).
        """
        self._warmup_rounds_done += 1
        if self._warmup_rounds_done >= self.config.warmup_rounds:
            self.warmup_complete = True
        return self.warmup_complete

    def get_current_sample_size(self, configured_sample_size: int) -> int:
        """Return the effective sample size for the current round.

        During warmup: returns min(4, configured). After: returns configured.

        Args:
            configured_sample_size: The operator-configured sample_size.

        Returns:
            Effective sample size for this round.
        """
        if self.warmup_complete:
            return configured_sample_size
        return min(4, configured_sample_size)

    @property
    def warmup_rounds_remaining(self) -> int:
        """Number of warmup rounds still to complete."""
        remaining = self.config.warmup_rounds - self._warmup_rounds_done
        return max(0, remaining)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Return cold-start state for persistence in .npz.

        Returns:
            Dict with bootstrap_step, warmup_complete,
            warmup_rounds_done, and startup_duration_seconds.
        """
        return {
            "bootstrap_step": self.bootstrap_step if self.bootstrap_step is not None else -1,
            "warmup_complete": self.warmup_complete,
            "warmup_rounds_done": self._warmup_rounds_done,
            "startup_duration_seconds": self.startup_time_seconds(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore cold-start state from persisted dict.

        Args:
            state: Dict with keys from get_state_dict().
        """
        bs = state.get("bootstrap_step", -1)
        self.bootstrap_step = bs if bs >= 0 else None
        self.warmup_complete = bool(state.get("warmup_complete", False))
        self._warmup_rounds_done = int(state.get("warmup_rounds_done", 0))
