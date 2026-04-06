"""
Evaluation dataset manager with hidden ground truth and honeypot injection.

Manages evaluation samples stored as JSON files, with ground truth labels
in a separate validator-only manifest. Supports deterministic round-based
sample rotation and honeypot injection at a configurable rate (default 20%).

Requirements: RWRD-02 (hidden evaluation datasets), RWRD-03 (honeypot injection)
"""

import hashlib
import json
from pathlib import Path

import bittensor as bt
import numpy as np


class EvaluationDataset:
    """Manages evaluation samples and hidden ground truth.

    Loads evaluation samples from a domain-specific directory containing:
    - samples.json: Evaluation samples WITHOUT ground truth labels
    - manifest.json: Ground truth labels and honeypot flags (validator-only)

    Provides deterministic round-based sample selection with configurable
    honeypot injection rate.
    """

    def __init__(self, data_dir: Path, domain: str):
        """Initialize evaluation dataset from a domain directory.

        Args:
            data_dir: Root evaluation data directory (e.g., data/evaluation).
            domain: Domain subdirectory name (e.g., "hallucination").
        """
        self.domain = domain
        self.data_dir = data_dir / domain
        self.samples = self._load_samples()
        self.manifest = self._load_manifest()

        # Partition samples into regular and honeypot sets
        self._regular_samples = [
            s
            for s in self.samples
            if not self.manifest.get(s["id"], {}).get("is_honeypot", False)
        ]
        self._honeypot_samples = [
            s
            for s in self.samples
            if self.manifest.get(s["id"], {}).get("is_honeypot", False)
        ]

        self.dataset_version = self._compute_version()

        bt.logging.info(
            f"Loaded evaluation dataset: {domain} | "
            f"{len(self.samples)} samples | "
            f"{len(self._honeypot_samples)} honeypots | "
            f"version {self.dataset_version}"
        )

    def _load_samples(self) -> list[dict]:
        """Load evaluation samples from samples.json.

        Returns:
            List of sample dicts (without ground truth labels).
        """
        samples_path = self.data_dir / "samples.json"
        with open(samples_path) as f:
            data = json.load(f)
        return data["samples"]

    def _load_manifest(self) -> dict:
        """Load ground truth manifest from manifest.json.

        Returns:
            Dict mapping sample_id to ground truth entry.
        """
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    def _compute_version(self) -> str:
        """Compute dataset version as SHA-256 hash of samples.json content.

        Returns:
            First 12 hex characters of the SHA-256 hash.
        """
        samples_path = self.data_dir / "samples.json"
        content = samples_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]

    def get_round_samples(
        self,
        round_num: int,
        n: int = 10,
        n_honeypots: int = 2,
        excluded_ids: set[str] | None = None,
    ) -> list[dict]:
        """Select samples for a specific evaluation round.

        Uses deterministic random selection seeded by round_num to ensure
        reproducible sample sets. Each round gets a mix of regular samples
        and honeypot samples.

        When ``excluded_ids`` is provided, samples with those IDs are
        filtered out before selection. This supports round-based challenge
        rotation (VHARD-01) -- miners who have seen certain samples in
        recent rounds will not receive them again.

        Args:
            round_num: Round number (typically validator step).
            n: Total number of samples per round.
            n_honeypots: Number of honeypot samples to include.
            excluded_ids: Sample IDs to exclude from selection. Default
                None means no exclusions (backward compatible).

        Returns:
            List of sample dicts for this round (shuffled).
        """
        n_regular = n - n_honeypots
        rng = np.random.default_rng(seed=round_num)

        # Filter out excluded IDs if provided
        if excluded_ids:
            available_regular = [
                s for s in self._regular_samples
                if s["id"] not in excluded_ids
            ]
            available_honeypots = [
                s for s in self._honeypot_samples
                if s["id"] not in excluded_ids
            ]
        else:
            available_regular = self._regular_samples
            available_honeypots = self._honeypot_samples

        if excluded_ids and (
            len(available_regular) < n_regular
            or len(available_honeypots) < n_honeypots
        ):
            bt.logging.warning(
                f"Rotation exclusion reduced pool: "
                f"{len(available_regular)} regular (need {n_regular}), "
                f"{len(available_honeypots)} honeypots (need {n_honeypots}) | "
                f"excluded={len(excluded_ids)} IDs"
            )

        # Select regular samples
        actual_n_regular = min(n_regular, len(available_regular))
        if actual_n_regular > 0:
            regular_indices = rng.choice(
                len(available_regular),
                size=actual_n_regular,
                replace=False,
            )
            selected_regular = [available_regular[i] for i in regular_indices]
        else:
            selected_regular = []

        # Select honeypot samples
        actual_n_honeypots = min(n_honeypots, len(available_honeypots))
        if actual_n_honeypots > 0:
            honeypot_indices = rng.choice(
                len(available_honeypots),
                size=actual_n_honeypots,
                replace=False,
            )
            selected_honeypots = [available_honeypots[i] for i in honeypot_indices]
        else:
            selected_honeypots = []

        # Combine and shuffle
        combined = selected_regular + selected_honeypots
        rng.shuffle(combined)

        return combined

    def get_ground_truth(self, sample_id: str) -> dict:
        """Get ground truth entry for a specific sample.

        Args:
            sample_id: The evaluation sample identifier.

        Returns:
            Ground truth dict with ground_truth_label and is_honeypot fields.
        """
        return self.manifest[sample_id]
