"""NegSl-AIS negative selection detector using dendritic features.

Implements the Umair et al. 2025 negative selection algorithm (Equations 20-21)
with adaptive r_self computed from training data geometry. Operates in the
10-dimensional dendritic feature space (not TF-IDF).

Wraps sync numpy computation in ``asyncio.to_thread()`` for thread safety
in the async miner forward loop, following the same pattern as LOFDetector.

State persistence via joblib for miner restart resilience.
"""

import asyncio
import logging

import joblib
import numpy as np

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.dendritic_features import DendriticFeatureExtractor
from antigence_subnet.miner.detectors.features import samples_to_texts


class NegSelAISDetector(BaseDetector):
    """NegSl-AIS negative selection detector using dendritic features.

    Implements Umair et al. 2025 (Equations 20-21) with adaptive r_self.
    Operates in 10-dim dendritic feature space (not TF-IDF).

    Detectors are generated in the unit hypercube [0,1]^10. A candidate
    becomes a valid detector only if its nearest distance to any self
    sample exceeds r_self (negative selection). The detector radius is
    ``R_q - r_self``.

    Args:
        num_detectors: Number of detectors to generate.
        r_self: Self radius. If ``None``, computed adaptively as the 95th
            percentile of nearest-neighbor distances in training data.
        max_attempts: Maximum random candidate attempts during fit.
        random_state: Seed for reproducible detector generation.
    """

    domain = "hallucination"

    def __init__(
        self,
        num_detectors: int = 50,
        r_self: float | None = None,
        max_attempts: int = 5000,
        random_state: int = 42,
    ):
        self._num_detectors = num_detectors
        self._r_self_param = r_self
        self._max_attempts = max_attempts
        self._random_state = random_state

        self._feature_extractor = DendriticFeatureExtractor()
        self._is_fitted = False
        self._valid_detectors: list[dict] = []  # list of {"center": np.array, "radius": float}
        self._self_features: np.ndarray | None = None
        self._effective_r_self: float = 0.0
        self._score_min: float = 0.0
        self._score_max: float = 1.0

        self._rng = np.random.default_rng(random_state)
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples via negative selection.

        Extracts dendritic features, computes adaptive r_self (if not
        provided), generates detectors in non-self space, and calibrates
        the anomaly score mapping.

        Args:
            samples: List of sample dicts with ``prompt`` and ``output`` keys.
        """
        texts = samples_to_texts(samples)
        self._self_features = self._feature_extractor.extract_batch(texts)

        # Determine r_self
        if self._r_self_param is not None:
            self._effective_r_self = self._r_self_param
        else:
            self._effective_r_self = self._compute_adaptive_r_self(self._self_features)

        self._logger.info(
            "NegSel fit: %d samples, r_self=%.4f (adaptive=%s)",
            len(texts),
            self._effective_r_self,
            self._r_self_param is None,
        )

        # Generate detectors (negative selection)
        self._valid_detectors = []
        attempts = 0
        while (
            len(self._valid_detectors) < self._num_detectors
            and attempts < self._max_attempts
        ):
            attempts += 1
            candidate = self._rng.uniform(0, 1, self._self_features.shape[1])

            # R_q = min distance to any self sample (Eq 21)
            r_q = float(np.min(np.linalg.norm(self._self_features - candidate, axis=1)))

            if r_q > self._effective_r_self:
                radius = r_q - self._effective_r_self

                # Duplicate check
                is_dup = any(
                    np.allclose(d["center"], candidate, atol=1e-5)
                    for d in self._valid_detectors
                )
                if not is_dup:
                    self._valid_detectors.append({"center": candidate, "radius": radius})

        if len(self._valid_detectors) < self._num_detectors:
            self._logger.warning(
                "NegSel: only %d/%d detectors generated in %d attempts",
                len(self._valid_detectors),
                self._num_detectors,
                self._max_attempts,
            )

        self._calibrate_scores()
        self._is_fitted = True

    def _compute_adaptive_r_self(self, features: np.ndarray) -> float:
        """Compute r_self from the self-data pairwise distance distribution.

        Uses the 95th percentile of nearest-neighbor distances among self
        samples. This ensures nearly all self-samples score 0 (within r_self
        of their nearest neighbor), while setting the boundary just beyond
        the natural self-cluster extent.

        Args:
            features: Self-sample feature matrix of shape ``(N, 10)``.

        Returns:
            Adaptive r_self value, floored at 0.05 and capped at 2.0.
        """
        n = features.shape[0]
        if n < 2:
            return 0.15  # Fallback to LLM_HALLUCINATION preset default

        # Pairwise distance matrix O(N^2), N typically 20-30
        dists = np.linalg.norm(
            features[:, np.newaxis] - features[np.newaxis, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        r_self = float(np.percentile(nn_dists, 95))
        return max(0.05, min(2.0, r_self))

    def _calibrate_scores(self) -> None:
        """Compute score calibration so self -> 0 and boundary -> 0.5.

        Strategy: self-samples should score near 0. The max self-sample
        score defines the boundary of "normal". Normalization maps the
        boundary to ~0.3, giving anomalies room to score 0.5-1.0.
        """
        if self._self_features is None or len(self._self_features) == 0:
            self._score_min = 0.0
            self._score_max = 1.0
            return

        self_scores = np.array([self._raw_score(s) for s in self._self_features])
        max_self_score = float(np.max(self_scores))

        self._score_min = 0.0
        if max_self_score > 0:
            # Normalize so max self score maps to ~0.3
            self._score_max = max_self_score / 0.3
        else:
            # All self-samples score 0 (ideal). Use r_self as reference.
            self._score_max = (
                self._effective_r_self / 0.5
                if self._effective_r_self > 0
                else 1.0
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _raw_score(self, features: np.ndarray) -> float:
        """Raw uncalibrated score: max(0, r_q - r_self)."""
        r_q = float(np.min(np.linalg.norm(self._self_features - features, axis=1)))
        return max(0.0, r_q - self._effective_r_self)

    def _get_calibrated_score(self, features: np.ndarray) -> float:
        """Calibrated anomaly score normalized to [0, 1]."""
        raw = self._raw_score(features)
        score_range = self._score_max - self._score_min
        if score_range <= 0:
            return 0.0
        normalized = (raw - self._score_min) / score_range
        return float(np.clip(normalized, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Detection (async + sync)
    # ------------------------------------------------------------------

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run anomaly detection on a single input via thread pool.

        Wraps sync numpy inference in ``asyncio.to_thread()`` for
        thread safety in the async miner forward loop.

        Args:
            prompt: Original prompt text.
            output: AI-generated output to verify.
            code: Optional code content (unused).
            context: Optional metadata (unused).

        Returns:
            DetectionResult with calibrated anomaly score and dendritic
            feature attribution.
        """
        return await asyncio.to_thread(self._sync_detect, prompt, output, code, context)

    def _sync_detect(
        self,
        prompt: str,
        output: str,
        code: str | None,
        context: str | None,
    ) -> DetectionResult:
        """Synchronous detection logic."""
        text = f"{prompt} {output}"
        features = self._feature_extractor.extract(text)
        score = self._get_calibrated_score(features)
        confidence = float(min(abs(score - 0.5) * 2.0, 1.0))
        anomaly_type = "anomaly" if score >= 0.5 else "normal"
        attribution = self._feature_extractor.extract_with_names(text)
        return DetectionResult(
            score=score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=attribution,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "NegSelAISDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "negsel-ais",
            "is_fitted": self._is_fitted,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save detector state to disk via joblib.

        Stores self-features, generated detectors, and score calibration
        parameters for restart resilience.

        Args:
            path: Directory to save state files in.
        """
        joblib.dump(
            {
                "self_features": self._self_features,
                "valid_detectors": self._valid_detectors,
                "effective_r_self": self._effective_r_self,
                "score_min": self._score_min,
                "score_max": self._score_max,
            },
            f"{path}/negsel_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load detector state from disk.

        Args:
            path: Directory containing ``negsel_state.joblib``.
        """
        state = joblib.load(f"{path}/negsel_state.joblib")
        self._self_features = state["self_features"]
        self._valid_detectors = state["valid_detectors"]
        self._effective_r_self = state["effective_r_self"]
        self._score_min = state["score_min"]
        self._score_max = state["score_max"]
        self._is_fitted = True
