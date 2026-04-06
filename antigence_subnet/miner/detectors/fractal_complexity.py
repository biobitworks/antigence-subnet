"""Fractal complexity anomaly detector using signal analysis on text.

Uses Higuchi Fractal Dimension, Hurst Exponent, and Shannon Entropy
to build an 8-feature complexity profile of text outputs, then applies
IsolationForest for anomaly scoring.

Combination approach adapted from Fractal Waves Project (Biobitworks).
All algorithms are published: Higuchi (1988), Hurst (1951), Shannon (1948).
"""

import asyncio

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.features import samples_to_texts
from antigence_subnet.miner.detectors.fractal_features import extract_fractal_features

FEATURE_NAMES = [
    "hfd_char_dist",
    "hfd_word_lengths",
    "hurst_word_lengths",
    "hurst_sentence_lengths",
    "shannon_char",
    "shannon_word",
    "shannon_bigram",
    "complexity_index",
]


class FractalComplexityDetector(BaseDetector):
    """Cross-domain anomaly detector using fractal complexity features.

    Extracts an 8-element fractal feature vector from text (Higuchi FD,
    Hurst exponent, Shannon entropy, complexity index) and scores anomalies
    using an IsolationForest backend.

    This detector is domain-agnostic -- it analyzes structural complexity
    of text rather than content similarity. Suitable as a complementary
    detector alongside domain-specific approaches.

    Args:
        contamination: IsolationForest contamination parameter.
        n_estimators: Number of isolation trees.
        random_state: Random seed for reproducibility.
    """

    domain = "fractal"

    def __init__(
        self,
        contamination: str | float = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._baseline_scores_sorted: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Extracts fractal features from each sample text and fits the
        IsolationForest model on the resulting (N, 8) feature matrix.
        Stores sorted baseline scores for percentile normalization.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        feature_matrix = np.array(
            [extract_fractal_features(text) for text in texts]
        )
        self.model.fit(feature_matrix)
        # Store sorted baseline scores for percentile normalization
        self._baseline_scores_sorted = np.sort(
            self.model.score_samples(feature_matrix)
        )
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run anomaly detection on a single input via thread pool.

        Wraps sync sklearn inference in asyncio.to_thread() for
        thread safety in the async miner forward loop.

        Args:
            prompt: Original prompt text.
            output: AI-generated output to verify.
            code: Optional code content (unused for fractal domain).
            context: Optional metadata (unused for fractal domain).

        Returns:
            DetectionResult with percentile-normalized anomaly score.
        """
        return await asyncio.to_thread(
            self._sync_detect, prompt, output, code, context
        )

    def _sync_detect(
        self,
        prompt: str,
        output: str,
        code: str | None,
        context: str | None,
    ) -> DetectionResult:
        """Synchronous detection logic."""
        text = f"{prompt} {output}"
        features = extract_fractal_features(text).reshape(1, -1)
        raw_score = self.model.score_samples(features)[0]

        # Hybrid percentile + deviation normalization
        # (same pattern as IsolationForestDetector)
        n = len(self._baseline_scores_sorted)
        idx = np.searchsorted(
            self._baseline_scores_sorted, raw_score, side="right"
        )

        if 0 < idx < n:
            # Score within baseline range: standard percentile
            anomaly_score = float(np.clip(1.0 - (idx / n), 0.0, 1.0))
        else:
            # Score outside baseline range: use deviation from median
            median_score = self._baseline_scores_sorted[n // 2]
            baseline_range = (
                self._baseline_scores_sorted[-1]
                - self._baseline_scores_sorted[0]
            )
            if baseline_range < 1e-10:
                anomaly_score = 0.5
            else:
                deviation = abs(raw_score - median_score) / baseline_range
                anomaly_score = float(np.clip(deviation, 0.0, 1.0))

        confidence = float(min(abs(anomaly_score - 0.5) * 2.0, 1.0))
        anomaly_type = "anomalous" if anomaly_score >= 0.5 else "normal"

        # Feature attribution: fractal feature names with their values
        feature_attribution = dict(
            zip(FEATURE_NAMES, features[0].tolist(), strict=True)
        )

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=feature_attribution,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "FractalComplexityDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "scikit-learn+nolds",
            "is_fitted": self._is_fitted,
        }

    def save_state(self, path: str) -> None:
        """Save model state to disk via joblib.

        Args:
            path: Directory to save state files in.
        """
        joblib.dump(
            {
                "model": self.model,
                "baseline_scores": self._baseline_scores_sorted,
            },
            f"{path}/fractal_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing fractal_state.joblib.
        """
        state = joblib.load(f"{path}/fractal_state.joblib")
        self.model = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._is_fitted = True
