"""BioDetector: domain-specific anomaly detection for computational biology pipeline outputs.

Uses 7 bio-specific features (numeric ranges, z-score outliers, unit mentions,
statistical terms) with IsolationForest for one-class anomaly detection.
Subclasses BaseDetector for integration with the Antigence subnet miner
detector registry.
"""


import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.domain_packs.bio.features import (
    extract_bio_features,
)

# Feature names for the bio-specific features
_BIO_FEATURE_NAMES = [
    "numeric_value_count",
    "out_of_range_count",
    "z_score_outlier_count",
    "negative_value_count",
    "unit_mention_count",
    "value_magnitude_range",
    "statistical_summary_count",
]


class BioDetector(BaseDetector):
    """Domain-specific detector for computational biology pipeline outputs.

    Detects anomalies in bio outputs: unexpected value ranges (pH outside 0-14,
    negative gene expression), statistical outliers (z-score > 3), and unit
    inconsistencies. Uses IsolationForest on 7 bio-specific features with
    percentile-normalized scoring.
    """

    domain = "bio"

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

    def _extract_domain_features(self, samples: list[dict]) -> np.ndarray:
        """Extract bio-specific features from a list of samples.

        Args:
            samples: List of sample dicts with prompt/output keys.

        Returns:
            2D numpy array of shape (n_samples, 7).
        """
        features = []
        for s in samples:
            f = extract_bio_features(s.get("prompt", ""), s.get("output", ""))
            features.append([f[name] for name in _BIO_FEATURE_NAMES])
        return np.array(features, dtype=np.float64)

    def _extract_single_features(self, prompt: str, output: str) -> np.ndarray:
        """Extract bio features for a single input.

        Returns:
            1D numpy array of shape (7,).
        """
        f = extract_bio_features(prompt, output)
        return np.array([f[name] for name in _BIO_FEATURE_NAMES], dtype=np.float64)

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Fits IsolationForest on 7-feature bio matrix. Stores sorted
        baseline scores for percentile normalization.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        X = self._extract_domain_features(samples)  # noqa: N806
        self.model.fit(X)
        # Store sorted baseline scores for percentile normalization
        self._baseline_scores_sorted = np.sort(self.model.score_samples(X))
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run anomaly detection on a single bio pipeline output.

        Transforms input to 7 bio features, scores with IsolationForest,
        and normalizes to [0, 1] via percentile against the training baseline.

        Args:
            prompt: Original prompt text.
            output: Bio pipeline output to verify.
            code: Optional code content (unused for bio domain).
            context: Optional metadata (unused for bio domain).

        Returns:
            DetectionResult with anomaly score, confidence, type, and
            feature attribution containing bio-specific features.
        """
        features = self._extract_single_features(prompt, output)
        X = features.reshape(1, -1)  # noqa: N806
        raw_score = self.model.score_samples(X)[0]

        # Hybrid percentile + deviation normalization
        # Same pattern as HallucinationDetector / IsolationForestDetector
        n = len(self._baseline_scores_sorted)
        idx = np.searchsorted(self._baseline_scores_sorted, raw_score, side="right")

        if 0 < idx < n:
            # Score within baseline range: standard percentile
            anomaly_score = float(np.clip(1.0 - (idx / n), 0.0, 1.0))
        else:
            # Score outside baseline range: use deviation from median
            median_score = self._baseline_scores_sorted[n // 2]
            baseline_range = (
                self._baseline_scores_sorted[-1] - self._baseline_scores_sorted[0]
            )
            if baseline_range < 1e-10:
                anomaly_score = 0.5
            else:
                deviation = abs(raw_score - median_score) / baseline_range
                anomaly_score = float(np.clip(deviation, 0.0, 1.0))

        confidence = float(min(abs(anomaly_score - 0.5) * 2.0, 1.0))
        anomaly_type = "bio_anomaly" if anomaly_score >= 0.5 else "normal"

        # Feature attribution: bio-specific feature values
        bio_features = extract_bio_features(prompt, output)
        feature_attribution: dict[str, float] = {
            k: v for k, v in bio_features.items()
        }

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=feature_attribution,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "BioDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "scikit-learn",
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
            f"{path}/bio_detector_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing bio_detector_state.joblib.
        """
        state = joblib.load(f"{path}/bio_detector_state.joblib")
        self.model = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._is_fitted = True
