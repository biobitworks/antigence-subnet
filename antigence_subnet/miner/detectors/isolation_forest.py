"""IsolationForest-based CPU anomaly detector for the hallucination domain.

Uses scikit-learn's IsolationForest with TF-IDF features. Trained on normal
(self) samples only. Anomaly scores are percentile-normalized to [0, 1]
against the training baseline distribution.
"""


import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.features import create_vectorizer, samples_to_texts


class IsolationForestDetector(BaseDetector):
    """CPU-based anomaly detector using IsolationForest with TF-IDF features."""

    domain = "hallucination"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        contamination: str | float = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.vectorizer = create_vectorizer(max_features, ngram_range)
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._baseline_scores_sorted: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Fits the TF-IDF vectorizer and IsolationForest model, then computes
        and stores the sorted baseline score distribution for percentile
        normalization during detection.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        X = self.vectorizer.fit_transform(texts)  # noqa: N806
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
        """Run anomaly detection on a single input.

        Transforms input text to TF-IDF features, computes IsolationForest
        score, and normalizes to [0, 1] via percentile against baseline.

        Args:
            prompt: Original prompt text.
            output: AI-generated output to verify.
            code: Optional code content (unused for hallucination domain).
            context: Optional metadata (unused for hallucination domain).

        Returns:
            DetectionResult with percentile-normalized anomaly score.
        """
        text = f"{prompt} {output}"
        X = self.vectorizer.transform([text])  # noqa: N806
        raw_score = self.model.score_samples(X)[0]

        # Hybrid percentile + deviation normalization.
        # Use searchsorted for in-range scores; fall back to deviation
        # from median for out-of-range scores (common with small datasets).
        n = len(self._baseline_scores_sorted)
        idx = np.searchsorted(self._baseline_scores_sorted, raw_score, side="right")

        if 0 < idx < n:
            # Score within baseline range: standard percentile
            anomaly_score = float(np.clip(
                1.0 - (idx / n),
                0.0,
                1.0,
            ))
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
        anomaly_type = "hallucination" if anomaly_score >= 0.5 else "normal"

        # Feature attribution: top-5 TF-IDF features by weight
        feature_names = self.vectorizer.get_feature_names_out()
        weights = X.toarray()[0]
        top_indices = np.argsort(weights)[-5:][::-1]
        feature_attribution: dict[str, float] = {
            feature_names[i]: float(weights[i])
            for i in top_indices
            if weights[i] > 0
        }

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=feature_attribution if feature_attribution else None,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "IsolationForestDetector",
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
                "vectorizer": self.vectorizer,
                "model": self.model,
                "baseline_scores": self._baseline_scores_sorted,
            },
            f"{path}/iforest_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing iforest_state.joblib.
        """
        state = joblib.load(f"{path}/iforest_state.joblib")
        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._is_fitted = True
