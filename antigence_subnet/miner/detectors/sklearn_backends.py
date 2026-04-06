"""LOF and OCSVM anomaly detectors ported from parent Antigence platform.

Uses shared TF-IDF features with async detect via asyncio.to_thread().
LOFDetector wraps sklearn LocalOutlierFactor (novelty=True) with jitter fix
for macOS arm64 stability. OCSVMDetector wraps sklearn OneClassSVM.

Both detectors normalize anomaly scores to [0, 1] using hybrid
percentile + deviation normalization against training baselines.
"""

import asyncio

import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.features import create_vectorizer, samples_to_texts


class LOFDetector(BaseDetector):
    """Local Outlier Factor detector for hallucination domain.

    Ported from parent Antigence platform LOFAdapter. Uses sklearn
    LocalOutlierFactor in novelty detection mode with brute-force
    algorithm and jitter fix for macOS arm64 singular matrix stability.

    Args:
        max_features: Maximum TF-IDF features.
        ngram_range: N-gram range for TF-IDF extraction.
        n_neighbors: LOF neighbor count (auto-selected if None).
        random_state: Random seed for jitter reproducibility.
    """

    domain = "hallucination"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        n_neighbors: int | None = None,
        random_state: int = 42,
    ):
        self.vectorizer = create_vectorizer(max_features, ngram_range)
        self._n_neighbors = n_neighbors
        self._random_state = random_state
        self._clf: LocalOutlierFactor | None = None
        self._baseline_scores_sorted: np.ndarray | None = None
        self._min_score = 0.0
        self._max_score = 1.0
        self._is_fitted = False

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Fits TF-IDF vectorizer and LOF model. Includes jitter fix from
        parent platform to prevent singular matrix errors on macOS arm64.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        X = self.vectorizer.fit_transform(texts)  # noqa: N806
        X_dense = X.toarray().astype(np.float64)  # noqa: N806

        # Jitter fix from parent Antigence platform (macOS arm64 singular matrix)
        # Adds small proportional noise to prevent degenerate distance matrices
        rng = np.random.default_rng(self._random_state)
        std = np.std(X_dense, axis=0)
        jitter_scale = max(1e-3, float(np.mean(std)) * 0.01)
        X_dense += rng.normal(0, jitter_scale, X_dense.shape)  # noqa: N806

        # Auto-select n_neighbors: min(5, n_samples-1), at least 1
        n = self._n_neighbors or min(5, X_dense.shape[0] - 1)
        n = max(1, n)

        self._clf = LocalOutlierFactor(
            n_neighbors=n, novelty=True, algorithm="brute"
        )
        self._clf.fit(X_dense)

        # Margin normalization from parent platform
        train_scores = -self._clf.decision_function(X_dense)
        self._min_score = float(np.min(train_scores))
        self._max_score = float(np.max(train_scores))
        margin = (self._max_score - self._min_score) * 0.1
        self._min_score -= margin
        self._max_score += margin
        if self._max_score <= self._min_score:
            self._max_score = self._min_score + 1.0

        # Baseline for percentile normalization in detect()
        self._baseline_scores_sorted = np.sort(self._clf.score_samples(X_dense))
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
            code: Optional code content (unused for hallucination domain).
            context: Optional metadata (unused for hallucination domain).

        Returns:
            DetectionResult with percentile-normalized anomaly score.
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
        X = self.vectorizer.transform([text])  # noqa: N806
        X_dense = X.toarray()  # noqa: N806
        raw_score = self._clf.score_samples(X_dense)[0]

        # Hybrid percentile + deviation normalization
        n = len(self._baseline_scores_sorted)
        idx = np.searchsorted(self._baseline_scores_sorted, raw_score, side="right")

        if 0 < idx < n:
            anomaly_score = float(np.clip(1.0 - (idx / n), 0.0, 1.0))
        else:
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
            "name": "LOFDetector",
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
                "model": self._clf,
                "baseline_scores": self._baseline_scores_sorted,
                "min_score": self._min_score,
                "max_score": self._max_score,
            },
            f"{path}/lof_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing lof_state.joblib.
        """
        state = joblib.load(f"{path}/lof_state.joblib")
        self.vectorizer = state["vectorizer"]
        self._clf = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._min_score = state["min_score"]
        self._max_score = state["max_score"]
        self._is_fitted = True


class OCSVMDetector(BaseDetector):
    """One-Class SVM detector for hallucination domain.

    Ported from parent Antigence platform OneClassSVMAdapter. Uses sklearn
    OneClassSVM with RBF kernel and margin-normalized scoring.

    Args:
        max_features: Maximum TF-IDF features.
        ngram_range: N-gram range for TF-IDF extraction.
        kernel: SVM kernel type (default 'rbf').
        nu: Upper bound on training error fraction (default 0.1).
        random_state: Random seed for reproducibility.
    """

    domain = "hallucination"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        kernel: str = "rbf",
        nu: float = 0.1,
        random_state: int = 42,
    ):
        self.vectorizer = create_vectorizer(max_features, ngram_range)
        self._kernel = kernel
        self._nu = nu
        self._random_state = random_state
        self._clf: OneClassSVM | None = None
        self._baseline_scores_sorted: np.ndarray | None = None
        self._min_score = 0.0
        self._max_score = 1.0
        self._is_fitted = False

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Fits TF-IDF vectorizer and OneClassSVM model with margin-normalized
        score bounds from training data.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        X = self.vectorizer.fit_transform(texts)  # noqa: N806
        X_dense = X.toarray().astype(np.float64)  # noqa: N806

        self._clf = OneClassSVM(kernel=self._kernel, nu=self._nu)
        self._clf.fit(X_dense)

        # Margin normalization from parent platform
        train_scores = -self._clf.decision_function(X_dense)
        self._min_score = float(np.min(train_scores))
        self._max_score = float(np.max(train_scores))
        margin = (self._max_score - self._min_score) * 0.1
        self._min_score -= margin
        self._max_score += margin
        if self._max_score <= self._min_score:
            self._max_score = self._min_score + 1.0

        # Baseline: store sorted normalized scores for percentile in detect()
        normalized_train = (train_scores - self._min_score) / (
            self._max_score - self._min_score
        )
        self._baseline_scores_sorted = np.sort(normalized_train)
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
            code: Optional code content (unused for hallucination domain).
            context: Optional metadata (unused for hallucination domain).

        Returns:
            DetectionResult with margin-normalized anomaly score.
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
        X = self.vectorizer.transform([text])  # noqa: N806
        X_dense = X.toarray()  # noqa: N806

        # OCSVM uses decision_function for scoring (not score_samples)
        raw = -self._clf.decision_function(X_dense)[0]
        normalized = (raw - self._min_score) / (self._max_score - self._min_score)
        anomaly_score = float(np.clip(normalized, 0.0, 1.0))

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
            "name": "OCSVMDetector",
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
                "model": self._clf,
                "baseline_scores": self._baseline_scores_sorted,
                "min_score": self._min_score,
                "max_score": self._max_score,
            },
            f"{path}/ocsvm_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing ocsvm_state.joblib.
        """
        state = joblib.load(f"{path}/ocsvm_state.joblib")
        self.vectorizer = state["vectorizer"]
        self._clf = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._min_score = state["min_score"]
        self._max_score = state["max_score"]
        self._is_fitted = True
