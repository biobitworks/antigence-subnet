"""HallucinationDetector: domain-specific anomaly detection for LLM hallucinations.

Supports two feature paths:
  - **sbert** (default): 384-dim dense semantic embeddings from all-MiniLM-L6-v2
    combined with 7 hallucination-specific features via np.hstack.
  - **tfidf** (fallback): 5000-dim sparse TF-IDF features combined with
    hallucination features via scipy sparse_hstack.

Falls back to tfidf gracefully when sentence-transformers is not installed.
Uses IsolationForest for one-class anomaly detection with percentile-normalized scoring.
"""

import warnings

import joblib
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from sklearn.ensemble import IsolationForest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
    extract_hallucination_features,
)
from antigence_subnet.miner.detectors.features import create_vectorizer, samples_to_texts

# Feature names for the hallucination-specific features
_HALLUCINATION_FEATURE_NAMES = [
    "claim_density",
    "citation_count",
    "citation_pattern_score",
    "hedging_ratio",
    "numeric_claim_density",
    "avg_sentence_length",
    "text_length",
]


class HallucinationDetector(BaseDetector):
    """Domain-specific detector for LLM hallucinations.

    Combines text embeddings (sbert or TF-IDF) with hallucination-specific features:
    claim density, citation count/quality, hedging ratio, numeric density,
    sentence length, and text length. Uses IsolationForest for one-class
    anomaly detection with percentile-normalized scoring.

    Args:
        max_features: Maximum number of TF-IDF features (only used when
            embedding_method is "tfidf").
        ngram_range: N-gram range for TF-IDF token extraction.
        contamination: IsolationForest contamination parameter.
        n_estimators: Number of IsolationForest estimators.
        random_state: Random seed for reproducibility.
        embedding_method: "sbert" for semantic embeddings (default) or
            "tfidf" for sparse TF-IDF features. Falls back to "tfidf"
            with a warning if sentence-transformers is not installed.
    """

    domain = "hallucination"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        contamination: str | float = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
        embedding_method: str = "sbert",
    ):
        self.embedding_method = embedding_method

        # Validate and potentially fall back
        if self.embedding_method == "sbert":
            from antigence_subnet.miner.detectors.embeddings import is_sbert_available

            if not is_sbert_available():
                warnings.warn(
                    "sentence-transformers not installed, falling back to tfidf",
                    stacklevel=2,
                )
                self.embedding_method = "tfidf"

        # Only create vectorizer for tfidf path
        if self.embedding_method == "tfidf":
            self.vectorizer = create_vectorizer(max_features, ngram_range)
        else:
            self.vectorizer = None

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._baseline_scores_sorted: np.ndarray | None = None
        self._is_fitted = False

    def _extract_domain_features(self, samples: list[dict]) -> np.ndarray:
        """Extract hallucination-specific features from a list of samples.

        Args:
            samples: List of sample dicts with prompt/output keys.

        Returns:
            2D numpy array of shape (n_samples, 7).
        """
        features = []
        for s in samples:
            f = extract_hallucination_features(
                s.get("prompt", ""), s.get("output", "")
            )
            features.append([f[name] for name in _HALLUCINATION_FEATURE_NAMES])
        return np.array(features, dtype=np.float64)

    def _extract_single_domain_features(self, prompt: str, output: str) -> np.ndarray:
        """Extract hallucination features for a single input.

        Returns:
            1D numpy array of shape (7,).
        """
        f = extract_hallucination_features(prompt, output)
        return np.array([f[name] for name in _HALLUCINATION_FEATURE_NAMES], dtype=np.float64)

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Fits the IsolationForest on the combined feature space of
        text embeddings (sbert or TF-IDF) + hallucination-specific features.
        Stores sorted baseline scores for percentile normalization.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        domain_matrix = self._extract_domain_features(samples)

        if self.embedding_method == "sbert":
            from antigence_subnet.miner.detectors.embeddings import encode_texts

            embedding_matrix = encode_texts(texts)
            X = np.hstack([embedding_matrix, domain_matrix])  # noqa: N806
        else:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            X = sparse_hstack([tfidf_matrix, domain_matrix]).tocsr()  # noqa: N806

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

        Transforms input to combined embedding + hallucination features,
        scores with IsolationForest, and normalizes to [0, 1] via
        percentile against the training baseline.

        Args:
            prompt: Original prompt text.
            output: AI-generated output to verify.
            code: Optional code content (unused for hallucination domain).
            context: Optional metadata (unused for hallucination domain).

        Returns:
            DetectionResult with anomaly score, confidence, type, and
            feature attribution containing hallucination-specific features.
        """
        text = f"{prompt} {output}"
        domain_vec = self._extract_single_domain_features(prompt, output).reshape(1, -1)

        if self.embedding_method == "sbert":
            from antigence_subnet.miner.detectors.embeddings import encode_texts

            embedding_vec = encode_texts([text])
            X = np.hstack([embedding_vec, domain_vec])  # noqa: N806
        else:
            tfidf_vec = self.vectorizer.transform([text])
            X = sparse_hstack([tfidf_vec, domain_vec]).tocsr()  # noqa: N806

        raw_score = self.model.score_samples(X)[0]

        # Hybrid percentile + deviation normalization
        # Same pattern as IsolationForestDetector
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
        anomaly_type = "hallucination" if anomaly_score >= 0.5 else "normal"

        # Feature attribution: hallucination-specific features + embedding info
        hall_features = extract_hallucination_features(prompt, output)
        feature_attribution: dict[str, float] = {
            k: v for k, v in hall_features.items()
        }

        if self.embedding_method == "sbert":
            # Top-5 embedding dimensions by magnitude
            emb_values = embedding_vec[0]
            top_indices = np.argsort(np.abs(emb_values))[-5:][::-1]
            for i in top_indices:
                feature_attribution[f"sbert_dim_{i}"] = float(emb_values[i])
        else:
            # Add top-5 TF-IDF features by weight
            feature_names = self.vectorizer.get_feature_names_out()
            weights = tfidf_vec.toarray()[0]
            top_indices = np.argsort(weights)[-5:][::-1]
            for i in top_indices:
                if weights[i] > 0:
                    feature_attribution[f"tfidf_{feature_names[i]}"] = float(weights[i])

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=feature_attribution,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "HallucinationDetector",
            "domain": self.domain,
            "version": "0.2.0",
            "backend": "scikit-learn",
            "is_fitted": self._is_fitted,
            "embedding_method": self.embedding_method,
        }

    def save_state(self, path: str) -> None:
        """Save model state to disk via joblib.

        Args:
            path: Directory to save state files in.
        """
        state = {
            "model": self.model,
            "baseline_scores": self._baseline_scores_sorted,
            "embedding_method": self.embedding_method,
        }
        # Only include vectorizer for tfidf path
        if self.embedding_method == "tfidf":
            state["vectorizer"] = self.vectorizer
        joblib.dump(state, f"{path}/hallucination_detector_state.joblib")

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing hallucination_detector_state.joblib.

        Raises:
            RuntimeError: If saved embedding_method doesn't match current config.
        """
        state = joblib.load(f"{path}/hallucination_detector_state.joblib")

        # Check embedding_method compatibility (default "tfidf" for old state files)
        saved_method = state.get("embedding_method", "tfidf")
        if saved_method != self.embedding_method:
            raise RuntimeError(
                f"State saved with {saved_method} but detector configured "
                f"for {self.embedding_method}"
            )

        self.model = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        if self.embedding_method == "tfidf":
            self.vectorizer = state.get("vectorizer", self.vectorizer)
        self._is_fitted = True
