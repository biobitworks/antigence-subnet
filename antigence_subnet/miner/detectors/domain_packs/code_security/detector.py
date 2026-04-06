"""CodeSecurityDetector: domain-specific anomaly detection for insecure code patterns.

Uses AST-based feature extraction (7 features) with IsolationForest for
one-class anomaly detection. Detects SQL injection, command injection,
hardcoded credentials, insecure deserialization, and path traversal patterns.
Subclasses BaseDetector for integration with the Antigence subnet miner
detector registry.
"""


import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
    code_from_sample,
    extract_code_security_features,
)

# Feature names for code security features
_CODE_SECURITY_FEATURE_NAMES = [
    "dangerous_call_count",
    "string_concat_in_call",
    "hardcoded_credential_score",
    "import_risk_score",
    "ast_node_depth",
    "exec_eval_usage",
    "total_function_calls",
]


class CodeSecurityDetector(BaseDetector):
    """Domain-specific detector for insecure code patterns.

    Uses AST-based feature extraction to identify dangerous function calls,
    string concatenation in queries (SQL injection), hardcoded credentials,
    risky imports, eval/exec usage, and deep nesting (obfuscation).
    IsolationForest provides one-class anomaly detection with percentile-
    normalized scoring.
    """

    domain = "code_security"

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

    def _extract_features_from_samples(self, samples: list[dict]) -> np.ndarray:
        """Extract code security features from a list of samples.

        Args:
            samples: List of sample dicts with code/output keys.

        Returns:
            2D numpy array of shape (n_samples, 7).
        """
        features = []
        for s in samples:
            code = code_from_sample(s)
            f = extract_code_security_features(code)
            features.append([f[name] for name in _CODE_SECURITY_FEATURE_NAMES])
        return np.array(features, dtype=np.float64)

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.

        Extracts AST-based features from each sample's code and fits
        IsolationForest. Stores sorted baseline scores for percentile
        normalization.

        Args:
            samples: List of normal sample dicts with code/output keys.
        """
        X = self._extract_features_from_samples(samples)  # noqa: N806
        self.model.fit(X)
        self._baseline_scores_sorted = np.sort(self.model.score_samples(X))
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run anomaly detection on a single code input.

        Uses code if provided, otherwise falls back to output.

        Args:
            prompt: Original prompt text.
            output: AI-generated output (used as code fallback).
            code: Optional code content (preferred if provided).
            context: Optional metadata (unused for code security domain).

        Returns:
            DetectionResult with anomaly score, confidence, type, and
            feature attribution containing code security feature values.
        """
        source = code if code else output
        features = extract_code_security_features(source)
        feature_vec = np.array(
            [features[name] for name in _CODE_SECURITY_FEATURE_NAMES],
            dtype=np.float64,
        ).reshape(1, -1)

        raw_score = self.model.score_samples(feature_vec)[0]

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
        anomaly_type = "insecure_code" if anomaly_score >= 0.5 else "normal"

        feature_attribution: dict[str, float] = {k: v for k, v in features.items()}

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=feature_attribution,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "CodeSecurityDetector",
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
            f"{path}/code_security_detector_state.joblib",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing code_security_detector_state.joblib.
        """
        state = joblib.load(f"{path}/code_security_detector_state.joblib")
        self.model = state["model"]
        self._baseline_scores_sorted = state["baseline_scores"]
        self._is_fitted = True
