"""
Abstract base detector interface for pluggable anomaly detection.

Miners implement BaseDetector subclasses for each domain they support.
The DetectionResult dataclass carries detector output back to the
miner forward function for synapse population.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result from a detector's detect() call.

    Attributes:
        score: Anomaly probability score (0.0=normal, 1.0=anomalous).
        confidence: Detector confidence in the score (0.0-1.0).
        anomaly_type: Type identifier string (e.g., 'fabricated_citation').
        feature_attribution: Optional feature importance scores.
    """

    score: float  # 0.0-1.0 anomaly score
    confidence: float  # 0.0-1.0 confidence
    anomaly_type: str  # type identifier string
    feature_attribution: dict[str, float] | None = None


class BaseDetector(ABC):
    """Abstract base for all domain detectors. Miners implement this interface.

    Each detector handles a specific domain (e.g., hallucination, code_security).
    Subclasses must implement fit() for training and detect() for inference.
    """

    domain: str  # The domain this detector handles

    @abstractmethod
    def fit(self, samples: list[dict]) -> None:
        """Train/fit the detector on normal (self) samples.

        Args:
            samples: List of sample dicts with keys matching
                VerificationSynapse fields (prompt, output, code, context).
        """
        ...

    @abstractmethod
    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run detection on input. Returns DetectionResult.

        Args:
            prompt: Original prompt or input text.
            output: AI-generated output to verify.
            code: Optional code content for code_security domain.
            context: Optional JSON-serialized metadata.

        Returns:
            DetectionResult with score, confidence, anomaly_type, and
            optional feature_attribution.
        """
        ...

    def save_state(self, path: str) -> None:  # noqa: B027
        """Save detector model state. Override for persistent detectors."""

    def load_state(self, path: str) -> None:  # noqa: B027
        """Load detector model state. Override for persistent detectors."""

    def get_info(self) -> dict:
        """Return detector metadata.

        Subclasses should override to set backend and is_fitted accurately.

        Returns:
            Dict with keys: name, domain, version, backend, is_fitted.
        """
        return {
            "name": self.__class__.__name__,
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "unknown",
            "is_fitted": False,
        }
