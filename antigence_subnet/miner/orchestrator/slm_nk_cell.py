"""SLM-powered NK Cell semantic anomaly gate for the immune orchestrator.

The rule-based NK Cell (nk_cell.py) catches statistical outliers via z-score
thresholds on dendritic features. SLMNKCell adds a complementary semantic
gate: it uses ModelManager.score() to compute cosine similarity between
prompt and output embeddings. If the similarity falls below a configurable
threshold, the output is flagged as semantically disconnected -- it may have
normal feature distributions but says something unrelated to the prompt.

Design decisions:
- D-01: SLMNKCell wraps ModelManager.score() for semantic similarity
- D-02: Returns DetectionResult(anomaly_type="slm_nk_semantic_disconnect")
- D-04: Graceful degradation -- returns None when ModelManager unavailable
- D-05: Default similarity_threshold=0.3, configurable via TOML/kwarg

Satisfies ImmuneCellType Protocol via duck typing (structural subtyping).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from antigence_subnet.miner.detector import DetectionResult

if TYPE_CHECKING:
    from antigence_subnet.miner.orchestrator.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SLMNKCell:
    """SLM-powered NK Cell semantic anomaly gate.

    Uses ModelManager.score(prompt, output) to compute semantic similarity.
    When similarity falls below the threshold, returns a DetectionResult
    flagging the output as semantically disconnected from the prompt.

    Satisfies ImmuneCellType Protocol via duck typing (structural subtyping).

    Args:
        model_manager: ModelManager instance for embed/score operations.
        similarity_threshold: Minimum cosine similarity to pass the gate.
            Outputs with similarity below this threshold are flagged.
            Defaults to 0.3.
        enabled: Whether this cell is active. When False, process()
            always returns None.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        similarity_threshold: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self._model_manager = model_manager
        self._similarity_threshold = similarity_threshold
        self._enabled = enabled
        self._unavailable_warned = False

    def process(
        self,
        features: np.ndarray,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
        *,
        similarity_threshold: float | None = None,
    ) -> DetectionResult | None:
        """Evaluate semantic similarity between prompt and output.

        Returns DetectionResult if similarity is below the threshold
        (semantic disconnect detected). Returns None if similarity
        is above threshold (output passes gate), or if the model
        manager is unavailable (graceful degradation).

        Args:
            features: Feature vector (shape (10,) from DendriticFeatureExtractor).
                Not used by SLMNKCell but required by ImmuneCellType Protocol.
            prompt: Original prompt text.
            output: AI-generated output text.
            code: Optional code content (unused, Protocol conformance).
            context: Optional JSON-serialized metadata (unused).
            similarity_threshold: Optional per-request threshold override.
                When not None, used instead of self._similarity_threshold
                for this invocation (per-domain config support).

        Returns:
            DetectionResult with score=1-similarity if semantic disconnect
            detected, None otherwise.
        """
        if not self._enabled:
            return None

        if not self._model_manager.is_available():
            if not self._unavailable_warned:
                logger.warning(
                    "SLMNKCell: ModelManager not available -- "
                    "semantic gate disabled (graceful degradation)"
                )
                self._unavailable_warned = True
            return None

        try:
            similarity = self._model_manager.score(prompt, output)
        except Exception:
            logger.warning(
                "SLMNKCell: ModelManager.score() raised exception -- "
                "returning None (graceful degradation)",
                exc_info=True,
            )
            return None

        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )

        if similarity >= effective_threshold:
            return None

        anomaly_score = 1.0 - similarity
        return DetectionResult(
            score=anomaly_score,
            confidence=anomaly_score,
            anomaly_type="slm_nk_semantic_disconnect",
        )
