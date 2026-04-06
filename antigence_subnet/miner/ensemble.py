"""
Ensemble detection -- averages results from multiple detectors on the same domain.

Simple average pooling strategy (v5.0). Weighted voting/Bayesian fusion
deferred to v6.0 per user decision.
"""

import numpy as np

from antigence_subnet.miner.detector import BaseDetector, DetectionResult


async def ensemble_detect(
    detectors: list[BaseDetector],
    prompt: str,
    output: str,
    code: str | None = None,
    context: str | None = None,
) -> DetectionResult:
    """Run multiple detectors and average their scores.

    Args:
        detectors: List of fitted BaseDetector instances for the same domain.
        prompt: Input prompt.
        output: AI output to verify.
        code: Optional code content.
        context: Optional context.

    Returns:
        DetectionResult with averaged score and confidence.
        anomaly_type from the first detector. feature_attribution=None.
    """
    if len(detectors) == 1:
        return await detectors[0].detect(prompt, output, code, context)

    results = []
    for detector in detectors:
        result = await detector.detect(prompt, output, code, context)
        results.append(result)

    avg_score = float(np.mean([r.score for r in results]))
    avg_confidence = float(np.mean([r.confidence for r in results]))

    return DetectionResult(
        score=avg_score,
        confidence=avg_confidence,
        anomaly_type=results[0].anomaly_type,
        feature_attribution=None,
    )
