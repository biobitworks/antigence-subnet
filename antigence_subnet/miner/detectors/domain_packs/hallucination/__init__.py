"""Hallucination domain pack for detecting fabricated citations,
unsupported claims, and hallucinated facts."""

from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
    HallucinationDetector,
)
from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
    extract_hallucination_features,
)

__all__ = ["HallucinationDetector", "extract_hallucination_features"]
