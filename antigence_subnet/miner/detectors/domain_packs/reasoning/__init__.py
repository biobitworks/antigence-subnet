"""Agent reasoning domain pack: chain-of-thought structural analysis for logical consistency."""

from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
    ReasoningDetector,
)
from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
    extract_reasoning_features,
)

__all__ = ["ReasoningDetector", "extract_reasoning_features"]
