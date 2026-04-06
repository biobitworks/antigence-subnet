"""Detector registry. Maps domain strings to detector classes."""

from antigence_subnet.miner.detector import BaseDetector
from antigence_subnet.miner.detectors.domain_packs.bio.detector import (  # noqa: E402
    BioDetector,
)
from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (  # noqa: E402
    CodeSecurityDetector,
)
from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (  # noqa: E402
    HallucinationDetector,
)
from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (  # noqa: E402
    ReasoningDetector,
)
from antigence_subnet.miner.detectors.fractal_complexity import (  # noqa: E402
    FractalComplexityDetector as FractalComplexityDetector,
)
from antigence_subnet.miner.detectors.isolation_forest import (  # noqa: E402
    IsolationForestDetector as IsolationForestDetector,
)
from antigence_subnet.miner.detectors.negsel import (  # noqa: E402
    NegSelAISDetector as NegSelAISDetector,
)
from antigence_subnet.miner.detectors.sklearn_backends import (  # noqa: E402
    LOFDetector as LOFDetector,
)
from antigence_subnet.miner.detectors.sklearn_backends import (  # noqa: E402
    OCSVMDetector as OCSVMDetector,
)

# AutoencoderDetector requires torch (optional dependency).
# Import conditionally to allow CPU-only environments (e.g., CI without torch).
try:
    from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector  # noqa: E402
except ImportError:
    AutoencoderDetector = None  # type: ignore[assignment, misc]

DETECTOR_REGISTRY: dict[str, list[type[BaseDetector]]] = {}


def register_detector(domain: str, detector_cls: type[BaseDetector]) -> None:
    """Register a detector class for a domain. Appends to list (supports ensemble).

    Args:
        domain: Domain string (e.g., 'hallucination').
        detector_cls: BaseDetector subclass to register.
    """
    if domain not in DETECTOR_REGISTRY:
        DETECTOR_REGISTRY[domain] = []
    DETECTOR_REGISTRY[domain].append(detector_cls)


def get_detector(domain: str) -> type[BaseDetector] | None:
    """Get the first detector class registered for a domain (backward compat).

    Args:
        domain: Domain string to look up.

    Returns:
        The first registered BaseDetector subclass, or None if not found.
    """
    detectors = DETECTOR_REGISTRY.get(domain, [])
    return detectors[0] if detectors else None


def get_detectors(domain: str) -> list[type[BaseDetector]]:
    """Get all detector classes registered for a domain.

    Args:
        domain: Domain string to look up.

    Returns:
        List of registered BaseDetector subclasses, or empty list if not found.
    """
    return DETECTOR_REGISTRY.get(domain, [])


# Register built-in detectors
# HallucinationDetector is the default domain pack for hallucination domain
# (replaces generic IsolationForestDetector with domain-specific features)
register_detector("hallucination", HallucinationDetector)
register_detector("code_security", CodeSecurityDetector)
register_detector("reasoning", ReasoningDetector)
register_detector("bio", BioDetector)
