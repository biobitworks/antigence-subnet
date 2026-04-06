"""
VerificationSynapse protocol definition.

Typed communication contract between validators and miners for the
Antigence verification subnet. Validators fill request fields and send
to miners; miners fill response fields and return.
"""

from typing import ClassVar

import bittensor as bt
from pydantic import Field

# Domain constants (D-05)
DOMAIN_HALLUCINATION = "hallucination"
DOMAIN_CODE_SECURITY = "code_security"
DOMAIN_REASONING = "reasoning"
DOMAIN_BIO = "bio"
KNOWN_DOMAINS = frozenset(
    {DOMAIN_HALLUCINATION, DOMAIN_CODE_SECURITY, DOMAIN_REASONING, DOMAIN_BIO}
)


class VerificationSynapse(bt.Synapse):
    """
    Protocol for verification requests.

    Validator sends input fields, miner fills output fields.
    Transported as JSON over HTTP via the Bittensor Axon/Dendrite pattern.
    """

    # --- Request fields (required, set by validator) ---
    prompt: str = Field(description="Original prompt or input text")
    output: str = Field(description="AI-generated output to verify")
    domain: str = Field(description="Detection domain identifier")

    # --- Request fields (optional) ---
    code: str | None = Field(
        default=None, description="Code content for code_security domain"
    )
    context: str | None = Field(
        default=None, description="JSON-serialized metadata"
    )
    seed: int | None = Field(
        default=None,
        description="Best-effort validator seed hint; miners may ignore it.",
    )

    # --- Response fields (optional, filled by miner) ---
    anomaly_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Anomaly probability score (0.0=normal, 1.0=anomalous)",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detector confidence in the score",
    )
    anomaly_type: str | None = Field(
        default=None,
        description="Type of anomaly detected (e.g., 'fabricated_citation', 'sql_injection')",
    )
    feature_attribution: dict[str, float] | None = Field(
        default=None,
        description="Feature importance scores explaining the detection",
    )

    # Body hash fields for integrity verification
    required_hash_fields: ClassVar[tuple[str, ...]] = ("prompt", "output", "domain")

    def deserialize(self) -> float | None:
        """Deserialize to anomaly_score for simple reward computation."""
        return self.anomaly_score
