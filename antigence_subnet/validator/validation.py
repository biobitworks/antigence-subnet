"""
Structural synapse validation module.

Pre-scoring gate that checks miner responses for structural validity
before they enter the reward computation pipeline. Invalid responses
are scored 0.0 (same as no-response) with logged rejection reasons.

Full implementation in Task 2 (02-01). This provides the initial
working version for reward.py integration.
"""

import bittensor as bt

# Known anomaly types across all domains
KNOWN_ANOMALY_TYPES = frozenset({
    # Hallucination domain
    "factual_error", "fabricated_citation", "unsupported_claim",
    # Code security domain
    "sql_injection", "xss", "code_backdoor", "buffer_overflow",
    # Reasoning domain
    "logic_inconsistency", "constraint_violation",
    # Bio domain
    "data_anomaly", "pipeline_error",
    # Testing
    "mock_anomaly",
    # None is valid (no anomaly detected)
    None,
})


def validate_response(response) -> tuple[bool, str]:
    """Validate structural integrity of a miner response.

    Checks:
        1. anomaly_score must not be None
        2. anomaly_score must be in [0.0, 1.0]
        3. anomaly_type must be in KNOWN_ANOMALY_TYPES (if provided)

    Args:
        response: A VerificationSynapse or compatible object with
            anomaly_score and anomaly_type attributes.

    Returns:
        Tuple of (is_valid, rejection_reason). If valid, reason is "".
    """
    # Check 1: anomaly_score must exist
    if response.anomaly_score is None:
        reason = "missing_anomaly_score"
        bt.logging.warning(f"Synapse validation rejected: {reason}")
        return False, reason

    # Check 2: anomaly_score must be in [0.0, 1.0]
    # Pydantic enforces this on VerificationSynapse, but defense-in-depth
    # for raw JSON that bypasses construction.
    if not (0.0 <= response.anomaly_score <= 1.0):
        reason = f"anomaly_score_out_of_range:{response.anomaly_score}"
        bt.logging.warning(f"Synapse validation rejected: {reason}")
        return False, reason

    # Check 3: anomaly_type must be known (if provided)
    if response.anomaly_type is not None and response.anomaly_type not in KNOWN_ANOMALY_TYPES:
        reason = f"unknown_anomaly_type:{response.anomaly_type}"
        bt.logging.warning(f"Synapse validation rejected: {reason}")
        return False, reason

    return True, ""
