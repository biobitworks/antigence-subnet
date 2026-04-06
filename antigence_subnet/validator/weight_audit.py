"""
Pre-commit weight audit and commit-reveal verification.

Implements CHEAT-05 (weight copying detection), CHEAT-06 (weight anomaly
detection), and NET-06 (commit-reveal status check).

Weight audit detects three anomaly types before weights are set on chain:
1. Near-uniform weights (no miner discrimination)
2. Extreme concentration (single miner dominance)
3. Cross-validator similarity (weight copying)

Commit-reveal check is informational: the SDK v10.2.0 set_weights()
automatically routes to commit-reveal when enabled on the subnet.
No custom implementation needed.
"""

import bittensor as bt
import numpy as np


def audit_weights(
    our_weights: np.ndarray,
    recent_validator_weights: dict[str, np.ndarray] | None = None,
    similarity_threshold: float = 0.99,
) -> list[str]:
    """Audit a weight vector for anomalous patterns before chain submission.

    Checks for three anomaly types:
    1. Near-uniform: weights show no discrimination among miners
    2. Extreme concentration: one miner gets > 50% of weight
    3. Cross-validator similarity: our weights match another validator's
       (possible weight copying)

    Args:
        our_weights: Weight vector to audit (float array).
        recent_validator_weights: Optional dict mapping validator hotkey
            (str) -> weight vector (np.ndarray) for cross-validator check.
            Uses 0.99 threshold (not 0.95) to avoid flagging legitimate
            convergence (pitfall 4 from research).
        similarity_threshold: Cosine similarity threshold for
            cross-validator weight copying detection. Default 0.99.

    Returns:
        List of warning strings (empty = clean weight vector).
    """
    warnings: list[str] = []

    if recent_validator_weights is None:
        recent_validator_weights = {}

    # Edge case: all-zero weights -- nothing to audit
    if np.sum(our_weights) == 0:
        return warnings

    # Check 1: Near-uniform weights
    non_zero = our_weights[our_weights > 0]
    if len(non_zero) >= 2:
        std = float(np.std(non_zero))
        if std < 0.001:
            msg = "WARN: weights are near-uniform -- no miner discrimination"
            warnings.append(msg)
            bt.logging.warning(msg)

    # Check 2: Extreme concentration
    max_weight = float(our_weights.max())
    if max_weight > 0.5:
        msg = f"WARN: extreme weight concentration: max={max_weight:.4f}"
        warnings.append(msg)
        bt.logging.warning(msg)

    # Check 3: Cross-validator similarity (weight copying)
    for val_hotkey, val_weights in recent_validator_weights.items():
        # Skip if lengths differ
        if len(val_weights) != len(our_weights):
            continue

        norm_ours = np.linalg.norm(our_weights)
        norm_theirs = np.linalg.norm(val_weights)

        # Skip if either norm is 0
        if norm_ours == 0 or norm_theirs == 0:
            continue

        similarity = float(np.dot(our_weights, val_weights) / (norm_ours * norm_theirs))
        if similarity > similarity_threshold:
            display_key = val_hotkey[:12] if len(val_hotkey) > 12 else val_hotkey
            msg = (
                f"WARN: weight vector similarity={similarity:.4f} "
                f"with validator {display_key}..."
            )
            warnings.append(msg)
            bt.logging.warning(msg)

    return warnings


def check_commit_reveal_enabled(subtensor, netuid: int) -> bool:
    """Check if commit-reveal is enabled for this subnet.

    Informational only: the SDK v10.2.0 set_weights() method automatically
    routes to commit_timelocked_weights_extrinsic() when commit-reveal is
    enabled on the subnet. No custom implementation needed.

    Args:
        subtensor: Subtensor instance (real or mock).
        netuid: Network UID to check.

    Returns:
        True if commit-reveal is active, False otherwise.
        Returns False gracefully if the method doesn't exist (mock/test).
    """
    try:
        result = subtensor.commit_reveal_enabled(netuid=netuid)
        bt.logging.info(
            f"Commit-reveal status for netuid {netuid}: "
            f"{'enabled' if result else 'disabled'}"
        )
        return bool(result)
    except AttributeError:
        bt.logging.info(
            f"Commit-reveal check unavailable for netuid {netuid} "
            f"(subtensor lacks method). Assuming disabled."
        )
        return False
