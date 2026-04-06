"""
Weight utility functions for chain-compatible weight setting.

Ported from opentensor/bittensor-subnet-template with adaptations for
the Antigence subnet. Provides normalization, uint16 conversion, and
full pipeline processing with chain parameter constraints.

Key functions:
    - normalize_max_weight: Normalize scores so sum=1 and max <= limit
    - convert_weights_and_uids_for_emit: Convert float weights to uint16
    - process_weights_for_netuid: Full pipeline with chain constraints
"""

import bittensor as bt
import numpy as np

U32_MAX = 4294967295
U16_MAX = 65535


def normalize_max_weight(
    x: np.ndarray, limit: float = 0.1
) -> np.ndarray:
    """Normalize array so sum(x) = 1 and max value <= limit.

    Handles edge cases:
    - All zeros -> uniform distribution
    - Single element -> 1.0
    - Already within limit -> simple normalization

    Args:
        x: Array of weight values to normalize.
        limit: Maximum allowed value for any single weight.

    Returns:
        Normalized array summing to 1.0 with max <= limit.
    """
    epsilon = 1e-7  # For numerical stability after normalization

    weights = x.copy()
    values = np.sort(weights)

    if x.sum() == 0 or len(x) * limit <= 1:
        return np.ones_like(x) / x.size
    else:
        estimation = values / values.sum()

        if estimation.max() <= limit:
            return weights / weights.sum()

        # Find the cumulative sum and sorted array
        cumsum = np.cumsum(estimation, 0)

        # Determine the index of cutoff
        estimation_sum = np.array(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (
            estimation / (estimation_sum + cumsum + epsilon) < limit
        ).sum()

        # Determine the cutoff based on the index
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights / weights.sum()

        return y


def convert_weights_and_uids_for_emit(
    uids: np.ndarray, weights: np.ndarray
) -> tuple[list[int], list[int]]:
    """Convert float weights to uint16 representation for chain emission.

    Max-upscales weights so the largest maps to U16_MAX (65535),
    then filters zero-valued entries.

    Args:
        uids: Array of UIDs as weight destinations.
        weights: Array of float weights corresponding to UIDs.

    Returns:
        Tuple of (uid_list, weight_list) with zeros filtered out.
        Weight values are integers in [0, 65535].

    Raises:
        ValueError: If weights contain negative values, UIDs are negative,
            or arrays have mismatched lengths.
    """
    uids = np.asarray(uids)
    weights = np.asarray(weights)

    # Get non-zero weights and corresponding uids
    non_zero_weights = weights[weights > 0]
    non_zero_weight_uids = uids[weights > 0]

    bt.logging.debug(f"weights: {weights}")
    bt.logging.debug(f"non_zero_weights: {non_zero_weights}")
    bt.logging.debug(f"uids: {uids}")
    bt.logging.debug(f"non_zero_weight_uids: {non_zero_weight_uids}")

    if np.min(weights) < 0:
        raise ValueError(
            f"Passed weight is negative cannot exist on chain {weights}"
        )
    if np.min(uids) < 0:
        raise ValueError(
            f"Passed uid is negative cannot exist on chain {uids}"
        )
    if len(uids) != len(weights):
        raise ValueError(
            f"Passed weights and uids must have the same length, got {len(uids)} and {len(weights)}"
        )
    if np.sum(weights) == 0:
        bt.logging.debug("nothing to set on chain")
        return [], []  # Nothing to set on chain.
    else:
        max_weight = float(np.max(weights))
        weights = [
            float(value) / max_weight for value in weights
        ]  # max-upscale values (max_weight = 1).
        bt.logging.debug(
            f"setting on chain max: {max_weight} and weights: {weights}"
        )

    weight_vals = []
    weight_uids = []
    for _i, (weight_i, uid_i) in enumerate(list(zip(weights, uids, strict=False))):
        uint16_val = round(
            float(weight_i) * int(U16_MAX)
        )  # convert to int representation.

        # Filter zeros
        if uint16_val != 0:
            weight_vals.append(uint16_val)
            weight_uids.append(uid_i)

    bt.logging.debug(f"final params: {weight_uids} : {weight_vals}")
    return weight_uids, weight_vals


def process_weights_for_netuid(
    uids,
    weights: np.ndarray,
    netuid: int,
    subtensor,
    metagraph=None,
    exclude_quantile: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full weight processing pipeline with chain parameter constraints.

    Queries the subtensor chain for min_allowed_weights and max_weight_limit,
    applies quantile exclusion, and normalizes weights for emission.

    Args:
        uids: Array of UIDs.
        weights: Array of float weights.
        netuid: Network UID for chain parameter queries.
        subtensor: Subtensor instance for chain queries.
        metagraph: Optional metagraph (fetched from chain if None).
        exclude_quantile: Quantile value for excluding lowest weights.

    Returns:
        Tuple of (processed_uids, normalized_weights).
    """
    bt.logging.debug("process_weights_for_netuid()")
    bt.logging.debug(f"weights: {weights}")
    bt.logging.debug(f"netuid: {netuid}")

    # Get latest metagraph from chain if metagraph is None.
    if metagraph is None:
        metagraph = subtensor.metagraph(netuid)

    # Cast weights to floats.
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
        weights = weights.astype(np.float32)

    # Network configuration parameters from subtensor.
    quantile = exclude_quantile / U16_MAX
    min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid)
    max_weight_limit = subtensor.max_weight_limit(netuid=netuid)
    bt.logging.debug(f"quantile: {quantile}")
    bt.logging.debug(f"min_allowed_weights: {min_allowed_weights}")
    bt.logging.debug(f"max_weight_limit: {max_weight_limit}")

    # Find all non-zero weights.
    non_zero_weight_idx = np.argwhere(weights > 0).squeeze()
    non_zero_weight_idx = np.atleast_1d(non_zero_weight_idx)
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]

    if non_zero_weights.size == 0 or metagraph.n < min_allowed_weights:
        bt.logging.warning("No non-zero weights returning all ones.")
        final_weights = np.ones(metagraph.n) / metagraph.n
        bt.logging.debug(f"final_weights: {final_weights}")
        return np.arange(len(final_weights)), final_weights

    elif non_zero_weights.size < min_allowed_weights:
        bt.logging.warning(
            "No non-zero weights less than min allowed weight, returning all ones."
        )
        weights = (
            np.ones(metagraph.n) * 1e-5
        )  # creating minimum even non-zero weights
        weights[non_zero_weight_idx] += non_zero_weights
        bt.logging.debug(f"final_weights: {weights}")
        normalized_weights = normalize_max_weight(
            x=weights, limit=max_weight_limit
        )
        return np.arange(len(normalized_weights)), normalized_weights

    bt.logging.debug(f"non_zero_weights: {non_zero_weights}")

    # Compute the exclude quantile and find weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(
        non_zero_weights
    )
    exclude_quantile = min([quantile, max_exclude])
    lowest_quantile = np.quantile(non_zero_weights, exclude_quantile)
    bt.logging.debug(f"max_exclude: {max_exclude}")
    bt.logging.debug(f"exclude_quantile: {exclude_quantile}")
    bt.logging.debug(f"lowest_quantile: {lowest_quantile}")

    # Exclude all weights below the allowed quantile.
    non_zero_weight_uids = non_zero_weight_uids[
        lowest_quantile <= non_zero_weights
    ]
    non_zero_weights = non_zero_weights[lowest_quantile <= non_zero_weights]
    bt.logging.debug(f"non_zero_weight_uids: {non_zero_weight_uids}")
    bt.logging.debug(f"non_zero_weights: {non_zero_weights}")

    # Normalize weights and return.
    normalized_weights = normalize_max_weight(
        x=non_zero_weights, limit=max_weight_limit
    )
    bt.logging.debug(f"final_weights: {normalized_weights}")

    return non_zero_weight_uids, normalized_weights
