"""
UID selection utilities for validators.

Provides functions to select random miner UIDs from the metagraph,
with support for exclusion lists.
"""

import random


def get_random_uids(
    validator, k: int, exclude: list[int] | None = None
) -> list[int]:
    """Get k random UIDs from metagraph, excluding specified UIDs.

    Excludes the validator's own UID automatically.

    Args:
        validator: The validator neuron instance with metagraph and uid.
        k: Number of UIDs to select.
        exclude: Optional list of UIDs to exclude.

    Returns:
        List of randomly selected UIDs.
    """
    available = list(range(validator.metagraph.n))

    # Exclude specified UIDs
    if exclude:
        available = [uid for uid in available if uid not in exclude]

    # Exclude own UID
    if validator.uid in available:
        available.remove(validator.uid)

    # Limit k to available count
    k = min(k, len(available))

    return random.sample(available, k) if available else []
