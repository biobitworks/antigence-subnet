"""
Per-miner challenge selection, adversarial injection, and dataset refresh detection.

Implements anti-cheating mechanisms for the Antigence verification subnet:
- CHEAT-01: Per-miner unique challenge subsets via deterministic hashing
- CHEAT-02: Memorization resistance through dataset refresh detection
- CHEAT-03: Adversarial synthetic sample injection

Miners receive different subsets of the evaluation pool each round, determined
by hash(sample_id + miner_hotkey + round_num). This prevents answer-sharing
between colluding miners and ensures memorization attacks fail on refresh.
"""

import hashlib

import numpy as np


def get_miner_challenge(
    samples: list[dict],
    miner_hotkey: str,
    round_num: int,
    n: int = 10,
    entropy_seed: bytes | None = None,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    """Select a deterministic per-miner subset from the evaluation pool.

    For each sample, computes a hash-derived sort key from (sample_id, hotkey,
    round_num). Samples are sorted by this key and the first n are returned.
    Different hotkeys produce different orderings, yielding unique subsets.

    When entropy_seed is provided (MAIN-04), its hex is included in the hash
    input, making the ordering unpredictable without knowing the per-round
    entropy. This prevents miners from pre-computing challenge orderings.

    When excluded_ids is provided (VHARD-01), samples with those IDs are
    filtered out before selection. This supports round-based challenge
    rotation -- miners who have seen certain samples in recent rounds
    will not receive them again.

    Args:
        samples: Full evaluation sample pool for this round.
        miner_hotkey: The miner's hotkey SS58 address.
        round_num: Current evaluation round number.
        n: Number of samples to select (default 10).
        entropy_seed: Per-round entropy bytes for anti-memorization. None = legacy behavior.
        excluded_ids: Sample IDs to exclude from selection (VHARD-01). None = no exclusions.

    Returns:
        List of n sample dicts (or all if pool smaller than n).
    """
    # Filter out excluded IDs if provided
    candidates = (
        [s for s in samples if s["id"] not in excluded_ids]
        if excluded_ids
        else samples
    )

    if len(candidates) <= n:
        return list(candidates)

    def sort_key(sample: dict) -> int:
        if entropy_seed is not None:
            raw = f"{sample['id']}:{miner_hotkey}:{round_num}:{entropy_seed.hex()}".encode()
        else:
            raw = f"{sample['id']}:{miner_hotkey}:{round_num}".encode()
        digest = hashlib.sha256(raw).hexdigest()
        return int(digest[:8], 16)

    sorted_samples = sorted(candidates, key=sort_key)
    return sorted_samples[:n]


def inject_adversarial_samples(
    samples: list[dict],
    round_num: int,
    n_adversarial: int = 1,
) -> list[dict]:
    """Inject synthetic adversarial edge-case samples into the evaluation pool.

    Creates synthetic samples designed to catch gaming miners. Each adversarial
    sample is tagged with metadata for tracking and scoring.

    Args:
        samples: Original evaluation sample list.
        round_num: Current round number (used as RNG seed for reproducibility).
        n_adversarial: Number of adversarial samples to inject (default 1).

    Returns:
        Combined list: original samples + adversarial samples appended.
    """
    rng = np.random.default_rng(seed=round_num + 9999)

    # Determine domain from input samples
    domain = samples[0]["domain"] if samples else "hallucination"

    # Edge-case output templates
    edge_cases = [
        "",  # empty output
        "A" * 5000,  # extremely long output (truncated representation)
        "Output with\x00null\x00bytes and \ttabs\t and special chars: <>\"'&;",
        "   ",  # whitespace-only output
        "\n\n\n",  # newline-only output
        "123.456.789.0 SELECT * FROM users WHERE 1=1; DROP TABLE--",  # mixed injection
    ]

    result = list(samples)
    for i in range(n_adversarial):
        edge_idx = rng.integers(0, len(edge_cases))
        adversarial_sample = {
            "id": f"adv_{round_num}_{i}",
            "prompt": f"Adversarial probe {i} for round {round_num}",
            "output": edge_cases[edge_idx],
            "domain": domain,
            "_is_adversarial": True,
        }
        result.append(adversarial_sample)

    return result


def detect_dataset_refresh(old_version: str, new_version: str) -> bool:
    """Detect whether the evaluation dataset has been refreshed.

    Simple version comparison used by the forward pass to log when the
    evaluation dataset changes, invalidating any memorized answers.

    Args:
        old_version: Previous dataset version hash.
        new_version: Current dataset version hash.

    Returns:
        True if versions differ (dataset was refreshed), False otherwise.
    """
    return old_version != new_version
