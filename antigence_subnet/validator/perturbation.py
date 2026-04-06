"""
Text perturbation engine and stability bonus computation.

Implements CHEAT-04: perturbation stability testing for the Antigence
verification subnet. Generates deterministic text variants via synonym
substitution, whitespace changes, casing changes, insertion, deletion,
and clause reordering. Miners that return consistent scores across
perturbed variants of the same input receive a stability bonus;
inconsistent miners are penalized.

MAIN-04: Composable mutations with per-round entropy eliminate the
memorization attack vector. When entropy_seed is provided, 1-3 mutation
types are composed per sample, making perturbations non-deterministic
across rounds while preserving within-round determinism for audit replay.

Stability bonus formula: 1 - std(scores), clamped to [0, 1]
"""

import hashlib
import random

import numpy as np

# Conservative synonym mappings that should not change anomaly classification
# Expanded to 24+ entries (MAIN-04) to increase the synonym space beyond
# what a memorizing miner can feasibly enumerate.
SYNONYM_MAP: dict[str, list[str]] = {
    # Original 8 entries
    "the": ["a", "this"],
    "is": ["represents", "constitutes"],
    "was": ["had been", "used to be"],
    "are": ["represent", "constitute"],
    "has": ["possesses", "contains"],
    "have": ["possess", "contain"],
    "will": ["shall", "is going to"],
    "can": ["is able to", "may"],
    # Expanded entries (MAIN-04)
    "important": ["significant", "crucial", "vital"],
    "large": ["big", "substantial", "considerable"],
    "small": ["tiny", "minor", "slight"],
    "good": ["excellent", "fine", "satisfactory"],
    "bad": ["poor", "inadequate", "unsatisfactory"],
    "show": ["demonstrate", "indicate", "reveal"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "produce", "generate"],
    "get": ["obtain", "acquire", "retrieve"],
    "give": ["provide", "supply", "offer"],
    "find": ["discover", "locate", "identify"],
    "know": ["understand", "recognize", "realize"],
    "think": ["believe", "consider", "suppose"],
    "see": ["observe", "notice", "perceive"],
    "come": ["arrive", "approach", "emerge"],
    "go": ["proceed", "advance", "move"],
}

# Filler words for insertion mutation (MAIN-04)
FILLER_WORDS: list[str] = [
    "very", "quite", "rather", "indeed", "actually",
    "really", "certainly", "essentially", "basically", "generally",
]

# Words eligible for deletion mutation (MAIN-04)
DELETABLE_WORDS: set[str] = {
    "very", "quite", "rather", "really", "actually",
    "just", "simply", "basically", "essentially", "certainly",
    "truly", "merely",
}

# All available mutation functions, indexed for composable selection
_MUTATION_TYPES = [
    "_synonym_mutation",
    "_whitespace_mutation",
    "_casing_mutation",
    "_insertion_mutation",
    "_deletion_mutation",
    "_reorder_mutation",
]


def perturb_text(text: str, seed: int = 42, entropy_seed: int | None = None) -> str:
    """Apply a deterministic text perturbation.

    Without entropy_seed (backward compat): selects one of three legacy
    mutation types based on seed % 3.

    With entropy_seed (MAIN-04 composable mode): creates an RNG from
    seed XOR entropy_seed, selects 1-3 mutation types from all 6
    available, and applies them in sequence.

    Args:
        text: Input text to perturb.
        seed: Random seed for deterministic mutation selection.
        entropy_seed: Per-round entropy for anti-memorization. None = legacy behavior.

    Returns:
        Perturbed text string.
    """
    if entropy_seed is None:
        # Legacy behavior: single mutation type based on seed % 3
        rng = random.Random(seed)
        mutation_type = seed % 3

        if mutation_type == 0:
            return _synonym_mutation(text, rng)
        elif mutation_type == 1:
            return _whitespace_mutation(text, rng)
        else:
            return _casing_mutation(text, rng)

    # Composable mode (MAIN-04): 1-3 mutations from all 6 types
    combined_seed = seed ^ entropy_seed
    rng = random.Random(combined_seed)
    n_mutations = (combined_seed % 3) + 1

    # Select n_mutations distinct mutation types
    mutation_indices = rng.sample(range(len(_MUTATION_TYPES)), k=n_mutations)

    # Dispatch table for mutation functions
    dispatch = {
        0: _synonym_mutation,
        1: _whitespace_mutation,
        2: _casing_mutation,
        3: _insertion_mutation,
        4: _deletion_mutation,
        5: _reorder_mutation,
    }

    result = text
    for idx in mutation_indices:
        result = dispatch[idx](result, rng)
    return result


def _synonym_mutation(text: str, rng: random.Random) -> str:
    """Replace eligible words with synonyms (30% chance per word).

    Guarantees at least one substitution if any eligible word exists,
    to ensure the mutation always produces a visible change.
    """
    words = text.split()
    result = []
    changed = False
    first_eligible_idx = None

    for idx, word in enumerate(words):
        # Strip punctuation for lookup, preserve it for reconstruction
        stripped = word.strip(".,!?;:\"'()[]")
        prefix = word[: len(word) - len(word.lstrip(".,!?;:\"'()[]"))]
        rstripped = word.rstrip(".,!?;:\"'()[]")
        suffix = word[len(rstripped):] if rstripped != word else ""

        lower = stripped.lower()
        if lower in SYNONYM_MAP:
            if first_eligible_idx is None:
                first_eligible_idx = idx
            if rng.random() < 0.3:
                synonym = rng.choice(SYNONYM_MAP[lower])
                # Preserve original casing if first char was uppercase
                if stripped and stripped[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                result.append(prefix + synonym + suffix)
                changed = True
                continue
        result.append(word)

    # Guarantee at least one change if eligible words exist
    if not changed and first_eligible_idx is not None:
        word = words[first_eligible_idx]
        stripped = word.strip(".,!?;:\"'()[]")
        prefix = word[: len(word) - len(word.lstrip(".,!?;:\"'()[]"))]
        rstripped2 = word.rstrip(".,!?;:\"'()[]")
        suffix = word[len(rstripped2):] if rstripped2 != word else ""
        lower = stripped.lower()
        synonym = rng.choice(SYNONYM_MAP[lower])
        if stripped and stripped[0].isupper():
            synonym = synonym[0].upper() + synonym[1:]
        result[first_eligible_idx] = prefix + synonym + suffix

    return " ".join(result)


def _whitespace_mutation(text: str, rng: random.Random) -> str:
    """Add or remove spaces around punctuation marks.

    Guarantees at least one whitespace change. If no punctuation-based change
    occurs, doubles a random interior space.
    """
    result = list(text)
    punctuation = ".,!?;:"
    insertions = []

    for i, char in enumerate(result):
        if char in punctuation and rng.random() < 0.5:
            # Add a space before punctuation (more reliably visible)
            if i > 0 and result[i - 1] != " ":
                insertions.append((i, " "))
            elif i < len(result) - 1 and result[i + 1] != " ":
                insertions.append((i + 1, " "))

    if insertions:
        # Apply insertions in reverse order to preserve indices
        for pos, char in reversed(insertions):
            result.insert(pos, char)
    else:
        # Fallback: double a random interior space to guarantee a visible change
        space_indices = [i for i, c in enumerate(result) if c == " "]
        if space_indices:
            idx = rng.choice(space_indices)
            result.insert(idx, " ")

    return "".join(result)


def _casing_mutation(text: str, rng: random.Random) -> str:
    """Uppercase random words (15% chance per word)."""
    words = text.split()
    result = []
    for word in words:
        if rng.random() < 0.15:
            result.append(word.upper())
        else:
            result.append(word)
    return " ".join(result)


def _insertion_mutation(text: str, rng: random.Random) -> str:
    """Insert a random filler word before words longer than 3 characters.

    10% chance per eligible word. Guarantees at least one insertion.
    """
    words = text.split()
    result = []
    inserted = False
    first_eligible_idx = None

    for _idx, word in enumerate(words):
        # Eligible if word is longer than 3 characters (likely adjective/adverb/noun)
        stripped = word.strip(".,!?;:\"'()[]")
        if len(stripped) > 3:
            if first_eligible_idx is None:
                first_eligible_idx = len(result)
            if rng.random() < 0.1:
                filler = rng.choice(FILLER_WORDS)
                result.append(filler)
                inserted = True
        result.append(word)

    # Guarantee at least one insertion
    if not inserted and first_eligible_idx is not None:
        filler = rng.choice(FILLER_WORDS)
        result.insert(first_eligible_idx, filler)

    return " ".join(result)


def _deletion_mutation(text: str, rng: random.Random) -> str:
    """Remove words that are in DELETABLE_WORDS (50% chance per eligible word).

    Guarantees at least one deletion if an eligible word exists.
    If no eligible word exists, returns text unchanged.
    """
    words = text.split()
    eligible_indices = [
        i for i, w in enumerate(words) if w.lower().strip(".,!?;:\"'()[]") in DELETABLE_WORDS
    ]

    if not eligible_indices:
        return text

    # Mark words for deletion (50% chance each)
    to_delete = set()
    for idx in eligible_indices:
        if rng.random() < 0.5:
            to_delete.add(idx)

    # Guarantee at least one deletion
    if not to_delete:
        to_delete.add(rng.choice(eligible_indices))

    result = [w for i, w in enumerate(words) if i not in to_delete]
    return " ".join(result)


def _reorder_mutation(text: str, rng: random.Random) -> str:
    """Swap clauses joined by 'and' or 'or'.

    If no conjunction found, reverses word order of a random sentence
    (split by period). Guarantees a visible change.
    """
    # Try to find and swap clauses around "and" or "or"
    for conjunction in ["and", "or"]:
        if f" {conjunction} " in text:
            parts = text.split(f" {conjunction} ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return f"{parts[1].strip()} {conjunction} {parts[0].strip()}"

    # Fallback: reverse word order of a random sentence
    sentences = text.split(".")
    non_empty = [(i, s) for i, s in enumerate(sentences) if s.strip()]
    if non_empty:
        idx, sentence = rng.choice(non_empty)
        words = sentence.strip().split()
        words.reverse()
        sentences[idx] = " ".join(words)
        return ".".join(sentences)

    # Final fallback: reverse all words
    words = text.split()
    words.reverse()
    return " ".join(words)


def _hash_int(s: str) -> int:
    """Compute a stable integer hash from a string."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def generate_perturbation_variants(
    sample: dict,
    round_num: int,
    n_variants: int = 2,
    entropy_seed: int | None = None,
) -> list[dict]:
    """Generate perturbed copies of a sample for stability testing.

    Creates variant copies with perturbed "output" fields. Only the output
    is modified -- prompt, domain, and other fields are preserved. Each variant
    is tagged with metadata for tracking.

    When entropy_seed is provided (MAIN-04), the actual number of variants
    is randomized to range [1, n_variants] based on entropy. This prevents
    miners from predicting how many variants they will receive.

    Args:
        sample: Original evaluation sample dict.
        round_num: Current round number (affects perturbation seed).
        n_variants: Maximum number of variants to generate (default 2).
        entropy_seed: Per-round entropy for anti-memorization. None = legacy behavior.

    Returns:
        List of perturbed sample dicts (original not included).
    """
    actual_n = (
        min((entropy_seed % 3) + 1, n_variants) if entropy_seed is not None else n_variants
    )

    variants = []
    for i in range(actual_n):
        variant = dict(sample)
        seed = _hash_int(f"{sample['id']}{round_num}{i}")
        variant["output"] = perturb_text(
            sample["output"], seed=seed, entropy_seed=entropy_seed,
        )
        variant["_is_perturbation"] = True
        variant["_original_id"] = sample["id"]
        variant["id"] = f"{sample['id']}_perturb_{i}"
        variants.append(variant)
    return variants


def compute_stability_bonus(scores: list[float]) -> float:
    """Compute stability bonus from anomaly scores across perturbation variants.

    Rewards miners that produce consistent scores for equivalent inputs.
    Formula: 1 - std(scores), clamped to [0, 1].

    Args:
        scores: List of anomaly scores from original + perturbed variants.

    Returns:
        Stability bonus in [0.0, 1.0]. Higher = more consistent.
    """
    if len(scores) < 2:
        return 1.0
    return float(max(0.0, 1.0 - np.std(scores)))
