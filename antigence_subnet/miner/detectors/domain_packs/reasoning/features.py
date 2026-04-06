"""Chain-of-thought reasoning feature extraction.

Extracts structural signals from reasoning chains: step counts, logical
connective density, negation patterns, contradiction indicators,
premise/conclusion balance, step length, and total length. These features
detect flawed reasoning patterns like logical contradictions, non sequiturs,
and constraint violations.
"""

import re

# Step markers -- explicit numbered steps
_STEP_NUMBERED_RE = re.compile(r"(?:Step|step)\s+\d+", re.IGNORECASE)

# Ordinal/sequential markers
_STEP_ORDINALS = frozenset({
    "first", "second", "third", "fourth", "fifth",
    "next", "then", "finally", "therefore", "thus", "hence",
})

# Numbered list items (e.g., "1. " or "1) ")
_NUMBERED_LIST_RE = re.compile(r"^\d+[.)]\s", re.MULTILINE)

# Logical connectives for density measurement
_LOGICAL_CONNECTIVES = frozenset({
    "therefore", "thus", "hence", "because", "since",
    "if", "then", "implies", "consequently",
    "however", "but", "although", "nevertheless",
})

# "follows that" as a multi-word connective
_FOLLOWS_THAT_RE = re.compile(r"\bfollows\s+that\b", re.IGNORECASE)

# Negation words
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor",
    "cannot", "can't", "don't", "doesn't", "isn't",
    "won't", "shouldn't", "wouldn't", "hasn't",
})

_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _NEGATION_WORDS) + r")\b",
    re.IGNORECASE,
)

# Premise markers
_PREMISE_WORDS = frozenset({"given", "assuming", "suppose", "let", "consider"})

# Conclusion markers
_CONCLUSION_WORDS = frozenset({"therefore", "thus", "hence", "so", "conclude", "follows"})

# Sentence splitting
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|[.!?]+$")

# "X is Y" pattern for contradiction detection
_IS_CLAIM_RE = re.compile(
    r"\b(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(not\s+)?(\w+(?:\s+\w+)?)",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    if not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _count_steps(text: str) -> int:
    """Count reasoning steps using multiple marker patterns.

    Combines numbered steps (Step 1, Step 2), ordinal markers (First, Then,
    Finally), and numbered lists (1. , 2. ) with deduplication by position.
    """
    positions = set()

    # Numbered step markers
    for m in _STEP_NUMBERED_RE.finditer(text):
        positions.add(m.start())

    # Ordinal markers -- check word boundaries
    text_lower = text.lower()
    words = text_lower.split()
    for i, w in enumerate(words):
        # Clean punctuation from word for matching
        clean = w.strip(".,;:!?")
        if clean in _STEP_ORDINALS:
            # Approximate position by finding this word in text
            offset = sum(len(words[j]) + 1 for j in range(i)) - 1 if i > 0 else 0
            pos = text_lower.find(clean, offset)
            if pos >= 0:
                positions.add(pos)

    # Numbered list items
    for m in _NUMBERED_LIST_RE.finditer(text):
        positions.add(m.start())

    return len(positions)


def _compute_contradiction_score(sentences: list[str]) -> float:
    """Compute contradiction score by finding opposing "X is Y" / "X is not Y" pairs.

    Returns:
        Score from 0.0 to 1.0 representing fraction of sentences involved in contradictions.
    """
    if not sentences:
        return 0.0

    # Extract claims: (subject, is_negated, predicate)
    claims = []
    for sent in sentences:
        for m in _IS_CLAIM_RE.finditer(sent):
            subject = m.group(1).lower().strip()
            negated = bool(m.group(2))
            predicate = m.group(3).lower().strip()
            claims.append((subject, negated, predicate))

    # Find contradicting pairs
    contradiction_count = 0
    seen = set()
    for i, (subj_a, neg_a, pred_a) in enumerate(claims):
        for j, (subj_b, neg_b, pred_b) in enumerate(claims):
            if i >= j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            # Same subject and predicate, opposite negation
            if subj_a == subj_b and pred_a == pred_b and neg_a != neg_b:
                contradiction_count += 1
                seen.add(pair_key)

    total = len(sentences) if sentences else 1
    return min(float(contradiction_count) / total, 1.0)


def extract_reasoning_features(prompt: str, output: str) -> dict[str, float]:
    """Extract chain-of-thought reasoning features from prompt and output.

    Args:
        prompt: The original prompt text.
        output: The AI-generated reasoning output to analyze.

    Returns:
        Dict with 7 float features:
            - step_count: number of reasoning steps detected
            - logical_connective_density: connective words / total words
            - negation_count: number of negation words
            - contradiction_score: fraction of sentences with contradictions
            - premise_conclusion_ratio: premise markers / conclusion markers
            - avg_step_length: average words per step (or per sentence)
            - total_length: total character count
    """
    combined = f"{prompt} {output}".strip()

    # Handle empty input
    if not combined:
        return {
            "step_count": 0.0,
            "logical_connective_density": 0.0,
            "negation_count": 0.0,
            "contradiction_score": 0.0,
            "premise_conclusion_ratio": 1.0,
            "avg_step_length": 0.0,
            "total_length": 0.0,
        }

    words = combined.lower().split()
    word_count = len(words)
    sentences = _split_sentences(combined)

    # Step count
    step_count = _count_steps(combined)

    # Logical connective density
    connective_count = 0
    for w in words:
        clean = w.strip(".,;:!?()\"'")
        if clean in _LOGICAL_CONNECTIVES:
            connective_count += 1
    # Also count "follows that" as a connective
    connective_count += len(_FOLLOWS_THAT_RE.findall(combined))
    logical_connective_density = float(connective_count / word_count) if word_count > 0 else 0.0

    # Negation count
    negation_count = float(len(_NEGATION_RE.findall(combined)))

    # Contradiction score
    contradiction_score = _compute_contradiction_score(sentences)

    # Premise/conclusion ratio
    premise_count = 0
    conclusion_count = 0
    for w in words:
        clean = w.strip(".,;:!?()\"'")
        if clean in _PREMISE_WORDS:
            premise_count += 1
        if clean in _CONCLUSION_WORDS:
            conclusion_count += 1

    if conclusion_count > 0 and premise_count > 0:
        premise_conclusion_ratio = min(float(premise_count / conclusion_count), 5.0)
    elif premise_count > 0 and conclusion_count == 0:
        premise_conclusion_ratio = 5.0
    else:
        premise_conclusion_ratio = 1.0

    # Average step length
    if step_count > 0:
        avg_step_length = float(word_count / step_count)
    elif sentences:
        avg_step_length = float(word_count / len(sentences))
    else:
        avg_step_length = float(word_count)

    # Total length
    total_length = float(len(prompt) + len(output))

    return {
        "step_count": float(step_count),
        "logical_connective_density": logical_connective_density,
        "negation_count": negation_count,
        "contradiction_score": contradiction_score,
        "premise_conclusion_ratio": premise_conclusion_ratio,
        "avg_step_length": avg_step_length,
        "total_length": total_length,
    }
