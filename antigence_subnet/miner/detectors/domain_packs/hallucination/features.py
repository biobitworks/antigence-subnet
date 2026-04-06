"""Hallucination-specific feature extraction.

Extracts domain-specific signals beyond generic TF-IDF: claim density,
citation patterns, hedging language, numeric density, and text structure.
These features improve anomaly detection for hallucinated content.
"""

import re

# Citation patterns
# Inline style: (Smith, 2020) or (Smith et al., 2020) or (Smith and Jones, 2020)
_PAREN_CITATION_RE = re.compile(
    r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)*,?\s*\d{4}\)"
)
# Narrative style: Smith (2020) or Smith et al. (2020) or Smith and Jones (2020)
_NARRATIVE_CITATION_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][a-z]+)?\s+\(\d{4}\)"
)
_BRACKET_CITATION_RE = re.compile(r"\[\d+\]")

# Hedging words that signal uncertainty
_HEDGING_WORDS = frozenset(
    {"may", "might", "possibly", "suggests", "appears", "likely", "approximately"}
)

# Numeric value pattern
_NUMERIC_RE = re.compile(r"\b\d+\.?\d*\b")

# Sentence splitting pattern
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|[.!?]+$")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses punctuation-based splitting ('. ', '! ', '? ') and
    filters out empty strings.
    """
    if not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def extract_hallucination_features(prompt: str, output: str) -> dict[str, float]:
    """Extract hallucination-specific features from prompt and output.

    Args:
        prompt: The original prompt text.
        output: The AI-generated output to analyze.

    Returns:
        Dict with 7 float features:
            - claim_density: declarative sentences / total sentences
            - citation_count: number of citation patterns found
            - citation_pattern_score: quality score of citation formatting
            - hedging_ratio: fraction of sentences with hedging words
            - numeric_claim_density: numeric values / total word count
            - avg_sentence_length: average words per sentence
            - text_length: total character count of prompt + output
    """
    combined = f"{prompt} {output}".strip()

    # Handle empty input
    if not combined:
        return {
            "claim_density": 0.0,
            "citation_count": 0.0,
            "citation_pattern_score": 0.0,
            "hedging_ratio": 0.0,
            "numeric_claim_density": 0.0,
            "avg_sentence_length": 0.0,
            "text_length": 0.0,
        }

    sentences = _split_sentences(combined)
    words = combined.split()
    word_count = len(words)
    sentence_count = len(sentences) if sentences else 1

    # Citation count: parenthetical (Smith, 2020), narrative Smith (2020), and bracket [1] styles
    paren_citations = _PAREN_CITATION_RE.findall(combined)
    narrative_citations = _NARRATIVE_CITATION_RE.findall(combined)
    bracket_citations = _BRACKET_CITATION_RE.findall(combined)
    citation_count = float(len(paren_citations) + len(narrative_citations) + len(bracket_citations))

    # Citation pattern score: quality of citation formatting
    # 1.0 if all citations match standard academic patterns, 0.0 if none
    if citation_count > 0:
        # Well-formed = parenthetical or narrative (not bracket-only)
        well_formed = len(paren_citations) + len(narrative_citations)
        citation_pattern_score = float(well_formed / citation_count)
    else:
        citation_pattern_score = 0.0

    # Claim density: fraction of sentences that are declarative
    # (Sentences NOT starting with question words and NOT ending with '?')
    question_words = {
        "who", "what", "when", "where", "why", "how", "which",
        "is", "are", "do", "does", "can", "could",
    }
    declarative_count = 0
    for sent in sentences:
        sent_lower = sent.lower().strip()
        first_word = sent_lower.split()[0] if sent_lower.split() else ""
        if first_word not in question_words and not sent_lower.endswith("?"):
            declarative_count += 1
    claim_density = float(declarative_count / sentence_count)

    # Hedging ratio: fraction of sentences containing hedging words
    hedging_count = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        if sent_words & _HEDGING_WORDS:
            hedging_count += 1
    hedging_ratio = float(hedging_count / sentence_count)

    # Numeric claim density: count of numeric values / word count
    numeric_values = _NUMERIC_RE.findall(combined)
    numeric_claim_density = float(len(numeric_values) / word_count) if word_count > 0 else 0.0

    # Average sentence length in words
    avg_sentence_length = float(word_count / sentence_count)

    # Text length: character count of prompt + output (not combined with space)
    text_length = float(len(prompt) + len(output))

    return {
        "claim_density": claim_density,
        "citation_count": citation_count,
        "citation_pattern_score": citation_pattern_score,
        "hedging_ratio": hedging_ratio,
        "numeric_claim_density": numeric_claim_density,
        "avg_sentence_length": avg_sentence_length,
        "text_length": text_length,
    }
