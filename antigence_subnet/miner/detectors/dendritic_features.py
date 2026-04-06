"""Rule-based 10-dimensional feature extraction for NegSel-AIS detection.

Extracts semantic features from raw text using regex patterns and word set
matching. No model dependency -- uses only ``re`` (stdlib) and ``numpy``.

Ported from the parent Antigence platform's DendriticAgent with feature
names aligned to the subnet CONTEXT.md specification.
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# Feature name contract (position -> CONTEXT.md name -> parent code name)
#   [0] claim_density    = source_credibility
#   [1] citation_count   = has_citation
#   [2] hedging_ratio    = has_hedging
#   [3] specificity      = specificity_score
#   [4] numeric_density  = has_numbers
#   [5] pamp_score       = pamp_score
#   [6] exaggeration     = exaggeration_score
#   [7] certainty        = has_certainty
#   [8] controversy      = controversy_score
#   [9] danger_signal    = danger_signal_count / 5.0
# ---------------------------------------------------------------------------


class DendriticFeatureExtractor:
    """Rule-based 10-dimensional feature vector from text.

    No model dependency -- uses regex patterns and word set matching.
    Features normalized to [0, 1] per-feature.

    Feature mapping (position -> CONTEXT.md name -> parent code name):
      [0] claim_density    = source_credibility (heuristic: base 0.5,
          +0.2 citation, +0.1 hedging, -0.2 exaggeration>0.5,
          -0.3 pamp>0.5, clamp [0,1])
      [1] citation_count   = has_citation (1.0 if any citation pattern
          matches, else 0.0)
      [2] hedging_ratio    = has_hedging (1.0 if any hedging word found,
          else 0.0)
      [3] specificity      = specificity_score (capitalized word ratio * 5,
          clamp [0,1])
      [4] numeric_density  = has_numbers (1.0 if quantitative pattern
          matches, else 0.0)
      [5] pamp_score       = pamp_score (danger pattern matches / 3.0,
          clamp [0,1])
      [6] exaggeration     = exaggeration_score (exaggeration word count
          / 3.0, clamp [0,1])
      [7] certainty        = has_certainty (1.0 if any certainty word
          found, else 0.0)
      [8] controversy      = controversy_score (controversy word count
          / 2.0, clamp [0,1])
      [9] danger_signal    = danger_signal_count / 5.0 (normalized danger
          count, clamp [0,1])
    """

    FEATURE_NAMES = [
        "claim_density",
        "citation_count",
        "hedging_ratio",
        "specificity",
        "numeric_density",
        "pamp_score",
        "exaggeration",
        "certainty",
        "controversy",
        "danger_signal",
    ]

    def __init__(self) -> None:
        """Initialize word sets and compile regex patterns."""
        # Word sets -- ported exactly from parent dendritic_agent.py
        self._hedging_words: set[str] = {
            "may", "might", "possibly", "perhaps", "could", "likely",
            "suggest", "appears", "seems", "indicate", "potentially",
        }
        self._certainty_words: set[str] = {
            "always", "never", "proven", "definitely", "certainly",
            "undoubtedly", "clearly", "obviously", "absolutely", "must",
        }
        self._exaggeration_words: set[str] = {
            "revolutionary", "breakthrough", "unprecedented", "miraculous",
            "amazing", "incredible", "extraordinary", "groundbreaking",
        }
        self._controversy_words: set[str] = {
            "controversial", "debate", "disputed", "disagree", "conflict",
        }

        # Citation patterns -- ported exactly from parent
        self._citation_patterns: list[re.Pattern[str]] = [
            re.compile(r"\b10\.\d{4,}/\S+"),                     # DOI
            re.compile(r"PMID:\s*\d+"),                          # PubMed ID
            re.compile(r"\[\d+\]"),                              # Numbered citation
            re.compile(r"\(\w+\s+et\s+al\.,?\s+\d{4}\)"),       # Author et al. (year)
            re.compile(r"\(\w+,?\s+\d{4}\)"),                   # (Author, year)
        ]

        # Danger / PAMP patterns -- ported exactly from parent
        self._danger_patterns: list[re.Pattern[str]] = [
            re.compile(r"\bcure[sd]?\b", re.I),
            re.compile(r"\b100\s*%\b"),
            re.compile(r"\bguarantee[sd]?\b", re.I),
            re.compile(r"\bproven\s+to\b", re.I),
            re.compile(r"\bno\s+side\s+effects?\b", re.I),
        ]

        # Quantitative numbers pattern
        self._numbers_pattern: re.Pattern[str] = re.compile(
            r"\b\d+\.?\d*\s*%|\b\d+\.?\d*\s*"
            r"(mg|kg|ml|mmol|patients?|subjects?|years?)\b",
            re.I,
        )

    def extract(self, text: str) -> np.ndarray:
        """Extract a 10-element feature vector from *text*.

        Args:
            text: Raw text to extract features from.

        Returns:
            numpy array of shape ``(10,)`` with all values in [0.0, 1.0].
        """
        text_lower = text.lower()
        words = text_lower.split()
        word_set = set(words)

        # --- Individual features (computed first so claim_density can use them) ---

        # [1] citation_count (binary)
        has_citation = 1.0 if any(p.search(text) for p in self._citation_patterns) else 0.0

        # [2] hedging_ratio (binary)
        has_hedging = 1.0 if (word_set & self._hedging_words) else 0.0

        # [3] specificity (capitalized word ratio * 5, clamp [0,1])
        all_words = text.split()
        capitalized = [w for w in all_words if w and w[0].isupper() and len(w) > 1]
        specificity = min(len(capitalized) / max(len(words), 1) * 5, 1.0)

        # [4] numeric_density (binary)
        has_numbers = 1.0 if self._numbers_pattern.search(text_lower) else 0.0

        # [5] pamp_score (danger pattern matches / 3.0, clamp [0,1])
        pamp_matches = sum(1 for p in self._danger_patterns if p.search(text))
        pamp_score = min(pamp_matches / 3.0, 1.0)

        # [6] exaggeration (exaggeration word count / 3.0, clamp [0,1])
        exaggeration_count = len(word_set & self._exaggeration_words)
        exaggeration = min(exaggeration_count / 3.0, 1.0)

        # [7] certainty (binary)
        has_certainty = 1.0 if (word_set & self._certainty_words) else 0.0

        # [8] controversy (controversy word count / 2.0, clamp [0,1])
        controversy_count = len(word_set & self._controversy_words)
        controversy = min(controversy_count / 2.0, 1.0)

        # [9] danger_signal (danger count / 5.0, clamp [0,1])
        danger_signal = min(pamp_matches / 5.0, 1.0)

        # [0] claim_density = source_credibility heuristic
        #     Depends on citation, hedging, exaggeration, pamp computed above
        credibility = 0.5
        if has_citation:
            credibility += 0.2
        if has_hedging:
            credibility += 0.1
        if exaggeration > 0.5:
            credibility -= 0.2
        if pamp_score > 0.5:
            credibility -= 0.3
        claim_density = max(0.0, min(1.0, credibility))

        vector = [
            claim_density,    # 0
            has_citation,     # 1
            has_hedging,      # 2
            specificity,      # 3
            has_numbers,      # 4
            pamp_score,       # 5
            exaggeration,     # 6
            has_certainty,    # 7
            controversy,      # 8
            danger_signal,    # 9
        ]

        return np.array(vector, dtype=np.float64)

    def extract_with_names(self, text: str) -> dict[str, float]:
        """Extract features and return as a named dictionary.

        Args:
            text: Raw text to extract features from.

        Returns:
            Dict mapping each feature name to its value.
        """
        vector = self.extract(text)
        return dict(zip(self.FEATURE_NAMES, vector.tolist(), strict=True))

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        """Extract features for a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            numpy array of shape ``(len(texts), 10)``.
        """
        return np.array([self.extract(t) for t in texts])
