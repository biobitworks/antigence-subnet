"""Fractal complexity feature extraction for text signals.

Extracts 8 fractal metrics from text using signal complexity analysis:
- Higuchi Fractal Dimension (HFD) on character frequency and word length sequences
- Hurst Exponent (H) on word length and sentence length sequences
- Shannon Entropy on character, word, and bigram distributions
- Complexity Index combining HFD and Hurst

All algorithms are published: Higuchi (1988), Hurst (1951), Shannon (1948).
Combination approach adapted from Fractal Waves Project (Biobitworks).
"""

import collections
import re

import numpy as np

# Minimum data points required for each algorithm
_MIN_HFD_POINTS = 10
_MIN_HURST_POINTS = 20

# Default values when data is insufficient
_DEFAULT_HFD = 1.5
_DEFAULT_HURST = 0.5

# Feature vector defaults for empty/whitespace-only input
_DEFAULTS = np.array([1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])

# Sentence boundary pattern
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _higuchi_fd(signal: np.ndarray, kmax: int = 10) -> float:
    """Compute Higuchi Fractal Dimension of a 1D signal.

    Implements the algorithm from Higuchi (1988) "Approach to an irregular
    time series on the basis of the fractal theory".

    Args:
        signal: 1D numeric array (the time series).
        kmax: Maximum interval length for curve length computation.

    Returns:
        Fractal dimension clipped to [1.0, 2.0], or 1.5 (default) if
        signal is too short.
    """
    n = len(signal)
    if n < kmax or n < _MIN_HFD_POINTS:
        return _DEFAULT_HFD

    # Ensure float for arithmetic
    signal = signal.astype(np.float64)

    lk = np.zeros(kmax)
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(1, k + 1):
            # Number of segments for this start offset
            num_segments = (n - m) // k
            if num_segments < 1:
                continue
            # Compute curve length for start offset m, interval k
            diff_sum = np.sum(
                np.abs(signal[m + np.arange(1, num_segments + 1) * k - 1]
                       - signal[m + np.arange(0, num_segments) * k - 1])
            )
            # Normalize
            norm_factor = (n - 1) / (num_segments * k)
            lengths.append(diff_sum * norm_factor / k)

        if lengths:
            lk[k - 1] = np.mean(lengths)
        else:
            lk[k - 1] = 0.0

    # Filter out zero or negative values for log fitting
    valid = lk > 0
    if np.sum(valid) < 2:
        return _DEFAULT_HFD

    ks = np.arange(1, kmax + 1)
    log_lk = np.log(lk[valid])
    log_inv_k = np.log(1.0 / ks[valid])

    # Linear fit: log(L(k)) vs log(1/k), slope is the fractal dimension
    coeffs = np.polyfit(log_inv_k, log_lk, 1)
    return float(np.clip(coeffs[0], 1.0, 2.0))


def _hurst_exponent(signal: np.ndarray) -> float:
    """Compute Hurst Exponent of a 1D signal via R/S analysis.

    Uses nolds.hurst_rs() if available, otherwise returns default 0.5.
    Based on Hurst (1951) "Long-term storage capacity of reservoirs".

    Args:
        signal: 1D numeric array (the time series).

    Returns:
        Hurst exponent clipped to [0.0, 1.0], or 0.5 (default) if
        signal is too short or nolds is unavailable.
    """
    if len(signal) < _MIN_HURST_POINTS:
        return _DEFAULT_HURST

    try:
        import nolds  # noqa: F811

        h = nolds.hurst_rs(signal)
        if not np.isfinite(h):
            return _DEFAULT_HURST
        return float(np.clip(h, 0.0, 1.0))
    except (ImportError, ValueError, RuntimeError):
        return _DEFAULT_HURST


def _shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy from frequency counts.

    Based on Shannon (1948) "A Mathematical Theory of Communication".

    Args:
        counts: Array of frequency counts (non-negative integers).

    Returns:
        Shannon entropy in bits (log base 2). Returns 0.0 for empty input.
    """
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_fractal_features(text: str) -> np.ndarray:
    """Extract 8-element fractal feature vector from text.

    Feature indices:
        [0] hfd_char_dist      -- Higuchi FD of character frequency sequence
        [1] hfd_word_lengths    -- Higuchi FD of word length sequence
        [2] hurst_word_lengths  -- Hurst exponent of word length sequence
        [3] hurst_sentence_lengths -- Hurst exponent of sentence length sequence
        [4] shannon_char        -- Shannon entropy of character distribution
        [5] shannon_word        -- Shannon entropy of word frequency distribution
        [6] shannon_bigram      -- Shannon entropy of bigram distribution
        [7] complexity_index    -- (HFD - 1) * (1 - Hurst + 0.5) combined metric

    Args:
        text: Input text string to analyze.

    Returns:
        np.ndarray of shape (8,) with float64 values. All values are finite.
    """
    if not text or not text.strip():
        return _DEFAULTS.copy()

    # -- Build signal arrays from text --

    # Character frequency sequence (sorted unique characters)
    char_freq_sequence = np.array(
        [text.count(c) for c in sorted(set(text))], dtype=np.float64
    )

    # Word lengths
    words = text.split()
    word_lengths = np.array([len(w) for w in words], dtype=np.float64)

    # Sentence lengths (split on .!?, filter empty)
    sentence_parts = _SENTENCE_SPLIT_RE.split(text)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]
    if sentence_parts:
        sentence_lengths = np.array(
            [len(s.split()) for s in sentence_parts], dtype=np.float64
        )
    else:
        # No sentence boundaries found -- treat whole text as one sentence
        sentence_lengths = np.array([len(words)], dtype=np.float64)

    # -- Shannon entropy inputs (frequency counts) --

    # Character frequency counts
    char_counter = collections.Counter(text)
    char_counts = np.array(list(char_counter.values()), dtype=np.float64)

    # Word frequency counts
    word_counter = collections.Counter(words)
    word_counts = np.array(list(word_counter.values()), dtype=np.float64)

    # Bigram frequency counts
    if len(words) >= 2:
        bigram_counter = collections.Counter(zip(words, words[1:], strict=False))
        bigram_counts = np.array(list(bigram_counter.values()), dtype=np.float64)
    else:
        bigram_counts = np.array([], dtype=np.float64)

    # -- Compute features --

    hfd_char_dist = _higuchi_fd(char_freq_sequence)
    hfd_word_lengths = _higuchi_fd(word_lengths)
    hurst_word_lengths = _hurst_exponent(word_lengths)
    hurst_sentence_lengths = _hurst_exponent(sentence_lengths)
    shannon_char = _shannon_entropy(char_counts)
    shannon_word = _shannon_entropy(word_counts)
    shannon_bigram = _shannon_entropy(bigram_counts)
    complexity_index = (hfd_word_lengths - 1.0) * (1.0 - hurst_word_lengths + 0.5)

    return np.array(
        [
            hfd_char_dist,
            hfd_word_lengths,
            hurst_word_lengths,
            hurst_sentence_lengths,
            shannon_char,
            shannon_word,
            shannon_bigram,
            complexity_index,
        ],
        dtype=np.float64,
    )
