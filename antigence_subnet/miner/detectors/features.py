"""Shared TF-IDF feature extraction for CPU/GPU parity.

Both IsolationForest and Autoencoder detectors use identical vectorizers
created via create_vectorizer() to ensure feature space consistency.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_STOP_WORDS = "english"


def create_vectorizer(
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with shared hyperparameters.

    Args:
        max_features: Maximum number of features to extract.
        ngram_range: N-gram range for token extraction.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=DEFAULT_STOP_WORDS,
        sublinear_tf=True,
    )


def samples_to_texts(samples: list[dict]) -> list[str]:
    """Convert sample dicts to text strings for vectorization.

    Concatenates prompt and output fields. Handles missing keys gracefully.

    Args:
        samples: List of sample dicts with optional 'prompt' and 'output' keys.

    Returns:
        List of text strings suitable for TF-IDF vectorization.
    """
    return [f"{s.get('prompt', '')} {s.get('output', '')}" for s in samples]
