"""Bio pipelines feature extraction.

Extracts domain-specific signals for computational biology outputs:
numeric value ranges, out-of-range detection for bio quantities,
z-score outliers, negative values, unit mentions, magnitude range,
and statistical summary terms. These features improve anomaly detection
for bio pipeline outputs with incorrect values or inconsistent units.
"""

import math
import re

# Numeric value pattern (integers, decimals, scientific notation)
_NUMERIC_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")

# Bio-relevant unit patterns (case-sensitive where appropriate)
_UNIT_PATTERNS = re.compile(
    r"\b(?:mM|uM|nM|pM|ng/mL|ug/mL|mg/mL|kDa|bp|kb|Mb|rpm|CFU|OD|pH|Celsius|Kelvin|mol/L|g/L)\b"
)

# Statistical summary terms (case-insensitive matching done separately)
_STAT_TERMS = [
    r"\bmean\b",
    r"\bmedian\b",
    r"\bstd\b",
    r"\bstandard deviation\b",
    r"\bvariance\b",
    r"\bCI\b",
    r"\bconfidence interval\b",
    r"\bp-value\b",
    r"\bp =",
    r"\br =",
    r"\bR\^2\b",
    r"\bR-squared\b",
    r"\bANOVA\b",
    r"\bt-test\b",
    r"\bchi-square\b",
]

# Range check keywords and their associated ranges
# Each entry: (keyword_pattern, min_value, max_value)
_RANGE_CHECKS = [
    (re.compile(r"\bpH\b", re.IGNORECASE), 0.0, 14.0),
    (re.compile(r"\b(?:temperature|temp)\b", re.IGNORECASE), -20.0, 200.0),
    (re.compile(r"\b(?:concentration|conc)\b", re.IGNORECASE), 0.0, float("inf")),
    (re.compile(r"\b(?:expression|fold change)\b", re.IGNORECASE), 0.0, float("inf")),
    (re.compile(r"\bp[- ]?value\b", re.IGNORECASE), 0.0, 1.0),
    (re.compile(r"\b(?:percentage|percent)\b|%", re.IGNORECASE), 0.0, 100.0),
]


def _find_numeric_values_with_positions(text: str) -> list[tuple[float, int]]:
    """Find all numeric values in text and return (value, start_position) pairs."""
    results = []
    for match in _NUMERIC_RE.finditer(text):
        try:
            val = float(match.group())
            results.append((val, match.start()))
        except ValueError:
            continue
    return results


def _count_out_of_range(text: str, values_with_pos: list[tuple[float, int]]) -> int:
    """Count numeric values near bio-domain keywords that fall outside expected ranges.

    For each numeric value, checks if any range keyword appears within 50 chars
    before it. If so, validates the value against that keyword's expected range.
    """
    violations = 0
    for val, pos in values_with_pos:
        # Look at the 50 characters before this number
        context_start = max(0, pos - 50)
        context = text[context_start:pos]
        for keyword_re, min_val, max_val in _RANGE_CHECKS:
            if keyword_re.search(context) and (val < min_val or val > max_val):
                    violations += 1
                    break  # One violation per numeric value
    return violations


def _count_z_score_outliers(values: list[float], threshold: float = 3.0) -> int:
    """Count values with |z-score| > threshold. Requires >= 3 values."""
    if len(values) < 3:
        return 0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance < 1e-10:
        return 0
    std = math.sqrt(variance)
    return sum(1 for v in values if abs((v - mean) / std) > threshold)


def extract_bio_features(prompt: str, output: str) -> dict[str, float]:
    """Extract bio-pipeline-specific features from prompt and output.

    Args:
        prompt: The original prompt text.
        output: The bio pipeline output to analyze.

    Returns:
        Dict with 7 float features:
            - numeric_value_count: count of all numeric values
            - out_of_range_count: values near bio keywords outside expected ranges
            - z_score_outlier_count: values with |z| > 3.0
            - negative_value_count: count of negative numeric values
            - unit_mention_count: count of bio-relevant unit mentions
            - value_magnitude_range: log10(max/min) of absolute values
            - statistical_summary_count: count of statistical terms
    """
    combined = f"{prompt} {output}".strip()

    # Handle empty input
    if not combined:
        return {
            "numeric_value_count": 0.0,
            "out_of_range_count": 0.0,
            "z_score_outlier_count": 0.0,
            "negative_value_count": 0.0,
            "unit_mention_count": 0.0,
            "value_magnitude_range": 0.0,
            "statistical_summary_count": 0.0,
        }

    # Find all numeric values with positions
    values_with_pos = _find_numeric_values_with_positions(combined)
    values = [v for v, _ in values_with_pos]

    # 1. numeric_value_count
    numeric_value_count = float(len(values))

    # 2. out_of_range_count
    out_of_range_count = float(_count_out_of_range(combined, values_with_pos))

    # 3. z_score_outlier_count
    z_score_outlier_count = float(_count_z_score_outliers(values))

    # 4. negative_value_count
    negative_value_count = float(sum(1 for v in values if v < 0))

    # 5. unit_mention_count
    unit_mention_count = float(len(_UNIT_PATTERNS.findall(combined)))

    # 6. value_magnitude_range
    abs_values = [abs(v) for v in values if abs(v) > 0]
    if len(abs_values) >= 2:
        max_abs = max(abs_values)
        min_abs = min(abs_values)
        value_magnitude_range = math.log10(max_abs / min_abs) if min_abs > 0 else 0.0
    else:
        value_magnitude_range = 0.0

    # 7. statistical_summary_count
    stat_count = 0
    for pattern in _STAT_TERMS:
        stat_count += len(re.findall(pattern, combined, re.IGNORECASE))
    statistical_summary_count = float(stat_count)

    return {
        "numeric_value_count": numeric_value_count,
        "out_of_range_count": out_of_range_count,
        "z_score_outlier_count": z_score_outlier_count,
        "negative_value_count": negative_value_count,
        "unit_mention_count": unit_mention_count,
        "value_magnitude_range": value_magnitude_range,
        "statistical_summary_count": statistical_summary_count,
    }
