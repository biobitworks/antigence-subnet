"""Canned codeword fixtures for Phase 1002 syndrome tests (SYNDROME-05).

Five hand-picked deterministic Codewords that cover:
    * 1 self (all features inside the [-1, +1] band)
    * 3 explicit anomaly types (total_collapse, alternating_spike, saturation_storm)
    * 1 unclassified (bucket signature not in the default lookup table)

All features are exact Python floats. No RNG, no numpy, no biological terms.
"""

from __future__ import annotations

from antigence_subnet.validator.deterministic_scoring.syndrome import Codeword

# Self: every feature is well inside [-1.0, +1.0] -> bucket signature (0,)*8 -> "self".
CW_SELF = Codeword(
    schema_version=1,
    features=(0.1, -0.2, 0.3, 0.0, -0.5, 0.4, 0.1, -0.3),
    domain="generic",
)

# Total collapse: every feature is strictly below -1.0 -> bucket signature (-1,)*8.
CW_TOTAL_COLLAPSE = Codeword(
    schema_version=1,
    features=(-2.5, -3.1, -1.8, -2.0, -2.3, -1.5, -4.0, -2.2),
    domain="generic",
)

# Alternating spike: features alternate spike/quiet -> bucket signature (1,0,1,0,1,0,1,0).
CW_ALTERNATING_SPIKE = Codeword(
    schema_version=1,
    features=(2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 4.0, 0.0),
    domain="generic",
)

# Saturation storm: every feature strictly above +1.0 -> bucket signature (1,)*8.
CW_SATURATION_STORM = Codeword(
    schema_version=1,
    features=(1.5, 2.1, 3.0, 1.2, 4.0, 1.1, 2.5, 1.8),
    domain="generic",
)

# Unclassified: mixed buckets that do not appear in the default lookup table.
CW_UNCLASSIFIED = Codeword(
    schema_version=1,
    features=(1.5, -2.0, 0.0, 1.5, -2.0, 0.0, 1.5, -2.0),
    domain="generic",
)

ALL_FIXTURES = (
    CW_SELF,
    CW_TOTAL_COLLAPSE,
    CW_ALTERNATING_SPIKE,
    CW_SATURATION_STORM,
    CW_UNCLASSIFIED,
)

EXPECTED_BUCKET_SIGNATURES = {
    "CW_SELF": (0, 0, 0, 0, 0, 0, 0, 0),
    "CW_TOTAL_COLLAPSE": (-1, -1, -1, -1, -1, -1, -1, -1),
    "CW_ALTERNATING_SPIKE": (1, 0, 1, 0, 1, 0, 1, 0),
    "CW_SATURATION_STORM": (1, 1, 1, 1, 1, 1, 1, 1),
    "CW_UNCLASSIFIED": (1, -1, 0, 1, -1, 0, 1, -1),
}

EXPECTED_CLASSES = {
    "CW_SELF": "self",
    "CW_TOTAL_COLLAPSE": "total_collapse",
    "CW_ALTERNATING_SPIKE": "alternating_spike",
    "CW_SATURATION_STORM": "saturation_storm",
    "CW_UNCLASSIFIED": "unclassified",
}
