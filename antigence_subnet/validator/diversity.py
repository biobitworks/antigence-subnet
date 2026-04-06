"""
Diversity weighting for sybil detection via cosine similarity.

Implements CHEAT-07: Miners with highly correlated score vectors
(cosine similarity > threshold) receive diminished rewards to prevent
sybil networks running identical detectors.

Note: compute_diversity_bonus() in reward.py is the active production
function used in the composite reward pipeline. This module provides
the standalone penalty computation for direct sybil detection use cases.
"""
