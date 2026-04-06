"""
Expected Calibration Error (ECE) computation and calibration bonus.

Implements RWRD-04: calibration-based reward component for the Antigence
verification subnet. Miners that report well-calibrated confidence scores
(confidence aligns with actual accuracy) receive a calibration bonus.

ECE formula (equal-width binning):
    ECE = sum_b (|B_b| / N) * |avg_confidence(B_b) - avg_accuracy(B_b)|

Calibration bonus:
    bonus = 1 - ECE  (clamped to [0, 1])
"""

import numpy as np


def compute_ece(
    confidences: list[float],
    accuracies: list[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error from confidence/accuracy pairs.

    Bins predictions into n_bins equal-width bins by confidence value.
    For each bin, computes the weighted absolute difference between
    average confidence and average accuracy.

    Args:
        confidences: List of confidence values in [0, 1] for each prediction.
        accuracies: List of binary accuracy values (0 or 1) indicating whether
            each prediction was correct.
        n_bins: Number of equal-width bins (default 10, bin edges at
            0.0, 0.1, 0.2, ..., 1.0).

    Returns:
        ECE value in [0.0, 1.0]. 0.0 means perfectly calibrated
        (confidence matches accuracy in every bin). Higher values
        indicate worse calibration.
    """
    if len(confidences) == 0:
        return 0.0

    confidences_arr = np.array(confidences, dtype=np.float64)
    accuracies_arr = np.array(accuracies, dtype=np.float64)
    n_samples = len(confidences_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i < n_bins - 1:
            # All bins except the last: lower <= conf < upper
            in_bin = (confidences_arr >= lower) & (confidences_arr < upper)
        else:
            # Last bin: lower <= conf <= upper (inclusive on right)
            in_bin = (confidences_arr >= lower) & (confidences_arr <= upper)

        bin_count = int(np.sum(in_bin))
        if bin_count == 0:
            continue

        avg_confidence = float(np.mean(confidences_arr[in_bin]))
        avg_accuracy = float(np.mean(accuracies_arr[in_bin]))

        ece += (bin_count / n_samples) * abs(avg_confidence - avg_accuracy)

    return float(np.clip(ece, 0.0, 1.0))


def compute_calibration_bonus(
    confidences: list[float],
    accuracies: list[int],
    n_bins: int = 10,
) -> float:
    """Compute calibration bonus from confidence/accuracy pairs.

    Higher bonus means better calibration. Perfect calibration yields
    bonus = 1.0, worst calibration yields bonus near 0.0.

    Args:
        confidences: List of confidence values in [0, 1].
        accuracies: List of binary accuracy values (0 or 1).
        n_bins: Number of ECE bins (default 10).

    Returns:
        Calibration bonus in [0.0, 1.0]. 1.0 = perfectly calibrated.
    """
    if len(confidences) == 0:
        return 1.0  # No data -> no penalty

    ece = compute_ece(confidences, accuracies, n_bins)
    return float(np.clip(1.0 - ece, 0.0, 1.0))
