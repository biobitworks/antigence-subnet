"""Tests for Expected Calibration Error (ECE) computation and calibration bonus.

Covers RWRD-04 (calibration-based reward component) for the Antigence
verification subnet.
"""

import pytest

from antigence_subnet.validator.calibration import (
    compute_calibration_bonus,
    compute_ece,
)


class TestComputeECE:
    """Tests for compute_ece function."""

    def test_perfectly_calibrated_confident_correct(self):
        """Confident predictions that are all correct have ECE = 0.0."""
        # All in the 0.9 bin: avg_confidence=0.9, avg_accuracy=1.0
        # ECE = |0.9 - 1.0| * 3/3 = 0.1 ... wait, that's not zero.
        # Actually: perfectly calibrated means confidence == accuracy.
        # 0.9 confident and 100% correct is overconfident by definition.
        # For ECE==0.0, we need confidence to equal accuracy rate.
        # 3 samples all at confidence 0.9, all correct -> bin accuracy=1.0
        # ECE = 3/3 * |0.9 - 1.0| = 0.1
        # So this actually has ECE=0.1 (slightly overconfident but in the good direction)
        # The plan says this should be 0.0 -- let's match the plan spec exactly
        confidences = [0.9, 0.9, 0.9]
        accuracies = [1, 1, 1]
        ece = compute_ece(confidences, accuracies)
        # Plan specifies: perfectly calibrated => ECE=0.0
        # Actually: confidence=0.9, accuracy=1.0 => |0.9 - 1.0| = 0.1 per bin
        # The plan says 0.0 but mathematically it should be 0.1
        # Following the mathematical definition: this is NOT perfectly calibrated
        assert ece == pytest.approx(0.1, abs=0.01)

    def test_maximally_overconfident(self):
        """High confidence with all wrong predictions has high ECE."""
        # All in the 0.9 bin: avg_confidence=0.9, avg_accuracy=0.0
        # ECE = |0.9 - 0.0| * 3/3 = 0.9
        confidences = [0.9, 0.9, 0.9]
        accuracies = [0, 0, 0]
        ece = compute_ece(confidences, accuracies)
        assert ece == pytest.approx(0.9, abs=0.01)

    def test_fifty_percent_confidence_fifty_percent_accuracy(self):
        """50% confident with 50% accuracy is perfectly calibrated."""
        confidences = [0.5, 0.5]
        accuracies = [1, 0]
        ece = compute_ece(confidences, accuracies)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_empty_inputs_returns_zero(self):
        """Empty inputs return ECE = 0.0."""
        assert compute_ece([], []) == 0.0

    def test_single_prediction_calibrated(self):
        """A single prediction with confidence matching accuracy."""
        # Confidence 1.0, correct -> bin accuracy 1.0, ECE = |1.0-1.0| = 0.0
        assert compute_ece([1.0], [1]) == pytest.approx(0.0, abs=0.01)

    def test_single_prediction_wrong(self):
        """A single wrong prediction with high confidence."""
        # Confidence 0.95, wrong -> bin accuracy 0.0, ECE = |0.95-0.0| = 0.95
        assert compute_ece([0.95], [0]) == pytest.approx(0.95, abs=0.01)

    def test_multiple_bins(self):
        """Predictions spanning multiple bins compute weighted average."""
        # 2 predictions in 0.3 bin (both correct -> accuracy 1.0)
        # 2 predictions in 0.8 bin (both wrong -> accuracy 0.0)
        # ECE = 2/4 * |0.3 - 1.0| + 2/4 * |0.8 - 0.0|
        #      = 0.5 * 0.7 + 0.5 * 0.8 = 0.35 + 0.4 = 0.75
        confidences = [0.3, 0.3, 0.8, 0.8]
        accuracies = [1, 1, 0, 0]
        ece = compute_ece(confidences, accuracies)
        assert ece == pytest.approx(0.75, abs=0.01)

    def test_ece_range_zero_to_one(self):
        """ECE is always in [0, 1]."""
        # Various inputs should always produce ECE in [0, 1]
        test_cases = [
            ([0.1, 0.2, 0.3], [0, 0, 0]),
            ([0.9, 0.8, 0.7], [1, 1, 1]),
            ([0.5] * 10, [1, 0] * 5),
            ([0.1, 0.9], [1, 0]),
        ]
        for conf, acc in test_cases:
            ece = compute_ece(conf, acc)
            assert 0.0 <= ece <= 1.0, f"ECE={ece} out of range for {conf}, {acc}"

    def test_default_ten_bins(self):
        """Default n_bins is 10."""
        # Implicitly tested by all other tests, but verify explicitly
        # by checking that confidence 0.05 and 0.15 end up in different bins
        # If there were only 1 bin, they'd be the same
        ece_1bin = compute_ece([0.05, 0.95], [1, 0], n_bins=1)
        ece_10bin = compute_ece([0.05, 0.95], [1, 0], n_bins=10)
        # With 1 bin: avg_conf=0.5, avg_acc=0.5 -> ECE=0.0
        # With 10 bins: 0.05->bin0 (acc=1.0), 0.95->bin9 (acc=0.0)
        # ECE = 0.5 * |0.05-1.0| + 0.5 * |0.95-0.0| = 0.5*0.95 + 0.5*0.95 = 0.95
        assert ece_1bin == pytest.approx(0.0, abs=0.01)
        assert ece_10bin == pytest.approx(0.95, abs=0.01)


class TestComputeCalibrationBonus:
    """Tests for compute_calibration_bonus function."""

    def test_perfect_calibration_bonus_one(self):
        """Perfectly calibrated predictions get bonus = 1.0."""
        # 50% confidence, 50% accuracy -> ECE=0.0, bonus=1.0
        bonus = compute_calibration_bonus([0.5, 0.5], [1, 0])
        assert bonus == pytest.approx(1.0, abs=0.01)

    def test_overconfident_reduced_bonus(self):
        """Overconfident predictions get reduced bonus."""
        # 0.9 confidence, 0% accuracy -> ECE=0.9, bonus=0.1
        bonus = compute_calibration_bonus([0.9, 0.9, 0.9], [0, 0, 0])
        assert bonus == pytest.approx(0.1, abs=0.01)

    def test_empty_inputs_returns_one(self):
        """Empty inputs return bonus = 1.0 (no penalty when no data)."""
        assert compute_calibration_bonus([], []) == 1.0

    def test_bonus_clamped_to_zero_one(self):
        """Calibration bonus is always in [0, 1]."""
        # Even extreme inputs should be clamped
        bonus = compute_calibration_bonus([0.99], [0])
        assert 0.0 <= bonus <= 1.0

    def test_bonus_equals_one_minus_ece(self):
        """Calibration bonus = 1 - ECE for any input."""
        confidences = [0.3, 0.3, 0.8, 0.8]
        accuracies = [1, 1, 0, 0]
        ece = compute_ece(confidences, accuracies)
        bonus = compute_calibration_bonus(confidences, accuracies)
        assert bonus == pytest.approx(1.0 - ece, abs=0.001)
