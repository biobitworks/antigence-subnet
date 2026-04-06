"""Tests for precision-first reward function with honeypot checking.

Covers RWRD-01 (precision-first scoring) and RWRD-03 (honeypot penalty).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from antigence_subnet.validator.reward import (
    BASE_WEIGHT,
    CALIBRATION_WEIGHT,
    DECISION_THRESHOLD,
    DIVERSITY_WEIGHT,
    FP_PENALTY_MULTIPLIER,
    PRECISION_WEIGHT,
    RECALL_WEIGHT,
    ROBUSTNESS_WEIGHT,
    check_honeypot_pass,
    compute_diversity_bonus,
    compute_reward,
    get_composite_rewards,
    get_rewards,
)


class TestComputeReward:
    """Tests for compute_reward function."""

    def test_precision_beats_recall(self):
        """A miner with 90% precision and 60% recall (0.81) outscores
        a miner with 60% precision and 90% recall (0.69)."""
        # High precision miner: 9 TP, 1 FP out of 10 flagged, 4 FN
        # precision = 9/10 = 0.9, recall = 9/13 ~ 0.69 (not exactly 0.6)
        # Instead, construct exact scenario:
        # Miner A: precision=0.9, recall=0.6
        #   TP=9, FP=1, FN=6 -> precision=9/10=0.9, recall=9/15=0.6
        scores_a = [0.8] * 9 + [0.8] + [0.2] * 6 + [0.2] * 4
        truths_a = ["anomalous"] * 9 + ["normal"] * 1 + ["anomalous"] * 6 + ["normal"] * 4
        # TP=9, FP=1, FN=6, TN=4 -> precision=9/10=0.9, recall=9/15=0.6
        reward_a = compute_reward(scores_a, truths_a)

        # Miner B: precision=0.6, recall=0.9
        #   TP=9, FP=6, FN=1 -> precision=9/15=0.6, recall=9/10=0.9
        scores_b = [0.8] * 9 + [0.8] * 6 + [0.2] * 1 + [0.2] * 4
        truths_b = ["anomalous"] * 9 + ["normal"] * 6 + ["anomalous"] * 1 + ["normal"] * 4
        # TP=9, FP=6, FN=1, TN=4 -> precision=9/15=0.6, recall=9/10=0.9
        reward_b = compute_reward(scores_b, truths_b)

        assert reward_a == pytest.approx(0.81, abs=0.01)
        assert reward_b == pytest.approx(0.69, abs=0.01)
        assert reward_a > reward_b

    def test_perfect_miner_scores_one(self):
        """A miner predicting all samples correctly gets reward=1.0."""
        # All anomalous correctly flagged, all normal correctly unflagged
        scores = [0.9, 0.9, 0.1, 0.1]
        truths = ["anomalous", "anomalous", "normal", "normal"]
        # TP=2, FP=0, FN=0, TN=2 -> precision=1.0, recall=1.0
        reward = compute_reward(scores, truths)
        assert reward == pytest.approx(1.0)

    def test_no_response_scored_zero(self):
        """An empty list of scores results in reward=0.0."""
        reward = compute_reward([], [])
        assert reward == pytest.approx(0.0)

    def test_false_positive_penalty(self):
        """A miner with 2 FP has lower reward than a miner with 0 FP."""
        # Miner A: 5 TP, 0 FP -> precision=1.0
        scores_a = [0.8] * 5 + [0.2] * 5
        truths_a = ["anomalous"] * 5 + ["normal"] * 5
        reward_a = compute_reward(scores_a, truths_a)

        # Miner B: 5 TP, 2 FP -> precision=5/7~0.714
        scores_b = [0.8] * 5 + [0.8] * 2 + [0.2] * 3
        truths_b = ["anomalous"] * 5 + ["normal"] * 2 + ["normal"] * 3
        reward_b = compute_reward(scores_b, truths_b)

        assert reward_a > reward_b

    def test_all_negative_predictions(self):
        """A miner that never flags anything gets 0.0 when anomalies exist."""
        scores = [0.1] * 10
        truths = ["anomalous"] * 5 + ["normal"] * 5
        # TP=0, FP=0, FN=5, TN=5 -> precision=0, recall=0
        reward = compute_reward(scores, truths)
        assert reward == pytest.approx(0.0)


class TestCheckHoneypotPass:
    """Tests for check_honeypot_pass function."""

    def test_honeypot_failure_zeroes_round(self):
        """Failing a honeypot means the check returns False."""
        manifest = {
            "hp-001": {
                "ground_truth_label": "anomalous",
                "is_honeypot": True,
            }
        }
        # Miner predicts normal (0.2) but truth is anomalous -> FAIL
        result = check_honeypot_pass("hp-001", 0.2, manifest)
        assert result is False

    def test_honeypot_pass_no_effect(self):
        """Passing a honeypot returns True."""
        manifest = {
            "hp-001": {
                "ground_truth_label": "anomalous",
                "is_honeypot": True,
            }
        }
        # Miner predicts anomalous (0.8) and truth is anomalous -> PASS
        result = check_honeypot_pass("hp-001", 0.8, manifest)
        assert result is True

    def test_non_honeypot_always_passes(self):
        """A non-honeypot sample always passes the honeypot check."""
        manifest = {
            "eval-001": {
                "ground_truth_label": "normal",
                "is_honeypot": False,
            }
        }
        result = check_honeypot_pass("eval-001", 0.1, manifest)
        assert result is True

    def test_honeypot_normal_correctly_predicted(self):
        """Normal honeypot correctly predicted as normal passes."""
        manifest = {
            "hp-002": {
                "ground_truth_label": "normal",
                "is_honeypot": True,
            }
        }
        # Miner predicts normal (0.3) and truth is normal -> PASS
        result = check_honeypot_pass("hp-002", 0.3, manifest)
        assert result is True


class TestGetRewards:
    """Integration tests for get_rewards function."""

    def _make_response(self, anomaly_score=None, anomaly_type=None):
        """Create a mock response object."""
        return SimpleNamespace(
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=0.9,
        )

    def test_get_rewards_integration(self):
        """get_rewards computes per-miner rewards from response dict."""
        miner_uids = [0, 1, 2]

        # Manifest with 3 samples (1 honeypot)
        manifest = {
            "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
            "s2": {"ground_truth_label": "normal", "is_honeypot": False},
            "s3": {"ground_truth_label": "anomalous", "is_honeypot": True},
        }

        # responses_by_sample: sample_id -> list of responses (one per miner)
        responses_by_sample = {
            "s1": [
                self._make_response(0.9, "factual_error"),  # miner 0: correct
                self._make_response(0.1),                    # miner 1: miss
                self._make_response(0.9, "factual_error"),  # miner 2: correct
            ],
            "s2": [
                self._make_response(0.1),                    # miner 0: correct
                self._make_response(0.9, "factual_error"),  # miner 1: FP
                self._make_response(0.1),                    # miner 2: correct
            ],
            "s3": [
                self._make_response(0.9, "factual_error"),  # miner 0: correct (honeypot pass)
                self._make_response(0.9, "factual_error"),  # miner 1: correct (honeypot pass)
                self._make_response(0.1),                    # miner 2: HONEYPOT FAIL
            ],
        }

        validator = SimpleNamespace()  # Not used directly by get_rewards

        rewards = get_rewards(validator, miner_uids, responses_by_sample, manifest)

        assert isinstance(rewards, np.ndarray)
        assert rewards.dtype == np.float32
        assert len(rewards) == 3

        # Miner 0: TP=2, TN=1, precision=1.0, recall=1.0 -> reward=1.0
        assert rewards[0] == pytest.approx(1.0)

        # Miner 1: passed honeypot on s3, but s1 FN, s2 FP, s3 TP
        # TP=1 (s3), FP=1 (s2), FN=1 (s1), TN=0 -> precision=1/2=0.5, recall=1/2=0.5
        assert rewards[1] == pytest.approx(0.5)

        # Miner 2: HONEYPOT FAIL on s3 -> entire round zeroed
        assert rewards[2] == pytest.approx(0.0)

    def test_get_rewards_no_response(self):
        """A miner with None anomaly_score gets reward=0.0."""
        miner_uids = [0]
        manifest = {
            "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
        }
        responses_by_sample = {
            "s1": [self._make_response(None)],  # No response
        }
        validator = SimpleNamespace()
        rewards = get_rewards(validator, miner_uids, responses_by_sample, manifest)
        assert rewards[0] == pytest.approx(0.0)

    def test_get_rewards_invalid_response_treated_as_no_response(self):
        """An invalid response (out-of-range score) is treated as no-response."""
        miner_uids = [0]
        manifest = {
            "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
        }
        responses_by_sample = {
            "s1": [self._make_response(1.5, "factual_error")],  # Out of range
        }
        validator = SimpleNamespace()
        rewards = get_rewards(validator, miner_uids, responses_by_sample, manifest)
        assert rewards[0] == pytest.approx(0.0)

    def test_get_rewards_matches_exact_scorer_default_contract(self):
        """The scorer facade must preserve exact-mode behavior by default."""
        from antigence_subnet.validator.scoring import build_validator_scorer

        miner_uids = [0, 1]
        manifest = {
            "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
            "s2": {"ground_truth_label": "normal", "is_honeypot": True},
        }
        responses_by_sample = {
            "s1": [
                self._make_response(0.9, "factual_error"),
                self._make_response(0.2),
            ],
            "s2": [
                self._make_response(0.1),
                self._make_response(0.9, "false_positive"),
            ],
        }
        validator = SimpleNamespace()

        scorer = build_validator_scorer()
        result = scorer.score_round(
            validator=validator,
            miner_uids=miner_uids,
            responses_by_sample=responses_by_sample,
            manifest=manifest,
        )

        np.testing.assert_array_equal(
            result.rewards,
            get_rewards(validator, miner_uids, responses_by_sample, manifest),
        )


class TestConstants:
    """Verify reward constants are set correctly."""

    def test_precision_weight(self):
        assert PRECISION_WEIGHT == 0.7

    def test_recall_weight(self):
        assert RECALL_WEIGHT == 0.3

    def test_decision_threshold(self):
        assert DECISION_THRESHOLD == 0.5

    def test_fp_penalty_multiplier(self):
        assert FP_PENALTY_MULTIPLIER == 3.0

    def test_composite_weights_sum_to_one(self):
        """BASE + CALIBRATION + ROBUSTNESS + DIVERSITY weights must sum to 1.0."""
        total = BASE_WEIGHT + CALIBRATION_WEIGHT + ROBUSTNESS_WEIGHT + DIVERSITY_WEIGHT
        assert total == pytest.approx(1.0)

    def test_base_weight(self):
        assert pytest.approx(0.70) == BASE_WEIGHT

    def test_calibration_weight(self):
        assert pytest.approx(0.10) == CALIBRATION_WEIGHT

    def test_robustness_weight(self):
        assert pytest.approx(0.10) == ROBUSTNESS_WEIGHT

    def test_diversity_weight(self):
        assert pytest.approx(0.10) == DIVERSITY_WEIGHT


class TestComputeDiversityBonus:
    """Tests for compute_diversity_bonus helper."""

    def test_unique_miner_gets_high_bonus(self):
        """A miner with a unique score vector gets bonus close to 1.0."""
        # Miner 1 has a very different vector from everyone else
        score_vectors = {
            1: np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]),
            2: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            3: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            4: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            5: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            6: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            7: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            8: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        }
        bonus = compute_diversity_bonus(1, score_vectors)
        # Cosine similarity between [0.9,0.1,...] and [0.5,0.5,...] is <1.0
        # Bonus = 1 - max_similarity, should be > 0.0
        assert bonus > 0.0
        assert bonus <= 1.0

    def test_identical_miners_get_low_bonus(self):
        """Miners with identical vectors get bonus = 0.0 (1.0 - 1.0 similarity)."""
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}
        bonus = compute_diversity_bonus(0, score_vectors)
        assert bonus == pytest.approx(0.0, abs=0.01)

    def test_insufficient_miners_returns_neutral(self):
        """Fewer than min_miners returns neutral bonus 0.5."""
        score_vectors = {
            0: np.array([0.5] * 10),
            1: np.array([0.5] * 10),
        }
        bonus = compute_diversity_bonus(0, score_vectors, min_miners=8)
        assert bonus == pytest.approx(0.5)

    def test_insufficient_history_returns_neutral(self):
        """UID with insufficient history returns neutral bonus 0.5."""
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}
        score_vectors[0] = np.array([0.5] * 3)  # Only 3 scores, need 10
        bonus = compute_diversity_bonus(0, score_vectors, min_history=10)
        assert bonus == pytest.approx(0.5)

    def test_bonus_range(self):
        """Diversity bonus is always in [0, 1]."""
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}
        for uid in score_vectors:
            bonus = compute_diversity_bonus(uid, score_vectors)
            assert 0.0 <= bonus <= 1.0


class TestCompositeRewards:
    """Tests for get_composite_rewards function."""

    def _make_response(self, anomaly_score=None, anomaly_type=None, confidence=0.9):
        """Create a mock response object."""
        return SimpleNamespace(
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=confidence,
        )

    def _make_manifest(self):
        """Create a simple manifest with 4 samples (1 honeypot)."""
        return {
            "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
            "s2": {"ground_truth_label": "normal", "is_honeypot": False},
            "s3": {"ground_truth_label": "anomalous", "is_honeypot": True},
            "s4": {"ground_truth_label": "normal", "is_honeypot": False},
        }

    def _make_perfect_responses(self, manifest):
        """Create perfectly correct responses for a miner."""
        responses = []
        for sample_id, truth in manifest.items():
            sample = {"id": sample_id}
            if truth["ground_truth_label"] == "anomalous":
                resp = self._make_response(0.9, "factual_error")
            else:
                resp = self._make_response(0.1)
            responses.append((sample, resp))
        return responses

    def test_constants_sum_to_one(self):
        """Verify 70+10+10+10 = 100%."""
        assert (
            pytest.approx(1.0)
            == BASE_WEIGHT + CALIBRATION_WEIGHT + ROBUSTNESS_WEIGHT + DIVERSITY_WEIGHT
        )

    def test_formula_with_known_values(self):
        """70% base=0.8 + 10% cal=1.0 + 10% stab=1.0 + 10% div=1.0 = 0.86."""
        # Perfect miner: all responses correct, well-calibrated, stable, unique
        validator = SimpleNamespace()
        miner_uids = [1]
        manifest = self._make_manifest()

        # Build responses for miner 1 -- all correct
        responses_by_miner = {1: self._make_perfect_responses(manifest)}

        # Perturbation map: empty (no perturbation samples -> stability defaults to 1.0)
        perturbation_map = {1: {}}

        # Confidence history: perfectly calibrated (confidence matches accuracy)
        # For simplicity, provide a sliding window that gives bonus=1.0
        confidence_history = {1: [([0.5, 0.5], [1, 0])]}

        # Score vectors: unique miner among many -> bonus close to 1.0
        # But with only 1 miner, it will get neutral (0.5) unless we have >= min_miners
        # Use many miners to get a real diversity bonus
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}
        # Make miner 1 unique
        score_vectors[1] = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])

        rewards = get_composite_rewards(
            validator, miner_uids, responses_by_miner, manifest,
            perturbation_map, confidence_history, score_vectors,
        )

        # base_reward = compute_reward on perfect -> 1.0
        # calibration_bonus = 1.0 (perfectly calibrated)
        # stability_bonus = 1.0 (default, no perturbation data)
        # diversity_bonus = 1 - max_cosine_sim (unique vs uniform vectors)
        # Exact diversity depends on cosine_sim, but should be > 0.5
        assert len(rewards) == 1
        assert rewards[0] > 0.7  # At minimum base component
        assert rewards[0] <= 1.0

    def test_base_only_miner(self):
        """Miner with cal=0, stab=0, div=0 scores 0.7 * base."""
        validator = SimpleNamespace()
        miner_uids = [1]
        manifest = self._make_manifest()
        responses_by_miner = {1: self._make_perfect_responses(manifest)}
        perturbation_map = {1: {}}

        # Maximally overconfident: ECE=0.9, bonus=0.1 (nearly zero)
        confidence_history = {1: [([0.9, 0.9, 0.9], [0, 0, 0])]}

        # All identical miners -> diversity bonus = 0.0
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}

        rewards = get_composite_rewards(
            validator, miner_uids, responses_by_miner, manifest,
            perturbation_map, confidence_history, score_vectors,
        )

        # base=1.0 (perfect accuracy)
        # calibration_bonus = 0.1 (ECE=0.9)
        # stability = 1.0 (default)
        # diversity = 0.0 (identical)
        # final = 0.7*1.0 + 0.1*0.1 + 0.1*1.0 + 0.1*0.0 = 0.7 + 0.01 + 0.1 + 0 = 0.81
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.81, abs=0.02)

    def test_honeypot_failure_zeroes_composite(self):
        """Honeypot failure zeroes entire composite reward."""
        validator = SimpleNamespace()
        miner_uids = [1]
        manifest = self._make_manifest()

        # Build responses for miner 1 -- fail honeypot s3
        responses = []
        for sample_id, truth in manifest.items():
            sample = {"id": sample_id}
            if sample_id == "s3":
                # Honeypot: truth is anomalous, but predict normal -> FAIL
                resp = self._make_response(0.1)
            elif truth["ground_truth_label"] == "anomalous":
                resp = self._make_response(0.9, "factual_error")
            else:
                resp = self._make_response(0.1)
            responses.append((sample, resp))

        responses_by_miner = {1: responses}
        perturbation_map = {1: {}}
        confidence_history = {1: [([0.5, 0.5], [1, 0])]}
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}

        rewards = get_composite_rewards(
            validator, miner_uids, responses_by_miner, manifest,
            perturbation_map, confidence_history, score_vectors,
        )

        assert rewards[0] == pytest.approx(0.0)

    def test_calibrated_miner_outscores_overconfident(self):
        """Well-calibrated miner outscores overconfident miner with same base accuracy."""
        validator = SimpleNamespace()
        miner_uids = [1, 2]
        manifest = self._make_manifest()

        # Both miners: same perfect responses
        responses_by_miner = {
            1: self._make_perfect_responses(manifest),
            2: self._make_perfect_responses(manifest),
        }
        perturbation_map = {1: {}, 2: {}}

        # Miner 1: well-calibrated
        confidence_history = {
            1: [([0.5, 0.5], [1, 0])],  # ECE=0.0, bonus=1.0
            2: [([0.9, 0.9, 0.9], [0, 0, 0])],  # ECE=0.9, bonus=0.1
        }

        # Same diversity for both
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}

        rewards = get_composite_rewards(
            validator, miner_uids, responses_by_miner, manifest,
            perturbation_map, confidence_history, score_vectors,
        )

        # Miner 1 has higher calibration bonus -> higher total
        assert rewards[0] > rewards[1]

    def test_output_clamped_to_zero_one(self):
        """Composite output is always in [0, 1]."""
        validator = SimpleNamespace()
        miner_uids = [1]
        manifest = self._make_manifest()
        responses_by_miner = {1: self._make_perfect_responses(manifest)}
        perturbation_map = {1: {}}
        confidence_history = {1: [([0.5, 0.5], [1, 0])]}
        score_vectors = {uid: np.array([0.5] * 10) for uid in range(8)}

        rewards = get_composite_rewards(
            validator, miner_uids, responses_by_miner, manifest,
            perturbation_map, confidence_history, score_vectors,
        )

        assert 0.0 <= rewards[0] <= 1.0
