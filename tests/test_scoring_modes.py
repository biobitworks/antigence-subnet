"""Contract tests for validator scoring modes."""

from types import SimpleNamespace

import numpy as np
import pytest

from antigence_subnet.validator.reward import get_rewards
from antigence_subnet.validator.scoring import (
    ExactScorer,
    SemanticScorer,
    ScoreResult,
    StatisticalScorer,
    build_validator_scorer,
)


def _make_response(anomaly_score=None, anomaly_type=None):
    return SimpleNamespace(
        anomaly_score=anomaly_score,
        anomaly_type=anomaly_type,
        confidence=0.9,
    )


def _round_fixture():
    miner_uids = [10, 11, 12]
    manifest = {
        "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
        "s2": {"ground_truth_label": "normal", "is_honeypot": False},
        "s3": {"ground_truth_label": "anomalous", "is_honeypot": True},
    }
    responses_by_sample = {
        "s1": [
            _make_response(0.9, "factual_error"),
            _make_response(0.1),
            _make_response(0.9, "factual_error"),
        ],
        "s2": [
            _make_response(0.1),
            _make_response(0.9, "factual_error"),
            _make_response(0.1),
        ],
        "s3": [
            _make_response(0.9, "factual_error"),
            _make_response(0.9, "factual_error"),
            _make_response(0.1),
        ],
    }
    validator = SimpleNamespace()
    return validator, miner_uids, responses_by_sample, manifest


def test_build_validator_scorer_exact_matches_current_reward_path_bit_for_bit():
    validator, miner_uids, responses_by_sample, manifest = _round_fixture()

    scorer = build_validator_scorer("exact")
    exact_result = scorer.score_round(
        validator=validator,
        miner_uids=miner_uids,
        responses_by_sample=responses_by_sample,
        manifest=manifest,
    )
    legacy_rewards = get_rewards(validator, miner_uids, responses_by_sample, manifest)

    assert isinstance(scorer, ExactScorer)
    assert isinstance(exact_result, ScoreResult)
    np.testing.assert_array_equal(exact_result.rewards, legacy_rewards)
    np.testing.assert_array_equal(exact_result.means, legacy_rewards)
    np.testing.assert_array_equal(exact_result.confidence_interval_lower, legacy_rewards)
    np.testing.assert_array_equal(exact_result.confidence_interval_upper, legacy_rewards)
    np.testing.assert_array_equal(exact_result.spread, np.zeros_like(legacy_rewards))
    assert exact_result.repeats == 1


@pytest.mark.parametrize("mode", [None, "exact"])
def test_build_validator_scorer_defaults_to_exact_when_mode_is_unspecified_or_exact(mode):
    scorer = build_validator_scorer(mode)

    assert isinstance(scorer, ExactScorer)


def test_build_validator_scorer_statistical_defaults_to_repeats_three_and_rejects_invalid_values():
    scorer = build_validator_scorer("statistical")

    assert isinstance(scorer, StatisticalScorer)
    assert scorer.repeats == 3

    with pytest.raises(ValueError, match="repeats"):
        build_validator_scorer("statistical", repeats=0)

    with pytest.raises(ValueError, match="repeats"):
        build_validator_scorer("statistical", repeats=-2)

    with pytest.raises(ValueError, match="unsupported"):
        build_validator_scorer("unknown")


def test_statistical_scoring_reports_mean_confidence_interval_and_small_sample_spread():
    validator, miner_uids, responses_by_sample, manifest = _round_fixture()
    per_repeat_rewards = [
        np.array([1.0, 0.5, 0.0], dtype=np.float32),
        np.array([0.7, 0.2, 0.0], dtype=np.float32),
        np.array([0.9, 0.4, 0.0], dtype=np.float32),
    ]

    class StubExactScorer:
        def __init__(self, rewards_per_repeat):
            self._rewards_per_repeat = rewards_per_repeat
            self.calls = 0

        def score_round(self, **kwargs):
            assert kwargs["validator"] is validator
            assert kwargs["miner_uids"] == miner_uids
            assert kwargs["responses_by_sample"] is responses_by_sample
            assert kwargs["manifest"] is manifest
            rewards = self._rewards_per_repeat[self.calls]
            self.calls += 1
            return ScoreResult(
                mode="exact",
                rewards=rewards,
                means=rewards,
                confidence_interval_lower=rewards,
                confidence_interval_upper=rewards,
                spread=np.zeros_like(rewards),
                repeats=1,
                samples=np.expand_dims(rewards, axis=0),
            )

    exact_stub = StubExactScorer(per_repeat_rewards)
    scorer = StatisticalScorer(exact_scorer=exact_stub, repeats=3)

    result = scorer.score_round(
        validator=validator,
        miner_uids=miner_uids,
        responses_by_sample=responses_by_sample,
        manifest=manifest,
    )

    repeated = np.stack(per_repeat_rewards)
    expected_means = repeated.mean(axis=0)

    np.testing.assert_allclose(result.rewards, expected_means)
    np.testing.assert_allclose(result.means, expected_means)
    np.testing.assert_allclose(result.samples, repeated)
    np.testing.assert_allclose(result.spread, repeated.std(axis=0, ddof=1))
    assert result.repeats == 3
    assert exact_stub.calls == 3
    assert np.all(result.confidence_interval_lower <= result.means)
    assert np.all(result.means <= result.confidence_interval_upper)


def test_statistical_mode_preserves_exact_semantics_for_zero_valid_and_honeypot_failure_cases():
    validator, miner_uids, responses_by_sample, manifest = _round_fixture()
    responses_by_sample["s1"][1] = _make_response(None)
    responses_by_sample["s2"][1] = _make_response(None)
    responses_by_sample["s3"][1] = _make_response(None)

    exact_result = build_validator_scorer("exact").score_round(
        validator=validator,
        miner_uids=miner_uids,
        responses_by_sample=responses_by_sample,
        manifest=manifest,
    )
    statistical_result = build_validator_scorer("statistical").score_round(
        validator=validator,
        miner_uids=miner_uids,
        responses_by_sample=responses_by_sample,
        manifest=manifest,
    )

    # miner 11 has zero valid responses; miner 12 still fails the honeypot
    assert exact_result.rewards[1] == pytest.approx(0.0)
    assert exact_result.rewards[2] == pytest.approx(0.0)
    assert statistical_result.rewards[1] == pytest.approx(0.0)
    assert statistical_result.rewards[2] == pytest.approx(0.0)
    assert statistical_result.means[1] == pytest.approx(0.0)
    assert statistical_result.means[2] == pytest.approx(0.0)
    assert statistical_result.spread[1] == pytest.approx(0.0)
    assert statistical_result.spread[2] == pytest.approx(0.0)


def test_statistical_mode_compute_multiplier_is_explicit_not_cached_single_pass_scoring():
    call_count = {"count": 0}

    class CountingExactScorer:
        def score_round(self, **kwargs):
            del kwargs
            call_count["count"] += 1
            rewards = np.array([0.5], dtype=np.float32)
            return ScoreResult(
                mode="exact",
                rewards=rewards,
                means=rewards,
                confidence_interval_lower=rewards,
                confidence_interval_upper=rewards,
                spread=np.zeros_like(rewards),
                repeats=1,
                samples=np.expand_dims(rewards, axis=0),
            )

    scorer = StatisticalScorer(exact_scorer=CountingExactScorer(), repeats=3)
    scorer.score_round(
        validator=SimpleNamespace(),
        miner_uids=[0],
        responses_by_sample={"s1": [_make_response(0.9)]},
        manifest={"s1": {"ground_truth_label": "anomalous", "is_honeypot": False}},
    )

    assert call_count["count"] == 3


def test_build_validator_scorer_semantic_exposes_locked_domain_thresholds():
    scorer = build_validator_scorer("semantic")

    assert isinstance(scorer, SemanticScorer)
    assert scorer.thresholds == {
        "hallucination": 0.85,
        "code": 0.95,
        "reasoning": 0.90,
        "bio": 0.92,
    }


@pytest.mark.parametrize("mode", ["exact", "statistical", "semantic"])
def test_build_validator_scorer_supports_the_phase83_benchmark_mode_matrix(mode):
    scorer = build_validator_scorer(mode)

    assert scorer.mode == mode


def test_semantic_scorer_reuses_model_manager_score_compatible_primitive():
    class StubSimilarityAdapter:
        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def score(self, prompt, output):
            self.calls.append((prompt, output))
            return 0.40

    scorer = SemanticScorer(similarity_adapter=StubSimilarityAdapter())
    result = scorer.score_round(
        validator=SimpleNamespace(),
        miner_uids=[0, 1],
        responses_by_sample={
            "s1": [_make_response(0.8), _make_response(0.2)],
        },
        manifest={
            "s1": {
                "domain": "reasoning",
                "prompt": "Explain the failure mode",
                "output": "The model violated the stated invariant.",
                "ground_truth_label": "anomalous",
                "is_honeypot": False,
            }
        },
    )

    assert scorer.similarity_adapter.calls == [
        (
            "Explain the failure mode",
            "The model violated the stated invariant.",
        )
    ]
    np.testing.assert_array_equal(result.rewards, np.array([1.0, 0.0], dtype=np.float32))


def test_semantic_scorer_applies_thresholds_as_deterministic_binary_pass_fail():
    class StubSimilarityAdapter:
        def __init__(self, similarity):
            self._similarity = similarity

        def is_available(self):
            return True

        def score(self, prompt, output):
            del prompt, output
            return self._similarity

    anomalous_result = SemanticScorer(
        similarity_adapter=StubSimilarityAdapter(0.70)
    ).score_round(
        validator=SimpleNamespace(),
        miner_uids=[0, 1],
        responses_by_sample={
            "low_sim": [_make_response(0.9), _make_response(0.2)],
        },
        manifest={
            "low_sim": {
                "domain": "hallucination",
                "prompt": "Prompt A",
                "output": "Output A",
                "ground_truth_label": "anomalous",
                "is_honeypot": False,
            }
        },
    )
    normal_result = SemanticScorer(
        similarity_adapter=StubSimilarityAdapter(0.97)
    ).score_round(
        validator=SimpleNamespace(),
        miner_uids=[0, 1],
        responses_by_sample={
            "high_sim": [_make_response(0.2), _make_response(0.9)],
        },
        manifest={
            "high_sim": {
                "domain": "hallucination",
                "prompt": "Prompt B",
                "output": "Output B",
                "ground_truth_label": "normal",
                "is_honeypot": False,
            }
        },
    )

    np.testing.assert_array_equal(
        anomalous_result.rewards,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        normal_result.rewards,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert anomalous_result.mode == "semantic"
    assert anomalous_result.repeats == 1


def test_semantic_mode_fails_clearly_when_dependency_is_unavailable():
    class MissingSimilarityAdapter:
        def is_available(self):
            return False

        def score(self, prompt, output):
            del prompt, output
            raise AssertionError("score should not be called when unavailable")

    scorer = SemanticScorer(similarity_adapter=MissingSimilarityAdapter())

    with pytest.raises(RuntimeError, match="semantic scoring requires sentence-transformers"):
        scorer.score_round(
            validator=SimpleNamespace(),
            miner_uids=[0],
            responses_by_sample={"s1": [_make_response(0.9)]},
            manifest={
                "s1": {
                    "domain": "bio",
                    "prompt": "Prompt",
                    "output": "Output",
                    "ground_truth_label": "anomalous",
                    "is_honeypot": False,
                }
            },
        )
