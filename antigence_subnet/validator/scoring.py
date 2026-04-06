"""Validator scoring strategies.

Exact scoring remains the default validator path. Statistical scoring is an
opt-in wrapper that repeats the exact scorer N times and summarizes the
distribution, which makes the compute multiplier explicit for operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from antigence_subnet.miner.orchestrator.model_manager import ModelManager
from antigence_subnet.validator.reward import DECISION_THRESHOLD, get_rewards
from antigence_subnet.validator.validation import validate_response


@dataclass(frozen=True)
class ScoreResult:
    """Normalized scorer output for validator reward strategies."""

    mode: str
    rewards: np.ndarray
    means: np.ndarray
    confidence_interval_lower: np.ndarray
    confidence_interval_upper: np.ndarray
    spread: np.ndarray
    repeats: int
    samples: np.ndarray


class ExactScorer:
    """Adapter around the existing exact reward path."""

    mode = "exact"

    def score_round(
        self,
        *,
        validator,
        miner_uids: list[int],
        responses_by_sample: dict,
        manifest: dict,
    ) -> ScoreResult:
        rewards = get_rewards(validator, miner_uids, responses_by_sample, manifest)
        zero_spread = np.zeros_like(rewards)
        return ScoreResult(
            mode=self.mode,
            rewards=rewards,
            means=rewards,
            confidence_interval_lower=rewards,
            confidence_interval_upper=rewards,
            spread=zero_spread,
            repeats=1,
            samples=np.expand_dims(rewards, axis=0),
        )


class StatisticalScorer:
    """Repeat exact scoring and expose uncertainty-aware summary statistics."""

    mode = "statistical"

    def __init__(
        self,
        *,
        exact_scorer: ExactScorer | None = None,
        repeats: int = 3,
        confidence_level: float = 0.95,
    ) -> None:
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        self.exact_scorer = exact_scorer or ExactScorer()
        self.repeats = repeats
        self.confidence_level = confidence_level

    def score_round(
        self,
        *,
        validator,
        miner_uids: list[int],
        responses_by_sample: dict,
        manifest: dict,
    ) -> ScoreResult:
        repeated_rewards = []
        for _ in range(self.repeats):
            # Statistical mode intentionally pays the full exact-scoring cost
            # on each repeat instead of caching one pass.
            exact_result = self.exact_scorer.score_round(
                validator=validator,
                miner_uids=miner_uids,
                responses_by_sample=responses_by_sample,
                manifest=manifest,
            )
            repeated_rewards.append(np.asarray(exact_result.rewards, dtype=np.float32))

        samples = np.stack(repeated_rewards)
        means = samples.mean(axis=0, dtype=np.float32)
        if self.repeats == 1:
            lower = means.copy()
            upper = means.copy()
            spread = np.zeros_like(means)
        else:
            spread = samples.std(axis=0, ddof=1).astype(np.float32)
            lower = np.zeros_like(means)
            upper = np.zeros_like(means)
            for idx in range(samples.shape[1]):
                column = samples[:, idx]
                if np.allclose(column, column[0]):
                    lower[idx] = means[idx]
                    upper[idx] = means[idx]
                    continue
                sem = stats.sem(column)
                ci_low, ci_high = stats.t.interval(
                    confidence=self.confidence_level,
                    df=len(column) - 1,
                    loc=float(means[idx]),
                    scale=float(sem),
                )
                lower[idx] = np.float32(ci_low)
                upper[idx] = np.float32(ci_high)

        return ScoreResult(
            mode=self.mode,
            rewards=means,
            means=means,
            confidence_interval_lower=lower,
            confidence_interval_upper=upper,
            spread=spread,
            repeats=self.repeats,
            samples=samples,
        )


class SemanticScorer:
    """Semantic scorer using the existing model-manager similarity primitive."""

    mode = "semantic"
    DEFAULT_THRESHOLDS = {
        "hallucination": 0.85,
        "code": 0.95,
        "reasoning": 0.90,
        "bio": 0.92,
    }

    def __init__(
        self,
        *,
        similarity_adapter: ModelManager | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.similarity_adapter = similarity_adapter or ModelManager()
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(thresholds)

    def score_round(
        self,
        *,
        validator,
        miner_uids: list[int],
        responses_by_sample: dict,
        manifest: dict,
    ) -> ScoreResult:
        del validator
        if not self.similarity_adapter.is_available():
            raise RuntimeError(
                "semantic scoring requires sentence-transformers support; "
                "install with: pip install 'antigence-subnet[sbert]'"
            )

        rewards = np.zeros(len(miner_uids), dtype=np.float32)
        sample_ids = list(responses_by_sample.keys())
        semantic_labels = {}
        for sample_id in sample_ids:
            sample_manifest = manifest[sample_id]
            domain = self._normalize_domain(sample_manifest.get("domain"))
            threshold = self.thresholds[domain]
            prompt = self._get_required_text(sample_manifest, "prompt", sample_id)
            output = self._get_required_text(sample_manifest, "output", sample_id)
            similarity = self.similarity_adapter.score(prompt, output)
            semantic_labels[sample_id] = similarity < threshold

        for miner_idx, uid in enumerate(miner_uids):
            del uid
            valid_samples = 0
            correct_predictions = 0
            honeypot_failed = False
            for sample_id in sample_ids:
                response = responses_by_sample[sample_id][miner_idx]
                is_valid, _reason = validate_response(response)
                if not is_valid:
                    if manifest[sample_id].get("is_honeypot", False):
                        honeypot_failed = True
                    continue
                predicted_anomalous = response.anomaly_score >= DECISION_THRESHOLD
                expected_anomalous = semantic_labels[sample_id]
                valid_samples += 1
                if predicted_anomalous == expected_anomalous:
                    correct_predictions += 1
                elif manifest[sample_id].get("is_honeypot", False):
                    honeypot_failed = True
            if honeypot_failed or valid_samples == 0:
                rewards[miner_idx] = 0.0
            else:
                rewards[miner_idx] = np.float32(correct_predictions / valid_samples)

        zero_spread = np.zeros_like(rewards)
        return ScoreResult(
            mode=self.mode,
            rewards=rewards,
            means=rewards,
            confidence_interval_lower=rewards,
            confidence_interval_upper=rewards,
            spread=zero_spread,
            repeats=1,
            samples=np.expand_dims(rewards, axis=0),
        )

    def _normalize_domain(self, domain: str | None) -> str:
        if not domain:
            raise ValueError("semantic scoring requires manifest domain")
        normalized = domain.lower().strip()
        aliases = {
            "code_security": "code",
            "code-security": "code",
            "biology": "bio",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.thresholds:
            raise ValueError(f"unsupported semantic scoring domain: {domain}")
        return normalized

    @staticmethod
    def _get_required_text(sample_manifest: dict, field: str, sample_id: str) -> str:
        value = sample_manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"semantic scoring requires manifest.{field} for sample {sample_id}"
            )
        return value


def build_validator_scorer(
    mode: str | None = None,
    *,
    repeats: int = 3,
    confidence_level: float = 0.95,
) -> ExactScorer | StatisticalScorer | SemanticScorer:
    """Build the validator scorer for the selected mode."""
    selected_mode = (mode or "exact").lower()
    if selected_mode == "exact":
        return ExactScorer()
    if selected_mode == "statistical":
        return StatisticalScorer(
            repeats=repeats,
            confidence_level=confidence_level,
        )
    if selected_mode == "semantic":
        return SemanticScorer()
    raise ValueError(f"unsupported validator scoring mode: {selected_mode}")
