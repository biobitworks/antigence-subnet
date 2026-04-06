"""
Precision-first reward function with honeypot checking, stability bonus,
and 4-component composite reward.

Implements RWRD-01 (precision-first scoring), RWRD-03 (honeypot penalty),
CHEAT-04 (perturbation stability bonus), RWRD-04 (calibration bonus),
RWRD-05 (robustness bonus), and RWRD-06 (diversity bonus) for the Antigence
verification subnet.

Base reward formula:
    reward = 0.7 * precision + 0.3 * recall

Hardened reward formula (Phase 5 - backward compatible):
    reward = (1 - stability_weight) * base + stability_weight * stability_bonus

Composite reward formula (Phase 6):
    reward = 0.70 * base + 0.10 * calibration + 0.10 * robustness + 0.10 * diversity

Honeypot rule:
    Any honeypot failure zeroes the miner's entire round reward.
"""

from typing import Any

import bittensor as bt
import numpy as np

from antigence_subnet.validator.calibration import compute_calibration_bonus
from antigence_subnet.validator.perturbation import compute_stability_bonus
from antigence_subnet.validator.validation import validate_response

# --- Reward constants ---
PRECISION_WEIGHT = 0.7
RECALL_WEIGHT = 0.3
FP_PENALTY_MULTIPLIER = 3.0  # Documented for Phase 6 refinement; not used in formula
DECISION_THRESHOLD = 0.5  # anomaly_score >= 0.5 means "anomalous" prediction

# --- Composite reward weights (Phase 6) ---
BASE_WEIGHT = 0.70
CALIBRATION_WEIGHT = 0.10
ROBUSTNESS_WEIGHT = 0.10
DIVERSITY_WEIGHT = 0.10


def compute_reward(
    anomaly_scores: list[float],
    ground_truths: list[str],
) -> float:
    """Compute precision-first reward for a single miner's round responses.

    Classifies each (score, truth) pair into TP/FP/FN/TN using a threshold
    of 0.5. Ground truth values are "anomalous" or "normal".

    Args:
        anomaly_scores: Miner's anomaly_score for each sample.
        ground_truths: Ground truth label for each sample ("anomalous" or "normal").

    Returns:
        Reward value in [0.0, 1.0].
    """
    if len(anomaly_scores) == 0:
        return 0.0

    tp = fp = fn = tn = 0
    for score, truth in zip(anomaly_scores, ground_truths, strict=False):
        predicted_anomalous = score >= DECISION_THRESHOLD
        actually_anomalous = truth == "anomalous"

        if predicted_anomalous and actually_anomalous:
            tp += 1
        elif predicted_anomalous and not actually_anomalous:
            fp += 1
        elif not predicted_anomalous and actually_anomalous:
            fn += 1
        else:
            tn += 1

    # Precision: of things flagged anomalous, how many actually were?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall: of things actually anomalous, how many did we catch?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    reward = PRECISION_WEIGHT * precision + RECALL_WEIGHT * recall
    return float(np.clip(reward, 0.0, 1.0))


def check_honeypot_pass(
    sample_id: str,
    anomaly_score: float,
    manifest: dict,
    threshold: float = 0.5,
) -> bool:
    """Check if a miner passed a honeypot sample.

    Non-honeypot samples always return True. For honeypot samples,
    the miner's prediction must match the ground truth label.

    Args:
        sample_id: The evaluation sample identifier.
        anomaly_score: The miner's anomaly_score for this sample.
        manifest: Ground truth manifest dict keyed by sample_id.
        threshold: Decision threshold for binary classification.

    Returns:
        True if the sample is not a honeypot or the miner answered correctly.
        False if the miner failed a honeypot sample.
    """
    truth = manifest[sample_id]
    if not truth.get("is_honeypot", False):
        return True  # Not a honeypot, always passes

    predicted_anomalous = anomaly_score >= threshold
    actually_anomalous = truth["ground_truth_label"] == "anomalous"
    return predicted_anomalous == actually_anomalous


def get_rewards(
    validator,
    miner_uids: list[int],
    responses_by_sample: dict,
    manifest: dict,
) -> np.ndarray:
    """Compute per-miner rewards from responses across evaluation samples.

    This is the canonical exact-scoring path. Phase 82 scorer strategies wrap
    this function so exact mode stays backward compatible for existing callers.

    For each miner:
    1. Validate each response (invalid -> treated as no-response)
    2. Check honeypot samples (any failure -> zero entire round)
    3. Compute precision-first reward from valid responses

    Args:
        validator: The validator neuron instance (for logging context).
        miner_uids: List of miner UIDs being scored.
        responses_by_sample: Dict mapping sample_id -> list of responses
            (one per miner, in same order as miner_uids).
        manifest: Ground truth manifest dict keyed by sample_id.

    Returns:
        numpy array of reward values (float32), one per miner.
    """
    n_miners = len(miner_uids)
    rewards = np.zeros(n_miners, dtype=np.float32)

    sample_ids = list(responses_by_sample.keys())

    for miner_idx in range(n_miners):
        uid = miner_uids[miner_idx]
        anomaly_scores = []
        ground_truths = []
        honeypot_failed = False

        for sample_id in sample_ids:
            response = responses_by_sample[sample_id][miner_idx]

            # Validate response structure
            is_valid, reason = validate_response(response)
            if not is_valid:
                bt.logging.debug(
                    f"Miner {uid} sample {sample_id}: invalid response ({reason})"
                )
                # Treat invalid as no-response: skip from precision/recall calc
                # But still check honeypot -- invalid response on honeypot = fail
                truth = manifest[sample_id]
                if truth.get("is_honeypot", False):
                    # Invalid response on honeypot is always a failure
                    honeypot_failed = True
                    bt.logging.debug(
                        f"Miner {uid}: honeypot {sample_id} failed (invalid response)"
                    )
                continue

            score = response.anomaly_score

            # Check honeypot
            if not check_honeypot_pass(sample_id, score, manifest):
                honeypot_failed = True
                bt.logging.debug(
                    f"Miner {uid}: honeypot {sample_id} failed "
                    f"(score={score})"
                )

            # Collect for reward computation (include all valid responses)
            anomaly_scores.append(score)
            ground_truths.append(manifest[sample_id]["ground_truth_label"])

        # Compute reward
        if honeypot_failed:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: reward=0.0 (honeypot failure)"
            )
        elif len(anomaly_scores) == 0:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: reward=0.0 (no valid responses)"
            )
        else:
            reward = compute_reward(anomaly_scores, ground_truths)
            rewards[miner_idx] = reward
            bt.logging.debug(
                f"Miner {uid}: reward={reward:.4f} "
                f"({len(anomaly_scores)} valid responses)"
            )

    return rewards


def get_hardened_rewards(
    validator,
    miner_uids: list[int],
    responses_by_miner: dict[int, list[tuple[dict, Any]]],
    manifest: dict,
    perturbation_map: dict[int, dict[str, list[str]]],
    stability_weight: float = 0.15,
) -> np.ndarray:
    """Compute per-miner rewards with stability bonus from hardened forward pass.

    For each miner:
    1. Separate responses into regular, honeypot, adversarial, perturbation
    2. Validate responses and check honeypots
    3. Compute base reward from non-perturbation, non-adversarial responses
    4. Compute stability bonus from original + perturbation variant scores
    5. Blend: (1 - stability_weight) * base + stability_weight * stability_bonus
    6. Honeypot failure zeroes entire round (overrides stability bonus)

    Args:
        validator: The validator neuron instance.
        miner_uids: List of miner UIDs being scored.
        responses_by_miner: Dict mapping uid -> [(sample, response), ...].
        manifest: Ground truth manifest dict keyed by sample_id.
        perturbation_map: Dict mapping uid -> {original_id: [perturbed_ids]}.
        stability_weight: Weight for stability bonus (default 0.15 = 15%).

    Returns:
        numpy array of reward values (float32), one per miner.
    """
    n_miners = len(miner_uids)
    rewards = np.zeros(n_miners, dtype=np.float32)

    for miner_idx, uid in enumerate(miner_uids):
        sample_responses = responses_by_miner.get(uid, [])
        uid_perturbation_map = perturbation_map.get(uid, {})

        # Collect perturbed sample IDs for this miner
        perturbed_ids = set()
        for _orig_id, pert_ids in uid_perturbation_map.items():
            perturbed_ids.update(pert_ids)

        anomaly_scores = []
        ground_truths = []
        honeypot_failed = False

        # Track scores for stability computation: original_id -> [scores]
        stability_scores: dict[str, list[float]] = {}
        for orig_id in uid_perturbation_map:
            stability_scores[orig_id] = []

        for sample, response in sample_responses:
            sample_id = sample.get("id", "")
            is_adversarial = sample.get("_is_adversarial", False)
            is_perturbation = sample.get("_is_perturbation", False)
            original_id = sample.get("_original_id")

            # Validate response structure
            is_valid, reason = validate_response(response)

            if not is_valid:
                bt.logging.debug(
                    f"Miner {uid} sample {sample_id}: invalid response ({reason})"
                )
                if sample_id in manifest:
                    truth = manifest[sample_id]
                    if truth.get("is_honeypot", False):
                        honeypot_failed = True
                        bt.logging.debug(
                            f"Miner {uid}: honeypot {sample_id} failed (invalid response)"
                        )
                continue

            score = response.anomaly_score

            # Track perturbation stability scores
            if is_perturbation and original_id in stability_scores:
                stability_scores[original_id].append(score)
                continue  # Do not include perturbation scores in base reward

            # Track original sample scores for stability
            if sample_id in stability_scores:
                stability_scores[sample_id].append(score)

            # Skip adversarial samples from base reward computation
            if is_adversarial:
                continue

            # Check honeypot
            if sample_id in manifest:
                if not check_honeypot_pass(sample_id, score, manifest):
                    honeypot_failed = True
                    bt.logging.debug(
                        f"Miner {uid}: honeypot {sample_id} failed "
                        f"(score={score})"
                    )

                anomaly_scores.append(score)
                ground_truths.append(manifest[sample_id]["ground_truth_label"])

        # Compute base reward
        if honeypot_failed:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: reward=0.0 (honeypot failure)"
            )
        elif len(anomaly_scores) == 0:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: reward=0.0 (no valid responses)"
            )
        else:
            base_reward = compute_reward(anomaly_scores, ground_truths)

            # Compute stability bonus
            all_stability_scores = []
            for _orig_id, scores_list in stability_scores.items():
                if len(scores_list) >= 2:
                    all_stability_scores.extend(scores_list)

            if all_stability_scores and len(all_stability_scores) >= 2:
                stability_bonus = compute_stability_bonus(all_stability_scores)
            else:
                stability_bonus = 1.0  # Default: no penalty

            final_reward = (
                (1 - stability_weight) * base_reward
                + stability_weight * stability_bonus
            )
            rewards[miner_idx] = float(np.clip(final_reward, 0.0, 1.0))

            bt.logging.debug(
                f"Miner {uid}: base={base_reward:.4f} "
                f"stability={stability_bonus:.4f} "
                f"final={rewards[miner_idx]:.4f} "
                f"({len(anomaly_scores)} valid responses)"
            )

    return rewards


def compute_diversity_bonus(
    uid: int,
    score_vectors: dict[int, np.ndarray],
    threshold: float = 0.95,
    min_history: int = 10,
    min_miners: int = 8,
) -> float:
    """Compute diversity bonus for a single miner based on score vector uniqueness.

    Inverts the diversity penalty concept: miners with unique detection
    approaches (low max cosine similarity) receive higher bonuses.

    Args:
        uid: The miner UID to compute bonus for.
        score_vectors: Dict mapping miner UID -> numpy array of recent scores.
        threshold: Cosine similarity threshold (unused in bonus, kept for API
            compatibility with diversity penalty parameters).
        min_history: Minimum score vector length required.
        min_miners: Minimum population size for diversity analysis.

    Returns:
        Diversity bonus in [0.0, 1.0]:
        - 1.0 = completely unique detector
        - 0.0 = identical to another miner
        - 0.5 = neutral default (insufficient data)
    """
    # Guard: small populations -> neutral default
    if len(score_vectors) < min_miners:
        return 0.5

    # Check if this UID has sufficient history
    if uid not in score_vectors or len(score_vectors[uid]) < min_history:
        return 0.5

    uid_vec = score_vectors[uid]
    uid_norm = np.linalg.norm(uid_vec)
    if uid_norm == 0:
        return 0.5

    # Find max cosine similarity against all other eligible miners
    max_similarity = 0.0
    has_comparison = False

    for other_uid, other_vec in score_vectors.items():
        if other_uid == uid:
            continue
        if len(other_vec) < min_history:
            continue
        other_norm = np.linalg.norm(other_vec)
        if other_norm == 0:
            continue

        similarity = float(np.dot(uid_vec, other_vec) / (uid_norm * other_norm))
        max_similarity = max(max_similarity, similarity)
        has_comparison = True

    if not has_comparison:
        return 0.5

    # Bonus = 1 - max_similarity, clamped to [0, 1]
    return float(np.clip(1.0 - max_similarity, 0.0, 1.0))


def get_composite_rewards(
    validator,
    miner_uids: list[int],
    responses_by_miner: dict[int, list[tuple[dict, Any]]],
    manifest: dict,
    perturbation_map: dict[int, dict[str, list[str]]],
    confidence_history: dict[int, list[tuple[list[float], list[int]]]],
    score_vectors: dict[int, np.ndarray],
    base_weight: float = BASE_WEIGHT,
    calibration_weight: float = CALIBRATION_WEIGHT,
    robustness_weight: float = ROBUSTNESS_WEIGHT,
    diversity_weight: float = DIVERSITY_WEIGHT,
) -> np.ndarray:
    """Compute per-miner rewards using 4-component composite formula.

    For each miner:
    1. Compute base reward (precision-first, same as get_hardened_rewards)
    2. Compute calibration bonus from confidence history
    3. Compute robustness (stability) bonus from perturbation variants
    4. Compute diversity bonus from score vector uniqueness
    5. Blend: base_weight*base + cal_weight*cal + rob_weight*rob + div_weight*div
    6. Honeypot failure zeroes entire round (overrides all bonuses)

    Args:
        validator: The validator neuron instance.
        miner_uids: List of miner UIDs being scored.
        responses_by_miner: Dict mapping uid -> [(sample, response), ...].
        manifest: Ground truth manifest dict keyed by sample_id.
        perturbation_map: Dict mapping uid -> {original_id: [perturbed_ids]}.
        confidence_history: Dict mapping uid -> [(confidences, accuracies), ...]
            sliding window of past rounds' confidence/accuracy pairs.
        score_vectors: Dict mapping uid -> numpy array of recent anomaly
            scores for diversity computation.
        base_weight: Weight for base reward component (default 0.70).
        calibration_weight: Weight for calibration bonus (default 0.10).
        robustness_weight: Weight for robustness/stability bonus (default 0.10).
        diversity_weight: Weight for diversity bonus (default 0.10).

    Returns:
        numpy array of reward values (float32), one per miner.
    """
    n_miners = len(miner_uids)
    rewards = np.zeros(n_miners, dtype=np.float32)

    for miner_idx, uid in enumerate(miner_uids):
        sample_responses = responses_by_miner.get(uid, [])
        uid_perturbation_map = perturbation_map.get(uid, {})

        anomaly_scores = []
        ground_truths = []
        honeypot_failed = False

        # Track scores for stability computation: original_id -> [scores]
        stability_scores: dict[str, list[float]] = {}
        for orig_id in uid_perturbation_map:
            stability_scores[orig_id] = []

        for sample, response in sample_responses:
            sample_id = sample.get("id", "")
            is_adversarial = sample.get("_is_adversarial", False)
            is_perturbation = sample.get("_is_perturbation", False)
            original_id = sample.get("_original_id")

            # Validate response structure
            is_valid, reason = validate_response(response)

            if not is_valid:
                bt.logging.debug(
                    f"Miner {uid} sample {sample_id}: invalid response ({reason})"
                )
                if sample_id in manifest:
                    truth = manifest[sample_id]
                    if truth.get("is_honeypot", False):
                        honeypot_failed = True
                        bt.logging.debug(
                            f"Miner {uid}: honeypot {sample_id} failed (invalid response)"
                        )
                continue

            score = response.anomaly_score

            # Track perturbation stability scores
            if is_perturbation and original_id in stability_scores:
                stability_scores[original_id].append(score)
                continue  # Do not include perturbation scores in base reward

            # Track original sample scores for stability
            if sample_id in stability_scores:
                stability_scores[sample_id].append(score)

            # Skip adversarial samples from base reward computation
            if is_adversarial:
                continue

            # Check honeypot
            if sample_id in manifest:
                if not check_honeypot_pass(sample_id, score, manifest):
                    honeypot_failed = True
                    bt.logging.debug(
                        f"Miner {uid}: honeypot {sample_id} failed "
                        f"(score={score})"
                    )

                anomaly_scores.append(score)
                ground_truths.append(manifest[sample_id]["ground_truth_label"])

        # Compute composite reward
        if honeypot_failed:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: composite_reward=0.0 (honeypot failure)"
            )
        elif len(anomaly_scores) == 0:
            rewards[miner_idx] = 0.0
            bt.logging.debug(
                f"Miner {uid}: composite_reward=0.0 (no valid responses)"
            )
        else:
            # 1. Base reward (precision-first)
            base_reward = compute_reward(anomaly_scores, ground_truths)

            # 2. Calibration bonus from confidence history
            uid_conf_history = confidence_history.get(uid, [])
            if uid_conf_history:
                # Flatten all confidence/accuracy pairs from sliding window
                all_confidences: list[float] = []
                all_accuracies: list[int] = []
                for confs, accs in uid_conf_history:
                    all_confidences.extend(confs)
                    all_accuracies.extend(accs)
                cal_bonus = compute_calibration_bonus(all_confidences, all_accuracies)
            else:
                cal_bonus = 1.0  # No history -> no penalty

            # 3. Robustness (stability) bonus from perturbation variants
            all_stability_scores: list[float] = []
            for _orig_id, scores_list in stability_scores.items():
                if len(scores_list) >= 2:
                    all_stability_scores.extend(scores_list)

            if all_stability_scores and len(all_stability_scores) >= 2:
                rob_bonus = compute_stability_bonus(all_stability_scores)
            else:
                rob_bonus = 1.0  # Default: no penalty

            # 4. Diversity bonus from score vector uniqueness
            div_bonus = compute_diversity_bonus(uid, score_vectors)

            # Composite blend
            final_reward = (
                base_weight * base_reward
                + calibration_weight * cal_bonus
                + robustness_weight * rob_bonus
                + diversity_weight * div_bonus
            )
            rewards[miner_idx] = float(np.clip(final_reward, 0.0, 1.0))

            bt.logging.debug(
                f"Miner {uid}: base={base_reward:.4f} "
                f"calibration={cal_bonus:.4f} "
                f"robustness={rob_bonus:.4f} "
                f"diversity={div_bonus:.4f} "
                f"composite={rewards[miner_idx]:.4f} "
                f"({len(anomaly_scores)} valid responses)"
            )

    return rewards
