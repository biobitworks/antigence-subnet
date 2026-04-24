"""
Hardened forward pass with per-miner challenge selection, perturbation
stability testing, composite rewards, and confidence tracking.

Anti-cheating integration:
- CHEAT-01: Per-miner unique challenge subsets (get_miner_challenge)
- CHEAT-03: Adversarial synthetic sample injection
- CHEAT-04: Perturbation stability bonus (in composite)
- CHEAT-07: Diversity bonus (in composite, replaces separate penalties)

Reward integration (Phase 6):
- RWRD-04: Calibration bonus from confidence history
- RWRD-05: Robustness bonus from perturbation stability
- RWRD-06: Diversity bonus from score vector uniqueness

Draws a 20-sample round pool, selects 10 unique per miner (8 challenge
+ 1 adversarial + 1 perturbation variant), queries miners individually,
computes composite rewards (base + calibration + robustness + diversity),
then applies EMA update.
"""

import secrets
import time

import bittensor as bt
import numpy as np

from antigence_subnet.base.validator import (
    VALIDATOR_POLICY_OPERATOR_MULTIBAND,
    resolve_validator_policy_config,
)
from antigence_subnet.protocol import VerificationSynapse
from antigence_subnet.utils.uids import get_random_uids
from antigence_subnet.validator.challenge import (
    detect_dataset_refresh,
    get_miner_challenge,
    inject_adversarial_samples,
)
from antigence_subnet.validator.collusion import CollusionConfig, CollusionDetector
from antigence_subnet.validator.perturbation import generate_perturbation_variants
from antigence_subnet.validator.reward import (
    get_composite_rewards,
)
from antigence_subnet.validator.rotation import ChallengeRotation
from antigence_subnet.validator.scoring import build_validator_scorer
from antigence_subnet.validator.validation import validate_response

# Optional metrics/health imports -- gracefully degrade if unavailable
try:
    from antigence_subnet.api.metrics import get_collector as _get_metrics_collector
except ImportError:
    _get_metrics_collector = None  # type: ignore[assignment]

try:
    from antigence_subnet.api.health import record_forward_complete as _record_forward_complete
except ImportError:
    _record_forward_complete = None  # type: ignore[assignment]


# Maximum score history entries per miner (rolling window)
MAX_SCORE_HISTORY = 50

# Maximum confidence history entries per miner (sliding window for ECE)
MAX_CONFIDENCE_HISTORY = 50


def _apply_validator_policy(*, anomaly_score: float, confidence: float, policy) -> str:
    """Map continuous detector outputs onto operator policy decisions."""
    if policy.mode == VALIDATOR_POLICY_OPERATOR_MULTIBAND:
        if confidence < policy.min_confidence:
            return "review"
        if anomaly_score >= policy.high_threshold:
            return "block"
        if anomaly_score < policy.low_threshold:
            return "allow"
        return "review"

    if confidence < policy.min_confidence:
        return "allow"
    if anomaly_score >= policy.high_threshold:
        return "block"
    return "allow"


def _score_non_exact_mode(
    validator,
    *,
    scorer,
    miner_uids: list[int],
    responses_by_miner: dict[int, list[tuple[dict, object]]],
    manifest: dict,
) -> np.ndarray:
    """Score miners individually for non-exact modes over their sampled payloads."""
    rewards = np.zeros(len(miner_uids), dtype=np.float32)

    for miner_idx, uid in enumerate(miner_uids):
        sample_responses = responses_by_miner.get(uid, [])
        responses_by_sample = {}
        miner_manifest = {}
        for sample, response in sample_responses:
            sample_id = sample.get("id", "")
            if sample.get("_is_adversarial", False) or sample.get("_is_perturbation", False):
                continue
            if sample_id not in manifest:
                continue
            responses_by_sample[sample_id] = [response]
            miner_manifest[sample_id] = {
                **manifest[sample_id],
                "prompt": sample.get("prompt", manifest[sample_id].get("prompt")),
                "output": sample.get("output", manifest[sample_id].get("output")),
                "domain": sample.get("domain", manifest[sample_id].get("domain")),
            }

        if not responses_by_sample:
            continue

        result = scorer.score_round(
            validator=validator,
            miner_uids=[uid],
            responses_by_sample=responses_by_sample,
            manifest=miner_manifest,
        )
        rewards[miner_idx] = float(result.rewards[0])

    return rewards


async def forward(validator) -> None:
    """Run a single hardened validator forward pass with composite rewards.

    1. Select 20-sample round pool (16 regular + 4 honeypots)
    2. Per-miner: select 8 challenge + 1 adversarial + 1 perturbation = 10
    3. Query each miner with their unique sample set
    4. Build confidence tracking from responses
    5. Build score vectors from score history
    6. Compute composite rewards (base + calibration + robustness + diversity)
    7. Update score history with composite rewards
    8. Update EMA scores

    Args:
        validator: The validator neuron instance with evaluation dataset loaded.
    """
    forward_start_time = time.monotonic()
    policy = resolve_validator_policy_config(validator.config)

    miner_uids = get_random_uids(
        validator,
        k=min(validator.config.neuron.sample_size, validator.metagraph.n),
    )

    if not miner_uids:
        bt.logging.warning("No available miner UIDs to query.")
        return

    if validator.evaluation is None:
        bt.logging.error(
            "No evaluation dataset loaded. Cannot run forward pass."
        )
        return

    # --- Dataset refresh detection (OPS-01) ---
    current_version = validator.evaluation.dataset_version
    last_version = getattr(validator, "_last_dataset_version", None)
    if last_version is not None and detect_dataset_refresh(last_version, current_version):
        bt.logging.warning(
            f"Dataset refresh detected | old={last_version} | new={current_version} | "
            "Resetting score_history and confidence_history (stale data from previous dataset)"
        )
        validator.score_history = {}
        validator.confidence_history = {}
    validator._last_dataset_version = current_version

    # --- Initialize challenge rotation tracker (VHARD-01) ---
    if not hasattr(validator, "challenge_rotation"):
        rotation_cfg = getattr(validator.config, "rotation", None)
        rotation_enabled = getattr(rotation_cfg, "enabled", True) if rotation_cfg else True
        rotation_window = int(
            getattr(rotation_cfg, "window", 10) if rotation_cfg else 10
        )
        if rotation_enabled:
            validator.challenge_rotation = ChallengeRotation(
                rotation_window=rotation_window
            )
        else:
            validator.challenge_rotation = None

    # --- Per-round entropy seed for anti-memorization (MAIN-04) ---
    round_entropy = secrets.token_bytes(32)
    entropy_int = int.from_bytes(round_entropy[:8], "big")
    bt.logging.info(
        f"Round entropy generated | step={validator.step} | "
        f"seed_hex={round_entropy.hex()[:16]}..."
    )

    # --- Stage 1: Round pool selection (20 samples: 16 regular + 4 honeypots) ---
    round_pool = validator.evaluation.get_round_samples(
        round_num=validator.step,
        n=20,
        n_honeypots=4,
    )
    scoring_cfg = getattr(validator.config, "scoring", None)
    # Optional best-effort hint only; miners may ignore this without affecting
    # the validator's default unseeded behavior.
    request_seed = (
        int(scoring_cfg.seed)
        if scoring_cfg is not None and hasattr(scoring_cfg, "seed") and scoring_cfg.seed is not None
        else None
    )

    # --- Stage 2: Per-miner challenge + perturbation ---
    # responses_by_miner: uid -> [(sample, response), ...]
    responses_by_miner: dict[int, list[tuple[dict, object]]] = {}
    # Track perturbation mapping: uid -> {original_sample_id: [perturbed_sample_ids]}
    perturbation_map: dict[int, dict[str, list[str]]] = {}

    for uid in miner_uids:
        miner_hotkey = validator.metagraph.hotkeys[uid]

        try:
            # Get per-miner exclusion set from rotation tracker (VHARD-01)
            rotation_excluded: set[str] = set()
            if validator.challenge_rotation is not None:
                rotation_excluded = validator.challenge_rotation.get_excluded(
                    miner_hotkey
                )

            # Select 8 unique samples for this miner from the pool
            miner_samples = get_miner_challenge(
                samples=round_pool,
                miner_hotkey=miner_hotkey,
                round_num=validator.step,
                n=8,
                entropy_seed=round_entropy,
                excluded_ids=rotation_excluded if rotation_excluded else None,
            )

            # Inject 1 adversarial sample (appended at end)
            miner_samples = inject_adversarial_samples(
                miner_samples,
                round_num=validator.step,
                n_adversarial=1,
            )
            # Now have 9 samples: 8 challenge + 1 adversarial

            # Pick one non-honeypot, non-adversarial sample for perturbation
            perturbation_map[uid] = {}
            perturbation_source = None
            perturbation_source_idx = None
            for idx, s in enumerate(miner_samples):
                if s.get("_is_adversarial"):
                    continue
                manifest_entry = validator.evaluation.manifest.get(s["id"], {})
                if manifest_entry.get("is_honeypot", False):
                    continue
                perturbation_source = s
                perturbation_source_idx = idx
                break

            if perturbation_source is not None:
                # Generate up to 3 perturbation variants (entropy randomizes actual count 1-3)
                variants = generate_perturbation_variants(
                    perturbation_source,
                    round_num=validator.step,
                    n_variants=3,
                    entropy_seed=entropy_int,
                )
                # Replace one regular (non-honeypot, non-adversarial) sample
                # to keep total at 10. We replace a sample OTHER than the source.
                replaced = False
                for idx, s in enumerate(miner_samples):
                    if idx == perturbation_source_idx:
                        continue
                    if s.get("_is_adversarial"):
                        continue
                    manifest_entry = validator.evaluation.manifest.get(s["id"], {})
                    if manifest_entry.get("is_honeypot", False):
                        continue
                    # Replace this sample with the perturbation variant
                    miner_samples[idx] = variants[0]
                    replaced = True
                    break

                if not replaced:
                    # If we couldn't replace, append and accept 10 samples
                    miner_samples.append(variants[0])

                perturbation_map[uid][perturbation_source["id"]] = [
                    v["id"] for v in variants
                ]
            else:
                # No suitable source for perturbation; stay at 9 samples
                pass

            # --- Stage 3: Query this miner ---
            miner_responses: list[tuple[dict, object]] = []
            for sample in miner_samples:
                synapse = VerificationSynapse(
                    prompt=sample["prompt"],
                    output=sample["output"],
                    domain=sample["domain"],
                    code=sample.get("code"),
                    context=sample.get("context"),
                    seed=request_seed,
                )
                responses = await validator.dendrite(
                    axons=[validator.metagraph.axons[uid]],
                    synapse=synapse,
                    deserialize=False,
                    timeout=validator.config.neuron.timeout,
                )
                # responses is a list with one element (single miner)
                response = responses[0]
                miner_responses.append((sample, response))

                # Record response/failure for microglia health tracking
                if hasattr(validator, "microglia") and validator.microglia is not None:
                    score = getattr(response, "anomaly_score", None)
                    if score is not None:
                        latency = getattr(
                            getattr(response, "dendrite", None),
                            "process_time",
                            0.0,
                        ) or 0.0
                        validator.microglia.record_response(
                            uid, float(score), float(latency), validator.step
                        )
                    else:
                        validator.microglia.record_failure(uid)

                # Record miner response time for Prometheus metrics
                try:
                    if _get_metrics_collector is not None:
                        resp_latency = getattr(
                            getattr(response, "dendrite", None),
                            "process_time",
                            0.0,
                        ) or 0.0
                        _get_metrics_collector().record_miner_response(
                            uid, float(resp_latency)
                        )
                except Exception:
                    pass

            responses_by_miner[uid] = miner_responses

            # Record which samples this miner saw (VHARD-01)
            if validator.challenge_rotation is not None:
                seen_ids = [
                    s["id"]
                    for s in miner_samples
                    if not s.get("_is_adversarial", False)
                    and not s.get("_is_perturbation", False)
                ]
                validator.challenge_rotation.record(
                    miner_hotkey, validator.step, seen_ids
                )

        except Exception as e:
            bt.logging.warning(
                f"Miner query failed | uid={uid} | hotkey={miner_hotkey} | "
                f"error={type(e).__name__}: {e}"
            )
            # Record failure for microglia health tracking
            if hasattr(validator, "microglia") and validator.microglia is not None:
                validator.microglia.record_failure(uid)
            # Empty response list -- miner produced nothing this round
            responses_by_miner[uid] = []
            continue

    # --- Response validation gate ---
    # Validate each response before scoring; invalid responses get anomaly_score
    # set to None so downstream reward computation treats them as zero.
    for uid in list(responses_by_miner.keys()):
        validated_responses: list[tuple[dict, object]] = []
        for sample, response in responses_by_miner[uid]:
            is_valid, rejection_reason = validate_response(response)
            if not is_valid:
                bt.logging.warning(
                    f"Response rejected | uid={uid} | "
                    f"sample_id={sample.get('id', 'unknown')} | "
                    f"reason={rejection_reason}"
                )
                response.anomaly_score = None
            validated_responses.append((sample, response))
        responses_by_miner[uid] = validated_responses

    # --- Early return when all miners failed ---
    active_miners = [
        uid for uid, resps in responses_by_miner.items() if len(resps) > 0
    ]
    if not active_miners:
        bt.logging.warning(
            "All miners failed in forward pass, no rewards to compute"
        )
        return

    # --- Stage 4: Build confidence tracking for ECE computation ---
    # Ensure confidence_history exists on validator
    if not hasattr(validator, "confidence_history"):
        validator.confidence_history = {}

    for uid in miner_uids:
        sample_responses = responses_by_miner.get(uid, [])
        round_confidences: list[float] = []
        round_accuracies: list[int] = []

        for sample, response in sample_responses:
            # Skip adversarial and perturbation samples from confidence tracking
            if sample.get("_is_adversarial", False):
                continue
            if sample.get("_is_perturbation", False):
                continue

            sample_id = sample.get("id", "")

            # Only track samples with manifest ground truth
            if sample_id not in validator.evaluation.manifest:
                continue

            # Skip if response is invalid (no anomaly_score or confidence)
            anomaly_score = getattr(response, "anomaly_score", None)
            confidence = getattr(response, "confidence", None)
            if anomaly_score is None or confidence is None:
                continue

            # Compute binary accuracy: 1 if prediction matches ground truth
            truth_label = validator.evaluation.manifest[sample_id].get(
                "ground_truth_label", ""
            )
            decision = _apply_validator_policy(
                anomaly_score=float(anomaly_score),
                confidence=float(confidence),
                policy=policy,
            )
            predicted_anomalous = decision == "block"
            actually_anomalous = truth_label == "anomalous"
            accuracy = 1 if predicted_anomalous == actually_anomalous else 0

            round_confidences.append(float(confidence))
            round_accuracies.append(accuracy)

        # Append this round's data to confidence history
        if round_confidences:
            if uid not in validator.confidence_history:
                validator.confidence_history[uid] = []
            validator.confidence_history[uid].append(
                (round_confidences, round_accuracies)
            )
            # Cap sliding window
            if len(validator.confidence_history[uid]) > MAX_CONFIDENCE_HISTORY:
                validator.confidence_history[uid] = (
                    validator.confidence_history[uid][-MAX_CONFIDENCE_HISTORY:]
                )

    # --- Stage 5: Build score vectors for diversity computation ---
    # Ensure score_history exists on validator
    if not hasattr(validator, "score_history"):
        validator.score_history = {}

    # Build score vectors from EXISTING history (before this round's rewards)
    score_vectors = {
        uid: np.array(history, dtype=np.float64)
        for uid, history in validator.score_history.items()
    }

    # --- Stage 6: Compute rewards according to configured scoring mode ---
    scoring_mode = getattr(scoring_cfg, "mode", "exact") if scoring_cfg else "exact"
    scoring_repeats = (
        int(getattr(scoring_cfg, "repeats", 3)) if scoring_cfg is not None else 3
    )
    scoring_ci_level = (
        float(getattr(scoring_cfg, "ci_level", 0.95))
        if scoring_cfg is not None
        else 0.95
    )

    if scoring_mode == "exact":
        # Read reward weights from validator config (Phase 26 - MAIN-06)
        reward_cfg = getattr(validator.config, "reward", None)
        reward_kwargs = {}
        if reward_cfg is not None:
            reward_kwargs = {
                "base_weight": float(getattr(reward_cfg, "base_weight", 0.70)),
                "calibration_weight": float(
                    getattr(reward_cfg, "calibration_weight", 0.10)
                ),
                "robustness_weight": float(
                    getattr(reward_cfg, "robustness_weight", 0.10)
                ),
                "diversity_weight": float(
                    getattr(reward_cfg, "diversity_weight", 0.10)
                ),
            }

        rewards = get_composite_rewards(
            validator=validator,
            miner_uids=miner_uids,
            responses_by_miner=responses_by_miner,
            manifest=validator.evaluation.manifest,
            perturbation_map=perturbation_map,
            confidence_history=validator.confidence_history,
            score_vectors=score_vectors,
            **reward_kwargs,
        )
    else:
        scorer = build_validator_scorer(
            scoring_mode,
            repeats=scoring_repeats,
            confidence_level=scoring_ci_level,
        )
        rewards = _score_non_exact_mode(
            validator,
            scorer=scorer,
            miner_uids=miner_uids,
            responses_by_miner=responses_by_miner,
            manifest=validator.evaluation.manifest,
        )

    # --- Stage 6b: Collusion detection and penalty (VHARD-05, VHARD-06) ---
    collusion_cfg_data = {}
    validator_cfg = getattr(validator.config, "validator", None)
    if validator_cfg is not None:
        collusion_section = getattr(validator_cfg, "collusion", None)
        if collusion_section is not None:
            collusion_cfg_data = {
                k: getattr(collusion_section, k)
                for k in [
                    "similarity_threshold",
                    "min_group_size",
                    "penalty",
                    "enabled",
                ]
                if hasattr(collusion_section, k)
            }
    collusion_config = CollusionConfig.from_dict(collusion_cfg_data)

    if collusion_config.enabled:
        # Build per-miner sample-keyed scores (handles per-miner unique challenges)
        miner_sample_scores: dict[int, dict[str, float]] = {}
        for uid in miner_uids:
            sample_responses = responses_by_miner.get(uid, [])
            uid_scores: dict[str, float] = {}
            for sample, response in sample_responses:
                score = getattr(response, "anomaly_score", None)
                if score is not None:
                    uid_scores[sample["id"]] = float(score)
            if uid_scores:
                miner_sample_scores[uid] = uid_scores

        detector = CollusionDetector(collusion_config)
        alerts = detector.detect(
            round_num=validator.step,
            miner_uids=list(miner_uids),
            miner_sample_scores=miner_sample_scores,
        )
        if alerts:
            detector.log_alerts(alerts)
            rewards = detector.apply_penalty(rewards, list(miner_uids), alerts)

    # Record per-miner rewards for Prometheus metrics
    try:
        if _get_metrics_collector is not None:
            collector = _get_metrics_collector()
            for i, uid in enumerate(miner_uids):
                collector.record_reward(uid, float(rewards[i]))
    except Exception:
        pass

    # --- Stage 7: Update score history with this round's composite rewards ---
    for i, uid in enumerate(miner_uids):
        if uid not in validator.score_history:
            validator.score_history[uid] = []
        validator.score_history[uid].append(float(rewards[i]))
        # Cap rolling window at MAX_SCORE_HISTORY
        if len(validator.score_history[uid]) > MAX_SCORE_HISTORY:
            validator.score_history[uid] = validator.score_history[uid][
                -MAX_SCORE_HISTORY:
            ]

    # --- Stage 7.5 (v13.1.1, Phase 1103): opt-in audit-chain write ---
    # STATEPOL-02: gated on validator.config.audit.enabled AND a resolved
    # validator.audit_chain_path. When disabled, NO chain.jsonl writes
    # happen. When enabled mid-session (STATEPOL-03), the chain starts
    # cleanly from GENESIS_PREV_HASH on the first append.
    _audit_cfg = getattr(validator.config, "audit", None)
    _audit_enabled = bool(getattr(_audit_cfg, "enabled", False))
    _chain_path_for_audit = getattr(validator, "audit_chain_path", None)
    if _audit_enabled and _chain_path_for_audit:
        try:
            from antigence_subnet.validator.audit_bridge import (
                RewardToAuditAdapter,
                next_round_index,
            )
            from antigence_subnet.validator.deterministic_scoring import (
                AuditChainWriter,
            )

            _chain_writer = AuditChainWriter(_chain_path_for_audit)
            _adapter = RewardToAuditAdapter(_chain_writer, ema_alpha=0.1)
            _idx = next_round_index(_chain_writer)
            _hotkey_fn = getattr(validator, "hotkey_for_uid", None)
            if callable(_hotkey_fn):
                _hotkeys_list = [str(_hotkey_fn(u)) for u in miner_uids]
            else:
                _hotkeys_list = [f"hk-unknown-{u}" for u in miner_uids]
            _adapter.record_round(
                round_index=_idx,
                miner_uids=list(miner_uids),
                rewards=[rewards[i] for i in range(len(miner_uids))],
                hotkeys=_hotkeys_list,
            )
        except Exception as _audit_exc:  # non-blocking by contract
            bt.logging.warning(
                f"audit-chain write failed (non-blocking): {_audit_exc!r}"
            )

    # --- Stage 8: Update EMA scores ---
    validator.update_scores(rewards, miner_uids)

    # --- Stage 8.5 (v13.1.1, Phase 1103): convergence detector hook ---
    # Non-blocking by contract (WIRE-02): any failure logs at WARNING and
    # returns control to the forward loop without raising. Gate requires
    # BOTH audit.enabled AND a resolved audit_chain_path -- if operators
    # set audit.enabled=false, no convergence hook call either (no inputs
    # to feed it). Reads the audit chain written in Stage 7.5 above.
    try:
        from antigence_subnet.validator import convergence_hook

        _chain_path = getattr(validator, "audit_chain_path", None)
        if _audit_enabled and _chain_path is not None:
            _events = convergence_hook.run_convergence_checks(_chain_path)
            if _events:
                bt.logging.info(
                    f"convergence_hook: emitted {len(_events)} event(s) "
                    f"| step={validator.step}"
                )
    except Exception as _hook_exc:  # non-blocking by contract (WIRE-02)
        bt.logging.warning(
            f"convergence_hook failed (non-blocking): {_hook_exc!r}"
        )

    # Record forward pass duration and completion for monitoring
    forward_duration = time.monotonic() - forward_start_time
    try:
        if _get_metrics_collector is not None:
            _get_metrics_collector().record_forward_pass(forward_duration)
    except Exception:
        pass
    try:
        if _record_forward_complete is not None:
            _record_forward_complete()
    except Exception:
        pass

    bt.logging.info(
        f"Composite rewards complete | Step {validator.step} | "
        f"Queried {len(miner_uids)} miners | Per-miner unique challenges | "
        f"Mean reward: {rewards.mean():.4f} | Duration: {forward_duration:.2f}s | "
        f"entropy_seed={round_entropy.hex()[:16]}"
    )
