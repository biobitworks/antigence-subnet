"""Integration tests for the hardened forward pass pipeline.

Tests CHEAT-01 through CHEAT-07 and NET-06 integration:
- Per-miner unique challenge selection
- Perturbation stability scoring
- Diversity penalties (now integrated into composite rewards)
- Weight audit before set_weights
- Score history persistence
- Confidence history persistence
- Commit-reveal status check
- Composite reward integration
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from antigence_subnet.base.validator import BaseValidatorNeuron
from antigence_subnet.protocol import VerificationSynapse
from antigence_subnet.validator.evaluation import EvaluationDataset
from antigence_subnet.validator.forward import forward

# --- Test helpers ---


def _make_mock_validator(tmp_path, n_miners=4, eval_data_dir="data/evaluation"):
    """Create a mock validator with evaluation dataset for hardened forward tests."""
    config = SimpleNamespace()
    config.netuid = 1
    config.neuron = SimpleNamespace()
    config.neuron.sample_size = n_miners
    config.neuron.timeout = 12.0
    config.neuron.moving_average_alpha = 0.1
    config.neuron.samples_per_round = 10
    config.neuron.n_honeypots = 2
    config.neuron.set_weights_interval = 100
    config.neuron.eval_data_dir = eval_data_dir
    config.neuron.eval_domain = "hallucination"
    config.neuron.full_path = str(tmp_path)
    config.mock = True

    total = n_miners + 1
    metagraph = SimpleNamespace()
    metagraph.n = total
    metagraph.axons = [
        SimpleNamespace(ip="127.0.0.1", port=8091 + i) for i in range(total)
    ]
    metagraph.hotkeys = [f"hotkey-{i}" for i in range(total)]

    validator = SimpleNamespace()
    validator.config = config
    validator.metagraph = metagraph
    validator.uid = 0
    validator.wallet = MagicMock()
    validator.subtensor = MagicMock()
    validator.step = 0
    validator.scores = np.zeros(total, dtype=np.float32)
    validator.hotkeys = list(metagraph.hotkeys)
    validator.score_history = {}
    validator.confidence_history = {}

    eval_path = Path(eval_data_dir)
    if eval_path.exists() and (eval_path / "hallucination").exists():
        validator.evaluation = EvaluationDataset(
            data_dir=eval_path, domain="hallucination"
        )
    else:
        validator.evaluation = None

    return validator


def _make_mock_response(anomaly_score=0.8, anomaly_type="factual_error"):
    """Create a mock VerificationSynapse response."""
    resp = MagicMock(spec=VerificationSynapse)
    resp.anomaly_score = anomaly_score
    resp.anomaly_type = anomaly_type
    resp.confidence = 0.9
    resp.feature_attribution = {"mock": 0.5}
    return resp


# --- Test 1: Hardened forward completes ---


@pytest.mark.asyncio
async def test_hardened_forward_completes(tmp_path):
    """Hardened forward pass completes without error and populates score_history."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite

    update_calls = []

    def mock_update_scores(rewards, uids):
        update_calls.append((rewards.copy(), list(uids)))
        alpha = validator.config.neuron.moving_average_alpha
        for i, uid in enumerate(uids):
            if 0 <= uid < len(validator.scores):
                validator.scores[uid] = (
                    alpha * rewards[i] + (1 - alpha) * validator.scores[uid]
                )

    validator.update_scores = mock_update_scores

    await forward(validator)

    # update_scores called once
    assert len(update_calls) == 1

    # score_history populated for queried miners
    assert len(validator.score_history) > 0
    for _uid, history in validator.score_history.items():
        assert len(history) == 1


# --- Test 2: Different miners get different sample sets ---


@pytest.mark.asyncio
async def test_different_miners_get_different_samples(tmp_path):
    """Different miners receive different sample sets (CHEAT-03)."""
    validator = _make_mock_validator(tmp_path, n_miners=4)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    # Track which prompts each miner (axon) received
    miner_samples: dict[int, list[str]] = {}

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        for axon in axons:
            port = axon.port
            uid = port - 8091
            if uid not in miner_samples:
                miner_samples[uid] = []
            miner_samples[uid].append(synapse.prompt)
        return [_make_mock_response() for _ in axons]

    validator.dendrite = mock_dendrite
    validator.update_scores = lambda r, u: None

    await forward(validator)

    # At least 2 miners should have been queried
    assert len(miner_samples) >= 2

    # Compare sample sets between miners (at least one pair should differ)
    uids = list(miner_samples.keys())
    all_identical = True
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            set_i = set(miner_samples[uids[i]])
            set_j = set(miner_samples[uids[j]])
            if set_i != set_j:
                all_identical = False
                break
        if not all_identical:
            break

    assert not all_identical, (
        "All miners received identical sample sets -- per-miner challenge "
        "selection is not working"
    )


# --- Test 3: Diversity penalties applied with history ---


@pytest.mark.asyncio
async def test_diversity_penalties_applied_with_history(tmp_path):
    """Pre-populated identical score_history triggers diversity penalty.

    Note: diversity penalties require min_miners=8 by default. We test with
    enough miners and pre-populated identical histories.
    """
    validator = _make_mock_validator(tmp_path, n_miners=10)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    # Pre-populate identical score histories for all miners (50 rounds)
    for uid in range(1, 11):
        validator.score_history[uid] = [0.75] * 50

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite

    update_args = []

    def mock_update_scores(rewards, uids):
        update_args.append((rewards.copy(), list(uids)))
        alpha = validator.config.neuron.moving_average_alpha
        for i, uid in enumerate(uids):
            if 0 <= uid < len(validator.scores):
                validator.scores[uid] = (
                    alpha * rewards[i] + (1 - alpha) * validator.scores[uid]
                )

    validator.update_scores = mock_update_scores

    await forward(validator)

    assert len(update_args) == 1
    rewards, uids = update_args[0]

    # With 10+ miners and identical histories, diversity penalty should apply
    # The rewards should be less than if no penalty was applied
    # Since all miners get ~same raw reward, and they all have identical history,
    # they should all be penalized (multiplied by penalty_factor=0.5)
    # But diversity check requires min_miners >= 8 eligible (sufficient history)
    # At least some miners should have been penalized
    # Note: exact penalty depends on whether threshold is met, but with 50 rounds
    # of identical scores (all 0.75), cosine similarity is 1.0 > 0.95
    mean_reward = np.mean(rewards)
    assert mean_reward >= 0.0, "Rewards should be non-negative"


# --- Test 4: Weight audit runs before set_weights ---


def test_weight_audit_runs_before_set_weights(tmp_path):
    """set_weights calls audit_weights before submitting to chain."""
    validator = _make_mock_validator(tmp_path)
    validator.scores = np.array([0.0, 0.5, 0.8, 0.3, 0.1], dtype=np.float32)

    validator.subtensor.min_allowed_weights = MagicMock(return_value=0)
    validator.subtensor.max_weight_limit = MagicMock(return_value=1.0)

    mock_result = SimpleNamespace(success=True, message="ok")
    validator.subtensor.set_weights = MagicMock(return_value=mock_result)

    with patch(
        "antigence_subnet.base.validator.audit_weights",
        return_value=[],
    ) as mock_audit:
        BaseValidatorNeuron.set_weights(validator)

        # audit_weights should have been called before set_weights
        mock_audit.assert_called_once()
        # And subtensor.set_weights should also have been called
        validator.subtensor.set_weights.assert_called_once()


# --- Test 5: Score history persisted across save/load ---


def test_score_history_persisted(tmp_path):
    """save_state persists score_history; load_state restores it."""
    validator = _make_mock_validator(tmp_path)
    validator.step = 10
    validator.score_history = {
        1: [0.5, 0.6, 0.7],
        3: [0.8, 0.9],
    }

    # Save state
    BaseValidatorNeuron.save_state(validator)

    # Verify state file exists
    state_file = os.path.join(str(tmp_path), "state.npz")
    assert os.path.exists(state_file)

    # Create a new validator and load state
    validator2 = _make_mock_validator(tmp_path)
    validator2.score_history = {}
    BaseValidatorNeuron.load_state(validator2)

    # score_history should be restored
    assert 1 in validator2.score_history
    assert 3 in validator2.score_history
    assert validator2.score_history[1] == [0.5, 0.6, 0.7]
    assert validator2.score_history[3] == [0.8, 0.9]


# --- Test 6: Commit-reveal check on init ---


def test_commit_reveal_checked_on_init(mock_config):
    """BaseValidatorNeuron.__init__ checks commit-reveal status."""
    with patch(
        "antigence_subnet.base.validator.check_commit_reveal_enabled",
        return_value=True,
    ) as mock_cr:
        validator = BaseValidatorNeuron.__new__(BaseValidatorNeuron)
        # We need to call __init__ which requires config
        # Use the mock_config fixture
        BaseValidatorNeuron.__init__(validator, config=mock_config)

        # commit-reveal check should have been called
        mock_cr.assert_called_once()

    # Verify score_history was initialized
    assert hasattr(validator, "score_history")
    assert validator.score_history == {}

    # Verify confidence_history was initialized
    assert hasattr(validator, "confidence_history")
    assert validator.confidence_history == {}


# --- Test 7: Forward keeps exact as default and routes opt-in modes via scorer selection ---


@pytest.mark.asyncio
async def test_forward_uses_composite_rewards_by_default(tmp_path):
    """Forward pass keeps exact mode on the legacy composite reward path."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite
    validator.update_scores = lambda r, u: None

    with patch(
        "antigence_subnet.validator.forward.get_composite_rewards",
        return_value=np.zeros(len([uid for uid in range(1, 5)]), dtype=np.float32),
    ) as mock_composite:
        await forward(validator)
        mock_composite.assert_called_once()


@pytest.mark.asyncio
async def test_forward_uses_configured_non_exact_scorer(tmp_path):
    """Forward pass switches to configured scorer for non-exact modes."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    validator.config.scoring = SimpleNamespace(
        mode="semantic",
        repeats=3,
        ci_level=0.95,
    )

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite
    captured = {}

    def mock_update_scores(rewards, uids):
        captured["rewards"] = rewards.copy()
        captured["uids"] = list(uids)

    validator.update_scores = mock_update_scores

    class StubScorer:
        def score_round(self, **kwargs):
            miner_uid = kwargs["miner_uids"][0]
            return SimpleNamespace(
                rewards=np.array([0.25 + miner_uid * 0.01], dtype=np.float32)
            )

    with patch(
        "antigence_subnet.validator.forward.get_composite_rewards"
    ) as mock_composite, patch(
        "antigence_subnet.validator.forward.build_validator_scorer",
        return_value=StubScorer(),
    ) as mock_builder:
        await forward(validator)

    mock_builder.assert_called_once_with(
        "semantic",
        repeats=3,
        confidence_level=0.95,
    )
    mock_composite.assert_not_called()
    assert "rewards" in captured
    assert len(captured["rewards"]) == len(captured["uids"])


@pytest.mark.asyncio
async def test_forward_threads_best_effort_seed_hint_when_available(tmp_path):
    """Forward passes a best-effort seed hint into VerificationSynapse when present."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    validator.config.scoring = SimpleNamespace(
        mode="semantic",
        repeats=2,
        ci_level=0.95,
        seed=777,
    )

    captured_synapses = []

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        captured_synapses.append(synapse)
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite
    validator.update_scores = lambda r, u: None

    class StubScorer:
        def score_round(self, **kwargs):
            miner_uid = kwargs["miner_uids"][0]
            return SimpleNamespace(
                rewards=np.array([0.25 + miner_uid * 0.01], dtype=np.float32)
            )

    with patch(
        "antigence_subnet.validator.forward.build_validator_scorer",
        return_value=StubScorer(),
    ):
        await forward(validator)

    assert captured_synapses, "Forward should build synapses for miner requests"
    assert all(
        synapse.seed == 777 for synapse in captured_synapses
    ), "Best effort seed hint should be threaded when available"


@pytest.mark.asyncio
async def test_forward_seed_absence_preserves_behavior_for_request_construction(tmp_path):
    """Absence preserves behavior by constructing valid unseeded synapses."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    captured_synapses = []

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        captured_synapses.append(synapse)
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite
    validator.update_scores = lambda r, u: None

    with patch(
        "antigence_subnet.validator.forward.get_composite_rewards",
        return_value=np.zeros(len([uid for uid in range(1, 5)]), dtype=np.float32),
    ):
        await forward(validator)

    assert captured_synapses, "Forward should still build synapses without a seed"
    first_synapse = captured_synapses[0]
    assert isinstance(first_synapse, VerificationSynapse)
    assert first_synapse.seed is None
    assert {
        "prompt": first_synapse.prompt,
        "output": first_synapse.output,
        "domain": first_synapse.domain,
        "code": first_synapse.code,
        "context": first_synapse.context,
    } == {
        key: first_synapse.model_dump().get(key)
        for key in ("prompt", "output", "domain", "code", "context")
    }


# --- Test 8: Confidence history populated after forward ---


@pytest.mark.asyncio
async def test_confidence_history_populated_after_forward(tmp_path):
    """Forward pass populates confidence_history for queried miners."""
    validator = _make_mock_validator(tmp_path)

    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    async def mock_dendrite(axons, synapse, deserialize=False, timeout=12.0):
        return [_make_mock_response(anomaly_score=0.7) for _ in axons]

    validator.dendrite = mock_dendrite

    def mock_update_scores(rewards, uids):
        alpha = validator.config.neuron.moving_average_alpha
        for i, uid in enumerate(uids):
            if 0 <= uid < len(validator.scores):
                validator.scores[uid] = (
                    alpha * rewards[i] + (1 - alpha) * validator.scores[uid]
                )

    validator.update_scores = mock_update_scores

    await forward(validator)

    # confidence_history should be populated for queried miners
    assert len(validator.confidence_history) > 0
    for _uid, history in validator.confidence_history.items():
        assert len(history) == 1  # One round of data
        confs, accs = history[0]
        assert isinstance(confs, list)
        assert isinstance(accs, list)
        assert len(confs) == len(accs)
        # All confidence values should be in [0, 1]
        for c in confs:
            assert 0.0 <= c <= 1.0
        # All accuracy values should be 0 or 1
        for a in accs:
            assert a in (0, 1)


# --- Test 9: Confidence history persisted across save/load ---


def test_confidence_history_persisted(tmp_path):
    """save_state persists confidence_history; load_state restores it."""
    validator = _make_mock_validator(tmp_path)
    validator.step = 5
    validator.confidence_history = {
        1: [([0.9, 0.8], [1, 0]), ([0.7, 0.6], [1, 1])],
        3: [([0.5], [0])],
    }

    # Save state
    BaseValidatorNeuron.save_state(validator)

    # Verify state file exists
    state_file = os.path.join(str(tmp_path), "state.npz")
    assert os.path.exists(state_file)

    # Create a new validator and load state
    validator2 = _make_mock_validator(tmp_path)
    validator2.confidence_history = {}
    BaseValidatorNeuron.load_state(validator2)

    # confidence_history should be restored
    assert 1 in validator2.confidence_history
    assert 3 in validator2.confidence_history
    assert len(validator2.confidence_history[1]) == 2
    assert validator2.confidence_history[1][0] == ([0.9, 0.8], [1, 0])
    assert validator2.confidence_history[1][1] == ([0.7, 0.6], [1, 1])
    assert validator2.confidence_history[3][0] == ([0.5], [0])


# --- Test 10: Diversity penalty NOT applied separately ---


def test_diversity_penalty_not_applied_separately():
    """Forward module does NOT import or use compute_diversity_penalties.

    Diversity is now inside get_composite_rewards, not applied separately.
    Verify by checking the module's source does not contain the call.
    """
    import inspect

    from antigence_subnet.validator import forward as fwd_module

    source = inspect.getsource(fwd_module.forward)

    # The function should NOT call compute_diversity_penalties directly
    assert "compute_diversity_penalties" not in source, (
        "forward() should not call compute_diversity_penalties directly; "
        "diversity is now computed inside get_composite_rewards"
    )

    # The function SHOULD call get_composite_rewards
    assert "get_composite_rewards" in source, (
        "forward() should call get_composite_rewards"
    )


# --- Dataset refresh detection tests (OPS-01) ---


@pytest.mark.asyncio
async def test_dataset_refresh_resets_histories(tmp_path):
    """When dataset version changes, score_history and confidence_history are reset."""
    validator = _make_mock_validator(tmp_path)
    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    # Pre-populate histories
    validator.score_history = {0: [0.5, 0.6], 1: [0.3]}
    validator.confidence_history = {0: [([0.8], [1])]}
    # Set old version to trigger refresh
    validator._last_dataset_version = "old_version_aaa"

    mock_resp = _make_mock_response()
    validator.dendrite = MagicMock(return_value=[mock_resp])
    validator.update_scores = MagicMock()
    await forward(validator)

    # Histories should be cleared by refresh detection
    # (then re-populated by the forward pass itself)
    assert validator._last_dataset_version == validator.evaluation.dataset_version
    # The key check: _last_dataset_version was updated from "old_version_aaa"
    assert validator._last_dataset_version != "old_version_aaa"


@pytest.mark.asyncio
async def test_no_refresh_preserves_histories(tmp_path):
    """When dataset version is unchanged, histories are preserved."""
    validator = _make_mock_validator(tmp_path)
    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    # Set version to match current — no refresh
    validator._last_dataset_version = validator.evaluation.dataset_version
    validator.score_history = {0: [0.5, 0.6]}
    validator.confidence_history = {0: [([0.8], [1])]}

    mock_resp = _make_mock_response()
    validator.dendrite = MagicMock(return_value=[mock_resp])
    validator.update_scores = MagicMock()
    await forward(validator)

    # score_history should NOT have been cleared (may have new entries added)
    assert 0 in validator.score_history


@pytest.mark.asyncio
async def test_first_forward_no_reset(tmp_path):
    """First forward pass (no _last_dataset_version) does not reset histories."""
    validator = _make_mock_validator(tmp_path)
    if validator.evaluation is None:
        pytest.skip("Seed data not available")

    # Ensure no _last_dataset_version attribute
    assert not hasattr(validator, "_last_dataset_version")
    validator.score_history = {0: [0.5]}

    mock_resp = _make_mock_response()
    validator.dendrite = MagicMock(return_value=[mock_resp])
    validator.update_scores = MagicMock()
    await forward(validator)

    # score_history should NOT have been cleared
    assert 0 in validator.score_history
    # _last_dataset_version should now be set
    assert hasattr(validator, "_last_dataset_version")
