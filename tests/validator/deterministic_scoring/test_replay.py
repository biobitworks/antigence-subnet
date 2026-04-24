"""Tests for the deterministic replay harness (DETSCORE-04)."""

from __future__ import annotations

import pathlib

import pytest

from antigence_subnet.validator.deterministic_scoring.chain import (
    GENESIS_PREV_HASH,
    AuditChainWriter,
    hash_record,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.replay import (
    ReplayResult,
    RoundInputs,
    replay_chain,
)
from antigence_subnet.validator.deterministic_scoring.serialization import (
    from_canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
)


def _three_round_inputs() -> list[RoundInputs]:
    return [
        RoundInputs(
            round_index=0,
            ema_alpha=0.1,
            raw_rewards=(
                (0, 1.0, "hk0"),
                (1, 0.5, "hk1"),
                (2, 0.2, "hk2"),
            ),
        ),
        RoundInputs(
            round_index=1,
            ema_alpha=0.1,
            raw_rewards=(
                (0, 0.9, "hk0"),
                (1, 0.6, "hk1"),
                (2, 0.3, "hk2"),
            ),
        ),
        RoundInputs(
            round_index=2,
            ema_alpha=0.1,
            raw_rewards=(
                (0, 0.8, "hk0"),
                (1, 0.7, "hk1"),
                (2, 0.4, "hk2"),
            ),
        ),
    ]


# ---- Empty + basic structural invariants -----------------------------------


def test_empty_replay():
    result = replay_chain([])
    assert isinstance(result, ReplayResult)
    assert result.records == ()
    assert result.hashes == ()


def test_three_round_replay_prev_hash_chain():
    rounds = _three_round_inputs()
    result = replay_chain(rounds)
    assert len(result.records) == 3
    assert len(result.hashes) == 3
    assert result.records[0].prev_hash == GENESIS_PREV_HASH
    for i in range(1, 3):
        assert result.records[i].prev_hash == result.hashes[i - 1]
    # Round indices are contiguous.
    for i, rec in enumerate(result.records):
        assert rec.round_index == i
    # Each hash matches hash_record(record).
    for rec, h in zip(result.records, result.hashes, strict=True):
        assert h == hash_record(rec)


# ---- EMA formula parity with base/validator.py update_scores ---------------


def test_ema_formula_matches_base_validator():
    """alpha * raw_reward + (1 - alpha) * prev_ema, with new UID starting at 0.0."""
    rounds = [
        RoundInputs(
            round_index=0,
            ema_alpha=0.1,
            raw_rewards=((1, 1.0, "hk1"),),
        )
    ]
    result = replay_chain(rounds, initial_ema={1: 0.0})
    assert result.records[0].scores[0].ema_score == 0.1  # 0.1*1.0 + 0.9*0.0


def test_ema_formula_multi_round_propagation():
    rounds = [
        RoundInputs(round_index=0, ema_alpha=0.5, raw_rewards=((0, 1.0, "a"),)),
        RoundInputs(round_index=1, ema_alpha=0.5, raw_rewards=((0, 0.0, "a"),)),
    ]
    result = replay_chain(rounds)
    # Round 0: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert result.records[0].scores[0].ema_score == 0.5
    # Round 1: 0.5 * 0.0 + 0.5 * 0.5 = 0.25
    assert result.records[1].scores[0].ema_score == 0.25


def test_ema_initial_map_used():
    rounds = [
        RoundInputs(round_index=0, ema_alpha=0.1, raw_rewards=((7, 0.5, "h"),)),
    ]
    result = replay_chain(rounds, initial_ema={7: 1.0})
    # 0.1 * 0.5 + 0.9 * 1.0 = 0.05 + 0.9 = 0.95
    assert result.records[0].scores[0].ema_score == pytest.approx(0.95, abs=1e-15)


# ---- Byte-identity across runs (DETSCORE-04 core) ---------------------------


def test_replay_byte_identity_across_runs():
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.1,
            raw_rewards=tuple(
                (uid, 0.1 * (i + 1) + 0.01 * uid, f"hk{uid}") for uid in range(4)
            ),
        )
        for i in range(5)
    ]
    a = replay_chain(rounds)
    b = replay_chain(rounds)
    assert a.records == b.records
    assert a.hashes == b.hashes


# ---- Chain-writer parity ----------------------------------------------------


def test_replay_output_verifies_via_chain_writer(tmp_path: pathlib.Path):
    """Writing the replay output through AuditChainWriter yields a valid chain."""
    rounds = _three_round_inputs()
    result = replay_chain(rounds)
    path = tmp_path / "chain.jsonl"
    writer = AuditChainWriter(path)
    for rec in result.records:
        writer.append(rec)
    # Should not raise.
    verify_chain(path)


# ---- Log-to-replay round-trip (closes the loop) -----------------------------


def test_log_to_replay_round_trip(tmp_path: pathlib.Path):
    """A recorded chain replays back to byte-identical records and hashes."""
    # 1. Build a 5-round chain via replay_chain + AuditChainWriter.
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.25,
            raw_rewards=tuple(
                (uid, 0.05 * (i + 1) + 0.01 * uid, f"hk{uid}") for uid in range(3)
            ),
        )
        for i in range(5)
    ]
    original = replay_chain(rounds)
    path = tmp_path / "chain.jsonl"
    writer = AuditChainWriter(path)
    for rec in original.records:
        writer.append(rec)
    verify_chain(path)

    # 2. Read JSONL back, reconstruct RoundInputs from each record.
    lines = path.read_bytes().splitlines()
    reconstructed_inputs: list[RoundInputs] = []
    for line in lines:
        rec = from_canonical_json(line, FrozenRoundRecord)
        raw_rewards = tuple(
            (s.uid, s.raw_reward, s.hotkey) for s in rec.scores
        )
        reconstructed_inputs.append(
            RoundInputs(
                round_index=rec.round_index,
                ema_alpha=rec.ema_alpha,
                raw_rewards=raw_rewards,
            )
        )

    # 3. Re-replay and assert byte-identity.
    replayed = replay_chain(reconstructed_inputs)
    assert replayed.records == original.records
    assert replayed.hashes == original.hashes


# ---- NaN/Inf rejection ------------------------------------------------------


def test_replay_rejects_nan_reward():
    rounds = [
        RoundInputs(
            round_index=0,
            ema_alpha=0.1,
            raw_rewards=((0, float("nan"), "hk"),),
        )
    ]
    with pytest.raises(ValueError, match="non-finite"):
        replay_chain(rounds)


def test_replay_rejects_positive_inf_reward():
    rounds = [
        RoundInputs(
            round_index=0,
            ema_alpha=0.1,
            raw_rewards=((0, float("inf"), "hk"),),
        )
    ]
    with pytest.raises(ValueError, match="non-finite"):
        replay_chain(rounds)


def test_replay_rejects_negative_inf_reward():
    rounds = [
        RoundInputs(
            round_index=0,
            ema_alpha=0.1,
            raw_rewards=((0, -float("inf"), "hk"),),
        )
    ]
    with pytest.raises(ValueError, match="non-finite"):
        replay_chain(rounds)


# ---- RoundInputs validation -------------------------------------------------


def test_round_inputs_rejects_list_raw_rewards():
    with pytest.raises(TypeError, match="raw_rewards must be tuple"):
        RoundInputs(
            round_index=0, ema_alpha=0.1, raw_rewards=[(0, 0.1, "hk")]  # type: ignore[arg-type]
        )


def test_round_inputs_rejects_bad_alpha():
    with pytest.raises(ValueError, match=r"ema_alpha must be in \[0.0, 1.0\]"):
        RoundInputs(round_index=0, ema_alpha=2.0, raw_rewards=())


def test_round_inputs_is_frozen():
    ri = RoundInputs(round_index=0, ema_alpha=0.1, raw_rewards=())
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        ri.round_index = 1  # type: ignore[misc]
