"""Tests for trajectory extraction over the Phase 1000 audit chain."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from antigence_subnet.validator.deterministic_scoring.chain import (
    AuditChainWriter,
    ChainIntegrityError,
    hash_record,
)
from antigence_subnet.validator.deterministic_scoring.replay import (
    RoundInputs,
    replay_chain,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)
from antigence_subnet.validator.deterministic_scoring.trajectory import (
    TrajectoryWindow,
    extract_trajectories,
)


def _build_chain(path: pathlib.Path, rounds: list[RoundInputs]) -> None:
    """Helper: run replay_chain and write via AuditChainWriter."""
    result = replay_chain(rounds)
    writer = AuditChainWriter(path)
    for rec in result.records:
        writer.append(rec)


def test_extract_empty_chain(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert extract_trajectories(path) == {}


def test_extract_missing_file_raises(tmp_path):
    path = tmp_path / "missing.jsonl"
    with pytest.raises(ChainIntegrityError):
        extract_trajectories(path)


def test_extract_tampered_chain_raises(tmp_path):
    path = tmp_path / "chain.jsonl"
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.1,
            raw_rewards=((0, 0.1 * (i + 1), "hk0"), (1, 0.2 * (i + 1), "hk1")),
        )
        for i in range(3)
    ]
    _build_chain(path, rounds)
    raw = path.read_bytes()
    lines = raw.splitlines()
    # Corrupt the middle line: flip a byte.
    middle = bytearray(lines[1])
    # Replace first digit 0 somewhere with 9 to break canonical bytes.
    for i, b in enumerate(middle):
        if chr(b) == "0":
            middle[i] = ord("9")
            break
    lines[1] = bytes(middle)
    path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ChainIntegrityError):
        extract_trajectories(path)


def test_extract_three_round_three_miner_trajectory(tmp_path):
    path = tmp_path / "chain.jsonl"
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.1,
            raw_rewards=(
                (0, 0.1, "hk0"),
                (1, 0.5, "hk1"),
                (2, 0.9, "hk2"),
            ),
        )
        for i in range(3)
    ]
    _build_chain(path, rounds)
    trajs = extract_trajectories(path)
    assert set(trajs.keys()) == {0, 1, 2}
    for uid in (0, 1, 2):
        win = trajs[uid]
        assert isinstance(win, TrajectoryWindow)
        assert len(win.ema_scores) == 3
        assert win.round_start == 0
        assert win.round_end == 2


def test_extract_window_size_truncates_to_last_N(tmp_path):
    path = tmp_path / "chain.jsonl"
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.1,
            raw_rewards=((0, 0.3, "hk0"), (1, 0.7, "hk1")),
        )
        for i in range(15)
    ]
    _build_chain(path, rounds)
    trajs = extract_trajectories(path, window_size=5)
    for uid in (0, 1):
        win = trajs[uid]
        assert len(win.ema_scores) == 5
        assert win.round_start == 10
        assert win.round_end == 14


def test_extract_uid_appearing_midway(tmp_path):
    path = tmp_path / "chain.jsonl"
    rounds = []
    for i in range(10):
        rewards = [(0, 0.5, "hk0")]
        if i >= 3:
            rewards.append((7, 0.4, "hk7"))
        rounds.append(
            RoundInputs(
                round_index=i, ema_alpha=0.1, raw_rewards=tuple(rewards)
            )
        )
    _build_chain(path, rounds)

    # With window_size >= 7, UID 7 runs from round 3..9 (7 entries).
    trajs = extract_trajectories(path, window_size=20)
    win7 = trajs[7]
    assert win7.round_start == 3
    assert win7.round_end == 9
    assert len(win7.ema_scores) == 7

    # With window_size=5, UID 7 keeps the last 5 entries (rounds 5..9).
    trajs5 = extract_trajectories(path, window_size=5)
    win7b = trajs5[7]
    assert win7b.round_start == 5
    assert win7b.round_end == 9
    assert len(win7b.ema_scores) == 5


def test_ema_values_match_replay(tmp_path):
    path = tmp_path / "chain.jsonl"
    rounds = [
        RoundInputs(
            round_index=i,
            ema_alpha=0.2,
            raw_rewards=((0, 0.1 * (i + 1), "hk0"), (1, 0.2 * (i + 1), "hk1")),
        )
        for i in range(5)
    ]
    result = replay_chain(rounds)
    writer = AuditChainWriter(path)
    for rec in result.records:
        writer.append(rec)
    trajs = extract_trajectories(path)
    # Recover per-uid ema_score from replay result.
    expected: dict[int, list[float]] = {0: [], 1: []}
    for rec in result.records:
        for s in rec.scores:
            expected[s.uid].append(s.ema_score)
    for uid in (0, 1):
        # Byte-identical path: exact float equality.
        assert list(trajs[uid].ema_scores) == expected[uid]


def test_trajectory_window_is_frozen():
    win = TrajectoryWindow(uid=0, round_start=0, round_end=2, ema_scores=(0.1, 0.2, 0.3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        win.uid = 99  # type: ignore[misc]
    assert isinstance(win.ema_scores, tuple)
