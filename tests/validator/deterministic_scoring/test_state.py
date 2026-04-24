"""Tests for frozen round-score dataclasses (DETSCORE-01)."""

from __future__ import annotations

import dataclasses

import pytest

from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)


# ---- FrozenRoundScore construction & immutability ---------------------------


def test_frozen_round_score_constructs():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="5F" + "a" * 46)
    assert s.uid == 1
    assert s.raw_reward == 0.5
    assert s.ema_score == 0.3
    assert s.hotkey.startswith("5F")


def test_frozen_round_score_rejects_mutation():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.raw_reward = 0.9  # type: ignore[misc]


def test_frozen_round_score_rejects_object_setattr():
    # slots=True means object.__setattr__ on a non-declared attr raises
    # AttributeError. This guards against attribute injection.
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    with pytest.raises(AttributeError):
        object.__setattr__(s, "new_attr", 1)


def test_frozen_round_score_is_hashable():
    s1 = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    s2 = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    # Goes into a set without error and dedupes equal instances.
    assert len({s1, s2}) == 1


def test_frozen_round_score_rejects_numpy_scalar_like():
    # We don't import numpy here (keeps the deterministic_scoring package
    # numpy-free). Simulate what np.float32 looks like from the dataclass's
    # perspective: a subclass of float.
    class NumpyFloatLike(float):
        pass

    bad = NumpyFloatLike(0.5)
    # type(bad) is NumpyFloatLike, not float -> must be rejected.
    with pytest.raises(TypeError, match="must be Python float"):
        FrozenRoundScore(uid=1, raw_reward=bad, ema_score=0.3, hotkey="hk")


def test_frozen_round_score_rejects_int_reward():
    with pytest.raises(TypeError, match="must be Python float"):
        FrozenRoundScore(uid=1, raw_reward=1, ema_score=0.3, hotkey="hk")  # type: ignore[arg-type]


def test_frozen_round_score_rejects_negative_uid():
    with pytest.raises(ValueError, match="uid must be >= 0"):
        FrozenRoundScore(uid=-1, raw_reward=0.5, ema_score=0.3, hotkey="hk")


def test_frozen_round_score_rejects_empty_hotkey():
    with pytest.raises(ValueError, match="hotkey must be non-empty"):
        FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="")


# ---- FrozenRoundRecord construction & immutability --------------------------


_GENESIS = "0" * 64


def test_frozen_round_record_empty_scores_ok():
    r = FrozenRoundRecord(
        round_index=0, ema_alpha=0.1, scores=(), prev_hash=_GENESIS
    )
    assert r.scores == ()


def test_frozen_round_record_scores_is_tuple():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    r = FrozenRoundRecord(
        round_index=0, ema_alpha=0.1, scores=(s,), prev_hash=_GENESIS
    )
    assert isinstance(r.scores, tuple)
    with pytest.raises(TypeError):
        # tuple itself is immutable
        r.scores[0] = s  # type: ignore[index]


def test_frozen_round_record_rejects_list_scores():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    with pytest.raises(TypeError, match="must be tuple"):
        FrozenRoundRecord(
            round_index=0, ema_alpha=0.1, scores=[s], prev_hash=_GENESIS  # type: ignore[arg-type]
        )


def test_frozen_round_record_rejects_mutation():
    r = FrozenRoundRecord(
        round_index=0, ema_alpha=0.1, scores=(), prev_hash=_GENESIS
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.round_index = 1  # type: ignore[misc]


def test_frozen_round_record_rejects_bad_prev_hash_length():
    with pytest.raises(ValueError, match="64 hex chars"):
        FrozenRoundRecord(
            round_index=0, ema_alpha=0.1, scores=(), prev_hash="abc"
        )


def test_frozen_round_record_rejects_non_hex_prev_hash():
    bad = "z" * 64
    with pytest.raises(ValueError, match="lowercase hex"):
        FrozenRoundRecord(
            round_index=0, ema_alpha=0.1, scores=(), prev_hash=bad
        )


def test_frozen_round_record_rejects_uppercase_hex():
    bad = "A" * 64  # uppercase -- not in our lowercase alphabet
    with pytest.raises(ValueError, match="lowercase hex"):
        FrozenRoundRecord(
            round_index=0, ema_alpha=0.1, scores=(), prev_hash=bad
        )


def test_frozen_round_record_rejects_alpha_out_of_range():
    with pytest.raises(ValueError, match=r"ema_alpha must be in \[0.0, 1.0\]"):
        FrozenRoundRecord(
            round_index=0, ema_alpha=1.5, scores=(), prev_hash=_GENESIS
        )


def test_frozen_round_record_rejects_negative_round_index():
    with pytest.raises(ValueError, match="round_index must be >= 0"):
        FrozenRoundRecord(
            round_index=-1, ema_alpha=0.1, scores=(), prev_hash=_GENESIS
        )


def test_frozen_round_record_rejects_non_score_in_scores():
    with pytest.raises(TypeError, match="must be FrozenRoundScore"):
        FrozenRoundRecord(
            round_index=0,
            ema_alpha=0.1,
            scores=("not a score",),  # type: ignore[arg-type]
            prev_hash=_GENESIS,
        )
