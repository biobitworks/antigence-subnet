"""Frozen round-score dataclasses for the deterministic scoring layer.

Both classes use ``@dataclass(frozen=True, slots=True)``. Mutation via normal
attribute assignment raises ``dataclasses.FrozenInstanceError``. Construction
rejects numpy scalars explicitly (via ``type(x) is float`` rather than
``isinstance``) so the serialized tree stays portable and hash-stable across
environments with different numpy versions installed.

This module is deliberately pure-stdlib -- no numpy, no bittensor -- so it
imports cleanly in a minimal test venv.
"""

from __future__ import annotations

from dataclasses import dataclass

_HEX_ALPHABET = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class FrozenRoundScore:
    """Per-miner score entry for a single evaluation round.

    Fields
    ------
    uid : int
        Non-negative miner UID from the metagraph.
    raw_reward : float
        Reward for this round. Must be a Python ``float`` (numpy scalars
        rejected to keep the canonical form portable).
    ema_score : float
        Running exponential-moving-average score *after* this round's update.
    hotkey : str
        Miner hotkey (ss58 address, stable identity across rounds).
    """

    uid: int
    raw_reward: float
    ema_score: float
    hotkey: str

    def __post_init__(self) -> None:
        # type(x) is float enforces exact Python float -- numpy.float32,
        # numpy.float64, bools (subclass of int), etc. are rejected.
        if type(self.uid) is not int or isinstance(self.uid, bool):
            raise TypeError(
                f"FrozenRoundScore.uid must be int, got {type(self.uid).__name__}"
            )
        if self.uid < 0:
            raise ValueError(f"FrozenRoundScore.uid must be >= 0, got {self.uid}")
        if type(self.raw_reward) is not float:
            raise TypeError(
                f"FrozenRoundScore.raw_reward must be Python float, got "
                f"{type(self.raw_reward).__name__}"
            )
        if type(self.ema_score) is not float:
            raise TypeError(
                f"FrozenRoundScore.ema_score must be Python float, got "
                f"{type(self.ema_score).__name__}"
            )
        if type(self.hotkey) is not str:
            raise TypeError(
                f"FrozenRoundScore.hotkey must be str, got "
                f"{type(self.hotkey).__name__}"
            )
        if len(self.hotkey) < 1:
            raise ValueError("FrozenRoundScore.hotkey must be non-empty")


@dataclass(frozen=True, slots=True)
class FrozenRoundRecord:
    """One evaluation round's score state plus chain linkage.

    ``scores`` is a tuple (hashable, immutable container). The record's own
    hash is *not* stored here -- callers compute it with
    :func:`antigence_subnet.validator.deterministic_scoring.chain.hash_record`
    before writing to the JSONL log.
    """

    round_index: int
    ema_alpha: float
    scores: tuple[FrozenRoundScore, ...]
    prev_hash: str

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or isinstance(self.round_index, bool):
            raise TypeError(
                f"FrozenRoundRecord.round_index must be int, got "
                f"{type(self.round_index).__name__}"
            )
        if self.round_index < 0:
            raise ValueError(
                f"FrozenRoundRecord.round_index must be >= 0, got {self.round_index}"
            )
        if type(self.ema_alpha) is not float:
            raise TypeError(
                f"FrozenRoundRecord.ema_alpha must be Python float, got "
                f"{type(self.ema_alpha).__name__}"
            )
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError(
                f"FrozenRoundRecord.ema_alpha must be in [0.0, 1.0], got "
                f"{self.ema_alpha}"
            )
        if type(self.scores) is not tuple:
            raise TypeError(
                f"FrozenRoundRecord.scores must be tuple, got "
                f"{type(self.scores).__name__}"
            )
        for i, s in enumerate(self.scores):
            if not isinstance(s, FrozenRoundScore):
                raise TypeError(
                    f"FrozenRoundRecord.scores[{i}] must be FrozenRoundScore, "
                    f"got {type(s).__name__}"
                )
        if type(self.prev_hash) is not str:
            raise TypeError(
                f"FrozenRoundRecord.prev_hash must be str, got "
                f"{type(self.prev_hash).__name__}"
            )
        if len(self.prev_hash) != 64:
            raise ValueError(
                f"FrozenRoundRecord.prev_hash must be 64 hex chars, got "
                f"len={len(self.prev_hash)}"
            )
        for c in self.prev_hash:
            if c not in _HEX_ALPHABET:
                raise ValueError(
                    f"FrozenRoundRecord.prev_hash must be lowercase hex, got "
                    f"invalid char {c!r}"
                )
