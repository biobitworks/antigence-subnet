"""Deterministic replay harness for the validator scoring state machine.

Given per-round ``(uid, raw_reward, hotkey)`` inputs plus an optional initial
EMA map, :func:`replay_chain` reproduces the full sequence of
:class:`FrozenRoundRecord` objects and their SHA-256 hashes -- byte-for-byte
identical across runs (DETSCORE-04).

Scope boundary
--------------
Replay operates on the *scoring state machine*, not on miner responses. The
upstream reward computation (what produced each ``raw_reward``) is the
existing mutable path in ``validator/reward.py`` and is **not** replayed
here. This keeps the new layer strictly additive: the reward values
themselves are treated as inputs.

EMA formula
-----------
Mirrored from ``antigence_subnet/base/validator.py``'s ``update_scores``::

    new_ema = alpha * raw_reward + (1 - alpha) * prev_ema.get(uid, 0.0)

New UIDs start from 0.0 prev_ema. NaN raw_rewards are **rejected** (not
sanitized) because determinism requires explicit failure rather than silent
coercion -- surfaced via the ``canonical_json`` NaN check when we hash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from antigence_subnet.validator.deterministic_scoring.chain import hash_record
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)


@dataclass(frozen=True, slots=True)
class RoundInputs:
    """Per-round input tuple driving a replay.

    ``raw_rewards`` is a tuple of ``(uid, raw_reward, hotkey)`` entries.
    Callers should pre-sort by uid to guarantee byte-stable output (the
    replay harness re-sorts defensively, but supplying sorted input keeps
    ``canonical_json`` output bit-stable between input construction and
    replay).
    """

    round_index: int
    ema_alpha: float
    raw_rewards: Tuple[Tuple[int, float, str], ...]

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or isinstance(self.round_index, bool):
            raise TypeError(
                f"RoundInputs.round_index must be int, got "
                f"{type(self.round_index).__name__}"
            )
        if self.round_index < 0:
            raise ValueError(
                f"RoundInputs.round_index must be >= 0, got {self.round_index}"
            )
        if type(self.ema_alpha) is not float:
            raise TypeError(
                f"RoundInputs.ema_alpha must be Python float, got "
                f"{type(self.ema_alpha).__name__}"
            )
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError(
                f"RoundInputs.ema_alpha must be in [0.0, 1.0], got "
                f"{self.ema_alpha}"
            )
        if type(self.raw_rewards) is not tuple:
            raise TypeError(
                f"RoundInputs.raw_rewards must be tuple, got "
                f"{type(self.raw_rewards).__name__}"
            )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Output of :func:`replay_chain`.

    ``records[i]`` is the i-th rebuilt FrozenRoundRecord and
    ``hashes[i] == hash_record(records[i])``. Both tuples have the same
    length as the input ``rounds`` sequence.
    """

    records: Tuple[FrozenRoundRecord, ...]
    hashes: Tuple[str, ...]


def replay_chain(
    rounds: Sequence[RoundInputs],
    initial_ema: Mapping[int, float] | None = None,
) -> ReplayResult:
    """Reproduce the frozen record sequence and hash chain from inputs.

    For each round, compute per-uid ``new_ema = alpha * raw_reward +
    (1 - alpha) * prev_ema.get(uid, 0.0)``, build ``FrozenRoundScore`` and
    ``FrozenRoundRecord`` tuples, hash each record, and link prev_hash.

    Raises:
        ValueError: if any ``raw_reward`` is NaN or +/-Inf (propagated from
            ``canonical_json`` during hashing).
    """
    prev_ema: dict[int, float] = dict(initial_ema or {})
    records: list[FrozenRoundRecord] = []
    hashes: list[str] = []

    for round_inputs in rounds:
        # Pre-check: reject NaN/Inf raw rewards explicitly. canonical_json
        # will also catch these at hash time, but raising here gives a
        # clearer error site ("raw_reward for uid=X is non-finite") before
        # we waste work building the record.
        for uid, raw_reward, _hotkey in round_inputs.raw_rewards:
            if type(raw_reward) is not float:
                raise TypeError(
                    f"replay_chain: raw_reward for uid={uid} must be Python "
                    f"float, got {type(raw_reward).__name__}"
                )
            if math.isnan(raw_reward) or math.isinf(raw_reward):
                raise ValueError(
                    f"replay_chain: raw_reward for uid={uid} is non-finite: "
                    f"{raw_reward!r}"
                )

        scores_list: list[FrozenRoundScore] = []
        for uid, raw_reward, hotkey in round_inputs.raw_rewards:
            new_ema = (
                round_inputs.ema_alpha * raw_reward
                + (1.0 - round_inputs.ema_alpha) * prev_ema.get(uid, 0.0)
            )
            scores_list.append(
                FrozenRoundScore(
                    uid=uid,
                    raw_reward=raw_reward,
                    ema_score=new_ema,
                    hotkey=hotkey,
                )
            )
            prev_ema[uid] = new_ema

        scores_tuple = tuple(sorted(scores_list, key=lambda s: s.uid))
        prev_hash = hashes[-1] if hashes else "0" * 64
        record = FrozenRoundRecord(
            round_index=round_inputs.round_index,
            ema_alpha=round_inputs.ema_alpha,
            scores=scores_tuple,
            prev_hash=prev_hash,
        )
        h = hash_record(record)
        records.append(record)
        hashes.append(h)

    return ReplayResult(records=tuple(records), hashes=tuple(hashes))
