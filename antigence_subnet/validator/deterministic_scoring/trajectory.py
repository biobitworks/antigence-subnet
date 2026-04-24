"""Pure-function trajectory extraction over the Phase 1000 audit chain (BPCONV-01).

Reads the JSONL audit log, calls :func:`verify_chain` first so a tampered
log fails loudly, and returns a per-UID :class:`TrajectoryWindow` holding
the last ``window_size`` EMA scores as an immutable tuple.

No state mutation. No miner network calls. No numpy. No bittensor.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Tuple

from antigence_subnet.validator.deterministic_scoring.chain import (
    ChainIntegrityError,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.serialization import (
    from_canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
)


@dataclass(frozen=True, slots=True)
class TrajectoryWindow:
    """Immutable snapshot of a single miner's EMA trajectory.

    Fields
    ------
    uid : int
        Miner UID (non-negative).
    round_start : int
        Inclusive first round represented in ``ema_scores``.
    round_end : int
        Inclusive last round represented in ``ema_scores``.
    ema_scores : tuple[float, ...]
        EMA score per round, in round order. Length must equal
        ``round_end - round_start + 1``.
    """

    uid: int
    round_start: int
    round_end: int
    ema_scores: Tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.uid) is not int or isinstance(self.uid, bool):
            raise TypeError(
                f"TrajectoryWindow.uid must be int, got {type(self.uid).__name__}"
            )
        if self.uid < 0:
            raise ValueError(f"TrajectoryWindow.uid must be >= 0, got {self.uid}")
        if type(self.round_start) is not int or isinstance(self.round_start, bool):
            raise TypeError(
                f"TrajectoryWindow.round_start must be int, got "
                f"{type(self.round_start).__name__}"
            )
        if type(self.round_end) is not int or isinstance(self.round_end, bool):
            raise TypeError(
                f"TrajectoryWindow.round_end must be int, got "
                f"{type(self.round_end).__name__}"
            )
        if self.round_end < self.round_start:
            raise ValueError(
                f"TrajectoryWindow.round_end ({self.round_end}) must be >= "
                f"round_start ({self.round_start})"
            )
        if type(self.ema_scores) is not tuple:
            raise TypeError(
                f"TrajectoryWindow.ema_scores must be tuple, got "
                f"{type(self.ema_scores).__name__}"
            )
        expected_len = self.round_end - self.round_start + 1
        if len(self.ema_scores) != expected_len:
            raise ValueError(
                f"TrajectoryWindow.ema_scores length ({len(self.ema_scores)}) "
                f"must equal round_end - round_start + 1 ({expected_len})"
            )
        for i, v in enumerate(self.ema_scores):
            if type(v) is not float:
                raise TypeError(
                    f"TrajectoryWindow.ema_scores[{i}] must be Python float, "
                    f"got {type(v).__name__}"
                )


def extract_trajectories(
    log_path: pathlib.Path | str,
    window_size: int = 20,
) -> dict[int, TrajectoryWindow]:
    """Return per-UID :class:`TrajectoryWindow` for the last ``window_size`` rounds.

    Pipeline:
        1. :func:`verify_chain` on the log path -- propagates
           :class:`ChainIntegrityError` for missing, malformed, or tampered
           logs.
        2. Parse each line via :func:`from_canonical_json` into a
           :class:`FrozenRoundRecord`.
        3. Collect ``(round_index, ema_score)`` per UID in round order.
        4. Keep only the last ``window_size`` entries per UID (ring-buffer
           semantics).
        5. Wrap each as an immutable :class:`TrajectoryWindow`.

    An empty chain file (zero bytes) yields ``{}``.

    Raises:
        ChainIntegrityError: propagated from :func:`verify_chain`.
        ValueError: if ``window_size < 1``.
    """
    if type(window_size) is not int or window_size < 1:
        raise ValueError(
            f"extract_trajectories.window_size must be int >= 1, got "
            f"{window_size!r}"
        )

    path = pathlib.Path(log_path)
    verify_chain(path)  # propagates on missing / tampered / malformed.

    raw = path.read_bytes()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return {}

    per_uid: dict[int, list[tuple[int, float]]] = {}
    for line in lines:
        rec = from_canonical_json(line, FrozenRoundRecord)
        for score in rec.scores:
            per_uid.setdefault(score.uid, []).append(
                (rec.round_index, score.ema_score)
            )

    out: dict[int, TrajectoryWindow] = {}
    for uid, entries in per_uid.items():
        # entries are already in round order because we read lines top-to-
        # bottom and verify_chain just asserted contiguous round_index.
        kept = entries[-window_size:]
        out[uid] = TrajectoryWindow(
            uid=uid,
            round_start=kept[0][0],
            round_end=kept[-1][0],
            ema_scores=tuple(s for _, s in kept),
        )
    return out


__all__ = [
    "TrajectoryWindow",
    "extract_trajectories",
]
