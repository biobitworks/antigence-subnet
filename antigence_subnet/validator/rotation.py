"""
Round-based challenge rotation tracker for anti-caching defense.

Tracks which evaluation samples each miner has seen within a configurable
rotation window. The validator uses this to exclude previously-seen samples
from future rounds, preventing miners from caching (memorizing) responses.

Requirements: VHARD-01 (round-based challenge rotation)
"""

from __future__ import annotations


class ChallengeRotation:
    """Tracks per-miner challenge history for round-based rotation.

    Maintains a sliding window of which sample IDs each miner received
    in recent rounds. The validator queries ``get_excluded(hotkey)`` before
    selecting samples for a miner and passes those IDs to exclude from
    the candidate pool.

    Args:
        rotation_window: Number of most recent rounds to track per miner.
            Samples seen within this window are excluded from future
            selection. Default: 10.
    """

    def __init__(self, rotation_window: int = 10) -> None:
        self.rotation_window = rotation_window
        # hotkey -> [(round_num, {sample_id, ...}), ...]
        self._history: dict[str, list[tuple[int, set[str]]]] = {}

    def record(
        self, miner_hotkey: str, round_num: int, sample_ids: list[str]
    ) -> None:
        """Record which samples a miner saw in a given round.

        Automatically evicts entries older than the rotation window.

        Args:
            miner_hotkey: The miner's hotkey SS58 address.
            round_num: The evaluation round number.
            sample_ids: List of sample IDs the miner received this round.
        """
        if miner_hotkey not in self._history:
            self._history[miner_hotkey] = []

        self._history[miner_hotkey].append((round_num, set(sample_ids)))

        # Evict entries beyond the rotation window.
        # Keep only the most recent `rotation_window` entries.
        if len(self._history[miner_hotkey]) > self.rotation_window:
            self._history[miner_hotkey] = self._history[miner_hotkey][
                -self.rotation_window :
            ]

    def get_excluded(self, miner_hotkey: str) -> set[str]:
        """Return all sample IDs this miner has seen within the rotation window.

        Args:
            miner_hotkey: The miner's hotkey SS58 address.

        Returns:
            Set of sample IDs to exclude from future challenge selection.
        """
        if miner_hotkey not in self._history:
            return set()

        result: set[str] = set()
        for _round_num, sample_ids in self._history[miner_hotkey]:
            result |= sample_ids
        return result

    def clear(self, miner_hotkey: str | None = None) -> None:
        """Clear challenge history.

        Args:
            miner_hotkey: If provided, clear only this miner's history.
                If None, clear all miners' history.
        """
        if miner_hotkey is not None:
            self._history.pop(miner_hotkey, None)
        else:
            self._history.clear()

    def to_dict(self) -> dict:
        """Serialize rotation state to a JSON-compatible dict.

        Returns:
            Dict with 'rotation_window' and 'history' keys.
        """
        return {
            "rotation_window": self.rotation_window,
            "history": {
                hotkey: [
                    [round_num, sorted(sample_ids)]
                    for round_num, sample_ids in entries
                ]
                for hotkey, entries in self._history.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChallengeRotation:
        """Deserialize rotation state from a dict.

        Args:
            data: Dict previously produced by ``to_dict()``.

        Returns:
            Restored ChallengeRotation instance.
        """
        instance = cls(rotation_window=data["rotation_window"])
        for hotkey, entries in data.get("history", {}).items():
            instance._history[hotkey] = [
                (entry[0], set(entry[1])) for entry in entries
            ]
        return instance
