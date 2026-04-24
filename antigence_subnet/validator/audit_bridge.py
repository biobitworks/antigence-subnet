"""Audit-chain bridge for v13.1 migration (experiment-first).

Wraps the verbatim copy of production ``get_rewards`` in
``experiments/v13.1-migration/production_copy/reward.py`` and emits one
``FrozenRoundRecord`` (bundling all per-miner ``FrozenRoundScore`` entries
for the round) to a SHA-256 linked audit chain JSONL file via the
Phase 1000 ``AuditChainWriter``.

Key boundaries
--------------
* ``numpy.float32`` / ``numpy.float64`` rewards are converted to Python
  ``float`` at the wrapper. ``FrozenRoundScore.__post_init__`` enforces
  exact Python ``float`` via ``type(x) is float`` (numpy scalars rejected
  downstream) -- so the conversion has to happen HERE.
* ``NaN`` and ``+-Inf`` are rejected at the wrapper, before construction
  of the dataclass, so the chain never contains poisoned records.
* ``deterministic_scoring`` stays bittensor-free. audit_bridge.py also
  imports zero bittensor symbols; the *production_copy/reward.py* copy
  does ``import bittensor as bt``, but tests install a stub for that.
* No biological-modeling terms. No ``random.Random`` seeds.

Public surface
--------------
* :class:`RewardToAuditAdapter`
* :func:`bridge_get_rewards`
* :func:`resume_chain_prev_hash`
* :func:`next_round_index`
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from antigence_subnet.validator.deterministic_scoring import (
    GENESIS_PREV_HASH,
    AuditChainWriter,
    FrozenRoundRecord,
    FrozenRoundScore,
    verify_chain,
)

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    import numpy as _np


__all__ = [
    "RewardToAuditAdapter",
    "bridge_get_rewards",
    "next_round_index",
    "resume_chain_prev_hash",
]


def _is_numpy_floating(x: Any) -> bool:
    """True iff ``x`` is a numpy floating scalar (float16/32/64/128, ...).

    Done by module-name string match so audit_bridge has zero numpy import
    at module load (numpy is only pulled in lazily inside
    ``bridge_get_rewards`` by the production_copy reward module).
    """
    t = type(x)
    if t.__module__ != "numpy":
        return False
    # Walk MRO for a numpy.floating base. Cheaper than importing numpy.
    return any(
        base.__module__ == "numpy" and base.__name__ == "floating"
        for base in t.__mro__
    )


def _is_numpy_integer(x: Any) -> bool:
    t = type(x)
    if t.__module__ != "numpy":
        return False
    return any(
        base.__module__ == "numpy" and base.__name__ == "integer"
        for base in t.__mro__
    )


class RewardToAuditAdapter:
    """Builds one :class:`FrozenRoundRecord` per evaluation round.

    Parameters
    ----------
    writer : AuditChainWriter
        The chain writer responsible for SHA-linkage and JSONL persistence.
    ema_alpha : float
        EMA alpha recorded on every round record. Must be a Python
        ``float`` strictly between 0.0 and 1.0 (both exclusive).

    Notes
    -----
    The adapter is boundary-only: it accepts numpy scalars as rewards and
    converts them to Python ``float`` with explicit ``float(...)`` casts
    before constructing ``FrozenRoundScore``. Bool and Python-int values
    are rejected (they would pass ``type(x) is float`` false-negatives
    and produce silently wrong canonical JSON).
    """

    def __init__(self, writer: AuditChainWriter, ema_alpha: float = 0.1) -> None:
        if not isinstance(writer, AuditChainWriter):
            raise TypeError(
                f"RewardToAuditAdapter.writer must be AuditChainWriter, got "
                f"{type(writer).__name__}"
            )
        if type(ema_alpha) is not float:
            raise TypeError(
                f"RewardToAuditAdapter.ema_alpha must be Python float, got "
                f"{type(ema_alpha).__name__}"
            )
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError(
                f"RewardToAuditAdapter.ema_alpha must be in (0.0, 1.0), got "
                f"{ema_alpha}"
            )
        self.writer = writer
        self.ema_alpha = ema_alpha

    @staticmethod
    def _coerce_float(x: Any, *, field: str) -> float:
        """Return ``x`` as a Python ``float`` or raise.

        * Python ``float`` passes through unchanged.
        * Python ``bool`` is rejected (subclass of int; would silently
          canonicalize to 0/1).
        * ``numpy.floating`` scalars are converted via ``float(x)``.
        * ``numpy.integer`` scalars are rejected (explicit TypeError).
        * Anything else raises TypeError.
        * ``NaN`` and ``+-Inf`` raise ValueError *after* coercion so
          FrozenRoundScore never sees them.
        """
        if isinstance(x, bool):
            raise TypeError(
                f"{field} must be Python float, got bool (use float casts "
                f"explicitly at the boundary)"
            )
        if type(x) is float:
            out = x
        elif _is_numpy_floating(x):
            out = float(x)
        elif _is_numpy_integer(x):
            raise TypeError(
                f"{field} must be float, got numpy integer "
                f"{type(x).__name__}; explicit float cast required upstream"
            )
        else:
            raise TypeError(
                f"{field} must be Python float or numpy.floating, got "
                f"{type(x).__module__}.{type(x).__name__}"
            )
        if math.isnan(out):
            raise ValueError(f"{field} is NaN (rejected at bridge boundary)")
        if math.isinf(out):
            sign = "+" if out > 0 else "-"
            raise ValueError(
                f"{field} is {sign}Inf (rejected at bridge boundary)"
            )
        return out

    def record_round(
        self,
        round_index: int,
        miner_uids: Sequence[int],
        rewards: Sequence[Any],
        hotkeys: Sequence[str],
        *,
        ema_scores: Sequence[Any] | None = None,
    ) -> str:
        """Build one FrozenRoundRecord, append, and return its hash."""
        if type(round_index) is not int or isinstance(round_index, bool):
            raise TypeError(
                f"round_index must be int, got {type(round_index).__name__}"
            )
        n = len(miner_uids)
        if len(rewards) != n:
            raise ValueError(
                f"rewards length {len(rewards)} != miner_uids length {n}"
            )
        if len(hotkeys) != n:
            raise ValueError(
                f"hotkeys length {len(hotkeys)} != miner_uids length {n}"
            )
        if ema_scores is not None and len(ema_scores) != n:
            raise ValueError(
                f"ema_scores length {len(ema_scores)} != miner_uids length {n}"
            )

        scores: list[FrozenRoundScore] = []
        for i, uid in enumerate(miner_uids):
            raw = self._coerce_float(rewards[i], field=f"rewards[{i}]")
            if ema_scores is None:
                ema = raw
            else:
                ema = self._coerce_float(ema_scores[i], field=f"ema_scores[{i}]")
            if type(uid) is not int or isinstance(uid, bool):
                raise TypeError(
                    f"miner_uids[{i}] must be int, got {type(uid).__name__}"
                )
            hk = hotkeys[i]
            if type(hk) is not str:
                raise TypeError(
                    f"hotkeys[{i}] must be str, got {type(hk).__name__}"
                )
            scores.append(
                FrozenRoundScore(uid=uid, raw_reward=raw, ema_score=ema, hotkey=hk)
            )

        record = FrozenRoundRecord(
            round_index=round_index,
            ema_alpha=self.ema_alpha,
            scores=tuple(scores),
            prev_hash=self.writer.latest_hash(),
        )
        return self.writer.append(record)


def next_round_index(writer: AuditChainWriter) -> int:
    """Return the next contiguous round_index for ``writer``'s chain.

    0 if the chain is empty; else ``last.round_index + 1``.
    """
    last = writer._last_round_index()  # noqa: SLF001 - experiment-only read
    return 0 if last is None else last + 1


def resume_chain_prev_hash(path: pathlib.Path | str) -> str:
    """Return the prev_hash to use for the next append on restart.

    * Missing or empty file -> ``GENESIS_PREV_HASH``.
    * Non-empty file -> ``verify_chain(path)`` (raises
      :class:`ChainIntegrityError` on tamper) then ``latest_hash()``.

    The verification walk makes this a safe startup gate for operators:
    a broken chain is surfaced immediately instead of silently continued.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return GENESIS_PREV_HASH
    raw = p.read_bytes()
    if not raw.strip():
        return GENESIS_PREV_HASH
    verify_chain(p)  # raises ChainIntegrityError on tamper
    return AuditChainWriter(p).latest_hash()


def bridge_get_rewards(
    validator: Any,
    miner_uids: list[int],
    responses_by_sample: dict,
    manifest: dict,
    chain_writer: AuditChainWriter,
    *,
    hotkeys_by_uid: dict[int, str] | None = None,
    round_index: int | None = None,
    ema_alpha: float = 0.1,
) -> _np.ndarray:
    """Call production_copy.reward.get_rewards and emit one chain record.

    Drop-in compatibility: returns the same ``numpy.ndarray`` the
    original ``get_rewards`` returns.

    Parameters
    ----------
    validator, miner_uids, responses_by_sample, manifest
        Forwarded verbatim to ``production_copy.reward.get_rewards``.
    chain_writer : AuditChainWriter
        Where to persist the audit record.
    hotkeys_by_uid : dict[int, str] or None
        Mapping from UID to ss58 hotkey. If None, each UID is paired with
        ``validator.hotkey_for_uid(uid)`` if present, otherwise a stable
        placeholder ``f"hk-unknown-{uid}"``.
    round_index : int or None
        Explicit round index (overrides chain-derived value). When None,
        the next contiguous index from the chain is used.
    ema_alpha : float
        EMA alpha recorded on the chain record.
    """
    # Lazy import: avoids pulling numpy + bittensor into audit_bridge's
    # module-load path. Tests install a bittensor stub before calling here.
    from antigence_subnet.validator import reward as prod_reward

    rewards = prod_reward.get_rewards(
        validator, miner_uids, responses_by_sample, manifest
    )

    adapter = RewardToAuditAdapter(chain_writer, ema_alpha=ema_alpha)

    idx = round_index if round_index is not None else next_round_index(chain_writer)

    if hotkeys_by_uid is None:
        hotkey_fn = getattr(validator, "hotkey_for_uid", None)
        if callable(hotkey_fn):
            hotkeys = [str(hotkey_fn(u)) for u in miner_uids]
        else:
            hotkeys = [f"hk-unknown-{u}" for u in miner_uids]
    else:
        missing = [u for u in miner_uids if u not in hotkeys_by_uid]
        if missing:
            raise ValueError(
                f"hotkeys_by_uid missing entries for uids: {missing}"
            )
        hotkeys = [hotkeys_by_uid[u] for u in miner_uids]

    # rewards is a numpy array; iterate scalar-by-scalar so _coerce_float
    # sees each numpy.floating independently (not a 0-d array).
    reward_list: list[Any] = [rewards[i] for i in range(len(miner_uids))]

    adapter.record_round(
        round_index=idx,
        miner_uids=list(miner_uids),
        rewards=reward_list,
        hotkeys=hotkeys,
    )
    return rewards
