"""Parallel audit-chain state hooks for :class:`BaseValidatorNeuron`.

Two stdlib+Phase-1100-only entry points that run *alongside* the existing
``.npz`` save_state / load_state — never replacing them. Both are no-ops
when ``validator.config.audit.enabled`` is False (STATEPOL-02).

* :func:`save_audit_state` — ensures the chain directory + parent exist
  before writes. The actual chain records are written by Phase 1100's
  ``AuditChainWriter.append`` inside ``bridge_get_rewards``; this is a
  best-effort preflight.

* :func:`load_audit_state` — on startup, verify the chain via
  ``resume_chain_prev_hash`` (Phase 1100) so a tampered file raises
  ``ChainIntegrityError`` immediately instead of silently continuing.
  Also resolves ``validator.audit_chain_path``:

  - If ``validator.config.audit.chain_path`` is a non-empty string, use it.
  - Otherwise, default to ``<neuron.full_path>/chain.jsonl``.

STATEPOL-01 guarantee: these functions never read or write the ``.npz``
state file, never mutate any attribute that the ``.npz`` path manages
(step, scores, hotkeys, score_history, confidence_history,
challenge_rotation). They add ONE attribute: ``validator.audit_chain_path``.

Zero bittensor imports. Safe to load under ``/tmp/chi-exp-venv``.
"""

from __future__ import annotations

import os
from typing import Any

# audit_bridge imports only from deterministic_scoring (bittensor-free).
from antigence_subnet.validator.audit_bridge import resume_chain_prev_hash

__all__ = [
    "audit_enabled",
    "resolve_audit_chain_path",
    "save_audit_state",
    "load_audit_state",
]


def audit_enabled(validator: Any) -> bool:
    """True iff ``validator.config.audit.enabled`` is True.

    Defensive: missing attributes return False (feature-off default).
    Accepts both dataclass-style configs (``.audit.enabled``) and the
    Bittensor ``bt.Config`` duck-type (also ``.audit.enabled`` via
    attribute access).
    """
    config = getattr(validator, "config", None)
    if config is None:
        return False
    audit = getattr(config, "audit", None)
    if audit is None:
        return False
    return bool(getattr(audit, "enabled", False))


def resolve_audit_chain_path(validator: Any) -> str:
    """Return the audit chain path for ``validator``, or empty string if none.

    Precedence:

    1. ``validator.config.audit.chain_path`` if non-empty string.
    2. ``<validator.config.neuron.full_path>/chain.jsonl`` as fallback.
    3. Empty string if neither is present (caller must handle).
    """
    config = getattr(validator, "config", None)
    if config is None:
        return ""
    audit = getattr(config, "audit", None)
    explicit = getattr(audit, "chain_path", "") if audit is not None else ""
    if isinstance(explicit, str) and explicit:
        return os.path.expanduser(explicit)

    neuron = getattr(config, "neuron", None)
    full_path = getattr(neuron, "full_path", "") if neuron is not None else ""
    if not isinstance(full_path, str) or not full_path:
        return ""
    return os.path.join(os.path.expanduser(full_path), "chain.jsonl")


def save_audit_state(validator: Any) -> None:
    """Preflight: ensure chain directory exists. No-op when disabled.

    STATEPOL-01: does not touch ``.npz`` state.
    STATEPOL-02: does nothing at all when ``audit.enabled=False``.

    The chain JSONL file is written lazily by ``AuditChainWriter`` on
    first ``append`` inside ``bridge_get_rewards``; this just makes sure
    the parent directory is writable so that first append does not fail.
    """
    if not audit_enabled(validator):
        return
    path = resolve_audit_chain_path(validator)
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_audit_state(validator: Any) -> None:
    """Resolve chain path + verify integrity on restart. No-op when disabled.

    STATEPOL-01: does not touch ``.npz`` state.
    STATEPOL-02: does nothing at all when ``audit.enabled=False``.
    STATEPOL-03: missing / empty chain file is fine (returns
    ``GENESIS_PREV_HASH`` via :func:`resume_chain_prev_hash`); a fresh
    chain starts on the next ``append``. No history replay.

    Side effects:

    * Sets ``validator.audit_chain_path`` to the resolved path (string).
    * Calls :func:`resume_chain_prev_hash` — which raises
      ``ChainIntegrityError`` if the chain file is tampered. Operators
      see the failure immediately at startup rather than silently
      continuing.
    """
    if not audit_enabled(validator):
        # Even when disabled, clear the attribute so forward.py's
        # ``if validator.audit_chain_path is not None`` gate stays
        # coherent across toggle events. (STATEPOL-03: must work for
        # mid-session enable too.)
        validator.audit_chain_path = None
        return
    path = resolve_audit_chain_path(validator)
    if not path:
        validator.audit_chain_path = None
        return
    # resume_chain_prev_hash verifies the chain (raises on tamper) and
    # returns GENESIS_PREV_HASH for missing/empty files -- exactly the
    # STATEPOL-03 "clean start, no replay" behavior.
    resume_chain_prev_hash(path)
    validator.audit_chain_path = path
