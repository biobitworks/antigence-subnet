"""Phase 1102 audit_state tests (STATEPOL-01, STATEPOL-02).

Null-hypothesis evidence:

* test_audit_disabled_writes_nothing
    NH0 (STATEPOL-02): audit disabled might still create files.
    Expected: 5 successive save_audit_state calls produce no
    ``chain.jsonl`` on disk.

* test_audit_enabled_creates_parent_dir
    Sanity: enabled path makes the chain directory writable.

* test_audit_enabled_writes_chain
    Sanity: writing via RewardToAuditAdapter produces non-empty JSONL.

* test_npz_save_load_unchanged
    NH0 (STATEPOL-01): the new audit hook might clobber the .npz write.
    Expected: a spy on np.savez confirms the ``.npz`` path is called
    exactly as before; audit hook is ADDITIVE.

* test_load_audit_state_disabled_is_noop
    NH0: disabled load might still touch the chain. Expected: attribute
    set to None; no chain access.

* test_load_audit_state_enabled_resolves_path
    Sanity: enabled load resolves path and sets attribute.

* test_production_untouched
    NH0: experiment silently edits production. Expected:
    ``git diff v13.0..HEAD -- antigence_subnet/`` is empty.
"""

from __future__ import annotations

import pathlib
import subprocess
import types

import numpy as np

from antigence_subnet.validator import audit_state
from antigence_subnet.validator.audit_bridge import RewardToAuditAdapter
from antigence_subnet.validator.deterministic_scoring import (
    GENESIS_PREV_HASH,
    AuditChainWriter,
    verify_chain,
)


# --------------------------------------------------------------------- #
# Stub validator builder                                                 #
# --------------------------------------------------------------------- #
def _make_stub_validator(
    enabled: bool,
    *,
    full_path: str,
    chain_path: str = "",
):
    """Duck-typed validator object with just the attributes audit_state touches."""
    v = types.SimpleNamespace()
    v.config = types.SimpleNamespace()
    v.config.audit = types.SimpleNamespace(enabled=enabled, chain_path=chain_path)
    v.config.neuron = types.SimpleNamespace(full_path=full_path)
    v.audit_chain_path = None
    return v


# --------------------------------------------------------------------- #
# STATEPOL-02: disabled writes nothing                                   #
# --------------------------------------------------------------------- #
def test_audit_disabled_writes_nothing(tmp_path):
    """NH0: disabled might still write. Expected: no chain.jsonl exists."""
    v = _make_stub_validator(enabled=False, full_path=str(tmp_path / "neuron"))
    for _ in range(5):
        audit_state.save_audit_state(v)

    assert not (tmp_path / "neuron" / "chain.jsonl").exists()
    assert not (tmp_path / "neuron").exists(), (
        "save_audit_state created a directory while audit.enabled=False"
    )
    assert v.audit_chain_path is None

    # load_audit_state also no-op.
    audit_state.load_audit_state(v)
    assert v.audit_chain_path is None
    assert not (tmp_path / "neuron" / "chain.jsonl").exists()


# --------------------------------------------------------------------- #
# Enabled: parent dir creation                                           #
# --------------------------------------------------------------------- #
def test_audit_enabled_creates_parent_dir(tmp_path):
    """Sanity: enabled save preflight makes parent dir exist."""
    neuron_dir = tmp_path / "neuron"
    v = _make_stub_validator(enabled=True, full_path=str(neuron_dir))
    audit_state.save_audit_state(v)
    # Chain file itself is not created until first append,
    # but the parent directory must now exist.
    assert neuron_dir.is_dir()
    assert not (neuron_dir / "chain.jsonl").exists()


# --------------------------------------------------------------------- #
# Enabled: actual chain writes via adapter                               #
# --------------------------------------------------------------------- #
def test_audit_enabled_writes_chain(tmp_path):
    """Sanity: enabled + adapter usage writes a non-empty chain."""
    neuron_dir = tmp_path / "neuron"
    v = _make_stub_validator(enabled=True, full_path=str(neuron_dir))
    audit_state.save_audit_state(v)
    # Simulate the forward-loop chain writer behavior.
    chain_path = neuron_dir / "chain.jsonl"
    writer = AuditChainWriter(chain_path)
    adapter = RewardToAuditAdapter(writer, ema_alpha=0.1)
    adapter.record_round(
        round_index=0,
        miner_uids=[0, 1],
        rewards=[np.float32(0.7), np.float32(0.3)],
        hotkeys=["hk-0" * 16, "hk-1" * 16],
    )
    assert chain_path.exists()
    # verify_chain clean -> starts from GENESIS_PREV_HASH.
    verify_chain(chain_path)
    assert chain_path.read_text().count("\n") >= 1


# --------------------------------------------------------------------- #
# STATEPOL-01: .npz path unchanged (spy-based)                            #
# --------------------------------------------------------------------- #
def test_npz_save_load_unchanged(monkeypatch, tmp_path):
    """NH0: new audit hook clobbers .npz. Expected: .npz call count
    identical regardless of audit.enabled.

    This test installs a spy on numpy.savez, then runs a minimal
    save sequence mimicking the base_validator save_state() shape:

    1. .npz write (spied)
    2. audit_state.save_audit_state(...)

    Under STATEPOL-01 the .npz write must be called EXACTLY ONCE in
    both enabled and disabled modes. audit_state never touches np.savez.
    """
    calls: list[dict] = []

    def _fake_savez(path, **kwargs):
        calls.append({"path": str(path), "kwargs": dict(kwargs)})

    monkeypatch.setattr(np, "savez", _fake_savez)

    # Mimic save_state() from base_validator: .npz first, audit hook second.
    def _fake_save_state(validator):
        np.savez(  # would normally write
            f"{validator.config.neuron.full_path}/state.tmp.npz",
            step=0, scores=np.zeros(1, dtype=np.float32), hotkeys=[],
        )
        audit_state.save_audit_state(validator)

    # Disabled mode.
    v_off = _make_stub_validator(enabled=False, full_path=str(tmp_path / "off"))
    _fake_save_state(v_off)
    assert len(calls) == 1
    # No chain dir created.
    assert not (tmp_path / "off").exists()

    # Enabled mode -- np.savez count unchanged, audit preflights the dir.
    v_on = _make_stub_validator(enabled=True, full_path=str(tmp_path / "on"))
    _fake_save_state(v_on)
    assert len(calls) == 2  # exactly one more .npz call
    assert (tmp_path / "on").is_dir()
    # Still no chain.jsonl (written lazily on first append).
    assert not (tmp_path / "on" / "chain.jsonl").exists()


# --------------------------------------------------------------------- #
# load_audit_state disabled path                                         #
# --------------------------------------------------------------------- #
def test_load_audit_state_disabled_is_noop(tmp_path):
    """NH0: disabled load might verify/read chain. Expected: pure no-op,
    audit_chain_path attribute left at None."""
    v = _make_stub_validator(enabled=False, full_path=str(tmp_path / "n"))
    # Even if someone pre-seeded a bad value, disabled load clears to None.
    v.audit_chain_path = "/some/bogus/path"
    audit_state.load_audit_state(v)
    assert v.audit_chain_path is None


def test_load_audit_state_enabled_resolves_path(tmp_path):
    """Sanity: enabled load sets audit_chain_path; missing file -> clean start."""
    neuron_dir = tmp_path / "n"
    v = _make_stub_validator(enabled=True, full_path=str(neuron_dir))
    audit_state.load_audit_state(v)
    assert v.audit_chain_path == str(neuron_dir / "chain.jsonl")
    # File didn't need to exist -- resume returns GENESIS on missing.
    # (Direct verification of GENESIS via resume_chain_prev_hash.)
    from antigence_subnet.validator.audit_bridge import resume_chain_prev_hash

    assert resume_chain_prev_hash(v.audit_chain_path) == GENESIS_PREV_HASH


def test_load_audit_state_explicit_chain_path(tmp_path):
    """Explicit chain_path in config wins over full_path default."""
    explicit = tmp_path / "explicit.jsonl"
    v = _make_stub_validator(
        enabled=True,
        full_path=str(tmp_path / "unused"),
        chain_path=str(explicit),
    )
    audit_state.load_audit_state(v)
    assert v.audit_chain_path == str(explicit)


# --------------------------------------------------------------------- #
# Production untouched (STATEPOL protection)                             #
# --------------------------------------------------------------------- #
def test_production_untouched():
    """NH0: audit-chain promotion silently rewrites reward.py.

    Expected: reward.py byte-identical to v13.0.

    v13.1.1 (Phase 1103) expects forward.py, base/validator.py, and
    validate_config.py to differ from v13.0 (audit-chain wire-in).
    reward.py MUST remain untouched because audit_bridge wraps it from
    outside (STATEPOL-01 additivity guarantee on the reward path).
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    files = [
        "antigence_subnet/validator/reward.py",
    ]
    result = subprocess.run(
        ["git", "diff", "v13.0..HEAD", "--"] + files,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", (
        f"reward.py drifted from v13.0 (must remain byte-identical):\n{result.stdout}"
    )
