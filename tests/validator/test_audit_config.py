"""Phase 1102 audit_config + validate_config tests (STATEPOL-02, WIRE-03).

Null-hypothesis evidence:

* test_default_audit_disabled
    NH0: audit might default ON (dangerous — chain writes without opt-in).
    Expected: ``AuditConfig()`` and ``audit_config_from_toml({})`` both
    return ``enabled=False``, empty ``chain_path``.

* test_toml_loads_audit_section
    NH0: TOML [validator.audit] might be silently ignored.
    Expected: values round-trip into the frozen dataclass.

* test_toml_loads_convergence_section
    NH0: [validator.convergence] overrides might not flow through.
    Expected: epsilon override lands in ``AuditConfig.convergence.epsilon``.

* test_config_precedence_cli_gt_toml
    NH0: CLI might not override TOML. Expected (WIRE-03):
    ``apply_audit_cli_overrides`` on a TOML-loaded config produces
    CLI > TOML > defaults.

* test_config_validates_audit_toml
    NH0: unknown TOML keys under [validator.audit] might pass silently.
    Expected: modified ``validate_config.py`` emits an "Unknown key"
    warning at section ``validator.audit``.

* test_config_validates_convergence_ranges
    NH0: bad ranges (e.g. epsilon=2.0) might slip past validation.
    Expected: validate_config emits a range error.

* test_audit_config_is_frozen
    NH0: mutability could allow in-place attacks on the config.
    Expected: both AuditConfig and ConvergenceConfig raise
    FrozenInstanceError on attribute set.

* test_audit_config_no_numpy_no_bittensor_import
    NH0: either dependency might sneak in via transitive imports.
    Expected: subprocess with clean sys.modules -> neither present.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import pytest

from antigence_subnet.validator import audit_config


# --------------------------------------------------------------------- #
# 1 -- default disabled                                                  #
# --------------------------------------------------------------------- #
def test_default_audit_disabled():
    """NH0: default ON. Expected: both construction paths OFF."""
    a = audit_config.AuditConfig()
    assert a.enabled is False
    assert a.chain_path == ""
    assert a.convergence.window_size == 20
    assert a.convergence.epsilon == 0.05

    b = audit_config.audit_config_from_toml({})
    assert b.enabled is False
    assert b.chain_path == ""

    c = audit_config.audit_config_from_toml({"validator": {}})
    assert c.enabled is False


# --------------------------------------------------------------------- #
# 2 -- TOML [validator.audit] loads                                      #
# --------------------------------------------------------------------- #
def test_toml_loads_audit_section():
    """NH0: section silently ignored. Expected: values flow through."""
    data = {
        "validator": {
            "audit": {"enabled": True, "chain_path": "/tmp/demo/chain.jsonl"}
        }
    }
    cfg = audit_config.audit_config_from_toml(data)
    assert cfg.enabled is True
    assert cfg.chain_path == "/tmp/demo/chain.jsonl"
    # Convergence defaults when section absent.
    assert cfg.convergence.epsilon == 0.05


# --------------------------------------------------------------------- #
# 3 -- TOML [validator.convergence] loads                                #
# --------------------------------------------------------------------- #
def test_toml_loads_convergence_section():
    """NH0: convergence overrides don't flow. Expected: epsilon override lands."""
    data = {
        "validator": {
            "convergence": {
                "window_size": 30,
                "sign_change_threshold": 6,
                "variance_bound": 2e-4,
                "top_quantile_cut": 0.75,
                "min_consecutive_rounds": 15,
                "epsilon": 0.1,
            }
        }
    }
    cfg = audit_config.audit_config_from_toml(data)
    assert cfg.convergence.window_size == 30
    assert cfg.convergence.sign_change_threshold == 6
    assert cfg.convergence.variance_bound == pytest.approx(2e-4)
    assert cfg.convergence.top_quantile_cut == 0.75
    assert cfg.convergence.min_consecutive_rounds == 15
    assert cfg.convergence.epsilon == 0.1


# --------------------------------------------------------------------- #
# 4 -- CLI > TOML > defaults precedence (WIRE-03)                        #
# --------------------------------------------------------------------- #
def test_config_precedence_cli_gt_toml():
    """NH0: CLI doesn't override TOML. Expected: WIRE-03 contract."""
    toml_data = {
        "validator": {
            "audit": {"enabled": False},
            "convergence": {"epsilon": 0.05, "window_size": 20},
        }
    }
    cfg_toml = audit_config.audit_config_from_toml(toml_data)
    assert cfg_toml.convergence.epsilon == 0.05
    assert cfg_toml.enabled is False

    cfg_final = audit_config.apply_audit_cli_overrides(
        cfg_toml,
        {
            "audit.enabled": True,  # CLI overrides TOML false
            "convergence.epsilon": 0.1,  # CLI overrides TOML 0.05
        },
    )
    assert cfg_final.enabled is True
    assert cfg_final.convergence.epsilon == 0.1
    # TOML-only values (no CLI override) preserved.
    assert cfg_final.convergence.window_size == 20


def test_apply_cli_overrides_unknown_key_raises():
    """Unknown CLI keys should fail loudly."""
    cfg = audit_config.AuditConfig()
    with pytest.raises(ValueError, match="Unknown CLI override"):
        audit_config.apply_audit_cli_overrides(cfg, {"audit.bogus": True})


# --------------------------------------------------------------------- #
# 5 + 6 -- validate_config.py rejects bad TOML (production_copy COPY)   #
# --------------------------------------------------------------------- #
def _write_toml(tmp_path, content: str):
    """Write a TOML config file and return the Path."""
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


def test_config_validates_audit_toml(tmp_path):
    """NH0: unknown [validator.audit] keys pass silently.
    Expected: warning emitted at section ``validator.audit``."""
    # v13.1.1 Phase 1103: import production validate_config; audit schema
    # is now baked into the production module (was: production_copy COPY).
    from antigence_subnet import validate_config as vc

    toml_text = textwrap.dedent("""
        [validator.audit]
        enabled = true
        chain_path = "/tmp/ok.jsonl"
        foo = 1
    """)
    p = _write_toml(tmp_path, toml_text)
    issues = vc.validate_config(p)
    unknown = [i for i in issues if "Unknown key 'foo'" in i.message]
    assert unknown, f"Expected unknown-key warning for audit.foo; got {issues}"
    assert unknown[0].section == "validator.audit"


def test_config_validates_convergence_ranges(tmp_path):
    """NH0: epsilon=2.0 slips through. Expected: range error."""
    from antigence_subnet import validate_config as vc

    toml_text = textwrap.dedent("""
        [validator.convergence]
        epsilon = 2.0
        window_size = -5
        variance_bound = 0
    """)
    p = _write_toml(tmp_path, toml_text)
    issues = vc.validate_config(p)
    errors = [i for i in issues if i.level == "error"]
    msgs = " | ".join(i.message for i in errors)
    assert "epsilon must be in (0.0, 1.0)" in msgs, msgs
    assert "window_size must be int > 0" in msgs, msgs
    assert "variance_bound must be > 0" in msgs, msgs


def test_config_validates_accepts_empty(tmp_path):
    """Absence of [validator.audit]/[validator.convergence] MUST be valid
    (STATEPOL-01 backward compat: existing configs still load)."""
    from antigence_subnet import validate_config as vc

    toml_text = textwrap.dedent("""
        [neuron]
        sample_size = 16
    """)
    p = _write_toml(tmp_path, toml_text)
    issues = vc.validate_config(p)
    errors = [i for i in issues if i.level == "error"]
    # No errors from audit/convergence sections (they're absent).
    audit_errors = [
        i for i in errors
        if i.section.startswith(("validator.audit", "validator.convergence"))
    ]
    assert not audit_errors, f"Unexpected audit/conv errors: {audit_errors}"


def test_config_validates_audit_type_errors(tmp_path):
    """Bool/str types enforced for audit.enabled and audit.chain_path."""
    from antigence_subnet import validate_config as vc

    toml_text = textwrap.dedent("""
        [validator.audit]
        enabled = "yes"
        chain_path = 123
    """)
    p = _write_toml(tmp_path, toml_text)
    issues = vc.validate_config(p)
    errors = [i for i in issues if i.level == "error"]
    msgs = " | ".join(i.message for i in errors)
    assert "enabled must be bool" in msgs, msgs
    assert "chain_path must be str" in msgs, msgs


# --------------------------------------------------------------------- #
# 7 -- frozen dataclass contract                                         #
# --------------------------------------------------------------------- #
def test_audit_config_is_frozen():
    """NH0: mutable config allows in-place attack.
    Expected: FrozenInstanceError."""
    a = audit_config.AuditConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.enabled = True  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        a.convergence.epsilon = 0.99  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 8 -- audit_config imports cleanly (no bittensor, no numpy)             #
# --------------------------------------------------------------------- #
def test_audit_config_no_numpy_no_bittensor_import():
    """NH0: transitive import pulls heavy deps.
    Expected: clean subprocess confirms neither is in sys.modules.

    v13.1.1 (Phase 1103): module now lives at
    ``antigence_subnet.validator.audit_config``. The bittensor-free
    and numpy-free contract survives promotion.
    """
    # Locate repo root (parents[2] from tests/validator/<file>.py).
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    prog = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(repo_root)!r})
        from antigence_subnet.validator import audit_config
        assert 'bittensor' not in sys.modules, (
            'bittensor leaked into audit_config import graph: '
            + str(sorted(k for k in sys.modules if 'bittensor' in k))
        )
        assert 'numpy' not in sys.modules, (
            'numpy leaked into audit_config import graph'
        )
        print('OK')
    """)
    result = subprocess.run(
        [sys.executable, "-c", prog],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert result.stdout.strip() == "OK"
