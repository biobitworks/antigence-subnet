"""Frozen-dataclass config surface for v13.1 audit chain + convergence.

Exposes two nested, immutable, bittensor-free config dataclasses:

* :class:`ConvergenceConfig` — the six Phase 1001 thresholds used by
  :mod:`convergence_hook` (window_size, sign_change_threshold,
  variance_bound, top_quantile_cut, min_consecutive_rounds, epsilon).
* :class:`AuditConfig` — top-level ``[validator.audit]`` section
  (``enabled`` + ``chain_path``) plus a nested ``ConvergenceConfig``
  for ``[validator.convergence]``.

The config is loaded from a TOML dict (``audit_config_from_toml``) and
then CLI overrides are layered on top (``apply_audit_cli_overrides``).
Precedence (closes **WIRE-03**):

    CLI flags  >  [validator.audit] / [validator.convergence] TOML  >  defaults

Hard contracts (enforced by tests):

* Zero bittensor imports. Importable under ``/tmp/chi-exp-venv`` cleanly.
* Zero numpy imports at module load.
* Both dataclasses are frozen; overrides return a new instance rather
  than mutating in place.
* Absence of either TOML section is valid — defaults are used
  (**STATEPOL-01/02** backward-compat: existing operator configs still
  load without migration).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

__all__ = [
    "AuditConfig",
    "ConvergenceConfig",
    "DEFAULT_CONVERGENCE_CONFIG_DICT",
    "apply_audit_cli_overrides",
    "audit_config_from_toml",
]


# --------------------------------------------------------------------- #
# Defaults (mirror Phase 1001 / 1101 ``DEFAULT_CONVERGENCE_CONFIG``)     #
# --------------------------------------------------------------------- #
DEFAULT_CONVERGENCE_CONFIG_DICT: dict[str, Any] = {
    "window_size": 20,
    "sign_change_threshold": 4,
    "variance_bound": 1e-4,
    "top_quantile_cut": 0.5,
    "min_consecutive_rounds": 10,
    "epsilon": 0.05,
}


@dataclasses.dataclass(frozen=True)
class ConvergenceConfig:
    """Phase 1001 detector thresholds. Frozen; override returns a new instance."""

    window_size: int = 20
    sign_change_threshold: int = 4
    variance_bound: float = 1e-4
    top_quantile_cut: float = 0.5
    min_consecutive_rounds: int = 10
    epsilon: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AuditConfig:
    """``[validator.audit]`` + nested ``[validator.convergence]``. Frozen.

    ``enabled=False`` by default (STATEPOL-02). ``chain_path`` empty string
    means "derive from validator.config.neuron.full_path at runtime"
    (see :mod:`audit_state`).
    """

    enabled: bool = False
    chain_path: str = ""
    convergence: ConvergenceConfig = dataclasses.field(
        default_factory=ConvergenceConfig
    )


# --------------------------------------------------------------------- #
# TOML loader                                                            #
# --------------------------------------------------------------------- #
def _coerce_bool(v: Any, *, field: str) -> bool:
    if isinstance(v, bool):
        return v
    raise TypeError(f"{field} must be bool, got {type(v).__name__}")


def _coerce_str(v: Any, *, field: str) -> str:
    if isinstance(v, str):
        return v
    raise TypeError(f"{field} must be str, got {type(v).__name__}")


def _coerce_int(v: Any, *, field: str) -> int:
    # Reject bool explicitly (subclass of int; silent footgun)
    if isinstance(v, bool):
        raise TypeError(f"{field} must be int, got bool")
    if isinstance(v, int):
        return v
    raise TypeError(f"{field} must be int, got {type(v).__name__}")


def _coerce_float(v: Any, *, field: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{field} must be float, got bool")
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"{field} must be float, got {type(v).__name__}")


def audit_config_from_toml(data: Mapping[str, Any]) -> AuditConfig:
    """Build :class:`AuditConfig` from a loaded TOML mapping.

    ``data`` is the full TOML dict (as returned by ``tomllib.loads``),
    not the ``[validator]`` section. Missing sections fall back to
    defaults. Unknown keys are **not** rejected here — that's
    ``validate_config.py``'s job; this function silently ignores them
    so partial configs continue to load.
    """
    validator = data.get("validator", {}) or {}
    audit_raw = validator.get("audit", {}) or {}
    conv_raw = validator.get("convergence", {}) or {}

    enabled = (
        _coerce_bool(audit_raw["enabled"], field="validator.audit.enabled")
        if "enabled" in audit_raw
        else False
    )
    chain_path = (
        _coerce_str(audit_raw["chain_path"], field="validator.audit.chain_path")
        if "chain_path" in audit_raw
        else ""
    )

    conv_kwargs: dict[str, Any] = {}
    for key, coerce in (
        ("window_size", _coerce_int),
        ("sign_change_threshold", _coerce_int),
        ("variance_bound", _coerce_float),
        ("top_quantile_cut", _coerce_float),
        ("min_consecutive_rounds", _coerce_int),
        ("epsilon", _coerce_float),
    ):
        if key in conv_raw:
            conv_kwargs[key] = coerce(
                conv_raw[key], field=f"validator.convergence.{key}"
            )
    convergence = ConvergenceConfig(**conv_kwargs)

    return AuditConfig(
        enabled=enabled,
        chain_path=chain_path,
        convergence=convergence,
    )


# --------------------------------------------------------------------- #
# CLI overrides                                                          #
# --------------------------------------------------------------------- #
_AUDIT_KEYS = {"audit.enabled", "audit.chain_path"}
_CONV_KEYS = {
    "convergence.window_size",
    "convergence.sign_change_threshold",
    "convergence.variance_bound",
    "convergence.top_quantile_cut",
    "convergence.min_consecutive_rounds",
    "convergence.epsilon",
}


def apply_audit_cli_overrides(
    cfg: AuditConfig, cli_kv: Mapping[str, Any]
) -> AuditConfig:
    """Return a new :class:`AuditConfig` with ``cli_kv`` overrides applied.

    Keys use dotted form matching the CLI flag names:
    ``audit.enabled``, ``audit.chain_path``,
    ``convergence.window_size``, ..., ``convergence.epsilon``.
    Unknown keys raise ``ValueError`` (strict; CLI parsing upstream
    should route only known flags here).

    Precedence (closes **WIRE-03**): this function layers ``cli_kv``
    on top of whatever ``cfg`` already holds (typically a TOML-loaded
    config), so the returned value is ``CLI > TOML > defaults``.
    """
    if not cli_kv:
        return cfg

    audit_kwargs: dict[str, Any] = {
        "enabled": cfg.enabled,
        "chain_path": cfg.chain_path,
    }
    conv_kwargs: dict[str, Any] = dataclasses.asdict(cfg.convergence)

    for raw_key, raw_val in cli_kv.items():
        if raw_key == "audit.enabled":
            audit_kwargs["enabled"] = _coerce_bool(raw_val, field=raw_key)
        elif raw_key == "audit.chain_path":
            audit_kwargs["chain_path"] = _coerce_str(raw_val, field=raw_key)
        elif raw_key == "convergence.window_size":
            conv_kwargs["window_size"] = _coerce_int(raw_val, field=raw_key)
        elif raw_key == "convergence.sign_change_threshold":
            conv_kwargs["sign_change_threshold"] = _coerce_int(
                raw_val, field=raw_key
            )
        elif raw_key == "convergence.variance_bound":
            conv_kwargs["variance_bound"] = _coerce_float(raw_val, field=raw_key)
        elif raw_key == "convergence.top_quantile_cut":
            conv_kwargs["top_quantile_cut"] = _coerce_float(
                raw_val, field=raw_key
            )
        elif raw_key == "convergence.min_consecutive_rounds":
            conv_kwargs["min_consecutive_rounds"] = _coerce_int(
                raw_val, field=raw_key
            )
        elif raw_key == "convergence.epsilon":
            conv_kwargs["epsilon"] = _coerce_float(raw_val, field=raw_key)
        else:
            raise ValueError(
                f"Unknown CLI override key {raw_key!r}; expected one of "
                f"{sorted(_AUDIT_KEYS | _CONV_KEYS)}"
            )

    return AuditConfig(
        enabled=audit_kwargs["enabled"],
        chain_path=audit_kwargs["chain_path"],
        convergence=ConvergenceConfig(**conv_kwargs),
    )
