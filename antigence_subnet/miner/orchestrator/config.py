"""
Orchestrator configuration dataclass with TOML parsing.

Reads the [miner.orchestrator] section from parsed TOML config dicts.
Uses dict.get() chains throughout so that missing sections produce safe
defaults -- v6.0 configs (which lack [miner.orchestrator]) work unchanged.

Design decisions:
- D-05: enabled defaults to False (v6.0 backward compat)
- D-06: Per-cell sub-tables map to nk_config, dca_config, danger_config, bcell_config
- D-07: dict.get() chain all the way down, never raises KeyError

Per-domain overrides (Phase 36):
- [miner.orchestrator.domains.<domain>] sections with optional keys
- DomainConfig dataclass with nk_z_threshold, dca_pamp_threshold, danger_alpha, danger_enabled
- Validation rejects invalid ranges, warns on unknown domains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from antigence_subnet.protocol import KNOWN_DOMAINS

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Per-domain orchestrator parameter overrides.

    All fields default to None meaning "use global default from
    [miner.orchestrator.nk/dca/danger]".

    Attributes:
        nk_z_threshold: Override NK Cell z-score threshold for this domain.
        dca_pamp_threshold: Override DCA PAMP threshold for this domain.
        danger_alpha: Override Danger Theory modulation alpha for this domain.
        danger_enabled: Override whether Danger Theory is enabled for this domain.
        slm_nk_similarity_threshold: Override SLM NK Cell similarity threshold
            for this domain. Must be in [0.0, 1.0] if present.
    """

    nk_z_threshold: float | None = None
    dca_pamp_threshold: float | None = None
    danger_alpha: float | None = None
    danger_enabled: bool | None = None
    slm_nk_similarity_threshold: float | None = None


def _validate_domain_config(domain_name: str, dc: DomainConfig) -> None:
    """Validate a DomainConfig, raising ValueError for invalid ranges.

    Args:
        domain_name: Domain name (for error messages and KNOWN_DOMAINS check).
        dc: DomainConfig to validate.

    Raises:
        ValueError: If nk_z_threshold < 0, danger_alpha not in [0.0, 1.0],
            or dca_pamp_threshold < 0.
    """
    if dc.nk_z_threshold is not None and dc.nk_z_threshold < 0:
        msg = f"Domain '{domain_name}': nk_z_threshold must be >= 0, got {dc.nk_z_threshold}"
        raise ValueError(msg)

    if dc.danger_alpha is not None and (dc.danger_alpha < 0.0 or dc.danger_alpha > 1.0):
        msg = f"Domain '{domain_name}': danger_alpha must be in [0.0, 1.0], got {dc.danger_alpha}"
        raise ValueError(msg)

    if dc.dca_pamp_threshold is not None and dc.dca_pamp_threshold < 0:
        msg = (
            f"Domain '{domain_name}': dca_pamp_threshold must be >= 0, "
            f"got {dc.dca_pamp_threshold}"
        )
        raise ValueError(msg)

    if dc.slm_nk_similarity_threshold is not None and (
        dc.slm_nk_similarity_threshold < 0.0 or dc.slm_nk_similarity_threshold > 1.0
    ):
        msg = (
            f"Domain '{domain_name}': slm_nk_similarity_threshold must be in "
            f"[0.0, 1.0], got {dc.slm_nk_similarity_threshold}"
        )
        raise ValueError(msg)

    if domain_name not in KNOWN_DOMAINS:
        logger.warning(
            "Unknown domain '%s' in [miner.orchestrator.domains]. "
            "Known domains: %s",
            domain_name,
            ", ".join(sorted(KNOWN_DOMAINS)),
        )


@dataclass
class SLMNKConfig:
    """Configuration for the SLM-powered NK Cell.

    Parsed from the ``[miner.orchestrator.slm_nk]`` TOML section.

    Attributes:
        enabled: Whether the SLM NK Cell is active. Defaults to True
            so miners who enable the orchestrator get semantic coverage
            automatically.
        similarity_threshold: Semantic similarity threshold below which
            outputs are flagged as anomalous. Defaults to 0.3 (per D-05).
    """

    enabled: bool = True
    similarity_threshold: float = 0.3

    @classmethod
    def from_toml_raw(cls, toml_raw: dict[str, Any]) -> SLMNKConfig:
        """Create SLMNKConfig from a parsed TOML dict.

        Reads ``[miner.orchestrator.slm_nk]`` section. If the section
        is missing, safe defaults apply.

        Args:
            toml_raw: Full parsed TOML config dict.

        Returns:
            SLMNKConfig with values from TOML or defaults.
        """
        slm_nk = toml_raw.get("miner", {}).get("orchestrator", {}).get("slm_nk", {})
        return cls(
            enabled=slm_nk.get("enabled", True),
            similarity_threshold=slm_nk.get("similarity_threshold", 0.3),
        )


def _validate_slm_nk_config(config: SLMNKConfig) -> None:
    """Validate SLMNKConfig, raising ValueError for invalid ranges.

    Args:
        config: SLMNKConfig to validate.

    Raises:
        ValueError: If similarity_threshold not in [0.0, 1.0].
    """
    if config.similarity_threshold < 0.0 or config.similarity_threshold > 1.0:
        msg = (
            f"similarity_threshold must be in [0.0, 1.0], "
            f"got {config.similarity_threshold}"
        )
        raise ValueError(msg)


def _validate_dca_adaptive_config(dca_config: dict[str, Any]) -> None:
    """Validate DCA adaptive configuration fields (D-06).

    Checks ``adapt_alpha`` key in the dca_config dict. Raises ValueError
    for invalid values. The ``adaptive`` flag (bool, default false) is the
    opt-in switch; ``adapt_alpha`` (float, default 0.1) is the EMA rate.

    Args:
        dca_config: The ``[miner.orchestrator.dca]`` dict.

    Raises:
        ValueError: If adapt_alpha is present and <= 0.0 or > 1.0.
    """
    adapt_alpha = dca_config.get("adapt_alpha")
    if adapt_alpha is not None and (adapt_alpha <= 0.0 or adapt_alpha > 1.0):
        msg = f"adapt_alpha must be in (0.0, 1.0], got {adapt_alpha}"
        raise ValueError(msg)


@dataclass
class FeedbackConfig:
    """Configuration for validator feedback loop.

    Parsed from ``[miner.orchestrator.feedback]`` TOML section.

    Attributes:
        enabled: Whether feedback tracking is active. Default False (opt-in per D-05).
        lookback_rounds: Number of rounds for weight correlation window. Default 5.
    """

    enabled: bool = False
    lookback_rounds: int = 5

    @classmethod
    def from_toml_raw(cls, toml_raw: dict[str, Any]) -> FeedbackConfig:
        """Create FeedbackConfig from a parsed TOML dict.

        Reads ``[miner.orchestrator.feedback]`` section. If the section
        is missing, safe defaults apply.

        Args:
            toml_raw: Full parsed TOML config dict.

        Returns:
            FeedbackConfig with values from TOML or defaults.
        """
        feedback = toml_raw.get("miner", {}).get("orchestrator", {}).get("feedback", {})
        return cls(
            enabled=feedback.get("enabled", False),
            lookback_rounds=feedback.get("lookback_rounds", 5),
        )


def _validate_feedback_config(config: FeedbackConfig) -> None:
    """Validate FeedbackConfig, raising ValueError for invalid ranges.

    Args:
        config: FeedbackConfig to validate.

    Raises:
        ValueError: If lookback_rounds < 1.
    """
    if config.lookback_rounds < 1:
        msg = f"lookback_rounds must be >= 1, got {config.lookback_rounds}"
        raise ValueError(msg)


@dataclass
class ModelConfig:
    """Configuration for the miner model manager.

    Parsed from the ``[miner.model]`` TOML section. This is a sibling of
    ``[miner.orchestrator]``, not nested inside it -- the model manager
    is a shared resource used by multiple immune cells.

    Attributes:
        model_name: HuggingFace model name for sentence-transformers.
            Defaults to all-MiniLM-L6-v2 (384-dim output).
        cache_dir: Directory to cache downloaded models. None uses
            sentence-transformers default (~/.cache/torch/sentence_transformers).
        device: Device for inference. "auto" detects GPU via torch.cuda,
            "cpu" forces CPU, "cuda" forces GPU.
    """

    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str | None = None
    device: str = "auto"

    @classmethod
    def from_toml_raw(cls, toml_raw: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from a parsed TOML dict.

        Reads ``[miner.model]`` section. If the section is missing,
        safe defaults apply -- v8.0 configs work unchanged.

        Args:
            toml_raw: Full parsed TOML config dict.

        Returns:
            ModelConfig with values from TOML or defaults.
        """
        model = toml_raw.get("miner", {}).get("model", {})
        return cls(
            model_name=model.get("model_name", "all-MiniLM-L6-v2"),
            cache_dir=model.get("cache_dir"),
            device=model.get("device", "auto"),
        )


@dataclass
class OrchestratorConfig:
    """Configuration for the immune cell orchestrator pipeline.

    Attributes:
        enabled: Whether the orchestrator pipeline is active. Defaults to
            False so v6.0 miners without [miner.orchestrator] config are
            unaffected.
        nk_config: Sub-config dict for NK Cell parameters (z_threshold, etc).
        dca_config: Sub-config dict for DCA parameters (escalation_tiers, etc).
        danger_config: Sub-config dict for Danger Theory parameters
            (pamp_weight, danger_weight, etc).
        bcell_config: Sub-config dict for B Cell parameters. Keys:
            max_memory (int), k (int), bcell_weight (float),
            half_life (float), eviction_threshold (float), jitter_sigma (float),
            memory_dir (str), embedding_mode (bool), embedding_sigma (float).
        domain_configs: Per-domain orchestrator parameter overrides parsed
            from [miner.orchestrator.domains.<domain>] TOML sections.
            Empty dict when no domain sections exist (v7.0 backward compat).
    """

    enabled: bool = False
    nk_config: dict[str, Any] = field(default_factory=dict)
    dca_config: dict[str, Any] = field(default_factory=dict)
    danger_config: dict[str, Any] = field(default_factory=dict)
    bcell_config: dict[str, Any] = field(default_factory=dict)
    domain_configs: dict[str, DomainConfig] = field(default_factory=dict)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    slm_nk_config: SLMNKConfig = field(default_factory=SLMNKConfig)
    feedback_config: FeedbackConfig = field(default_factory=FeedbackConfig)

    def get_domain_config(self, domain: str) -> DomainConfig | None:
        """Return the DomainConfig for a domain, or None if not configured.

        Args:
            domain: Domain name (e.g., 'hallucination', 'code_security').

        Returns:
            DomainConfig with overrides, or None if domain has no overrides.
        """
        return self.domain_configs.get(domain)

    @classmethod
    def from_toml_raw(cls, toml_raw: dict[str, Any]) -> OrchestratorConfig:
        """Create OrchestratorConfig from a parsed TOML dict.

        Reads [miner.orchestrator] section. If the section or any sub-table
        is missing, safe defaults apply. Parses [miner.orchestrator.domains.*]
        sub-tables into DomainConfig instances with validation.

        Args:
            toml_raw: Full parsed TOML config dict (e.g., from load_toml_config).

        Returns:
            OrchestratorConfig with values from TOML or defaults.

        Raises:
            ValueError: If any domain config has invalid parameter ranges.
        """
        orchestrator = toml_raw.get("miner", {}).get("orchestrator", {})

        # Parse per-domain overrides
        domains_raw = orchestrator.get("domains", {})
        domain_configs: dict[str, DomainConfig] = {}
        for domain_name, domain_dict in domains_raw.items():
            dc = DomainConfig(
                nk_z_threshold=domain_dict.get("nk_z_threshold"),
                dca_pamp_threshold=domain_dict.get("dca_pamp_threshold"),
                danger_alpha=domain_dict.get("danger_alpha"),
                danger_enabled=domain_dict.get("danger_enabled"),
                slm_nk_similarity_threshold=domain_dict.get("slm_nk_similarity_threshold"),
            )
            _validate_domain_config(domain_name, dc)
            domain_configs[domain_name] = dc

        # Parse and validate SLM NK config
        slm_nk_config = SLMNKConfig.from_toml_raw(toml_raw)
        _validate_slm_nk_config(slm_nk_config)

        # Validate DCA adaptive config (D-06)
        dca_config = orchestrator.get("dca", {})
        _validate_dca_adaptive_config(dca_config)

        # Parse and validate feedback config (Phase 45)
        feedback_config = FeedbackConfig.from_toml_raw(toml_raw)
        _validate_feedback_config(feedback_config)

        return cls(
            enabled=orchestrator.get("enabled", False),
            nk_config=orchestrator.get("nk", {}),
            dca_config=dca_config,
            danger_config=orchestrator.get("danger", {}),
            bcell_config=orchestrator.get("bcell", {}),
            domain_configs=domain_configs,
            model_config=ModelConfig.from_toml_raw(toml_raw),
            slm_nk_config=slm_nk_config,
            feedback_config=feedback_config,
        )
