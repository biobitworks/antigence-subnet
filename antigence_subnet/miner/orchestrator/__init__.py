"""
Immune cell orchestrator package.

Provides the type system, registry, configuration, and pipeline coordination
for immune-inspired detection components (NK Cell, Dendritic Cell, B Cell).

Exports:
    ImmuneCellType: Protocol that all immune cells must satisfy.
    NKCell: Real NK Cell implementation (Phase 32).
    FeatureStatistics: Per-feature statistics dataclass for NKCell.
    NKCellStub: Backward-compatible alias for NKCell (zero-arg init).
    DendriticCell: Real DCA implementation (Phase 33).
    DCAResult: Frozen dataclass returned by DendriticCell.classify().
    DendriticCellStub: Backward-compatible alias for DendriticCell (zero-arg init).
    BCell: Real B Cell adaptive memory implementation (Phase 37).
    BCellStub: Backward-compatible alias for BCell (zero-arg init).
    ImmuneCellRegistry: Per-instance registry for cell lookup.
    OrchestratorConfig: Dataclass for [miner.orchestrator] TOML config.
    ModelConfig: Dataclass for [miner.model] TOML config (Phase 41).
    ModelManager: SLM model lifecycle with embed()/score() API (Phase 41).
    SLMNKConfig: Dataclass for [miner.orchestrator.slm_nk] TOML config (Phase 42).
    SLMNKCell: SLM-powered semantic anomaly gate (Phase 42).
    DangerTheoryModulator: Two-signal costimulation score modulator (Phase 34).
    ImmuneOrchestrator: Central pipeline coordinator (Phase 34, Phase 37 BCell, Phase 43 embedding).
    ValidatorFeedbackTracker: Metagraph weight feedback loop (Phase 45).
    DetectionRecord: Per-detection record for feedback correlation (Phase 45).
    FeedbackConfig: Feedback loop configuration dataclass (Phase 45).
"""

from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager
from antigence_subnet.miner.orchestrator.b_cell import BCell
from antigence_subnet.miner.orchestrator.cells import (
    BCellStub,
    DendriticCell,
    DendriticCellStub,
    ImmuneCellType,
    NKCell,
    NKCellStub,
)
from antigence_subnet.miner.orchestrator.config import (
    FeedbackConfig,
    ModelConfig,
    OrchestratorConfig,
    SLMNKConfig,
)
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
from antigence_subnet.miner.orchestrator.dendritic_cell import DCAResult
from antigence_subnet.miner.orchestrator.feedback import (
    DetectionRecord,
    ValidatorFeedbackTracker,
)
from antigence_subnet.miner.orchestrator.model_manager import ModelManager
from antigence_subnet.miner.orchestrator.nk_cell import FeatureStatistics
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
from antigence_subnet.miner.orchestrator.registry import ImmuneCellRegistry
from antigence_subnet.miner.orchestrator.slm_nk_cell import SLMNKCell
from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry

__all__ = [
    "AdaptiveWeightManager",
    "BCell",
    "BCellStub",
    "DCAResult",
    "DangerTheoryModulator",
    "DendriticCell",
    "DendriticCellStub",
    "DetectionRecord",
    "FeatureStatistics",
    "FeedbackConfig",
    "ImmuneCellRegistry",
    "ImmuneCellType",
    "ImmuneOrchestrator",
    "MinerTelemetry",
    "ModelConfig",
    "ModelManager",
    "NKCell",
    "NKCellStub",
    "OrchestratorConfig",
    "SLMNKCell",
    "SLMNKConfig",
    "ValidatorFeedbackTracker",
]
