"""
Immune cell type system for the orchestrator pipeline.

Defines the ImmuneCellType Protocol that all immune cells (NK, Dendritic,
B Cell, T Cell) must satisfy. Immune cells are orchestrators that wrap
detector outputs with biological decision logic -- they are NOT detectors
themselves and do not subclass BaseDetector.

NKCell is the real implementation (Phase 32). DendriticCell is the real
implementation (Phase 33). BCell is the real implementation (Phase 37).
BCellStub is a backward-compatible alias for BCell (zero-arg init).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.b_cell import BCell
from antigence_subnet.miner.orchestrator.dendritic_cell import DendriticCell
from antigence_subnet.miner.orchestrator.nk_cell import NKCell


@runtime_checkable
class ImmuneCellType(Protocol):
    """Protocol for immune cell components in the orchestrator pipeline.

    All immune cells must implement process() which takes extracted features,
    the original prompt/output pair, and optional code/context, returning
    either a DetectionResult or None (if the cell has no opinion).

    This is runtime_checkable so the ImmuneCellRegistry can validate
    registrations at runtime via isinstance().
    """

    def process(
        self,
        features: np.ndarray,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult | None: ...


class NKCellStub(NKCell):
    """Backward-compatible alias for NKCell.

    Existing code that creates NKCellStub() with no arguments continues
    to work: initializes NKCell with empty feature_stats (returns None
    for all inputs, matching original stub behavior).

    New code should use NKCell directly.
    """

    def __init__(self) -> None:
        """Initialize as NKCell with empty feature stats (stub behavior)."""
        super().__init__(feature_stats=[])


class DendriticCellStub(DendriticCell):
    """Backward-compatible alias for DendriticCell.

    Existing code that creates DendriticCellStub() with no arguments continues
    to work: initializes DendriticCell with default weights and thresholds
    (pamp_threshold=0.3, standard signal weights, standard tier map).

    New code should use DendriticCell directly.
    """

    def __init__(self) -> None:
        """Initialize as DendriticCell with default config (stub behavior)."""
        super().__init__()


class BCellStub(BCell):
    """Backward-compatible alias for BCell.

    Existing code that creates BCellStub() continues to work:
    initializes BCell with default parameters (empty memory = no influence).
    New code should use BCell directly.

    Same pattern as NKCellStub -> NKCell and DendriticCellStub -> DendriticCell.
    """

    def __init__(self) -> None:
        """Initialize as BCell with default config (stub behavior)."""
        super().__init__()
