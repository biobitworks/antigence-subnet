"""ImmuneOrchestrator -- central coordinator for the immune detection pipeline.

Wires NK Cell gate, DCA signal classification, detector execution,
Danger Theory modulation, and B Cell memory influence into a single
async process() call.

Pipeline (per D-06, extended Phase 43):
1. Extract dendritic features from output text
2. NK Cell gate (rule-based + SLM) -- if triggered, return immediately
3. DCA classify -- get maturation state and recommended tier
4. Select and run detectors for the tier
5. Apply Danger Theory modulation
6. B Cell memory influence (embedding-aware when model_manager available)
7. Return final DetectionResult

The orchestrator does NOT own detector training or state persistence.
Detectors are provided pre-fitted by the miner initialization code.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.dendritic_features import DendriticFeatureExtractor
from antigence_subnet.miner.ensemble import ensemble_detect
from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager
from antigence_subnet.miner.orchestrator.b_cell import BCell
from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
from antigence_subnet.miner.orchestrator.dendritic_cell import DendriticCell
from antigence_subnet.miner.orchestrator.feedback import ValidatorFeedbackTracker
from antigence_subnet.miner.orchestrator.model_manager import ModelManager
from antigence_subnet.miner.orchestrator.nk_cell import NKCell
from antigence_subnet.miner.orchestrator.slm_nk_cell import SLMNKCell
from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry

logger = logging.getLogger(__name__)


class ImmuneOrchestrator:
    """Central coordinator for the immune detection pipeline.

    Pipeline:
    1. Extract dendritic features from output text
    2. NK Cell gate (rule-based + SLM) -- if triggered, return immediately
    3. DCA classify -- get maturation state and recommended tier
    4. Select and run detectors for the tier
    5. Apply Danger Theory modulation
    6. B Cell memory influence
    7. Return final DetectionResult
    """

    def __init__(
        self,
        feature_extractor: DendriticFeatureExtractor,
        nk_cell: NKCell,
        dendritic_cell: DendriticCell,
        danger_modulator: DangerTheoryModulator,
        detectors: dict[str, list[BaseDetector]],
        config: OrchestratorConfig | None = None,
        slm_nk_cell: SLMNKCell | None = None,
        b_cell: BCell | None = None,
        adaptive_weights: AdaptiveWeightManager | None = None,
        model_manager: ModelManager | None = None,
        telemetry: MinerTelemetry | None = None,
        feedback: ValidatorFeedbackTracker | None = None,
    ) -> None:
        """Initialize the immune orchestrator with pipeline components.

        Args:
            feature_extractor: DendriticFeatureExtractor for text -> 10-dim features.
            nk_cell: NKCell for fast-path anomaly gating.
            dendritic_cell: DendriticCell for signal classification and tier routing.
            danger_modulator: DangerTheoryModulator for costimulation modulation.
            detectors: Dict mapping domain -> list of fitted BaseDetector instances.
            config: Optional OrchestratorConfig for per-domain parameter overrides.
            slm_nk_cell: Optional SLMNKCell for semantic fast-path gate (Phase 42).
            b_cell: Optional BCell for adaptive memory influence (Phase 43).
            adaptive_weights: Optional AdaptiveWeightManager for DCA weight learning (Phase 44).
            model_manager: Optional ModelManager for computing embeddings (Phase 43).
                Used to compute 384-dim embeddings passed to BCell.influence().
            telemetry: Optional MinerTelemetry for detection telemetry logging (Phase 44).
            feedback: Optional ValidatorFeedbackTracker for metagraph weight
                feedback loop (Phase 45).
        """
        self._extractor = feature_extractor
        self._nk_cell = nk_cell
        self._dc = dendritic_cell
        self._danger = danger_modulator
        self._detectors = detectors
        self._config = config
        self._slm_nk = slm_nk_cell
        self._b_cell = b_cell
        self._adaptive_weights = adaptive_weights
        self._model_manager = model_manager
        self._telemetry = telemetry
        self._feedback = feedback

    @classmethod
    def from_config(
        cls,
        config: OrchestratorConfig,
        detectors: dict[str, list[BaseDetector]],
        audit_dir: str | None = None,
        telemetry: MinerTelemetry | None = None,
    ) -> ImmuneOrchestrator:
        """Factory that creates orchestrator from OrchestratorConfig.

        Creates NKCell from audit JSON if audit_dir provided (else empty stats),
        DendriticCell from dca_config, DangerTheoryModulator from danger_config.
        Creates ModelManager from model_config and injects into BCell when
        embedding_mode is configured (Phase 43).
        Creates AdaptiveWeightManager when adaptive=true in dca_config (Phase 44).

        Args:
            config: OrchestratorConfig with nk_config, dca_config, danger_config,
                bcell_config, and model_config.
            detectors: Dict mapping domain -> list of fitted BaseDetector instances.
            audit_dir: Optional path to directory containing per-domain audit JSONs.
                If None, NKCell is created with empty feature_stats (no fast-path).
            telemetry: Optional MinerTelemetry for detection telemetry logging (Phase 44).

        Returns:
            Configured ImmuneOrchestrator instance.
        """
        extractor = DendriticFeatureExtractor()

        # NK Cell: from audit JSON if available, else stub with empty stats
        if audit_dir is not None:
            # Try to load audit data for any available domain
            import json
            from pathlib import Path

            audit_path = Path(audit_dir)
            nk_cell: NKCell | None = None
            for domain_file in sorted(audit_path.glob("*.json")):
                try:
                    nk_cell = NKCell.from_audit_json(
                        str(domain_file),
                        z_threshold=config.nk_config.get("z_threshold", 3.0),
                    )
                    break
                except (json.JSONDecodeError, KeyError):
                    continue
            if nk_cell is None:
                nk_cell = NKCell(feature_stats=[])
        else:
            nk_cell = NKCell(feature_stats=[])

        # Adaptive weights: create if adaptive=true in dca_config (Phase 44)
        weight_manager: AdaptiveWeightManager | None = None
        if config.dca_config.get("adaptive", False):
            alpha = config.dca_config.get("adapt_alpha", 0.1)
            weight_manager = AdaptiveWeightManager(alpha=alpha)
            weight_manager.load(domain="default")

        # DCA: from dca_config (with optional weight_manager)
        dc = DendriticCell.from_config(config.dca_config, weight_manager=weight_manager)

        # Danger modulator: from danger_config
        danger = DangerTheoryModulator.from_config(config.danger_config)

        # ModelManager: create if model_config is available (Phase 41+)
        model_manager = None
        if hasattr(config, "model_config"):
            model_manager = ModelManager(config=config.model_config)

        # SLM NK Cell: from slm_nk_config + model_manager (Phase 42)
        slm_nk_cell_instance: SLMNKCell | None = None
        if config.slm_nk_config.enabled and model_manager is not None:
            slm_nk_cell_instance = SLMNKCell(
                model_manager=model_manager,
                similarity_threshold=config.slm_nk_config.similarity_threshold,
                enabled=config.slm_nk_config.enabled,
            )

        # BCell: create from bcell_config with model_manager injection (Phase 43)
        b_cell = None
        if config.bcell_config:
            if (
                model_manager is not None
                and config.bcell_config.get("embedding_mode", False)
            ):
                # Create BCell with model_manager for embedding mode
                b_cell = BCell(
                    model_manager=model_manager,
                    **config.bcell_config,
                )
            else:
                b_cell = BCell.from_config(config.bcell_config)

        # Feedback tracker (Phase 45)
        feedback_tracker: ValidatorFeedbackTracker | None = None
        if config.feedback_config.enabled:
            feedback_tracker = ValidatorFeedbackTracker(
                lookback_rounds=config.feedback_config.lookback_rounds,
                enabled=True,
            )

        return cls(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors=detectors,
            config=config,
            slm_nk_cell=slm_nk_cell_instance,
            b_cell=b_cell,
            adaptive_weights=weight_manager,
            model_manager=model_manager,
            telemetry=telemetry,
            feedback=feedback_tracker,
        )

    def save_state(self, domain: str = "default") -> None:
        """Persist adaptive weight state to disk.

        Called by miner shutdown hooks to save adapted weights across
        restarts. No-op when adaptive weights are not enabled.

        Args:
            domain: Domain name for weight file isolation. Default "default".
        """
        if self._adaptive_weights is not None:
            self._adaptive_weights.save(domain)
            logger.info("Saved adaptive weight state for domain '%s'", domain)

    def process_feedback(
        self,
        current_weight: float,
        avg_score: float,
        detection_count: int,
        domain: str = "all",
    ) -> float:
        """Process one feedback round: record weight, compute signal, apply to BCell and DCA.

        Called by the miner after each evaluation round with the miner's
        current metagraph weight. Returns the feedback signal for logging.

        Per D-02: positive signal reinforces BCell, negative decays.
        Per D-04: signal also fed to DCA as importance signal.
        Per D-06: returns 0.0 when disabled or metagraph unavailable.

        Args:
            current_weight: Miner's current weight in metagraph.
            avg_score: Average detection score this round.
            detection_count: Number of detections this round.
            domain: Domain for this round.

        Returns:
            Feedback signal in [-1.0, 1.0]. 0.0 when disabled.
        """
        if self._feedback is None or not self._feedback.enabled:
            return 0.0

        signal = self._feedback.record_round(
            current_weight=current_weight,
            avg_score=avg_score,
            detection_count=detection_count,
            domain=domain,
        )

        if signal == 0.0:
            return 0.0

        # Apply to BCell (per D-03)
        if self._b_cell is not None:
            self._feedback.apply_feedback_to_bcell_correlated(self._b_cell, signal)

        # Apply to DCA (per D-04)
        if self._adaptive_weights is not None:
            # Use recent detection features for feature-specific feedback
            recent = self._feedback.get_recent_detections()
            if recent:
                avg_features = np.mean([d.features for d in recent], axis=0)
                self._feedback.apply_to_dca(
                    self._adaptive_weights,
                    signal,
                    features=avg_features,
                )
            else:
                self._feedback.apply_to_dca(self._adaptive_weights, signal)

        logger.info(
            "Feedback processed [domain=%s]: signal=%.4f, weight=%.6f, "
            "bcell=%s, dca=%s",
            domain,
            signal,
            current_weight,
            self._b_cell is not None,
            self._adaptive_weights is not None,
        )

        return signal

    async def process(
        self,
        prompt: str,
        output: str,
        domain: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run the full immune pipeline.

        Steps:
        1. Extract features from output (or output+code for code domains)
        2. NK Cell gate check (rule-based + SLM)
        3. DCA signal classification and tier routing
        4. Run detectors from selected tier (or all if tier is empty)
        5. Apply Danger Theory modulation
        6. B Cell memory influence (embedding-aware when model_manager available)
        7. Return DetectionResult

        Args:
            prompt: Original prompt text.
            output: AI-generated output text.
            domain: Domain string (e.g., 'hallucination', 'code_security').
            code: Optional code content for code_security domain.
            context: Optional JSON-serialized metadata.

        Returns:
            DetectionResult with modulated score, confidence, and anomaly_type.
        """
        # Step 1: Feature extraction
        text = output if code is None else f"{output}\n{code}"
        features = self._extractor.extract(text)

        # Resolve domain-specific overrides (Phase 36)
        domain_cfg = None
        if self._config is not None:
            domain_cfg = self._config.get_domain_config(domain)

        # Step 2a: Rule-based NK Cell fast-path (cheaper, checked first)
        nk_kwargs: dict[str, Any] = {}
        if domain_cfg and domain_cfg.nk_z_threshold is not None:
            nk_kwargs["z_threshold"] = domain_cfg.nk_z_threshold
        nk_result = self._nk_cell.process(
            features, prompt, output, code, context, **nk_kwargs,
        )
        if nk_result is not None:
            return nk_result

        # Step 2b: SLM NK Cell fast-path (semantic, checked second per D-03)
        if self._slm_nk is not None:
            slm_nk_kwargs: dict[str, float] = {}
            if domain_cfg and domain_cfg.slm_nk_similarity_threshold is not None:
                slm_nk_kwargs["similarity_threshold"] = domain_cfg.slm_nk_similarity_threshold
            slm_nk_result = self._slm_nk.process(
                features, prompt, output, code, context, **slm_nk_kwargs,
            )
            if slm_nk_result is not None:
                return slm_nk_result

        # Step 3: DCA classification (with domain override)
        dca_kwargs: dict[str, float] = {}
        if domain_cfg and domain_cfg.dca_pamp_threshold is not None:
            dca_kwargs["pamp_threshold"] = domain_cfg.dca_pamp_threshold
        dca_result = self._dc.classify(features, **dca_kwargs)

        # Step 4: Select detectors for tier
        domain_detectors = self._detectors.get(domain, [])
        if dca_result.recommended_tier:
            # Filter to tier-specified detectors by matching class name
            tier_detectors = [
                d for d in domain_detectors
                if any(key in type(d).__name__.lower() for key in dca_result.recommended_tier)
            ]
            if not tier_detectors:
                tier_detectors = domain_detectors  # fallback to all
        else:
            tier_detectors = domain_detectors  # empty tier = full ensemble

        # Step 5: Run ensemble
        result = await ensemble_detect(tier_detectors, prompt, output, code, context)

        # Step 5a: Record detection for feedback correlation (Phase 45)
        if self._feedback is not None and self._feedback.enabled:
            self._feedback.record_detection(
                features=features,
                anomaly_score=result.score,
                domain=domain,
            )

        # Step 5b: Adaptive weight update (Phase 44)
        if self._adaptive_weights is not None:
            old_weights = self._adaptive_weights.get_weights()
            outcome = 1.0 if result.score > 0.5 else -1.0
            self._adaptive_weights.adapt(features, outcome)
            new_weights = self._adaptive_weights.get_weights()
            old_weight_summary = {
                cat: {name: round(weight, 4) for name, (_, weight) in weights.items()}
                for cat, weights in old_weights.items()
            }
            new_weight_summary = {
                cat: {name: round(weight, 4) for name, (_, weight) in weights.items()}
                for cat, weights in new_weights.items()
            }
            logger.debug(
                "DCA weight adaptation [domain=%s]: old=%s -> new=%s",
                domain,
                old_weight_summary,
                new_weight_summary,
            )
            if self._telemetry is not None:
                self._telemetry.record(domain, result.score, result.confidence)

        # Step 6: Danger modulation (with domain override)
        danger_kwargs: dict[str, Any] = {}
        if domain_cfg and domain_cfg.danger_alpha is not None:
            danger_kwargs["alpha"] = domain_cfg.danger_alpha
        if domain_cfg and domain_cfg.danger_enabled is not None:
            danger_kwargs["enabled"] = domain_cfg.danger_enabled
        result = self._danger.modulate_result(result, features, **danger_kwargs)

        # Step 7: B Cell memory influence (per D-09, Phase 43 embedding-aware)
        if self._b_cell is not None:
            embedding = None
            if self._model_manager is not None and getattr(
                self._b_cell, "_embedding_mode", False
            ):
                try:
                    embedding = self._model_manager.embed(output)
                except Exception:
                    logger.warning(
                        "ModelManager.embed() failed, falling back to "
                        "feature-only BCell influence"
                    )
            result = self._b_cell.influence(features, result, embedding=embedding)

        return result
