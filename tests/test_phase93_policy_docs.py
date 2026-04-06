"""Contract tests for Phase 93 ADR and migration guidance artifacts."""

from __future__ import annotations

from pathlib import Path


PHASE_DIR = Path(".planning/phases/93-decision-policy-adr-operator-migration")
ADR_PATH = PHASE_DIR / "93-ADR.md"
MIGRATION_PATH = PHASE_DIR / "93-migration-guide.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase93_adr_chooses_only_measured_policy_layer_default():
    text = _read(ADR_PATH)

    assert "# ADR:" in text
    assert "Accepted" in text
    assert "operator_multiband" in text
    assert "global_threshold" in text
    assert "domain_thresholds" in text
    assert "confidence_modulated_static" in text
    assert "0.493536" in text
    assert "score" in text
    assert "confidence" in text
    assert "detector contract" in text.lower()
    assert "policy layer" in text.lower()
    assert "Phase 92" in text
    assert "Phase 84" in text
    assert "NO-GO" in text
    assert "does not change detector outputs" in text.lower()
    forbidden = ("swarmagent", "swarmpool", "mean3", "median3", "benchmark redesign")
    lowered = text.lower()
    for token in forbidden:
        assert token not in lowered


def test_phase93_migration_guide_explains_score_vs_decision_layers_for_operators():
    text = _read(MIGRATION_PATH)
    lowered = text.lower()

    assert "score layer" in lowered
    assert "decision layer" in lowered
    assert "reward.decision_threshold=0.5" in text
    assert "[validator.policy]" in text
    assert 'mode = "operator_multiband"' in text
    assert 'mode = "global_threshold"' in text
    assert "high_threshold = 0.5" in text
    assert "low_threshold = 0.493536" in text
    assert "min_confidence = 0.6" in text
    assert "allow" in lowered
    assert "review" in lowered
    assert "block" in lowered
    assert "legacy" in lowered
    assert "fallback" in lowered
