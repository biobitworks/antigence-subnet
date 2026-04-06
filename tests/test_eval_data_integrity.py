"""Evaluation data integrity tests for the Antigence Subnet.

Validates that all 4 domain evaluation datasets pass schema, balance,
duplicate ID, and cross-reference checks using the existing
validate_eval_data.py script.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_evaluation_data_valid():
    """Validate all 4 domain evaluation datasets pass integrity checks."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_eval_data.py", "--domain", "all"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"Evaluation data validation failed:\n{result.stdout}\n{result.stderr}"
    )


def test_evaluation_data_minimum_samples():
    """Verify each domain has at least 200 samples (total >= 800)."""
    data_dir = PROJECT_ROOT / "data" / "evaluation"
    domains = ["hallucination", "code_security", "reasoning", "bio"]
    total = 0
    for domain in domains:
        samples_path = data_dir / domain / "samples.json"
        assert samples_path.exists(), f"Missing samples.json for domain: {domain}"
        with open(samples_path) as f:
            data = json.load(f)
        count = len(data["samples"])
        assert count >= 200, f"Domain {domain} has {count} samples (need >= 200)"
        total += count
    assert total >= 800, f"Total samples {total} < 800 minimum"
