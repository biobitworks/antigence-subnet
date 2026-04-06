"""
Training data loading utility for miner detectors.

Loads normal (self) samples from evaluation data directories for one-class
anomaly detection training. Reads samples.json for data and manifest.json
for ground truth labels, returning only normal samples for detector fitting.
"""

import json
from pathlib import Path


def load_training_samples(data_dir: str, domain: str) -> list[dict]:
    """Load normal (self) samples for detector training.

    One-class anomaly detection: fit on normal samples only.
    Reads samples.json for data and manifest.json for labels.

    Args:
        data_dir: Path to evaluation data root (e.g., "data/evaluation").
        domain: Domain subdirectory (e.g., "hallucination").

    Returns:
        List of sample dicts with prompt, output, domain keys.
        Only samples labeled "normal" in manifest are returned.

    Raises:
        FileNotFoundError: If samples.json or manifest.json not found.
    """
    samples_path = Path(data_dir) / domain / "samples.json"
    manifest_path = Path(data_dir) / domain / "manifest.json"

    with open(samples_path) as f:
        all_samples = json.load(f)["samples"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    # One-class: fit on NORMAL samples only (self-tolerance)
    normal_samples = [
        s for s in all_samples
        if manifest.get(s["id"], {}).get("ground_truth_label") == "normal"
    ]
    return normal_samples
