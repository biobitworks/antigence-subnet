#!/usr/bin/env python3
"""Combine detector, orchestrator, and Ollama baselines into a single v9.2 baseline JSON.

Reads from:
  - data/benchmarks/v9.2-baseline-detectors.json
  - data/benchmarks/v9.2-baseline-orchestrator.json
  - data/benchmarks/v9.2-baseline-ollama-*.json (glob, may not exist yet)

Produces:
  - data/benchmarks/v9.2-baseline.json

Gracefully handles missing Ollama files (warns but produces partial baseline).
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

DETECTOR_PATH = DATA_DIR / "v9.2-baseline-detectors.json"
ORCHESTRATOR_PATH = DATA_DIR / "v9.2-baseline-orchestrator.json"
OLLAMA_GLOB = str(DATA_DIR / "v9.2-baseline-ollama-*.json")
OUTPUT_PATH = DATA_DIR / "v9.2-baseline.json"


def load_json(path: Path, label: str) -> dict | list | None:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        print(f"  WARNING: {label} not found at {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Combine v9.2 baselines into single JSON")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output path for combined baseline")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ANTIGENCE SUBNET -- COMBINE v9.2 BASELINES")
    print("=" * 60)

    combined = {
        "version": "9.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "sections": {},
    }

    # Load detector baseline
    print("\n  Loading detector baseline...")
    detectors = load_json(DETECTOR_PATH, "Detector baseline")
    if detectors is not None:
        combined["sections"]["detectors"] = {
            "source": str(DETECTOR_PATH.relative_to(PROJECT_ROOT)),
            "entry_count": len(detectors),
            "data": detectors,
        }
        print(f"    Loaded {len(detectors)} detector entries")
    else:
        print("    SKIPPED (file not found)")

    # Load orchestrator baseline
    print("  Loading orchestrator baseline...")
    orchestrator = load_json(ORCHESTRATOR_PATH, "Orchestrator baseline")
    if orchestrator is not None:
        domains = orchestrator.get("domains", {})
        combined["sections"]["orchestrator"] = {
            "source": str(ORCHESTRATOR_PATH.relative_to(PROJECT_ROOT)),
            "domain_count": len(domains),
            "data": orchestrator,
        }
        print(f"    Loaded {len(domains)} orchestrator domains")
    else:
        print("    SKIPPED (file not found)")

    # Load Ollama baselines (glob)
    print("  Loading Ollama baselines...")
    ollama_files = sorted(glob.glob(OLLAMA_GLOB))
    if ollama_files:
        ollama_data = {}
        for fpath in ollama_files:
            p = Path(fpath)
            # Extract domain from filename: v9.2-baseline-ollama-{domain}.json
            domain = p.stem.replace("v9.2-baseline-ollama-", "")
            data = load_json(p, f"Ollama {domain}")
            if data is not None:
                ollama_data[domain] = data
                print(f"    Loaded Ollama baseline for domain: {domain}")

        if ollama_data:
            combined["sections"]["ollama"] = {
                "source_pattern": "data/benchmarks/v9.2-baseline-ollama-*.json",
                "domain_count": len(ollama_data),
                "domains": ollama_data,
            }
    else:
        print("    WARNING: No Ollama baseline files found (expected after Plan 02)")

    # Summary
    sections = list(combined["sections"].keys())
    print(f"\n  Combined sections: {sections}")
    print(f"  Missing sections: {[s for s in ['detectors', 'orchestrator', 'ollama'] if s not in sections]}")

    # Write output
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Output: {output_path}")

    # Exit code: 0 if at least detectors + orchestrator present
    if "detectors" in combined["sections"] and "orchestrator" in combined["sections"]:
        print("\n  Status: OK (core baselines present)")
        return 0
    else:
        print("\n  Status: PARTIAL (missing core baselines)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
