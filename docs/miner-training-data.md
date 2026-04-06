# Miner Training Data Guide

## Overview

The Antigence subnet uses an immune-inspired **self/non-self** detection paradigm.
Miners run anomaly detectors ("antibodies") that learn what **normal** AI output looks like (self),
then flag deviations as anomalous (non-self).

**Key principle:** One-class anomaly detection. Detectors are trained on **normal samples only** (self-tolerance).
Anomalous outputs are anything that deviates from the learned normal distribution.

- **Self** = correct, supported, faithful AI outputs
- **Non-self** = hallucinated, refuted, unfaithful AI outputs

Fit your detector on self only. Non-self is what the detector learns to reject.

## Seed Data

The subnet ships with 30 bootstrap samples in `data/evaluation/hallucination/`:

- `samples.json` -- 30 prompt/output pairs (15 normal, 15 anomalous)
- `manifest.json` -- ground truth labels and honeypot flags

The reference miner loads these at startup via `load_training_samples()`, which
filters to the 15 normal samples for one-class fitting. This is sufficient for
testing but too small for competitive detection.

To improve your miner's accuracy, train on larger public datasets listed below.

## Primary Datasets

| Dataset | Size | License | Source |
|---------|------|---------|--------|
| TruthfulQA | 817 questions | Apache 2.0 | [HuggingFace: truthful_qa](https://huggingface.co/datasets/truthful_qa) |
| HaluEval | 35,000 samples | MIT | [HuggingFace: pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) |
| FEVER | 185,445 claims | CC-BY-SA 3.0 | [fever.ai](https://fever.ai/) |
| FaithDial | 50,761 turns | Apache 2.0 | [HuggingFace: McGill-NLP/FaithDial](https://huggingface.co/datasets/McGill-NLP/FaithDial) |
| SummEval | 1,600 samples | MIT | [GitHub: Yale-LILY/SummEval](https://github.com/Yale-LILY/SummEval) |

## Secondary Datasets

| Dataset | Size | License | Source |
|---------|------|---------|--------|
| RAGTruth | 18,000 samples | MIT | [HuggingFace: RAGTruth](https://huggingface.co/datasets/wandbdata/RAGTruth) |
| DefAn | 20,000+ samples | CC-BY 4.0 | [HuggingFace: DefAn](https://huggingface.co/datasets/anonysubmission/DefAn) |
| Vectara Hallucination Leaderboard | 7,700+ samples | Apache 2.0 | [HuggingFace: vectara/hallucination_evaluation_dataset](https://huggingface.co/datasets/vectara/hallucination_evaluation_dataset) |

## Usage Notes

### Recommended Starting Points

- **TF-IDF detectors (IsolationForest):** Start with **TruthfulQA** -- small enough for fast iteration, good mix of factual domains. Scale up to **HaluEval** for larger training sets.
- **Neural detectors (Autoencoder):** **HaluEval** provides the volume needed for neural training. Combine with **FaithDial** for dialogue-grounded hallucination detection.
- **Fact verification:** **FEVER** is the standard benchmark for claim verification. Works well for detectors that assess factual accuracy against knowledge bases.

### Self/Non-Self Mapping

When adapting these datasets for one-class training:

| Dataset | Self (normal) | Non-self (anomalous) |
|---------|---------------|----------------------|
| TruthfulQA | Truthful answers | Untruthful/misleading answers |
| HaluEval | Supported/faithful outputs | Hallucinated outputs |
| FEVER | Supported claims | Refuted claims |
| FaithDial | Faithful responses | Hallucinated responses |
| SummEval | Consistent summaries | Inconsistent summaries |

### Data Format

Convert datasets to the format expected by `load_training_samples()`:

```json
{
  "samples": [
    {
      "id": "unique-id",
      "prompt": "The input prompt or context",
      "output": "The AI-generated output to evaluate",
      "domain": "hallucination",
      "metadata": {"source": "dataset-name"}
    }
  ]
}
```

With a corresponding manifest:

```json
{
  "unique-id": {
    "ground_truth_label": "normal",
    "ground_truth_type": null,
    "is_honeypot": false
  }
}
```

### Combining Datasets

For competitive mining, combine multiple datasets:

1. Start with TruthfulQA (broad factual coverage)
2. Add HaluEval (scale and diversity)
3. Add domain-specific data (FEVER for fact verification, FaithDial for dialogue)

### TruthfulQA Caveat

TruthfulQA was designed for evaluating LLM truthfulness, not anomaly detection training.
Research shows 79.6% accuracy using simple decision trees on structural features alone,
meaning models can exploit question patterns rather than learning semantic truthfulness.
This is less of a concern for anomaly detector training (where you fit on normal output
distributions) than for LLM evaluation benchmarks, but be aware that structural patterns
in the questions may influence TF-IDF feature extraction.

## Custom Training Data

Miners are encouraged to curate their own training data beyond public datasets.
The `--neuron.training_data_dir` and `--neuron.training_domain` CLI arguments
allow pointing the reference miner at any directory following the samples.json +
manifest.json format.

For custom detectors implementing `BaseDetector`, the `fit(samples)` method
receives the list of normal sample dicts -- you can preprocess and train
however your detector requires.
