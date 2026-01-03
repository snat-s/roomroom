#!/usr/bin/env python3
"""
Calibration script for Binoculars detector.

Loads GhostBuster_v3 dataset, computes Binoculars scores,
and finds optimal thresholds for accuracy and low-FPR modes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset as hf_load_dataset
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from binoculars import Binoculars

DATASET = "jionghong94/GhostBuster_v3"
SPLIT = "train"
CACHE_DIR = "./data/"


def load_data():
    """Load and prepare the calibration dataset."""
    print(f"Loading {DATASET} ({SPLIT} split)...")

    dataset = hf_load_dataset(DATASET, split=SPLIT, cache_dir=CACHE_DIR)
    df = dataset.to_pandas()

    df = df.rename(columns={"texts": "text", "labels": "label", "domains": "domain"})
    df["is_ai"] = df["label"] == "gpt"

    return df


def compute_scores(
    texts: list[str],
    observer: str,
    performer: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute Binoculars scores for all texts."""
    print(f"  Observer: {observer}")
    print(f"  Performer: {performer}")

    bino = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        use_bfloat16=True,
        max_token_observed=512,
        mode="low-fpr",
    )

    scores = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\nScoring {len(texts)} texts in {n_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_scores = bino.compute_score(batch)

        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]

        scores.extend(batch_scores)

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(texts):
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(scores)


def find_optimal_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.0001,
) -> dict:
    """
    Find optimal thresholds for accuracy and low-FPR modes.

    Args:
        scores: Binoculars scores (lower = more likely AI)
        labels: Binary labels (True = AI-generated)
        target_fpr: Target false positive rate for low-FPR mode

    Returns:
        Dict with thresholds and metrics
    """
    # Invert scores for sklearn (expects higher = positive class)
    inverted_scores = -scores

    # Find threshold for best F1 (accuracy mode)
    precision, recall, pr_thresholds = precision_recall_curve(labels, inverted_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    accuracy_threshold = -pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else scores.mean()

    # Find threshold for target FPR (low-fpr mode)
    fpr, tpr, roc_thresholds = roc_curve(labels, inverted_scores)

    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) > 0:
        best_low_fpr_idx = valid_idx[np.argmax(tpr[valid_idx])]
        low_fpr_threshold = -roc_thresholds[best_low_fpr_idx] if best_low_fpr_idx < len(roc_thresholds) else scores.mean()
    else:
        best_low_fpr_idx = np.argmin(fpr[fpr > 0]) if np.any(fpr > 0) else 0
        low_fpr_threshold = -roc_thresholds[best_low_fpr_idx] if best_low_fpr_idx < len(roc_thresholds) else scores.mean()

    def compute_metrics(threshold):
        preds = scores < threshold
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        tn = np.sum(~preds & ~labels)
        fn = np.sum(~preds & labels)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "fpr": fpr}

    return {
        "accuracy_threshold": float(accuracy_threshold),
        "accuracy_metrics": compute_metrics(accuracy_threshold),
        "low_fpr_threshold": float(low_fpr_threshold),
        "low_fpr_metrics": compute_metrics(low_fpr_threshold),
        "score_stats": {
            "ai_mean": float(scores[labels].mean()),
            "ai_std": float(scores[labels].std()),
            "human_mean": float(scores[~labels].mean()),
            "human_std": float(scores[~labels].std()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Binoculars thresholds")
    parser.add_argument(
        "--observer",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Observer model name or path",
    )
    parser.add_argument(
        "--performer",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Performer model name or path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for scoring",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.0001,
        help="Target false positive rate for low-FPR mode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (optional)",
    )
    args = parser.parse_args()

    df = load_data()

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42)
        print(f"Sampled {args.max_samples} samples for calibration")

    scores = compute_scores(
        texts=df["text"].tolist(),
        observer=args.observer,
        performer=args.performer,
        batch_size=args.batch_size,
    )

    labels = df["is_ai"].values
    results = find_optimal_thresholds(
        scores=scores,
        labels=labels,
        target_fpr=args.target_fpr,
    )

    results["model"] = {
        "observer": args.observer,
        "performer": args.performer,
    }
    results["samples"] = {
        "total": len(df),
        "ai": int(labels.sum()),
        "human": int((~labels).sum()),
    }

    print("\n=== Results ===")
    print(f"Model: {args.observer} / {args.performer}")
    print(f"Samples: {len(df)} total ({results['samples']['ai']} AI, {results['samples']['human']} human)")
    print(f"\nAccuracy threshold: {results['accuracy_threshold']:.4f}")
    print(f"  Precision: {results['accuracy_metrics']['precision']:.4f}")
    print(f"  Recall: {results['accuracy_metrics']['recall']:.4f}")
    print(f"  F1: {results['accuracy_metrics']['f1']:.4f}")
    print(f"\nLow-FPR threshold: {results['low_fpr_threshold']:.4f}")
    print(f"  FPR: {results['low_fpr_metrics']['fpr']:.6f}")
    print(f"  Recall: {results['low_fpr_metrics']['recall']:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
