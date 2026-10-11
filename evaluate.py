import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


def apply_guarded_temporal_decision_rule(
    raw_scores: np.ndarray,
    threshold: float = 0.309,
    bypass_threshold: float = 0.4625,
    window_size: int = 3,
    min_confirmations: int = 2,
) -> np.ndarray:
    """
    Applies the guarded temporal decision rule to streaming window diagnosis scores (Paper Section III-D).

    Rules:
      1. If raw_score >= bypass_threshold (theta_h), alarm triggers immediately (zero confirmation delay).
      2. If raw_score >= threshold (theta), alarm requires at least min_confirmations (M=2)
         crossings within the last window_size (C=3) consecutive windows to suppress transient spikes.
    """
    n = len(raw_scores)
    decisions = np.zeros(n, dtype=np.int64)
    buffer: List[float] = []

    for t in range(n):
        score = float(raw_scores[t])
        buffer.append(score)
        if len(buffer) > window_size:
            buffer.pop(0)

        # High-confidence bypass
        if score >= bypass_threshold:
            decisions[t] = 1
        elif score >= threshold:
            # Check confirmation buffer
            active_count = sum(1 for s in buffer if s >= threshold)
            if active_count >= min_confirmations:
                decisions[t] = 1
            else:
                decisions[t] = 0
        else:
            decisions[t] = 0

    return decisions


def compute_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    post_processed_decisions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes comprehensive diagnostic effectiveness metrics."""
    binary_preds = (scores >= threshold).astype(np.int64)

    metrics = {
        "f1_raw": float(f1_score(targets, binary_preds, zero_division=0)),
        "precision_raw": float(precision_score(targets, binary_preds, zero_division=0)),
        "recall_raw": float(recall_score(targets, binary_preds, zero_division=0)),
        "accuracy_raw": float(accuracy_score(targets, binary_preds)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(targets, scores))
        metrics["pr_auc"] = float(average_precision_score(targets, scores))
    except Exception:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    if post_processed_decisions is not None:
        metrics["f1_guarded"] = float(f1_score(targets, post_processed_decisions, zero_division=0))
        metrics["precision_guarded"] = float(precision_score(targets, post_processed_decisions, zero_division=0))
        metrics["recall_guarded"] = float(recall_score(targets, post_processed_decisions, zero_division=0))
        metrics["accuracy_guarded"] = float(accuracy_score(targets, post_processed_decisions))

    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    """Pretty prints the diagnostic evaluation results."""
    print("=" * 65)
    print(f"{'Metric':<35} | {'Value':<15}")
    print("-" * 65)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<35} | {v:<15.4f}")
        else:
            print(f"{k:<35} | {str(v):<15}")
    print("=" * 65)
