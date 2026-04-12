"""
Evaluation metrics for CTR prediction and recommendation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from typing import Dict, List, Optional, Tuple
import torch


class AUCMeter:
    """Running AUC computation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, preds: np.ndarray, labels: np.ndarray):
        self.predictions.extend(preds.flatten().tolist())
        self.labels.extend(labels.flatten().tolist())

    def compute(self) -> float:
        if len(set(self.labels)) < 2:
            return 0.5
        return roc_auc_score(self.labels, self.predictions)


class LogLossMeter:
    """Running LogLoss computation."""

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, preds: np.ndarray, labels: np.ndarray):
        preds = np.clip(preds, self.eps, 1 - self.eps)
        loss = -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
        self.total_loss += loss * len(labels)
        self.count += len(labels)

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_loss / self.count


class BucketedMetrics:
    """
    Compute metrics bucketed by frequency (Head/Mid/Tail).
    Critical for evaluating Trie-based routing effectiveness.
    """

    def __init__(self, bucket_boundaries: List[float] = None):
        # Default: Head (top 20%), Mid (20-80%), Tail (bottom 20%)
        self.bucket_boundaries = bucket_boundaries or [0.2, 0.8]
        self.reset()

    def reset(self):
        self.buckets = {
            'head': {'preds': [], 'labels': []},
            'mid': {'preds': [], 'labels': []},
            'tail': {'preds': [], 'labels': []},
        }

    def update(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        frequency_scores: np.ndarray
    ):
        """
        Args:
            preds: Prediction scores
            labels: Ground truth labels
            frequency_scores: Normalized frequency score (0-1) for each sample,
                            based on the tokens it contains
        """
        head_mask = frequency_scores >= self.bucket_boundaries[1]
        tail_mask = frequency_scores <= self.bucket_boundaries[0]
        mid_mask = ~head_mask & ~tail_mask

        for name, mask in [('head', head_mask), ('mid', mid_mask), ('tail', tail_mask)]:
            if mask.sum() > 0:
                self.buckets[name]['preds'].extend(preds[mask].tolist())
                self.buckets[name]['labels'].extend(labels[mask].tolist())

    def compute(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for bucket_name, data in self.buckets.items():
            if len(data['labels']) > 0 and len(set(data['labels'])) >= 2:
                results[bucket_name] = {
                    'auc': roc_auc_score(data['labels'], data['preds']),
                    'logloss': log_loss(data['labels'], data['preds']),
                    'count': len(data['labels']),
                }
            else:
                results[bucket_name] = {
                    'auc': 0.5,
                    'logloss': 0.0,
                    'count': len(data['labels']),
                }
        return results


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    frequency_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for CTR prediction.

    Returns:
        Dict containing AUC, LogLoss, and optionally bucketed metrics
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()

    # Ensure labels are binary integers
    labels = labels.astype(np.int32)

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

    # Compute logloss manually to avoid sklearn format issues
    logloss = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    metrics = {
        'auc': roc_auc_score(labels, predictions) if len(set(labels)) >= 2 else 0.5,
        'logloss': logloss,
        'accuracy': accuracy_score(labels, (predictions > 0.5).astype(int)),
    }

    if frequency_scores is not None:
        bucketed = BucketedMetrics()
        bucketed.update(predictions, labels, frequency_scores)
        bucket_results = bucketed.compute()
        for bucket_name, bucket_metrics in bucket_results.items():
            for metric_name, value in bucket_metrics.items():
                metrics[f'{bucket_name}_{metric_name}'] = value

    return metrics


class ExpertUtilizationMeter:
    """Track MoE expert utilization for load balancing analysis."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()

    def reset(self):
        self.expert_counts = np.zeros(self.num_experts)
        self.total_samples = 0

    def update(self, expert_indices: np.ndarray):
        """
        Args:
            expert_indices: Shape (batch_size,) or (batch_size, top_k)
        """
        expert_indices = np.asarray(expert_indices).flatten()
        for idx in expert_indices:
            self.expert_counts[idx] += 1
        self.total_samples += len(expert_indices)

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {'utilization_entropy': 0.0, 'utilization_std': 0.0}

        probs = self.expert_counts / self.total_samples
        probs = probs + 1e-10  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(self.num_experts)

        return {
            'utilization_entropy': entropy / max_entropy,  # Normalized to [0, 1]
            'utilization_std': np.std(self.expert_counts / self.total_samples),
            'expert_distribution': probs.tolist(),
        }
