"""
Statistical computations for Trie nodes.

These statistics form the basis for:
1. Trie structure construction (hierarchical organization)
2. MoE routing signals
3. LLM prompt generation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


@dataclass
class NodeStatistics:
    """
    Statistics maintained at each Trie node.

    These are the "semantic-free" signals that drive routing
    when actual token semantics are unavailable.
    """
    # Frequency statistics
    count: int = 0
    frequency: float = 0.0  # Normalized frequency (0-1)
    frequency_rank: int = 0  # Rank in frequency ordering

    # Label statistics (for CTR prediction)
    positive_count: int = 0
    negative_count: int = 0
    ctr: float = 0.0  # Click-through rate
    ctr_lift: float = 0.0  # CTR relative to global average

    # Information-theoretic measures
    entropy: float = 0.0  # Entropy of label distribution
    mutual_info: float = 0.0  # Mutual information with label
    fisher_info: float = 0.0  # Fisher information

    # Temporal statistics (for drift detection)
    recent_ctr: float = 0.0  # CTR in recent time window
    ctr_drift: float = 0.0  # Change in CTR over time
    stability_score: float = 1.0  # How stable is this token

    # Co-occurrence statistics
    cooccurrence_entropy: float = 0.0  # Diversity of co-occurring tokens
    avg_cooccurrence_strength: float = 0.0  # Average PMI with co-occurring tokens

    def to_vector(self) -> np.ndarray:
        """Convert statistics to a feature vector for routing."""
        return np.array([
            self.frequency,
            self.ctr,
            self.ctr_lift,
            self.entropy,
            self.mutual_info,
            self.stability_score,
            self.cooccurrence_entropy,
        ], dtype=np.float32)

    @property
    def frequency_bucket(self) -> str:
        """Categorize into Head/Mid/Tail based on frequency."""
        if self.frequency_rank <= 0.2:  # Top 20%
            return "head"
        elif self.frequency_rank <= 0.8:  # Middle 60%
            return "mid"
        else:  # Bottom 20%
            return "tail"

    @property
    def info_bucket(self) -> str:
        """Categorize by information content."""
        if self.ctr_lift > 1.5:
            return "high_info"
        elif self.ctr_lift < 0.5:
            return "low_info"
        else:
            return "neutral"


class TrieStatistics:
    """
    Compute and manage statistics for Trie construction.

    This is the core engine that extracts statistical structure
    from encrypted/hashed tokens.
    """

    def __init__(self, global_ctr: float = 0.5, smoothing: float = 1.0):
        """
        Args:
            global_ctr: Global click-through rate (for lift computation)
            smoothing: Laplace smoothing parameter
        """
        self.global_ctr = global_ctr
        self.smoothing = smoothing

        # Raw counts
        self.token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.token_positive: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cooccurrence: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

        # Computed statistics
        self.statistics: Dict[str, Dict[str, NodeStatistics]] = {}

        self.total_samples = 0
        self.total_positive = 0

    def update(
        self,
        field: str,
        token: str,
        label: int,
        cooccurring_tokens: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Update statistics with a single observation.

        Args:
            field: Feature field name (e.g., "C1", "user_id")
            token: Token value
            label: Binary label (0 or 1)
            cooccurring_tokens: List of (field, token) pairs that co-occur
        """
        self.token_counts[field][token] += 1
        if label == 1:
            self.token_positive[field][token] += 1

        self.total_samples += 1
        if label == 1:
            self.total_positive += 1

        # Update co-occurrence
        if cooccurring_tokens:
            for other_field, other_token in cooccurring_tokens:
                if other_field != field or other_token != token:
                    key = tuple(sorted([(field, token), (other_field, other_token)]))
                    self.cooccurrence[field][key] += 1

    def update_batch(
        self,
        field: str,
        tokens: List[str],
        labels: List[int],
    ):
        """Batch update for efficiency."""
        for token, label in zip(tokens, labels):
            self.update(field, token, label)

    def compute_statistics(self) -> Dict[str, Dict[str, NodeStatistics]]:
        """
        Compute all statistics after data collection.

        Returns:
            Nested dict: field -> token -> NodeStatistics
        """
        self.global_ctr = self.total_positive / max(self.total_samples, 1)

        for field, token_counts in self.token_counts.items():
            self.statistics[field] = {}

            # Sort by frequency for rank computation
            sorted_tokens = sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            total_field_count = sum(token_counts.values())

            for rank, (token, count) in enumerate(sorted_tokens):
                positive = self.token_positive[field].get(token, 0)
                negative = count - positive

                # Basic statistics
                freq = count / max(total_field_count, 1)
                freq_rank = rank / max(len(sorted_tokens) - 1, 1)

                # CTR and lift
                ctr = (positive + self.smoothing) / (count + 2 * self.smoothing)
                ctr_lift = ctr / max(self.global_ctr, 1e-8)

                # Entropy of label distribution at this token
                p = ctr
                if 0 < p < 1:
                    entropy = -p * math.log2(p) - (1-p) * math.log2(1-p)
                else:
                    entropy = 0.0

                # Mutual information approximation
                # MI(token, label) based on frequency and CTR deviation
                mutual_info = abs(ctr - self.global_ctr) * freq

                stats = NodeStatistics(
                    count=count,
                    frequency=freq,
                    frequency_rank=freq_rank,
                    positive_count=positive,
                    negative_count=negative,
                    ctr=ctr,
                    ctr_lift=ctr_lift,
                    entropy=entropy,
                    mutual_info=mutual_info,
                )

                self.statistics[field][token] = stats

        return self.statistics

    def get_node_statistics(self, field: str, token: str) -> Optional[NodeStatistics]:
        """Get statistics for a specific token."""
        return self.statistics.get(field, {}).get(token)

    def get_field_summary(self, field: str) -> Dict[str, float]:
        """Get summary statistics for a field."""
        if field not in self.statistics:
            return {}

        stats_list = list(self.statistics[field].values())
        if not stats_list:
            return {}

        freqs = [s.frequency for s in stats_list]
        ctrs = [s.ctr for s in stats_list]
        lifts = [s.ctr_lift for s in stats_list]

        return {
            'num_tokens': len(stats_list),
            'avg_frequency': np.mean(freqs),
            'std_frequency': np.std(freqs),
            'avg_ctr': np.mean(ctrs),
            'std_ctr': np.std(ctrs),
            'avg_lift': np.mean(lifts),
            'max_lift': max(lifts),
            'min_lift': min(lifts),
            'head_count': sum(1 for s in stats_list if s.frequency_bucket == 'head'),
            'tail_count': sum(1 for s in stats_list if s.frequency_bucket == 'tail'),
        }

    def export_for_llm_prompt(self, field: str, top_k: int = 10) -> str:
        """
        Export statistics in a format suitable for LLM prompts.

        Note: No actual token values are exposed - only statistical summaries.
        """
        if field not in self.statistics:
            return f"No statistics available for field {field}"

        summary = self.get_field_summary(field)
        stats_list = sorted(
            self.statistics[field].items(),
            key=lambda x: x[1].mutual_info,
            reverse=True
        )[:top_k]

        lines = [
            f"Field: {field}",
            f"Total unique tokens: {summary['num_tokens']}",
            f"Head/Mid/Tail distribution: {summary['head_count']}/{summary['num_tokens'] - summary['head_count'] - summary['tail_count']}/{summary['tail_count']}",
            f"Average CTR: {summary['avg_ctr']:.4f} (global: {self.global_ctr:.4f})",
            f"CTR lift range: [{summary['min_lift']:.2f}, {summary['max_lift']:.2f}]",
            "",
            f"Top {top_k} high-information tokens (by mutual information):",
        ]

        for i, (token_id, stats) in enumerate(stats_list):
            # Note: token_id is already encrypted/hashed - we only show statistics
            lines.append(
                f"  [{i+1}] freq_bucket={stats.frequency_bucket}, "
                f"ctr_lift={stats.ctr_lift:.2f}, "
                f"info_bucket={stats.info_bucket}, "
                f"count={stats.count}"
            )

        return "\n".join(lines)
