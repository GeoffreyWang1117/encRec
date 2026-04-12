"""
Comprehensive Recommendation System Metrics.

All standard metrics for evaluating recommendation quality, diversity,
coverage, novelty, and fairness. Designed for KDD-level experiments.

References:
- RecBole: https://recbole.io/
- Recommender Systems Handbook (Ricci et al.)
- Beyond Accuracy (Herlocker et al., 2004)
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class RecommendationMetrics:
    """Container for all recommendation metrics."""
    # Accuracy metrics
    hit: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0  # Mean Average Precision

    # Ranking metrics
    auc: float = 0.0
    gauc: float = 0.0  # Group AUC

    # Beyond-accuracy metrics
    coverage: float = 0.0
    diversity: float = 0.0  # Intra-List Diversity
    novelty: float = 0.0
    serendipity: float = 0.0

    # Fairness metrics
    popularity_bias: float = 0.0
    long_tail_coverage: float = 0.0

    # Efficiency metrics
    latency_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hit': self.hit,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'ndcg': self.ndcg,
            'mrr': self.mrr,
            'map': self.map_score,
            'auc': self.auc,
            'gauc': self.gauc,
            'coverage': self.coverage,
            'diversity': self.diversity,
            'novelty': self.novelty,
            'serendipity': self.serendipity,
            'popularity_bias': self.popularity_bias,
            'long_tail_coverage': self.long_tail_coverage,
            'latency_ms': self.latency_ms,
            'tokens_used': self.tokens_used,
        }


class MetricsCalculator:
    """
    Calculate all recommendation metrics.

    Usage:
        calc = MetricsCalculator(item_popularity, item_embeddings)
        metrics = calc.calculate_all(recommended, ground_truth, k=10)
    """

    def __init__(
        self,
        item_popularity: Optional[Dict[str, int]] = None,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        item_categories: Optional[Dict[str, str]] = None,
        all_items: Optional[Set[str]] = None,
    ):
        """
        Initialize calculator with item metadata.

        Args:
            item_popularity: Dict mapping item_id -> interaction count
            item_embeddings: Dict mapping item_id -> embedding vector
            item_categories: Dict mapping item_id -> category
            all_items: Set of all item IDs in catalog
        """
        self.item_popularity = item_popularity or {}
        self.item_embeddings = item_embeddings or {}
        self.item_categories = item_categories or {}
        self.all_items = all_items or set()

        # Precompute popularity statistics
        if self.item_popularity:
            pop_values = list(self.item_popularity.values())
            self.total_interactions = sum(pop_values)
            self.max_popularity = max(pop_values) if pop_values else 1

            # Define long-tail threshold (bottom 80% of items by popularity)
            sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1])
            cutoff = int(len(sorted_items) * 0.8)
            self.long_tail_items = set(item for item, _ in sorted_items[:cutoff])
        else:
            self.total_interactions = 1
            self.max_popularity = 1
            self.long_tail_items = set()

    # ==================== Accuracy Metrics ====================

    def hit_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Hit@K: Whether any relevant item appears in top-K.

        Returns 1 if at least one ground truth item is in top-K recommendations.
        """
        if not ground_truth:
            return 0.0
        return 1.0 if any(item in ground_truth for item in recommended[:k]) else 0.0

    def precision_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Precision@K = |Recommended ∩ Relevant| / K

        Fraction of recommended items that are relevant.
        """
        if k == 0:
            return 0.0
        hits = sum(1 for item in recommended[:k] if item in ground_truth)
        return hits / k

    def recall_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Recall@K = |Recommended ∩ Relevant| / |Relevant|

        Fraction of relevant items that are recommended.
        """
        if not ground_truth:
            return 0.0
        hits = sum(1 for item in recommended[:k] if item in ground_truth)
        return hits / len(ground_truth)

    def f1_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        F1@K = 2 * Precision * Recall / (Precision + Recall)

        Harmonic mean of precision and recall.
        """
        precision = self.precision_at_k(recommended, ground_truth, k)
        recall = self.recall_at_k(recommended, ground_truth, k)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def ndcg_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.

        Measures ranking quality with position-based discounting.
        """
        if not ground_truth:
            return 0.0

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))

        return dcg / idcg if idcg > 0 else 0.0

    def mrr(self, recommended: List[str], ground_truth: List[str], k: int = None) -> float:
        """
        MRR: Mean Reciprocal Rank.

        1 / (rank of first relevant item).
        """
        if not ground_truth:
            return 0.0

        search_list = recommended[:k] if k else recommended
        for i, item in enumerate(search_list):
            if item in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    def average_precision(self, recommended: List[str], ground_truth: List[str], k: int = None) -> float:
        """
        AP: Average Precision.

        Average of precision at each relevant position.
        """
        if not ground_truth:
            return 0.0

        search_list = recommended[:k] if k else recommended
        precisions = []
        num_hits = 0

        for i, item in enumerate(search_list):
            if item in ground_truth:
                num_hits += 1
                precisions.append(num_hits / (i + 1))

        return sum(precisions) / len(ground_truth) if precisions else 0.0

    # ==================== Ranking Metrics ====================

    def auc(
        self,
        scores: List[Tuple[str, float]],
        ground_truth: List[str]
    ) -> float:
        """
        AUC: Area Under the ROC Curve.

        P(score(positive) > score(negative))

        Args:
            scores: List of (item_id, score) tuples
            ground_truth: List of positive item IDs
        """
        if not scores or not ground_truth:
            return 0.5

        ground_truth_set = set(ground_truth)
        positives = [(item, score) for item, score in scores if item in ground_truth_set]
        negatives = [(item, score) for item, score in scores if item not in ground_truth_set]

        if not positives or not negatives:
            return 0.5

        # Count correct orderings
        correct = 0
        total = 0

        for _, pos_score in positives:
            for _, neg_score in negatives:
                total += 1
                if pos_score > neg_score:
                    correct += 1
                elif pos_score == neg_score:
                    correct += 0.5

        return correct / total if total > 0 else 0.5

    # ==================== Beyond-Accuracy Metrics ====================

    def catalog_coverage(self, all_recommendations: List[List[str]]) -> float:
        """
        Catalog Coverage: Fraction of items ever recommended.

        Measures how much of the catalog is exposed to users.
        """
        if not self.all_items:
            return 0.0

        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / len(self.all_items)

    def intra_list_diversity(self, recommended: List[str], k: int = None) -> float:
        """
        ILD: Intra-List Diversity.

        Average pairwise distance between recommended items.
        Higher is more diverse.
        """
        if not self.item_embeddings:
            # Fallback: use category-based diversity
            return self._category_diversity(recommended, k)

        items = recommended[:k] if k else recommended
        if len(items) < 2:
            return 0.0

        distances = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i] in self.item_embeddings and items[j] in self.item_embeddings:
                    emb_i = self.item_embeddings[items[i]]
                    emb_j = self.item_embeddings[items[j]]
                    # Cosine distance = 1 - cosine similarity
                    sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                    distances.append(1 - sim)

        return np.mean(distances) if distances else 0.0

    def _category_diversity(self, recommended: List[str], k: int = None) -> float:
        """Category-based diversity when embeddings unavailable."""
        if not self.item_categories:
            return 0.0

        items = recommended[:k] if k else recommended
        categories = set()
        for item in items:
            if item in self.item_categories:
                categories.add(self.item_categories[item])

        return len(categories) / len(items) if items else 0.0

    def novelty(self, recommended: List[str], k: int = None) -> float:
        """
        Novelty: How unexpected/novel are the recommendations.

        Based on item popularity: recommending unpopular items is more novel.
        Novelty = mean(-log2(popularity / total))
        """
        if not self.item_popularity:
            return 0.0

        items = recommended[:k] if k else recommended
        novelties = []

        for item in items:
            pop = self.item_popularity.get(item, 1)
            # Normalize and compute self-information
            prob = pop / self.total_interactions
            novelties.append(-np.log2(prob + 1e-10))

        return np.mean(novelties) if novelties else 0.0

    def serendipity(
        self,
        recommended: List[str],
        ground_truth: List[str],
        expected: List[str],  # Items user would expect (e.g., popular items)
        k: int = None
    ) -> float:
        """
        Serendipity: Unexpected AND relevant recommendations.

        Serendipity = |Recommended ∩ Relevant - Expected| / K
        """
        items = recommended[:k] if k else recommended
        if not items:
            return 0.0

        ground_truth_set = set(ground_truth)
        expected_set = set(expected)

        serendipitous = sum(
            1 for item in items
            if item in ground_truth_set and item not in expected_set
        )

        return serendipitous / len(items)

    # ==================== Fairness Metrics ====================

    def popularity_bias(self, recommended: List[str], k: int = None) -> float:
        """
        Popularity Bias: How much recommendations favor popular items.

        Average popularity of recommended items / max popularity.
        Lower is less biased toward popular items.
        """
        if not self.item_popularity:
            return 0.0

        items = recommended[:k] if k else recommended
        pops = [self.item_popularity.get(item, 0) for item in items]

        return np.mean(pops) / self.max_popularity if pops else 0.0

    def long_tail_coverage(self, recommended: List[str], k: int = None) -> float:
        """
        Long-Tail Coverage: Fraction of recommendations from long-tail items.

        Long-tail = bottom 80% items by popularity.
        """
        if not self.long_tail_items:
            return 0.0

        items = recommended[:k] if k else recommended
        if not items:
            return 0.0

        long_tail_count = sum(1 for item in items if item in self.long_tail_items)
        return long_tail_count / len(items)

    # ==================== Aggregation ====================

    def calculate_all(
        self,
        recommended: List[str],
        ground_truth: List[str],
        k: int = 10,
        scores: Optional[List[Tuple[str, float]]] = None,
        expected: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> RecommendationMetrics:
        """
        Calculate all metrics for a single recommendation.

        Args:
            recommended: List of recommended item IDs
            ground_truth: List of relevant item IDs (ground truth)
            k: Cutoff for @K metrics
            scores: Optional (item_id, score) pairs for AUC
            expected: Optional expected items for serendipity
            latency_ms: Inference latency
            tokens_used: Number of tokens used

        Returns:
            RecommendationMetrics with all metrics populated
        """
        metrics = RecommendationMetrics()

        # Accuracy metrics
        metrics.hit = self.hit_at_k(recommended, ground_truth, k)
        metrics.precision = self.precision_at_k(recommended, ground_truth, k)
        metrics.recall = self.recall_at_k(recommended, ground_truth, k)
        metrics.f1 = self.f1_at_k(recommended, ground_truth, k)
        metrics.ndcg = self.ndcg_at_k(recommended, ground_truth, k)
        metrics.mrr = self.mrr(recommended, ground_truth, k)
        metrics.map_score = self.average_precision(recommended, ground_truth, k)

        # Ranking metrics
        if scores:
            metrics.auc = self.auc(scores, ground_truth)

        # Beyond-accuracy metrics
        metrics.diversity = self.intra_list_diversity(recommended, k)
        metrics.novelty = self.novelty(recommended, k)

        if expected:
            metrics.serendipity = self.serendipity(recommended, ground_truth, expected, k)

        # Fairness metrics
        metrics.popularity_bias = self.popularity_bias(recommended, k)
        metrics.long_tail_coverage = self.long_tail_coverage(recommended, k)

        # Efficiency metrics
        metrics.latency_ms = latency_ms
        metrics.tokens_used = tokens_used

        return metrics

    def aggregate_metrics(
        self,
        metrics_list: List[RecommendationMetrics],
        all_recommendations: Optional[List[List[str]]] = None,
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple samples.

        Returns mean and std for each metric.
        """
        if not metrics_list:
            return {}

        result = {}

        # Get all metric names
        sample = metrics_list[0].to_dict()

        for metric_name in sample.keys():
            values = [m.to_dict()[metric_name] for m in metrics_list]
            result[f'{metric_name}_mean'] = np.mean(values)
            result[f'{metric_name}_std'] = np.std(values)

        # Add catalog coverage if we have all recommendations
        if all_recommendations and self.all_items:
            result['catalog_coverage'] = self.catalog_coverage(all_recommendations)

        return result


class StatisticalTests:
    """Statistical significance tests for comparing methods."""

    @staticmethod
    def paired_ttest(
        method1_results: List[float],
        method2_results: List[float],
    ) -> Dict:
        """
        Paired t-test for comparing two methods.

        Returns:
            t_statistic, p_value, significant (p < 0.05), effect_size (Cohen's d)
        """
        from scipy import stats

        if len(method1_results) != len(method2_results):
            raise ValueError("Result lists must have same length")

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(method1_results, method2_results)

        # Effect size (Cohen's d)
        diff = np.array(method1_results) - np.array(method2_results)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else
                          'medium' if abs(cohens_d) > 0.5 else 'small',
        }

    @staticmethod
    def wilcoxon_test(
        method1_results: List[float],
        method2_results: List[float],
    ) -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric alternative to t-test).
        """
        from scipy import stats

        statistic, p_value = stats.wilcoxon(method1_results, method2_results)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }

    @staticmethod
    def format_significance(p_value: float) -> str:
        """Format p-value with significance stars."""
        if p_value < 0.001:
            return f"{p_value:.4f}***"
        elif p_value < 0.01:
            return f"{p_value:.4f}**"
        elif p_value < 0.05:
            return f"{p_value:.4f}*"
        else:
            return f"{p_value:.4f}"


def create_metrics_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['hit', 'precision', 'recall', 'ndcg', 'mrr'],
    k: int = 10,
) -> str:
    """
    Create a formatted table of results.

    Args:
        results: Dict mapping method_name -> {metric_name: value}
        metrics: List of metric names to include
        k: The K value used (for display)

    Returns:
        Formatted table string
    """
    # Header
    header = f"{'Method':<20}"
    for m in metrics:
        header += f" {m}@{k:<8}"

    lines = [header, "-" * len(header)]

    # Rows
    for method, method_results in results.items():
        row = f"{method:<20}"
        for m in metrics:
            mean_key = f"{m}_mean"
            std_key = f"{m}_std"

            if mean_key in method_results:
                mean_val = method_results[mean_key]
                std_val = method_results.get(std_key, 0)
                row += f" {mean_val:.4f}±{std_val:.4f}"
            else:
                row += f" {'N/A':<12}"

        lines.append(row)

    return "\n".join(lines)
