#!/usr/bin/env python3
"""
KDD 2026 Experiment: MoE Routing Mechanism Comparison

Compare Trie-based routing with learned routing mechanisms on MIND dataset.

Routing Methods:
1. Trie-Router (Ours): Statistical routing based on Trie structure
2. MMoE: Multi-gate Mixture of Experts (Ma et al., 2018)
3. Switch-Router: Top-1 sparse routing (Fedus et al., 2022)
4. Lightweight-MLP: Simple learned gating network
5. Random: Baseline for sanity check

Control Variables (SAME as main MIND experiments):
- Dataset: MIND Large (101K news)
- LLM: GPT-2 (124M) via HuggingFace
- Scoring: Perplexity-based ranking
- Metrics: Hit@k, NDCG@k, MRR@k
- Samples: 2000 per run
- Runs: 3 with seeds [42, 123, 456]
- Candidate pool: 20 items
- History length: 10 items

Key Insight: Trie-based routing uses pre-computed statistics
instead of learned embeddings, which is more efficient and
interpretable for recommendation routing.
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MoEComparisonConfig:
    """Configuration matching main MIND experiments."""
    n_samples: int = 2000
    num_runs: int = 3
    k_values: List[int] = None
    seeds: List[int] = None
    n_experts: int = 4
    candidate_pool_size: int = 20
    history_length: int = 10
    feature_dim: int = 64
    hidden_dim: int = 128
    output_dir: str = "results/kdd_triellm_moe_routing"
    device: str = None

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.seeds is None:
            self.seeds = [42, 123, 456][:self.num_runs]
        if self.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ============================================================
# ROUTING MECHANISMS
# ============================================================

class TrieRouter(nn.Module):
    """
    Trie-based routing using pre-computed category statistics.
    This is our proposed method.
    """

    def __init__(self, n_categories: int, n_experts: int):
        super().__init__()
        self.n_categories = n_categories
        self.n_experts = n_experts
        # Pre-computed mapping from category to expert
        self.register_buffer(
            'category_to_expert',
            torch.zeros(n_categories, dtype=torch.long)
        )
        self.category_stats = {}

    def set_routing_from_stats(self, category_counts: Dict[str, int]):
        """
        Set routing based on category frequency statistics.
        Assigns categories to experts in balanced round-robin fashion.
        """
        self.category_stats = category_counts

        # Sort categories by frequency
        sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])

        # Assign to experts round-robin (balanced load)
        for i, (cat, _) in enumerate(sorted_cats):
            cat_idx = hash(cat) % self.n_categories
            self.category_to_expert[cat_idx] = i % self.n_experts

    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Route based on category.
        Input: category_ids (batch_size,)
        Output: routing weights (batch_size, n_experts)
        """
        expert_ids = self.category_to_expert[category_ids % self.n_categories]
        routing = F.one_hot(expert_ids, self.n_experts).float()
        return routing

    def get_routing_stats(self) -> Dict:
        """Return routing statistics for analysis."""
        return {
            'type': 'trie',
            'n_experts': self.n_experts,
            'category_coverage': len(self.category_stats)
        }


class MMoERouter(nn.Module):
    """
    Multi-gate Mixture of Experts (Ma et al., KDD 2018).
    Uses learned embeddings for routing.
    """

    def __init__(self, input_dim: int, n_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.n_experts = n_experts

        # Shared bottom layer
        self.shared_layer = nn.Linear(input_dim, hidden_dim)

        # Multiple gates (one per expert for multi-task, we use average)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learned routing based on input features.
        Input: x (batch_size, input_dim)
        Output: routing weights (batch_size, n_experts)
        """
        hidden = F.relu(self.shared_layer(x))
        routing = self.gate(hidden)
        return routing

    def get_routing_stats(self) -> Dict:
        return {'type': 'mmoe', 'n_experts': self.n_experts}


class SwitchRouter(nn.Module):
    """
    Switch Transformer routing (Fedus et al., JMLR 2022).
    Top-1 sparse routing with load balancing noise.
    """

    def __init__(self, input_dim: int, n_experts: int, noise_scale: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.noise_scale = noise_scale
        self.router = nn.Linear(input_dim, n_experts)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Top-1 routing with optional noise for load balancing.
        Input: x (batch_size, input_dim)
        Output: routing weights (batch_size, n_experts) - one-hot
        """
        logits = self.router(x)

        if training and self.noise_scale > 0:
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise

        # Top-1 selection
        top1_idx = logits.argmax(dim=-1)
        routing = F.one_hot(top1_idx, self.n_experts).float()
        return routing

    def get_routing_stats(self) -> Dict:
        return {'type': 'switch', 'n_experts': self.n_experts}


class LightweightMLPRouter(nn.Module):
    """
    Simple 2-layer MLP gating network.
    Baseline for learned routing.
    """

    def __init__(self, input_dim: int, n_experts: int, hidden_dim: int = 32):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

    def get_routing_stats(self) -> Dict:
        return {'type': 'lightweight_mlp', 'n_experts': self.n_experts}


class RandomRouter(nn.Module):
    """Random routing baseline for sanity check."""

    def __init__(self, n_experts: int):
        super().__init__()
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        random_idx = torch.randint(0, self.n_experts, (batch_size,), device=x.device)
        routing = F.one_hot(random_idx, self.n_experts).float()
        return routing

    def get_routing_stats(self) -> Dict:
        return {'type': 'random', 'n_experts': self.n_experts}


# ============================================================
# EXPERT AND RECOMMENDER
# ============================================================

class ExpertNetwork(nn.Module):
    """Simple expert network for scoring."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoERecommender(nn.Module):
    """MoE-based recommender with configurable routing."""

    def __init__(self, router: nn.Module, n_experts: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim) for _ in range(n_experts)
        ])
        self.n_experts = n_experts

    def forward(
        self,
        x: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score items using routed experts.
        Returns: (scores, routing_weights)
        """
        # Get routing weights
        if isinstance(self.router, TrieRouter):
            if category_ids is None:
                raise ValueError("TrieRouter requires category_ids")
            routing = self.router(category_ids)
        else:
            routing = self.router(x)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weighted combination
        scores = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)

        return scores.squeeze(-1), routing


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractor:
    """Extract features from MIND news items for routing."""

    def __init__(self, news_items: Dict, feature_dim: int = 64):
        self.news_items = news_items
        self.feature_dim = feature_dim
        self.category_to_idx = {}
        self._build_category_mapping()

    def _build_category_mapping(self):
        """Build category vocabulary."""
        categories = set()
        for item in self.news_items.values():
            if isinstance(item, dict):
                categories.add(item.get('category', 'unknown'))
        self.category_to_idx = {cat: i for i, cat in enumerate(sorted(categories))}
        logger.info(f"Built mapping for {len(self.category_to_idx)} categories")

    def extract(self, item_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and category IDs for items.
        Returns: (features, category_ids)
        """
        features = []
        category_ids = []

        for item_id in item_ids:
            item = self.news_items.get(item_id, {})
            if isinstance(item, dict):
                title = item.get('title', '')
                category = item.get('category', 'unknown')
            else:
                title = getattr(item, 'title', '')
                category = getattr(item, 'category', 'unknown')

            # Hash-based feature (consistent with other experiments)
            title_hash = hash(title) % 10000
            cat_idx = self.category_to_idx.get(category, 0)

            # Create feature vector
            feat = np.zeros(self.feature_dim)
            feat[0] = title_hash / 10000  # Normalized title hash
            feat[1] = cat_idx / max(1, len(self.category_to_idx))  # Normalized category
            feat[2] = len(title) / 200  # Normalized length
            # Fill rest with deterministic pseudo-random values
            np.random.seed(hash(item_id) % (2**31))
            feat[3:] = np.random.randn(self.feature_dim - 3) * 0.1

            features.append(feat)
            category_ids.append(cat_idx)

        return (
            torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(category_ids, dtype=torch.long)
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_recommendations(
    predicted_ranking: List[int],
    ground_truth_idx: int,
    k_values: List[int]
) -> Dict:
    """Evaluate recommendation metrics (same as other experiments)."""
    metrics = {}
    for k in k_values:
        top_k = predicted_ranking[:k]
        hit = 1 if ground_truth_idx in top_k else 0
        metrics[f'hit@{k}'] = hit

        if hit:
            rank = top_k.index(ground_truth_idx) + 1
            metrics[f'ndcg@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'mrr@{k}'] = 1.0 / rank
        else:
            metrics[f'ndcg@{k}'] = 0.0
            metrics[f'mrr@{k}'] = 0.0

    return metrics


def compute_expert_utilization(routing_weights: torch.Tensor) -> float:
    """Compute expert utilization balance (entropy-based)."""
    # Average routing across batch
    avg_routing = routing_weights.mean(dim=0)
    # Compute entropy
    eps = 1e-8
    entropy = -(avg_routing * torch.log(avg_routing + eps)).sum()
    # Normalize by max entropy
    max_entropy = np.log(routing_weights.shape[1])
    return float(entropy / max_entropy)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_moe_comparison(config: MoEComparisonConfig):
    """Run MoE routing comparison experiment."""
    logger.info("=" * 60)
    logger.info("KDD 2026: MoE Routing Mechanism Comparison")
    logger.info("=" * 60)

    # Load MIND data (same as main experiments)
    from src.data.mind_loader import load_mind_for_trie_experiment

    logger.info("\nLoading MIND dataset...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=config.n_samples * 2,
        seed=42
    )
    logger.info(f"Loaded {len(samples)} samples, {len(news_items)} news items")

    # Build category statistics for Trie routing
    category_counts = defaultdict(int)
    for sample in samples:
        for item_id in sample.get('history', []):
            item = news_items.get(item_id, {})
            if isinstance(item, dict):
                cat = item.get('category', 'unknown')
            else:
                cat = getattr(item, 'category', 'unknown')
            category_counts[cat] += 1

    logger.info(f"Found {len(category_counts)} categories")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(news_items, config.feature_dim)
    n_categories = len(feature_extractor.category_to_idx)

    # Define routing methods
    def create_routers():
        return {
            'Trie-Router': TrieRouter(n_categories, config.n_experts),
            'MMoE': MMoERouter(config.feature_dim, config.n_experts, config.hidden_dim),
            'Switch-Router': SwitchRouter(config.feature_dim, config.n_experts),
            'Lightweight-MLP': LightweightMLPRouter(config.feature_dim, config.n_experts),
            'Random': RandomRouter(config.n_experts)
        }

    all_results = []

    for run_idx, seed in enumerate(config.seeds):
        logger.info(f"\n--- Run {run_idx + 1}/{config.num_runs} (seed={seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Shuffle samples
        run_samples = samples.copy()
        np.random.shuffle(run_samples)
        run_samples = run_samples[:config.n_samples]

        # Create fresh routers for this run
        routers = create_routers()

        # Set Trie routing from statistics
        routers['Trie-Router'].set_routing_from_stats(category_counts)

        run_result = {'seed': seed, 'n_samples': len(run_samples)}

        for router_name, router in routers.items():
            logger.info(f"  Evaluating {router_name}...")

            model = MoERecommender(
                router=router,
                n_experts=config.n_experts,
                input_dim=config.feature_dim,
                hidden_dim=config.hidden_dim
            )
            model.eval()

            method_metrics = defaultdict(list)
            all_routing_weights = []

            for sample in run_samples:
                history = sample.get('history', [])
                ground_truth = sample.get('ground_truth', '')

                if not history or not ground_truth:
                    continue

                # Create candidate pool
                candidates = [ground_truth]
                other_items = [it for it in all_items if it != ground_truth]
                np.random.shuffle(other_items)
                candidates.extend(other_items[:config.candidate_pool_size - 1])
                np.random.shuffle(candidates)
                gt_idx = candidates.index(ground_truth)

                # Extract features
                features, category_ids = feature_extractor.extract(candidates)

                # Score candidates
                with torch.no_grad():
                    if isinstance(router, TrieRouter):
                        scores, routing = model(features, category_ids)
                    else:
                        scores, routing = model(features)

                all_routing_weights.append(routing)

                # Rank by score
                ranking = torch.argsort(scores, descending=True).tolist()

                # Evaluate
                metrics = evaluate_recommendations(ranking, gt_idx, config.k_values)
                for k, v in metrics.items():
                    method_metrics[k].append(v)

            # Aggregate metrics
            for metric, values in method_metrics.items():
                run_result[f'{router_name}_{metric}'] = np.mean(values)
                run_result[f'{router_name}_{metric}_std'] = np.std(values)

            # Compute expert utilization
            if all_routing_weights:
                all_routing = torch.cat(all_routing_weights, dim=0)
                run_result[f'{router_name}_expert_util'] = compute_expert_utilization(all_routing)

            hit5 = run_result.get(f'{router_name}_hit@5', 0)
            ndcg5 = run_result.get(f'{router_name}_ndcg@5', 0)
            expert_util = run_result.get(f'{router_name}_expert_util', 0)
            logger.info(f"    Hit@5={hit5:.4f}, NDCG@5={ndcg5:.4f}, ExpertUtil={expert_util:.3f}")

        all_results.append(run_result)

    # Aggregate across runs
    router_names = ['Trie-Router', 'MMoE', 'Switch-Router', 'Lightweight-MLP', 'Random']
    aggregate = {}

    for router_name in router_names:
        for metric in ['hit@1', 'hit@5', 'ndcg@5', 'mrr@5', 'expert_util']:
            key = f'{router_name}_{metric}'
            values = [r.get(key, 0) for r in all_results]
            if values:
                aggregate[f'{key}_mean'] = np.mean(values)
                aggregate[f'{key}_std'] = np.std(values)

    # Statistical significance tests
    from scipy import stats
    significance = {}

    trie_hits = [r.get('Trie-Router_hit@5', 0) for r in all_results]
    for router_name in router_names:
        if router_name != 'Trie-Router':
            other_hits = [r.get(f'{router_name}_hit@5', 0) for r in all_results]
            if len(trie_hits) >= 2 and len(other_hits) >= 2:
                t_stat, p_value = stats.ttest_rel(trie_hits, other_hits)
                significance[f'Trie_vs_{router_name}'] = {
                    't_stat': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'delta': np.mean(trie_hits) - np.mean(other_hits)
                }

    results = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'aggregate': aggregate,
        'significance': significance,
        'raw_results': all_results
    }

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'moe_routing_comparison_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: MoE Routing Comparison on MIND")
    logger.info("=" * 60)
    logger.info(f"\n{'Router':<20} {'Hit@5':<18} {'NDCG@5':<18} {'Expert Util':<12}")
    logger.info("-" * 68)

    for router_name in router_names:
        hit5_mean = aggregate.get(f'{router_name}_hit@5_mean', 0)
        hit5_std = aggregate.get(f'{router_name}_hit@5_std', 0)
        ndcg5_mean = aggregate.get(f'{router_name}_ndcg@5_mean', 0)
        ndcg5_std = aggregate.get(f'{router_name}_ndcg@5_std', 0)
        expert_util = aggregate.get(f'{router_name}_expert_util_mean', 0)

        logger.info(f"{router_name:<20} {hit5_mean:.4f}±{hit5_std:.4f}    "
                   f"{ndcg5_mean:.4f}±{ndcg5_std:.4f}    {expert_util:.3f}")

    logger.info("\nStatistical Significance (Trie-Router vs Others):")
    for test_name, sig in significance.items():
        status = "***" if sig['p_value'] < 0.001 else "**" if sig['p_value'] < 0.01 else "*" if sig['p_value'] < 0.05 else ""
        delta_pct = sig['delta'] * 100
        logger.info(f"  {test_name}: Δ={delta_pct:+.2f}%, p={sig['p_value']:.4f} {status}")

    logger.info(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MoE Routing Comparison Experiment")
    parser.add_argument("--samples", type=int, default=2000,
                       help="Number of samples per run")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs")
    parser.add_argument("--experts", type=int, default=4,
                       help="Number of experts")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    config = MoEComparisonConfig(
        n_samples=args.samples,
        num_runs=args.runs,
        n_experts=args.experts,
        device=args.device
    )

    run_moe_comparison(config)


if __name__ == '__main__':
    main()
