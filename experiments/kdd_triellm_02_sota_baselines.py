"""
Run SOTA Baselines for KDD 2026 Paper.

Implements parallel execution strategy:
- GPU 0 (11.7GB): NRMS + NAML (lightweight neural models)
- GPU 1 (19.6GB): PLM-NR + TALLRec (needs PLM/LLM)
- CPU: Prompt4NR (uses existing LLM or local fallback)

Usage:
    python experiments/run_sota_baselines.py --samples 5000 --runs 5
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationSample:
    """A single recommendation sample."""
    user_id: str
    history: List[str]
    ground_truth: str
    candidates: List[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for baseline experiments."""
    n_samples: int = 5000
    num_runs: int = 5
    k_values: List[int] = None  # Top-k for evaluation
    seeds: List[int] = None
    output_dir: str = "results/sota_baselines"

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024][:self.num_runs]


def load_mind_data(n_samples: int = 5000) -> Tuple[List[RecommendationSample], Dict, List[str]]:
    """Load MIND dataset for experiments."""
    from src.data.mind_loader import load_mind_dataset

    logger.info(f"Loading MIND dataset (n_samples={n_samples})...")

    try:
        samples, news_items = load_mind_dataset(n_samples=n_samples)
        all_items = list(news_items.keys())

        rec_samples = [
            RecommendationSample(
                user_id=s.get('user_id', f'user_{i}'),
                history=s['history'],
                ground_truth=s['ground_truth'],
            )
            for i, s in enumerate(samples)
        ]

        logger.info(f"Loaded {len(rec_samples)} samples, {len(news_items)} news items")
        return rec_samples, news_items, all_items

    except Exception as e:
        logger.warning(f"Failed to load MIND: {e}. Creating synthetic data.")
        return create_synthetic_data(n_samples)


def create_synthetic_data(n_samples: int) -> Tuple[List[RecommendationSample], Dict, List[str]]:
    """Create synthetic data for testing."""
    # Generate synthetic news items
    news_items = {}
    categories = ['politics', 'sports', 'tech', 'entertainment', 'business']

    for i in range(1000):
        item_id = f"N{i:05d}"
        cat = categories[i % len(categories)]
        news_items[item_id] = {
            'title': f"News article about {cat} topic {i}",
            'abstract': f"This is a detailed abstract about {cat} news item {i}",
            'category': cat,
            'subcategory': f"{cat}_sub{i % 3}",
        }

    all_items = list(news_items.keys())

    # Generate samples
    samples = []
    for i in range(n_samples):
        history_size = np.random.randint(3, 15)
        history = list(np.random.choice(all_items, size=history_size, replace=False))
        remaining = [it for it in all_items if it not in history]
        ground_truth = np.random.choice(remaining)

        samples.append(RecommendationSample(
            user_id=f"user_{i}",
            history=history,
            ground_truth=ground_truth,
        ))

    return samples, news_items, all_items


def evaluate_recommendations(
    recommendations: List[str],
    ground_truth: str,
    k_values: List[int],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    for k in k_values:
        top_k = recommendations[:k]

        # Hit@k
        hit = 1.0 if ground_truth in top_k else 0.0
        metrics[f'hit@{k}'] = hit

        # NDCG@k
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        metrics[f'ndcg@{k}'] = ndcg

        # MRR@k
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        metrics[f'mrr@{k}'] = mrr

    return metrics


def run_single_baseline(
    baseline_name: str,
    samples: List[RecommendationSample],
    news_items: Dict,
    all_items: List[str],
    config: ExperimentConfig,
    seed: int,
    device: str = None,
) -> Dict:
    """Run a single baseline experiment."""
    from src.baselines.news_recommendation import (
        NRMSRecommender, NAMLRecommender, PLMNRRecommender,
        TALLRecRecommender, Prompt4NRLocal
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize recommender
    recommender_classes = {
        'NRMS': NRMSRecommender,
        'NAML': NAMLRecommender,
        'PLM-NR': PLMNRRecommender,
        'TALLRec': TALLRecRecommender,
        'Prompt4NR': Prompt4NRLocal,
    }

    if baseline_name not in recommender_classes:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    logger.info(f"[{baseline_name}] Initializing on {device}...")
    start_time = time.time()

    recommender = recommender_classes[baseline_name](items=news_items, device=device)

    # Train
    train_time_start = time.time()
    recommender.fit(samples, all_items)
    train_time = time.time() - train_time_start

    # Evaluate
    all_metrics = {f'hit@{k}': [] for k in config.k_values}
    all_metrics.update({f'ndcg@{k}': [] for k in config.k_values})
    all_metrics.update({f'mrr@{k}': [] for k in config.k_values})

    eval_time_start = time.time()
    for sample in samples:
        # Create candidate pool (ground truth + negatives)
        neg_items = [it for it in all_items if it != sample.ground_truth and it not in sample.history]
        candidates = [sample.ground_truth] + list(np.random.choice(neg_items, size=min(19, len(neg_items)), replace=False))
        np.random.shuffle(candidates)

        # Get recommendations
        recommendations = recommender.recommend(sample.history, candidates, k=max(config.k_values))

        # Compute metrics
        metrics = evaluate_recommendations(recommendations, sample.ground_truth, config.k_values)
        for key, value in metrics.items():
            all_metrics[key].append(value)

    eval_time = time.time() - eval_time_start
    total_time = time.time() - start_time

    # Aggregate metrics
    results = {
        'baseline': baseline_name,
        'seed': seed,
        'n_samples': len(samples),
        'train_time': train_time,
        'eval_time': eval_time,
        'total_time': total_time,
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"[{baseline_name}] Completed. Hit@5: {results['hit@5']:.4f}, Time: {total_time:.1f}s")
    return results


def run_baseline_on_device(args):
    """Wrapper for parallel execution."""
    baseline_name, samples, news_items, all_items, config, seed, device = args
    try:
        return run_single_baseline(baseline_name, samples, news_items, all_items, config, seed, device)
    except Exception as e:
        logger.error(f"Error running {baseline_name}: {e}")
        return {'baseline': baseline_name, 'seed': seed, 'error': str(e)}


def run_parallel_experiments(
    samples: List[RecommendationSample],
    news_items: Dict,
    all_items: List[str],
    config: ExperimentConfig,
):
    """Run all baselines with parallel execution."""
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {num_gpus}")

    # Device assignment strategy
    baseline_devices = {
        'NRMS': 'cuda:0' if num_gpus > 0 else 'cpu',
        'NAML': 'cuda:0' if num_gpus > 0 else 'cpu',
        'PLM-NR': 'cuda:1' if num_gpus > 1 else ('cuda:0' if num_gpus > 0 else 'cpu'),
        'TALLRec': 'cuda:1' if num_gpus > 1 else ('cuda:0' if num_gpus > 0 else 'cpu'),
        'Prompt4NR': 'cpu',  # Uses local LLM or falls back to heuristics
    }

    all_results = []
    baselines = ['NRMS', 'NAML', 'PLM-NR', 'TALLRec', 'Prompt4NR']

    # Run experiments for each seed
    for seed in config.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Run with seed={seed}")
        logger.info(f"{'='*60}")

        # Run baselines sequentially (due to GPU memory constraints)
        # Group 1: NRMS + NAML (lightweight, can share GPU 0)
        # Group 2: PLM-NR + TALLRec (heavy, need separate runs)
        # Group 3: Prompt4NR (CPU-based)

        for baseline in baselines:
            device = baseline_devices[baseline]
            result = run_single_baseline(
                baseline, samples, news_items, all_items, config, seed, device
            )
            all_results.append(result)

            # Clear GPU cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results


def compute_statistics(results: List[Dict], config: ExperimentConfig) -> Dict:
    """Compute aggregate statistics with significance tests."""
    from scipy import stats

    baselines = list(set(r['baseline'] for r in results if 'error' not in r))
    stats_results = {}

    for baseline in baselines:
        baseline_results = [r for r in results if r['baseline'] == baseline and 'error' not in r]

        if not baseline_results:
            continue

        stats_results[baseline] = {}

        for k in config.k_values:
            # Process hit@k, ndcg@k, mrr@k
            for metric_prefix in ['hit', 'ndcg', 'mrr']:
                key = f'{metric_prefix}@{k}'
                if key in baseline_results[0]:
                    values = [r[key] for r in baseline_results if key in r]
                    if values:
                        stats_results[baseline][key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'ci_95': (
                                np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values)),
                                np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)),
                            ),
                        }

    # Pairwise comparisons with our method (assuming Trie-LLM is baseline to compare)
    if 'NRMS' in stats_results and len(baselines) > 1:
        for other_baseline in baselines:
            if other_baseline == 'NRMS':
                continue

            nrms_values = [r['hit@5'] for r in results if r['baseline'] == 'NRMS' and 'error' not in r]
            other_values = [r['hit@5'] for r in results if r['baseline'] == other_baseline and 'error' not in r]

            if len(nrms_values) == len(other_values) and len(nrms_values) > 1:
                t_stat, p_value = stats.ttest_rel(nrms_values, other_values)
                cohens_d = (np.mean(nrms_values) - np.mean(other_values)) / np.sqrt(
                    (np.std(nrms_values)**2 + np.std(other_values)**2) / 2
                )

                stats_results[f'NRMS_vs_{other_baseline}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                }

    return stats_results


def save_results(results: List[Dict], stats: Dict, config: ExperimentConfig):
    """Save results to files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save statistics
    stats_file = output_dir / f"statistics_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Save summary table
    summary_file = output_dir / f"summary_{timestamp}.md"
    with open(summary_file, 'w') as f:
        f.write("# SOTA Baselines Comparison\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Samples**: {config.n_samples}\n")
        f.write(f"**Runs**: {config.num_runs}\n\n")

        f.write("## Results\n\n")
        f.write("| Baseline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@5 | MRR@5 |\n")
        f.write("|----------|-------|-------|-------|--------|--------|-------|\n")

        for baseline in ['NRMS', 'NAML', 'PLM-NR', 'TALLRec', 'Prompt4NR']:
            if baseline in stats:
                s = stats[baseline]
                f.write(f"| {baseline} | "
                       f"{s.get('hit@1', {}).get('mean', 0):.4f} | "
                       f"{s.get('hit@3', {}).get('mean', 0):.4f} | "
                       f"{s.get('hit@5', {}).get('mean', 0):.4f} | "
                       f"{s.get('hit@10', {}).get('mean', 0):.4f} | "
                       f"{s.get('ndcg@5', {}).get('mean', 0):.4f} | "
                       f"{s.get('mrr@5', {}).get('mean', 0):.4f} |\n")

        f.write("\n## Statistical Significance\n\n")
        for key, val in stats.items():
            if key.startswith('NRMS_vs_'):
                f.write(f"- **{key}**: p={val.get('p_value', 1):.4f}, Cohen's d={val.get('cohens_d', 0):.3f}\n")

    logger.info(f"Results saved to {output_dir}")
    return results_file, stats_file, summary_file


def main():
    parser = argparse.ArgumentParser(description="Run SOTA Baselines for KDD 2026")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--output", type=str, default="results/sota_baselines", help="Output directory")
    parser.add_argument("--baselines", type=str, nargs='+',
                       default=['NRMS', 'NAML', 'PLM-NR', 'TALLRec', 'Prompt4NR'],
                       help="Baselines to run")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_samples=args.samples,
        num_runs=args.runs,
        output_dir=args.output,
    )

    logger.info("="*60)
    logger.info("SOTA Baselines Experiment")
    logger.info("="*60)
    logger.info(f"Config: {config}")

    # Load data
    samples, news_items, all_items = load_mind_data(config.n_samples)

    # Run experiments
    results = run_parallel_experiments(samples, news_items, all_items, config)

    # Compute statistics
    stats = compute_statistics(results, config)

    # Save results
    save_results(results, stats, config)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for baseline in ['NRMS', 'NAML', 'PLM-NR', 'TALLRec', 'Prompt4NR']:
        if baseline in stats:
            hit5 = stats[baseline].get('hit@5', {}).get('mean', 0)
            std5 = stats[baseline].get('hit@5', {}).get('std', 0)
            print(f"{baseline:12s}: Hit@5 = {hit5:.4f} +/- {std5:.4f}")

    print("="*60)


if __name__ == "__main__":
    main()
