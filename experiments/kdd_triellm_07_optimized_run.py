"""
Optimized Trie+LLM experiment based on ablation findings.

Key optimizations:
- Disabled CTR signals (hurt performance by -0.73%)
- Disabled early exit (hurt performance by -0.47%)
- Keep compression (helped by +2.14%)
- Keep trie filtering (helped by +0.44%)

Usage:
    python experiments/run_optimized_triellm.py --samples 5000 --runs 5 --device cuda:1
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.kdd_triellm_01_ablation import (
    ExperimentConfig, LocalLLMRanker, TrieStatistics,
    TrieLLMRecommender, evaluate_recommendations
)
from src.data.mind_loader import load_mind_for_trie_experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_optimized_experiment(
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Run optimized Trie+LLM experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Running Optimized Trie+LLM (seed={seed})")
    start_time = time.time()

    # Build components
    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=config.device)
    recommender = TrieLLMRecommender(
        news_items=news_items,
        trie_stats=trie_stats,
        llm_ranker=llm_ranker,
        config=config,
    )

    # Evaluate with optimized settings
    all_metrics = defaultdict(list)
    meta_stats = defaultdict(int)

    for i, sample in enumerate(samples):
        # Sample negative items
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=min(19, len(neg_items)), replace=False
        ))
        np.random.shuffle(candidates)

        # Optimized configuration:
        # - NO early exit (hurt performance)
        # - NO CTR signals (hurt performance)
        # - YES compression (helped)
        # - YES trie filtering (helped)
        recs, meta = recommender.recommend(
            history=sample['history'],
            candidates=candidates,
            k=max(config.k_values),
            use_early_exit=False,      # Disabled
            use_compression=True,       # Enabled
            use_trie_filtering=True,    # Enabled
            use_ctr_signals=False,      # Disabled
        )

        meta_stats['llm_calls'] += 1 if meta['llm_called'] else 0
        meta_stats['total_tokens'] += meta.get('tokens', 0)

        metrics = evaluate_recommendations(recs, sample['ground_truth'], config.k_values)
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{len(samples)}, Hit@5: {np.mean(all_metrics['hit@5']):.4f}")

    total_time = time.time() - start_time

    results = {
        'method': 'Optimized_TrieLLM',
        'seed': seed,
        'n_samples': len(samples),
        'total_time': total_time,
        'llm_call_rate': meta_stats['llm_calls'] / len(samples),
        'avg_tokens': meta_stats['total_tokens'] / max(meta_stats['llm_calls'], 1),
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"  Optimized_TrieLLM: Hit@5={results['hit@5']:.4f}, Time={total_time:.1f}s")

    # Cleanup
    del llm_ranker, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/optimized_triellm")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_samples=args.samples,
        num_runs=args.runs,
        device=args.device or ('cuda:1' if torch.cuda.is_available() else 'cpu'),
    )

    logger.info("="*60)
    logger.info("OPTIMIZED TRIE+LLM EXPERIMENT")
    logger.info("Configuration: NO early_exit, NO ctr_signals, YES compression")
    logger.info("="*60)

    # Load data
    samples, news_items, all_items = load_mind_for_trie_experiment(config.n_samples)
    logger.info(f"Loaded {len(samples)} samples, {len(news_items)} news items")

    # Run experiments
    all_results = []
    seeds = [42, 123, 456, 789, 1024][:config.num_runs]

    for seed in seeds:
        logger.info(f"\n{'='*40}")
        logger.info(f"Run with seed={seed}")
        logger.info(f"{'='*40}")

        result = run_optimized_experiment(samples, news_items, all_items, config, seed)
        all_results.append(result)

    # Compute statistics
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregate results
    metrics = ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'ndcg@5', 'mrr@5']
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in all_results if metric in r]
        if values:
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }

    # Save results
    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    for metric, stats in summary.items():
        logger.info(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Compare with baselines
    logger.info("\n" + "="*60)
    logger.info("COMPARISON WITH BASELINES (5000 samples, 5 runs)")
    logger.info("="*60)
    logger.info(f"Prompt4NR    : Hit@5 = 0.4208")
    logger.info(f"PLM-NR       : Hit@5 = 0.3880")
    logger.info(f"NAML         : Hit@5 = 0.3431")
    logger.info(f"Optimized TLM: Hit@5 = {summary.get('hit@5', {}).get('mean', 0):.4f}")


if __name__ == "__main__":
    main()
