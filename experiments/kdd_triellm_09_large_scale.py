"""
Large-Scale MIND Experiments for KDD 2026.

Extends the scalability experiments to industrial-scale:
- 50K, 100K samples
- Multiple runs with statistical analysis
- Full metrics suite

Usage:
    python experiments/kdd_triellm_09_large_scale.py --samples 50000 --runs 3 --device cuda:0
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

from src.data.mind_loader import load_mind_for_trie_experiment
from experiments.kdd_triellm_01_ablation import (
    ExperimentConfig, LocalLLMRanker, TrieStatistics,
    TrieLLMRecommender, evaluate_recommendations
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_large_scale_experiment(
    n_samples: int,
    news_items: Dict,
    all_items: List[str],
    device: str,
    seed: int = 42,
    candidate_pool_size: int = 20,
) -> Dict:
    """Run large-scale experiment on MIND."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Starting large-scale experiment: n_samples={n_samples}, seed={seed}")

    # Load samples
    samples, _, _ = load_mind_for_trie_experiment(n_samples, seed=seed)
    logger.info(f"Loaded {len(samples)} samples")

    start_time = time.time()

    config = ExperimentConfig(
        n_samples=n_samples,
        candidate_pool_size=candidate_pool_size,
        device=device,
        early_exit_threshold=0.5,  # Optimal from sensitivity analysis
        history_length=15,  # Optimal from sensitivity analysis
    )

    # Build components
    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=device)
    recommender = TrieLLMRecommender(
        news_items=news_items,
        trie_stats=trie_stats,
        llm_ranker=llm_ranker,
        config=config,
    )

    # Evaluate with optimized configuration (from ablation findings)
    all_metrics = defaultdict(list)
    latencies = []
    meta_stats = defaultdict(int)

    for i, sample in enumerate(samples):
        req_start = time.time()

        # Sample candidates
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        n_neg = min(candidate_pool_size - 1, len(neg_items))
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=n_neg, replace=False
        ))
        np.random.shuffle(candidates)

        # Use optimized configuration (no CTR, no early exit, with compression)
        recs, meta = recommender.recommend(
            history=sample['history'],
            candidates=candidates,
            k=10,
            use_early_exit=False,  # Hurts performance per ablation
            use_compression=True,   # Helps performance per ablation
            use_trie_filtering=True,
            use_ctr_signals=False,  # Hurts performance per ablation
        )

        req_latency = (time.time() - req_start) * 1000  # ms
        latencies.append(req_latency)

        # Track meta stats
        if meta.get('cache_hit'):
            meta_stats['cache_hits'] += 1
        if meta.get('early_exit'):
            meta_stats['early_exits'] += 1
        if meta.get('llm_called'):
            meta_stats['llm_calls'] += 1
        meta_stats['total_tokens'] += meta.get('tokens', 0)

        # Compute metrics
        metrics = evaluate_recommendations(recs, sample['ground_truth'], [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 1000 == 0:
            hit5 = np.mean(all_metrics['hit@5'])
            avg_latency = np.mean(latencies)
            logger.info(f"  Progress: {i+1}/{n_samples}, Hit@5: {hit5:.4f}, Latency: {avg_latency:.1f}ms")

    total_time = time.time() - start_time

    results = {
        'n_samples': n_samples,
        'candidate_pool_size': candidate_pool_size,
        'seed': seed,
        'total_time': total_time,
        'avg_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_rps': n_samples / total_time,
        'cache_hit_rate': meta_stats['cache_hits'] / n_samples,
        'early_exit_rate': meta_stats['early_exits'] / n_samples,
        'llm_call_rate': meta_stats['llm_calls'] / n_samples,
        'avg_tokens': meta_stats['total_tokens'] / max(meta_stats['llm_calls'], 1),
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"Completed: Hit@5={results['hit@5']:.4f}, Latency={results['avg_latency_ms']:.1f}ms, "
               f"Throughput={results['throughput_rps']:.2f} req/s")

    # Clean up GPU memory
    del llm_ranker, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Large-Scale MIND Experiment")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results/kdd_triellm_large_scale")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("LARGE-SCALE MIND EXPERIMENT")
    logger.info(f"Samples: {args.samples}, Runs: {args.runs}")
    logger.info("="*60)

    # Check GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB")

    # Load full dataset once
    _, news_items, all_items = load_mind_for_trie_experiment(100)
    logger.info(f"Loaded {len(news_items)} news items, {len(all_items)} total items")

    # Run experiments
    all_results = []
    seeds = [42, 123, 456, 789, 1024][:args.runs]

    for seed in seeds:
        logger.info(f"\n{'='*40}")
        logger.info(f"Run with seed={seed}")
        logger.info(f"{'='*40}")

        result = run_large_scale_experiment(
            n_samples=args.samples,
            news_items=news_items,
            all_items=all_items,
            device=args.device,
            seed=seed,
        )
        all_results.append(result)

    # Compute aggregate statistics
    logger.info("\n" + "="*60)
    logger.info("AGGREGATE RESULTS")
    logger.info("="*60)

    metrics_to_report = ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'ndcg@5', 'mrr@5']

    for metric in metrics_to_report:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"{metric}: {mean_val:.4f} +/- {std_val:.4f}")

    # Performance metrics
    latencies = [r['avg_latency_ms'] for r in all_results]
    throughputs = [r['throughput_rps'] for r in all_results]
    logger.info(f"\nLatency: {np.mean(latencies):.1f} +/- {np.std(latencies):.1f} ms")
    logger.info(f"Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f} req/s")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"large_scale_{args.samples}_{timestamp}.json", 'w') as f:
        json.dump({
            'config': {
                'n_samples': args.samples,
                'runs': args.runs,
                'device': args.device,
            },
            'results': all_results,
            'aggregate': {
                metric: {
                    'mean': np.mean([r[metric] for r in all_results]),
                    'std': np.std([r[metric] for r in all_results]),
                } for metric in metrics_to_report
            }
        }, f, indent=2, default=str)

    # Save summary
    with open(output_dir / f"summary_{args.samples}_{timestamp}.md", 'w') as f:
        f.write(f"# Large-Scale MIND Experiment Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Samples**: {args.samples}\n")
        f.write(f"**Runs**: {args.runs}\n\n")

        f.write("## Results\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|--------|------|-----|\n")
        for metric in metrics_to_report:
            values = [r[metric] for r in all_results]
            f.write(f"| {metric} | {np.mean(values):.4f} | {np.std(values):.4f} |\n")

        f.write("\n## Performance\n\n")
        f.write(f"- **Latency**: {np.mean(latencies):.1f} +/- {np.std(latencies):.1f} ms\n")
        f.write(f"- **Throughput**: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f} req/s\n")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
