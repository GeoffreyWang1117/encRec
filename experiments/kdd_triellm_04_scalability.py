"""
Scalability Analysis for Trie+LLM.

Tests performance across:
1. Sample sizes: [1000, 2000, 5000, 10000]
2. Candidate pool sizes: [10, 20, 50, 100]

Usage:
    python experiments/scalability_experiments.py --device cuda:0
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_scalability_experiment(
    n_samples: int,
    candidate_pool_size: int,
    news_items: Dict,
    all_items: List[str],
    device: str,
    seed: int = 42,
) -> Dict:
    """Run experiment with specific scale parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Testing n_samples={n_samples}, candidates={candidate_pool_size}")

    # Load samples
    samples, _, _ = load_mind_for_trie_experiment(n_samples, seed=seed)

    start_time = time.time()

    config = ExperimentConfig(
        n_samples=n_samples,
        candidate_pool_size=candidate_pool_size,
        device=device,
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

    # Evaluate
    all_metrics = defaultdict(list)
    latencies = []

    for i, sample in enumerate(samples):
        req_start = time.time()

        # Sample candidates
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        n_neg = min(candidate_pool_size - 1, len(neg_items))
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=n_neg, replace=False
        ))
        np.random.shuffle(candidates)

        recs, meta = recommender.recommend(
            history=sample['history'],
            candidates=candidates,
            k=10,
            use_early_exit=False,
            use_compression=True,
            use_trie_filtering=True,
            use_ctr_signals=False,
        )

        req_latency = (time.time() - req_start) * 1000  # ms
        latencies.append(req_latency)

        metrics = evaluate_recommendations(recs, sample['ground_truth'], [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{n_samples}, Hit@5: {np.mean(all_metrics['hit@5']):.4f}, "
                       f"Latency: {np.mean(latencies):.1f}ms")

    total_time = time.time() - start_time

    results = {
        'n_samples': n_samples,
        'candidate_pool_size': candidate_pool_size,
        'seed': seed,
        'total_time': total_time,
        'avg_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'throughput_rps': n_samples / total_time,
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"  Done: Hit@5={results['hit@5']:.4f}, Latency={results['avg_latency_ms']:.1f}ms, "
               f"Throughput={results['throughput_rps']:.2f} req/s")

    del llm_ranker, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results/scalability")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("SCALABILITY ANALYSIS")
    logger.info("="*60)

    # Load full dataset once
    _, news_items, all_items = load_mind_for_trie_experiment(100)
    logger.info(f"Loaded {len(news_items)} news items")

    all_results = []

    # Test 1: Vary sample size (fixed 20 candidates)
    logger.info("\n" + "="*40)
    logger.info("Test 1: Varying Sample Size")
    logger.info("="*40)

    sample_sizes = [1000, 2000, 5000]
    for n_samples in sample_sizes:
        result = run_scalability_experiment(
            n_samples=n_samples,
            candidate_pool_size=20,
            news_items=news_items,
            all_items=all_items,
            device=args.device,
        )
        all_results.append(result)

    # Test 2: Vary candidate pool size (fixed 2000 samples)
    logger.info("\n" + "="*40)
    logger.info("Test 2: Varying Candidate Pool Size")
    logger.info("="*40)

    candidate_sizes = [10, 20, 50]
    for cand_size in candidate_sizes:
        if cand_size == 20:
            continue  # Already tested above
        result = run_scalability_experiment(
            n_samples=2000,
            candidate_pool_size=cand_size,
            news_items=news_items,
            all_items=all_items,
            device=args.device,
        )
        all_results.append(result)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"scalability_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("SCALABILITY SUMMARY")
    logger.info("="*60)

    logger.info("\nSample Size Scaling (20 candidates):")
    logger.info("| Samples | Hit@5 | Latency (ms) | Throughput |")
    logger.info("|---------|-------|--------------|------------|")
    for r in all_results:
        if r['candidate_pool_size'] == 20:
            logger.info(f"| {r['n_samples']:>7} | {r['hit@5']:.4f} | {r['avg_latency_ms']:>12.1f} | {r['throughput_rps']:>10.2f} |")

    logger.info("\nCandidate Pool Scaling (2000 samples):")
    logger.info("| Candidates | Hit@5 | Latency (ms) | Throughput |")
    logger.info("|------------|-------|--------------|------------|")
    for r in all_results:
        if r['n_samples'] == 2000:
            logger.info(f"| {r['candidate_pool_size']:>10} | {r['hit@5']:.4f} | {r['avg_latency_ms']:>12.1f} | {r['throughput_rps']:>10.2f} |")

    # Save markdown summary
    with open(output_dir / f"scalability_summary.md", 'w') as f:
        f.write("# Scalability Analysis\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Sample Size Scaling\n\n")
        f.write("| Samples | Hit@5 | Latency (ms) | Throughput (req/s) |\n")
        f.write("|---------|-------|--------------|--------------------|\n")
        for r in all_results:
            if r['candidate_pool_size'] == 20:
                f.write(f"| {r['n_samples']} | {r['hit@5']:.4f} | {r['avg_latency_ms']:.1f} | {r['throughput_rps']:.2f} |\n")

        f.write("\n## Candidate Pool Scaling\n\n")
        f.write("| Candidates | Hit@5 | Latency (ms) | Throughput (req/s) |\n")
        f.write("|------------|-------|--------------|--------------------|\n")
        for r in all_results:
            if r['n_samples'] == 2000:
                f.write(f"| {r['candidate_pool_size']} | {r['hit@5']:.4f} | {r['avg_latency_ms']:.1f} | {r['throughput_rps']:.2f} |\n")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
