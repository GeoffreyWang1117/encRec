"""
Hyperparameter Sensitivity Analysis for Trie+LLM.

Tests the following hyperparameters:
1. Early Exit Threshold τ: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
2. History Length: [3, 5, 10, 15, 20]
3. Candidate Pool Size: [10, 20, 30, 50, 100]

Usage:
    python experiments/hyperparameter_sensitivity.py --param threshold
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
import logging
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.kdd_triellm_01_ablation import (
    ExperimentConfig, load_mind_data, TrieStatistics,
    LocalLLMRanker, TrieLLMRecommender, evaluate_recommendations
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_sensitivity_experiment(
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    param_name: str,
    param_value: float,
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Run experiment with specific parameter value."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Testing {param_name}={param_value}, seed={seed}")
    start_time = time.time()

    # Modify config based on parameter
    modified_config = ExperimentConfig(**asdict(config))

    if param_name == 'threshold':
        modified_config.early_exit_threshold = param_value
    elif param_name == 'history_length':
        modified_config.history_length = int(param_value)
    elif param_name == 'candidate_pool':
        modified_config.candidate_pool_size = int(param_value)

    # Build components
    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=config.device)
    recommender = TrieLLMRecommender(
        news_items=news_items,
        trie_stats=trie_stats,
        llm_ranker=llm_ranker,
        config=modified_config,
    )

    # Evaluate
    all_metrics = defaultdict(list)
    meta_stats = defaultdict(int)

    for i, sample in enumerate(samples):
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=min(19, len(neg_items)), replace=False
        ))
        np.random.shuffle(candidates)

        recs, meta = recommender.recommend(
            history=sample['history'],
            candidates=candidates,
            k=max(config.k_values),
        )

        if meta['early_exit']:
            meta_stats['early_exits'] += 1
        if meta['llm_called']:
            meta_stats['llm_calls'] += 1
        meta_stats['total_tokens'] += meta.get('tokens', 0)

        metrics = evaluate_recommendations(recs, sample['ground_truth'], config.k_values)
        for key, value in metrics.items():
            all_metrics[key].append(value)

    total_time = time.time() - start_time

    results = {
        'param_name': param_name,
        'param_value': param_value,
        'seed': seed,
        'n_samples': len(samples),
        'total_time': total_time,
        'early_exit_rate': meta_stats['early_exits'] / len(samples),
        'llm_call_rate': meta_stats['llm_calls'] / len(samples),
        'avg_tokens': meta_stats['total_tokens'] / max(meta_stats['llm_calls'], 1),
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"  {param_name}={param_value}: Hit@5={results['hit@5']:.4f}, "
               f"Early Exit={results['early_exit_rate']:.1%}")

    del llm_ranker, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_threshold_sensitivity(samples, news_items, all_items, config):
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    for seed in config.seeds:
        for threshold in thresholds:
            result = run_sensitivity_experiment(
                samples, news_items, all_items,
                'threshold', threshold, config, seed
            )
            results.append(result)
    return results


def run_history_sensitivity(samples, news_items, all_items, config):
    history_lengths = [3, 5, 10, 15, 20]
    results = []
    for seed in config.seeds:
        for length in history_lengths:
            result = run_sensitivity_experiment(
                samples, news_items, all_items,
                'history_length', length, config, seed
            )
            results.append(result)
    return results


def aggregate_results(results, param_name):
    by_value = defaultdict(list)
    for r in results:
        by_value[r['param_value']].append(r)

    aggregated = {}
    for value, runs in by_value.items():
        aggregated[value] = {
            'hit@5_mean': np.mean([r['hit@5'] for r in runs]),
            'hit@5_std': np.std([r['hit@5'] for r in runs]),
            'early_exit_rate': np.mean([r['early_exit_rate'] for r in runs]),
        }
    return aggregated


def save_results(results, param_name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_path / f"sensitivity_{param_name}_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    aggregated = aggregate_results(results, param_name)
    with open(output_path / f"sensitivity_{param_name}_summary.md", 'w') as f:
        f.write(f"# {param_name} Sensitivity\n\n")
        f.write(f"| {param_name} | Hit@5 | Early Exit |\n")
        f.write("|------------|-------|------------|\n")
        for value in sorted(aggregated.keys()):
            s = aggregated[value]
            f.write(f"| {value} | {s['hit@5_mean']:.4f}±{s['hit@5_std']:.4f} | {s['early_exit_rate']:.1%} |\n")

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default="threshold")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/sensitivity")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_samples=args.samples,
        num_runs=args.runs,
        device=args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    )

    logger.info("HYPERPARAMETER SENSITIVITY ANALYSIS")
    samples, news_items, all_items = load_mind_data(config.n_samples)

    if args.param == 'threshold':
        results = run_threshold_sensitivity(samples, news_items, all_items, config)
        save_results(results, 'threshold', args.output)
    elif args.param == 'history':
        results = run_history_sensitivity(samples, news_items, all_items, config)
        save_results(results, 'history_length', args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
