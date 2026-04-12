"""
Error Analysis for Trie+LLM.

Analyzes:
1. Performance by news category
2. Performance by user history length
3. Performance by item popularity
4. Failure case analysis

Usage:
    python experiments/error_analysis.py --samples 2000 --device cuda:0
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
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


def run_error_analysis(
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    device: str,
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    """Run experiment and collect detailed per-sample results."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = ExperimentConfig(
        n_samples=len(samples),
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

    # Compute item popularity
    item_counts = Counter()
    for sample in samples:
        for item_id in sample['history']:
            item_counts[item_id] += 1

    detailed_results = []

    for i, sample in enumerate(samples):
        # Sample candidates
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=min(19, len(neg_items)), replace=False
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

        # Get ground truth info
        gt_item = news_items.get(sample['ground_truth'], {})
        gt_category = gt_item.get('category', 'unknown')
        gt_popularity = item_counts.get(sample['ground_truth'], 0)

        # Get user history info
        history_categories = [news_items.get(h, {}).get('category', 'unknown') for h in sample['history']]
        dominant_cat = Counter(history_categories).most_common(1)[0][0] if history_categories else 'unknown'

        # Evaluate
        hit_at_5 = 1.0 if sample['ground_truth'] in recs[:5] else 0.0
        hit_at_1 = 1.0 if sample['ground_truth'] in recs[:1] else 0.0

        rank = recs.index(sample['ground_truth']) + 1 if sample['ground_truth'] in recs else -1

        detailed_results.append({
            'sample_id': i,
            'ground_truth': sample['ground_truth'],
            'gt_category': gt_category,
            'gt_popularity': gt_popularity,
            'history_length': len(sample['history']),
            'dominant_history_cat': dominant_cat,
            'category_match': gt_category == dominant_cat,
            'hit_at_1': hit_at_1,
            'hit_at_5': hit_at_5,
            'rank': rank,
            'recs_top3': recs[:3],
        })

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{len(samples)}")

    del llm_ranker, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detailed_results, dict(item_counts)


def analyze_by_category(results: List[Dict]) -> Dict:
    """Analyze performance by news category."""
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['gt_category']].append(r['hit_at_5'])

    analysis = {}
    for cat, hits in by_cat.items():
        if len(hits) >= 10:  # Only categories with enough samples
            analysis[cat] = {
                'count': len(hits),
                'hit_at_5': np.mean(hits),
                'std': np.std(hits),
            }

    return dict(sorted(analysis.items(), key=lambda x: -x[1]['hit_at_5']))


def analyze_by_history_length(results: List[Dict]) -> Dict:
    """Analyze performance by user history length."""
    bins = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 20), (21, float('inf'))]
    bin_names = ['1-3', '4-6', '7-10', '11-15', '16-20', '21+']

    analysis = {}
    for (low, high), name in zip(bins, bin_names):
        hits = [r['hit_at_5'] for r in results if low <= r['history_length'] <= high]
        if hits:
            analysis[name] = {
                'count': len(hits),
                'hit_at_5': np.mean(hits),
                'std': np.std(hits),
            }

    return analysis


def analyze_by_popularity(results: List[Dict], item_counts: Dict) -> Dict:
    """Analyze performance by item popularity."""
    # Compute popularity percentiles
    counts = list(item_counts.values())
    if not counts:
        return {}

    p25, p50, p75 = np.percentile(counts, [25, 50, 75])

    bins = [
        ('cold (0)', lambda x: x == 0),
        ('rare (1-p25)', lambda x: 1 <= x <= p25),
        ('moderate (p25-p50)', lambda x: p25 < x <= p50),
        ('popular (p50-p75)', lambda x: p50 < x <= p75),
        ('very popular (>p75)', lambda x: x > p75),
    ]

    analysis = {}
    for name, condition in bins:
        hits = [r['hit_at_5'] for r in results if condition(r['gt_popularity'])]
        if hits:
            analysis[name] = {
                'count': len(hits),
                'hit_at_5': np.mean(hits),
                'std': np.std(hits),
            }

    return analysis


def analyze_category_match(results: List[Dict]) -> Dict:
    """Analyze performance when category matches vs doesn't match."""
    match_hits = [r['hit_at_5'] for r in results if r['category_match']]
    nomatch_hits = [r['hit_at_5'] for r in results if not r['category_match']]

    return {
        'category_match': {
            'count': len(match_hits),
            'hit_at_5': np.mean(match_hits) if match_hits else 0,
        },
        'category_mismatch': {
            'count': len(nomatch_hits),
            'hit_at_5': np.mean(nomatch_hits) if nomatch_hits else 0,
        }
    }


def find_failure_cases(results: List[Dict], news_items: Dict, n: int = 10) -> List[Dict]:
    """Find interesting failure cases for qualitative analysis."""
    failures = [r for r in results if r['hit_at_5'] == 0]

    # Sort by: category match but still failed (most surprising)
    surprising = [f for f in failures if f['category_match']]
    surprising.sort(key=lambda x: -x['gt_popularity'])  # Popular items that failed

    cases = []
    for f in surprising[:n]:
        gt_item = news_items.get(f['ground_truth'], {})
        rec_items = [news_items.get(r, {}) for r in f['recs_top3']]

        cases.append({
            'ground_truth': {
                'id': f['ground_truth'],
                'title': gt_item.get('title', '')[:80],
                'category': f['gt_category'],
            },
            'recommendations': [
                {'title': r.get('title', '')[:60], 'category': r.get('category', '')}
                for r in rec_items
            ],
            'history_length': f['history_length'],
            'category_match': f['category_match'],
        })

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results/error_analysis")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("ERROR ANALYSIS")
    logger.info("="*60)

    # Load data
    samples, news_items, all_items = load_mind_for_trie_experiment(args.samples)
    logger.info(f"Loaded {len(samples)} samples, {len(news_items)} news items")

    # Run experiment
    logger.info("\nRunning recommendation experiment...")
    detailed_results, item_counts = run_error_analysis(
        samples, news_items, all_items, args.device
    )

    overall_hit5 = np.mean([r['hit_at_5'] for r in detailed_results])
    logger.info(f"Overall Hit@5: {overall_hit5:.4f}")

    # Analyze by category
    logger.info("\n" + "="*40)
    logger.info("Analysis by Category")
    logger.info("="*40)
    cat_analysis = analyze_by_category(detailed_results)
    for cat, stats in list(cat_analysis.items())[:10]:
        logger.info(f"  {cat:20s}: Hit@5={stats['hit_at_5']:.4f} (n={stats['count']})")

    # Analyze by history length
    logger.info("\n" + "="*40)
    logger.info("Analysis by History Length")
    logger.info("="*40)
    hist_analysis = analyze_by_history_length(detailed_results)
    for length, stats in hist_analysis.items():
        logger.info(f"  {length:10s}: Hit@5={stats['hit_at_5']:.4f} (n={stats['count']})")

    # Analyze by popularity
    logger.info("\n" + "="*40)
    logger.info("Analysis by Item Popularity")
    logger.info("="*40)
    pop_analysis = analyze_by_popularity(detailed_results, item_counts)
    for pop, stats in pop_analysis.items():
        logger.info(f"  {pop:25s}: Hit@5={stats['hit_at_5']:.4f} (n={stats['count']})")

    # Analyze category match
    logger.info("\n" + "="*40)
    logger.info("Analysis by Category Match")
    logger.info("="*40)
    match_analysis = analyze_category_match(detailed_results)
    for match_type, stats in match_analysis.items():
        logger.info(f"  {match_type:20s}: Hit@5={stats['hit_at_5']:.4f} (n={stats['count']})")

    # Failure cases
    logger.info("\n" + "="*40)
    logger.info("Failure Case Analysis")
    logger.info("="*40)
    failure_cases = find_failure_cases(detailed_results, news_items, n=5)
    for i, case in enumerate(failure_cases):
        logger.info(f"\nCase {i+1}:")
        logger.info(f"  Ground Truth: [{case['ground_truth']['category']}] {case['ground_truth']['title']}")
        logger.info(f"  Top Recommendations:")
        for j, rec in enumerate(case['recommendations']):
            logger.info(f"    {j+1}. [{rec['category']}] {rec['title']}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    analysis_results = {
        'overall_hit5': overall_hit5,
        'by_category': cat_analysis,
        'by_history_length': hist_analysis,
        'by_popularity': pop_analysis,
        'by_category_match': match_analysis,
        'failure_cases': failure_cases,
    }

    with open(output_dir / f"error_analysis_{timestamp}.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    # Save markdown summary
    with open(output_dir / f"error_analysis_summary.md", 'w') as f:
        f.write("# Error Analysis\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Samples**: {args.samples}\n")
        f.write(f"**Overall Hit@5**: {overall_hit5:.4f}\n\n")

        f.write("## Performance by Category\n\n")
        f.write("| Category | Hit@5 | Count |\n")
        f.write("|----------|-------|-------|\n")
        for cat, stats in list(cat_analysis.items())[:10]:
            f.write(f"| {cat} | {stats['hit_at_5']:.4f} | {stats['count']} |\n")

        f.write("\n## Performance by History Length\n\n")
        f.write("| History Length | Hit@5 | Count |\n")
        f.write("|----------------|-------|-------|\n")
        for length, stats in hist_analysis.items():
            f.write(f"| {length} | {stats['hit_at_5']:.4f} | {stats['count']} |\n")

        f.write("\n## Performance by Item Popularity\n\n")
        f.write("| Popularity | Hit@5 | Count |\n")
        f.write("|------------|-------|-------|\n")
        for pop, stats in pop_analysis.items():
            f.write(f"| {pop} | {stats['hit_at_5']:.4f} | {stats['count']} |\n")

        f.write("\n## Category Match Analysis\n\n")
        f.write("| Type | Hit@5 | Count |\n")
        f.write("|------|-------|-------|\n")
        for match_type, stats in match_analysis.items():
            f.write(f"| {match_type} | {stats['hit_at_5']:.4f} | {stats['count']} |\n")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
