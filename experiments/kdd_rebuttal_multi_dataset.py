"""
KDD 2026 Rebuttal: Multi-Dataset Validation Experiments.

Runs Trie+LLM (GPT-2) on Amazon Electronics and MovieLens 1M
to address reviewer concern about single-dataset evaluation.

Usage:
    # Amazon Electronics
    python experiments/kdd_rebuttal_multi_dataset.py \
        --dataset amazon --category Electronics \
        --data_dir ~/DataSets/amazon \
        --samples 5000 --runs 3 --device cuda:1

    # MovieLens 1M
    python experiments/kdd_rebuttal_multi_dataset.py \
        --dataset movielens \
        --data_dir data/ml-1m \
        --samples 5000 --runs 3 --device cuda:1
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# Dataset Loaders (adapted from kdd_triellm_explore_comprehensive.py)
# =====================================================================

@dataclass
class RecSample:
    user_id: str
    history: List[str]
    candidates: List[str]
    ground_truth: str  # Single ground truth item ID


def load_amazon(data_dir: str, category: str = "Electronics",
                max_samples: int = 5000, seed: int = 42) -> Tuple[List[RecSample], Dict, List[str]]:
    """Load Amazon dataset and create recommendation samples."""
    import pandas as pd
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    reviews_path = data_dir / f"{category}_reviews.parquet"
    meta_path = data_dir / f"{category}_meta.parquet"

    logger.info(f"Loading Amazon {category} from {reviews_path}")

    # Sample subset of reviews for tractability (43M is too much)
    df = pd.read_parquet(reviews_path)
    logger.info(f"Loaded {len(df)} reviews total")

    # Filter to active users first, then subsample
    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 10].index
    df = df[df['user_id'].isin(active_users)]
    logger.info(f"Filtered to {len(df)} reviews from {len(active_users)} active users")

    # If still too large, randomly sample USERS (not reviews) to preserve histories
    if len(active_users) > 20000:
        sampled_users = pd.Series(active_users).sample(n=20000, random_state=seed)
        df = df[df['user_id'].isin(sampled_users)]
        logger.info(f"Subsampled to {len(df)} reviews from 20000 users")

    # Load metadata (vectorized)
    items = {}
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path, columns=['parent_asin', 'title', 'main_category'])
        meta_df = meta_df.dropna(subset=['parent_asin'])
        meta_df['parent_asin'] = meta_df['parent_asin'].astype(str)
        meta_df['title'] = meta_df['title'].fillna('').astype(str).str[:100]
        meta_df['main_category'] = meta_df['main_category'].fillna('unknown').astype(str).str[:50]
        items = {
            row.parent_asin: {
                'title': row.title, 'category': row.main_category,
                'abstract': '', 'subcategory': row.main_category,
            }
            for row in meta_df.itertuples()
        }
        logger.info(f"Loaded {len(items)} product metadata")
        del meta_df

    # Build user interaction sequences (vectorized groupby)
    asin_col = 'parent_asin' if 'parent_asin' in df.columns else 'asin'
    df = df[['user_id', asin_col, 'rating', 'timestamp']].copy()
    df.columns = ['_uid', '_asin', '_rating', '_ts']
    df['_uid'] = df['_uid'].astype(str)
    df['_asin'] = df['_asin'].astype(str)
    df['_rating'] = df['_rating'].astype(float)
    df['_ts'] = pd.to_numeric(df['_ts'], errors='coerce').fillna(0)

    # Register all unique items
    unique_asins = df['_asin'].unique()
    for a in unique_asins:
        if a not in items:
            items[a] = {'title': a, 'category': 'unknown',
                        'abstract': '', 'subcategory': 'unknown'}

    # Vectorized groupby instead of row-by-row loop
    logger.info("Building user sequences via groupby...")
    user_items = {}
    for uid, group in df.groupby('_uid'):
        user_items[uid] = list(zip(group['_asin'], group['_rating'], group['_ts']))
    del df
    logger.info(f"Built {len(user_items)} user sequences")

    all_items_arr = np.array(list(items.keys()))
    logger.info(f"Total items: {len(all_items_arr)}, Total users: {len(user_items)}")

    # Create samples: users with >= 10 interactions, rating >= 4 as positive
    samples = []
    user_ids = list(user_items.keys())
    random.shuffle(user_ids)

    logger.info("Creating recommendation samples...")
    for uid in user_ids:
        interactions = user_items[uid]
        if len(interactions) < 10:
            continue

        interactions.sort(key=lambda x: x[2])

        # Last item as ground truth if rating >= 4
        for i in range(len(interactions) - 1, max(len(interactions) - 5, -1), -1):
            asin, rating, _ = interactions[i]
            if rating >= 4.0:
                history = [x[0] for x in interactions[:i]][-15:]
                if len(history) < 3:
                    continue

                # Fast negative sampling from pre-computed array
                negatives = list(np.random.choice(all_items_arr, size=19, replace=False))
                # Remove ground truth and history items if sampled
                negatives = [n for n in negatives if n != asin and n not in history][:19]
                candidates = [asin] + negatives
                random.shuffle(candidates)

                samples.append(RecSample(
                    user_id=uid,
                    history=history,
                    candidates=candidates,
                    ground_truth=asin,
                ))
                break

        if len(samples) >= max_samples:
            break
        if len(samples) % 1000 == 0 and len(samples) > 0:
            logger.info(f"  Created {len(samples)} samples so far...")

    logger.info(f"Created {len(samples)} Amazon samples")
    return samples, items, list(all_items_arr)


def load_movielens(data_dir: str, max_samples: int = 5000,
                   seed: int = 42) -> Tuple[List[RecSample], Dict, List[str]]:
    """Load MovieLens 1M and create recommendation samples."""
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)

    # Load movies
    items = {}
    movies_path = data_dir / "movies.dat"
    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                mid, title, genres = parts[:3]
                items[mid] = {
                    'title': title,
                    'category': genres.split('|')[0],
                    'abstract': genres,
                    'subcategory': genres.split('|')[0],
                }

    logger.info(f"Loaded {len(items)} movies")

    # Load ratings
    user_items = defaultdict(list)
    ratings_path = data_dir / "ratings.dat"
    with open(ratings_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                uid, mid, rating, ts = parts[:4]
                user_items[uid].append((mid, int(rating), int(ts)))

    all_items = list(items.keys())
    logger.info(f"Total items: {len(all_items)}, Total users: {len(user_items)}")

    # Create samples
    samples = []
    user_ids = list(user_items.keys())
    random.shuffle(user_ids)

    for uid in user_ids:
        interactions = user_items[uid]
        if len(interactions) < 15:
            continue

        interactions.sort(key=lambda x: x[2])

        # Last high-rated item as ground truth
        for i in range(len(interactions) - 1, max(len(interactions) - 5, -1), -1):
            mid, rating, _ = interactions[i]
            if rating >= 4:
                history = [x[0] for x in interactions[:i]][-15:]
                if len(history) < 5:
                    continue

                neg_pool = [it for it in all_items if it != mid and it not in history]
                n_neg = min(19, len(neg_pool))
                negatives = list(np.random.choice(neg_pool, size=n_neg, replace=False))
                candidates = [mid] + negatives
                random.shuffle(candidates)

                samples.append(RecSample(
                    user_id=uid,
                    history=history,
                    candidates=candidates,
                    ground_truth=mid,
                ))
                break

        if len(samples) >= max_samples:
            break

    logger.info(f"Created {len(samples)} MovieLens samples")
    return samples, items, all_items


# =====================================================================
# Experiment Core (reuses ablation infrastructure)
# =====================================================================

def run_experiment(
    samples: List[RecSample],
    news_items: Dict,
    all_items: List[str],
    device: str,
    seed: int,
    candidate_pool_size: int = 20,
) -> Dict:
    """Run Trie+LLM experiment on any dataset."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ExperimentConfig = _mod.ExperimentConfig
    LocalLLMRanker = _mod.LocalLLMRanker
    TrieStatistics = _mod.TrieStatistics
    TrieLLMRecommender = _mod.TrieLLMRecommender
    evaluate_recommendations = _mod.evaluate_recommendations

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    logger.info(f"Running experiment: {len(samples)} samples, seed={seed}")

    # Convert samples to the format expected by ablation infrastructure
    mind_format_samples = []
    for s in samples:
        mind_format_samples.append({
            'user_id': s.user_id,
            'history': s.history,
            'ground_truth': s.ground_truth,
        })

    config = ExperimentConfig(
        n_samples=len(samples),
        candidate_pool_size=candidate_pool_size,
        device=device,
        early_exit_threshold=0.5,
        history_length=15,
    )

    # Build components
    trie_stats = TrieStatistics(news_items, mind_format_samples)
    llm_ranker = LocalLLMRanker(device=device)
    recommender = TrieLLMRecommender(
        news_items=news_items,
        trie_stats=trie_stats,
        llm_ranker=llm_ranker,
        config=config,
    )

    all_metrics = defaultdict(list)
    latencies = []
    meta_stats = defaultdict(int)
    start_time = time.time()

    for i, sample in enumerate(samples):
        req_start = time.time()

        recs, meta = recommender.recommend(
            history=sample.history,
            candidates=sample.candidates,
            k=10,
            use_early_exit=False,
            use_compression=True,
            use_trie_filtering=True,
            use_ctr_signals=False,
        )

        req_latency = (time.time() - req_start) * 1000
        latencies.append(req_latency)

        if meta.get('cache_hit'):
            meta_stats['cache_hits'] += 1
        if meta.get('early_exit'):
            meta_stats['early_exits'] += 1
        if meta.get('llm_called'):
            meta_stats['llm_calls'] += 1
        meta_stats['total_tokens'] += meta.get('tokens', 0)

        metrics = evaluate_recommendations(recs, sample.ground_truth, [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 500 == 0:
            hit5 = np.mean(all_metrics['hit@5'])
            avg_lat = np.mean(latencies)
            logger.info(f"  [{i+1}/{len(samples)}] Hit@5={hit5:.4f}, Latency={avg_lat:.1f}ms")

    total_time = time.time() - start_time

    results = {
        'n_samples': len(samples),
        'seed': seed,
        'total_time': total_time,
        'avg_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'throughput_rps': len(samples) / total_time,
        'cache_hit_rate': meta_stats['cache_hits'] / len(samples),
        'early_exit_rate': meta_stats['early_exits'] / len(samples),
        'llm_call_rate': meta_stats['llm_calls'] / max(len(samples), 1),
        'avg_tokens': meta_stats['total_tokens'] / max(meta_stats['llm_calls'], 1),
    }

    for key, values in all_metrics.items():
        results[key] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))

    logger.info(f"Done: Hit@5={results['hit@5']:.4f}, NDCG@5={results.get('ndcg@5', 0):.4f}")

    del llm_ranker, recommender
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_trie_only(
    samples: List[RecSample],
    news_items: Dict,
    seed: int,
) -> Dict:
    """Run Trie-only baseline."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    TrieStatistics = _mod.TrieStatistics
    evaluate_recommendations = _mod.evaluate_recommendations

    np.random.seed(seed)
    random.seed(seed)

    mind_format = [{'user_id': s.user_id, 'history': s.history,
                    'ground_truth': s.ground_truth} for s in samples]
    trie_stats = TrieStatistics(news_items, mind_format)

    all_metrics = defaultdict(list)
    latencies = []

    for sample in samples:
        start = time.time()

        # Score by category match + popularity
        user_cats = defaultdict(int)
        for h in sample.history[-15:]:
            cat = trie_stats.get_category(h)
            user_cats[cat] += 1

        scores = []
        for cid in sample.candidates:
            ctr = trie_stats.get_ctr(cid)
            cat = trie_stats.get_category(cid)
            cat_match = 1.0 + user_cats.get(cat, 0) * 0.2
            score = ctr * cat_match
            scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [x[0] for x in scores[:10]]

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        metrics = evaluate_recommendations(recs, sample.ground_truth, [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

    results = {'method': 'trie_only', 'seed': seed, 'n_samples': len(samples),
               'avg_latency_ms': np.mean(latencies)}
    for key, values in all_metrics.items():
        results[key] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))

    logger.info(f"Trie-only: Hit@5={results['hit@5']:.4f}")
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="KDD Rebuttal: Multi-Dataset Validation")
    parser.add_argument("--dataset", type=str, required=True, choices=["amazon", "movielens"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="Electronics",
                        help="Amazon category (e.g., Electronics, Books)")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/kdd_rebuttal_{args.dataset}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"KDD REBUTTAL: {args.dataset.upper()} VALIDATION")
    logger.info(f"Samples: {args.samples}, Runs: {args.runs}, Device: {args.device}")
    logger.info("=" * 60)

    seeds = [42, 123, 456, 789, 1024][:args.runs]
    all_trie_results = []
    all_triellm_results = []

    for seed in seeds:
        logger.info(f"\n{'='*40} Run seed={seed} {'='*40}")

        # Load data
        if args.dataset == "amazon":
            samples, items, all_items = load_amazon(
                args.data_dir, args.category, args.samples, seed)
        else:
            samples, items, all_items = load_movielens(
                args.data_dir, args.samples, seed)

        if not samples:
            logger.error("No samples loaded!")
            return

        # Run Trie-only
        trie_result = run_trie_only(samples, items, seed)
        all_trie_results.append(trie_result)

        # Run Trie+LLM
        triellm_result = run_experiment(
            samples, items, all_items, args.device, seed)
        all_triellm_results.append(triellm_result)

    # Aggregate
    def aggregate(results_list, method_name):
        agg = {'method': method_name}
        metric_keys = [k for k in results_list[0] if k.startswith('hit@') or
                       k.startswith('ndcg@') or k.startswith('mrr@')]
        metric_keys = [k for k in metric_keys if not k.endswith('_std')]
        for k in metric_keys:
            vals = [r[k] for r in results_list]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        if 'avg_latency_ms' in results_list[0]:
            lats = [r['avg_latency_ms'] for r in results_list]
            agg['avg_latency_ms'] = {'mean': float(np.mean(lats)), 'std': float(np.std(lats))}
        if 'avg_tokens' in results_list[0]:
            toks = [r.get('avg_tokens', 0) for r in results_list]
            agg['avg_tokens'] = {'mean': float(np.mean(toks)), 'std': float(np.std(toks))}
        return agg

    trie_agg = aggregate(all_trie_results, 'trie_only')
    triellm_agg = aggregate(all_triellm_results, 'trie_llm')

    # Statistical test
    from scipy import stats as scipy_stats
    trie_hits = [r['hit@5'] for r in all_trie_results]
    triellm_hits = [r['hit@5'] for r in all_triellm_results]
    if len(trie_hits) > 1:
        t_stat, p_val = scipy_stats.ttest_rel(triellm_hits, trie_hits)
    else:
        t_stat, p_val = 0, 1

    final = {
        'dataset': args.dataset,
        'category': args.category if args.dataset == 'amazon' else 'N/A',
        'n_samples': args.samples,
        'n_runs': args.runs,
        'timestamp': datetime.now().isoformat(),
        'trie_only': trie_agg,
        'trie_llm': triellm_agg,
        'raw_trie': all_trie_results,
        'raw_triellm': all_triellm_results,
        'statistical_test': {
            'method': 'paired_t_test',
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
        }
    }

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"rebuttal_{args.dataset}_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for method, agg in [('Trie-only', trie_agg), ('Trie+LLM', triellm_agg)]:
        h5 = agg.get('hit@5', {})
        n5 = agg.get('ndcg@5', {})
        logger.info(f"{method}: Hit@5={h5.get('mean',0):.4f}±{h5.get('std',0):.4f}, "
                     f"NDCG@5={n5.get('mean',0):.4f}±{n5.get('std',0):.4f}")
    logger.info(f"Statistical test: t={t_stat:.3f}, p={p_val:.4f}")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
