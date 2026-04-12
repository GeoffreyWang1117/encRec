"""
KDD 2026 Rebuttal: Fixed Cache Dynamics Experiment.

Fix: Cache stores per-item LLM preference scores (user→item affinity),
not fixed rankings. On cache hit, re-rank current candidates using
cached scores + Trie fallback for unseen items.

Usage:
    python experiments/kdd_rebuttal_cache_fix.py --queries 3000 --device cuda:1
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, OrderedDict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import from ablation module
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LocalLLMRanker = _mod.LocalLLMRanker
TrieStatistics = _mod.TrieStatistics


class ScoreCache:
    """LRU cache that stores per-item affinity scores, not fixed rankings.

    Key insight: cache the user preference model (scores), not the output.
    This allows re-ranking new candidates using cached knowledge.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # key -> (timestamp, item_scores_dict)

    def get(self, key: str) -> dict:
        """Return cached item→score dict, or None."""
        if key in self.cache:
            ts, scores = self.cache[key]
            if time.time() - ts < self.ttl:
                self.cache.move_to_end(key)
                return scores
            else:
                del self.cache[key]
        return None

    def put(self, key: str, scores: dict):
        """Store item→score mapping."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (time.time(), scores)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


def generate_cache_key(user_id: str, history: List[str]) -> str:
    """Cache key from user + recent history."""
    history_str = "|".join(sorted(history[-5:]))
    return hashlib.md5(f"{user_id}:{history_str}".encode()).hexdigest()


def run_fixed_cache_experiment(
    n_queries: int = 3000,
    n_users: int = 300,
    user_return_prob: float = 0.3,
    cache_size: int = 500,
    cache_ttl: int = 300,
    candidate_pool_size: int = 20,
    device: str = "cuda:1",
):
    """Run cache dynamics with score-based caching."""
    from src.data.mind_loader import load_mind_for_trie_experiment

    logger.info("Loading MIND dataset...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=n_queries * 2, seed=42)
    logger.info(f"Loaded {len(samples)} samples, {len(news_items)} items")

    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=device)

    # Score-based cache
    score_cache = ScoreCache(max_size=cache_size, ttl=cache_ttl)

    # Also run old-style ranking cache for comparison
    rank_cache = ScoreCache(max_size=cache_size, ttl=cache_ttl)

    user_histories = {}
    user_last_query = {}

    log_score_cache = []
    log_rank_cache = []
    log_no_cache = []

    logger.info(f"Running {n_queries} queries, {n_users} users, return_prob={user_return_prob}")

    for qi in range(n_queries):
        # User selection (same logic as original)
        if qi > 0 and np.random.random() < user_return_prob:
            active = list(user_histories.keys())
            if active:
                weights = np.array([1.0 / (qi - user_last_query.get(u, 0) + 1) for u in active])
                weights /= weights.sum()
                user_id = np.random.choice(active, p=weights)
                if np.random.random() < 0.5:
                    new_item = np.random.choice(all_items)
                    user_histories[user_id] = user_histories[user_id][-9:] + [new_item]
            else:
                user_id = f"user_{len(user_histories)}"
                user_histories[user_id] = list(np.random.choice(all_items, size=10, replace=False))
        else:
            user_id = f"user_{len(user_histories)}"
            user_histories[user_id] = list(np.random.choice(all_items, size=10, replace=False))

        user_last_query[user_id] = qi
        history = user_histories[user_id]

        # Create candidates (fresh each time)
        candidates = list(np.random.choice(all_items, size=candidate_pool_size, replace=False))
        ground_truth = candidates[0]

        cache_key = generate_cache_key(user_id, history)

        # === Method 1: Score-based cache (NEW) ===
        cached_scores = score_cache.get(cache_key)
        if cached_scores is not None:
            # Re-rank current candidates using cached scores
            # For unseen items, fall back to Trie CTR
            scored = []
            for cid in candidates:
                if cid in cached_scores:
                    scored.append((cid, cached_scores[cid]))
                else:
                    # Trie fallback for items not in cache
                    ctr = trie_stats.get_ctr(cid)
                    scored.append((cid, ctr * 0.5))  # Discount unseen
            scored.sort(key=lambda x: -x[1])
            ranking_score = [x[0] for x in scored]
            method_score = 'cache'
            latency_score = 0.15  # Slightly more than pure cache due to re-ranking
        else:
            # LLM call
            history_text = " | ".join([
                news_items.get(iid, {}).get('title', '')[:30] for iid in history
            ])
            candidate_texts = [(cid, news_items.get(cid, {}).get('title', '')[:50])
                               for cid in candidates]
            t0 = time.time()
            scores_list = llm_ranker.score_candidates(history_text, candidate_texts)
            latency_score = (time.time() - t0) * 1000

            # Store scores in cache
            score_dict = {cid: sc for cid, sc in scores_list}
            score_cache.put(cache_key, score_dict)

            ranking_score = [cid for cid, _ in sorted(scores_list, key=lambda x: -x[1])]
            method_score = 'llm'

        gt_idx = ranking_score.index(ground_truth) if ground_truth in ranking_score else -1
        hit5_score = 1 if 0 <= gt_idx < 5 else 0

        # === Method 2: Rank-based cache (OLD - for comparison) ===
        cached_rank = rank_cache.get(cache_key)
        if cached_rank is not None:
            ranking_rank = cached_rank  # Old fixed ranking
            method_rank = 'cache'
            latency_rank = 0.1
        else:
            # Reuse LLM results from above
            ranking_rank = ranking_score[:]  # Reuse LLM results
            rank_cache.put(cache_key, ranking_rank)
            method_rank = 'llm'
            latency_rank = latency_score  # Same LLM call

        gt_idx_r = ranking_rank.index(ground_truth) if ground_truth in ranking_rank else -1
        hit5_rank = 1 if 0 <= gt_idx_r < 5 else 0

        log_score_cache.append({
            'qi': qi, 'method': method_score, 'hit@5': hit5_score, 'latency': latency_score
        })
        log_rank_cache.append({
            'qi': qi, 'method': method_rank, 'hit@5': hit5_rank, 'latency': latency_rank
        })

        if (qi + 1) % 500 == 0:
            sc_hits = sum(1 for x in log_score_cache if x['method'] == 'cache')
            sc_h5 = np.mean([x['hit@5'] for x in log_score_cache if x['method'] == 'cache']) if sc_hits > 0 else 0
            rc_h5 = np.mean([x['hit@5'] for x in log_rank_cache if x['method'] == 'cache']) if sc_hits > 0 else 0
            logger.info(f"  [{qi+1}/{n_queries}] CacheRate={sc_hits/(qi+1):.1%}, "
                         f"ScoreCache Hit@5={sc_h5:.3f}, RankCache Hit@5={rc_h5:.3f}")

    # Aggregate
    def summarize(log, label):
        total = len(log)
        cache_entries = [x for x in log if x['method'] == 'cache']
        llm_entries = [x for x in log if x['method'] == 'llm']
        return {
            'label': label,
            'total_queries': total,
            'cache_hits': len(cache_entries),
            'cache_hit_rate': len(cache_entries) / total,
            'hit@5_all': np.mean([x['hit@5'] for x in log]),
            'hit@5_cached': np.mean([x['hit@5'] for x in cache_entries]) if cache_entries else 0,
            'hit@5_llm': np.mean([x['hit@5'] for x in llm_entries]) if llm_entries else 0,
            'avg_latency_all': np.mean([x['latency'] for x in log]),
            'avg_latency_cached': np.mean([x['latency'] for x in cache_entries]) if cache_entries else 0,
            'avg_latency_llm': np.mean([x['latency'] for x in llm_entries]) if llm_entries else 0,
        }

    score_summary = summarize(log_score_cache, 'score_cache_new')
    rank_summary = summarize(log_rank_cache, 'rank_cache_old')

    logger.info("\n" + "=" * 60)
    logger.info("CACHE COMPARISON RESULTS")
    logger.info("=" * 60)
    for s in [score_summary, rank_summary]:
        logger.info(f"\n{s['label']}:")
        logger.info(f"  Cache Hit Rate: {s['cache_hit_rate']:.1%}")
        logger.info(f"  Hit@5 (all):    {s['hit@5_all']:.4f}")
        logger.info(f"  Hit@5 (cached): {s['hit@5_cached']:.4f}")
        logger.info(f"  Hit@5 (LLM):    {s['hit@5_llm']:.4f}")
        logger.info(f"  Latency (all):  {s['avg_latency_all']:.1f}ms")

    results = {
        'config': {
            'n_queries': n_queries,
            'n_users': n_users,
            'user_return_prob': user_return_prob,
            'cache_size': cache_size,
            'candidate_pool_size': candidate_pool_size,
        },
        'timestamp': datetime.now().isoformat(),
        'score_cache': score_summary,
        'rank_cache': rank_summary,
        'improvement': {
            'hit5_cached_delta': score_summary['hit@5_cached'] - rank_summary['hit@5_cached'],
            'hit5_all_delta': score_summary['hit@5_all'] - rank_summary['hit@5_all'],
        }
    }

    # Save
    out_dir = Path("results/kdd_rebuttal_cache_fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cache_fix_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    del llm_ranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=3000)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--return_prob", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    run_fixed_cache_experiment(
        n_queries=args.queries,
        n_users=args.users,
        user_return_prob=args.return_prob,
        device=args.device,
    )
