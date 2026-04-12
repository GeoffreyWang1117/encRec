"""
MIND News Recommendation with Trie-Augmented LLM.

The MIND (Microsoft News Dataset) is ideal for demonstrating Trie-LLM value because:
1. News have semantic titles that LLM can understand
2. Clear user click history for personalization
3. Industrial-scale: 50K+ news, 150K+ user sessions

This experiment validates:
- Context compression effectiveness
- Early exit for high-confidence cases
- Cache hit for similar user histories
- LLM quality improvement over pure statistics
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trie.retrieval_trie import RetrievalTrie
from src.llm.ollama_client import OllamaClient, OllamaConfig


@dataclass
class NewsArticle:
    """News article with semantic information."""
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str


@dataclass
class UserSession:
    """User session with history and impressions."""
    user_id: str
    history: List[str]  # News IDs clicked before
    impressions: List[Tuple[str, int]]  # (news_id, label) pairs
    ground_truth: List[str]  # Clicked news in impressions


class MINDDataLoader:
    """Load and process MIND dataset."""

    def __init__(self, data_dir: str = "data/mind/MINDsmall_train"):
        self.data_dir = Path(data_dir)
        self.news: Dict[str, NewsArticle] = {}
        self.sessions: List[UserSession] = []

    def load(self, max_sessions: int = None) -> Tuple[Dict[str, NewsArticle], List[UserSession]]:
        """Load news and behavior data."""
        # Load news
        logger.info("Loading news articles...")
        news_path = self.data_dir / "news.tsv"
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    news_id, category, subcategory, title, abstract = parts[:5]
                    self.news[news_id] = NewsArticle(
                        news_id=news_id,
                        category=category,
                        subcategory=subcategory,
                        title=title,
                        abstract=abstract[:200] if abstract else "",
                    )

        logger.info(f"Loaded {len(self.news)} news articles")

        # Load behaviors
        logger.info("Loading user behaviors...")
        behaviors_path = self.data_dir / "behaviors.tsv"
        count = 0
        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    _, user_id, _, history_str, impressions_str = parts[:5]

                    # Parse history
                    history = history_str.split() if history_str else []

                    # Parse impressions (news_id-label pairs)
                    impressions = []
                    ground_truth = []
                    for item in impressions_str.split():
                        if '-' in item:
                            news_id, label = item.rsplit('-', 1)
                            impressions.append((news_id, int(label)))
                            if label == '1':
                                ground_truth.append(news_id)

                    if history and impressions and ground_truth:
                        self.sessions.append(UserSession(
                            user_id=user_id,
                            history=history,
                            impressions=impressions,
                            ground_truth=ground_truth,
                        ))
                        count += 1

                        if max_sessions and count >= max_sessions:
                            break

        logger.info(f"Loaded {len(self.sessions)} valid user sessions")
        return self.news, self.sessions


class MINDTrieBuilder:
    """Build Trie from MIND news data."""

    def __init__(self, news: Dict[str, NewsArticle], sessions: List[UserSession]):
        self.news = news
        self.sessions = sessions

    def build(self) -> RetrievalTrie:
        """Build Trie with news statistics."""
        logger.info("Building Trie from MIND data...")
        trie = RetrievalTrie()

        # Count clicks and impressions for each news
        news_clicks = defaultdict(int)
        news_impressions = defaultdict(int)

        for session in self.sessions:
            for news_id, label in session.impressions:
                news_impressions[news_id] += 1
                if label == 1:
                    news_clicks[news_id] += 1

        # Insert into Trie
        for news_id, article in self.news.items():
            clicks = news_clicks.get(news_id, 0)
            impressions = news_impressions.get(news_id, 1)

            # Insert multiple times based on impressions
            for _ in range(impressions):
                label = 1 if np.random.random() < (clicks / impressions) else 0
                trie.insert(news_id, label, article.category)

        trie.build_indexes()
        logger.info(f"Trie built with {trie.total_items} news items")
        return trie


class NewsContextCompressor:
    """Compress news recommendation context."""

    def __init__(self, trie: RetrievalTrie, news: Dict[str, NewsArticle]):
        self.trie = trie
        self.news = news

    def compress_history(self, history: List[str], max_items: int = 5) -> str:
        """Create compressed user history."""
        compressed = []
        for news_id in history[-max_items:]:
            article = self.news.get(news_id)
            if article:
                # Short title + category
                short_title = article.title[:30] + "..." if len(article.title) > 30 else article.title
                compressed.append(f"[{article.category}]{short_title}")
        return " → ".join(compressed)

    def compress_candidates(self, candidates: List[str], max_items: int = 15) -> str:
        """Create compressed candidate list grouped by category."""
        by_cat = defaultdict(list)
        for news_id in candidates[:max_items]:
            article = self.news.get(news_id)
            stats = self.trie.get(news_id)
            if article:
                ctr = stats.ctr if stats else 0
                short_title = article.title[:25] + "..." if len(article.title) > 25 else article.title
                by_cat[article.category].append((news_id, short_title, ctr))

        lines = []
        for cat, items in by_cat.items():
            items.sort(key=lambda x: x[2], reverse=True)
            item_strs = [f"{nid}:{title}" for nid, title, _ in items[:3]]
            lines.append(f"[{cat}]: {', '.join(item_strs)}")

        return "\n".join(lines)

    def create_full_prompt(self, history: List[str], candidates: List[str], num_recs: int) -> str:
        """Create verbose full prompt (baseline)."""
        history_titles = []
        for nid in history[-10:]:
            article = self.news.get(nid)
            if article:
                history_titles.append(f"- {article.title}")

        candidate_titles = []
        for nid in candidates[:30]:
            article = self.news.get(nid)
            if article:
                candidate_titles.append(f"- {nid}: {article.title} ({article.category})")

        return f"""你是一个新闻推荐专家。

用户最近阅读的新闻：
{chr(10).join(history_titles)}

待推荐的候选新闻：
{chr(10).join(candidate_titles)}

请根据用户的阅读偏好，从候选新闻中选择{num_recs}个最可能感兴趣的新闻。
考虑类别偏好、话题相关性和新闻热度。

请输出{num_recs}个新闻ID，每行一个，最推荐的在前。
"""

    def create_compressed_prompt(self, history: List[str], candidates: List[str], num_recs: int) -> str:
        """Create compressed prompt."""
        compressed_history = self.compress_history(history)
        compressed_candidates = self.compress_candidates(candidates)

        return f"""新闻推荐
用户历史: {compressed_history}
候选:
{compressed_candidates}
选{num_recs}个，每行一个ID"""


class MINDEarlyExit:
    """Early exit based on category matching and CTR confidence."""

    def __init__(self, trie: RetrievalTrie, news: Dict[str, NewsArticle], threshold: float = 0.7):
        self.trie = trie
        self.news = news
        self.threshold = threshold

    def check(self, history: List[str], candidates: List[str], num_recs: int) -> Tuple[bool, List[str], float]:
        """Check if we can skip LLM."""
        # Get user's category preferences from history
        user_categories = defaultdict(int)
        for nid in history[-20:]:
            article = self.news.get(nid)
            if article:
                user_categories[article.category] += 1

        if not user_categories:
            return False, [], 0.0

        top_category = max(user_categories, key=user_categories.get)
        preference_strength = user_categories[top_category] / len(history[-20:])

        # Score candidates
        candidate_scores = []
        for nid in candidates:
            article = self.news.get(nid)
            stats = self.trie.get(nid)
            if article and stats:
                # Score = category match * CTR
                cat_match = 1.0 if article.category == top_category else 0.3
                score = cat_match * stats.ctr * (1 + np.log1p(stats.frequency))
                candidate_scores.append((nid, score))

        if len(candidate_scores) < num_recs:
            return False, [], 0.0

        # Sort by score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = candidate_scores[:num_recs]
        rest = candidate_scores[num_recs:num_recs + 10]

        if not rest:
            return False, [], 0.0

        # Calculate confidence
        avg_top = np.mean([s for _, s in top_k])
        avg_rest = np.mean([s for _, s in rest])

        confidence = preference_strength * (avg_top - avg_rest) / max(avg_top, 0.01)
        confidence = min(confidence, 1.0)

        if confidence >= self.threshold:
            return True, [nid for nid, _ in top_k], confidence

        return False, [], confidence


def calculate_metrics(recommended: List[str], ground_truth: List[str], k: int) -> Dict:
    """Calculate recommendation metrics."""
    if not ground_truth:
        return {'hit': 0, 'ndcg': 0, 'mrr': 0}

    # Hit@K
    hit = 1 if any(r in ground_truth for r in recommended[:k]) else 0

    # NDCG@K
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recommended[:k]) if r in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    ndcg = dcg / idcg if idcg > 0 else 0

    # MRR
    mrr = 0
    for i, r in enumerate(recommended[:k]):
        if r in ground_truth:
            mrr = 1.0 / (i + 1)
            break

    return {'hit': hit, 'ndcg': ndcg, 'mrr': mrr}


def run_mind_experiment(
    api_key: str,
    data_dir: str = "data/mind/MINDsmall_train",
    num_sessions: int = 200,
    num_recs: int = 5,
    early_exit_threshold: float = 0.5,
):
    """Run the MIND news recommendation experiment."""
    print("=" * 70)
    print("MIND NEWS RECOMMENDATION WITH TRIE-AUGMENTED LLM")
    print("=" * 70)

    # Load data
    loader = MINDDataLoader(data_dir)
    news, sessions = loader.load(max_sessions=num_sessions * 2)

    # Build Trie
    builder = MINDTrieBuilder(news, sessions)
    trie = builder.build()

    # Initialize components
    compressor = NewsContextCompressor(trie, news)
    early_exit = MINDEarlyExit(trie, news, threshold=early_exit_threshold)

    # Initialize LLM
    config = OllamaConfig(api_key=api_key)
    llm = OllamaClient(config)

    # Cache
    cache = {}

    results = {
        'trie_only': [],
        'trie_llm': [],
        'pure_llm': [],
    }

    compression_metrics = []
    test_sessions = sessions[:num_sessions]

    print(f"\nTesting on {len(test_sessions)} user sessions...")

    for idx, session in enumerate(test_sessions):
        if idx % 50 == 0:
            print(f"\n--- Session {idx + 1}/{len(test_sessions)} ---")

        history = session.history
        candidates = [nid for nid, _ in session.impressions]
        ground_truth = session.ground_truth

        # Measure compression
        full_prompt = compressor.create_full_prompt(history, candidates, num_recs)
        compressed_prompt = compressor.create_compressed_prompt(history, candidates, num_recs)
        compression_metrics.append({
            'full_tokens': len(full_prompt.split()),
            'compressed_tokens': len(compressed_prompt.split()),
            'ratio': len(compressed_prompt.split()) / len(full_prompt.split()),
        })

        # METHOD 1: Trie-only (statistical ranking)
        start = time.time()
        from src.trie.retrieval_trie import TrieCandidateRetrieval
        retriever = TrieCandidateRetrieval(trie)
        trie_recs = retriever.retrieve(history, k=num_recs)
        trie_latency = (time.time() - start) * 1000
        trie_metrics = calculate_metrics(trie_recs, ground_truth, num_recs)

        results['trie_only'].append({
            'latency_ms': trie_latency,
            'tokens': 0,
            **trie_metrics,
        })

        # METHOD 2: Trie+LLM with optimizations
        start = time.time()

        # Check cache
        cache_key = hashlib.md5(",".join(sorted(history[-5:])).encode()).hexdigest()
        if cache_key in cache:
            trie_llm_recs = cache[cache_key]
            trie_llm_latency = (time.time() - start) * 1000
            trie_llm_metrics = calculate_metrics(trie_llm_recs, ground_truth, num_recs)

            results['trie_llm'].append({
                'latency_ms': trie_llm_latency,
                'tokens': 0,
                'cache_hit': True,
                'early_exit': False,
                **trie_llm_metrics,
            })
            continue

        # Check early exit
        should_exit, early_recs, confidence = early_exit.check(history, candidates, num_recs)

        if should_exit:
            trie_llm_recs = early_recs
            cache[cache_key] = trie_llm_recs
            trie_llm_latency = (time.time() - start) * 1000
            trie_llm_metrics = calculate_metrics(trie_llm_recs, ground_truth, num_recs)

            results['trie_llm'].append({
                'latency_ms': trie_llm_latency,
                'tokens': 0,
                'cache_hit': False,
                'early_exit': True,
                **trie_llm_metrics,
            })

            if idx % 50 == 0:
                print(f"  Early exit (conf={confidence:.2f}): {trie_llm_latency:.2f}ms, "
                      f"Hit@{num_recs}={trie_llm_metrics['hit']}")
            continue

        # Call LLM with compressed prompt
        try:
            response = llm.chat(
                messages=[{"role": "user", "content": compressed_prompt}],
                model="glm-4.6",
                temperature=0.3,
                max_tokens=200,
            )
            content = response.get('message', {}).get('content', '')

            # Parse recommendations
            trie_llm_recs = []
            for nid in candidates:
                if nid in content:
                    trie_llm_recs.append(nid)
                    if len(trie_llm_recs) >= num_recs:
                        break

            # Fallback
            while len(trie_llm_recs) < num_recs and trie_recs:
                for r in trie_recs:
                    if r not in trie_llm_recs:
                        trie_llm_recs.append(r)
                        break

            tokens = len(compressed_prompt.split()) + len(content.split())

        except Exception as e:
            logger.error(f"LLM error: {e}")
            trie_llm_recs = trie_recs
            tokens = 0

        cache[cache_key] = trie_llm_recs
        trie_llm_latency = (time.time() - start) * 1000
        trie_llm_metrics = calculate_metrics(trie_llm_recs, ground_truth, num_recs)

        results['trie_llm'].append({
            'latency_ms': trie_llm_latency,
            'tokens': tokens,
            'cache_hit': False,
            'early_exit': False,
            **trie_llm_metrics,
        })

        if idx % 50 == 0:
            print(f"  LLM called: {trie_llm_latency:.1f}ms, {tokens} tokens, "
                  f"Hit@{num_recs}={trie_llm_metrics['hit']}")

        # METHOD 3: Pure LLM (every 20th session)
        if idx % 20 == 0:
            start = time.time()
            try:
                response = llm.chat(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="glm-4.6",
                    temperature=0.3,
                    max_tokens=200,
                )
                full_content = response.get('message', {}).get('content', '')
                full_tokens = len(full_prompt.split()) + len(full_content.split())
            except:
                full_content = ""
                full_tokens = len(full_prompt.split())

            # Parse
            pure_llm_recs = []
            for nid in candidates:
                if nid in full_content:
                    pure_llm_recs.append(nid)
                    if len(pure_llm_recs) >= num_recs:
                        break

            pure_llm_latency = (time.time() - start) * 1000
            pure_llm_metrics = calculate_metrics(pure_llm_recs, ground_truth, num_recs)

            results['pure_llm'].append({
                'latency_ms': pure_llm_latency,
                'tokens': full_tokens,
                **pure_llm_metrics,
            })

            if idx % 50 == 0:
                print(f"  Pure LLM: {pure_llm_latency:.1f}ms, {full_tokens} tokens, "
                      f"Hit@{num_recs}={pure_llm_metrics['hit']}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Quality metrics
    print("\n1. RECOMMENDATION QUALITY")
    print("-" * 50)
    print(f"{'Method':<15} {'Hit@5':<10} {'NDCG@5':<10} {'MRR':<10}")
    print("-" * 50)

    for method, method_results in results.items():
        if not method_results:
            continue
        hit = np.mean([r['hit'] for r in method_results])
        ndcg = np.mean([r['ndcg'] for r in method_results])
        mrr = np.mean([r['mrr'] for r in method_results])
        print(f"{method:<15} {hit:<10.4f} {ndcg:<10.4f} {mrr:<10.4f}")

    # Latency metrics
    print("\n2. LATENCY ANALYSIS")
    print("-" * 50)
    print(f"{'Method':<15} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 50)

    for method, method_results in results.items():
        if not method_results:
            continue
        latencies = [r['latency_ms'] for r in method_results]
        print(f"{method:<15} {np.mean(latencies):<12.2f} "
              f"{np.percentile(latencies, 50):<12.2f} "
              f"{np.percentile(latencies, 99):<12.2f}")

    # Speedup
    if results['pure_llm'] and results['trie_llm']:
        pure_avg = np.mean([r['latency_ms'] for r in results['pure_llm']])
        trie_llm_avg = np.mean([r['latency_ms'] for r in results['trie_llm']])
        print(f"\n  Speedup (Trie+LLM vs Pure LLM): {pure_avg / trie_llm_avg:.1f}x")

    # Compression
    print("\n3. TOKEN COMPRESSION")
    print("-" * 50)
    avg_ratio = np.mean([c['ratio'] for c in compression_metrics])
    avg_full = np.mean([c['full_tokens'] for c in compression_metrics])
    avg_compressed = np.mean([c['compressed_tokens'] for c in compression_metrics])

    print(f"  Full prompt tokens: {avg_full:.1f}")
    print(f"  Compressed tokens: {avg_compressed:.1f}")
    print(f"  Compression ratio: {avg_ratio:.1%}")
    print(f"  Token reduction: {(1 - avg_ratio) * 100:.1f}%")

    # Optimization rates
    print("\n4. OPTIMIZATION RATES")
    print("-" * 50)
    trie_llm = results['trie_llm']
    early_exits = sum(1 for r in trie_llm if r.get('early_exit'))
    cache_hits = sum(1 for r in trie_llm if r.get('cache_hit'))
    llm_calls = len(trie_llm) - early_exits - cache_hits

    print(f"  Early exit rate: {early_exits / len(trie_llm):.1%} ({early_exits}/{len(trie_llm)})")
    print(f"  Cache hit rate: {cache_hits / len(trie_llm):.1%} ({cache_hits}/{len(trie_llm)})")
    print(f"  LLM call rate: {llm_calls / len(trie_llm):.1%} ({llm_calls}/{len(trie_llm)})")

    # Cost analysis
    print("\n5. COST ANALYSIS")
    print("-" * 50)
    trie_llm_tokens = sum(r['tokens'] for r in trie_llm)
    pure_llm_tokens = sum(r['tokens'] for r in results['pure_llm'])

    # Scale to same requests
    if results['pure_llm']:
        scale = len(trie_llm) / len(results['pure_llm'])
        pure_scaled = pure_llm_tokens * scale
        reduction = (pure_scaled - trie_llm_tokens) / pure_scaled * 100

        print(f"  Trie+LLM total tokens: {trie_llm_tokens}")
        print(f"  Pure LLM total tokens (scaled): {pure_scaled:.0f}")
        print(f"  Token cost reduction: {reduction:.1f}%")

    # Save results
    output_dir = Path("results/mind_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'quality': {
            method: {
                'hit': np.mean([r['hit'] for r in res]),
                'ndcg': np.mean([r['ndcg'] for r in res]),
                'mrr': np.mean([r['mrr'] for r in res]),
            } for method, res in results.items() if res
        },
        'latency': {
            method: {
                'mean': np.mean([r['latency_ms'] for r in res]),
                'p50': np.percentile([r['latency_ms'] for r in res], 50),
                'p99': np.percentile([r['latency_ms'] for r in res], 99),
            } for method, res in results.items() if res
        },
        'compression': {
            'ratio': avg_ratio,
            'reduction_pct': (1 - avg_ratio) * 100,
        },
        'optimization': {
            'early_exit_rate': early_exits / len(trie_llm),
            'cache_hit_rate': cache_hits / len(trie_llm),
            'llm_call_rate': llm_calls / len(trie_llm),
        },
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/mind/MINDsmall_train')
    parser.add_argument('--num_sessions', type=int, default=200)
    parser.add_argument('--num_recs', type=int, default=5)
    parser.add_argument('--early_exit_threshold', type=float, default=0.5)
    args = parser.parse_args()

    run_mind_experiment(
        api_key=args.api_key,
        data_dir=args.data_dir,
        num_sessions=args.num_sessions,
        num_recs=args.num_recs,
        early_exit_threshold=args.early_exit_threshold,
    )
