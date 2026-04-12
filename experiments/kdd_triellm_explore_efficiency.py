"""
Trie-Augmented LLM Efficiency Demonstration.

This experiment focuses on demonstrating the key efficiency benefits:
1. Early Exit - Skip LLM for high-confidence cases
2. Caching - Reuse results for similar queries
3. Compression - Reduce token usage

For KDD-level publication with industrial-scale data.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trie.retrieval_trie import RetrievalTrie
from src.llm.ollama_client import OllamaClient, OllamaConfig


@dataclass
class EfficiencyResult:
    """Result for a single request."""
    request_id: int
    method: str
    latency_ms: float
    tokens_used: int
    early_exit: bool = False
    cache_hit: bool = False
    llm_called: bool = False


class IndustrialTrieBuilder:
    """Build Trie from industrial-scale Criteo data."""

    def __init__(self, data_path: str, sample_size: int = None):
        self.data_path = data_path
        self.sample_size = sample_size

    def build(self) -> RetrievalTrie:
        """Build Trie with realistic industrial statistics."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)

        if self.sample_size:
            df = df.sample(n=min(self.sample_size, len(df)), random_state=42)

        logger.info(f"Building Trie from {len(df)} samples...")
        trie = RetrievalTrie()

        # Use first 5 categorical features
        sparse_cols = [f'C{i}' for i in range(1, 6)]

        for idx, row in df.iterrows():
            if idx % 100000 == 0:
                logger.info(f"  Processing {idx}/{len(df)}")

            label = int(row['label'])
            for col in sparse_cols:
                token = str(row.get(col, 'UNK'))
                if token and token != 'nan':
                    category = f"cat_{col}"
                    trie.insert(token, label, category)

        trie.build_indexes()
        logger.info(f"Trie built: {trie.total_items} unique tokens")
        return trie


class StatisticalEarlyExitDemo:
    """Demonstrate early exit mechanism."""

    def __init__(self, trie: RetrievalTrie, threshold: float = 0.7):
        self.trie = trie
        self.threshold = threshold

    def check_early_exit(self, candidates: List[str], num_recs: int) -> Tuple[bool, List[str], float]:
        """Check if we can skip LLM based on statistics."""
        if not candidates:
            return False, [], 0.0

        # Get CTR stats for candidates
        stats_list = []
        for item_id in candidates[:num_recs * 3]:
            stats = self.trie.get(item_id)
            if stats:
                stats_list.append((item_id, stats.ctr, stats.frequency))

        if len(stats_list) < num_recs:
            return False, [], 0.0

        # Sort by CTR
        stats_list.sort(key=lambda x: x[1], reverse=True)

        # Calculate confidence based on CTR gap
        top_k_ctrs = [s[1] for s in stats_list[:num_recs]]
        rest_ctrs = [s[1] for s in stats_list[num_recs:]]

        if not rest_ctrs:
            return False, [], 0.0

        avg_top = np.mean(top_k_ctrs)
        avg_rest = np.mean(rest_ctrs)

        # Confidence = how much better top-k is than rest
        confidence = (avg_top - avg_rest) / max(avg_top, 0.01)

        if confidence >= self.threshold:
            return True, [s[0] for s in stats_list[:num_recs]], confidence

        return False, [], confidence


class ContextCompressorDemo:
    """Demonstrate context compression."""

    def __init__(self, trie: RetrievalTrie):
        self.trie = trie

    def create_full_prompt(self, history: List[str], candidates: List[str], num_recs: int) -> str:
        """Create verbose full prompt (baseline)."""
        history_str = ", ".join(history)
        candidates_str = ", ".join(candidates)

        return f"""你是一个推荐系统专家。

用户的完整浏览历史如下：
{history_str}

可供选择的推荐候选商品列表：
{candidates_str}

请根据用户的历史偏好，从候选商品中选择{num_recs}个最合适的推荐。

要求：
1. 考虑用户的历史行为模式
2. 选择与用户兴趣相关的商品
3. 保持推荐的多样性
4. 优先选择热门且相关的商品

请按推荐优先级从高到低输出{num_recs}个商品ID，每行一个。
"""

    def create_compressed_prompt(self, history: List[str], candidates: List[str], num_recs: int) -> str:
        """Create compressed prompt using Trie stats."""
        # Compress history to top-5 informative items with stats
        history_stats = []
        for item in history[:10]:
            stats = self.trie.get(item)
            if stats:
                history_stats.append(f"{item[:8]}(★{stats.ctr:.0%})")

        # Compress candidates by category
        by_cat = defaultdict(list)
        for item in candidates[:20]:
            stats = self.trie.get(item)
            cat = stats.category if stats else "unk"
            ctr = stats.ctr if stats else 0
            by_cat[cat].append((item[:8], ctr))

        cat_summary = []
        for cat, items in by_cat.items():
            items.sort(key=lambda x: x[1], reverse=True)
            top3 = [f"{i[0]}★{i[1]:.0%}" for i in items[:3]]
            cat_summary.append(f"[{cat}]:{','.join(top3)}")

        return f"""推荐任务
历史: {' '.join(history_stats[:5])}
候选:
{chr(10).join(cat_summary)}
选{num_recs}个，每行一个ID"""

    def measure_compression(self, history: List[str], candidates: List[str], num_recs: int) -> Dict:
        """Measure compression ratio."""
        full = self.create_full_prompt(history, candidates, num_recs)
        compressed = self.create_compressed_prompt(history, candidates, num_recs)

        full_tokens = len(full.split())
        compressed_tokens = len(compressed.split())

        return {
            'full_tokens': full_tokens,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': compressed_tokens / full_tokens,
            'tokens_saved': full_tokens - compressed_tokens,
            'reduction_pct': (1 - compressed_tokens / full_tokens) * 100,
        }


class ResultCacheDemo:
    """Demonstrate caching mechanism."""

    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def _make_key(self, history: List[str]) -> str:
        """Create cache key from history."""
        import hashlib
        recent = sorted(history[-5:])
        return hashlib.md5(",".join(recent).encode()).hexdigest()

    def get(self, history: List[str]) -> Tuple[bool, List[str]]:
        """Check cache."""
        key = self._make_key(history)
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key]
        self.misses += 1
        return False, []

    def put(self, history: List[str], result: List[str]):
        """Store result."""
        key = self._make_key(history)
        self.cache[key] = result

    def get_stats(self) -> Dict:
        """Get cache stats."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1) * 100,
            'cache_size': len(self.cache),
        }


def run_efficiency_experiment(
    api_key: str,
    data_path: str = "data/criteo/criteo_5m.parquet",
    sample_size: int = 500000,
    num_requests: int = 100,
    num_recs: int = 10,
    early_exit_threshold: float = 0.5,
):
    """Run the efficiency demonstration experiment."""
    print("=" * 70)
    print("TRIE-AUGMENTED LLM EFFICIENCY DEMONSTRATION")
    print("=" * 70)

    # Build Trie
    builder = IndustrialTrieBuilder(data_path, sample_size)
    trie = builder.build()

    # Initialize components
    early_exit = StatisticalEarlyExitDemo(trie, threshold=early_exit_threshold)
    compressor = ContextCompressorDemo(trie)
    cache = ResultCacheDemo()

    # Initialize LLM
    config = OllamaConfig(api_key=api_key)
    llm = OllamaClient(config)

    # Get sample items for testing
    all_items = [s.item_id for s in trie.get_top_by_freq(1000)]

    results = {
        'pure_llm': [],
        'trie_llm': [],
        'trie_only': [],
    }

    compression_metrics = []

    print(f"\nRunning {num_requests} requests...")

    for req_id in range(num_requests):
        if req_id % 20 == 0:
            print(f"\n--- Request {req_id + 1}/{num_requests} ---")

        # Simulate user history and candidates
        np.random.seed(42 + req_id)
        history = list(np.random.choice(all_items, size=10, replace=False))
        candidates = list(np.random.choice(all_items, size=50, replace=False))

        # Measure compression
        comp = compressor.measure_compression(history, candidates, num_recs)
        compression_metrics.append(comp)

        # METHOD 1: Trie-only (pure statistics)
        start = time.time()
        from src.trie.retrieval_trie import TrieCandidateRetrieval
        retriever = TrieCandidateRetrieval(trie)
        trie_recs = retriever.retrieve(history, k=num_recs)
        trie_latency = (time.time() - start) * 1000

        results['trie_only'].append(EfficiencyResult(
            request_id=req_id,
            method='trie_only',
            latency_ms=trie_latency,
            tokens_used=0,
            early_exit=True,  # Always "early exit" since no LLM
            llm_called=False,
        ))

        # METHOD 2: Trie+LLM with optimizations
        start = time.time()

        # Check cache first
        cache_hit, cached_result = cache.get(history)
        if cache_hit:
            trie_llm_latency = (time.time() - start) * 1000
            results['trie_llm'].append(EfficiencyResult(
                request_id=req_id,
                method='trie_llm',
                latency_ms=trie_llm_latency,
                tokens_used=0,
                cache_hit=True,
                llm_called=False,
            ))
            continue

        # Check early exit
        should_exit, early_recs, confidence = early_exit.check_early_exit(candidates, num_recs)

        if should_exit:
            cache.put(history, early_recs)
            trie_llm_latency = (time.time() - start) * 1000
            results['trie_llm'].append(EfficiencyResult(
                request_id=req_id,
                method='trie_llm',
                latency_ms=trie_llm_latency,
                tokens_used=0,
                early_exit=True,
                llm_called=False,
            ))
            if req_id % 20 == 0:
                print(f"  Early exit (confidence={confidence:.2f}): {trie_llm_latency:.2f}ms")
            continue

        # Need to call LLM - use compressed prompt
        compressed_prompt = compressor.create_compressed_prompt(history, candidates, num_recs)

        try:
            response = llm.chat(
                messages=[{"role": "user", "content": compressed_prompt}],
                model="glm-4.6",
                temperature=0.3,
                max_tokens=256,
            )
            content = response.get('message', {}).get('content', '')
            llm_tokens = comp['compressed_tokens'] + len(content.split())
            cache.put(history, trie_recs)  # Cache result

        except Exception as e:
            logger.error(f"LLM error: {e}")
            llm_tokens = 0

        trie_llm_latency = (time.time() - start) * 1000
        results['trie_llm'].append(EfficiencyResult(
            request_id=req_id,
            method='trie_llm',
            latency_ms=trie_llm_latency,
            tokens_used=llm_tokens,
            llm_called=True,
        ))

        if req_id % 20 == 0:
            print(f"  LLM called: {trie_llm_latency:.1f}ms, {llm_tokens} tokens")

        # METHOD 3: Pure LLM (every 10th request, expensive)
        if req_id % 10 == 0:
            start = time.time()
            full_prompt = compressor.create_full_prompt(history, candidates, num_recs)

            try:
                response = llm.chat(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="glm-4.6",
                    temperature=0.3,
                    max_tokens=256,
                )
                content = response.get('message', {}).get('content', '')
                full_tokens = comp['full_tokens'] + len(content.split())
            except:
                full_tokens = comp['full_tokens']

            pure_llm_latency = (time.time() - start) * 1000
            results['pure_llm'].append(EfficiencyResult(
                request_id=req_id,
                method='pure_llm',
                latency_ms=pure_llm_latency,
                tokens_used=full_tokens,
                llm_called=True,
            ))

            if req_id % 20 == 0:
                print(f"  Pure LLM: {pure_llm_latency:.1f}ms, {full_tokens} tokens")

    # Print summary
    print("\n" + "=" * 70)
    print("EFFICIENCY EXPERIMENT SUMMARY")
    print("=" * 70)

    # Latency analysis
    print("\n1. LATENCY ANALYSIS")
    print("-" * 50)

    for method, method_results in results.items():
        if not method_results:
            continue
        latencies = [r.latency_ms for r in method_results]
        print(f"\n{method}:")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  P50 latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Min latency: {np.min(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")

    # Speedup calculation
    if results['pure_llm'] and results['trie_llm']:
        pure_llm_avg = np.mean([r.latency_ms for r in results['pure_llm']])
        trie_llm_avg = np.mean([r.latency_ms for r in results['trie_llm']])
        speedup = pure_llm_avg / trie_llm_avg if trie_llm_avg > 0 else float('inf')
        print(f"\n  Speedup (Trie+LLM vs Pure LLM): {speedup:.1f}x")

    # Token analysis
    print("\n2. TOKEN ANALYSIS")
    print("-" * 50)

    avg_compression = np.mean([c['compression_ratio'] for c in compression_metrics])
    avg_tokens_saved = np.mean([c['tokens_saved'] for c in compression_metrics])
    avg_reduction = np.mean([c['reduction_pct'] for c in compression_metrics])

    print(f"  Compression ratio: {avg_compression:.2%}")
    print(f"  Avg tokens saved: {avg_tokens_saved:.1f}")
    print(f"  Token reduction: {avg_reduction:.1f}%")

    for method, method_results in results.items():
        if not method_results:
            continue
        tokens = [r.tokens_used for r in method_results]
        print(f"\n  {method} avg tokens: {np.mean(tokens):.1f}")

    # Early exit and cache analysis
    print("\n3. OPTIMIZATION METRICS")
    print("-" * 50)

    trie_llm_results = results['trie_llm']
    early_exits = sum(1 for r in trie_llm_results if r.early_exit)
    cache_hits = sum(1 for r in trie_llm_results if r.cache_hit)
    llm_calls = sum(1 for r in trie_llm_results if r.llm_called)

    total = len(trie_llm_results)
    print(f"  Early exit rate: {early_exits / total:.1%} ({early_exits}/{total})")
    print(f"  Cache hit rate: {cache_hits / total:.1%} ({cache_hits}/{total})")
    print(f"  LLM call rate: {llm_calls / total:.1%} ({llm_calls}/{total})")

    cache_stats = cache.get_stats()
    print(f"\n  Cache stats: {cache_stats}")

    # Cost analysis
    print("\n4. COST ANALYSIS")
    print("-" * 50)

    # Estimate costs (rough: $0.01 per 1K tokens)
    pure_llm_tokens = sum(r.tokens_used for r in results['pure_llm'])
    trie_llm_tokens = sum(r.tokens_used for r in results['trie_llm'])

    # Scale to same number of requests
    if results['pure_llm']:
        scale = len(trie_llm_results) / len(results['pure_llm'])
        pure_llm_tokens_scaled = pure_llm_tokens * scale

        print(f"  Pure LLM (scaled): {pure_llm_tokens_scaled:.0f} tokens")
        print(f"  Trie+LLM: {trie_llm_tokens:.0f} tokens")

        cost_reduction = (pure_llm_tokens_scaled - trie_llm_tokens) / pure_llm_tokens_scaled * 100
        print(f"  Cost reduction: {cost_reduction:.1f}%")

    # Save results
    output_dir = Path("results/efficiency_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'config': {
            'sample_size': sample_size,
            'num_requests': num_requests,
            'early_exit_threshold': early_exit_threshold,
        },
        'latency': {
            method: {
                'mean': np.mean([r.latency_ms for r in res]),
                'p50': np.percentile([r.latency_ms for r in res], 50),
                'p99': np.percentile([r.latency_ms for r in res], 99),
            } for method, res in results.items() if res
        },
        'compression': {
            'ratio': avg_compression,
            'tokens_saved': avg_tokens_saved,
            'reduction_pct': avg_reduction,
        },
        'optimization': {
            'early_exit_rate': early_exits / total,
            'cache_hit_rate': cache_hits / total,
            'llm_call_rate': llm_calls / total,
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
    parser.add_argument('--data_path', type=str, default='data/criteo/criteo_5m.parquet')
    parser.add_argument('--sample_size', type=int, default=500000)
    parser.add_argument('--num_requests', type=int, default=100)
    parser.add_argument('--early_exit_threshold', type=float, default=0.5)
    args = parser.parse_args()

    run_efficiency_experiment(
        api_key=args.api_key,
        data_path=args.data_path,
        sample_size=args.sample_size,
        num_requests=args.num_requests,
        early_exit_threshold=args.early_exit_threshold,
    )
