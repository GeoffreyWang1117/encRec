"""
Experiment: Trie-Augmented LLM Recommendation.

Compare:
1. Pure LLM recommendation (baseline)
2. Trie + LLM (our method)
3. Trie-only (statistical baseline)

Metrics:
- Recommendation quality (simulated)
- Latency
- Token usage
- Cache hit rate
- Early exit rate
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trie.retrieval_trie import RetrievalTrie, ItemStats
from src.llm.trie_augmented_llm_rec import (
    TrieAugmentedLLMRecommender,
    TrieContextCompressor,
    RecommendationRequest,
    StatisticalEarlyExit,
)
from src.llm.ollama_client import OllamaClient, OllamaConfig


def build_mock_trie(num_items: int = 10000) -> RetrievalTrie:
    """Build a mock Trie with simulated item data."""
    trie = RetrievalTrie()
    np.random.seed(42)

    categories = ['电子产品', '服装', '图书', '家居', '运动', '美妆', '食品', '玩具']

    for i in range(num_items):
        item_id = f'item_{i:05d}'
        # Simulate CTR with category effects
        category = np.random.choice(categories)
        base_ctr = {
            '电子产品': 0.08, '服装': 0.06, '图书': 0.04,
            '家居': 0.05, '运动': 0.07, '美妆': 0.09,
            '食品': 0.10, '玩具': 0.05,
        }[category]

        # Add item with random positive labels
        num_interactions = np.random.randint(10, 1000)
        for _ in range(num_interactions):
            p = np.clip(base_ctr + np.random.normal(0, 0.02), 0.01, 0.99)
            label = np.random.binomial(1, p)
            trie.insert(item_id, label, category)

    trie.build_indexes()
    return trie


def simulate_user_session(
    trie: RetrievalTrie,
    num_history: int = 10,
) -> List[str]:
    """Simulate a user's browsing history."""
    # Get some items and sample
    popular = trie.get_top_by_freq(100)
    history = np.random.choice([s.item_id for s in popular], size=num_history, replace=False)
    return list(history)


def run_pure_llm_recommendation(
    llm_client: OllamaClient,
    user_history: List[str],
    all_items: List[str],
    num_recs: int = 10,
) -> Dict:
    """Baseline: Pure LLM without Trie augmentation."""
    start_time = time.time()

    # Build verbose prompt (no compression)
    history_str = ", ".join(user_history)
    candidates_str = ", ".join(all_items[:50])  # Limited by context

    prompt = f"""你是一个推荐系统。

用户浏览历史: {history_str}

可选商品列表: {candidates_str}

请从以上商品中选择{num_recs}个最适合该用户的推荐，每行输出一个商品ID。
考虑用户的历史偏好，选择相关但有多样性的商品。
"""

    response = llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        model="glm-4.6",
        temperature=0.3,
        max_tokens=512,
    )

    latency = (time.time() - start_time) * 1000
    tokens = len(prompt.split()) * 1.5  # Rough estimate

    return {
        'method': 'pure_llm',
        'latency_ms': latency,
        'tokens_used': tokens,
        'response': response.get('message', {}).get('content', ''),
    }


def run_trie_augmented_recommendation(
    recommender: TrieAugmentedLLMRecommender,
    user_history: List[str],
    num_recs: int = 10,
) -> Dict:
    """Our method: Trie-augmented LLM."""
    request = RecommendationRequest(
        user_id="test_user",
        user_history=user_history,
        num_recommendations=num_recs,
    )

    result = recommender.recommend(request)

    return {
        'method': 'trie_augmented',
        'latency_ms': result.latency_ms,
        'tokens_used': result.tokens_used,
        'items': result.items,
        'early_exit': result.early_exit,
        'cache_hit': result.cache_hit,
    }


def run_trie_only_recommendation(
    trie: RetrievalTrie,
    user_history: List[str],
    num_recs: int = 10,
) -> Dict:
    """Statistical baseline: Trie-only without LLM."""
    from src.trie.retrieval_trie import TrieCandidateRetrieval

    start_time = time.time()

    retriever = TrieCandidateRetrieval(trie)
    candidates = retriever.retrieve(user_history, k=num_recs)

    latency = (time.time() - start_time) * 1000

    return {
        'method': 'trie_only',
        'latency_ms': latency,
        'tokens_used': 0,
        'items': candidates,
    }


def measure_compression_ratio(
    trie: RetrievalTrie,
    user_history: List[str],
    candidates: List[str],
) -> Dict:
    """Measure context compression effectiveness."""
    compressor = TrieContextCompressor(trie)

    # Original context (verbose)
    original_history = ", ".join(user_history)
    original_candidates = ", ".join(candidates)
    original_tokens = len(original_history.split()) + len(original_candidates.split())

    # Compressed context
    compressed_history = compressor.compress_history(user_history)
    compressed_candidates = compressor.compress_candidates(candidates)
    compressed_tokens = len(compressed_history.split()) + len(compressed_candidates.split())

    return {
        'original_tokens': original_tokens,
        'compressed_tokens': compressed_tokens,
        'compression_ratio': compressed_tokens / max(original_tokens, 1),
        'tokens_saved': original_tokens - compressed_tokens,
    }


def run_experiment(
    api_key: str,
    num_users: int = 10,
    num_items: int = 10000,
    output_dir: str = "results/trie_llm_experiment",
):
    """Run the full experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Trie-Augmented LLM Recommendation Experiment")
    print("="*60)

    # Build Trie
    print("\n1. Building mock Trie with simulated data...")
    trie = build_mock_trie(num_items)
    print(f"   Items in Trie: {trie.total_items}")

    # Initialize LLM client
    print("\n2. Initializing LLM client...")
    config = OllamaConfig(api_key=api_key)
    llm_client = OllamaClient(config)

    # Initialize recommender
    print("\n3. Initializing Trie-Augmented Recommender...")
    recommender = TrieAugmentedLLMRecommender(
        trie=trie,
        llm_client=llm_client,
        model="glm-4.6",
        enable_cache=True,
        enable_early_exit=True,
    )

    # Run experiments
    print(f"\n4. Running experiments with {num_users} simulated users...")

    results = {
        'pure_llm': [],
        'trie_augmented': [],
        'trie_only': [],
        'compression': [],
    }

    all_items = [f'item_{i:05d}' for i in range(num_items)]

    for user_idx in range(num_users):
        print(f"\n   User {user_idx + 1}/{num_users}")

        # Simulate user history
        user_history = simulate_user_session(trie)

        # Measure compression
        candidates = list(np.random.choice(all_items, 50, replace=False))
        compression = measure_compression_ratio(trie, user_history, candidates)
        results['compression'].append(compression)
        print(f"   Compression ratio: {compression['compression_ratio']:.2%}")

        # Run Trie-only (fast baseline)
        trie_result = run_trie_only_recommendation(trie, user_history)
        results['trie_only'].append(trie_result)
        print(f"   Trie-only: {trie_result['latency_ms']:.2f}ms")

        # Run Trie-augmented
        trie_aug_result = run_trie_augmented_recommendation(recommender, user_history)
        results['trie_augmented'].append(trie_aug_result)
        print(f"   Trie+LLM: {trie_aug_result['latency_ms']:.2f}ms, "
              f"early_exit={trie_aug_result['early_exit']}, "
              f"cache_hit={trie_aug_result['cache_hit']}")

        # Run pure LLM (expensive, only every 3rd user)
        if user_idx % 3 == 0:
            try:
                llm_result = run_pure_llm_recommendation(
                    llm_client, user_history, all_items
                )
                results['pure_llm'].append(llm_result)
                print(f"   Pure LLM: {llm_result['latency_ms']:.2f}ms")
            except Exception as e:
                print(f"   Pure LLM failed: {e}")

    # Aggregate results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    # Latency comparison
    print("\nLatency (ms):")
    for method in ['trie_only', 'trie_augmented', 'pure_llm']:
        if results[method]:
            latencies = [r['latency_ms'] for r in results[method]]
            print(f"  {method}: mean={np.mean(latencies):.2f}, "
                  f"std={np.std(latencies):.2f}, "
                  f"p50={np.percentile(latencies, 50):.2f}, "
                  f"p99={np.percentile(latencies, 99):.2f}")

    # Token usage
    print("\nToken Usage:")
    for method in ['trie_augmented', 'pure_llm']:
        if results[method]:
            tokens = [r.get('tokens_used', 0) for r in results[method]]
            print(f"  {method}: mean={np.mean(tokens):.1f}")

    # Compression
    print("\nContext Compression:")
    compressions = results['compression']
    print(f"  Compression ratio: {np.mean([c['compression_ratio'] for c in compressions]):.2%}")
    print(f"  Tokens saved: {np.mean([c['tokens_saved'] for c in compressions]):.1f}")

    # Early exit and cache
    print("\nOptimization Metrics:")
    trie_aug = results['trie_augmented']
    early_exits = sum(1 for r in trie_aug if r.get('early_exit'))
    cache_hits = sum(1 for r in trie_aug if r.get('cache_hit'))
    print(f"  Early exit rate: {early_exits / len(trie_aug):.1%}")
    print(f"  Cache hit rate: {cache_hits / len(trie_aug):.1%}")

    # Save results
    summary = {
        'config': {
            'num_users': num_users,
            'num_items': num_items,
        },
        'latency': {
            method: {
                'mean': np.mean([r['latency_ms'] for r in results[method]]),
                'std': np.std([r['latency_ms'] for r in results[method]]),
            } for method in ['trie_only', 'trie_augmented'] if results[method]
        },
        'compression': {
            'mean_ratio': np.mean([c['compression_ratio'] for c in compressions]),
            'mean_tokens_saved': np.mean([c['tokens_saved'] for c in compressions]),
        },
        'optimization': {
            'early_exit_rate': early_exits / len(trie_aug),
            'cache_hit_rate': cache_hits / len(trie_aug),
        },
    }

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--num_items', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='results/trie_llm_experiment')
    args = parser.parse_args()

    run_experiment(
        api_key=args.api_key,
        num_users=args.num_users,
        num_items=args.num_items,
        output_dir=args.output_dir,
    )
