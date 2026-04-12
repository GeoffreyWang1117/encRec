"""
Radix Trie Optimization Experiments for KDD 2026.

This script validates the Radix Trie implementation:
1. Small-scale correctness test
2. Memory efficiency comparison
3. Query performance comparison
4. Integration with recommendation pipeline
5. Large-scale industrial validation

Usage:
    python experiments/radix_trie_experiments.py --phase small
    python experiments/radix_trie_experiments.py --phase large
"""

import os
import sys
import json
import time
import random
import string
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trie.retrieval_trie import RetrievalTrie
from src.trie.radix_trie import RadixTrie, compare_tries


def generate_synthetic_items(
    n_items: int,
    id_length: int = 10,
    categories: List[str] = None,
    prefix_similarity: float = 0.5,
) -> List[Tuple[str, int, str]]:
    """
    Generate synthetic item data for testing.

    Args:
        n_items: Number of items
        id_length: Length of item IDs
        categories: List of categories
        prefix_similarity: Probability of sharing prefix with previous item

    Returns:
        List of (item_id, label, category) tuples
    """
    if categories is None:
        categories = ["news", "sports", "tech", "politics", "entertainment",
                      "business", "health", "science", "travel", "lifestyle"]

    items = []
    prefixes = []

    for i in range(n_items):
        # Determine prefix
        if prefixes and random.random() < prefix_similarity:
            # Reuse existing prefix
            prefix = random.choice(prefixes)
            suffix_len = id_length - len(prefix)
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=max(suffix_len, 1)))
            item_id = prefix + suffix
        else:
            # Generate new ID
            item_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=id_length))
            # Store prefix for reuse
            prefix_len = random.randint(3, min(6, id_length))
            prefixes.append(item_id[:prefix_len])
            if len(prefixes) > 100:
                prefixes = prefixes[-100:]

        # Generate label and category
        label = 1 if random.random() < 0.3 else 0  # 30% CTR
        category = random.choice(categories)

        items.append((item_id[:id_length], label, category))

    return items


def load_mind_items(data_path: str, max_items: int = None) -> List[Tuple[str, int, str]]:
    """Load items from MIND dataset."""
    news_path = Path(data_path) / "news.tsv"

    items = []
    with open(news_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else "unknown"
                # Simulate CTR
                label = 1 if random.random() < 0.25 else 0
                items.append((news_id, label, category))

    return items


def run_small_scale_experiments() -> Dict[str, Any]:
    """
    Phase 1: Small-scale experiments for correctness and feasibility.
    """
    print("\n" + "="*70)
    print("PHASE 1: Small-Scale Experiments")
    print("="*70)

    results = {
        'phase': 'small',
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # Experiment 1.1: Basic correctness
    print("\n--- Experiment 1.1: Basic Correctness ---")
    test_items = [
        ("news_001", 1, "politics"),
        ("news_002", 0, "politics"),
        ("news_003", 1, "sports"),
        ("news_010", 1, "sports"),
        ("news_011", 0, "tech"),
        ("news_100", 1, "tech"),
        ("movie_001", 1, "action"),
        ("movie_002", 1, "comedy"),
        ("movie_010", 0, "action"),
        ("tech_article_001", 1, "tech"),
        ("tech_article_002", 0, "tech"),
    ]

    std_trie = RetrievalTrie()
    radix_trie = RadixTrie()

    for item_id, label, category in test_items:
        std_trie.insert(item_id, label, category)
        radix_trie.insert(item_id, label, category)

    # Verify all lookups
    errors = []
    for item_id, _, _ in test_items:
        std_result = std_trie.get(item_id)
        radix_result = radix_trie.get(item_id)

        if std_result is None or radix_result is None:
            errors.append(f"Lookup failed for {item_id}")
        elif std_result.frequency != radix_result.frequency:
            errors.append(f"Frequency mismatch for {item_id}")
        elif abs(std_result.ctr - radix_result.ctr) > 0.001:
            errors.append(f"CTR mismatch for {item_id}")

    # Verify prefix search
    std_prefix = std_trie.prefix_search("news_", max_results=100)
    radix_prefix = radix_trie.prefix_search("news_", max_results=100)

    if len(std_prefix) != len(radix_prefix):
        errors.append(f"Prefix search count mismatch: {len(std_prefix)} vs {len(radix_prefix)}")

    exp1_result = {
        'name': 'basic_correctness',
        'items': len(test_items),
        'errors': len(errors),
        'error_details': errors[:5],  # First 5 errors
        'passed': len(errors) == 0
    }
    results['experiments'].append(exp1_result)
    print(f"  Items: {len(test_items)}, Errors: {len(errors)}, Passed: {exp1_result['passed']}")

    # Experiment 1.2: Scaling test (1K, 10K, 100K items)
    print("\n--- Experiment 1.2: Scaling Test ---")

    for n_items in [1000, 10000, 100000]:
        print(f"\n  Testing with {n_items:,} items...")

        items = generate_synthetic_items(n_items, id_length=12, prefix_similarity=0.6)

        # Build both tries
        std_trie = RetrievalTrie()
        radix_trie = RadixTrie()

        start = time.time()
        for item_id, label, category in items:
            std_trie.insert(item_id, label, category)
        std_build_time = time.time() - start

        start = time.time()
        for item_id, label, category in items:
            radix_trie.insert(item_id, label, category)
        radix_build_time = time.time() - start

        # Memory stats
        radix_mem = radix_trie.memory_stats()

        # Query performance (sample 1000 queries)
        sample_items = random.sample([item_id for item_id, _, _ in items], min(1000, len(items)))

        start = time.time()
        for item_id in sample_items:
            std_trie.get(item_id)
        std_query_time = (time.time() - start) * 1000 / len(sample_items)

        start = time.time()
        for item_id in sample_items:
            radix_trie.get(item_id)
        radix_query_time = (time.time() - start) * 1000 / len(sample_items)

        # Prefix search
        prefixes = [item_id[:4] for item_id in sample_items[:100]]

        start = time.time()
        for prefix in prefixes:
            std_trie.prefix_search(prefix, max_results=20)
        std_prefix_time = (time.time() - start) * 1000 / len(prefixes)

        start = time.time()
        for prefix in prefixes:
            radix_trie.prefix_search(prefix, max_results=20)
        radix_prefix_time = (time.time() - start) * 1000 / len(prefixes)

        exp_result = {
            'name': f'scaling_{n_items}',
            'n_items': n_items,
            'std_build_time_s': std_build_time,
            'radix_build_time_s': radix_build_time,
            'radix_nodes': radix_mem['total_nodes'],
            'node_reduction_pct': (1 - radix_mem['total_nodes'] / (std_trie.total_items * 12)) * 100,  # Approximation
            'avg_edge_length': radix_mem['avg_edge_length'],
            'std_query_ms': std_query_time,
            'radix_query_ms': radix_query_time,
            'std_prefix_ms': std_prefix_time,
            'radix_prefix_ms': radix_prefix_time,
            'memory_mb': radix_mem['estimated_mb'],
        }
        results['experiments'].append(exp_result)

        print(f"    Build: std={std_build_time:.3f}s, radix={radix_build_time:.3f}s")
        print(f"    Nodes: radix={radix_mem['total_nodes']:,}, avg_edge={radix_mem['avg_edge_length']:.2f}")
        print(f"    Query: std={std_query_time:.4f}ms, radix={radix_query_time:.4f}ms")
        print(f"    Prefix: std={std_prefix_time:.4f}ms, radix={radix_prefix_time:.4f}ms")

    # Experiment 1.3: Different prefix similarity levels
    print("\n--- Experiment 1.3: Prefix Similarity Impact ---")

    for similarity in [0.2, 0.5, 0.8]:
        items = generate_synthetic_items(50000, id_length=15, prefix_similarity=similarity)

        radix_trie = RadixTrie()
        for item_id, label, category in items:
            radix_trie.insert(item_id, label, category)

        mem_stats = radix_trie.memory_stats()

        exp_result = {
            'name': f'prefix_sim_{similarity}',
            'prefix_similarity': similarity,
            'n_items': 50000,
            'total_nodes': mem_stats['total_nodes'],
            'avg_edge_length': mem_stats['avg_edge_length'],
            'compression_ratio': mem_stats['compression_ratio'],
        }
        results['experiments'].append(exp_result)

        print(f"  Similarity={similarity}: nodes={mem_stats['total_nodes']:,}, avg_edge={mem_stats['avg_edge_length']:.2f}")

    return results


def run_large_scale_experiments(data_path: str = None) -> Dict[str, Any]:
    """
    Phase 2: Large-scale experiments with real data.
    """
    print("\n" + "="*70)
    print("PHASE 2: Large-Scale Industrial Experiments")
    print("="*70)

    results = {
        'phase': 'large',
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # Try MIND Large dataset
    mind_large_path = data_path or "data/mind/MINDlarge_train"

    if Path(mind_large_path).exists():
        print(f"\n--- Loading MIND Large from {mind_large_path} ---")

        items = load_mind_items(mind_large_path, max_items=None)
        print(f"Loaded {len(items):,} items from MIND Large")

        # Build both tries
        print("\nBuilding Standard Trie...")
        std_trie = RetrievalTrie()
        start = time.time()
        for item_id, label, category in items:
            std_trie.insert(item_id, label, category)
        std_build_time = time.time() - start
        print(f"  Build time: {std_build_time:.2f}s")

        print("Building Radix Trie...")
        radix_trie = RadixTrie()
        start = time.time()
        for item_id, label, category in items:
            radix_trie.insert(item_id, label, category)
        radix_build_time = time.time() - start
        print(f"  Build time: {radix_build_time:.2f}s")

        # Memory comparison
        radix_mem = radix_trie.memory_stats()

        # Estimate standard trie memory (rough)
        # Each node has ~64 bytes overhead + dict for children
        std_nodes_estimate = sum(len(item_id) for item_id, _, _ in items)
        std_mem_estimate = std_nodes_estimate * 100 / (1024 * 1024)  # MB

        print(f"\n--- Memory Comparison ---")
        print(f"  Standard Trie (estimate): {std_mem_estimate:.1f} MB")
        print(f"  Radix Trie: {radix_mem['estimated_mb']:.1f} MB")
        print(f"  Savings: {(1 - radix_mem['estimated_mb']/std_mem_estimate)*100:.1f}%")
        print(f"  Radix nodes: {radix_mem['total_nodes']:,}")
        print(f"  Avg edge length: {radix_mem['avg_edge_length']:.2f}")

        # Query performance
        print(f"\n--- Query Performance (10K queries) ---")
        sample_items = random.sample([item_id for item_id, _, _ in items], min(10000, len(items)))

        # Exact lookup
        start = time.time()
        for item_id in sample_items:
            std_trie.get(item_id)
        std_query = (time.time() - start) * 1000 / len(sample_items)

        start = time.time()
        for item_id in sample_items:
            radix_trie.get(item_id)
        radix_query = (time.time() - start) * 1000 / len(sample_items)

        print(f"  Exact lookup: std={std_query:.4f}ms, radix={radix_query:.4f}ms")

        # Prefix search
        prefixes = list(set(item_id[:3] for item_id, _, _ in items[:1000]))[:100]

        start = time.time()
        for prefix in prefixes:
            std_trie.prefix_search(prefix, max_results=50)
        std_prefix = (time.time() - start) * 1000 / len(prefixes)

        start = time.time()
        for prefix in prefixes:
            radix_trie.prefix_search(prefix, max_results=50)
        radix_prefix = (time.time() - start) * 1000 / len(prefixes)

        print(f"  Prefix search: std={std_prefix:.4f}ms, radix={radix_prefix:.4f}ms")

        # Category retrieval
        categories = list(set(cat for _, _, cat in items))[:10]

        start = time.time()
        for cat in categories * 100:
            std_trie.get_by_category(cat, max_results=50)
        std_cat = (time.time() - start) * 1000 / (len(categories) * 100)

        start = time.time()
        for cat in categories * 100:
            radix_trie.get_by_category(cat, max_results=50)
        radix_cat = (time.time() - start) * 1000 / (len(categories) * 100)

        print(f"  Category lookup: std={std_cat:.4f}ms, radix={radix_cat:.4f}ms")

        # Correctness verification
        print(f"\n--- Correctness Verification ---")
        errors = 0
        for item_id in sample_items[:1000]:
            std_result = std_trie.get(item_id)
            radix_result = radix_trie.get(item_id)
            if (std_result is None) != (radix_result is None):
                errors += 1
            elif std_result and radix_result:
                if std_result.frequency != radix_result.frequency:
                    errors += 1
        print(f"  Errors in 1000 samples: {errors}")

        results['experiments'].append({
            'name': 'mind_large',
            'n_items': len(items),
            'std_build_time_s': std_build_time,
            'radix_build_time_s': radix_build_time,
            'std_mem_mb_estimate': std_mem_estimate,
            'radix_mem_mb': radix_mem['estimated_mb'],
            'memory_savings_pct': (1 - radix_mem['estimated_mb']/std_mem_estimate) * 100,
            'radix_nodes': radix_mem['total_nodes'],
            'avg_edge_length': radix_mem['avg_edge_length'],
            'std_query_ms': std_query,
            'radix_query_ms': radix_query,
            'std_prefix_ms': std_prefix,
            'radix_prefix_ms': radix_prefix,
            'errors': errors,
        })

    else:
        print(f"\nMIND Large not found at {mind_large_path}")
        print("Running with synthetic large-scale data...")

        # Generate 500K synthetic items
        for n_items in [100000, 500000]:
            print(f"\n--- Synthetic {n_items:,} items ---")

            items = generate_synthetic_items(n_items, id_length=15, prefix_similarity=0.5)

            std_trie = RetrievalTrie()
            radix_trie = RadixTrie()

            start = time.time()
            for item_id, label, category in items:
                std_trie.insert(item_id, label, category)
            std_build = time.time() - start

            start = time.time()
            for item_id, label, category in items:
                radix_trie.insert(item_id, label, category)
            radix_build = time.time() - start

            mem_stats = radix_trie.memory_stats()

            print(f"  Build: std={std_build:.2f}s, radix={radix_build:.2f}s")
            print(f"  Memory: {mem_stats['estimated_mb']:.1f} MB")
            print(f"  Nodes: {mem_stats['total_nodes']:,}")

            results['experiments'].append({
                'name': f'synthetic_{n_items}',
                'n_items': n_items,
                'std_build_time_s': std_build,
                'radix_build_time_s': radix_build,
                'radix_nodes': mem_stats['total_nodes'],
                'memory_mb': mem_stats['estimated_mb'],
            })

    return results


def run_recommendation_integration_test():
    """
    Test Radix Trie integration with recommendation pipeline.
    """
    print("\n" + "="*70)
    print("PHASE 3: Recommendation Pipeline Integration")
    print("="*70)

    from src.trie.radix_trie import RadixTrie

    # Create Radix Trie with simulated recommendation data
    radix_trie = RadixTrie()

    # Simulate news categories
    categories = ["politics", "sports", "tech", "business", "entertainment"]

    # Generate items
    items = []
    for cat in categories:
        for i in range(200):
            item_id = f"{cat[:3]}_{i:04d}"
            label = 1 if random.random() < (0.2 + 0.1 * categories.index(cat)) else 0
            items.append((item_id, label, cat))
            radix_trie.insert(item_id, label, cat)

    print(f"\nCreated {len(items)} items across {len(categories)} categories")

    # Simulate user history
    user_history = ["pol_0010", "pol_0023", "tec_0005", "spo_0100"]

    print(f"\nUser history: {user_history}")

    # Get recommendations using category-based retrieval
    recommendations = []
    seen_items = set(user_history)

    # Find user's preferred categories from history
    user_cats = defaultdict(int)
    for item_id in user_history:
        stats = radix_trie.get(item_id)
        if stats:
            user_cats[stats.category] += 1

    print(f"User category preferences: {dict(user_cats)}")

    # Get top items from preferred categories
    for cat, count in sorted(user_cats.items(), key=lambda x: -x[1]):
        cat_items = radix_trie.get_by_category(cat, max_results=20)
        for item in cat_items:
            if item.item_id not in seen_items:
                recommendations.append((item.item_id, item.ctr, item.category))
                seen_items.add(item.item_id)
                if len(recommendations) >= 10:
                    break

    # Add exploration items
    top_ctr = radix_trie.get_top_by_ctr(20)
    for item in top_ctr:
        if item.item_id not in seen_items:
            recommendations.append((item.item_id, item.ctr, item.category))
            if len(recommendations) >= 15:
                break

    print(f"\nTop 10 recommendations:")
    for i, (item_id, ctr, cat) in enumerate(recommendations[:10], 1):
        print(f"  {i}. {item_id} (CTR={ctr:.1%}, category={cat})")

    return {
        'n_items': len(items),
        'n_categories': len(categories),
        'user_history': user_history,
        'n_recommendations': len(recommendations),
    }


def main():
    parser = argparse.ArgumentParser(description='Radix Trie Experiments for KDD 2026')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['small', 'large', 'integration', 'all'],
                       help='Experiment phase to run')
    parser.add_argument('--data_path', type=str, default='data/mind/MINDlarge_train',
                       help='Path to MIND dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'phases': {}
    }

    if args.phase in ['small', 'all']:
        results = run_small_scale_experiments()
        all_results['phases']['small'] = results

    if args.phase in ['large', 'all']:
        results = run_large_scale_experiments(args.data_path)
        all_results['phases']['large'] = results

    if args.phase in ['integration', 'all']:
        results = run_recommendation_integration_test()
        all_results['phases']['integration'] = results

    # Save results
    output_path = args.output or f"results/radix_trie_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if 'small' in all_results['phases']:
        small = all_results['phases']['small']
        for exp in small.get('experiments', []):
            if 'scaling' in exp.get('name', ''):
                print(f"\n{exp['name']}:")
                print(f"  Query speedup: {exp['std_query_ms']/exp['radix_query_ms']:.2f}x")
                print(f"  Memory: {exp.get('memory_mb', 'N/A')} MB")

    if 'large' in all_results['phases']:
        large = all_results['phases']['large']
        for exp in large.get('experiments', []):
            if 'mind' in exp.get('name', ''):
                print(f"\n{exp['name']}:")
                print(f"  Memory savings: {exp.get('memory_savings_pct', 0):.1f}%")
                print(f"  Query: std={exp['std_query_ms']:.4f}ms, radix={exp['radix_query_ms']:.4f}ms")


if __name__ == '__main__':
    main()
