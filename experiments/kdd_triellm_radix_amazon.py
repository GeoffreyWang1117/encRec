"""
Large-Scale Radix Trie Experiments with Amazon Dataset.

This script tests Radix Trie on hierarchical category data where
edge compression provides maximum benefit.

Datasets:
- Amazon Electronics (1.8M products)
- Amazon Home & Kitchen (4.1M products)
- Multiple categories with hierarchical structure

Key hypothesis: Radix Trie excels when item IDs share common prefixes
(e.g., category hierarchies like "Electronics/Phones/Apple/...")
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trie.retrieval_trie import RetrievalTrie
from src.trie.radix_trie import RadixTrie


def load_amazon_data(
    data_path: str,
    category: str = "Electronics",
    max_items: int = None,
    use_category_prefix: bool = True,
) -> Tuple[List[Tuple[str, int, str]], Dict[str, Any]]:
    """
    Load Amazon dataset with hierarchical category information.

    Args:
        data_path: Path to Amazon parquet files
        category: Category to load (Electronics, Home_and_Kitchen, etc.)
        max_items: Maximum items to load
        use_category_prefix: If True, create IDs with category prefix for Radix Trie advantage

    Returns:
        List of (item_id, label, category) and metadata dict
    """
    meta_path = Path(data_path) / f"{category}_meta.parquet"
    reviews_path = Path(data_path) / f"{category}_reviews.parquet"

    print(f"Loading Amazon {category} from {data_path}...")

    # Load metadata
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        print(f"  Loaded {len(meta_df):,} products from metadata")
    else:
        print(f"  Metadata not found at {meta_path}")
        meta_df = None

    # Load reviews for CTR estimation
    if reviews_path.exists():
        reviews_df = pd.read_parquet(reviews_path)
        if max_items:
            reviews_df = reviews_df.head(max_items * 10)  # Sample more reviews
        print(f"  Loaded {len(reviews_df):,} reviews")
    else:
        print(f"  Reviews not found at {reviews_path}")
        reviews_df = None

    items = []
    item_stats = defaultdict(lambda: {'views': 0, 'positive': 0})

    if reviews_df is not None:
        # Calculate item-level statistics from reviews
        # Rating >= 4 is considered positive
        for _, row in reviews_df.iterrows():
            asin = row.get('parent_asin') or row.get('asin', 'unknown')
            rating = row.get('rating', 3)
            item_stats[asin]['views'] += 1
            if rating >= 4:
                item_stats[asin]['positive'] += 1

    # Create items with category prefix for better Radix Trie compression
    if meta_df is not None:
        for idx, row in meta_df.iterrows():
            if max_items and len(items) >= max_items:
                break

            asin = row.get('parent_asin')
            if asin is None or (hasattr(asin, '__len__') and len(asin) == 0):
                asin = f'item_{idx}'

            # Get category hierarchy
            categories = row.get('categories')

            # Handle numpy arrays
            if hasattr(categories, 'tolist'):
                categories = categories.tolist()

            # Handle various formats
            cat_path = category  # default
            if categories is not None:
                try:
                    if isinstance(categories, (list, tuple)) and len(categories) > 0:
                        # Take first 3 category levels
                        cat_items = categories[:3] if not isinstance(categories[0], (list, tuple)) else categories
                        if isinstance(cat_items[0], (list, tuple)):
                            cat_items = cat_items[0][:3]
                        cat_path = '/'.join(str(c) for c in cat_items[:3])
                    elif isinstance(categories, str):
                        cat_path = categories
                except Exception:
                    cat_path = category

            # Create item ID with optional category prefix
            if use_category_prefix:
                item_id = f"{cat_path}/{asin}"
            else:
                item_id = str(asin)

            # Get label from reviews
            stats = item_stats.get(str(asin), {'views': 1, 'positive': 0})
            label = 1 if stats['positive'] > 0 else 0

            items.append((item_id, label, cat_path))

    # If no metadata, use reviews directly
    if not items and reviews_df is not None:
        seen_asins = set()
        for _, row in reviews_df.iterrows():
            if max_items and len(items) >= max_items:
                break

            asin = row.get('parent_asin') or row.get('asin', 'unknown')
            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            rating = row.get('rating', 3)
            label = 1 if rating >= 4 else 0

            if use_category_prefix:
                item_id = f"{category}/{asin}"
            else:
                item_id = asin

            items.append((item_id, label, category))

    metadata = {
        'category': category,
        'n_items': len(items),
        'n_reviews': len(reviews_df) if reviews_df is not None else 0,
        'use_category_prefix': use_category_prefix,
    }

    print(f"  Created {len(items):,} items")

    return items, metadata


def run_trie_comparison(
    items: List[Tuple[str, int, str]],
    name: str,
    n_queries: int = 10000,
) -> Dict[str, Any]:
    """
    Run comprehensive comparison between Standard and Radix Trie.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Items: {len(items):,}")
    print(f"{'='*60}")

    results = {
        'name': name,
        'n_items': len(items),
        'timestamp': datetime.now().isoformat(),
    }

    # Build Standard Trie
    print("\nBuilding Standard Trie...")
    std_trie = RetrievalTrie()
    start = time.time()
    for item_id, label, category in items:
        std_trie.insert(item_id, label, category)
    std_build_time = time.time() - start
    results['std_build_time_s'] = std_build_time
    print(f"  Build time: {std_build_time:.3f}s")

    # Count standard trie nodes
    def count_nodes(node) -> int:
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count

    std_nodes = count_nodes(std_trie.root)
    results['std_nodes'] = std_nodes
    print(f"  Nodes: {std_nodes:,}")

    # Build Radix Trie
    print("\nBuilding Radix Trie...")
    radix_trie = RadixTrie()
    start = time.time()
    for item_id, label, category in items:
        radix_trie.insert(item_id, label, category)
    radix_build_time = time.time() - start
    results['radix_build_time_s'] = radix_build_time
    print(f"  Build time: {radix_build_time:.3f}s")

    # Radix Trie stats
    radix_mem = radix_trie.memory_stats()
    results['radix_nodes'] = radix_mem['total_nodes']
    results['radix_avg_edge_len'] = radix_mem['avg_edge_length']
    results['radix_mem_mb'] = radix_mem['estimated_mb']
    print(f"  Nodes: {radix_mem['total_nodes']:,}")
    print(f"  Avg edge length: {radix_mem['avg_edge_length']:.2f}")

    # Calculate node reduction
    node_reduction = (1 - radix_mem['total_nodes'] / std_nodes) * 100
    results['node_reduction_pct'] = node_reduction
    print(f"  Node reduction: {node_reduction:.1f}%")

    # Estimate memory savings
    std_mem_estimate = std_nodes * 100 / (1024 * 1024)  # Rough estimate
    mem_savings = (1 - radix_mem['estimated_mb'] / std_mem_estimate) * 100
    results['std_mem_mb_estimate'] = std_mem_estimate
    results['memory_savings_pct'] = mem_savings
    print(f"  Memory savings: {mem_savings:.1f}%")

    # Query performance
    print(f"\nQuery Performance ({n_queries:,} queries)...")
    sample_size = min(n_queries, len(items))
    sample_items = random.sample([item_id for item_id, _, _ in items], sample_size)

    # Exact lookup
    start = time.time()
    for item_id in sample_items:
        std_trie.get(item_id)
    std_query = (time.time() - start) * 1000 / sample_size
    results['std_query_ms'] = std_query

    start = time.time()
    for item_id in sample_items:
        radix_trie.get(item_id)
    radix_query = (time.time() - start) * 1000 / sample_size
    results['radix_query_ms'] = radix_query

    query_speedup = std_query / radix_query if radix_query > 0 else 1
    results['query_speedup'] = query_speedup
    print(f"  Exact lookup: std={std_query:.4f}ms, radix={radix_query:.4f}ms (speedup: {query_speedup:.2f}x)")

    # Prefix search (important for category-based retrieval)
    # Extract various prefix lengths
    prefixes_short = list(set(item_id.split('/')[0] if '/' in item_id else item_id[:4]
                              for item_id, _, _ in items))[:100]
    prefixes_long = list(set('/'.join(item_id.split('/')[:2]) if '/' in item_id else item_id[:8]
                             for item_id, _, _ in items))[:100]

    # Short prefix search
    start = time.time()
    for prefix in prefixes_short:
        std_trie.prefix_search(prefix, max_results=50)
    std_prefix_short = (time.time() - start) * 1000 / len(prefixes_short)

    start = time.time()
    for prefix in prefixes_short:
        radix_trie.prefix_search(prefix, max_results=50)
    radix_prefix_short = (time.time() - start) * 1000 / len(prefixes_short)

    results['std_prefix_short_ms'] = std_prefix_short
    results['radix_prefix_short_ms'] = radix_prefix_short
    prefix_short_speedup = std_prefix_short / radix_prefix_short if radix_prefix_short > 0 else 1
    results['prefix_short_speedup'] = prefix_short_speedup
    print(f"  Short prefix: std={std_prefix_short:.4f}ms, radix={radix_prefix_short:.4f}ms (speedup: {prefix_short_speedup:.2f}x)")

    # Long prefix search
    start = time.time()
    for prefix in prefixes_long:
        std_trie.prefix_search(prefix, max_results=50)
    std_prefix_long = (time.time() - start) * 1000 / len(prefixes_long)

    start = time.time()
    for prefix in prefixes_long:
        radix_trie.prefix_search(prefix, max_results=50)
    radix_prefix_long = (time.time() - start) * 1000 / len(prefixes_long)

    results['std_prefix_long_ms'] = std_prefix_long
    results['radix_prefix_long_ms'] = radix_prefix_long
    prefix_long_speedup = std_prefix_long / radix_prefix_long if radix_prefix_long > 0 else 1
    results['prefix_long_speedup'] = prefix_long_speedup
    print(f"  Long prefix: std={std_prefix_long:.4f}ms, radix={radix_prefix_long:.4f}ms (speedup: {prefix_long_speedup:.2f}x)")

    # Category retrieval
    categories = list(set(cat for _, _, cat in items))[:20]

    start = time.time()
    for cat in categories * 50:
        std_trie.get_by_category(cat, max_results=50)
    std_cat = (time.time() - start) * 1000 / (len(categories) * 50)

    start = time.time()
    for cat in categories * 50:
        radix_trie.get_by_category(cat, max_results=50)
    radix_cat = (time.time() - start) * 1000 / (len(categories) * 50)

    results['std_category_ms'] = std_cat
    results['radix_category_ms'] = radix_cat
    print(f"  Category lookup: std={std_cat:.4f}ms, radix={radix_cat:.4f}ms")

    # Correctness verification
    print("\nCorrectness verification...")
    errors = 0
    for item_id in sample_items[:1000]:
        std_result = std_trie.get(item_id)
        radix_result = radix_trie.get(item_id)
        if (std_result is None) != (radix_result is None):
            errors += 1
        elif std_result and radix_result:
            if std_result.frequency != radix_result.frequency:
                errors += 1
    results['errors'] = errors
    print(f"  Errors: {errors}/1000")

    return results


def run_amazon_experiments(data_path: str, max_items: int = None) -> Dict[str, Any]:
    """
    Run experiments on Amazon dataset with hierarchical categories.
    """
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': [],
    }

    # Categories to test
    categories = [
        ('Electronics', 100000),      # Large, diverse
        ('Home_and_Kitchen', 100000), # Large, hierarchical
        ('All_Beauty', 50000),        # Smaller
    ]

    for category, default_max in categories:
        items_limit = max_items or default_max

        # Test with category prefix (should favor Radix Trie)
        print(f"\n{'#'*70}")
        print(f"# Amazon {category} - WITH Category Prefix")
        print(f"{'#'*70}")

        try:
            items_with_prefix, meta = load_amazon_data(
                data_path,
                category=category,
                max_items=items_limit,
                use_category_prefix=True,
            )

            if items_with_prefix:
                results = run_trie_comparison(
                    items_with_prefix,
                    f"amazon_{category.lower()}_with_prefix",
                    n_queries=min(10000, len(items_with_prefix)),
                )
                results['metadata'] = meta
                all_results['experiments'].append(results)

        except Exception as e:
            print(f"Error loading {category}: {e}")
            continue

        # Test without category prefix (baseline)
        print(f"\n{'#'*70}")
        print(f"# Amazon {category} - WITHOUT Category Prefix (Baseline)")
        print(f"{'#'*70}")

        try:
            items_no_prefix, meta = load_amazon_data(
                data_path,
                category=category,
                max_items=items_limit,
                use_category_prefix=False,
            )

            if items_no_prefix:
                results = run_trie_comparison(
                    items_no_prefix,
                    f"amazon_{category.lower()}_no_prefix",
                    n_queries=min(10000, len(items_no_prefix)),
                )
                results['metadata'] = meta
                all_results['experiments'].append(results)

        except Exception as e:
            print(f"Error loading {category} (no prefix): {e}")

    return all_results


def run_mind_comparison():
    """
    Run MIND dataset comparison for reference.
    """
    print(f"\n{'#'*70}")
    print(f"# MIND Large - Reference Comparison")
    print(f"{'#'*70}")

    mind_path = "data/mind/MINDlarge_train"
    if not Path(mind_path).exists():
        print(f"MIND Large not found at {mind_path}")
        return None

    # Load MIND data
    news_path = Path(mind_path) / "news.tsv"
    items = []

    with open(news_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100000:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else "unknown"
                label = 1 if random.random() < 0.25 else 0
                items.append((news_id, label, category))

    print(f"Loaded {len(items):,} MIND items")

    results = run_trie_comparison(items, "mind_large", n_queries=10000)
    return results


def print_summary(all_results: Dict[str, Any]):
    """Print summary table of all experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\n{'Experiment':<40} {'Nodes Red.':<12} {'Mem Save':<12} {'Query':<12} {'Prefix':<12}")
    print("-"*80)

    for exp in all_results.get('experiments', []):
        name = exp.get('name', 'unknown')[:38]
        node_red = f"{exp.get('node_reduction_pct', 0):.1f}%"
        mem_save = f"{exp.get('memory_savings_pct', 0):.1f}%"
        query = f"{exp.get('query_speedup', 1):.2f}x"
        prefix = f"{exp.get('prefix_short_speedup', 1):.2f}x"

        print(f"{name:<40} {node_red:<12} {mem_save:<12} {query:<12} {prefix:<12}")

    print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Amazon Radix Trie Experiments')
    parser.add_argument('--data_path', type=str, default='/home/coder-gw/DataSets/amazon',
                       help='Path to Amazon dataset')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum items per category')
    parser.add_argument('--include_mind', action='store_true',
                       help='Include MIND comparison')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')

    args = parser.parse_args()

    print("="*70)
    print("Large-Scale Radix Trie Experiments with Amazon Dataset")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Max items: {args.max_items or 'default per category'}")

    # Run Amazon experiments
    all_results = run_amazon_experiments(args.data_path, args.max_items)

    # Optionally include MIND comparison
    if args.include_mind:
        mind_results = run_mind_comparison()
        if mind_results:
            all_results['experiments'].append(mind_results)

    # Print summary
    print_summary(all_results)

    # Save results
    output_path = args.output or f"results/radix_amazon_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    with_prefix = [e for e in all_results['experiments'] if 'with_prefix' in e.get('name', '')]
    no_prefix = [e for e in all_results['experiments'] if 'no_prefix' in e.get('name', '')]

    if with_prefix and no_prefix:
        avg_node_red_prefix = np.mean([e['node_reduction_pct'] for e in with_prefix])
        avg_node_red_no = np.mean([e['node_reduction_pct'] for e in no_prefix])

        avg_mem_save_prefix = np.mean([e['memory_savings_pct'] for e in with_prefix])
        avg_mem_save_no = np.mean([e['memory_savings_pct'] for e in no_prefix])

        print(f"\nWith Category Prefix (Radix Trie advantage):")
        print(f"  Avg node reduction: {avg_node_red_prefix:.1f}%")
        print(f"  Avg memory savings: {avg_mem_save_prefix:.1f}%")

        print(f"\nWithout Category Prefix (Baseline):")
        print(f"  Avg node reduction: {avg_node_red_no:.1f}%")
        print(f"  Avg memory savings: {avg_mem_save_no:.1f}%")

        print(f"\nConclusion: Category prefix increases memory savings by {avg_mem_save_prefix - avg_mem_save_no:.1f}%")


if __name__ == '__main__':
    main()
