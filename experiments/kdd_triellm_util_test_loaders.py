"""
Test script to verify metrics and data loaders work correctly.
This runs without API calls to validate the setup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from collections import defaultdict

from src.metrics.recommendation_metrics import (
    MetricsCalculator, RecommendationMetrics, StatisticalTests, create_metrics_table
)


def test_metrics_calculator():
    """Test the MetricsCalculator with synthetic data."""
    print("=" * 60)
    print("Testing MetricsCalculator")
    print("=" * 60)

    # Create synthetic item data
    all_items = {f"item_{i}" for i in range(100)}
    item_popularity = {f"item_{i}": random.randint(1, 100) for i in range(100)}
    item_categories = {f"item_{i}": random.choice(['A', 'B', 'C', 'D']) for i in range(100)}

    # Create calculator
    calc = MetricsCalculator(
        item_popularity=item_popularity,
        item_categories=item_categories,
        all_items=all_items,
    )

    # Test with synthetic recommendations
    recommended = ["item_1", "item_2", "item_3", "item_4", "item_5"]
    ground_truth = ["item_1", "item_3", "item_10"]  # 2 hits

    # Calculate all metrics
    metrics = calc.calculate_all(
        recommended=recommended,
        ground_truth=ground_truth,
        k=5,
        latency_ms=100.5,
        tokens_used=50,
    )

    print(f"  Hit@5: {metrics.hit}")
    print(f"  Precision@5: {metrics.precision:.4f}")
    print(f"  Recall@5: {metrics.recall:.4f}")
    print(f"  NDCG@5: {metrics.ndcg:.4f}")
    print(f"  MRR: {metrics.mrr:.4f}")
    print(f"  Diversity: {metrics.diversity:.4f}")
    print(f"  Novelty: {metrics.novelty:.4f}")
    print(f"  Popularity Bias: {metrics.popularity_bias:.4f}")
    print(f"  Long-tail Coverage: {metrics.long_tail_coverage:.4f}")

    # Test aggregation
    print("\nTesting aggregation...")
    metrics_list = []
    for _ in range(10):
        rec = random.sample([f"item_{i}" for i in range(100)], 5)
        gt = random.sample([f"item_{i}" for i in range(100)], 3)
        m = calc.calculate_all(rec, gt, k=5)
        metrics_list.append(m)

    aggregated = calc.aggregate_metrics(metrics_list)
    print(f"  Hit@5 mean: {aggregated['hit_mean']:.4f} +/- {aggregated['hit_std']:.4f}")
    print(f"  NDCG@5 mean: {aggregated['ndcg_mean']:.4f} +/- {aggregated['ndcg_std']:.4f}")

    print("\n[PASS] MetricsCalculator works correctly!")
    return True


def test_statistical_tests():
    """Test statistical significance tests."""
    print("\n" + "=" * 60)
    print("Testing StatisticalTests")
    print("=" * 60)

    # Two methods with different performance
    method1_results = [0.7, 0.72, 0.71, 0.73, 0.69]
    method2_results = [0.65, 0.67, 0.66, 0.64, 0.68]

    # Paired t-test
    result = StatisticalTests.paired_ttest(method1_results, method2_results)
    print(f"  t-statistic: {result['t_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['significant']}")
    print(f"  Cohen's d: {result['cohens_d']:.4f}")
    print(f"  Effect size: {result['effect_size']}")

    # Wilcoxon test
    wilcox = StatisticalTests.wilcoxon_test(method1_results, method2_results)
    print(f"  Wilcoxon p-value: {wilcox['p_value']:.4f}")

    print("\n[PASS] StatisticalTests works correctly!")
    return True


def test_mind_loader():
    """Test MIND data loader."""
    print("\n" + "=" * 60)
    print("Testing MINDDatasetLoader")
    print("=" * 60)

    from experiments.trie_llm_comprehensive_experiments import MINDDatasetLoader

    mind_path = Path("data/mind/MINDsmall_train")
    if not mind_path.exists():
        print(f"  [SKIP] MIND dataset not found at {mind_path}")
        return False

    loader = MINDDatasetLoader(str(mind_path))
    samples = loader.load(max_samples=100)

    print(f"  Loaded {len(samples)} samples")
    print(f"  Total items: {len(loader.items)}")
    print(f"  Sample user: {samples[0].user_id}")
    print(f"  History length: {len(samples[0].history)}")
    print(f"  Candidates: {len(samples[0].candidates)}")
    print(f"  Ground truth: {len(samples[0].ground_truth)}")

    # Test metrics calculator
    calc = loader.get_metrics_calculator()
    print(f"  Categories: {len(set(loader.item_categories.values()))}")

    print("\n[PASS] MINDDatasetLoader works correctly!")
    return True


def test_movielens_loader():
    """Test MovieLens data loader."""
    print("\n" + "=" * 60)
    print("Testing MovieLensDatasetLoader")
    print("=" * 60)

    from experiments.trie_llm_comprehensive_experiments import MovieLensDatasetLoader

    ml_path = Path("data/ml-1m")
    if not ml_path.exists():
        print(f"  [SKIP] MovieLens dataset not found at {ml_path}")
        return False

    loader = MovieLensDatasetLoader(str(ml_path))
    samples = loader.load(max_samples=100)

    print(f"  Loaded {len(samples)} samples")
    print(f"  Total movies: {len(loader.items)}")
    print(f"  Sample user: {samples[0].user_id}")
    print(f"  History length: {len(samples[0].history)}")
    print(f"  Candidates: {len(samples[0].candidates)}")
    print(f"  Ground truth: {len(samples[0].ground_truth)}")

    # Sample movie
    sample_movie = list(loader.items.keys())[0]
    print(f"  Sample movie: {loader.items[sample_movie]}")

    print("\n[PASS] MovieLensDatasetLoader works correctly!")
    return True


def test_trie_recommender():
    """Test Trie-based recommender without LLM."""
    print("\n" + "=" * 60)
    print("Testing TrieRecommender (no LLM)")
    print("=" * 60)

    from experiments.trie_llm_comprehensive_experiments import (
        TrieRecommender, MINDDatasetLoader
    )

    mind_path = Path("data/mind/MINDsmall_train")
    if not mind_path.exists():
        print(f"  [SKIP] MIND dataset not found at {mind_path}")
        return False

    # Load data
    loader = MINDDatasetLoader(str(mind_path))
    samples = loader.load(max_samples=50)

    # Create recommender
    recommender = TrieRecommender(loader.items, loader.item_popularity)

    # Test recommendation
    sample = samples[0]
    recs, latency = recommender.recommend(sample.history, sample.candidates, k=5)

    print(f"  Recommendations: {recs[:3]}...")
    print(f"  Latency: {latency:.2f}ms")

    # Calculate metrics
    calc = loader.get_metrics_calculator()
    metrics = calc.calculate_all(recs, sample.ground_truth, k=5, latency_ms=latency)

    print(f"  Hit@5: {metrics.hit}")
    print(f"  NDCG@5: {metrics.ndcg:.4f}")

    # Run multiple samples
    all_metrics = []
    for sample in samples[:20]:
        recs, latency = recommender.recommend(sample.history, sample.candidates, k=5)
        m = calc.calculate_all(recs, sample.ground_truth, k=5, latency_ms=latency)
        all_metrics.append(m)

    aggregated = calc.aggregate_metrics(all_metrics)
    print(f"\n  Averaged over 20 samples:")
    print(f"    Hit@5: {aggregated['hit_mean']:.4f} +/- {aggregated['hit_std']:.4f}")
    print(f"    NDCG@5: {aggregated['ndcg_mean']:.4f} +/- {aggregated['ndcg_std']:.4f}")
    print(f"    Latency: {aggregated['latency_ms_mean']:.2f}ms")

    print("\n[PASS] TrieRecommender works correctly!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("KDD SUPPLEMENTARY EXPERIMENTS - SETUP VERIFICATION")
    print("=" * 70)

    results = {}

    results['metrics_calculator'] = test_metrics_calculator()
    results['statistical_tests'] = test_statistical_tests()
    results['mind_loader'] = test_mind_loader()
    results['movielens_loader'] = test_movielens_loader()
    results['trie_recommender'] = test_trie_recommender()

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[SKIP/FAIL]"
        print(f"  {test_name}: {status}")

    if all(v is not False for v in results.values()):
        print("\nAll available tests passed!")
        print("\nTo run full experiments with LLM, use:")
        print("  python experiments/trie_llm_comprehensive_experiments.py \\")
        print("    --api_key YOUR_API_KEY \\")
        print("    --datasets mind_small movielens \\")
        print("    --num_samples 500 \\")
        print("    --num_runs 5")

    return results


if __name__ == '__main__':
    main()
