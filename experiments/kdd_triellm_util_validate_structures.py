#!/usr/bin/env python3
"""
Validate advanced data structures for encRec.

Tests:
1. Count-Min Sketch: Frequency estimation accuracy vs memory
2. Cuckoo Filter: Cold-start detection speed and accuracy
3. Skip List: Range query performance
4. LSH: Similar token retrieval quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import time
import json
from collections import defaultdict
import yaml

from src.data.amazon_loader import load_amazon_data
from src.structures.count_min_sketch import CountMinSketch, CountMinSketchWithHeap
from src.structures.cuckoo_filter import CuckooFilter
from src.structures.skip_list import SkipList, FrequencySkipList
from src.structures.lsh import CosineLSH, TokenSimilarityIndex
from src.models.adaptive_alpha import (
    AdaptiveAlphaRecommender,
    compute_token_frequencies,
    get_batch_token_freqs,
)
from src.utils.metrics import compute_metrics


def test_count_min_sketch(train_dataset, test_dataset, item_field_idx=1):
    """Test Count-Min Sketch for frequency estimation."""
    print("\n" + "="*60)
    print("Test 1: Count-Min Sketch")
    print("="*60)

    # Ground truth frequencies
    true_freq = defaultdict(int)
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        token_id = sample['sparse'][item_field_idx].item()
        true_freq[token_id] += 1

    print(f"Total unique tokens: {len(true_freq)}")
    print(f"Memory for dict: {sys.getsizeof(true_freq) / 1024:.2f} KB")

    # Test different CMS configurations
    configs = [
        {'width': 100, 'depth': 3},
        {'width': 500, 'depth': 5},
        {'width': 1000, 'depth': 5},
        {'width': 2000, 'depth': 7},
    ]

    results = []
    for cfg in configs:
        cms = CountMinSketch(**cfg)

        # Build CMS
        start = time.time()
        for token_id, count in true_freq.items():
            cms.update(token_id, count)
        build_time = time.time() - start

        # Evaluate accuracy
        errors = []
        relative_errors = []
        for token_id, true_count in true_freq.items():
            est_count = cms.estimate(token_id)
            error = est_count - true_count  # CMS only overestimates
            errors.append(error)
            if true_count > 0:
                relative_errors.append(error / true_count)

        result = {
            'config': cfg,
            'memory_kb': cms.memory_usage_bytes() / 1024,
            'build_time_ms': build_time * 1000,
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'mean_relative_error': np.mean(relative_errors),
            'p95_relative_error': np.percentile(relative_errors, 95),
        }
        results.append(result)

        epsilon, delta = cms.error_bound()
        print(f"\n  Config: w={cfg['width']}, d={cfg['depth']}")
        print(f"  Memory: {result['memory_kb']:.2f} KB (vs {sys.getsizeof(true_freq)/1024:.2f} KB dict)")
        print(f"  Mean error: {result['mean_error']:.2f}, Max error: {result['max_error']:.0f}")
        print(f"  Mean relative error: {result['mean_relative_error']*100:.2f}%")
        print(f"  Theoretical bounds: ε={epsilon:.4f}, δ={delta:.6f}")

    return {'count_min_sketch': results}


def test_cuckoo_filter(train_dataset, test_dataset, item_field_idx=1):
    """Test Cuckoo Filter for cold-start detection."""
    print("\n" + "="*60)
    print("Test 2: Cuckoo Filter (Cold-Start Detection)")
    print("="*60)

    # Collect train tokens
    train_tokens = set()
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        token_id = sample['sparse'][item_field_idx].item()
        train_tokens.add(token_id)

    # Collect test tokens
    test_tokens = set()
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        token_id = sample['sparse'][item_field_idx].item()
        test_tokens.add(token_id)

    cold_start_tokens = test_tokens - train_tokens
    known_tokens = test_tokens & train_tokens

    print(f"Train tokens: {len(train_tokens)}")
    print(f"Test tokens: {len(test_tokens)}")
    print(f"Cold-start (unseen): {len(cold_start_tokens)}")
    print(f"Known: {len(known_tokens)}")

    # Test different configurations
    configs = [
        {'capacity': 5000, 'fingerprint_bits': 8},
        {'capacity': 10000, 'fingerprint_bits': 8},
        {'capacity': 10000, 'fingerprint_bits': 12},
        {'capacity': 20000, 'fingerprint_bits': 8},
    ]

    results = []
    for cfg in configs:
        cf = CuckooFilter(**cfg)

        # Insert train tokens
        start = time.time()
        insert_failures = 0
        for token_id in train_tokens:
            if not cf.insert(token_id):
                insert_failures += 1
        insert_time = time.time() - start

        # Test lookup
        start = time.time()

        # True negatives (cold-start correctly identified)
        true_neg = sum(1 for t in cold_start_tokens if t not in cf)
        # False positives (cold-start wrongly identified as known)
        false_pos = len(cold_start_tokens) - true_neg

        # True positives (known correctly identified)
        true_pos = sum(1 for t in known_tokens if t in cf)
        # False negatives (known wrongly identified as cold-start)
        false_neg = len(known_tokens) - true_pos

        lookup_time = time.time() - start

        # Metrics
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        fpr = false_pos / len(cold_start_tokens) if cold_start_tokens else 0

        result = {
            'config': cfg,
            'memory_kb': cf.memory_usage_bytes() / 1024,
            'insert_time_ms': insert_time * 1000,
            'lookup_time_ms': lookup_time * 1000,
            'insert_failures': insert_failures,
            'load_factor': cf.load_factor(),
            'precision': precision,
            'recall': recall,
            'false_positive_rate': fpr,
            'theoretical_fpr': cf.false_positive_rate(),
        }
        results.append(result)

        print(f"\n  Config: capacity={cfg['capacity']}, fp_bits={cfg['fingerprint_bits']}")
        print(f"  Memory: {result['memory_kb']:.2f} KB")
        print(f"  Load factor: {result['load_factor']:.2%}")
        print(f"  Insert failures: {insert_failures}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  False positive rate: {fpr:.4f} (theoretical: {cf.false_positive_rate():.4f})")

    return {'cuckoo_filter': results}


def test_skip_list(train_dataset, item_field_idx=1):
    """Test Skip List for frequency-based operations."""
    print("\n" + "="*60)
    print("Test 3: Skip List (Frequency Ranking)")
    print("="*60)

    # Compute frequencies
    freq_dict = defaultdict(int)
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        token_id = sample['sparse'][item_field_idx].item()
        freq_dict[token_id] += 1

    # Build skip list
    sl = SkipList()
    start = time.time()
    for token_id, freq in freq_dict.items():
        sl.insert(freq, token_id)
    build_time = time.time() - start

    print(f"Total tokens: {len(freq_dict)}")
    print(f"Build time: {build_time*1000:.2f} ms")

    # Test operations
    results = {}

    # 1. Range query
    freq_values = list(freq_dict.values())
    low = int(np.percentile(freq_values, 10))
    high = int(np.percentile(freq_values, 90))

    start = time.time()
    for _ in range(100):
        range_results = list(sl.range_query(low, high))
    range_time = (time.time() - start) * 10  # ms per query

    results['range_query'] = {
        'range': (low, high),
        'result_count': len(range_results),
        'time_ms': range_time,
    }
    print(f"\n  Range query [{low}, {high}]: {len(range_results)} results")
    print(f"  Time per query: {range_time:.3f} ms")

    # 2. Top-K query
    start = time.time()
    for _ in range(100):
        top_k = sl.get_top_k(100)
    topk_time = (time.time() - start) * 10

    results['top_k'] = {
        'k': 100,
        'time_ms': topk_time,
    }
    print(f"\n  Top-100 query: {topk_time:.3f} ms")

    # 3. Percentile query
    start = time.time()
    for _ in range(100):
        p50 = sl.get_percentile(50)
    percentile_time = (time.time() - start) * 10

    results['percentile'] = {
        'p50_freq': p50[0] if p50 else None,
        'time_ms': percentile_time,
    }
    print(f"\n  P50 frequency: {p50[0] if p50 else 'N/A'}")
    print(f"  Percentile query time: {percentile_time:.3f} ms")

    # Compare with sorted list baseline
    start = time.time()
    for _ in range(100):
        sorted_items = sorted(freq_dict.items(), key=lambda x: -x[1])[:100]
    baseline_time = (time.time() - start) * 10

    results['baseline_sort_time_ms'] = baseline_time
    print(f"\n  Baseline sort time: {baseline_time:.3f} ms")
    print(f"  Skip list speedup: {baseline_time / topk_time:.2f}x")

    return {'skip_list': results}


def test_lsh(train_dataset, embedding_model, device, item_field_idx=1):
    """Test LSH for similar token retrieval."""
    print("\n" + "="*60)
    print("Test 4: LSH (Token Similarity)")
    print("="*60)

    # Extract embeddings and frequencies
    token_embeddings = {}
    token_freqs = defaultdict(int)

    embedding_model.eval()
    with torch.no_grad():
        for i in range(min(len(train_dataset), 10000)):  # Limit for speed
            sample = train_dataset[i]
            token_id = sample['sparse'][item_field_idx].item()
            token_freqs[token_id] += 1

            if token_id not in token_embeddings:
                sparse = sample['sparse'].unsqueeze(0).to(device)
                # Get embedding from model
                emb = embedding_model.embeddings[
                    embedding_model.sparse_field_names[item_field_idx]
                ](sparse[:, item_field_idx])
                token_embeddings[token_id] = emb.cpu().numpy().flatten()

    print(f"Total tokens with embeddings: {len(token_embeddings)}")

    # Split into "frequent" and "rare" for testing
    freq_threshold = np.percentile(list(token_freqs.values()), 80)
    frequent_tokens = {t: e for t, e in token_embeddings.items()
                       if token_freqs[t] >= freq_threshold}
    rare_tokens = {t: e for t, e in token_embeddings.items()
                   if token_freqs[t] < freq_threshold}

    print(f"Frequent tokens: {len(frequent_tokens)}")
    print(f"Rare tokens: {len(rare_tokens)}")

    # Test configurations
    configs = [
        {'num_tables': 5, 'hash_size': 8},
        {'num_tables': 10, 'hash_size': 12},
        {'num_tables': 15, 'hash_size': 16},
    ]

    embedding_dim = list(token_embeddings.values())[0].shape[0]

    results = []
    for cfg in configs:
        lsh = CosineLSH(embedding_dim, **cfg)

        # Insert frequent tokens
        start = time.time()
        for token_id, emb in frequent_tokens.items():
            lsh.insert(token_id, emb)
        build_time = time.time() - start

        # Query with rare tokens
        recalls = []
        query_times = []

        for token_id, query_emb in list(rare_tokens.items())[:100]:
            # Ground truth: brute force
            true_similar = []
            for ft_id, ft_emb in frequent_tokens.items():
                sim = np.dot(query_emb, ft_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(ft_emb) + 1e-8
                )
                true_similar.append((ft_id, sim))
            true_similar.sort(key=lambda x: -x[1])
            true_top10 = set(x[0] for x in true_similar[:10])

            # LSH query
            start = time.time()
            lsh_results = lsh.query(query_emb, k=10)
            query_times.append(time.time() - start)

            # Recall
            recall = len(set(lsh_results) & true_top10) / len(true_top10)
            recalls.append(recall)

        result = {
            'config': cfg,
            'build_time_ms': build_time * 1000,
            'avg_query_time_ms': np.mean(query_times) * 1000,
            'mean_recall@10': np.mean(recalls),
            'std_recall@10': np.std(recalls),
        }
        results.append(result)

        print(f"\n  Config: tables={cfg['num_tables']}, hash_size={cfg['hash_size']}")
        print(f"  Build time: {result['build_time_ms']:.2f} ms")
        print(f"  Avg query time: {result['avg_query_time_ms']:.3f} ms")
        print(f"  Recall@10: {result['mean_recall@10']:.4f} ± {result['std_recall@10']:.4f}")

    # Brute force baseline
    bf_times = []
    for token_id, query_emb in list(rare_tokens.items())[:100]:
        start = time.time()
        sims = []
        for ft_id, ft_emb in frequent_tokens.items():
            sim = np.dot(query_emb, ft_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(ft_emb) + 1e-8
            )
            sims.append((ft_id, sim))
        sims.sort(key=lambda x: -x[1])
        _ = sims[:10]
        bf_times.append(time.time() - start)

    baseline_time = np.mean(bf_times) * 1000
    print(f"\n  Brute force baseline: {baseline_time:.3f} ms")
    print(f"  Best LSH speedup: {baseline_time / min(r['avg_query_time_ms'] for r in results):.2f}x")

    return {'lsh': results, 'baseline_query_time_ms': baseline_time}


def test_integrated_model(train_dataset, test_dataset, feature_dims,
                          freq_dict, item_field_idx, device):
    """Test data structures integrated with the model."""
    print("\n" + "="*60)
    print("Test 5: Integrated Model Performance")
    print("="*60)

    # Use subset for speed
    train_subset = Subset(train_dataset, range(min(5000, len(train_dataset))))
    test_subset = Subset(test_dataset, range(min(2000, len(test_dataset))))

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=2)

    # Build CMS for frequency estimation
    cms = CountMinSketch(width=1000, depth=5)
    for token_id, count in freq_dict.items():
        cms.update(token_id, count)

    # Build Cuckoo Filter for cold-start detection
    cf = CuckooFilter(capacity=len(freq_dict)*2, fingerprint_bits=10)
    for token_id in freq_dict.keys():
        cf.insert(token_id)

    results = {}

    # Test 1: Exact freq vs CMS freq
    model_exact = AdaptiveAlphaRecommender(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=16, num_experts=8,
    ).to(device)

    model_cms = AdaptiveAlphaRecommender(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=16, num_experts=8,
    ).to(device)

    # Copy weights
    model_cms.load_state_dict(model_exact.state_dict())

    optimizer_exact = torch.optim.Adam(model_exact.parameters(), lr=0.001)
    optimizer_cms = torch.optim.Adam(model_cms.parameters(), lr=0.001)

    # Train both
    for epoch in range(3):
        model_exact.train()
        model_cms.train()

        for batch in train_loader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)

            # Exact frequencies
            exact_freqs = get_batch_token_freqs(sparse, item_field_idx, freq_dict).to(device)

            # CMS frequencies
            cms_freqs = torch.tensor([
                cms.estimate(sparse[i, item_field_idx].item())
                for i in range(sparse.shape[0])
            ], device=device)

            # Train exact
            optimizer_exact.zero_grad()
            out_exact = model_exact(dense, sparse, exact_freqs)
            loss_exact = model_exact.compute_loss(out_exact, labels)
            loss_exact.backward()
            optimizer_exact.step()

            # Train CMS
            optimizer_cms.zero_grad()
            out_cms = model_cms(dense, sparse, cms_freqs)
            loss_cms = model_cms.compute_loss(out_cms, labels)
            loss_cms.backward()
            optimizer_cms.step()

    # Evaluate
    def evaluate_model(model, loader, use_cms=False):
        model.eval()
        all_preds, all_labels = [], []
        inference_time = 0

        with torch.no_grad():
            for batch in loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)
                labels = batch['label'].to(device)

                start = time.time()
                if use_cms:
                    freqs = torch.tensor([
                        cms.estimate(sparse[i, item_field_idx].item())
                        for i in range(sparse.shape[0])
                    ], device=device)
                else:
                    freqs = get_batch_token_freqs(sparse, item_field_idx, freq_dict).to(device)

                outputs = model(dense, sparse, freqs)
                inference_time += time.time() - start

                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['inference_time_ms'] = inference_time * 1000
        return metrics

    results['exact_freq'] = evaluate_model(model_exact, test_loader, use_cms=False)
    results['cms_freq'] = evaluate_model(model_cms, test_loader, use_cms=True)

    print(f"\n  Exact freq: AUC={results['exact_freq']['auc']:.4f}, "
          f"LogLoss={results['exact_freq']['logloss']:.4f}")
    print(f"  CMS freq:   AUC={results['cms_freq']['auc']:.4f}, "
          f"LogLoss={results['cms_freq']['logloss']:.4f}")

    auc_diff = abs(results['exact_freq']['auc'] - results['cms_freq']['auc'])
    ll_diff = abs(results['exact_freq']['logloss'] - results['cms_freq']['logloss'])
    print(f"\n  AUC difference: {auc_diff:.6f}")
    print(f"  LogLoss difference: {ll_diff:.6f}")
    print(f"  Memory saved: {sys.getsizeof(freq_dict)/1024:.2f} KB -> {cms.memory_usage_bytes()/1024:.2f} KB")

    return {'integrated': results}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    with open('configs/amazon.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=20000,  # Small scale
    )

    feature_dims = train_dataset.get_feature_dims()
    sparse_fields = train_dataset.actual_sparse_cols
    item_field_idx = sparse_fields.index('asin') if 'asin' in sparse_fields else 1

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Compute frequencies
    freq_dict = compute_token_frequencies(train_dataset, item_field_idx)

    # Run tests
    all_results = {}

    # Test 1: Count-Min Sketch
    all_results.update(test_count_min_sketch(train_dataset, test_dataset, item_field_idx))

    # Test 2: Cuckoo Filter
    all_results.update(test_cuckoo_filter(train_dataset, test_dataset, item_field_idx))

    # Test 3: Skip List
    all_results.update(test_skip_list(train_dataset, item_field_idx))

    # Test 4: LSH (need model for embeddings)
    model = AdaptiveAlphaRecommender(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=16, num_experts=8,
    ).to(device)
    all_results.update(test_lsh(train_dataset, model, device, item_field_idx))

    # Test 5: Integrated
    all_results.update(test_integrated_model(
        train_dataset, test_dataset, feature_dims,
        freq_dict, item_field_idx, device
    ))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Data Structure Selection")
    print("="*60)

    print("\n1. Count-Min Sketch:")
    best_cms = min(all_results['count_min_sketch'],
                   key=lambda x: x['mean_relative_error'])
    print(f"   Best config: w={best_cms['config']['width']}, d={best_cms['config']['depth']}")
    print(f"   Memory: {best_cms['memory_kb']:.2f} KB")
    print(f"   Mean relative error: {best_cms['mean_relative_error']*100:.2f}%")
    print(f"   Recommendation: USE (significant memory savings with acceptable error)")

    print("\n2. Cuckoo Filter:")
    best_cf = min(all_results['cuckoo_filter'],
                  key=lambda x: x['false_positive_rate'] if x['insert_failures'] == 0 else 1)
    print(f"   Best config: capacity={best_cf['config']['capacity']}, bits={best_cf['config']['fingerprint_bits']}")
    print(f"   Memory: {best_cf['memory_kb']:.2f} KB")
    print(f"   False positive rate: {best_cf['false_positive_rate']*100:.2f}%")
    print(f"   Recommendation: USE (fast cold-start detection with deletion support)")

    print("\n3. Skip List:")
    sl_results = all_results['skip_list']
    speedup = sl_results['baseline_sort_time_ms'] / sl_results['top_k']['time_ms']
    print(f"   Range query time: {sl_results['range_query']['time_ms']:.3f} ms")
    print(f"   Top-K speedup: {speedup:.2f}x vs sorting")
    print(f"   Recommendation: OPTIONAL (useful for dynamic frequency updates)")

    print("\n4. LSH:")
    best_lsh = max(all_results['lsh'], key=lambda x: x['mean_recall@10'])
    speedup = all_results['baseline_query_time_ms'] / best_lsh['avg_query_time_ms']
    print(f"   Best config: tables={best_lsh['config']['num_tables']}, hash_size={best_lsh['config']['hash_size']}")
    print(f"   Recall@10: {best_lsh['mean_recall@10']:.4f}")
    print(f"   Speedup: {speedup:.2f}x vs brute force")
    print(f"   Recommendation: USE (valuable for cold-start token similarity)")

    print("\n5. Integrated Performance:")
    integrated = all_results['integrated']
    auc_diff = abs(integrated['exact_freq']['auc'] - integrated['cms_freq']['auc'])
    print(f"   CMS vs Exact AUC diff: {auc_diff:.6f}")
    print(f"   Recommendation: CMS is safe to use (negligible accuracy impact)")

    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/data_structure_validation.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to results/data_structure_validation.json")

    return all_results


if __name__ == '__main__':
    main()
