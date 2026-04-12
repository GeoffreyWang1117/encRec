#!/usr/bin/env python3
"""
实际效率测试 - 测量真实推理延迟
用于: KDD 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.criteo_loader import load_criteo_data, SPARSE_COLS
from src.models.backbone import DeepFM
from src.models.moe import TrieMoERecommender
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder


def measure_inference_latency(model, data_loader, device, num_batches=100, warmup=10):
    """测量推理延迟"""
    model.eval()

    # Warmup
    batch_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(warmup):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                batch = next(batch_iter)

            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            _ = model(dense, sparse)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Actual measurement
    latencies = []
    batch_iter = iter(data_loader)

    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                batch = next(batch_iter)

            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dense, sparse)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }


def main():
    print("="*70)
    print("REAL EFFICIENCY TEST")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_ds, val_ds, test_ds = load_criteo_data(
        'data/criteo/criteo_5m.parquet',
        sample_size=10000
    )

    feature_dims = train_ds.get_feature_dims()
    dense_dim = feature_dims['dense_dim']
    sparse_dims = feature_dims['sparse_dims']

    batch_sizes = [32, 64, 128, 256, 512]

    print("\n" + "="*70)
    print("DEEPFM INFERENCE LATENCY")
    print("="*70)

    # Create DeepFM model
    deepfm = DeepFM(
        dense_dim=dense_dim,
        sparse_dims=sparse_dims,
        embedding_dim=32,
        hidden_dims=[256, 128, 64],
    ).to(device)

    print(f"\n{'Batch Size':<12} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-"*60)

    for bs in batch_sizes:
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        latency = measure_inference_latency(deepfm, loader, device, num_batches=50)
        print(f"{bs:<12} {latency['mean_ms']:<12.3f} {latency['p50_ms']:<12.3f} "
              f"{latency['p95_ms']:<12.3f} {latency['p99_ms']:<12.3f}")

    print("\n" + "="*70)
    print("TRIE-MOE INFERENCE LATENCY")
    print("="*70)

    # Build Trie
    builder = TrieBuilder(
        hierarchy_config=['frequency', 'info'],
        num_experts=8,
        expert_strategy='frequency_aware',
    )

    for field in SPARSE_COLS:
        if field in train_ds.freq_stats:
            freq_dict = train_ds.freq_stats[field]
            for token, count in freq_dict.items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)

    builder.statistics.compute_statistics()

    for field in SPARSE_COLS:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(field_name=field, hierarchy_config=['frequency', 'info'])
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(8, 'frequency_aware')
            builder.tries[field] = trie

    # Create FastTrieEncoder
    trie_encoder = FastTrieEncoder(
        tries=builder.tries,
        vocab_sizes=sparse_dims,
        routing_dim=32,
    ).to(device)

    # Create Trie-MoE model
    triemoe = TrieMoERecommender(
        dense_dim=dense_dim,
        sparse_dims=sparse_dims,
        embedding_dim=32,
        trie_routing_dim=32,
        num_experts=8,
        top_k=2,
        hidden_dims=[128, 64],
    ).to(device)

    print(f"\n{'Batch Size':<12} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-"*60)

    # Wrap model to include trie encoding
    class TrieMoEWithEncoder(nn.Module):
        def __init__(self, model, encoder):
            super().__init__()
            self.model = model
            self.encoder = encoder

        def forward(self, dense, sparse):
            trie_features = self.encoder(sparse)
            return self.model(dense, sparse, trie_routing_vec=trie_features)

    wrapped_model = TrieMoEWithEncoder(triemoe, trie_encoder).to(device)

    for bs in batch_sizes:
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        latency = measure_inference_latency(wrapped_model, loader, device, num_batches=50)
        print(f"{bs:<12} {latency['mean_ms']:<12.3f} {latency['p50_ms']:<12.3f} "
              f"{latency['p95_ms']:<12.3f} {latency['p99_ms']:<12.3f}")

    print("\n" + "="*70)
    print("THROUGHPUT COMPARISON (samples/sec)")
    print("="*70)

    bs = 256
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # DeepFM throughput
    deepfm_latency = measure_inference_latency(deepfm, loader, device, num_batches=100)
    deepfm_throughput = (bs / deepfm_latency['mean_ms']) * 1000

    # Trie-MoE throughput
    triemoe_latency = measure_inference_latency(wrapped_model, loader, device, num_batches=100)
    triemoe_throughput = (bs / triemoe_latency['mean_ms']) * 1000

    print(f"\nBatch size = {bs}")
    print(f"DeepFM:   {deepfm_throughput:,.0f} samples/sec ({deepfm_latency['mean_ms']:.2f} ms/batch)")
    print(f"Trie-MoE: {triemoe_throughput:,.0f} samples/sec ({triemoe_latency['mean_ms']:.2f} ms/batch)")
    print(f"Overhead: {(triemoe_latency['mean_ms']/deepfm_latency['mean_ms'] - 1)*100:.1f}%")

    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
