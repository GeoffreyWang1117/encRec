#!/usr/bin/env python3
"""
Scalability analysis for CIKM 2026 paper.

Analyzes:
1. Training time vs dataset size
2. Memory usage vs dataset size
3. Inference latency
4. Trie construction overhead
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc

from src.models.backbone import DeepFM, DLRM, DCNv2


def measure_memory():
    """Measure current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    return 0


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def create_synthetic_dataset(num_samples, dense_dim=13, num_sparse_fields=26, vocab_size=10000):
    """Create synthetic dataset for scalability testing."""
    dense = torch.randn(num_samples, dense_dim)
    sparse = torch.randint(0, vocab_size, (num_samples, num_sparse_fields))
    labels = torch.randint(0, 2, (num_samples,)).float()
    return TensorDataset(dense, sparse, labels)


def measure_training_time(model, dataloader, device, num_batches=100):
    """Measure training time for a fixed number of batches."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    reset_memory()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    batch_count = 0
    
    for batch in dataloader:
        if batch_count >= num_batches:
            break
        dense, sparse, labels = batch
        dense = dense.to(device)
        sparse = sparse.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(dense, sparse)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs['logits'], labels)
        loss.backward()
        optimizer.step()
        batch_count += 1
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    memory = measure_memory()
    
    return {'time_seconds': elapsed, 'time_per_batch': elapsed / num_batches, 'memory_mb': memory}


def measure_inference_latency(model, dataloader, device, num_batches=100):
    """Measure inference latency."""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            dense, sparse, _ = batch
            dense = dense.to(device)
            sparse = sparse.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dense, sparse)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)
    
    batch_size = len(batch[0])
    return {
        'avg_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'throughput_qps': batch_size / (np.mean(latencies) / 1000),
    }


def run_scalability_experiment(model_class, model_kwargs, data_sizes, device, batch_size=1024):
    """Run scalability experiments."""
    results = []
    
    for num_samples in data_sizes:
        print(f"  Testing {num_samples:,} samples...")
        dataset = create_synthetic_dataset(num_samples, dense_dim=model_kwargs.get('dense_dim', 13))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        reset_memory()
        model = model_class(**model_kwargs).to(device)
        
        train_info = measure_training_time(model, dataloader, device, num_batches=50)
        inference_info = measure_inference_latency(model, dataloader, device, num_batches=50)
        
        results.append({
            'num_samples': num_samples,
            **train_info,
            **inference_info,
        })
        
        del model, dataset, dataloader
        reset_memory()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='results/scalability.json')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_sizes = [10000, 50000, 100000, 500000]
    sparse_dims = {f'f{i}': 10000 for i in range(26)}
    
    models = {
        'DeepFM': (DeepFM, {'dense_dim': 13, 'sparse_dims': sparse_dims, 'embedding_dim': 16, 'hidden_dims': [256,128,64], 'dropout': 0.1}),
        'DLRM': (DLRM, {'dense_dim': 13, 'sparse_dims': sparse_dims, 'embedding_dim': 16, 'dropout': 0.1}),
        'DCNv2': (DCNv2, {'dense_dim': 13, 'sparse_dims': sparse_dims, 'embedding_dim': 16, 'dropout': 0.1}),
    }
    
    all_results = {}
    for name, (cls, kwargs) in models.items():
        print(f"\nTesting {name}")
        all_results[name] = run_scalability_experiment(cls, kwargs, data_sizes, device)
    
    # Print summary
    print("\n" + "="*70)
    print("SCALABILITY SUMMARY")
    print("="*70)
    for name, results in all_results.items():
        print(f"\n{name}:")
        for r in results:
            print(f"  {r['num_samples']:>8,} samples: {r['time_seconds']:.1f}s, {r['memory_mb']:.0f}MB, {r['throughput_qps']:.0f} QPS")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
