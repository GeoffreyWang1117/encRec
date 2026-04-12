#!/usr/bin/env python3
"""
Complete analysis for paper: trains models and evaluates on cold-start buckets.
Runs 10 experiments for statistical significance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import yaml
import json
import time
from collections import defaultdict
from datetime import datetime

from src.data.amazon_loader import load_amazon_data
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.encoder import TrieEncoder
from src.models.backbone import DeepFM, DLRM, DCNv2, FinalMLP, DCNv3
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics, AUCMeter


FREQUENCY_BUCKETS = {
    'very_rare': (1, 5),
    'rare': (6, 20),
    'moderate': (21, 100),
    'frequent': (101, float('inf')),
}


def compute_item_frequencies(dataset, field_idx=1):
    """Compute item frequencies from training data."""
    freq_counter = defaultdict(int)
    for i in range(len(dataset)):
        sample = dataset[i]
        item_id = sample['sparse'][field_idx].item()
        freq_counter[item_id] += 1
    return freq_counter


def get_bucket_indices(dataset, frequencies, field_idx=1):
    """Get indices for each frequency bucket."""
    bucket_indices = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        item_id = sample['sparse'][field_idx].item()
        freq = frequencies.get(item_id, 0)
        for bucket_name, (low, high) in FREQUENCY_BUCKETS.items():
            if low <= freq <= high:
                bucket_indices[bucket_name].append(i)
                break
    return bucket_indices


def train_epoch(model, dataloader, optimizer, device, trie_encoder=None, sparse_fields=None, vocab_maps=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        if trie_encoder is not None:
            trie_routing = trie_encoder(sparse, sparse_fields, vocab_maps).to(device)
            outputs = model(dense, sparse, trie_routing)
        else:
            outputs = model(dense, sparse)
        loss = model.compute_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, trie_encoder=None, sparse_fields=None, vocab_maps=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)
            if trie_encoder is not None:
                trie_routing = trie_encoder(sparse, sparse_fields, vocab_maps).to(device)
                outputs = model(dense, sparse, trie_routing)
            else:
                outputs = model(dense, sparse)
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return compute_metrics(np.array(all_preds), np.array(all_labels))


def evaluate_bucket(model, dataset, indices, device, batch_size=512, trie_encoder=None, sparse_fields=None, vocab_maps=None):
    if len(indices) == 0:
        return {'auc': 0.5, 'logloss': 1.0, 'count': 0}
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    metrics = evaluate(model, loader, device, trie_encoder, sparse_fields, vocab_maps)
    metrics['count'] = len(indices)
    return metrics


def build_trie(train_dataset, sparse_fields, config):
    trie_config = config['model']['trie']
    builder = TrieBuilder(
        hierarchy_config=trie_config['hierarchy'],
        num_experts=trie_config['num_experts'],
        expert_strategy=trie_config['expert_strategy'],
    )
    for field in sparse_fields:
        if field in train_dataset.freq_stats:
            for token, count in train_dataset.freq_stats[field].items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)
    builder.statistics.compute_statistics()
    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(field_name=field, hierarchy_config=trie_config['hierarchy'])
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(trie_config['num_experts'], trie_config['expert_strategy'])
            builder.tries[field] = trie
    return builder


def create_model(name, feature_dims, embedding_dim=16, hidden_dims=[256, 128, 64], dropout=0.1):
    common = {'dense_dim': feature_dims['dense_dim'], 'sparse_dims': feature_dims['sparse_dims'], 'embedding_dim': embedding_dim}
    if name == 'deepfm':
        return DeepFM(**common, hidden_dims=hidden_dims, dropout=dropout)
    elif name == 'dlrm':
        return DLRM(**common, dropout=dropout)
    elif name == 'dcnv2':
        return DCNv2(**common, dropout=dropout)
    elif name == 'finalmlp':
        return FinalMLP(**common, dropout=dropout)
    elif name == 'dcnv3':
        return DCNv3(**common, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


def run_single_experiment(model_name, train_loader, val_loader, test_dataset, bucket_indices,
                          feature_dims, config, device, seed, train_dataset=None, sparse_fields=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    embedding_dim = config['model'].get('embedding_dim', 16)
    hidden_dims = config['model'].get('backbone', {}).get('hidden_dims', [256, 128, 64])
    dropout = config['model'].get('backbone', {}).get('dropout', 0.1)
    epochs = config['training'].get('epochs', 5)
    lr = config['training'].get('learning_rate', 0.001)

    trie_encoder, vocab_maps = None, None

    if model_name == 'trie_moe':
        trie_builder = build_trie(train_dataset, sparse_fields, config)
        vocab_maps = {field: {idx: token for token, idx in vocab.items()} for field, vocab in train_dataset.vocab.items()}
        trie_encoder = TrieEncoder(tries=trie_builder.tries, routing_dim=32, aggregation='mean', learnable_projection=True).to(device)
        model = TrieMoERecommender(
            dense_dim=feature_dims['dense_dim'], sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=embedding_dim, trie_routing_dim=32, num_experts=config['model']['trie']['num_experts'],
            expert_hidden_dim=64, expert_output_dim=64, top_k=2, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(trie_encoder.parameters()), lr=lr)
    else:
        model = create_model(model_name, feature_dims, embedding_dim, hidden_dims, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device, trie_encoder, sparse_fields, vocab_maps)

    # Evaluate overall
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    overall = evaluate(model, test_loader, device, trie_encoder, sparse_fields, vocab_maps)

    # Evaluate per bucket
    bucket_results = {}
    for bucket_name, indices in bucket_indices.items():
        bucket_results[bucket_name] = evaluate_bucket(model, test_dataset, indices, device, 512, trie_encoder, sparse_fields, vocab_maps)

    return {'overall': overall, 'buckets': bucket_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/amazon.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--models', nargs='+', default=['deepfm', 'dlrm', 'dcnv2', 'trie_moe'])
    parser.add_argument('--output', type=str, default='results/full_analysis.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load base config if exists
    if 'defaults' in config:
        base_path = Path(args.config).parent / 'base.yaml'
        if base_path.exists():
            with open(base_path, 'r') as f:
                base = yaml.safe_load(f)
            for k, v in base.items():
                if k not in config:
                    config[k] = v
                elif isinstance(v, dict):
                    config[k] = {**v, **config.get(k, {})}

    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=config['data'].get('sample_size'),
    )

    feature_dims = train_dataset.get_feature_dims()
    sparse_fields = train_dataset.actual_sparse_cols
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Compute frequencies and buckets
    print("Computing item frequencies...")
    item_field_idx = sparse_fields.index('asin') if 'asin' in sparse_fields else 1
    train_frequencies = compute_item_frequencies(train_dataset, item_field_idx)
    bucket_indices = get_bucket_indices(test_dataset, train_frequencies, item_field_idx)

    print("Bucket distribution:")
    for name, indices in bucket_indices.items():
        print(f"  {name}: {len(indices)} samples")

    batch_size = config['data'].get('batch_size', 1024)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_results = defaultdict(list)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} ({args.num_runs} runs)")
        print('='*60)

        for run in range(args.num_runs):
            print(f"  Run {run+1}/{args.num_runs}", end=" ", flush=True)
            seed = 42 + run
            start = time.time()

            try:
                result = run_single_experiment(
                    model_name, train_loader, val_loader, test_dataset, bucket_indices,
                    feature_dims, config, device, seed, train_dataset, sparse_fields
                )
                elapsed = time.time() - start
                result['time'] = elapsed
                all_results[model_name].append(result)
                print(f"- AUC: {result['overall']['auc']:.4f}, LogLoss: {result['overall']['logloss']:.4f}, Time: {elapsed:.1f}s")
            except Exception as e:
                print(f"- FAILED: {e}")

    # Compute statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary = {}
    for model, runs in all_results.items():
        aucs = [r['overall']['auc'] for r in runs]
        loglosses = [r['overall']['logloss'] for r in runs]

        summary[model] = {
            'overall': {
                'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
                'logloss_mean': np.mean(loglosses), 'logloss_std': np.std(loglosses),
            },
            'buckets': {}
        }

        print(f"\n{model.upper()}:")
        print(f"  Overall: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, LogLoss={np.mean(loglosses):.4f}±{np.std(loglosses):.4f}")

        for bucket in FREQUENCY_BUCKETS.keys():
            bucket_loglosses = [r['buckets'].get(bucket, {}).get('logloss', 1.0) for r in runs]
            bucket_aucs = [r['buckets'].get(bucket, {}).get('auc', 0.5) for r in runs]
            summary[model]['buckets'][bucket] = {
                'logloss_mean': np.mean(bucket_loglosses), 'logloss_std': np.std(bucket_loglosses),
                'auc_mean': np.mean(bucket_aucs), 'auc_std': np.std(bucket_aucs),
            }
            print(f"  {bucket}: LogLoss={np.mean(bucket_loglosses):.4f}±{np.std(bucket_loglosses):.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'config': args.config,
        'num_runs': args.num_runs,
        'summary': summary,
        'raw_results': {k: [{kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                            for kk, vv in r['overall'].items()} for r in v] for k, v in all_results.items()}
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
