#!/usr/bin/env python3
"""
Supplement SOTA Comparison with DCNv2 and AutoInt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime

from src.data.criteo_loader import load_criteo_data, SPARSE_COLS
from src.models.backbone import DeepFM, DCNv2, AutoInt
from src.models.moe import TrieMoERecommender
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder
from src.utils.metrics import compute_metrics


def build_trie(dataset, sparse_fields, num_experts=8):
    builder = TrieBuilder(hierarchy_config=['frequency', 'info'], num_experts=num_experts, expert_strategy='frequency_aware')
    for field in sparse_fields:
        if field in dataset.freq_stats:
            for token, count in dataset.freq_stats[field].items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)
    builder.statistics.compute_statistics()
    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(field_name=field, hierarchy_config=['frequency', 'info'])
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(num_experts, 'frequency_aware')
            builder.tries[field] = trie
    return builder


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=10, lr=0.001, use_trie=False, trie_encoder=None,
                       sparse_fields=None, vocab_maps=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc, best_model_state = 0, None
    train_time = 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            dense, sparse, labels = batch['dense'].to(device), batch['sparse'].to(device), batch['label'].to(device)
            if use_trie and trie_encoder:
                output = model(dense, sparse, trie_routing_vec=trie_encoder(sparse, sparse_fields, vocab_maps))
            else:
                output = model(dense, sparse)
            loss = criterion(output['logits'].squeeze(), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        train_time += time.time() - epoch_start

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                dense, sparse = batch['dense'].to(device), batch['sparse'].to(device)
                if use_trie and trie_encoder:
                    output = model(dense, sparse, trie_routing_vec=trie_encoder(sparse, sparse_fields, vocab_maps))
                else:
                    output = model(dense, sparse)
                val_preds.extend(torch.sigmoid(output['logits']).cpu().numpy())
                val_labels.extend(batch['label'].numpy())
        val_auc = compute_metrics(np.array(val_preds), np.array(val_labels))['auc']
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            dense, sparse = batch['dense'].to(device), batch['sparse'].to(device)
            if use_trie and trie_encoder:
                output = model(dense, sparse, trie_routing_vec=trie_encoder(sparse, sparse_fields, vocab_maps))
            else:
                output = model(dense, sparse)
            test_preds.extend(torch.sigmoid(output['logits']).cpu().numpy())
            test_labels.extend(batch['label'].numpy())

    test_auc = compute_metrics(np.array(test_preds), np.array(test_labels))['auc']
    return {'test_auc': test_auc, 'train_time': train_time, 'params': sum(p.numel() for p in model.parameters())}


def main():
    print("=" * 70)
    print("SUPPLEMENT: DCNv2 & AutoInt SOTA Comparison")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_sizes = [50000, 100000]
    num_runs = 5
    epochs = 10

    all_results = {}

    for data_size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Data Size: {data_size:,}")
        print(f"{'='*60}")

        train_dataset, val_dataset, test_dataset = load_criteo_data(
            'data/criteo/criteo_5m.parquet', sample_size=data_size
        )
        sparse_fields = SPARSE_COLS
        feature_dims = train_dataset.get_feature_dims()
        vocab_maps = {f: {i: t for t, i in v.items()} for f, v in train_dataset.vocab.items()}
        vocab_sizes = {f: len(v) for f, v in train_dataset.vocab.items()}

        batch_size = min(512, data_size // 10)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

        trie_builder = build_trie(train_dataset, sparse_fields)
        trie_encoder = FastTrieEncoder(tries=trie_builder.tries, vocab_sizes=vocab_sizes).to(device)

        models_config = {
            'DCNv2': {
                'class': DCNv2,
                'kwargs': {
                    'dense_dim': feature_dims['dense_dim'],
                    'sparse_dims': feature_dims['sparse_dims'],
                    'embedding_dim': 16,
                    'cross_num_layers': 3,
                    'cross_low_rank': 32,
                    'dropout': 0.2,
                },
                'use_trie': False,
            },
            'AutoInt': {
                'class': AutoInt,
                'kwargs': {
                    'dense_dim': feature_dims['dense_dim'],
                    'sparse_dims': feature_dims['sparse_dims'],
                    'embedding_dim': 16,
                    'num_attention_layers': 3,
                    'num_heads': 4,
                    'attention_dim': 32,
                    'dropout': 0.2,
                },
                'use_trie': False,
            },
            'Trie-MoE': {
                'class': TrieMoERecommender,
                'kwargs': {
                    'dense_dim': feature_dims['dense_dim'],
                    'sparse_dims': feature_dims['sparse_dims'],
                    'embedding_dim': 16,
                    'num_experts': 4,
                    'dropout': 0.2,
                },
                'use_trie': True,
            },
        }

        results = {name: {'aucs': [], 'times': [], 'params': None} for name in models_config}

        for model_name, config in models_config.items():
            print(f"\n--- {model_name} ---")
            for run in range(num_runs):
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                try:
                    model = config['class'](**config['kwargs']).to(device)
                    result = train_and_evaluate(
                        model, train_loader, val_loader, test_loader, device, epochs=epochs,
                        use_trie=config['use_trie'],
                        trie_encoder=trie_encoder if config['use_trie'] else None,
                        sparse_fields=sparse_fields if config['use_trie'] else None,
                        vocab_maps=vocab_maps if config['use_trie'] else None,
                    )
                    results[model_name]['aucs'].append(result['test_auc'])
                    results[model_name]['times'].append(result['train_time'])
                    results[model_name]['params'] = result['params']
                    print(f"  Run {run+1}: AUC={result['test_auc']:.4f}")
                except Exception as e:
                    print(f"  Run {run+1}: Error - {e}")

        summary = {}
        for model_name, r in results.items():
            if r['aucs']:
                summary[model_name] = {
                    'auc_mean': np.mean(r['aucs']),
                    'auc_std': np.std(r['aucs']),
                    'time_mean': np.mean(r['times']),
                    'params': r['params'],
                    'runs': r['aucs'],
                }
        all_results[data_size] = summary

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for data_size in data_sizes:
        print(f"\n--- {data_size:,} samples ---")
        print(f"{'Model':<15} {'AUC (mean±std)':<22} {'Params':<12}")
        print("-" * 50)
        for model_name in ['DCNv2', 'AutoInt', 'Trie-MoE']:
            if model_name in all_results[data_size]:
                r = all_results[data_size][model_name]
                print(f"{model_name:<15} {r['auc_mean']:.4f}±{r['auc_std']:.4f}        {r['params']:,}")

    # Compare with previous results
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS SOTA RESULTS")
    print("=" * 70)

    prev_results = {
        50000: {'FinalMLP': 0.7553, 'Trie-MoE': 0.7517, 'DLRM': 0.7438, 'DeepFM': 0.7423, 'PLE': 0.7418},
        100000: {'FinalMLP': 0.7619, 'Trie-MoE': 0.7578, 'DLRM': 0.7539, 'DeepFM': 0.7490, 'PLE': 0.7455},
    }

    for data_size in data_sizes:
        print(f"\n--- {data_size:,} samples (All models ranked) ---")
        combined = dict(prev_results[data_size])
        for model_name in ['DCNv2', 'AutoInt']:
            if model_name in all_results[data_size]:
                combined[model_name] = all_results[data_size][model_name]['auc_mean']

        sorted_models = sorted(combined.items(), key=lambda x: -x[1])
        for rank, (name, auc) in enumerate(sorted_models, 1):
            marker = " ← Trie-MoE" if name == "Trie-MoE" else (" ← NEW" if name in ['DCNv2', 'AutoInt'] else "")
            print(f"  {rank}. {name:<12}: {auc:.4f}{marker}")

    # Save results
    results_path = Path('results/sota_supplement')
    results_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_results = {str(size): {m: {'auc_mean': r['auc_mean'], 'auc_std': r['auc_std'], 'params': r['params'], 'runs': r['runs']}
                                for m, r in res.items()} for size, res in all_results.items()}
    with open(results_path / f'results_{timestamp}.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path / f'results_{timestamp}.json'}")


if __name__ == '__main__':
    main()
