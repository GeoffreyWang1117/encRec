#!/usr/bin/env python3
"""
Comprehensive paper experiment runner.

Runs all baseline models (DeepFM, DLRM, DCNv2, AutoInt, FinalMLP, DCNv3)
and Trie-MoE with multiple runs for statistical significance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import json
import time
from collections import defaultdict
from datetime import datetime

from src.data.amazon_loader import AmazonDataset, load_amazon_data
from src.data.criteo_loader import load_criteo_data, create_criteo_dataloaders
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.encoder import TrieEncoder
from src.models.backbone import DeepFM, DLRM, DenseBaseline, DCNv2, AutoInt, FinalMLP, DCNv3
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics, AUCMeter, LogLossMeter


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'defaults' in config:
        base_configs = config.pop('defaults')
        merged_config = {}
        for base_name in base_configs:
            if isinstance(base_name, str):
                base_path = config_path.parent / f"{base_name}.yaml"
                if base_path.exists():
                    with open(base_path, 'r') as f:
                        base_config = yaml.safe_load(f)
                    merged_config = deep_merge(merged_config, base_config)
        config = deep_merge(merged_config, config)
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_model(model_name: str, feature_dims: dict, model_config: dict, backbone_config: dict):
    """Create model by name."""
    common_args = {
        'dense_dim': feature_dims['dense_dim'],
        'sparse_dims': feature_dims['sparse_dims'],
        'embedding_dim': model_config['embedding_dim'],
    }

    if model_name == 'deepfm':
        return DeepFM(
            **common_args,
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif model_name == 'dlrm':
        return DLRM(
            **common_args,
            dropout=backbone_config['dropout'],
        )
    elif model_name == 'dcnv2':
        return DCNv2(
            **common_args,
            cross_num_layers=backbone_config.get('cross_num_layers', 3),
            deep_hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif model_name == 'autoint':
        return AutoInt(
            **common_args,
            num_attention_layers=backbone_config.get('num_attention_layers', 3),
            num_heads=backbone_config.get('num_heads', 2),
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif model_name == 'finalmlp':
        return FinalMLP(
            **common_args,
            stream1_dims=backbone_config.get('stream1_dims', [256, 128]),
            stream2_dims=backbone_config.get('stream2_dims', [256, 128]),
            dropout=backbone_config['dropout'],
        )
    elif model_name == 'dcnv3':
        return DCNv3(
            **common_args,
            cross_num_layers=backbone_config.get('cross_num_layers', 3),
            deep_hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, dataloader, optimizer, device, gradient_clip=1.0,
                trie_encoder=None, sparse_fields=None, vocab_maps=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    auc_meter = AUCMeter()

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

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        auc_meter.update(preds, labels.cpu().numpy())

    return {'loss': total_loss / len(dataloader), 'auc': auc_meter.compute()}


def evaluate(model, dataloader, device, trie_encoder=None, sparse_fields=None, vocab_maps=None):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

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

            loss = model.compute_loss(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def build_trie(train_dataset, sparse_fields, config):
    """Build Trie structure from dataset."""
    trie_config = config['model']['trie']

    builder = TrieBuilder(
        hierarchy_config=trie_config['hierarchy'],
        num_experts=trie_config['num_experts'],
        expert_strategy=trie_config['expert_strategy'],
    )

    for field in sparse_fields:
        if field in train_dataset.freq_stats:
            freq_dict = train_dataset.freq_stats[field]
            for token, count in freq_dict.items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)

    builder.statistics.compute_statistics()

    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(
                field_name=field,
                hierarchy_config=trie_config['hierarchy']
            )
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(trie_config['num_experts'], trie_config['expert_strategy'])
            builder.tries[field] = trie

    return builder


def run_single_experiment(model_name, train_loader, val_loader, test_loader,
                          feature_dims, config, device, seed,
                          train_dataset=None, sparse_fields=None):
    """Run a single experiment with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_config = config['model']
    backbone_config = model_config['backbone']
    train_config = config['training']

    trie_encoder = None
    vocab_maps = None

    if model_name == 'trie_moe':
        # Build Trie
        trie_builder = build_trie(train_dataset, sparse_fields, config)

        vocab_maps = {}
        for field, vocab in train_dataset.vocab.items():
            vocab_maps[field] = {idx: token for token, idx in vocab.items()}

        trie_encoder = TrieEncoder(
            tries=trie_builder.tries,
            routing_dim=model_config['moe'].get('routing_dim', 32),
            aggregation='mean',
            learnable_projection=True,
        ).to(device)

        model = TrieMoERecommender(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            trie_routing_dim=trie_encoder.routing_dim,
            num_experts=model_config['trie']['num_experts'],
            expert_hidden_dim=model_config['moe']['expert_hidden_dim'],
            expert_output_dim=model_config['moe']['expert_output_dim'],
            top_k=model_config['moe']['top_k'],
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        ).to(device)

        optimizer = optim.Adam(
            list(model.parameters()) + list(trie_encoder.parameters()),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 0)
        )
    else:
        model = create_model(model_name, feature_dims, model_config, backbone_config)
        model = model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 0)
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['epochs'])

    start_time = time.time()
    best_val_auc = 0.0

    for epoch in range(train_config['epochs']):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            gradient_clip=train_config.get('gradient_clip', 1.0),
            trie_encoder=trie_encoder,
            sparse_fields=sparse_fields,
            vocab_maps=vocab_maps
        )

        val_metrics = evaluate(
            model, val_loader, device,
            trie_encoder=trie_encoder,
            sparse_fields=sparse_fields,
            vocab_maps=vocab_maps
        )

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']

        scheduler.step()

    training_time = time.time() - start_time

    test_metrics = evaluate(
        model, test_loader, device,
        trie_encoder=trie_encoder,
        sparse_fields=sparse_fields,
        vocab_maps=vocab_maps
    )
    test_metrics['training_time'] = training_time

    return test_metrics


def run_experiments_amazon(config, device, num_runs=10, models=None):
    """Run experiments on Amazon dataset."""
    print("\n" + "="*60)
    print("Running Amazon Electronics Experiments")
    print("="*60)

    if models is None:
        models = ['deepfm', 'dlrm', 'dcnv2', 'autoint', 'finalmlp', 'dcnv3', 'trie_moe']

    # Load data
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=config['data'].get('sample_size'),
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    feature_dims = train_dataset.get_feature_dims()
    sparse_fields = train_dataset.actual_sparse_cols

    results = defaultdict(list)

    for model_name in models:
        print(f"\n--- Training {model_name.upper()} ---")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}", end=" ")
            seed = 42 + run

            try:
                metrics = run_single_experiment(
                    model_name, train_loader, val_loader, test_loader,
                    feature_dims, config, device, seed,
                    train_dataset=train_dataset, sparse_fields=sparse_fields
                )
                results[model_name].append(metrics)
                print(f"- AUC: {metrics['auc']:.4f}, LogLoss: {metrics['logloss']:.4f}, Time: {metrics['training_time']:.1f}s")
            except Exception as e:
                print(f"- FAILED: {e}")

    return dict(results)


def compute_statistics(results):
    """Compute mean and std for each metric."""
    stats = {}
    for model_name, runs in results.items():
        if not runs:
            continue

        aucs = [r['auc'] for r in runs]
        loglosses = [r['logloss'] for r in runs]
        times = [r.get('training_time', 0) for r in runs]

        stats[model_name] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'logloss_mean': np.mean(loglosses),
            'logloss_std': np.std(loglosses),
            'time_mean': np.mean(times),
            'num_runs': len(runs),
        }

    return stats


def print_latex_table(stats, dataset_name):
    """Print results as LaTeX table."""
    print(f"\n{'='*60}")
    print(f"LaTeX Table for {dataset_name}")
    print("="*60)

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(f"\\caption{{Results on {dataset_name} ({list(stats.values())[0]['num_runs']} runs)}}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Model & AUC & LogLoss & Time (s) \\")
    print(r"\midrule")

    # Sort by AUC
    sorted_models = sorted(stats.items(), key=lambda x: x[1]['auc_mean'], reverse=True)
    best_auc = sorted_models[0][1]['auc_mean']

    for model_name, s in sorted_models:
        auc_str = f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}"
        if s['auc_mean'] == best_auc:
            auc_str = r"\textbf{" + auc_str + "}"

        print(f"{model_name.upper()} & {auc_str} & {s['logloss_mean']:.4f} & {s['time_mean']:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def save_results(results, stats, output_dir):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for model, runs in results.items():
            serializable_results[model] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in run.items()}
                for run in runs
            ]
        json.dump(serializable_results, f, indent=2)

    # Save statistics
    with open(output_dir / f"stats_{timestamp}.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run paper experiments')
    parser.add_argument('--config', type=str, default='configs/amazon.yaml')
    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'criteo'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/paper_experiments')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = load_config(args.config)

    if args.dataset == 'amazon':
        results = run_experiments_amazon(config, device, args.num_runs, args.models)
        dataset_name = "Amazon Electronics"
    else:
        raise NotImplementedError("Criteo dataset runner not yet implemented")

    stats = compute_statistics(results)

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    for model_name, s in stats.items():
        print(f"{model_name.upper():12} | AUC: {s['auc_mean']:.4f}±{s['auc_std']:.4f} | "
              f"LogLoss: {s['logloss_mean']:.4f}±{s['logloss_std']:.4f} | "
              f"Time: {s['time_mean']:.1f}s")

    print_latex_table(stats, dataset_name)
    save_results(results, stats, args.output_dir)


if __name__ == '__main__':
    main()
