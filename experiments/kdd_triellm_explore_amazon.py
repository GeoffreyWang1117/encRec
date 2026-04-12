#!/usr/bin/env python3
"""
Train models on Amazon Encrypted dataset.

Compares baseline (DeepFM) vs Trie-MoE on Amazon data with hashed features.
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
from collections import defaultdict

from src.data.amazon_loader import AmazonDataset, load_amazon_data
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.encoder import TrieEncoder
from src.trie.statistics import TrieStatistics
from src.models.backbone import DeepFM
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics, AUCMeter, BucketedMetrics
from src.utils.logger import ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with inheritance support."""
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


def build_trie_from_amazon_dataset(
    dataset: AmazonDataset,
    sparse_fields: list,
    config: dict,
) -> TrieBuilder:
    """Build Trie structure from Amazon dataset statistics."""
    print("Building Trie structure from Amazon data...")

    trie_config = config['model']['trie']

    builder = TrieBuilder(
        hierarchy_config=trie_config['hierarchy'],
        num_experts=trie_config['num_experts'],
        expert_strategy=trie_config['expert_strategy'],
    )

    # Use frequency stats already computed in dataset
    for field in sparse_fields:
        if field in dataset.freq_stats:
            freq_dict = dataset.freq_stats[field]
            # Update builder statistics
            for token, count in freq_dict.items():
                for _ in range(min(count, 100)):  # Cap iterations for efficiency
                    builder.statistics.update(field, token, label=1)

    builder.statistics.compute_statistics()

    # Build Trie for each field
    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(
                field_name=field,
                hierarchy_config=trie_config['hierarchy']
            )
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(trie_config['num_experts'], trie_config['expert_strategy'])
            builder.tries[field] = trie

    print(f"Built Tries for {len(builder.tries)} fields")
    for field, trie in builder.tries.items():
        summary = trie.summarize()
        print(f"  {field}: {summary['total_tokens']} tokens, {summary['leaf_nodes']} leaves")

    return builder


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    trie_encoder=None,
    sparse_fields=None,
    vocab_maps=None,
    gradient_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    auc_meter = AUCMeter()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if trie_encoder is not None:
            # Trie-MoE model
            trie_routing = trie_encoder(sparse, sparse_fields, vocab_maps).to(device)
            outputs = model(dense, sparse, trie_routing)
        else:
            # Baseline model
            outputs = model(dense, sparse)

        loss = model.compute_loss(outputs, labels)
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        auc_meter.update(preds, labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    return {
        'loss': total_loss / len(dataloader),
        'auc': auc_meter.compute(),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    trie_encoder=None,
    sparse_fields=None,
    vocab_maps=None,
) -> dict:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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


def main():
    parser = argparse.ArgumentParser(description='Train on Amazon Encrypted')
    parser.add_argument('--config', type=str, default='configs/amazon.yaml')
    parser.add_argument('--model', type=str, default='both',
                       choices=['deepfm', 'trie_moe', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    config['training']['epochs'] = args.epochs

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading Amazon encrypted data...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=config['data'].get('sample_size'),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get feature dims
    feature_dims = train_dataset.get_feature_dims()
    print(f"Feature dims: {feature_dims}")

    sparse_fields = train_dataset.actual_sparse_cols
    results = {}

    # Train DeepFM baseline
    if args.model in ['deepfm', 'both']:
        print("\n" + "="*50)
        print("Training DeepFM Baseline")
        print("="*50)

        model = DeepFM(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=config['model']['embedding_dim'],
            hidden_dims=config['model']['backbone']['hidden_dims'],
            dropout=config['model']['backbone']['dropout'],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        for epoch in range(config['training']['epochs']):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} - Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        test_metrics = evaluate(model, test_loader, device)
        results['DeepFM'] = test_metrics
        print(f"DeepFM Test - AUC: {test_metrics['auc']:.4f}, LogLoss: {test_metrics['logloss']:.4f}")

    # Train Trie-MoE
    if args.model in ['trie_moe', 'both']:
        print("\n" + "="*50)
        print("Training Trie-MoE")
        print("="*50)

        # Build Trie
        trie_builder = build_trie_from_amazon_dataset(train_dataset, sparse_fields, config)

        # Create vocab maps
        vocab_maps = {}
        for field, vocab in train_dataset.vocab.items():
            vocab_maps[field] = {idx: token for token, idx in vocab.items()}

        # Create Trie encoder
        trie_encoder = TrieEncoder(
            tries=trie_builder.tries,
            routing_dim=config['model']['moe'].get('routing_dim', 32),
            aggregation='mean',
            learnable_projection=True,
        ).to(device)

        model = TrieMoERecommender(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=config['model']['embedding_dim'],
            trie_routing_dim=trie_encoder.routing_dim,
            num_experts=config['model']['trie']['num_experts'],
            expert_hidden_dim=config['model']['moe']['expert_hidden_dim'],
            expert_output_dim=config['model']['moe']['expert_output_dim'],
            top_k=config['model']['moe']['top_k'],
            hidden_dims=config['model']['backbone']['hidden_dims'],
            dropout=config['model']['backbone']['dropout'],
        ).to(device)

        optimizer = optim.Adam(
            list(model.parameters()) + list(trie_encoder.parameters()),
            lr=config['training']['learning_rate']
        )

        for epoch in range(config['training']['epochs']):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device,
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
            print(f"Epoch {epoch+1} - Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        test_metrics = evaluate(
            model, test_loader, device,
            trie_encoder=trie_encoder,
            sparse_fields=sparse_fields,
            vocab_maps=vocab_maps
        )
        results['Trie-MoE'] = test_metrics
        print(f"Trie-MoE Test - AUC: {test_metrics['auc']:.4f}, LogLoss: {test_metrics['logloss']:.4f}")

    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY (Amazon Encrypted)")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name}: AUC={metrics['auc']:.4f}, LogLoss={metrics['logloss']:.4f}")


if __name__ == '__main__':
    main()
