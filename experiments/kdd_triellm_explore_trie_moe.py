#!/usr/bin/env python3
"""
Train Trie-guided MoE model - the main contribution of this project.

This script:
1. Builds Trie structure from token statistics
2. Creates Trie-based routing vectors
3. Trains MoE model with Trie guidance
4. Evaluates on head/mid/tail buckets separately
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

from src.data.criteo_loader import (
    CriteoDataset, load_criteo_data, create_criteo_dataloaders,
    SPARSE_COLS, DENSE_COLS
)
from src.trie.builder import TrieBuilder
from src.trie.encoder import TrieEncoder
from src.trie.statistics import TrieStatistics
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics, AUCMeter, BucketedMetrics, ExpertUtilizationMeter
from src.utils.logger import ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with inheritance support."""
    from pathlib import Path

    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle defaults/inheritance
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

        # Merge current config on top
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


def build_trie_from_dataset(
    dataset: CriteoDataset,
    sparse_fields: list,
    config: dict,
) -> TrieBuilder:
    """Build Trie structure from dataset statistics."""
    print("Building Trie structure...")

    trie_config = config['model']['trie']

    builder = TrieBuilder(
        hierarchy_config=trie_config['hierarchy'],
        num_experts=trie_config['num_experts'],
        expert_strategy=trie_config['expert_strategy'],
    )

    # Collect statistics from dataset
    for idx in tqdm(range(len(dataset)), desc="Collecting statistics"):
        sample = dataset[idx]
        label = int(sample['label'].item())

        for i, field in enumerate(sparse_fields):
            token_idx = sample['sparse'][i].item()
            # Convert index back to token (for Trie)
            token = str(token_idx)
            builder.statistics.update(field, token, label)

    # Compute statistics
    builder.statistics.compute_statistics()

    # Build Trie for each field
    for field in sparse_fields:
        if field in builder.statistics.statistics:
            from src.trie.builder import StatisticalTrie
            trie = StatisticalTrie(
                field_name=field,
                hierarchy_config=trie_config['hierarchy']
            )
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(trie_config['num_experts'], trie_config['expert_strategy'])
            builder.tries[field] = trie

    print(f"Built Tries for {len(builder.tries)} fields")

    # Print summary
    for field, trie in builder.tries.items():
        summary = trie.summarize()
        print(f"  {field}: {summary['total_tokens']} tokens, {summary['leaf_nodes']} leaves")

    return builder


def compute_frequency_scores(
    dataset: CriteoDataset,
    trie_builder: TrieBuilder,
    sparse_fields: list,
) -> np.ndarray:
    """Compute frequency score for each sample (for bucketed evaluation)."""
    scores = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        freq_sum = 0.0
        count = 0

        for i, field in enumerate(sparse_fields):
            token_idx = sample['sparse'][i].item()
            token = str(token_idx)

            if field in trie_builder.tries:
                node = trie_builder.tries[field].get_node_for_token(token)
                if node:
                    freq_sum += node.avg_frequency
                    count += 1

        scores.append(freq_sum / max(count, 1))

    return np.array(scores)


def create_trie_routing_batch(
    sparse_indices: torch.Tensor,
    trie_encoder: TrieEncoder,
    sparse_fields: list,
    vocab_maps: dict,
) -> torch.Tensor:
    """Create Trie routing vectors for a batch."""
    return trie_encoder(sparse_indices, sparse_fields, vocab_maps)


def train_epoch(
    model: nn.Module,
    trie_encoder: TrieEncoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    sparse_fields: list,
    vocab_maps: dict,
    gradient_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_lb_loss = 0.0
    auc_meter = AUCMeter()
    expert_meter = ExpertUtilizationMeter(num_experts=8)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Create Trie routing vectors
        trie_routing = create_trie_routing_batch(
            sparse, trie_encoder, sparse_fields, vocab_maps
        ).to(device)

        # Forward pass
        outputs = model(
            dense, sparse, trie_routing,
            return_routing=True,
        )

        # Loss
        loss = model.compute_loss(outputs, labels)

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_lb_loss += outputs.get('load_balance_loss', torch.tensor(0.0)).item()

        preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        auc_meter.update(preds, labels.cpu().numpy())

        # Expert utilization
        if 'routing_info' in outputs:
            expert_indices = outputs['routing_info']['expert_indices'].cpu().numpy()
            expert_meter.update(expert_indices)

        pbar.set_postfix({'loss': loss.item()})

    expert_stats = expert_meter.compute()

    return {
        'loss': total_loss / len(dataloader),
        'lb_loss': total_lb_loss / len(dataloader),
        'auc': auc_meter.compute(),
        'expert_entropy': expert_stats['utilization_entropy'],
    }


def evaluate(
    model: nn.Module,
    trie_encoder: TrieEncoder,
    dataloader: DataLoader,
    device: torch.device,
    sparse_fields: list,
    vocab_maps: dict,
    frequency_scores: np.ndarray,
) -> dict:
    """Evaluate model with bucketed metrics."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_routing_info = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)

            # Trie routing
            trie_routing = create_trie_routing_batch(
                sparse, trie_encoder, sparse_fields, vocab_maps
            ).to(device)

            outputs = model(dense, sparse, trie_routing, return_routing=True)
            loss = model.compute_loss(outputs, labels)

            total_loss += loss.item()

            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # Collect routing info
            if 'routing_info' in outputs:
                all_routing_info['expert_indices'].extend(
                    outputs['routing_info']['expert_indices'].cpu().numpy().tolist()
                )

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get corresponding frequency scores
    eval_freq_scores = frequency_scores[:len(all_preds)]

    metrics = compute_metrics(all_preds, all_labels, eval_freq_scores)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Trie-guided MoE model')
    parser.add_argument('--config', type=str, default='configs/criteo.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs from config')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.epochs:
        config['training']['epochs'] = args.epochs

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_path = config['data']['path']
    sample_size = config['data'].get('sample_size')

    train_dataset, val_dataset, test_dataset = load_criteo_data(
        data_path,
        sample_size=sample_size,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Define sparse fields
    sparse_fields = config['model'].get('sparse_fields', SPARSE_COLS)

    # Build Trie structure
    trie_builder = build_trie_from_dataset(train_dataset, sparse_fields, config)

    # Build reverse vocab map (index -> token)
    vocab_maps = {}
    for field, vocab in train_dataset.vocab.items():
        vocab_maps[field] = {idx: token for token, idx in vocab.items()}

    # Create Trie encoder
    trie_encoder = TrieEncoder(
        tries=trie_builder.tries,
        routing_dim=config['model']['moe'].get('routing_dim', 32),
        aggregation='mean',
        learnable_projection=True,
    )

    # Compute frequency scores for bucketed evaluation
    print("Computing frequency scores...")
    val_freq_scores = compute_frequency_scores(val_dataset, trie_builder, sparse_fields)
    test_freq_scores = compute_frequency_scores(test_dataset, trie_builder, sparse_fields)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_criteo_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )

    # Get feature dimensions
    feature_dims = train_dataset.get_feature_dims()
    print(f"Feature dims: {feature_dims}")

    # Create model
    model_config = config['model']
    moe_config = model_config['moe']

    model = TrieMoERecommender(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=model_config['embedding_dim'],
        trie_routing_dim=trie_encoder.routing_dim,  # Use output dim, not input dim
        num_experts=model_config['trie']['num_experts'],
        expert_hidden_dim=moe_config['expert_hidden_dim'],
        expert_output_dim=moe_config['expert_output_dim'],
        top_k=moe_config['top_k'],
        routing_mode=moe_config['routing_mode'],
        hidden_dims=model_config['backbone']['hidden_dims'],
        dropout=model_config['backbone']['dropout'],
    )

    model = model.to(device)
    trie_encoder = trie_encoder.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    train_config = config['training']
    optimizer = optim.Adam(
        list(model.parameters()) + list(trie_encoder.parameters()),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['epochs'],
    )

    # Logger
    logger = ExperimentLogger(
        experiment_name=f"{config['experiment']['name']}_trie_moe",
        log_dir=config['experiment']['log_dir'],
    )
    logger.log_config(config)

    # Training loop
    best_auc = 0.0
    for epoch in range(train_config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{train_config['epochs']} ===")

        # Train
        train_metrics = train_epoch(
            model, trie_encoder, train_loader, optimizer, device,
            sparse_fields, vocab_maps,
            gradient_clip=train_config['gradient_clip'],
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
              f"LB Loss: {train_metrics['lb_loss']:.4f}, Expert Entropy: {train_metrics['expert_entropy']:.4f}")

        # Validate
        val_metrics = evaluate(
            model, trie_encoder, val_loader, device,
            sparse_fields, vocab_maps, val_freq_scores,
        )
        print(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Bucketed metrics
        if 'head_auc' in val_metrics:
            print(f"  Head AUC: {val_metrics['head_auc']:.4f}, "
                  f"Mid AUC: {val_metrics['mid_auc']:.4f}, "
                  f"Tail AUC: {val_metrics['tail_auc']:.4f}")

        # Log
        logger.log_metrics({
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'train_expert_entropy': train_metrics['expert_entropy'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_head_auc': val_metrics.get('head_auc', 0),
            'val_tail_auc': val_metrics.get('tail_auc', 0),
            'lr': scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            logger.log_model(model, name="best_trie_moe_model")

        scheduler.step()

    # Final test evaluation
    print("\n=== Test Evaluation ===")
    test_metrics = evaluate(
        model, trie_encoder, test_loader, device,
        sparse_fields, vocab_maps, test_freq_scores,
    )
    print(f"Test - AUC: {test_metrics['auc']:.4f}, LogLoss: {test_metrics['logloss']:.4f}")

    if 'head_auc' in test_metrics:
        print(f"  Head AUC: {test_metrics['head_auc']:.4f}, "
              f"Mid AUC: {test_metrics['mid_auc']:.4f}, "
              f"Tail AUC: {test_metrics['tail_auc']:.4f}")

    logger.log_metrics({
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'test_head_auc': test_metrics.get('head_auc', 0),
        'test_tail_auc': test_metrics.get('tail_auc', 0),
    })
    logger.finish()


if __name__ == '__main__':
    main()
