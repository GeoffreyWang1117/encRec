#!/usr/bin/env python3
"""
Train baseline models (DeepFM, DLRM) without Trie-MoE.

This establishes performance baselines for comparison.
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

from src.data.criteo_loader import CriteoDataset, load_criteo_data, create_criteo_dataloaders
from src.models.backbone import DeepFM, DLRM, DenseBaseline, DCNv2, AutoInt, FinalMLP, DCNv3
from src.utils.metrics import compute_metrics, AUCMeter, LogLossMeter
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    auc_meter = AUCMeter()
    logloss_meter = LogLossMeter()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(dense, sparse)
        loss = model.compute_loss(outputs, labels)

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        auc_meter.update(preds, labels.cpu().numpy())
        logloss_meter.update(preds, labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    return {
        'loss': total_loss / len(dataloader),
        'auc': auc_meter.compute(),
        'logloss': logloss_meter.compute(),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
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
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--config', type=str, default='configs/criteo.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='deepfm',
                       choices=['deepfm', 'dlrm', 'dense', 'dcnv2', 'autoint', 'finalmlp', 'dcnv3'],
                       help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs from config')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
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
    backbone_config = model_config['backbone']

    if args.model == 'deepfm':
        model = DeepFM(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif args.model == 'dlrm':
        model = DLRM(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            dropout=backbone_config['dropout'],
        )
    elif args.model == 'dcnv2':
        model = DCNv2(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            cross_num_layers=backbone_config.get('cross_num_layers', 3),
            deep_hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif args.model == 'autoint':
        model = AutoInt(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            num_attention_layers=backbone_config.get('num_attention_layers', 3),
            num_heads=backbone_config.get('num_heads', 2),
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    elif args.model == 'finalmlp':
        model = FinalMLP(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            stream1_dims=backbone_config.get('stream1_dims', [256, 128]),
            stream2_dims=backbone_config.get('stream2_dims', [256, 128]),
            dropout=backbone_config['dropout'],
        )
    elif args.model == 'dcnv3':
        model = DCNv3(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            cross_num_layers=backbone_config.get('cross_num_layers', 3),
            deep_hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )
    else:
        model = DenseBaseline(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dims=backbone_config['hidden_dims'],
            dropout=backbone_config['dropout'],
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    train_config = config['training']
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['epochs'],
    )

    # Logger
    logger = ExperimentLogger(
        experiment_name=f"{config['experiment']['name']}_{args.model}_baseline",
        log_dir=config['experiment']['log_dir'],
    )
    logger.log_config(config)

    # Training loop
    best_auc = 0.0
    for epoch in range(train_config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{train_config['epochs']} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            gradient_clip=train_config['gradient_clip'],
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Log
        logger.log_metrics({
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'lr': scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            logger.log_model(model, name=f"best_model_{args.model}")

        scheduler.step()

    # Final test evaluation
    print("\n=== Test Evaluation ===")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test - AUC: {test_metrics['auc']:.4f}, LogLoss: {test_metrics['logloss']:.4f}")

    logger.log_metrics({'test_auc': test_metrics['auc'], 'test_logloss': test_metrics['logloss']})
    logger.finish()


if __name__ == '__main__':
    main()
