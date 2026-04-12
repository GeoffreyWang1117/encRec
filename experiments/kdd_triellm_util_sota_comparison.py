#!/usr/bin/env python3
"""
Comprehensive SOTA Comparison Experiments for RecSys 2026 Paper.

This script runs all baseline comparisons:
1. CTR Backbones: DeepFM, DLRM, DCNv2, AutoInt, FinalMLP
2. Cold-Start Methods: DropoutNet, MeLU, MMoE, PLE
3. Our Methods: AdaptiveAlpha, EnhancedAdaptive

Datasets: Amazon Electronics, Criteo, MovieLens-1M
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DeepFM, DLRM, DCNv2, AutoInt, FinalMLP, DCNv3,
    DropoutNet, MeLU, MMoE, PLE, WarmUp,
    AdaptiveAlphaRecommender, EnhancedAdaptiveRecommender,
)


def compute_token_frequencies(
    sparse_tensor: torch.Tensor,
    sparse_dims: Dict[str, int],
) -> torch.Tensor:
    """
    Compute token frequencies from sparse tensor.

    Args:
        sparse_tensor: Sparse features tensor (num_samples, num_fields)
        sparse_dims: Dict mapping field names to vocabulary sizes

    Returns:
        Frequency tensor (num_fields, max_vocab_size)
    """
    num_samples, num_fields = sparse_tensor.shape
    max_vocab = max(sparse_dims.values())

    # Initialize frequency tensor
    freq_tensor = torch.zeros(num_fields, max_vocab, dtype=torch.float32)

    # Count frequencies for each field
    for field_idx in range(num_fields):
        field_values = sparse_tensor[:, field_idx]
        for token_id in field_values:
            if token_id < max_vocab:
                freq_tensor[field_idx, token_id] += 1

    return freq_tensor


# ============================================
# Configuration
# ============================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'sota_comparison'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default hyperparameters
DEFAULT_CONFIG = {
    'embedding_dim': 16,
    'hidden_dims': [128, 64],
    'dropout': 0.1,
    'lr': 0.001,
    'batch_size': 1024,
    'epochs': 10,
    'early_stopping_patience': 3,
    'num_runs': 5,
    'cold_start_threshold': 5,
}


# ============================================
# Data Loading
# ============================================

def load_movielens(data_dir: str = None, sample_size: int = None) -> Dict:
    """Load MovieLens-1M dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'

    data_path = Path(data_dir) / 'ml-1m'

    # Load ratings
    ratings = pd.read_csv(
        data_path / 'ratings.dat',
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    # Load users
    users = pd.read_csv(
        data_path / 'users.dat',
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        engine='python'
    )

    # Load movies
    movies = pd.read_csv(
        data_path / 'movies.dat',
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )

    # Merge
    df = ratings.merge(users, on='user_id').merge(movies, on='movie_id')

    # Binary label
    df['label'] = (df['rating'] >= 4).astype(int)

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    # Encode categorical features
    sparse_fields = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    sparse_dims = {}

    for field in sparse_fields:
        df[field] = pd.Categorical(df[field]).codes
        sparse_dims[field] = df[field].nunique()

    # No dense features for MovieLens
    dense_dim = 1  # Placeholder

    return {
        'df': df,
        'sparse_fields': sparse_fields,
        'sparse_dims': sparse_dims,
        'dense_dim': dense_dim,
        'label_col': 'label',
    }


def load_amazon(data_dir: str = None, sample_size: int = None) -> Dict:
    """Load Amazon Electronics dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'amazon'

    # Check for preprocessed data (try parquet first, then csv)
    parquet_path = Path(data_dir) / 'electronics_encrypted.parquet'
    csv_path = Path(data_dir) / 'amazon_electronics.csv'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        # Map columns for parquet format
        df['label'] = (df['rating'] >= 4).astype(int)
        df['item_id'] = df['asin']
        df['category'] = df['parent_asin']
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Amazon dataset not found at {data_dir}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    # Sparse fields
    sparse_fields = ['user_id', 'item_id', 'category']
    sparse_dims = {}

    for field in sparse_fields:
        if field in df.columns:
            df[field] = pd.Categorical(df[field]).codes
            sparse_dims[field] = df[field].nunique()

    dense_dim = 1

    return {
        'df': df,
        'sparse_fields': sparse_fields,
        'sparse_dims': sparse_dims,
        'dense_dim': dense_dim,
        'label_col': 'label',
    }


def load_criteo(data_dir: str = None, sample_size: int = 100000) -> Dict:
    """Load Criteo dataset (sampled)."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'criteo'

    data_path = Path(data_dir)

    # Try parquet files first (sorted by preference)
    parquet_files = [
        'criteo_real_1m.parquet',
        'criteo_5m.parquet',
        'criteo_synthetic_1m.parquet',
        'criteo_synthetic_100k.parquet',
    ]

    df = None
    for pq_file in parquet_files:
        pq_path = data_path / pq_file
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            break

    if df is None:
        # Try CSV
        csv_path = data_path / 'criteo_sample.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, nrows=sample_size)
        else:
            # Try raw Criteo
            raw_path = data_path / 'train.txt'
            if raw_path.exists():
                cols = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
                df = pd.read_csv(raw_path, sep='\t', names=cols, nrows=sample_size)
            else:
                raise FileNotFoundError(f"Criteo dataset not found at {data_dir}")

    # Dense features (I1-I13)
    dense_cols = [f'I{i}' for i in range(1, 14)]
    for col in dense_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Sparse features (C1-C26)
    sparse_fields = [f'C{i}' for i in range(1, 27)]
    sparse_dims = {}

    max_vocab = 10000
    for field in sparse_fields:
        if field in df.columns:
            df[field] = df[field].fillna('missing')
            # Convert to categorical codes (as int32 to avoid overflow)
            codes = pd.Categorical(df[field]).codes.astype(np.int32)
            # Apply modulo to cap vocabulary and ensure valid indices
            df[field] = codes % max_vocab
            sparse_dims[field] = max_vocab

    return {
        'df': df,
        'sparse_fields': [f for f in sparse_fields if f in df.columns],
        'sparse_dims': sparse_dims,
        'dense_dim': len(dense_cols),
        'dense_cols': dense_cols,
        'label_col': 'label',
    }


def prepare_data(
    data: Dict,
    train_size: int = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Prepare train/val/test dataloaders."""

    df = data['df']
    sparse_fields = data['sparse_fields']
    label_col = data['label_col']
    dense_dim = data['dense_dim']

    # Sample if train_size specified
    if train_size and train_size < len(df):
        df = df.sample(n=train_size, random_state=42)

    # Split
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=42)

    def make_tensors(subset_df):
        # Dense features
        if 'dense_cols' in data and data['dense_cols']:
            dense = torch.tensor(
                subset_df[data['dense_cols']].values, dtype=torch.float32
            )
        else:
            dense = torch.zeros(len(subset_df), dense_dim, dtype=torch.float32)

        # Sparse features
        sparse = torch.tensor(
            subset_df[sparse_fields].values, dtype=torch.long
        )

        # Labels
        labels = torch.tensor(subset_df[label_col].values, dtype=torch.float32)

        return dense, sparse, labels

    train_dense, train_sparse, train_labels = make_tensors(train_df)
    val_dense, val_sparse, val_labels = make_tensors(val_df)
    test_dense, test_sparse, test_labels = make_tensors(test_df)

    # Compute token frequencies for cold-start analysis
    token_freqs = compute_token_frequencies(train_sparse, data['sparse_dims'])

    # Dataloaders
    batch_size = DEFAULT_CONFIG['batch_size']

    train_loader = DataLoader(
        TensorDataset(train_dense, train_sparse, train_labels),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_dense, val_sparse, val_labels),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(test_dense, test_sparse, test_labels),
        batch_size=batch_size
    )

    meta = {
        'sparse_dims': data['sparse_dims'],
        'dense_dim': dense_dim,
        'token_freqs': token_freqs,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
    }

    return train_loader, val_loader, test_loader, meta


# ============================================
# Wrapper Classes for Adaptive Models
# ============================================

class AdaptiveAlphaWrapper(nn.Module):
    """Wrapper for AdaptiveAlphaRecommender that handles token frequency computation."""

    def __init__(self, model: nn.Module, token_freqs: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer('token_freqs', token_freqs)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with automatic token frequency computation."""
        batch_size = sparse.shape[0]
        num_fields = sparse.shape[1]

        # Compute min frequency across fields for each sample
        batch_freqs = []
        for i in range(batch_size):
            min_freq = float('inf')
            for j in range(num_fields):
                token_idx = sparse[i, j].item()
                if token_idx < self.token_freqs.shape[1]:
                    freq = self.token_freqs[j, token_idx].item()
                    min_freq = min(min_freq, freq)
            batch_freqs.append(min_freq if min_freq != float('inf') else 1)

        token_freqs = torch.tensor(batch_freqs, device=sparse.device, dtype=torch.float32)
        return self.model(dense, sparse, token_freqs)


class EnhancedAdaptiveWrapper(nn.Module):
    """Wrapper for EnhancedAdaptiveRecommender that handles token frequency computation."""

    def __init__(self, model: nn.Module, token_freqs: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer('token_freqs', token_freqs)

        # Initialize data structures from frequency tensor
        freq_dict = {}
        for j in range(token_freqs.shape[1]):
            total = token_freqs[:, j].sum().item()
            if total > 0:
                freq_dict[j] = int(total)
        if freq_dict:
            self.model.init_data_structures(freq_dict)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with automatic token frequency computation."""
        batch_size = sparse.shape[0]
        num_fields = sparse.shape[1]

        # Compute min frequency across fields for each sample
        batch_freqs = []
        for i in range(batch_size):
            min_freq = float('inf')
            for j in range(num_fields):
                token_idx = sparse[i, j].item()
                if token_idx < self.token_freqs.shape[1]:
                    freq = self.token_freqs[j, token_idx].item()
                    min_freq = min(min_freq, freq)
            batch_freqs.append(min_freq if min_freq != float('inf') else 1)

        token_freqs = torch.tensor(batch_freqs, device=sparse.device, dtype=torch.float32)
        return self.model(dense, sparse, token_freqs)


# ============================================
# Model Factory
# ============================================

def create_model(
    model_name: str,
    dense_dim: int,
    sparse_dims: Dict[str, int],
    token_freqs: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Create model by name."""

    embedding_dim = kwargs.get('embedding_dim', DEFAULT_CONFIG['embedding_dim'])
    hidden_dims = kwargs.get('hidden_dims', DEFAULT_CONFIG['hidden_dims'])
    dropout = kwargs.get('dropout', DEFAULT_CONFIG['dropout'])

    common_args = {
        'dense_dim': dense_dim,
        'sparse_dims': sparse_dims,
        'embedding_dim': embedding_dim,
        'dropout': dropout,
    }

    if model_name == 'DeepFM':
        return DeepFM(**common_args, hidden_dims=hidden_dims)

    elif model_name == 'DLRM':
        return DLRM(**common_args, bottom_mlp_dims=hidden_dims, top_mlp_dims=hidden_dims)

    elif model_name == 'DCNv2':
        return DCNv2(**common_args, deep_hidden_dims=hidden_dims)

    elif model_name == 'AutoInt':
        return AutoInt(**common_args, deep_hidden_dims=hidden_dims)

    elif model_name == 'FinalMLP':
        return FinalMLP(**common_args, stream1_dims=hidden_dims, stream2_dims=hidden_dims)

    elif model_name == 'DCNv3':
        return DCNv3(**common_args, deep_hidden_dims=hidden_dims)

    elif model_name == 'DropoutNet':
        return DropoutNet(**common_args, hidden_dims=hidden_dims, input_dropout=0.5)

    elif model_name == 'MeLU':
        return MeLU(**common_args, hidden_dims=hidden_dims)

    elif model_name == 'MMoE':
        return MMoE(**common_args, num_experts=8, expert_hidden_dims=hidden_dims)

    elif model_name == 'PLE':
        return PLE(**common_args, num_shared_experts=4, num_specific_experts=4)

    elif model_name == 'WarmUp':
        return WarmUp(**common_args, hidden_dims=hidden_dims)

    elif model_name == 'AdaptiveAlpha':
        if token_freqs is None:
            raise ValueError("AdaptiveAlpha requires token_freqs")
        base_model = AdaptiveAlphaRecommender(
            **common_args,
            hidden_dims=hidden_dims,
            num_experts=8,
        )
        return AdaptiveAlphaWrapper(base_model, token_freqs)

    elif model_name == 'EnhancedAdaptive':
        if token_freqs is None:
            raise ValueError("EnhancedAdaptive requires token_freqs")
        base_model = EnhancedAdaptiveRecommender(
            **common_args,
            hidden_dims=hidden_dims,
            num_experts=8,
        )
        return EnhancedAdaptiveWrapper(base_model, token_freqs)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================
# Training and Evaluation
# ============================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0

    for dense, sparse, labels in train_loader:
        dense = dense.to(device)
        sparse = sparse.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(dense, sparse)
        logits = output['logits']

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Add auxiliary losses if available
        if 'aux_loss' in output:
            loss = loss + 0.01 * output['aux_loss']

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    token_freqs: Optional[torch.Tensor] = None,
    cold_threshold: int = 5,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    all_is_cold = []

    with torch.no_grad():
        for dense, sparse, labels in data_loader:
            dense = dense.to(device)
            sparse = sparse.to(device)

            output = model(dense, sparse)
            logits = output['logits']
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

            # Check cold-start
            if token_freqs is not None:
                # Check if any token in the sample is cold
                batch_cold = []
                for i in range(sparse.shape[0]):
                    is_cold = False
                    for j in range(sparse.shape[1]):
                        token_idx = sparse[i, j].item()
                        if token_idx < token_freqs.shape[1]:
                            freq = token_freqs[j, token_idx].item()
                            if freq <= cold_threshold:
                                is_cold = True
                                break
                    batch_cold.append(is_cold)
                all_is_cold.extend(batch_cold)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # Overall metrics
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, np.clip(preds, 1e-7, 1-1e-7))

    results = {
        'auc': auc,
        'logloss': logloss,
    }

    # Cold-start metrics
    if all_is_cold:
        is_cold = np.array(all_is_cold)
        if is_cold.sum() > 0:
            cold_preds = preds[is_cold]
            cold_labels = labels[is_cold]
            results['cold_auc'] = roc_auc_score(cold_labels, cold_preds) if len(np.unique(cold_labels)) > 1 else 0.5
            results['cold_logloss'] = log_loss(cold_labels, np.clip(cold_preds, 1e-7, 1-1e-7))
            results['cold_ratio'] = is_cold.mean()

    return results


def train_and_evaluate(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    meta: Dict,
    seed: int = 42,
) -> Dict:
    """Full training and evaluation pipeline."""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_model(
        model_name,
        dense_dim=meta['dense_dim'],
        sparse_dims=meta['sparse_dims'],
        token_freqs=meta.get('token_freqs'),
    )
    model = model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_CONFIG['lr'])

    # Training
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(DEFAULT_CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_results = evaluate(model, val_loader, DEVICE, meta.get('token_freqs'))

        if val_results['auc'] > best_val_auc:
            best_val_auc = val_results['auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_CONFIG['early_stopping_patience']:
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test evaluation
    test_results = evaluate(model, test_loader, DEVICE, meta.get('token_freqs'))

    return test_results


# ============================================
# Main Experiment
# ============================================

def run_experiment(
    dataset_name: str,
    model_names: List[str],
    train_sizes: List[int] = None,
    num_runs: int = 5,
) -> Dict:
    """Run full experiment on a dataset."""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    if dataset_name == 'movielens':
        data = load_movielens()
    elif dataset_name == 'amazon':
        data = load_amazon()
    elif dataset_name == 'criteo':
        data = load_criteo()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if train_sizes is None:
        train_sizes = [1000, 2000, 5000, 10000]

    results = defaultdict(lambda: defaultdict(list))

    for train_size in train_sizes:
        print(f"\nTrain size: {train_size}")
        print("-" * 40)

        # Prepare data
        train_loader, val_loader, test_loader, meta = prepare_data(
            data, train_size=train_size
        )

        for model_name in tqdm(model_names, desc="Models"):
            for run in range(num_runs):
                try:
                    run_results = train_and_evaluate(
                        model_name,
                        train_loader,
                        val_loader,
                        test_loader,
                        meta,
                        seed=42 + run,
                    )

                    for metric, value in run_results.items():
                        results[f"{model_name}_{train_size}"][metric].append(value)

                except Exception as e:
                    print(f"Error in {model_name}: {e}")
                    continue

    return dict(results)


def summarize_results(results: Dict) -> pd.DataFrame:
    """Summarize results into a DataFrame."""
    rows = []

    for key, metrics in results.items():
        parts = key.rsplit('_', 1)
        model_name = parts[0]
        train_size = int(parts[1])

        row = {
            'model': model_name,
            'train_size': train_size,
        }

        for metric, values in metrics.items():
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='SOTA Comparison Experiments')
    parser.add_argument('--dataset', type=str, default='movielens',
                        choices=['movielens', 'amazon', 'criteo'])
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to evaluate')
    parser.add_argument('--train-sizes', type=int, nargs='+', default=[1000, 2000, 5000],
                        help='Training sizes to evaluate')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # Default models
    if args.models is None:
        args.models = [
            # CTR Backbones
            'DeepFM', 'DCNv2', 'AutoInt', 'FinalMLP',
            # Cold-Start Methods
            'DropoutNet', 'MMoE', 'PLE',
            # Our Methods
            'AdaptiveAlpha', 'EnhancedAdaptive',
        ]

    print(f"Device: {DEVICE}")
    print(f"Models: {args.models}")
    print(f"Train sizes: {args.train_sizes}")

    # Run experiment
    results = run_experiment(
        dataset_name=args.dataset,
        model_names=args.models,
        train_sizes=args.train_sizes,
        num_runs=args.num_runs,
    )

    # Summarize
    df = summarize_results(results)
    print("\nResults Summary:")
    print(df.to_string())

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or f'{RESULTS_DIR}/{args.dataset}_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Also save raw results
    with open(output_file.replace('.csv', '_raw.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {k: {m: list(v) for m, v in metrics.items()}
                       for k, metrics in results.items()}
        json.dump(json_results, f, indent=2)


if __name__ == '__main__':
    main()
