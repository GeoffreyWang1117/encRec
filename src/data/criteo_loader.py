"""
Criteo CTR dataset loader.

The Criteo dataset contains:
- Label: Click/no-click (binary)
- I1-I13: 13 integer features (mostly count features)
- C1-C26: 26 categorical features (hashed to 32-bit integers)

This is the ideal dataset for studying encrypted/hashed feature recommendation
as the categorical features are completely opaque with no semantic meaning.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import hashlib


# Column names for Criteo dataset
DENSE_COLS = [f'I{i}' for i in range(1, 14)]  # I1-I13
SPARSE_COLS = [f'C{i}' for i in range(1, 27)]  # C1-C26
LABEL_COL = 'label'


class CriteoDataset(Dataset):
    """
    PyTorch Dataset for Criteo CTR prediction.

    Handles:
    - Dense feature normalization
    - Sparse feature vocabulary building
    - Frequency statistics for Trie construction
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vocab: Optional[Dict[str, Dict]] = None,
        freq_stats: Optional[Dict[str, Counter]] = None,
        build_vocab: bool = True,
        max_vocab_size: int = 1000000,
        min_freq: int = 5,
    ):
        self.data = data.reset_index(drop=True)
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Build or use provided vocabulary
        if vocab is None and build_vocab:
            self.vocab, self.freq_stats = self._build_vocab()
        else:
            self.vocab = vocab or {}
            self.freq_stats = freq_stats or {}

        # Preprocess dense features
        self._preprocess_dense()

        # Encode sparse features
        self._encode_sparse()

    def _build_vocab(self) -> Tuple[Dict[str, Dict], Dict[str, Counter]]:
        """Build vocabulary and frequency statistics for sparse features."""
        vocab = {}
        freq_stats = {}

        for col in SPARSE_COLS:
            # Count frequencies
            counter = Counter(self.data[col].fillna('__MISSING__').astype(str))
            freq_stats[col] = counter

            # Build vocab (filter by min_freq, limit by max_vocab_size)
            sorted_items = counter.most_common(self.max_vocab_size)
            col_vocab = {'__PAD__': 0, '__UNK__': 1, '__MISSING__': 2}

            for token, freq in sorted_items:
                if freq >= self.min_freq and token not in col_vocab:
                    col_vocab[token] = len(col_vocab)

            vocab[col] = col_vocab

        return vocab, freq_stats

    def _preprocess_dense(self):
        """Normalize dense features with log transformation."""
        self.dense_data = np.zeros((len(self.data), len(DENSE_COLS)), dtype=np.float32)

        for i, col in enumerate(DENSE_COLS):
            values = self.data[col].fillna(0).values.astype(np.float32)
            # Log transform for count features (common in CTR)
            values = np.log1p(np.maximum(values, 0))
            # Z-score normalization
            mean, std = values.mean(), values.std() + 1e-8
            self.dense_data[:, i] = (values - mean) / std

    def _encode_sparse(self):
        """Encode sparse features to indices."""
        self.sparse_data = np.zeros((len(self.data), len(SPARSE_COLS)), dtype=np.int64)

        for i, col in enumerate(SPARSE_COLS):
            col_vocab = self.vocab.get(col, {})
            unk_idx = col_vocab.get('__UNK__', 1)

            values = self.data[col].fillna('__MISSING__').astype(str)
            self.sparse_data[:, i] = values.map(
                lambda x: col_vocab.get(x, unk_idx)
            ).values

        # Store labels
        self.labels = self.data[LABEL_COL].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'dense': torch.from_numpy(self.dense_data[idx]),
            'sparse': torch.from_numpy(self.sparse_data[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for model construction."""
        return {
            'dense_dim': len(DENSE_COLS),
            'sparse_dims': {col: len(self.vocab[col]) for col in SPARSE_COLS},
            'num_sparse_fields': len(SPARSE_COLS),
        }

    def get_token_frequencies(self, col: str) -> Dict[str, int]:
        """Get token frequencies for a sparse column (for Trie construction)."""
        return dict(self.freq_stats.get(col, {}))

    def get_all_frequencies(self) -> Dict[str, Dict[str, int]]:
        """Get all token frequencies."""
        return {col: dict(counter) for col, counter in self.freq_stats.items()}


def load_criteo_data(
    data_path: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[CriteoDataset, CriteoDataset, CriteoDataset]:
    """
    Load and split Criteo dataset.

    Args:
        data_path: Path to parquet file
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        sample_size: Optional sampling for development
        random_seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    data_path = Path(data_path)

    # Load data
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)

    # Shuffle and split
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Build vocab on training data only
    train_dataset = CriteoDataset(train_df, build_vocab=True)

    # Use same vocab for val/test
    val_dataset = CriteoDataset(
        val_df,
        vocab=train_dataset.vocab,
        freq_stats=train_dataset.freq_stats,
        build_vocab=False
    )
    test_dataset = CriteoDataset(
        test_df,
        vocab=train_dataset.vocab,
        freq_stats=train_dataset.freq_stats,
        build_vocab=False
    )

    return train_dataset, val_dataset, test_dataset


def create_criteo_dataloaders(
    train_dataset: CriteoDataset,
    val_dataset: CriteoDataset,
    test_dataset: CriteoDataset,
    batch_size: int = 1024,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
