"""
Amazon Reviews dataset loader with support for both plaintext and encrypted modes.

The Amazon dataset serves as:
1. Plaintext baseline (with full text/title features)
2. Encrypted comparison (using only hashed IDs)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import hashlib


class AmazonDataset(Dataset):
    """
    PyTorch Dataset for Amazon Reviews.

    Supports two modes:
    - plaintext: Uses text features (for baseline/interpretability)
    - encrypted: Uses only ID features (simulates encrypted scenario)
    """

    def __init__(
        self,
        reviews_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame] = None,
        mode: str = "encrypted",  # "plaintext" or "encrypted"
        vocab: Optional[Dict[str, Dict]] = None,
        build_vocab: bool = True,
        max_vocab_size: int = 500000,
        min_freq: int = 3,
        rating_threshold: float = 4.0,  # For binary CTR-like task
    ):
        """
        Args:
            reviews_df: Reviews dataframe with user_id, asin, rating, etc.
            meta_df: Product metadata (optional, for plaintext mode)
            mode: "plaintext" or "encrypted"
            vocab: Pre-built vocabulary
            build_vocab: Whether to build vocab from data
            max_vocab_size: Maximum vocabulary size per field
            min_freq: Minimum frequency threshold
            rating_threshold: Threshold for positive label (rating >= threshold)
        """
        self.mode = mode
        self.rating_threshold = rating_threshold
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Prepare data
        self.data = self._prepare_data(reviews_df, meta_df)

        # Define feature columns based on mode
        if mode == "encrypted":
            self.sparse_cols = ['user_id', 'asin', 'parent_asin']
            self.dense_cols = ['helpful_vote', 'verified_purchase_int']
        else:  # plaintext - would include text features
            self.sparse_cols = ['user_id', 'asin', 'parent_asin', 'main_category']
            self.dense_cols = ['helpful_vote', 'verified_purchase_int', 'average_rating']

        # Build or use provided vocabulary
        if vocab is None and build_vocab:
            self.vocab, self.freq_stats = self._build_vocab()
        else:
            self.vocab = vocab or {}
            self.freq_stats = {}

        # Preprocess features
        self._preprocess()

    def _prepare_data(
        self,
        reviews_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Prepare and merge data."""
        df = reviews_df.copy()

        # Create binary label
        df['label'] = (df['rating'] >= self.rating_threshold).astype(int)

        # Convert boolean to int
        df['verified_purchase_int'] = df['verified_purchase'].astype(int)

        # Merge with metadata if available
        if meta_df is not None and self.mode == "plaintext":
            meta_cols = ['parent_asin', 'main_category', 'average_rating']
            meta_subset = meta_df[meta_cols].drop_duplicates(subset=['parent_asin'])
            df = df.merge(meta_subset, on='parent_asin', how='left')

        return df

    def _build_vocab(self) -> Tuple[Dict[str, Dict], Dict[str, Counter]]:
        """Build vocabulary for sparse features."""
        vocab = {}
        freq_stats = {}

        for col in self.sparse_cols:
            if col not in self.data.columns:
                continue

            counter = Counter(self.data[col].fillna('__MISSING__').astype(str))
            freq_stats[col] = counter

            sorted_items = counter.most_common(self.max_vocab_size)
            col_vocab = {'__PAD__': 0, '__UNK__': 1, '__MISSING__': 2}

            for token, freq in sorted_items:
                if freq >= self.min_freq and token not in col_vocab:
                    col_vocab[token] = len(col_vocab)

            vocab[col] = col_vocab

        return vocab, freq_stats

    def _preprocess(self):
        """Preprocess all features."""
        # Dense features
        self.dense_data = np.zeros((len(self.data), len(self.dense_cols)), dtype=np.float32)
        for i, col in enumerate(self.dense_cols):
            if col in self.data.columns:
                values = self.data[col].fillna(0).values.astype(np.float32)
                values = np.log1p(np.maximum(values, 0))
                mean, std = values.mean(), values.std() + 1e-8
                self.dense_data[:, i] = (values - mean) / std

        # Sparse features
        actual_sparse_cols = [c for c in self.sparse_cols if c in self.data.columns]
        self.sparse_data = np.zeros((len(self.data), len(actual_sparse_cols)), dtype=np.int64)

        for i, col in enumerate(actual_sparse_cols):
            col_vocab = self.vocab.get(col, {})
            unk_idx = col_vocab.get('__UNK__', 1)
            values = self.data[col].fillna('__MISSING__').astype(str)
            self.sparse_data[:, i] = values.map(lambda x: col_vocab.get(x, unk_idx)).values

        # Labels
        self.labels = self.data['label'].values.astype(np.float32)

        # Store actual sparse columns
        self.actual_sparse_cols = actual_sparse_cols

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
            'dense_dim': len(self.dense_cols),
            'sparse_dims': {col: len(self.vocab.get(col, {})) for col in self.actual_sparse_cols},
            'num_sparse_fields': len(self.actual_sparse_cols),
        }

    def get_token_frequencies(self, col: str) -> Dict[str, int]:
        """Get token frequencies for Trie construction."""
        return dict(self.freq_stats.get(col, {}))


def load_amazon_data(
    reviews_path: Union[str, Path],
    meta_path: Optional[Union[str, Path]] = None,
    mode: str = "encrypted",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[AmazonDataset, AmazonDataset, AmazonDataset]:
    """
    Load and split Amazon dataset.

    Args:
        reviews_path: Path to reviews parquet file
        meta_path: Path to metadata parquet file (optional)
        mode: "plaintext" or "encrypted"
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        sample_size: Optional sampling for development
        random_seed: Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    reviews_df = pd.read_parquet(reviews_path)
    meta_df = pd.read_parquet(meta_path) if meta_path else None

    # Sample if requested
    if sample_size and sample_size < len(reviews_df):
        reviews_df = reviews_df.sample(n=sample_size, random_state=random_seed)

    # Shuffle and split
    reviews_df = reviews_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n = len(reviews_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = reviews_df.iloc[:train_end]
    val_df = reviews_df.iloc[train_end:val_end]
    test_df = reviews_df.iloc[val_end:]

    # Build vocab on training data
    train_dataset = AmazonDataset(train_df, meta_df, mode=mode, build_vocab=True)

    val_dataset = AmazonDataset(
        val_df, meta_df, mode=mode,
        vocab=train_dataset.vocab, build_vocab=False
    )
    test_dataset = AmazonDataset(
        test_df, meta_df, mode=mode,
        vocab=train_dataset.vocab, build_vocab=False
    )

    return train_dataset, val_dataset, test_dataset
