"""
MovieLens dataset loader for CTR-style experiments.

Converts MovieLens ratings to binary classification (rating >= 4 is positive).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import Counter


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens."""

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        movies_df: Optional[pd.DataFrame] = None,
        vocab: Optional[Dict[str, Dict]] = None,
        build_vocab: bool = True,
        rating_threshold: float = 4.0,
    ):
        self.rating_threshold = rating_threshold

        # Prepare data
        self.data = self._prepare_data(ratings_df, users_df, movies_df)

        # Define feature columns
        self.sparse_cols = ['user_id', 'movie_id']
        self.dense_cols = []  # No dense features in basic MovieLens

        if users_df is not None:
            self.sparse_cols.extend(['gender', 'age', 'occupation'])

        if movies_df is not None:
            self.sparse_cols.append('genre')

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
        ratings_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame],
        movies_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Prepare and merge data."""
        df = ratings_df.copy()

        # Create binary label
        df['label'] = (df['rating'] >= self.rating_threshold).astype(int)

        # Merge with user info if available
        if users_df is not None:
            df = df.merge(users_df, on='user_id', how='left')

        # Merge with movie info if available
        if movies_df is not None:
            # Take first genre only for simplicity
            movies_df = movies_df.copy()
            movies_df['genre'] = movies_df['genres'].str.split('|').str[0]
            df = df.merge(movies_df[['movie_id', 'genre']], on='movie_id', how='left')

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

            col_vocab = {'__PAD__': 0, '__UNK__': 1, '__MISSING__': 2}

            for token, freq in counter.most_common():
                if token not in col_vocab:
                    col_vocab[token] = len(col_vocab)

            vocab[col] = col_vocab

        return vocab, freq_stats

    def _preprocess(self):
        """Preprocess all features."""
        # Dense features (empty for basic MovieLens)
        self.dense_data = np.zeros((len(self.data), max(1, len(self.dense_cols))), dtype=np.float32)

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

    def get_feature_dims(self) -> Dict:
        """Get feature dimensions for model construction."""
        return {
            'dense_dim': max(1, len(self.dense_cols)),
            'sparse_dims': {col: len(self.vocab.get(col, {})) for col in self.actual_sparse_cols},
            'num_sparse_fields': len(self.actual_sparse_cols),
        }


def load_movielens_data(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    use_side_info: bool = True,
) -> Tuple[MovieLensDataset, MovieLensDataset, MovieLensDataset]:
    """
    Load and split MovieLens dataset.

    Args:
        data_dir: Path to ml-1m directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        sample_size: Optional sampling for development
        random_seed: Random seed
        use_side_info: Whether to use user/movie side information

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    data_path = Path(data_dir)

    # Load ratings
    ratings_df = pd.read_csv(
        data_path / 'ratings.dat',
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    users_df = None
    movies_df = None

    if use_side_info:
        # Load users
        users_df = pd.read_csv(
            data_path / 'users.dat',
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            engine='python'
        )
        users_df = users_df.drop('zipcode', axis=1)

        # Load movies
        movies_df = pd.read_csv(
            data_path / 'movies.dat',
            sep='::',
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )

    # Sample if requested
    if sample_size and sample_size < len(ratings_df):
        ratings_df = ratings_df.sample(n=sample_size, random_state=random_seed)

    # Shuffle and split
    ratings_df = ratings_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n = len(ratings_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = ratings_df.iloc[:train_end]
    val_df = ratings_df.iloc[train_end:val_end]
    test_df = ratings_df.iloc[val_end:]

    # Build vocab on training data
    train_dataset = MovieLensDataset(
        train_df, users_df, movies_df, build_vocab=True
    )

    val_dataset = MovieLensDataset(
        val_df, users_df, movies_df,
        vocab=train_dataset.vocab, build_vocab=False
    )
    test_dataset = MovieLensDataset(
        test_df, users_df, movies_df,
        vocab=train_dataset.vocab, build_vocab=False
    )

    return train_dataset, val_dataset, test_dataset
