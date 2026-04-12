"""
Feature encryption utilities for creating encrypted versions of plaintext datasets.

This module provides controlled encryption to simulate real-world scenarios where
features are hashed/encrypted for privacy or commercial protection.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path


class FeatureEncryptor:
    """
    Encrypt/hash features to simulate encrypted feature scenarios.

    Supports multiple encryption strategies:
    - hash: Simple hash (deterministic, preserves frequency)
    - salted_hash: Salted hash (deterministic per salt)
    - random_map: Random mapping (preserves structure, not frequency)
    """

    def __init__(
        self,
        strategy: str = "hash",
        salt: Optional[str] = None,
        hash_bits: int = 32,
        preserve_frequency: bool = True,
    ):
        """
        Args:
            strategy: "hash", "salted_hash", or "random_map"
            salt: Salt for salted_hash strategy
            hash_bits: Number of bits for hash output
            preserve_frequency: Whether to preserve token frequency distribution
        """
        self.strategy = strategy
        self.salt = salt or "encRec_default_salt"
        self.hash_bits = hash_bits
        self.preserve_frequency = preserve_frequency

        # For random_map strategy
        self._random_maps: Dict[str, Dict] = {}

    def _hash_value(self, value: str, field: Optional[str] = None) -> str:
        """Hash a single value."""
        if self.strategy == "salted_hash":
            to_hash = f"{self.salt}_{field}_{value}" if field else f"{self.salt}_{value}"
        else:
            to_hash = str(value)

        hash_obj = hashlib.sha256(to_hash.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        # Truncate to specified bits
        truncated = hash_int % (2 ** self.hash_bits)
        return f"ENC_{truncated:08x}"

    def _get_random_map(self, field: str, unique_values: List) -> Dict:
        """Get or create random mapping for a field."""
        if field not in self._random_maps:
            rng = np.random.RandomState(hash(field) % (2**32))
            encrypted_ids = [f"RND_{i:08x}" for i in rng.permutation(len(unique_values))]
            self._random_maps[field] = dict(zip(unique_values, encrypted_ids))
        return self._random_maps[field]

    def encrypt_column(
        self,
        data: pd.Series,
        field_name: str,
    ) -> pd.Series:
        """Encrypt a single column."""
        if self.strategy in ["hash", "salted_hash"]:
            return data.fillna("__NULL__").astype(str).apply(
                lambda x: self._hash_value(x, field_name)
            )
        elif self.strategy == "random_map":
            unique_vals = data.fillna("__NULL__").astype(str).unique().tolist()
            mapping = self._get_random_map(field_name, unique_vals)
            return data.fillna("__NULL__").astype(str).map(mapping)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def encrypt_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_plaintext: bool = True,
    ) -> pd.DataFrame:
        """
        Encrypt specified columns in a dataframe.

        Args:
            df: Input dataframe
            columns: Columns to encrypt
            drop_plaintext: Whether to drop original plaintext columns

        Returns:
            Dataframe with encrypted columns
        """
        result = df.copy()

        for col in columns:
            if col in df.columns:
                encrypted_col = f"{col}_enc"
                result[encrypted_col] = self.encrypt_column(df[col], col)

                if drop_plaintext:
                    result = result.drop(columns=[col])
                    result = result.rename(columns={encrypted_col: col})

        return result


def encrypt_amazon_dataset(
    reviews_path: Union[str, Path],
    output_path: Union[str, Path],
    meta_path: Optional[Union[str, Path]] = None,
    encryption_strategy: str = "hash",
    columns_to_encrypt: Optional[List[str]] = None,
    drop_text_columns: bool = True,
) -> pd.DataFrame:
    """
    Create an encrypted version of Amazon Reviews dataset.

    This simulates a real-world scenario where user/item IDs are hashed
    and text content is unavailable.

    Args:
        reviews_path: Path to original reviews parquet
        output_path: Path to save encrypted version
        meta_path: Optional metadata path
        encryption_strategy: Encryption strategy to use
        columns_to_encrypt: Columns to encrypt (default: user_id, asin, parent_asin)
        drop_text_columns: Whether to remove text columns (title, text)

    Returns:
        Encrypted dataframe
    """
    # Load data
    df = pd.read_parquet(reviews_path)

    # Default columns to encrypt
    if columns_to_encrypt is None:
        columns_to_encrypt = ['user_id', 'asin', 'parent_asin']

    # Initialize encryptor
    encryptor = FeatureEncryptor(strategy=encryption_strategy)

    # Encrypt ID columns
    df = encryptor.encrypt_dataframe(df, columns_to_encrypt)

    # Drop text columns if requested (simulating unavailable semantic info)
    if drop_text_columns:
        text_cols = ['title', 'text', 'images']
        cols_to_drop = [c for c in text_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    # Save encrypted version
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Encrypted dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


class EncryptionAnalyzer:
    """
    Analyze properties of encrypted features.

    Useful for understanding what statistical properties are preserved
    after encryption (frequency, co-occurrence, etc.).
    """

    @staticmethod
    def compare_frequency_distributions(
        original: pd.Series,
        encrypted: pd.Series,
    ) -> Dict[str, float]:
        """Compare frequency distributions before and after encryption."""
        from scipy.stats import spearmanr, entropy

        orig_freq = original.value_counts(normalize=True)
        enc_freq = encrypted.value_counts(normalize=True)

        # Check if frequency ranking is preserved
        orig_rank = orig_freq.rank(ascending=False)
        enc_rank = enc_freq.rank(ascending=False)

        # Align by original values (if mapping is known)
        # For hash encryption, frequencies should match exactly

        results = {
            'original_entropy': entropy(orig_freq),
            'encrypted_entropy': entropy(enc_freq),
            'original_unique': len(orig_freq),
            'encrypted_unique': len(enc_freq),
        }

        return results

    @staticmethod
    def analyze_cooccurrence_preservation(
        df_original: pd.DataFrame,
        df_encrypted: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> Dict[str, float]:
        """
        Check if co-occurrence patterns are preserved after encryption.

        This is crucial for Trie construction - we need co-occurrence
        structure to remain stable even without semantic meaning.
        """
        from scipy.stats import chi2_contingency

        # Build co-occurrence matrices
        def build_cooc_matrix(df, c1, c2):
            return pd.crosstab(df[c1], df[c2])

        orig_cooc = build_cooc_matrix(df_original, col1, col2)
        enc_cooc = build_cooc_matrix(df_encrypted, col1, col2)

        # Chi-square test for independence
        orig_chi2, orig_p, _, _ = chi2_contingency(orig_cooc)
        enc_chi2, enc_p, _, _ = chi2_contingency(enc_cooc)

        return {
            'original_chi2': orig_chi2,
            'encrypted_chi2': enc_chi2,
            'original_p_value': orig_p,
            'encrypted_p_value': enc_p,
            'structure_preserved': abs(orig_chi2 - enc_chi2) / orig_chi2 < 0.1,
        }
