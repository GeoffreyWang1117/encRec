"""
Base classes for recommendation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for sparse features with support for multiple fields.

    Handles:
    - Multiple sparse fields with different vocabulary sizes
    - Shared or separate embedding dimensions
    - Embedding aggregation strategies
    """

    def __init__(
        self,
        sparse_dims: Dict[str, int],  # field_name -> vocab_size
        embedding_dim: int = 16,
        use_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.sparse_dims = sparse_dims
        self.embedding_dim = embedding_dim
        self.field_names = list(sparse_dims.keys())
        self.num_fields = len(sparse_dims)

        # Create embedding table for each field
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0,  # Reserve 0 for padding
            )
            for field, vocab_size in sparse_dims.items()
        })

        # Initialize embeddings
        self._init_embeddings()

        # Load pretrained if available
        if use_pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight.data[1:])  # Skip padding
            emb.weight.data[0].zero_()  # Zero padding

    def _load_pretrained(self, path: str):
        """Load pretrained embeddings."""
        state_dict = torch.load(path)
        for field, emb in self.embeddings.items():
            if field in state_dict:
                emb.weight.data = state_dict[field]

    def forward(
        self,
        sparse_indices: torch.Tensor,  # (batch_size, num_fields)
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sparse_indices: (batch_size, num_fields) tensor of token indices

        Returns:
            (batch_size, num_fields, embedding_dim) tensor
        """
        batch_size = sparse_indices.shape[0]
        embeddings_list = []

        for i, field in enumerate(self.field_names):
            field_indices = sparse_indices[:, i]  # (batch_size,)
            field_emb = self.embeddings[field](field_indices)  # (batch_size, emb_dim)
            embeddings_list.append(field_emb)

        # Stack: (batch_size, num_fields, embedding_dim)
        return torch.stack(embeddings_list, dim=1)

    def get_flat_embeddings(
        self,
        sparse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get flattened embeddings (batch_size, num_fields * embedding_dim)."""
        emb = self.forward(sparse_indices)
        return emb.view(emb.shape[0], -1)


class BaseRecommender(nn.Module):
    """
    Base class for recommendation models.

    Defines the interface for:
    - Dense and sparse feature processing
    - Forward pass returning logits
    - Optional auxiliary outputs (e.g., expert assignments)
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        output_dim: int = 1,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_dims = sparse_dims
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_sparse_fields = len(sparse_dims)

        # Embedding layer
        self.embedding = EmbeddingLayer(sparse_dims, embedding_dim)

        # Dense feature processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

    def forward(
        self,
        dense: torch.Tensor,  # (batch_size, dense_dim)
        sparse: torch.Tensor,  # (batch_size, num_fields)
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            dense: Dense features
            sparse: Sparse feature indices

        Returns:
            Dict containing at least 'logits' key
        """
        raise NotImplementedError

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from model outputs."""
        logits = outputs['logits']
        return F.binary_cross_entropy_with_logits(logits.squeeze(), labels)


class MLPBlock(nn.Module):
    """Standard MLP block with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_norm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeatureInteraction(nn.Module):
    """
    Feature interaction layer (FM-style).

    Computes second-order feature interactions efficiently.
    """

    def __init__(self, reduce: bool = True):
        super().__init__()
        self.reduce = reduce

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, num_fields, embedding_dim)

        Returns:
            If reduce: (batch_size, embedding_dim)
            Else: (batch_size, num_fields * (num_fields - 1) / 2)
        """
        # FM interaction: (sum_squared - squared_sum) / 2
        sum_of_emb = embeddings.sum(dim=1)  # (batch, emb_dim)
        sum_squared = sum_of_emb ** 2

        squared_sum = (embeddings ** 2).sum(dim=1)

        interaction = 0.5 * (sum_squared - squared_sum)

        if self.reduce:
            return interaction  # (batch, emb_dim)
        else:
            return interaction.sum(dim=1, keepdim=True)  # (batch, 1)
