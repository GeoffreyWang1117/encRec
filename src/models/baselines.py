#!/usr/bin/env python3
"""
Baseline CTR Models for KDD Paper Comparison.

Implements:
- DeepFM (Guo et al., 2017)
- AutoInt (Song et al., 2019)
- FiBiNET (Huang et al., 2019)
- EDCN (Chen et al., 2021)
- MaskNet (Wang et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class DeepFM(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.
    Guo et al., IJCAI 2017.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Embeddings for FM and DNN
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # First-order weights for sparse features
        self.first_order_weights = nn.ModuleDict({
            name: nn.Embedding(dim, 1)
            for name, dim in sparse_dims.items()
        })

        # Dense feature processing
        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)
            self.dense_linear = nn.Linear(dense_dim, 1)

        # DNN part
        dnn_input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        layers = []
        prev_dim = dnn_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

        # Output bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        batch_size = sparse.shape[0]

        # First-order term
        first_order = torch.zeros(batch_size, 1, device=sparse.device)
        for i, name in enumerate(self.sparse_field_names):
            first_order += self.first_order_weights[name](sparse[:, i])

        if self.dense_dim > 0:
            dense = self.dense_bn(dense)
            first_order += self.dense_linear(dense)

        # Second-order FM term (sum of squares - square of sums)
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            if self.training and self.embedding_dropout > 0:
                emb = F.dropout(emb, p=self.embedding_dropout)
            sparse_embs.append(emb)

        emb_stack = torch.stack(sparse_embs, dim=1)  # (batch, num_fields, emb_dim)
        sum_of_embs = emb_stack.sum(dim=1)  # (batch, emb_dim)
        sum_of_squares = (emb_stack ** 2).sum(dim=1)  # (batch, emb_dim)
        second_order = 0.5 * (sum_of_embs ** 2 - sum_of_squares).sum(dim=1, keepdim=True)

        # DNN part
        sparse_flat = torch.cat(sparse_embs, dim=1)
        if self.dense_dim > 0:
            dnn_input = torch.cat([dense, sparse_flat], dim=1)
        else:
            dnn_input = sparse_flat
        dnn_out = self.dnn(dnn_input)

        # Combine
        logit = first_order + second_order + dnn_out + self.bias
        return torch.sigmoid(logit.squeeze(-1))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_fields, embed_dim)
        batch_size, num_fields, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_fields, self.embed_dim)

        return self.W_o(context)


class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.
    Song et al., CIKM 2019.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 3,
        attention_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)
            self.dense_proj = nn.Linear(dense_dim, embedding_dim)
            total_fields = self.num_sparse_fields + 1
        else:
            total_fields = self.num_sparse_fields

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.output_layer = nn.Linear(total_fields * embedding_dim, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        # Sparse embeddings
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            if self.training and self.embedding_dropout > 0:
                emb = F.dropout(emb, p=self.embedding_dropout)
            sparse_embs.append(emb)

        # Stack as field embeddings
        emb_stack = torch.stack(sparse_embs, dim=1)  # (batch, num_fields, emb_dim)

        # Add dense as another field
        if self.dense_dim > 0:
            dense = self.dense_bn(dense)
            dense_emb = self.dense_proj(dense).unsqueeze(1)  # (batch, 1, emb_dim)
            emb_stack = torch.cat([emb_stack, dense_emb], dim=1)

        # Self-attention layers
        x = emb_stack
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            x = norm(x + attn(x))

        # Flatten and output
        x = x.flatten(start_dim=1)
        return torch.sigmoid(self.output_layer(x).squeeze(-1))


class SENet(nn.Module):
    """Squeeze-and-Excitation Network for feature importance."""

    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.fc1 = nn.Linear(num_fields, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, num_fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_fields, emb_dim)
        # Squeeze: global average pooling
        z = x.mean(dim=2)  # (batch, num_fields)
        # Excitation
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        # Scale
        return x * z.unsqueeze(2)


class BilinearInteraction(nn.Module):
    """Bilinear feature interaction layer."""

    def __init__(self, embedding_dim: int, num_fields: int, bilinear_type: str = 'field_all'):
        super().__init__()
        self.bilinear_type = bilinear_type
        self.num_fields = num_fields

        if bilinear_type == 'field_all':
            self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        elif bilinear_type == 'field_each':
            self.W = nn.ParameterList([
                nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
                for _ in range(num_fields)
            ])
        elif bilinear_type == 'field_interaction':
            num_interactions = num_fields * (num_fields - 1) // 2
            self.W = nn.ParameterList([
                nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
                for _ in range(num_interactions)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_fields, emb_dim)
        interactions = []
        idx = 0
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                if self.bilinear_type == 'field_all':
                    v_i = torch.matmul(x[:, i, :], self.W)
                elif self.bilinear_type == 'field_each':
                    v_i = torch.matmul(x[:, i, :], self.W[i])
                elif self.bilinear_type == 'field_interaction':
                    v_i = torch.matmul(x[:, i, :], self.W[idx])
                    idx += 1
                interactions.append(v_i * x[:, j, :])

        return torch.stack(interactions, dim=1)  # (batch, num_interactions, emb_dim)


class FiBiNET(nn.Module):
    """
    FiBiNET: Combining Feature Importance and Bilinear feature Interaction.
    Huang et al., RecSys 2019.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        bilinear_type: str = 'field_all',
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        # SENET layer
        self.senet = SENet(self.num_sparse_fields)

        # Bilinear interaction layers
        self.bilinear_raw = BilinearInteraction(embedding_dim, self.num_sparse_fields, bilinear_type)
        self.bilinear_senet = BilinearInteraction(embedding_dim, self.num_sparse_fields, bilinear_type)

        # Calculate number of interactions
        num_interactions = self.num_sparse_fields * (self.num_sparse_fields - 1) // 2

        # DNN
        dnn_input_dim = dense_dim + 2 * num_interactions * embedding_dim

        layers = []
        prev_dim = dnn_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        # Sparse embeddings
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            if self.training and self.embedding_dropout > 0:
                emb = F.dropout(emb, p=self.embedding_dropout)
            sparse_embs.append(emb)

        emb_stack = torch.stack(sparse_embs, dim=1)  # (batch, num_fields, emb_dim)

        # SENET
        senet_emb = self.senet(emb_stack)

        # Bilinear interactions
        bilinear_raw = self.bilinear_raw(emb_stack)  # (batch, num_inter, emb_dim)
        bilinear_senet = self.bilinear_senet(senet_emb)

        # Flatten interactions
        bilinear_raw_flat = bilinear_raw.flatten(start_dim=1)
        bilinear_senet_flat = bilinear_senet.flatten(start_dim=1)

        # DNN input
        if self.dense_dim > 0:
            dense = self.dense_bn(dense)
            dnn_input = torch.cat([dense, bilinear_raw_flat, bilinear_senet_flat], dim=1)
        else:
            dnn_input = torch.cat([bilinear_raw_flat, bilinear_senet_flat], dim=1)

        return torch.sigmoid(self.dnn(dnn_input).squeeze(-1))


class MaskBlock(nn.Module):
    """Mask Block for MaskNet."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: current hidden, emb: original embedding
        mask = self.fc2(F.relu(self.fc1(self.ln(emb))))
        mask = self.dropout(mask)
        return x * mask


class MaskNet(nn.Module):
    """
    MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models.
    Wang et al., DLP-KDD 2021.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_blocks: int = 3,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Mask blocks
        self.mask_blocks = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.mask_blocks.append(MaskBlock(input_dim, input_dim // 2, prev_dim, dropout))
            self.fc_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        # Sparse embeddings
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            if self.training and self.embedding_dropout > 0:
                emb = F.dropout(emb, p=self.embedding_dropout)
            sparse_embs.append(emb)

        sparse_flat = torch.cat(sparse_embs, dim=1)

        if self.dense_dim > 0:
            dense = self.dense_bn(dense)
            x = torch.cat([dense, sparse_flat], dim=1)
        else:
            x = sparse_flat

        emb_input = x  # Save for mask blocks

        # Forward through mask blocks
        for mask_block, fc_layer in zip(self.mask_blocks, self.fc_layers):
            x = mask_block(x, emb_input)
            x = fc_layer(x)

        return torch.sigmoid(self.output_layer(x).squeeze(-1))


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline without explicit feature interactions.
    Used as a sanity check baseline.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Sparse embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        # MLP
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor, **kwargs) -> torch.Tensor:
        # Sparse embeddings
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            if self.training and self.embedding_dropout > 0:
                emb = F.dropout(emb, p=self.embedding_dropout)
            sparse_embs.append(emb)

        sparse_flat = torch.cat(sparse_embs, dim=1)

        if self.dense_dim > 0:
            dense = self.dense_bn(dense)
            x = torch.cat([dense, sparse_flat], dim=1)
        else:
            x = sparse_flat

        return self.mlp(x).squeeze(-1)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss."""
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs)
        else:
            logits = outputs
        return F.binary_cross_entropy_with_logits(logits, labels)


# Factory function
def get_model(
    model_name: str,
    dense_dim: int,
    sparse_dims: Dict[str, int],
    embedding_dim: int = 16,
    embedding_dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """Get model by name."""
    models = {
        'deepfm': DeepFM,
        'autoint': AutoInt,
        'fibinet': FiBiNET,
        'masknet': MaskNet,
        'simplemlp': SimpleMLP,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](
        dense_dim=dense_dim,
        sparse_dims=sparse_dims,
        embedding_dim=embedding_dim,
        embedding_dropout=embedding_dropout,
        **kwargs
    )
