"""
Fusion Models: Combining DCNv2/FinalMLP backbone with cold-start improvements.

Key insight from experiments:
- DCNv2/FinalMLP excel at cold-start (0.761 cold AUC) due to explicit feature crossing
- SimilarityPriorMoE provides meaningful cold-start prior (0.738 cold AUC)
- Hypothesis: Combining explicit crossing with similarity prior may yield further gains

Three fusion approaches:
1. DCNv2WithFreqAware: Frequency-aware embedding adjustment + DCNv2 backbone
2. DCNv2WithSimilarityPrior: DCNv2 + similarity-based output ensemble
3. FinalMLPWithFreqAware: FinalMLP + frequency-aware gating

Author: Anonymous
Date: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FrequencyAwareEmbedding(nn.Module):
    """
    Embedding layer with frequency-aware adjustment.

    For cold tokens (low frequency), applies learnable transformation:
    embedding' = embedding * (1 - α) + global_prior * α

    Where α = sigmoid(w * log(freq+1) + b) - higher for cold tokens
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 alpha_init_w: float = -0.5, alpha_init_b: float = 0.5):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim

        # Global prior (learned average embedding for cold tokens)
        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))

        # Adaptive alpha parameters
        self.alpha_w = nn.Parameter(torch.tensor(alpha_init_w))
        self.alpha_b = nn.Parameter(torch.tensor(alpha_init_b))

        # Frequency-conditioned transformation
        self.freq_transform = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),
            nn.Tanh(),
        )

    def compute_alpha(self, freqs: torch.Tensor) -> torch.Tensor:
        """Compute mixing weight α based on frequency."""
        log_freq = torch.log(freqs.float() + 1)
        return torch.sigmoid(self.alpha_w * log_freq + self.alpha_b)

    def forward(self, indices: torch.Tensor, freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            indices: [batch] token indices
            freqs: [batch] token frequencies (optional)
        Returns:
            embeddings: [batch, embedding_dim]
        """
        emb = self.embedding(indices)

        if freqs is None:
            return emb

        alpha = self.compute_alpha(freqs).unsqueeze(-1)  # [batch, 1]

        # Mix with global prior based on frequency
        adjusted = emb * (1 - alpha) + self.global_prior * alpha

        return adjusted


class CrossNetV2(nn.Module):
    """Cross Network V2 with low-rank matrix decomposition."""

    def __init__(self, input_dim: int, num_layers: int = 3, low_rank: int = 32):
        super().__init__()
        self.num_layers = num_layers

        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.num_layers):
            v_out = torch.matmul(x, self.V[i])
            uv_out = torch.matmul(v_out, self.U[i].T)
            x = x0 * (uv_out + self.bias[i]) + x
        return x


class DCNv2WithFreqAware(nn.Module):
    """
    DCNv2 with Frequency-Aware Embeddings.

    Combines:
    - Frequency-aware embedding adjustment (helps cold tokens)
    - Cross Network V2 (captures feature interactions)
    - Deep Network (non-linear transformations)
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        cross_num_layers: int = 3,
        cross_low_rank: int = 32,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim

        # Frequency-aware embeddings
        self.embeddings = nn.ModuleDict({
            name: FrequencyAwareEmbedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Cross Network V2
        self.cross_net = CrossNetV2(
            input_dim=input_dim,
            num_layers=cross_num_layers,
            low_rank=cross_low_rank,
        )

        # Deep Network
        layers = []
        prev_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim + deep_hidden_dims[-1], 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None,
                field_freqs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        # Default frequency
        if token_freqs is None:
            token_freqs = torch.ones(batch_size, device=device)

        # Normalize dense
        dense = self.dense_bn(dense)

        # Frequency-aware sparse embeddings
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i], token_freqs)
            sparse_embs.append(emb)

        sparse_flat = torch.cat(sparse_embs, dim=1)

        # Concatenate features
        x0 = torch.cat([dense, sparse_flat], dim=1)

        # Cross Network
        cross_out = self.cross_net(x0)

        # Deep Network
        deep_out = self.deep_net(x0)

        # Combine
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output_layer(combined).squeeze(-1)

        return torch.sigmoid(logits)


class SimilarityEnsemble(nn.Module):
    """
    Ensemble output based on similarity to warm examples.

    For cold samples, blend DCNv2 prediction with neighbors' predictions.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Project features for similarity computation
        self.sim_proj = nn.Linear(feature_dim, hidden_dim)

        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Blending weight based on frequency
        self.alpha_w = nn.Parameter(torch.tensor(-0.5))
        self.alpha_b = nn.Parameter(torch.tensor(0.3))

        # Cache for warm examples
        self.register_buffer('cache_features', torch.zeros(5000, hidden_dim))
        self.register_buffer('cache_preds', torch.zeros(5000))
        self.register_buffer('cache_count', torch.zeros(1, dtype=torch.long))

    def compute_alpha(self, freqs: torch.Tensor) -> torch.Tensor:
        log_freq = torch.log(freqs.float() + 1)
        return torch.sigmoid(self.alpha_w * log_freq + self.alpha_b)

    def update_cache(self, features: torch.Tensor, preds: torch.Tensor, freqs: torch.Tensor):
        """Update cache with warm examples during training."""
        if not self.training:
            return

        warm_mask = freqs > 10
        if not warm_mask.any():
            return

        warm_features = self.sim_proj(features[warm_mask]).detach()
        warm_preds = preds[warm_mask].detach()

        num_warm = warm_features.shape[0]
        start_idx = self.cache_count.item() % self.cache_features.shape[0]
        end_idx = min(start_idx + num_warm, self.cache_features.shape[0])
        actual_num = end_idx - start_idx

        self.cache_features[start_idx:end_idx] = warm_features[:actual_num]
        self.cache_preds[start_idx:end_idx] = warm_preds[:actual_num]
        self.cache_count += actual_num

    def forward(self, features: torch.Tensor, base_pred: torch.Tensor,
                freqs: torch.Tensor) -> torch.Tensor:
        """
        Blend base prediction with similarity-based prediction for cold samples.
        """
        alpha = self.compute_alpha(freqs)

        if self.cache_count.item() == 0:
            return base_pred

        # Project features
        proj = self.sim_proj(features)
        proj_norm = F.normalize(proj, dim=-1)

        # Find similar cached examples
        cache_size = min(self.cache_count.item(), self.cache_features.shape[0])
        cache_norm = F.normalize(self.cache_features[:cache_size], dim=-1)

        similarities = torch.mm(proj_norm, cache_norm.t())  # [batch, cache_size]

        # Top-k neighbors
        top_k = min(10, cache_size)
        top_sims, top_idx = torch.topk(similarities, top_k, dim=-1)
        top_sims = F.softmax(top_sims / self.temperature.clamp(min=0.1), dim=-1)

        # Neighbor predictions
        neighbor_preds = self.cache_preds[top_idx]  # [batch, k]
        sim_pred = (top_sims * neighbor_preds).sum(dim=-1)

        # Update cache during training
        self.update_cache(features, base_pred, freqs)

        # Blend: cold samples use more similarity-based prediction
        final_pred = alpha * sim_pred + (1 - alpha) * base_pred

        return final_pred


class DCNv2WithSimilarityEnsemble(nn.Module):
    """
    DCNv2 + Similarity-based output ensemble.

    For cold samples, blends DCNv2 prediction with neighbors' predictions.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        cross_num_layers: int = 3,
        cross_low_rank: int = 32,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim

        # Standard embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Cross Network V2
        self.cross_net = CrossNetV2(
            input_dim=input_dim,
            num_layers=cross_num_layers,
            low_rank=cross_low_rank,
        )

        # Deep Network
        layers = []
        prev_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*layers)

        # Pre-output layer
        self.pre_output = nn.Linear(input_dim + deep_hidden_dims[-1], 64)

        # Similarity ensemble
        self.sim_ensemble = SimilarityEnsemble(feature_dim=64, hidden_dim=32)

        # Final output
        self.output_layer = nn.Linear(64, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        if token_freqs is None:
            token_freqs = torch.ones(batch_size, device=device)

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]
        sparse_flat = torch.cat(sparse_embs, dim=1)

        # Concatenate features
        x0 = torch.cat([dense, sparse_flat], dim=1)

        # Cross + Deep
        cross_out = self.cross_net(x0)
        deep_out = self.deep_net(x0)

        combined = torch.cat([cross_out, deep_out], dim=1)

        # Pre-output features
        features = F.relu(self.pre_output(combined))

        # Base prediction
        base_logits = self.output_layer(features).squeeze(-1)
        base_pred = torch.sigmoid(base_logits)

        # Similarity ensemble
        final_pred = self.sim_ensemble(features, base_pred, token_freqs)

        return final_pred


class FinalMLPWithFreqAware(nn.Module):
    """
    FinalMLP with Frequency-Aware Feature Gate.

    Extends FinalMLP's feature gate to consider token frequency.
    Cold tokens get different gating than warm tokens.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        stream1_dims: List[int] = [256, 128],
        stream2_dims: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Frequency-aware feature gate
        self.freq_gate = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # +1 for frequency
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

        # Stream 1
        self.stream1 = self._build_stream(input_dim, stream1_dims, dropout)

        # Stream 2
        self.stream2 = self._build_stream(input_dim, stream2_dims, dropout)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(stream1_dims[-1] + stream2_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _build_stream(self, input_dim, hidden_dims, dropout):
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
        return nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        if token_freqs is None:
            token_freqs = torch.ones(batch_size, device=device)

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]
        sparse_flat = torch.cat(sparse_embs, dim=1)

        # Concatenate features
        x = torch.cat([dense, sparse_flat], dim=1)

        # Frequency-aware gating
        log_freq = torch.log(token_freqs.float() + 1).unsqueeze(-1)
        gate_input = torch.cat([x, log_freq], dim=1)
        gate = self.freq_gate(gate_input)
        x = x * gate

        # Two streams
        stream1_out = self.stream1(x)
        stream2_out = self.stream2(x)

        # Fusion
        combined = torch.cat([stream1_out, stream2_out], dim=1)
        logits = self.fusion(combined).squeeze(-1)

        return torch.sigmoid(logits)


def create_fusion_model(
    model_name: str,
    dense_dim: int,
    sparse_dims: Dict[str, int],
    embedding_dim: int = 16,
    **kwargs
) -> nn.Module:
    """Create fusion model by name."""

    if model_name == 'DCNv2FreqAware':
        return DCNv2WithFreqAware(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
        )
    elif model_name == 'DCNv2SimEnsemble':
        return DCNv2WithSimilarityEnsemble(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
        )
    elif model_name == 'FinalMLPFreqAware':
        return FinalMLPWithFreqAware(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
        )
    else:
        raise ValueError(f"Unknown fusion model: {model_name}")
