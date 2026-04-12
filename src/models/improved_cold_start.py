"""
Improved Cold-Start Models based on SOTA techniques.

Plan B: DropoutNet-style embedding dropout during training
Plan A: Similarity-based prior using neighbor embeddings
Plan C: FreqAwareNet - simplified frequency-aware architecture without MoE

Author: Anonymous
Date: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# Plan B: DropoutNet-style Training
# =============================================================================

class DropoutNetRecommender(nn.Module):
    """
    DropoutNet-style model that randomly drops embeddings during training
    to simulate cold-start and force the model to use other features.

    Key insight: By training with missing embeddings, the model learns to
    leverage cross-feature interactions even when some features are unreliable.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        embedding_dropout: float = 0.3,  # Key parameter: dropout rate for embeddings
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        self.dense_linear = nn.Linear(dense_dim, embedding_dim)

        # Feature interaction (FM-style)
        total_emb_dim = (self.num_sparse_fields + 1) * embedding_dim

        # DNN layers
        layers = []
        prev_dim = total_emb_dim
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

        # Content network (used when embeddings are dropped)
        # This learns to predict from remaining features
        self.content_net = nn.Sequential(
            nn.Linear(dense_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None,
                use_data_structures: bool = True) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        # Process dense features
        dense_normed = self.dense_bn(dense)
        dense_emb = self.dense_linear(dense_normed)

        # Process sparse features with dropout during training
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])

            # Apply embedding dropout during training
            if self.training and self.embedding_dropout > 0:
                # Randomly drop entire embeddings (not individual dimensions)
                mask = torch.bernoulli(
                    torch.ones(batch_size, 1, device=device) * (1 - self.embedding_dropout)
                )
                # When dropped, use content network output as fallback
                content_fallback = self.content_net(dense)
                emb = emb * mask + content_fallback * (1 - mask)

            sparse_embs.append(emb)

        # Concatenate all embeddings
        all_embs = torch.cat(sparse_embs + [dense_emb], dim=1)

        # DNN prediction
        logits = self.dnn(all_embs).squeeze(-1)

        return torch.sigmoid(logits)


# =============================================================================
# Plan A: Similarity-based Prior
# =============================================================================

class SimilarityPriorRouter(nn.Module):
    """
    Router that uses similarity-based prior instead of uniform.
    For cold-start tokens, use the routing decisions of similar warm tokens.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        hidden_dim: int = 32,
        alpha_init_w: float = -0.5,
        alpha_init_b: float = 0.5,
    ):
        super().__init__()

        self.num_experts = num_experts

        # Adaptive alpha parameters
        self.alpha_w = nn.Parameter(torch.tensor(alpha_init_w))
        self.alpha_b = nn.Parameter(torch.tensor(alpha_init_b))

        # Learned router
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Similarity projection for finding neighbors
        self.sim_proj = nn.Linear(input_dim, 64)

        # Cache for warm token routing (updated during training)
        self.register_buffer('routing_cache', torch.zeros(10000, num_experts))
        self.register_buffer('embedding_cache', torch.zeros(10000, 64))
        self.register_buffer('cache_count', torch.zeros(1, dtype=torch.long))

    def compute_alpha(self, freqs: torch.Tensor) -> torch.Tensor:
        log_freq = torch.log(freqs.float() + 1)
        return torch.sigmoid(self.alpha_w * log_freq + self.alpha_b)

    def get_similarity_prior(self, x: torch.Tensor, is_cold: torch.Tensor) -> torch.Tensor:
        """Get routing prior based on similar warm tokens."""
        batch_size = x.shape[0]
        device = x.device

        # Project to similarity space
        x_proj = self.sim_proj(x)

        # Default to uniform
        prior = torch.ones(batch_size, self.num_experts, device=device) / self.num_experts

        if self.cache_count.item() == 0:
            return prior

        # For cold tokens, find similar cached tokens
        cache_size = min(self.cache_count.item(), self.embedding_cache.shape[0])
        cached_embs = self.embedding_cache[:cache_size]
        cached_routing = self.routing_cache[:cache_size]

        # Compute similarities
        x_norm = F.normalize(x_proj, dim=-1)
        cache_norm = F.normalize(cached_embs, dim=-1)

        # Top-k similar tokens
        similarities = torch.mm(x_norm, cache_norm.t())  # [batch, cache_size]
        top_k = min(5, cache_size)
        top_sims, top_idx = torch.topk(similarities, top_k, dim=-1)

        # Weighted average of neighbor routing
        top_sims = F.softmax(top_sims * 10, dim=-1)  # Temperature scaling
        neighbor_routing = cached_routing[top_idx]  # [batch, k, num_experts]

        # Weighted sum
        sim_prior = (top_sims.unsqueeze(-1) * neighbor_routing).sum(dim=1)

        # Only use similarity prior for cold tokens
        cold_mask = is_cold.unsqueeze(-1).float()
        prior = prior * (1 - cold_mask) + sim_prior * cold_mask

        return prior

    def update_cache(self, x: torch.Tensor, routing: torch.Tensor, freqs: torch.Tensor):
        """Update cache with warm token routing (freq > threshold)."""
        if not self.training:
            return

        warm_mask = freqs > 10
        if not warm_mask.any():
            return

        warm_x = x[warm_mask]
        warm_routing = routing[warm_mask]
        warm_proj = self.sim_proj(warm_x).detach()

        # Add to cache (circular buffer)
        num_warm = warm_x.shape[0]
        start_idx = self.cache_count.item() % self.routing_cache.shape[0]
        end_idx = min(start_idx + num_warm, self.routing_cache.shape[0])
        actual_num = end_idx - start_idx

        self.routing_cache[start_idx:end_idx] = warm_routing[:actual_num].detach()
        self.embedding_cache[start_idx:end_idx] = warm_proj[:actual_num]
        self.cache_count += actual_num

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # Compute alpha
        alpha = self.compute_alpha(freqs).unsqueeze(-1)

        # Learned routing
        learned_logits = self.router(x)
        learned_routing = F.softmax(learned_logits, dim=-1)

        # Get similarity-based prior for cold tokens
        is_cold = freqs <= 5
        sim_prior = self.get_similarity_prior(x, is_cold)

        # Adaptive mixing
        final_routing = alpha * sim_prior + (1 - alpha) * learned_routing

        # Update cache with warm token routing
        self.update_cache(x, learned_routing, freqs)

        return final_routing, alpha.squeeze(-1)


class SimilarityPriorMoE(nn.Module):
    """
    MoE with similarity-based prior instead of uniform.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_experts: int = 8,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        self.dense_linear = nn.Linear(dense_dim, embedding_dim)

        # Input dimension
        total_dim = (self.num_sparse_fields + 1) * embedding_dim

        # Similarity-based router
        self.router = SimilarityPriorRouter(
            input_dim=total_dim,
            num_experts=num_experts,
        )

        # Expert networks
        expert_output_dim = 64
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], expert_output_dim),
            )
            for _ in range(num_experts)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(expert_output_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None,
                use_data_structures: bool = True) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        # Default frequency
        if token_freqs is None:
            token_freqs = torch.ones(batch_size, device=device)

        # Process features
        dense_normed = self.dense_bn(dense)
        dense_emb = self.dense_linear(dense_normed)

        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]

        features = torch.cat(sparse_embs + [dense_emb], dim=1)

        # Get routing with similarity prior
        routing, alpha = self.router(features, token_freqs)

        # Expert outputs
        expert_outputs = torch.stack([exp(features) for exp in self.experts], dim=1)

        # Weighted combination
        output = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # Final prediction
        logits = self.output_head(output).squeeze(-1)

        return torch.sigmoid(logits)


# =============================================================================
# Plan C: FreqAwareNet - Simplified Architecture
# =============================================================================

class FrequencyAwareLayer(nn.Module):
    """
    Lightweight frequency-aware transformation layer.
    Adjusts embedding based on frequency without complex routing.

    embedding' = embedding * (1 + α(freq) * scale) + α(freq) * bias

    Where α(freq) = sigmoid(w * log(freq+1) + b)
    """

    def __init__(self, embedding_dim: int, alpha_init_w: float = -0.5, alpha_init_b: float = 0.5):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Learnable alpha parameters
        self.alpha_w = nn.Parameter(torch.tensor(alpha_init_w))
        self.alpha_b = nn.Parameter(torch.tensor(alpha_init_b))

        # Learnable scale and bias for frequency-aware transformation
        self.scale = nn.Parameter(torch.zeros(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def compute_alpha(self, freqs: torch.Tensor) -> torch.Tensor:
        log_freq = torch.log(freqs.float() + 1)
        return torch.sigmoid(self.alpha_w * log_freq + self.alpha_b)

    def forward(self, embedding: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [batch, embedding_dim]
            freqs: [batch]
        Returns:
            adjusted_embedding: [batch, embedding_dim]
        """
        alpha = self.compute_alpha(freqs).unsqueeze(-1)  # [batch, 1]

        # Frequency-aware transformation
        # High alpha (cold) -> more bias/scale adjustment
        # Low alpha (warm) -> keep original embedding
        adjusted = embedding * (1 + alpha * self.scale) + alpha * self.bias

        return adjusted


class FreqAwareNet(nn.Module):
    """
    Simplified frequency-aware network without MoE.

    Key insight: Instead of complex routing, directly adjust embeddings
    based on frequency using learnable transformations.

    Benefits:
    - Much simpler than MoE
    - Fewer parameters to learn
    - No routing overhead
    - Preserves adaptive α idea
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        use_freq_aware: bool = True,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())
        self.embedding_dim = embedding_dim
        self.use_freq_aware = use_freq_aware

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Frequency-aware layers (one per field)
        if use_freq_aware:
            self.freq_aware_layers = nn.ModuleDict({
                name: FrequencyAwareLayer(embedding_dim)
                for name in sparse_dims.keys()
            })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        self.dense_linear = nn.Linear(dense_dim, embedding_dim)

        # Feature interaction layer
        total_dim = (self.num_sparse_fields + 1) * embedding_dim

        # Cross layer (simplified DCN-style)
        self.cross_w = nn.Parameter(torch.randn(total_dim, 1) * 0.01)
        self.cross_b = nn.Parameter(torch.zeros(total_dim))

        # DNN layers
        layers = []
        prev_dim = total_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)

        # Final output
        self.output = nn.Linear(hidden_dim + total_dim, 1)  # Combine DNN and cross

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor,
                token_freqs: Optional[torch.Tensor] = None,
                use_data_structures: bool = True) -> torch.Tensor:
        batch_size = dense.shape[0]
        device = dense.device

        # Default frequency
        if token_freqs is None:
            token_freqs = torch.ones(batch_size, device=device)

        # Process dense
        dense_normed = self.dense_bn(dense)
        dense_emb = self.dense_linear(dense_normed)

        # Process sparse with frequency-aware adjustment
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])

            if self.use_freq_aware:
                emb = self.freq_aware_layers[name](emb, token_freqs)

            sparse_embs.append(emb)

        # Concatenate
        x0 = torch.cat(sparse_embs + [dense_emb], dim=1)

        # Cross layer
        cross_term = x0 * (torch.mm(x0, self.cross_w) + self.cross_b)

        # DNN
        dnn_out = self.dnn(x0)

        # Combine
        combined = torch.cat([cross_term, dnn_out], dim=1)
        logits = self.output(combined).squeeze(-1)

        return torch.sigmoid(logits)


# =============================================================================
# Factory function for creating models
# =============================================================================

def create_improved_model(
    model_name: str,
    dense_dim: int,
    sparse_dims: Dict[str, int],
    embedding_dim: int = 16,
    **kwargs
) -> nn.Module:
    """Create improved cold-start model by name."""

    if model_name == 'DropoutNet':
        return DropoutNetRecommender(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
            embedding_dropout=kwargs.get('embedding_dropout', 0.3),
        )
    elif model_name == 'SimilarityPriorMoE':
        return SimilarityPriorMoE(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
            num_experts=kwargs.get('num_experts', 8),
        )
    elif model_name == 'FreqAwareNet':
        return FreqAwareNet(
            dense_dim=dense_dim,
            sparse_dims=sparse_dims,
            embedding_dim=embedding_dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
